"""Extract weekly XAUUSD files from a verified publisher release and audit the year."""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def sha256(file):
    with file.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(file):
    return json.loads(file.read_text(encoding='utf-8-sig'))


def quoted(path):
    return "'" + path.as_posix().replace("'", "''") + "'"


def extract(folder):
    release = read_json(folder / 'provenance.json')
    archive = folder / release['archive']
    if sha256(archive) != release['sha256']:
        raise ValueError('Empreinte ZIP différente du manifeste')
    year = folder.name
    if not re.fullmatch(r'\d{4}', year):
        raise ValueError('Le dossier de la release doit être son année')
    pattern = re.compile(rf'XAUUSD_{year}_w(\d{{2}})\.csv\.gz')
    output = folder / 'brut'
    update_dir = folder / 'mise_a_jour'
    update_manifest = update_dir / 'provenance.json'
    updates = read_json(update_manifest) if update_manifest.exists() else {}
    update_files = {item['file']: item for item in updates.get('files', [])}

    output.mkdir(exist_ok=True)
    files = {}
    with ZipFile(archive) as bundle, TemporaryDirectory(prefix='extraction_', dir=folder) as staging:
        staging = Path(staging)
        members = {}
        for member in bundle.infolist():
            name = PurePosixPath(member.filename).name
            if pattern.fullmatch(name):
                if name in members:
                    raise ValueError(f'Doublon dans le ZIP : {name}')
                members[name] = member
        if not members:
            raise ValueError('Aucun fichier hebdomadaire dans le ZIP')
        for name in sorted(set(members) | set(update_files)):
            if not pattern.fullmatch(name):
                raise ValueError(f'Nom inattendu : {name}')
            destination = output / name
            temp = staging / name
            update = update_files.get(name)
            if update:
                shutil.copyfile(update_dir / name, temp)
                if sha256(temp) != update['sha256']:
                    raise ValueError(f'Empreinte différente : {name}')
            else:
                with bundle.open(members[name]) as source, temp.open('xb') as target:
                    shutil.copyfileobj(source, target, 1024 * 1024)
            digest = sha256(temp)
            if destination.exists() and sha256(destination) != digest:
                raise FileExistsError(f'Un fichier différent existe : {destination}')
            if not destination.exists():
                temp.rename(destination)
            files[name] = {'sha256': digest, 'source': updates['base'] if update else release['url']}

    if {file.name for file in output.glob('*.csv.gz')} != set(files):
        raise ValueError(f'Contenu inattendu sous {output}')
    return release, updates, output, files


def audit(folder, duckdb, public_weeks=False):
    if public_weeks:
        source = folder / 'semaines_publiques'
        audit_path = source / 'audit_annuel.json'
        release = read_json(source / 'provenance.json')
        updates = release
        raw = source / 'brut'
        files = {item['file']: {'sha256': item['sha256'], 'source': item['source']} for item in release['files']}
        if {path.name for path in raw.glob('*.csv.gz')} != set(files):
            raise ValueError('Les semaines disponibles ne correspondent pas au manifeste')
        if any(sha256(raw / name) != item['sha256'] for name, item in files.items()):
            raise ValueError('Empreinte différente pour une des semaines')
    else:
        audit_path = folder / 'audit_annuel.json'
        release, updates, raw, files = extract(folder)
    pattern = quoted(raw.resolve() / '*.csv.gz')
    query = f"""
SET TimeZone='UTC';
SET memory_limit='2GB';
SET threads=3;
WITH ticks AS (
    SELECT filename, try_cast(timestamp AS TIMESTAMP_MS) AS t, ask, bid, ask_volume, bid_volume
    FROM read_csv({pattern}, header=true, filename=true, auto_detect=false, delim=',',
        columns={{'timestamp':'VARCHAR','ask':'DECIMAL(18,9)','bid':'DECIMAL(18,9)',
                  'ask_volume':'DECIMAL(18,9)','bid_volume':'DECIMAL(18,9)'}})
)
SELECT filename, count(*) AS ticks, strftime(min(t), '%Y-%m-%d %H:%M:%S.%g') AS first_utc, strftime(max(t), '%Y-%m-%d %H:%M:%S.%g') AS last_utc,
    count(*) FILTER (WHERE t IS NULL OR ask IS NULL OR bid IS NULL
        OR ask_volume IS NULL OR bid_volume IS NULL) AS invalid_rows,
    count(*) FILTER (WHERE ask <= 0 OR bid <= 0) AS nonpositive_prices,
    count(*) FILTER (WHERE ask < bid) AS negative_spreads,
    count(*) FILTER (WHERE ask_volume < 0 OR bid_volume < 0) AS negative_volumes
FROM ticks GROUP BY filename ORDER BY filename;
"""
    result = subprocess.run([str(duckdb), '-batch', '-bail', '-csv'], input=query, text=True,
                            capture_output=True, encoding='utf-8', check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    fields = ('ticks', 'invalid_rows', 'nonpositive_prices', 'negative_spreads', 'negative_volumes')
    details = []
    for row in csv.DictReader(io.StringIO(result.stdout)):
        name = Path(row.pop('filename')).name
        details.append({'file': name, **files[name], **row, **{key: int(row[key]) for key in fields}})
    if set(files) != {row['file'] for row in details}:
        raise ValueError('Des semaines sont vides ou manquantes')
    for row in details:
        if not row['first_utc'].startswith(folder.name) or not row['last_utc'].startswith(folder.name):
            raise ValueError(f'Hors année : {row}')
    if any(left['last_utc'] >= right['first_utc'] for left, right in zip(details, details[1:])):
        raise ValueError('Chevauchement d’horodatages entre deux semaines')
    for item in updates.get('files', []):
        actual, = [row for row in details if row['file'] == item['file']]
        if any(actual[key] != item[expected] for key, expected in (
            ('ticks', 'rows'), ('first_utc', 'first_tick_utc'), ('last_utc', 'last_tick_utc')
        )):
            raise ValueError(f'Annexe différente des métadonnées : {item["file"]}')

    summary = {
        'source': release['source'], 'status': 'audited', 'validation_status': 'pending',
        'release': release, 'first_utc': details[0]['first_utc'], 'last_utc': details[-1]['last_utc'],
        **{key: sum(row[key] for row in details) for key in fields}, 'files': details,
    }
    pending = audit_path.with_suffix('.tmp')
    pending.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    pending.replace(audit_path)
    print(f"{folder.name}: {summary['ticks']:,} ticks, {len(details)} fichiers, "
          f"{summary['first_utc']} -> {summary['last_utc']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-dir', required=True, type=Path)
    parser.add_argument('--duckdb', required=True, type=Path)
    parser.add_argument('--public-weeks', action='store_true')
    args = parser.parse_args()
    audit(args.release_dir, args.duckdb, args.public_weeks)
