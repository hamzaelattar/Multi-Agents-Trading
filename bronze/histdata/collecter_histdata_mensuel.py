"""Download and audit monthly XAUUSD archives from the public HistData forms."""

import argparse
import csv
from datetime import date, timedelta
import hashlib
from http.client import IncompleteRead
import io
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from zipfile import BadZipFile, ZipFile

from collecter_histdata_pilote import entry, main as download


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def sql_path(path):
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def write_json(path, value):
    pending = path.with_suffix('.tmp')
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    pending.replace(path)


def audit_csv(path, duckdb):
    query = f"""
    SET threads=3;
    SET memory_limit='2GB';
    SELECT count(*) AS ticks, min(stamp) AS first_raw, max(stamp) AS last_raw,
           count(DISTINCT left(stamp, 8)) AS dates,
           count(DISTINCT left(stamp, 11)) AS hours,
           count(DISTINCT left(stamp, 13)) AS minutes,
           count(*) FILTER (WHERE try_strptime(stamp,'%Y%m%d %H%M%S%g') IS NULL
               OR ask IS NULL OR bid IS NULL OR volume IS NULL) AS invalid_rows,
           count(*) FILTER (WHERE ask <= 0 OR bid <= 0) AS nonpositive_prices,
           count(*) FILTER (WHERE ask < bid) AS negative_spreads,
           count(*) FILTER (WHERE volume <> 0) AS nonzero_source_volume
    FROM read_csv({sql_path(path)}, header=false, auto_detect=false, delim=',',
        columns={{'stamp':'VARCHAR','bid':'DECIMAL(20,9)','ask':'DECIMAL(20,9)','volume':'DECIMAL(20,9)'}});
    """
    result = subprocess.run([str(duckdb), '-batch', '-bail', '-csv'], input=query,
                            text=True, capture_output=True, encoding='utf-8', check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    summary, = csv.DictReader(io.StringIO(result.stdout))
    return {key: value if key.endswith('_raw') else int(value) for key, value in summary.items()}


def audit_archive(archive, month, duckdb):
    manifest = archive.with_suffix('.json')
    origin = json.loads(manifest.read_text(encoding='utf-8'))
    if origin['month'] != month or origin['sha256'] != sha256(archive):
        raise ValueError(f'Empreinte ou mois incohérent pour {archive.name}')
    audit_path = archive.with_suffix('.audit.json')
    if audit_path.exists():
        saved = json.loads(audit_path.read_text(encoding='utf-8'))
        if saved['archive_sha256'] != origin['sha256'] or any(
            sha256(archive.parent / 'brut' / item['file']) != item['sha256'] for item in saved['files']
        ):
            raise ValueError(f'Les données auditées ont changé pour {month}')
        return saved

    stem = 'DAT_ASCII_XAUUSD_T_' + month.replace('-', '')
    names = [stem + suffix for suffix in ('.csv', '.txt')]
    raw = archive.parent / 'brut'
    with TemporaryDirectory(prefix='audit_', dir=archive.parent) as staging, ZipFile(archive) as zipfile:
        staging = Path(staging)
        if sorted(zipfile.namelist()) != sorted(names):
            raise ValueError(f'Membres inattendus dans {archive.name}')
        if sum(zipfile.getinfo(name).file_size for name in names) > shutil.disk_usage(archive).free:
            raise OSError('Espace disque insuffisant pour extraire le lot')
        files = []
        for name in names:
            path = staging / name
            with zipfile.open(name) as source, path.open('xb') as destination:
                shutil.copyfileobj(source, destination, 1024 * 1024)
            digest = sha256(path)
            if (raw / name).exists() and sha256(raw / name) != digest:
                raise ValueError(f'Le fichier brut existant diffère de l’archive : {name}')
            files.append({'file': name, 'bytes': path.stat().st_size, 'sha256': digest})

        summary = audit_csv(staging / names[0], duckdb)
        prefix = month.replace('-', '')
        if (not summary['ticks'] or summary['invalid_rows'] or summary['nonpositive_prices']
                or not all(summary[key].startswith(prefix) for key in ('first_raw', 'last_raw'))):
            raise ValueError(f'Lot incohérent : {month} {summary}')

        raw.mkdir(exist_ok=True)
        for name in names:
            target = raw / name
            if not target.exists():
                (staging / name).rename(target)

    report = {'source': 'HistData', 'month': month, 'status': 'audited',
              'validation_status': 'pending', 'archive_sha256': origin['sha256'], 'files': files,
              'clock': 'raw', 'quoting_volumes_available': False, **summary}
    write_json(audit_path, report)
    return report


def months(start, end):
    current = date.fromisoformat(start + '-01')
    last = date.fromisoformat(end + '-01')
    if last < current:
        raise ValueError('La borne de fin précède le premier mois')
    while current <= last:
        yield current.strftime('%Y-%m')
        current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)


def collect(out, duckdb, start, end, pause, offline):
    previous_request = None
    journal = out / 'collecte_mensuelle.jsonl'
    out.mkdir(parents=True, exist_ok=True)
    for month in months(start, end):
        _, _, name = entry(month)
        archive = out / month[:4] / month[5:] / name
        try:
            if not archive.exists():
                if offline:
                    raise FileNotFoundError(archive)
                if previous_request is not None:
                    time.sleep(max(0, pause - (time.monotonic() - previous_request)))
                download(archive.parent, month)
                previous_request = time.monotonic()
            report = audit_archive(archive, month, duckdb)
        except (OSError, ValueError, RuntimeError, IncompleteRead, BadZipFile) as error:
            with journal.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'month': month, 'status': 'stopped', 'error': str(error)}) + '\n')
            raise
        print(f"{month} : {report['ticks']:,} ticks, {report['dates']} jours, "
              f"{report['first_raw']} -> {report['last_raw']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--duckdb', required=True, type=Path)
    parser.add_argument('--from', required=True, dest='start')
    parser.add_argument('--through', required=True, dest='end')
    parser.add_argument('--pause', type=float, default=5)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()
    collect(args.out, args.duckdb, args.start, args.end, args.pause, args.offline)
