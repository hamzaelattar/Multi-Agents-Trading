"""Keep JForex evidence and HistData candidates separate from the originals."""

import hashlib
import csv
from datetime import datetime
from decimal import Decimal
import io
import json
from pathlib import Path
import shutil
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def copy_exact(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    original = digest(source)
    if not target.exists():
        with source.open('rb') as src, target.open('xb') as dest:
            shutil.copyfileobj(src, dest, 1024 * 1024)
    if digest(target) != original:
        raise ValueError(f'Fichier cible différent : {target}')


def stage(root):
    data = root / 'data'
    bronze = root / 'data/source_runs'
    for name in ('recovered/dukascopy', 'recovered/autres_fournisseurs', 'quarantine', 'manifests', 'reports', 'logs'):
        (data / name).mkdir(parents=True, exist_ok=True)
    sources = [bronze / 'verifications_jforex_2026-11-24', bronze / 'jforex_borne_2026-11-23']
    for source in sources:
        if not (source / 'heures.tsv').is_file():
            continue
        suffix = 'borne' if source.name.startswith('jforex_borne') else 'verifications'
        # A running attempt has a growing journal. Archive it only after its final CSV is available.
        if suffix == 'borne' and not (source / '2026-11-23.csv').is_file():
            continue
        copy_exact(source / 'heures.tsv', data / 'logs' / f'JForex_{suffix}_2026-11-24.tsv')
        for manifest in sorted(source.glob('*.audit.json')):
            day = manifest.name.removesuffix('.audit.json')
            data_file = source / f'{day}.csv'
            detail = json.loads(manifest.read_text(encoding='utf-8'))
            if not data_file.is_file() or digest(data_file) != detail['sha256']:
                raise ValueError(f'Empreinte JForex invalide : {day}')
            copy_exact(manifest, data / 'manifests' / f'JForex_{day}.audit.json')
            if detail['status'] == 'audited':
                copy_exact(data_file, data / 'recovered' / 'dukascopy' / day[:4] / day[5:7] /
                           f'XAUUSD_{day}_JForex_UTC.csv')
    followup = sources[0]
    for name in ('preuves_decimales_2021-01-05.json', 'comparaisons_verification.json'):
        copy_exact(followup / name, data / 'reports' / name)
    evidence = followup / 'preuves_decimales_2021-01-05.json'
    proof = json.loads(evidence.read_text(encoding='utf-8'))
    repairs = [r for r in proof['repair_evidence'] if r['status'] == 'exact_unique']
    if len(repairs) != 121 or not all(r['exact_splits'] == 1 for r in repairs):
        raise ValueError('Les preuves ne confirment pas les 121 reconstructions attendues.')
    payload = io.StringIO(newline='')
    writer = csv.writer(payload)
    writer.writerow(['timestamp_utc', 'ask', 'bid', 'ask_volume', 'bid_volume'])
    for row in repairs:
        t = datetime.fromisoformat(row['timestamp_utc']).isoformat(timespec='milliseconds').replace('+00:00', 'Z')
        writer.writerow([t, *(format(Decimal(v), 'f') for v in row['reference_values'])])
    normalized = payload.getvalue().encode('utf-8')
    patch = data / 'recovered/dukascopy/2021/01/XAUUSD_2021-01-05_reconstruction_decimales_unique.csv'
    patch.parent.mkdir(parents=True, exist_ok=True)
    if not patch.exists():
        with patch.open('xb') as file:
            file.write(normalized)
    if patch.read_bytes() != normalized:
        raise ValueError('Le lot de reconstructions existant diffère de ses preuves.')
    manifest = data / 'manifests/JForex_2021-01-05_reconstruction.audit.json'
    if not manifest.exists():
        with manifest.open('x', encoding='utf-8') as file:
            json.dump({'status': 'compared_pending_validation', 'ticks': len(repairs),
                       'sha256': digest(patch), 'source_reference_sha256': proof['jforex_sha256'],
                       'evidence_sha256': digest(evidence), 'add_to_tick_count': False}, file, indent=2)
            file.write('\n')
    hist = data / 'recovered/autres_fournisseurs/histdata/2025/08'
    nominal = hist / 'XAUUSD_2025-08-05_HistData_UTC.csv'
    if nominal.is_file():
        target = data / 'quarantine' / 'XAUUSD_2025-08-05_HistData_UTC_plus5_non_valide.csv'
        if target.exists():
            raise FileExistsError(target)
        nominal.rename(target)
    print('Bronze lots staged individually, original CSVs untouched.')


if __name__ == '__main__':
    stage(Path(sys.argv[1]).resolve(strict=True))
