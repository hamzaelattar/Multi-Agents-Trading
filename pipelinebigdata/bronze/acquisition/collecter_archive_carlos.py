"""Download and audit the pinned CarlosSilva1 XAUUSD Parquet release."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import io
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import quote
from urllib.request import urlopen

from recuperer_fichier_public import download_file

DATASET = 'CarlosSilva1/xauusd-ticks'
REVISION = 'd14fc64dc3b0452732e60662d94b500a6bb3d534'


def collect(out, year, duckdb):
    api = f'https://huggingface.co/api/datasets/{DATASET}/revision/{REVISION}?blobs=true'
    with urlopen(api, timeout=30) as response:
        catalogue = json.load(response)
    if catalogue['sha'] != REVISION:
        raise ValueError('Révision différente de celle du pilote')
    year_dir = out / f'year={year}'
    pattern = re.compile(rf'year={year}/month=\d{{2}}/XAUUSD-{year}-\d{{2}}-part\d{{4}}\.parquet')
    partitions = [file for file in catalogue['siblings'] if pattern.fullmatch(file['rfilename'])]
    if not partitions:
        raise ValueError(f'Aucune publication pour {year}')
    base = f'https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}'

    def acquire(item):
        name = item['rfilename']
        digest = item['lfs']['sha256']
        download_file(base + '/' + quote(name, safe='=/'), out / name, item['size'], digest)
        print(name, flush=True)
        return {'file': name, 'sha256': digest, 'bytes': item['size']}

    with ThreadPoolExecutor(max_workers=2) as workers:
        files = list(workers.map(acquire, partitions))

    if {file.relative_to(out).as_posix() for file in year_dir.glob('month=*/*.parquet')} != {row['file'] for row in files}:
        raise ValueError(f'Partitions inattendues pour {year}')
    path = (year_dir.resolve() / 'month=*/*.parquet').as_posix().replace("'", "''")
    query = f"""
SET threads=3;
SET memory_limit='2GB';
SELECT count(*) AS ticks, min(timestamp) AS first_utc, max(timestamp) AS last_utc,
       count(DISTINCT timestamp::DATE) AS dates,
       count(*) FILTER (WHERE timestamp IS NULL OR ask_price IS NULL OR bid_price IS NULL
           OR ask_volume IS NULL OR bid_volume IS NULL) AS invalid_rows,
       count(*) FILTER (WHERE NOT isfinite(ask_price) OR NOT isfinite(bid_price)
           OR NOT isfinite(ask_volume) OR NOT isfinite(bid_volume)) AS nonfinite,
       count(*) FILTER (WHERE ask_price <= 0 OR bid_price <= 0) AS nonpositive_prices,
       count(*) FILTER (WHERE ask_price < bid_price) AS negative_spreads,
       count(*) FILTER (WHERE ask_volume < 0 OR bid_volume < 0) AS negative_volumes
FROM read_parquet('{path}');
"""
    result = subprocess.run([str(duckdb), '-batch', '-bail', '-csv'], input=query, text=True,
                            capture_output=True, encoding='utf-8', check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    summary, = csv.DictReader(io.StringIO(result.stdout))
    summary = {key: value if key.endswith('_utc') else int(value) for key, value in summary.items()}
    if not all(summary[key].startswith(str(year)) for key in ('first_utc', 'last_utc')):
        raise ValueError(f'Bornes UTC incohérentes : {summary}')
    report = {'source': DATASET, 'license': 'CC-BY-4.0 (publisher)', 'revision': REVISION,
              'status': 'audited', 'validation_status': 'pending', **summary, 'files': files}
    audit = year_dir / 'audit_annuel.json'
    pending = audit.with_suffix('.tmp')
    pending.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    pending.replace(audit)
    print(f"{year}: {summary['ticks']:,} ticks, {len(files)} partitions", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--duckdb', required=True, type=Path)
    parser.add_argument('--year', required=True, type=int)
    args = parser.parse_args()
    collect(args.out, args.year, args.duckdb)
