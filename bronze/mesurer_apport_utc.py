"""Measure minutes present in an audited UTC source and absent from both local offset candidates."""

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess


def sql_path(path):
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def coverage(data, duckdb, year):
    providers = data / 'recovered/autres_fournisseurs'
    touhid = providers / 'touhid_xauusd' / str(year)
    carlos = providers / 'carlos_silva_xauusd' / f'year={year}'
    if (touhid / 'audit_annuel.json').is_file():
        manifest_path = touhid / 'audit_annuel.json'
        source = sql_path(touhid / 'brut/*.csv.gz')
        relation = f"""read_csv({source}, auto_detect=false, header=true, delim=',',
            columns={{'timestamp':'VARCHAR','ask':'VARCHAR','bid':'VARCHAR',
                      'ask_volume':'VARCHAR','bid_volume':'VARCHAR'}})"""
    elif (carlos / 'audit_annuel.json').is_file():
        manifest_path = carlos / 'audit_annuel.json'
        relation = f"read_parquet({sql_path(carlos / 'month=*/*.parquet')})"
    else:
        raise FileNotFoundError(f'Aucune source UTC auditée pour {year}')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig'))


    metrics = measure(data, duckdb, year, relation, "timestamp::TIMESTAMP_MS", (2, 3))
    if metrics['ticks_utc'] != manifest['ticks']:
        raise ValueError(f'Effectif différent de la source auditée : {year}')
    return {'source': manifest['source'], **metrics}


def coverage_histdata(data, duckdb, year):
    folder = data / 'recovered/autres_fournisseurs/histdata' / str(year)
    monthly = []
    files = []
    for manifest in sorted(folder.glob('*/HISTDATA_*.audit.json')):
        report = json.loads(manifest.read_text(encoding='utf-8'))
        tick_file, = [row['file'] for row in report['files'] if row['file'].endswith('.csv')]
        files.append(sql_path(manifest.parent / 'brut' / tick_file))
        monthly.append(report)
    if not monthly:
        raise FileNotFoundError(f'Aucune archive HistData auditée pour {year}')
    relation = f"""read_csv([{', '.join(files)}], auto_detect=false, header=false, delim=',',
        columns={{'stamp':'VARCHAR','bid':'VARCHAR','ask':'VARCHAR','volume':'VARCHAR'}})"""
    metrics = measure(data, duckdb, year, relation, "strptime(left(stamp, 13), '%Y%m%d %H%M')", (6, 7, 8))
    if metrics['ticks_utc'] != sum(month['ticks'] for month in monthly):
        raise ValueError(f'Effectif différent de l’archive auditée : {year}')
    return {'source': 'HistData', 'clock': 'raw', 'mois_audites': len(monthly),
            'decalages_brut_vers_local_testes': [6, 7, 8],
            'minutes_brutes': metrics['minutes_utc'], 'ticks_bruts': metrics['ticks_utc'],
            'minutes_sans_locale': metrics['minutes_inedites_utc'],
            'ticks_dans_minutes_sans_locale': metrics['ticks_dans_minutes_inedites']}


def measure(data, duckdb, year, relation, clock, offsets):
    joins = '\n  '.join(
        f'LEFT JOIN originals AS local{shift} ON local{shift}.minute_exportee = external.minute_utc + INTERVAL {shift} HOUR'
        for shift in offsets
    )
    absent = ' AND '.join(f'local{shift}.minute_exportee IS NULL' for shift in offsets)
    audit = data / 'source_runs/audit_local'
    old = sql_path(audit / 'minutes_ancien.parquet')
    recent = sql_path(audit / 'minutes_recent.parquet')
    query = f"""
SET threads=3;
SET memory_limit='2GB';
WITH originals AS (
    SELECT DISTINCT minute_exportee
      FROM read_parquet([{old}, {recent}])
     WHERE minute_exportee >= TIMESTAMP '{year}-01-01'
       AND minute_exportee < TIMESTAMP '{year+1}-01-02'
),
external AS (
    SELECT date_trunc('minute', {clock}) AS minute_utc, count(*) AS ticks
      FROM {relation}
     GROUP BY 1
)
SELECT count(*) AS minutes_utc, sum(external.ticks) AS ticks_utc,
       count(*) FILTER (WHERE {absent}) AS minutes_inedites_utc,
       coalesce(sum(external.ticks) FILTER (
         WHERE {absent}), 0) AS ticks_dans_minutes_inedites
  FROM external
  {joins};
"""
    result = subprocess.run([str(duckdb), '-batch', '-bail', '-csv'], input=query, text=True,
                            capture_output=True, encoding='utf-8', check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    raw, = csv.DictReader(io.StringIO(result.stdout))
    metrics = {key: int(value) for key, value in raw.items()}
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--duckdb', required=True, type=Path)
    parser.add_argument('years', nargs='+', type=int)
    parser.add_argument('--histdata', action='store_true')
    args = parser.parse_args()
    name = 'Apport_minutes_HistData_XAUUSD.json' if args.histdata else 'Apport_minutes_UTC_XAUUSD.json'
    output = args.data / 'reports' / name
    compare = coverage_histdata if args.histdata else coverage
    results = json.loads(output.read_text(encoding='utf-8')) if output.is_file() else {}
    for year in args.years:
        result = compare(args.data, args.duckdb, year)
        results[str(year)] = result
        print(year, result, flush=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(output)
