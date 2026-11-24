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
    SELECT date_trunc('minute', timestamp::TIMESTAMP_MS) AS minute_utc, count(*) AS ticks
      FROM {relation}
     GROUP BY 1
)
SELECT count(*) AS minutes_utc, sum(external.ticks) AS ticks_utc,
       count(*) FILTER (WHERE local2.minute_exportee IS NULL AND local3.minute_exportee IS NULL) AS minutes_inedites_utc,
       coalesce(sum(external.ticks) FILTER (
         WHERE local2.minute_exportee IS NULL AND local3.minute_exportee IS NULL), 0) AS ticks_dans_minutes_inedites
  FROM external
  LEFT JOIN originals AS local2 ON local2.minute_exportee = external.minute_utc + INTERVAL 2 HOUR
  LEFT JOIN originals AS local3 ON local3.minute_exportee = external.minute_utc + INTERVAL 3 HOUR;
"""
    result = subprocess.run([str(duckdb), '-batch', '-bail', '-csv'], input=query, text=True,
                            capture_output=True, encoding='utf-8', check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    raw, = csv.DictReader(io.StringIO(result.stdout))
    metrics = {key: int(value) for key, value in raw.items()}
    if metrics['ticks_utc'] != manifest['ticks']:
        raise ValueError(f'Effectif différent de la source auditée : {year}')
    return {'source': manifest['source'], **metrics}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--duckdb', required=True, type=Path)
    parser.add_argument('years', nargs='+', type=int)
    args = parser.parse_args()
    output = args.data / 'reports/Apport_minutes_UTC_XAUUSD.json'
    results = json.loads(output.read_text(encoding='utf-8')) if output.is_file() else {}
    for year in args.years:
        result = coverage(args.data, args.duckdb, year)
        results[str(year)] = result
        print(year, result, flush=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(results, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(output)
