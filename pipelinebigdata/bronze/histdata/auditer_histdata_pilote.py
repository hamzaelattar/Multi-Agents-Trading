"""Audit one isolated HistData XAUUSD month and compare its 5 August UTC quotes with JForex."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import zipfile


DAY = '2025-08-05'


def quoted(file):
    return "'" + file.as_posix().replace("'", "''") + "'"


def month_candidate(archive, evidence, duckdb):
    """Create a separate, explicitly unvalidated UTC price candidate for August 2025."""
    proof = json.loads(evidence.read_text(encoding='utf-8'))
    with archive.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    best = proof['strongest']
    if (proof['day'] != DAY or proof['archive_sha256'] != source_hash or
            best['utc_minus_raw_hours'] != 4 or best['timestamp_ask_bid_matches'] != 156842):
        raise ValueError('Preuve du décalage estival manquante ou incompatible.')
    out = archive.parent / 'XAUUSD_2025-08_HistData_prix_candidats.parquet'
    manifest = out.with_suffix('.json')
    if out.exists() or manifest.exists():
        raise FileExistsError(out)
    with TemporaryDirectory(prefix='candidat_histdata_', dir=archive.parent) as temp:
        folder = Path(temp)
        raw = folder / 'raw.csv'
        with zipfile.ZipFile(archive) as zipped:
            member, = [name for name in zipped.namelist() if name.lower().endswith('.csv')]
            with zipped.open(member) as src, raw.open('xb') as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
        parquet = folder / 'candidate.parquet'
        stats = folder / 'stats.json'
        query = f"""
SET threads=3; SET memory_limit='3GB'; SET TimeZone='UTC';
COPY (WITH ticks AS (
  SELECT (try_strptime(stamp,'%Y%m%d %H%M%S%g') + INTERVAL 4 HOUR)::TIMESTAMPTZ AS timestamp_utc, ask, bid
  FROM read_csv({quoted(raw)}, header=false, auto_detect=false, parallel=false,
    columns={{'stamp':'VARCHAR','bid':'DECIMAL(20,9)','ask':'DECIMAL(20,9)','volume':'DECIMAL(20,9)'}}))
  SELECT timestamp_utc, ask, bid, NULL::DECIMAL(20,9) AS ask_volume, NULL::DECIMAL(20,9) AS bid_volume
  FROM ticks WHERE timestamp_utc >= TIMESTAMPTZ '2025-08-01 00:00:00+00'
    AND timestamp_utc < TIMESTAMPTZ '2025-09-01 00:00:00+00'
) TO {quoted(parquet)} (FORMAT PARQUET, COMPRESSION ZSTD);
COPY (SELECT count(*) AS ticks, min(timestamp_utc) AS first_utc, max(timestamp_utc) AS last_utc,
           count(DISTINCT cast(timestamp_utc AS DATE)) AS days_with_ticks,
           count(DISTINCT date_trunc('hour',timestamp_utc)) AS hours_with_ticks,
           count(DISTINCT date_trunc('minute',timestamp_utc)) AS minutes_with_ticks,
           count(*) FILTER (WHERE ask IS NULL OR bid IS NULL OR ask <= 0 OR bid <= 0 OR ask < bid) AS invalid_quotes,
           count(ask_volume) + count(bid_volume) AS filled_volume_fields
      FROM read_parquet({quoted(parquet)})) TO {quoted(stats)} (FORMAT JSON, ARRAY true);
"""
        result = subprocess.run([str(duckdb), '-batch', '-bail', '-csv'], input=query, text=True,
                                encoding='utf-8', capture_output=True, cwd=folder)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        summary, = json.loads(stats.read_text(encoding='utf-8'))
        if summary['invalid_quotes'] or summary['filled_volume_fields'] or summary['ticks'] == 0:
            raise ValueError('Audit incomplet du candidat de prix.')
        with parquet.open('rb') as stream:
            sha = hashlib.file_digest(stream, 'sha256').hexdigest()
        parquet.rename(out)
    with manifest.open('x', encoding='utf-8') as stream:
        json.dump({'source': 'HistData', 'status': 'audited', 'validation_status': 'pending',
                   'scope': 'prix seuls, conversion candidate, archive du mois civil source limitée aux dates UTC d’août',
                   'utc_minus_raw_hours_candidate': 4, 'calibration_day': DAY, 'archive_sha256': source_hash,
                   'sha256': sha, **summary}, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(f'HistData août : {summary["ticks"]:,} ticks candidats, volumes absents.')


def main(archive, jforex, duckdb):
    directory = archive.parent
    out = directory / f'XAUUSD_{DAY}_HistData_UTC.csv'
    report = directory / 'audit_et_comparaison_2025-08.json'
    if out.exists() or report.exists():
        raise FileExistsError('Le résultat HistData existe déjà.')
    with TemporaryDirectory(prefix='audit_histdata_', dir=directory) as temp:
        folder = Path(temp)
        raw = folder / 'raw.csv'
        with zipfile.ZipFile(archive) as body:
            member, = [name for name in body.namelist() if name.lower().endswith('.csv')]
            with body.open(member) as src, raw.open('xb') as dest:
                shutil.copyfileobj(src, dest, 1024 * 1024)

        expressions = {
            'month': '''SELECT count(*) AS ticks, min(t) AS first_utc, max(t) AS last_utc,
                count(DISTINCT cast(t AS DATE)) AS dates_with_ticks,
                count(DISTINCT date_trunc('hour', t)) AS hours_with_ticks,
                count(DISTINCT date_trunc('minute', t)) AS minutes_with_ticks,
                count(*) FILTER (WHERE t IS NULL OR ask IS NULL OR bid IS NULL) AS invalid_rows,
                count(*) FILTER (WHERE ask <= 0 OR bid <= 0) AS nonpositive_prices,
                count(*) FILTER (WHERE ask < bid) AS negative_spreads,
                count(*) FILTER (WHERE volume <> 0) AS nonzero_source_volume,
                count(*) FILTER (WHERE volume IS NULL) AS null_source_volume,
                count(*) - count(DISTINCT (t,ask,bid,volume)) AS exact_duplicates,
                count(*) FILTER (WHERE t < previous) AS reversals
                FROM (SELECT *, lag(t) OVER (ORDER BY i) AS previous FROM ticks)''',
            'days': '''SELECT cast(t AS DATE) AS date_utc, count(*) AS ticks,
                count(DISTINCT date_trunc('hour', t)) AS hours_with_ticks,
                count(DISTINCT date_trunc('minute', t)) AS minutes_with_ticks
                FROM ticks WHERE t IS NOT NULL GROUP BY 1 ORDER BY 1''',
            'comparison': '''WITH hm AS (SELECT date_trunc('minute',t) AS minute,
                    arg_max(ask,i) AS ask,arg_max(bid,i) AS bid FROM hist GROUP BY 1),
                jm AS (SELECT date_trunc('minute',t) AS minute,arg_max(ask,t) AS ask,arg_max(bid,t) AS bid FROM jf GROUP BY 1),
                both_minutes AS (SELECT abs(hm.ask-jm.ask) AS ask_diff, abs(hm.bid-jm.bid) AS bid_diff FROM hm JOIN jm USING (minute))
                SELECT (SELECT count(*) FROM hist) AS histdata_ticks,
                   (SELECT count(*) FROM jf) AS jforex_ticks,
                   (SELECT count(*) FROM hm) AS histdata_minutes,
                   (SELECT count(*) FROM jm) AS jforex_minutes,
                   (SELECT count(*) FROM (SELECT t FROM hist INTERSECT ALL SELECT t FROM jf)) AS exact_timestamp_pairs,
                   (SELECT count(*) FROM (SELECT t,ask,bid FROM hist INTERSECT ALL SELECT t,ask,bid FROM jf)) AS exact_timestamp_price_pairs,
                   count(*) AS common_minutes, median(ask_diff) AS median_absolute_ask_difference,
                   median(bid_diff) AS median_absolute_bid_difference
                FROM both_minutes''',
            'hours': '''WITH h AS (SELECT hour(t) AS hour_utc,count(*) AS ticks FROM hist GROUP BY 1),
                j AS (SELECT hour(t) AS hour_utc,count(*) AS ticks FROM jf GROUP BY 1)
                SELECT s.i AS hour_utc,coalesce(h.ticks,0) AS histdata_ticks,coalesce(j.ticks,0) AS jforex_ticks
                FROM range(24) s(i) LEFT JOIN h ON s.i=h.hour_utc LEFT JOIN j ON s.i=j.hour_utc ORDER BY s.i''',
        }
        query = f"""
SET threads=3; SET memory_limit='3GB'; SET TimeZone='UTC';
CREATE TEMP TABLE ticks AS SELECT row_number() OVER () AS i,
    try_strptime(stamp,'%Y%m%d %H%M%S%g') + INTERVAL 5 HOUR AS t, ask, bid, volume
FROM read_csv({quoted(raw)}, header=false, auto_detect=false, parallel=false,
    columns={{'stamp':'VARCHAR','bid':'DECIMAL(20,9)','ask':'DECIMAL(20,9)','volume':'DECIMAL(20,9)'}});
CREATE TEMP VIEW hist AS SELECT * FROM ticks WHERE t >= TIMESTAMP '{DAY}' AND t < TIMESTAMP '{DAY}' + INTERVAL 1 DAY;
CREATE TEMP TABLE jf AS SELECT cast(timestamp_utc AS TIMESTAMPTZ)::TIMESTAMP AS t,ask,bid
    FROM read_csv({quoted(jforex)}, header=true, auto_detect=false,
      columns={{'timestamp_utc':'VARCHAR','ask':'DECIMAL(20,9)','bid':'DECIMAL(20,9)','ask_volume':'VARCHAR','bid_volume':'VARCHAR'}});
"""
        for name, statement in expressions.items():
            query += f'COPY ({statement}) TO {quoted(folder / (name + ".json"))} (FORMAT JSON, ARRAY true);\n'
        query += f"""COPY (SELECT strftime(t,'%Y-%m-%dT%H:%M:%S.%gZ') AS timestamp_utc,ask,bid,
                   NULL::DOUBLE AS ask_volume,NULL::DOUBLE AS bid_volume FROM hist ORDER BY i)
                   TO {quoted(folder / 'day.csv')} (FORMAT CSV, HEADER true);"""
        result = subprocess.run([str(duckdb), '-batch', '-bail', '-csv'], input=query, encoding='utf-8',
                                capture_output=True, cwd=folder)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        report_data = {name: json.loads((folder / (name + '.json')).read_text(encoding='utf-8')) for name in expressions}
        month, = report_data['month']
        comparison, = report_data['comparison']
        assert month['invalid_rows'] == 0 and comparison['jforex_ticks'] == 156842
        day_file = folder / 'day.csv'
        with day_file.open('rb') as stream:
            day_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        verified_times = comparison['histdata_ticks'] > 0 and report_data['days']
        if not verified_times:
            raise ValueError('Le jour demandé est absent de l’archive.')
        day_file.rename(out)
    report_data.update(source='HistData, distinct from Dukascopy FX quotes', status='audited', validation_status='pending',
                       documented_clock='EST UTC-05:00 fixed', normalized_day_sha256=day_hash,
                       normalized_volumes='NULL for both sides; the original source field remains in the ZIP')
    with report.open('x', encoding='utf-8') as stream:
        json.dump(report_data, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'month': month, 'comparison': comparison}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--jforex', type=Path)
    parser.add_argument('--candidat-mensuel', type=Path, help='Preuve de calibration du 5 août 2025')
    parser.add_argument('--duckdb', type=Path, required=True)
    args = parser.parse_args()
    if args.candidat_mensuel:
        month_candidate(args.archive.resolve(strict=True), args.candidat_mensuel.resolve(strict=True), args.duckdb.resolve(strict=True))
    elif args.jforex:
        main(args.archive.resolve(strict=True), args.jforex.resolve(strict=True), args.duckdb.resolve(strict=True))
    else:
        parser.error('--jforex ou --candidat-mensuel est requis.')
