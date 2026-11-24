"""Profile each raw XAUUSD export using the DuckDB CLI and temporary Parquet partitions."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import chdir
import csv
from datetime import date, timedelta
import io
from itertools import count
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


FILES = {
    'ancien': 'XAUUSD_Ticks_2004.01.01_2021.10.15.csv',
    'recent': 'XAUUSD_Ticks_2021.01.04_2026.11.20.csv',
}
HEADER = b'Time (EET),Ask,Bid,AskVolume,BidVolume'
NUMBER = r'[+-]?[0-9]+,[0-9]+'
RAW_FORMAT = r'[0-9]{4}\.[0-9]{2}\.[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3},' + ','.join([NUMBER] * 4)
VALUES = {'ask': 2, 'bid': 4, 'ask_volume': 6, 'bid_volume': 8}


def sql_string(value):
    return "'" + str(value).replace("'", "''") + "'"


def number_fields():
    for name, position in VALUES.items():
        whole = f'parts[{position}]'
        fraction = f'parts[{position + 1}]'
        yield (f"CASE WHEN standard OR (width = 9 AND regexp_full_match({whole}, '[+-]?[0-9]+') "
               f"AND regexp_full_match({fraction}, '[0-9]+')) THEN "
               f"try_cast({whole} || '.' || {fraction} AS DOUBLE) END AS {name}")
        yield f"width = 9 AND {whole} = '' AND {fraction} = '' AS empty_{name}"
        yield f"width = 9 AND ({whole} = '' OR {fraction} = '') AS empty_component_{name}"


def sql_rows(exe, query, threads=3, memory_mb=3500):
    command = f"SET threads={threads}; SET memory_limit='{memory_mb}MB'; SET preserve_insertion_order=true;\n{query}"
    result = subprocess.run([str(exe), '-batch', '-bail', '-csv'], input=command,
                            encoding='utf-8', capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return list(csv.DictReader(io.StringIO(result.stdout)))


def physical_layout(path):
    with path.open('rb') as stream:
        if stream.readline().removeprefix(b'\xef\xbb\xbf').rstrip(b'\r\n') != HEADER:
            raise ValueError(f'Unexpected header: {path}')
        complete = 0
        last = b''
        while block := stream.read(16 * 1024**2):
            complete += block.count(b'\n')
            last = block
    fragment = last.rsplit(b'\n', 1)[-1]
    return complete, fragment.decode('ascii')


def stage(exe, source, target, fragment, *, preceding_rows=0, has_predecessor=False, threads=4):
    fields = ',\n'.join(number_fields())
    last = f'raw = {sql_string(fragment)}' if fragment else 'false'
    source_offset = preceding_rows - has_predecessor
    sql_rows(exe, f"""
SET partitioned_write_flush_threshold=65536;
SET partitioned_write_max_open_files=16;
COPY (
    WITH raw_rows AS (
        SELECT raw, row_number() OVER () AS row_id,
               string_split(raw, ',') AS parts
        FROM read_csv({sql_string(source.as_posix())}, columns={{'raw':'VARCHAR'}},
                      delim='|', quote='', escape='', header=true,
                      auto_detect=false, parallel={str(threads > 1).lower()})
    ), parsed AS (
        SELECT *, len(parts) AS width, {last} AS truncated,
               try_strptime(parts[1], '%Y.%m.%d %H:%M:%S.%g') AS t,
               regexp_full_match(raw, {sql_string(RAW_FORMAT)}) AS standard
        FROM raw_rows
    ), typed AS (
        SELECT raw, row_id + {source_offset} AS row_id, width, truncated, t,
               lag(t) OVER () AS previous_t,
               parts[1] = '' OR parts[1] IS NULL AS empty_time,
               standard, CASE WHEN t IS NULL THEN 'invalide' ELSE replace(left(parts[1], 7), '.', '-') END AS mois,
               {fields}
        FROM parsed
    )
    SELECT * FROM typed WHERE row_id > {preceding_rows}
) TO {sql_string(target.as_posix())} (FORMAT parquet, PARTITION_BY (mois), COMPRESSION zstd);
""", threads=threads, memory_mb=1100 if threads == 1 else 3500)


def stage_large(exe, source, target, fragment, block_size=384 * 1024**2):
    """Split on line endings, keeping a predecessor for each source-order boundary."""
    target.mkdir()

    def process_shard(path, output, preceding_rows, has_predecessor):
        stage(exe, path, output, fragment, preceding_rows=preceding_rows,
              has_predecessor=has_predecessor, threads=1)
        path.unlink()
        return output

    def collect(pending):
        done, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in done:
            output = future.result()
            for month in output.iterdir():
                destination = target / month.name
                destination.mkdir(exist_ok=True)
                for parquet in month.glob('*.parquet'):
                    parquet.rename(destination / f'{output.name}_{parquet.name}')
                month.rmdir()
            output.rmdir()
        return pending

    with source.open('rb') as stream, ThreadPoolExecutor(max_workers=4) as executor:
        header = stream.readline()
        offset = 0
        previous = b''
        remainder = b''
        pending = set()
        for index in count(1):
            read = stream.read(block_size)
            if not read and not remainder:
                break
            block = remainder + read
            end = block.rfind(b'\n') + 1
            if read:
                remainder = block[end:]
                block = block[:end]
            else:
                remainder = b''
            if not block:
                continue
            path = target.parent / f'shard_{index:03d}.csv'
            output = target.parent / f'shard_{index:03d}'
            with path.open('wb') as file:
                file.write(header)
                file.write(previous)
                file.write(block)
            pending.add(executor.submit(process_shard, path, output, offset, bool(previous)))
            if read:
                offset += block.count(b'\n')
                before_last = block.rfind(b'\n', 0, len(block) - 1)
                previous = block[before_last + 1:]
            if len(pending) >= 6:
                pending = collect(pending)
            if index % 8 == 0:
                print(f'{source.name}: {offset:,} rows dispatched to shards', flush=True)
        while pending:
            pending = collect(pending)


def as_number(value):
    if value == '':
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def typed_rows(exe, statement):
    return [{key: as_number(val) for key, val in row.items()} for row in sql_rows(exe, statement)]


def counts_summary(exe, table):
    details = []
    for name in VALUES:
        details.extend([
            f'count({name}) AS {name}_lisible',
            f'count(*) FILTER (WHERE empty_{name}) AS {name}_vide',
            f'count(*) FILTER (WHERE empty_component_{name}) AS {name}_composant_vide',
            f'count(*) FILTER (WHERE width = 9 AND {name} IS NULL) AS {name}_non_lisible_structure_standard',
        ])
    return typed_rows(exe, f"""
SELECT count(*) AS lignes_lues,
       count(*) FILTER (WHERE truncated) AS finales_tronquees,
       count(*) FILTER (WHERE NOT truncated AND t IS NULL) AS horodatages_invalides,
       count(*) FILTER (WHERE NOT truncated AND empty_time) AS horodatages_vides,
       count(*) FILTER (WHERE NOT truncated AND standard) AS format_standard,
       count(*) FILTER (WHERE NOT truncated AND NOT standard) AS format_non_standard,
       min(t) FILTER (WHERE NOT truncated) AS premier_tick,
       max(t) FILTER (WHERE NOT truncated) AS dernier_tick,
       count(*) FILTER (WHERE NOT truncated AND t < previous_t) AS reculs_adjacent,
       count(*) FILTER (WHERE NOT truncated AND t = previous_t) AS egalites_adjacentes,
       count(*) FILTER (WHERE NOT truncated AND t IS NOT NULL AND previous_t IS NOT NULL) AS paires_adjacentes_lisibles,
       count(*) FILTER (WHERE NOT truncated AND t < previous_t AND cast(t AS DATE) <> cast(previous_t AS DATE)) AS reculs_changement_date,
       min(date_diff('millisecond', previous_t, t)) FILTER (WHERE NOT truncated) AS pire_recul_ms
FROM {table};
""")[0] | typed_rows(exe, f"""
SELECT {', '.join(details)},
       count(*) FILTER (WHERE ask IS NOT NULL AND bid IS NOT NULL) AS paires_de_prix_lisibles,
       count(*) FILTER (WHERE ask <= 0) AS ask_non_positifs,
       count(*) FILTER (WHERE bid <= 0) AS bid_non_positifs,
       count(*) FILTER (WHERE ask <= 0 OR bid <= 0) AS ticks_prix_non_positifs,
       count(*) FILTER (WHERE ask < bid) AS spread_negatif,
       count(*) FILTER (WHERE ask = bid) AS spread_nul,
       min(ask - bid) AS spread_minimum,
       min(ask) AS ask_minimum, max(ask) AS ask_maximum,
       min(bid) AS bid_minimum, max(bid) AS bid_maximum
FROM {table} WHERE NOT truncated;
""")[0]


def volume_stats(exe, table, column):
    return typed_rows(exe, f"""
WITH frequencies AS (
    SELECT {column} AS v, count(*) AS n FROM {table}
    WHERE NOT truncated AND {column} IS NOT NULL GROUP BY 1
), ranked AS (
    SELECT *, sum(n) OVER () AS population,
           sum(n) OVER (ORDER BY v ROWS UNBOUNDED PRECEDING) AS cumul
    FROM frequencies
)
SELECT coalesce(sum(n), 0) AS observations, coalesce(sum(n) FILTER (WHERE v < 0),0) AS negatifs,
       coalesce(sum(n) FILTER (WHERE v = 0),0) AS zeros, min(v) AS minimum,
       sum(v*n)/sum(n) AS moyenne,
       sqrt(greatest(0, sum(v*v*n)/sum(n) - pow(sum(v*n)/sum(n),2))) AS ecart_type,
       max(v) AS maximum,
       min(v) FILTER (WHERE cumul*100 >= population) AS p01,
       min(v) FILTER (WHERE cumul*4 >= population) AS p25,
       min(v) FILTER (WHERE cumul*2 >= population) AS p50,
       min(v) FILTER (WHERE cumul*4 >= population*3) AS p75,
       min(v) FILTER (WHERE cumul*20 >= population*19) AS p95,
       min(v) FILTER (WHERE cumul*100 >= population*99) AS p99,
       coalesce(sum(n) FILTER (WHERE v > 0 AND v <= 0.0001),0) AS positif_jusque_00001,
       coalesce(sum(n) FILTER (WHERE v > 0.0001 AND v <= 0.001),0) AS de_00001_a_0001,
       coalesce(sum(n) FILTER (WHERE v > 0.001 AND v <= 0.01),0) AS de_0001_a_001,
       coalesce(sum(n) FILTER (WHERE v > 0.01 AND v <= 0.1),0) AS de_001_a_01,
       coalesce(sum(n) FILTER (WHERE v > 0.1),0) AS superieur_a_01
FROM ranked;
""")[0]


def per_calendar(daily, period):
    by_day = {date.fromisoformat(row['jour']): row['ticks'] for row in daily}
    first, last = min(by_day), max(by_day)
    periods = {}
    for offset in range((last - first).days + 1):
        day = first + timedelta(days=offset)
        key = day.isoformat()[:period]
        bucket = periods.setdefault(key, {'periode': key, 'ticks': 0, 'jours_couverts': 0,
                                         'jours_dans_fenetre': 0, 'lun_ven_sans_ticks': 0})
        count = by_day.get(day, 0)
        bucket['jours_dans_fenetre'] += 1
        bucket['ticks'] += count
        bucket['jours_couverts'] += count > 0
        bucket['lun_ven_sans_ticks'] += count == 0 and day.weekday() < 5
    return list(periods.values())


def duplicates(exe, partitions):
    rows = []
    for folder in sorted(partitions.glob('mois=*')):
        table = f"read_parquet({sql_string((folder / '*.parquet').as_posix())}, hive_partitioning=false)"
        row = typed_rows(exe, f"""
WITH distinct_rows AS (
    SELECT t, raw, count(*) AS n FROM {table} WHERE NOT truncated GROUP BY t, raw
), per_timestamp AS (
    SELECT t, sum(n) AS ticks, count(*) AS variants,
           sum(n - 1) AS repetitions, count(*) FILTER (WHERE n > 1) AS groupes
    FROM distinct_rows GROUP BY t
)
SELECT coalesce(sum(repetitions),0) AS copies_identiques_excedentaires,
       coalesce(sum(groupes),0) AS groupes_lignes_identiques,
       count(*) FILTER (WHERE t IS NOT NULL AND ticks > 1) AS horodatages_repetes,
       count(*) FILTER (WHERE t IS NOT NULL AND variants > 1) AS horodatages_avec_lignes_distinctes
FROM per_timestamp;
""")[0]
        rows.append({'mois': folder.name[5:]} | row)
    keys = rows[0].keys() - {'mois'}
    return {'totaux': {key: sum(row[key] for row in rows) for key in sorted(keys)}, 'par_mois': rows}


def summarize(exe, partitions, source, physical_complete, fragment):
    table = f"read_parquet({sql_string((partitions / '*/*.parquet').as_posix())}, hive_partitioning=false)"
    checks = counts_summary(exe, table)
    checks['lignes_physiques_donnees'] = physical_complete + bool(fragment)
    checks['lignes_terminees'] = physical_complete
    checks['lignes_ignorees_par_lecteur'] = physical_complete + bool(fragment) - checks['lignes_lues']
    widths = typed_rows(exe, f"""
SELECT width AS composants, count(*) AS lignes, min(raw) AS exemple,
       count(*) FILTER (WHERE starts_with(raw, ',') OR contains(raw, ',,') OR ends_with(raw, ',')) AS lignes_avec_composant_brut_vide
FROM {table} WHERE NOT truncated GROUP BY width ORDER BY width;
""")
    checks['lignes_avec_composant_brut_vide'] = sum(row['lignes_avec_composant_brut_vide'] for row in widths)
    daily = typed_rows(exe, f"""
SELECT cast(t AS DATE) AS jour, count(*) AS ticks, min(t) AS premier_tick, max(t) AS dernier_tick
FROM {table} WHERE NOT truncated AND t IS NOT NULL GROUP BY 1 ORDER BY 1;
""")
    covered_days = {date.fromisoformat(row['jour']) for row in daily}
    first, last = min(covered_days), max(covered_days)
    days = (first + timedelta(days=offset) for offset in range((last - first).days + 1))
    missing = [day.isoformat() for day in days if day.weekday() < 5 and day not in covered_days]
    gaps = typed_rows(exe, f"""
WITH gaps AS (
    SELECT t, previous_t, date_diff('millisecond', previous_t, t) AS gap_ms
    FROM {table} WHERE NOT truncated AND t > previous_t
)
SELECT count(*) FILTER (WHERE gap_ms >= 3600000 AND cast(t AS DATE) = cast(previous_t AS DATE)) AS intra_jour_ge_1h,
       count(*) FILTER (WHERE gap_ms >= 86400000) AS ge_24h,
       count(*) FILTER (WHERE gap_ms >= 259200000) AS ge_72h,
       max(gap_ms) AS max_ms
FROM gaps;
""")[0]
    gaps['principaux'] = typed_rows(exe, f"""
SELECT row_id + 1 AS ligne_csv, previous_t AS depuis, t AS jusque,
       date_diff('millisecond', previous_t, t) AS gap_ms,
       cast(t AS DATE) = cast(previous_t AS DATE) AS intra_jour
FROM {table} WHERE NOT truncated AND t > previous_t ORDER BY gap_ms DESC, ligne_csv LIMIT 25;
""")
    gaps['exemples_intra_jour'] = typed_rows(exe, f"""
SELECT row_id + 1 AS ligne_csv, previous_t AS depuis, t AS jusque,
       date_diff('millisecond', previous_t, t) AS gap_ms
FROM {table} WHERE NOT truncated AND cast(t AS DATE) = cast(previous_t AS DATE)
AND date_diff('millisecond', previous_t, t) >= 3600000
ORDER BY gap_ms DESC, ligne_csv LIMIT 25;
""")
    reversals = typed_rows(exe, f"""
SELECT row_id + 1 AS ligne_csv, previous_t AS precedent, t AS suivant,
       date_diff('millisecond', previous_t, t) AS delta_ms
FROM {table} WHERE NOT truncated AND t < previous_t ORDER BY ligne_csv LIMIT 25;
""")
    print(f'{source.name}: coverage and field checks completed', flush=True)
    volumes = {name: volume_stats(exe, table, name) for name in ('ask_volume', 'bid_volume')}
    print(f'{source.name}: volume distributions completed', flush=True)
    strict = (checks['reculs_adjacent'] == checks['egalites_adjacentes'] == checks['horodatages_invalides'] == 0
              and checks['paires_adjacentes_lisibles'] == physical_complete - 1
              and checks['lignes_ignorees_par_lecteur'] == 0)
    if strict:
        zero = dict.fromkeys(('copies_identiques_excedentaires', 'groupes_lignes_identiques',
                              'horodatages_repetes', 'horodatages_avec_lignes_distinctes'), 0)
        repeated = {'totaux': zero, 'par_mois': [{'mois': p.name[5:]} | zero for p in sorted(partitions.glob('mois=*'))],
                    'methode': 'unicite_prouvee_par_horodatages_strictement_croissants'}
    else:
        repeated = duplicates(exe, partitions)
        repeated['methode'] = 'groupement_exact_par_horodatage_et_ligne_brute'
    print(f'{source.name}: exact duplicates completed', flush=True)
    return {'fichier': source.name, 'octets': source.stat().st_size, 'controle': checks,
            'format': widths, 'couverture_annuelle': per_calendar(daily, 4),
            'couverture_mensuelle': per_calendar(daily, 7), 'couverture_journaliere': daily,
            'lun_ven_sans_ticks': missing, 'intervalles': gaps, 'exemples_reculs': reversals,
            'volumes': volumes, 'doublons': repeated}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parent.parent / 'data_trading_agent')
    parser.add_argument('--duckdb', type=Path, default=Path(os.environ.get('LOCALAPPDATA', '.')) / 'duckdb/cli/2.0.1/duckdb.exe')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--export', choices=FILES, required=True)
    args = parser.parse_args()
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = (args.data / FILES[args.export]).resolve()
    physical_complete, fragment = physical_layout(source)
    print(f'{source.name}: {physical_complete:,} newline-terminated rows; fragment={bool(fragment)}', flush=True)
    duckdb = args.duckdb.resolve(strict=True)
    version = sql_rows(duckdb, 'SELECT version() AS version;')[0]['version']
    with TemporaryDirectory(prefix=f'xauusd_{args.export}_', dir=args.scratch.resolve()) as temp, chdir(temp):
        partitions = Path(temp).resolve() / 'partitions'
        if source.stat().st_size >= 10 * 1024**3:
            stage_large(duckdb, source, partitions, fragment)
        else:
            stage(duckdb, source, partitions, fragment)
        print(f'{source.name}: DuckDB staging complete', flush=True)
        report = summarize(duckdb, partitions, source, physical_complete, fragment)
    temporary_output = args.out.with_suffix(args.out.suffix + '.tmp')
    temporary_output.write_text(json.dumps({'duckdb': version, **report}, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary_output.replace(args.out)
    print(args.out.resolve(), flush=True)


if __name__ == '__main__':
    main()
