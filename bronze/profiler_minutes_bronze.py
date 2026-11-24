"""Count observed minutes in the original XAUUSD files without changing either CSV."""

import argparse
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


FILES = {
    'ancien': 'XAUUSD_Ticks_2004.01.01_2021.10.15.csv',
    'recent': 'XAUUSD_Ticks_2021.01.04_2026.11.20.csv',
}
OVERLAP_START = '2021.01.04 01:00:00.413'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def query(exe, sql, cwd):
    result = subprocess.run([str(exe), '-batch', '-bail', '-csv'], input=sql,
                            encoding='utf-8', capture_output=True, cwd=cwd)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def profile(source, target, duckdb, kind):
    # The recent file replaces the independently verified common suffix of the old file.
    filter_sql = f"WHERE left(raw, 23) < {literal(OVERLAP_START)}" if kind == 'ancien' else ''
    with TemporaryDirectory(prefix='xauusd_minutes_', dir=target.parent) as scratch:
        temporary = Path(scratch) / 'minutes.parquet'
        print(f'Reading {source.name}, {source.stat().st_size:,} bytes', flush=True)
        print(query(duckdb, f"""
SET threads=4;
SET memory_limit='4GB';
SET preserve_insertion_order=false;
COPY (
    WITH counts AS (
        SELECT left(raw, 16) AS minute_texte, count(*) AS ticks,
               count(*) FILTER (WHERE length(raw) - length(replace(raw, ',', '')) <> 8) AS largeurs_atypiques
        FROM read_csv({literal(source.as_posix())}, columns={{'raw':'VARCHAR'}},
                      delim='|', quote='', escape='', header=true, auto_detect=false, parallel=true)
        {filter_sql}
        GROUP BY 1
    )
    SELECT strptime(minute_texte, '%Y.%m.%d %H:%M') AS minute_exportee, ticks, largeurs_atypiques
    FROM counts ORDER BY minute_exportee
) TO {literal(temporary.as_posix())} (FORMAT PARQUET, COMPRESSION zstd);
SELECT count(*) AS minutes_observees, sum(ticks) AS ticks,
       sum(largeurs_atypiques) AS lignes_largeur_atypique, min(minute_exportee) AS premiere_minute,
       max(minute_exportee) AS derniere_minute
FROM read_parquet({literal(temporary.as_posix())});
""", scratch).strip(), flush=True)
        temporary.rename(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--duckdb', type=Path, required=True)
    parser.add_argument('--export', choices=FILES, required=True)
    args = parser.parse_args()
    source = (args.data / FILES[args.export]).resolve(strict=True)
    target = args.out.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    profile(source, target, args.duckdb.resolve(strict=True), args.export)


if __name__ == '__main__':
    main()
