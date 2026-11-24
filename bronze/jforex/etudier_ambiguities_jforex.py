"""Reconcile the first January pilot with exact decimal arithmetic and no writes to the sources."""

from collections import Counter
from datetime import timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_UP
import hashlib
import json
from pathlib import Path
import runpy
import sys


def encodings(tokens, width=4):
    if width == 0:
        if not tokens:
            yield ()
        return
    if not tokens or not tokens[0].lstrip(b'+-').isdigit():
        return
    for size in (1, 2):
        if size > len(tokens) or not width - 1 <= len(tokens) - size <= 2 * (width - 1):
            continue
        if size == 2 and not tokens[1].isdigit():
            continue
        value = Decimal(b'.'.join(tokens[:size]).decode('ascii'))
        for tail in encodings(tokens[size:], width - 1):
            yield (value, *tail)


def main(root, output):
    repo = root / 'github'
    original = root / 'data_trading_agent'
    pilot = root / 'bronze_dukascopy/pilote_jforex/XAUUSD_2021-01-05_JForex_UTC.csv'
    module = runpy.run_path(str(repo / 'bronze/jforex/comparer_pilote_jforex.py'))
    reference = module['provider_ticks'](pilot)
    providers = Counter(t for t, _ in reference)
    by_time = dict(reference)
    offset = timedelta(hours=2)
    start, end = (module[k] + offset for k in ('START', 'END'))
    label_start, label_end = [clock.strftime(module['LOCAL_FORMAT'])[:23].encode('ascii') for clock in (start, end)]

    files = sorted(original.glob('XAUUSD_Ticks_*.csv'))
    assert len(files) == 2
    with files[0].open('rb') as a, files[1].open('rb') as b:
        block = []
        for file in (a, b):
            header = module['lire_entete'](file)
            file.seek(module['trouver_debut_recouvrement'](file, header, label_start))
            rows = []
            while (row := file.readline()) and row[:23] < label_end:
                rows.append(row)
            block.append(rows)
    assert block[0] == block[1] and len(block[0]) == 183840
    local_rows = module['utc_window'](module['local_ticks'](files[1]), 120)
    repairs = []
    volume = []
    decimal_results = Counter()
    for (time, quote), raw in zip(local_rows, block[1], strict=True):
        if time not in by_time or providers[time] != 1:
            continue
        provider = by_time[time]
        if quote is None:
            tokens = raw.rstrip(b'\r\n').split(b',')[1:]
            choices = list(encodings(tokens))
            exact = sum(candidate == provider for candidate in choices)
            repairs.append({'timestamp_utc': time.isoformat(), 'source_raw_sha256': hashlib.sha256(raw).hexdigest(),
                            'splits': len(choices), 'exact_splits': exact,
                            'status': 'exact_unique' if exact == 1 else 'unresolved',
                            'reference_values': [str(v) for v in provider]})
        else:
            assert quote[:2] == provider[:2]
            for side, left, right in zip(('ask_volume', 'bid_volume'), quote[2:], provider[2:]):
                if left == right:
                    continue
                decimal_results[side] += 1
                decimal_results['local_lower_than_jforex' if left < right else 'local_higher_than_jforex'] += 1
                step = Decimal('0.00001')
                for name, rounding in [('half_up', ROUND_HALF_UP), ('half_even', ROUND_HALF_EVEN),
                                       ('down', ROUND_DOWN), ('up', ROUND_UP)]:
                    decimal_results[name] += left == right.quantize(step, rounding=rounding)
                volume.append({'timestamp_utc': time.isoformat(), 'field': side,
                               'local': str(left), 'jforex': str(right), 'delta_local_minus_jforex': str(left - right)})
    counts = {'ambiguous_apparied': len(repairs),
              'ambiguous_exact_unique': sum(r['exact_splits'] == 1 for r in repairs),
              'ambiguous_unresolved': sum(r['exact_splits'] != 1 for r in repairs),
              'ambiguous_without_reference': sum(q is None for _, q in local_rows) - len(repairs),
              'volume_differences': len(volume), 'volume_rounding_tests': dict(decimal_results)}
    assert len(repairs) == 122 and len(volume) == 36
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'origin': [f.name for f in files], 'jforex_sha256': hashlib.sha256(pilot.read_bytes()).hexdigest(),
                   'counts': counts, 'repair_evidence': repairs, 'volume_evidence': volume}, stream, indent=2)
        stream.write('\n')
    print(json.dumps(counts, indent=2))


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:3]))
