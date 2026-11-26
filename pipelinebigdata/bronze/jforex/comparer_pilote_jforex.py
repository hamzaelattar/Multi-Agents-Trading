"""Compare the fixed XAUUSD pilot day without rewriting source ticks."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'audit'))
from fusionner_ticks import lire_entete, trouver_debut_recouvrement


UTC = timezone.utc
START = datetime(2021, 1, 5, tzinfo=UTC)
END = START + timedelta(days=1)
OFFSETS = range(-240, 241, 60)
LOCAL_FORMAT = '%Y.%m.%d %H:%M:%S.%f'
HEADER = ['timestamp_utc', 'ask', 'bid', 'ask_volume', 'bid_volume']
UTC_PATTERN = re.compile(r'2021-01-05T\d{2}:\d{2}:\d{2}\.\d{3}Z')
WHOLE = re.compile(rb'[+-]?\d+')
FRACTION = re.compile(rb'\d+')


def local_ticks(path):
    earliest = (START + timedelta(minutes=min(OFFSETS))).strftime(LOCAL_FORMAT)[:23].encode('ascii')
    latest = (END + timedelta(minutes=max(OFFSETS))).strftime(LOCAL_FORMAT)[:23].encode('ascii')
    rows = []
    with path.open('rb') as stream:
        origin = lire_entete(stream)
        stream.seek(trouver_debut_recouvrement(stream, origin, earliest))
        for raw in stream:
            if raw[:23] >= latest:
                break
            timestamp, *values = raw.rstrip(b'\r\n').split(b',')
            clock = datetime.strptime(timestamp.decode('ascii'), LOCAL_FORMAT).replace(tzinfo=UTC)
            pairs = list(zip(values[::2], values[1::2]))
            numbers = None
            if len(values) == 8 and all(WHOLE.fullmatch(a) and FRACTION.fullmatch(b) for a, b in pairs):
                numbers = tuple(Decimal((a + b'.' + b).decode('ascii')) for a, b in pairs)
            rows.append((clock, numbers))
    return rows


def provider_ticks(path):
    rows = []
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != HEADER:
            raise ValueError('Attendu : timestamp_utc,ask,bid,ask_volume,bid_volume')
        for row in reader:
            if len(row) != 5 or not UTC_PATTERN.fullmatch(row[0]):
                raise ValueError(f'Ligne {reader.line_num} : horodatage UTC ou colonnes non conformes.')
            timestamp = datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%S.%fZ').replace(tzinfo=UTC)
            numbers = tuple(Decimal(field) for field in row[1:])
            if not all(number.is_finite() for number in numbers):
                raise ValueError(f'Ligne {reader.line_num} : valeur décimale non finie.')
            rows.append((timestamp, numbers))
    if not rows:
        raise ValueError('Export fournisseur vide. La couverture reste non vérifiée.')
    return rows


def utc_window(rows, offset):
    delta = timedelta(minutes=offset)
    return [(clock - delta, quote) for clock, quote in rows if START + delta <= clock < END + delta]


def count_prices(rows):
    return Counter((t, quote[:2]) for t, quote in rows if quote is not None)


def matched(left, right):
    return sum((left & right).values())


def compare(local, provider):
    provider_times = Counter(t for t, _ in provider or [])
    provider_prices = count_prices(provider or [])
    results = []
    for offset in OFFSETS:
        window = utc_window(local, offset)
        standard = sum(quote is not None for _, quote in window)
        evidence = {'offset_minutes_local_minus_utc': offset, 'local_ticks': len(window),
                    'local_ambiguous_or_unreadable': len(window) - standard}
        if provider is not None:
            clocks = matched(Counter(t for t, _ in window), provider_times)
            standard_clocks = matched(Counter(t for t, q in window if q is not None), provider_times)
            prices = matched(count_prices(window), provider_prices)
            evidence |= {'timestamp_matches': clocks,
                         'local_without_reference_timestamp_match': len(window) - clocks,
                         'reference_without_local_timestamp_match': len(provider) - clocks,
                         'standard_timestamp_price_matches': prices,
                         'standard_timestamp_matches_without_same_prices': standard_clocks - prices}
        results.append(evidence)
    if provider is None:
        return {'offset_candidates': results}

    # Confirm all candidates sharing the maximum independently matched timestamp/price count.
    best_count = max(row['standard_timestamp_price_matches'] for row in results)
    strongest = [row for row in results if row['standard_timestamp_price_matches'] == best_count and best_count > 0]
    provider_quotes = Counter(provider)
    for row in strongest:
        window = utc_window(local, row['offset_minutes_local_minus_utc'])
        row['exact_timestamps_prices_volumes_by_local_volume_factor'] = {}
        for exponent in (-6, -3, 0, 3, 6):
            factor = Decimal(10) ** exponent
            local_quotes = Counter((t, (*q[:2], q[2] * factor, q[3] * factor))
                                   for t, q in window if q is not None)
            row['exact_timestamps_prices_volumes_by_local_volume_factor'][f'1e{exponent}'] = matched(
                local_quotes, provider_quotes)
    return {'offset_candidates': results, 'highest_price_match_count': best_count,
            'strongest_offsets_minutes': [row['offset_minutes_local_minus_utc'] for row in strongest],
            'same_day_only': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local', action='append', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    reference = provider_ticks(args.reference) if args.reference else None
    report = {'status': 'compared_pending_validation' if reference else 'awaiting_official_export',
              'window_utc': [START.isoformat(), END.isoformat()],
              'reference_ticks': len(reference) if reference else None,
              'local_exports': {source.name: compare(local_ticks(source), reference) for source in args.local}}
    if reference:
        report['reference_sha256'] = hashlib.sha256(args.reference.read_bytes()).hexdigest()
        report['reference_exact_duplicates'] = len(reference) - len(Counter(reference))
        report['reference_same_timestamp_excess'] = len(reference) - len(Counter(t for t, _ in reference))
    with args.output.open('x', encoding='utf-8') as output:
        json.dump(report, output, indent=2)
        output.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
