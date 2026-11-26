"""Test raw HistData clock offsets against a designated JForex reference day."""

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def histdata_day(archive, start, offset):
    stop = start + timedelta(days=1)
    delta = timedelta(hours=offset)
    first, last = [(clock - delta).strftime('%Y%m%d %H%M%S').encode('ascii') for clock in (start, stop)]
    with zipfile.ZipFile(archive) as zipped:
        member, = [f for f in zipped.namelist() if f.lower().endswith('.csv')]
        with zipped.open(member) as stream:
            for line in stream:
                if line[:15] < first:
                    continue
                if line[:15] >= last:
                    break
                stamp, bid, ask, volume = line.decode('ascii').rstrip().split(',')
                t = datetime.strptime(stamp, '%Y%m%d %H%M%S%f').replace(tzinfo=timezone.utc) + delta
                yield t, Decimal(ask), Decimal(bid), Decimal(volume)


def times(rows):
    result = Counter()
    for (clock, _, _), count in rows.items():
        result[clock] += count
    return result


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def reference_day(file, start):
    with file.open(encoding='utf-8-sig', newline='') as stream:
        rows = Counter((datetime.fromisoformat(row['timestamp_utc']), Decimal(row['ask']), Decimal(row['bid']))
                       for row in csv.DictReader(stream))
    if not rows or any(not start <= t < start + timedelta(days=1) for t, _, _ in rows):
        raise ValueError('Le CSV JForex ne couvre pas la journée déclarée.')
    return rows


def main(args):
    start = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    jet = reference_day(args.reference, start)
    clocks = times(jet)
    candidates = []
    for offset in (4, 5):
        rows = list(histdata_day(args.archive, start, offset))
        h = Counter((t, a, b) for t, a, b, _ in rows)
        hou = Counter(t.hour for t, _, _, _ in rows)
        candidates.append({'utc_minus_raw_hours': offset, 'ticks': len(rows), 'tick_hours': dict(sorted(hou.items())),
                           'source_volume_nonzero': sum(v != 0 for _, _, _, v in rows),
                           'timestamp_matches': sum((times(h) & clocks).values()),
                           'timestamp_ask_bid_matches': sum((h & jet).values())})
    strongest = max(candidates, key=lambda row: row['timestamp_ask_bid_matches'])
    if sum(row['timestamp_ask_bid_matches']==strongest['timestamp_ask_bid_matches'] for row in candidates) != 1:
        strongest = None
    if args.candidate:
        if not strongest or strongest['timestamp_ask_bid_matches'] != sum(jet.values()):
            raise ValueError('Une référence JForex intégralement appariée est nécessaire pour créer un CSV candidat.')
        with args.candidate.open('x', encoding='utf-8', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['timestamp_utc', 'ask', 'bid', 'ask_volume', 'bid_volume'])
            for t, ask, bid, _ in histdata_day(args.archive, start, strongest['utc_minus_raw_hours']):
                writer.writerow([t.isoformat(timespec='milliseconds').replace('+00:00', 'Z'), ask, bid, None, None])
    summary = {'day': args.date, 'source': 'HistData – prices only', 'archive_sha256': sha256(args.archive),
               'jforex_sha256': sha256(args.reference), 'jforex_ticks': sum(jet.values()),
               'status': 'compared_pending_validation', 'results': candidates,
               'strongest': strongest, 'candidate_sha256': sha256(args.candidate) if args.candidate else None}
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2)
        stream.write('\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--date', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path)
    sys.exit(main(parser.parse_args()))
