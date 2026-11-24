"""Summarize official demo checks and append a dated verification section to the Bronze HTML."""

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import re
import runpy
import sys


DAYS = ('2021-01-05', '2022-07-05', '2007-03-20', '2007-04-10', '2025-08-05')
LONG_DATE = '2026-11-24'
SECTION = 'verifications-jforex-2026-11-24'


def count_minutes(rows):
    return len({t.replace(second=0, microsecond=0) for t, _ in rows})


def count_hours(rows):
    return Counter(t.hour for t, _ in rows)


def number(n):
    return f'{n:,}'.replace(',', ' ') if isinstance(n, int) else str(n)


def table(columns, rows):
    header = ''.join(f'<th scope="col">{escape(c)}</th>' for c in columns)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(number(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>'


def read_export(module, file):
    with file.open('rb') as stream:
        stream.readline()
        if not stream.read(1):
            return []
    return module['provider_ticks'](file)


def inspect_dates(root, journal):
    repo = root / 'github'
    module = runpy.run_path(str(repo / 'bronze/jforex/comparer_pilote_jforex.py'))
    data = root / 'data_trading_agent'
    old = data / 'XAUUSD_Ticks_2004.01.01_2021.10.15.csv'
    recent = data / 'XAUUSD_Ticks_2021.01.04_2026.11.20.csv'
    bronze = root / 'bronze_dukascopy'
    followup = bronze / f'verifications_jforex_{LONG_DATE}'
    initial = bronze / 'pilote_jforex/XAUUSD_2021-01-05_JForex_UTC.csv'
    results = {}
    for day in DAYS:
        start = datetime.fromisoformat(day).replace(tzinfo=timezone.utc)
        module['compare'].__globals__.update(START=start, END=start + timedelta(days=1),
                                          UTC_PATTERN=re.compile(re.escape(day) + r'T\d{2}:\d{2}:\d{2}\.\d{3}Z'))
        source = old if day.startswith('2007') else recent
        local = module['local_ticks'](source)
        file = followup / f'{day}.csv'
        seen = [row for row in journal if row['date_utc'] == day]
        rows = read_export(module, file) if file.is_file() else []
        assert len(rows) == sum(int(row['ticks']) for row in seen)
        reversals = sum(a[0] > b[0] for a, b in zip(rows, rows[1:]))
        added_count = len(rows)
        inputs = [file] if file.is_file() else []
        if day == DAYS[0]:
            original = read_export(module, initial)
            reversals += sum(a[0] > b[0] for a, b in zip(original, original[1:]))
            rows = original + rows
            inputs.insert(0, initial)
        comparison = module['compare'](local, rows if rows else None)
        offsets = comparison.get('strongest_offsets_minutes', [])
        strongest = [row for row in comparison['offset_candidates'] if row['offset_minutes_local_minus_utc'] in offsets]
        chosen = strongest[0] if len(strongest) == 1 else None
        hours = count_hours(rows)
        reference_counts = Counter(t for t, _ in rows)
        local_window = module['utc_window'](local, offsets[0]) if chosen else None
        if chosen:
            local_hours = count_hours(local_window)
            head_to_head = sum(local_hours[h] for h in hours)
        else:
            local_hours = None
            head_to_head = None
        results[day] = {
            'requested_hours': 6 if day == DAYS[0] else 24, 'attempted_hours': len(seen),
            'followup_ticks': added_count, 'jforex_total_ticks': len(rows), 'jforex_hours': len(hours),
            'jforex_minutes': count_minutes(rows), 'jforex_hours_counts': dict(sorted(hours.items())),
            'same_timestamp_excess': len(rows) - len(reference_counts),
            'exact_duplicates': len(rows) - len(Counter(rows)),
            'jforex_nonpositive_prices': sum(q[0] <= 0 or q[1] <= 0 for _, q in rows),
            'jforex_negative_spread': sum(q[0] < q[1] for _, q in rows),
            'jforex_negative_volume': sum(q[2] < 0 or q[3] < 0 for _, q in rows),
            'jforex_reversals_in_export_order': reversals,
            'jforex_first_utc': min(reference_counts).isoformat() if rows else None,
            'jforex_last_utc': max(reference_counts).isoformat() if rows else None,
            'best_comparison': chosen, 'local_total_on_jforex_covered_hours': head_to_head,
            'local_hours_counts_utc': dict(sorted(local_hours.items())) if chosen else None,
            'local_ticks_calendar_as_labelled': len(module['utc_window'](local, 0)),
            'local_min_ticks_over_candidate_offsets': min(r['local_ticks'] for r in comparison['offset_candidates']),
            'local_max_ticks_over_candidate_offsets': max(r['local_ticks'] for r in comparison['offset_candidates']),
            'aggregate_offset_comparison': comparison,
            'status_log': seen,
            'files': [{'filename': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs],
        }
    return results


def paragraph(day, r):
    first = f'<p><strong>{escape(day)}</strong> · {number(r["jforex_total_ticks"])} ticks JForex dans {r["jforex_hours"]} heures.'
    comparison = r['best_comparison']
    if comparison:
        delta = comparison['offset_minutes_local_minus_utc']
        clocks = comparison['timestamp_matches']
        prices = comparison['standard_timestamp_price_matches']
        volumes = comparison['exact_timestamps_prices_volumes_by_local_volume_factor']['1e0']
        with_reference_hours = r['local_total_on_jforex_covered_hours']
        return (first + f' Meilleur décalage local − UTC : {delta:+d} min, sur {number(prices)} prix exactement appariés. '
                f'{number(clocks)} horodatages concordent, {number(volumes)} lignes concordent aussi sur les deux volumes sans facteur. '
                f'Les heures renseignées par JForex contiennent {number(with_reference_hours)} ticks dans le CSV local. '
                'Le décalage déduit concerne cette journée et les ticks appariés.</p>')
    if r['local_max_ticks_over_candidate_offsets'] == 0 and r['jforex_total_ticks']:
        return first + ' Le CSV local ne contient aucun tick pour les fenêtres candidates. Les ticks JForex constituent des données nouvelles pour cette journée. Une deuxième source reste nécessaire pour établir la complétude.</p>'
    if not r['jforex_total_ticks']:
        return first + ' La source ne fournit aucune référence exploitable dans cet essai. Une réponse vide laisse la couverture non vérifiée.</p>'
    return first + ' Aucun décalage unique ne ressort des rapprochements horodatage et prix.</p>'


def build_section(results, precision):
    metrics = []
    for day, r in results.items():
        hit = r['best_comparison']
        local = hit['local_ticks'] if hit else r['local_ticks_calendar_as_labelled']
        amount = r['followup_ticks']
        metrics.append((day, f'{r["attempted_hours"]}/{r["requested_hours"]}', amount, r['jforex_total_ticks'],
                        local, r['jforex_hours'], hit['offset_minutes_local_minus_utc'] if hit else 'N/D'))
    shifts = ''.join(paragraph(day, r) for day, r in results.items())
    coverage = table(['Date UTC', 'Heures testées', 'Nouveaux ticks de cette vérification', 'Total JForex disponible',
                      'Ticks locaux de la fenêtre', 'Heures avec ticks JForex', 'Décalage apparié (min)'], metrics)
    h = results[DAYS[0]]
    clock = h['local_hours_counts_utc'] or {}
    provider = h['jforex_hours_counts']
    missing_hours = [f'{hh:02d} h' for hh in range(24) if clock.get(hh, 0) > provider.get(hh, 0)]
    summaries = [(date, filename['filename'], filename['sha256']) for date, item in results.items() for filename in item['files']]
    checks = table(['Date UTC', 'Doublons exacts', 'Reculs', 'Prix ≤ 0', 'Spreads négatifs', 'Volumes négatifs'],
                   [(day, r['exact_duplicates'], r['jforex_reversals_in_export_order'], r['jforex_nonpositive_prices'],
                     r['jforex_negative_spread'], r['jforex_negative_volume']) for day, r in results.items() if r['jforex_total_ticks']])
    hour_rows = [(day, f'{int(row["heure_utc"]):02d} h', row['statut'], int(row['ticks']))
                 for day, info in results.items() for row in info['status_log']]
    return f'''
<section id="{SECTION}">
<h2>Vérifications JForex DEMO du 24 novembre 2026</h2>
<p>L’export initial du 5 janvier 2021, les nouveaux lots et les CSV historiques sont conservés séparément. Les appels officiels <code>IHistory.getTicks()</code> couvrent des heures UTC isolées. Un retour vide constitue une heure non vérifiée et ne prouve pas une fermeture.</p>
{coverage}
<p>Lorsque l’offset n’est pas identifié, le nombre local se rapporte à la date telle qu’elle est étiquetée dans l’export. Les résultats de couverture sont des observations sur les fichiers obtenus.</p>
{shifts}
<p>Le 5 janvier 2021, heures déficitaires par rapport aux originaux après alignement UTC+2 : {escape(', '.join(missing_hours) or 'aucune')} UTC. Le résultat de reprise est enregistré séparément du fichier pilote.</p>
<h3>Rupture de fréquence en 2007</h3>
<p>Le changement de fréquence des CSV locaux est établi par l’audit antérieur. Les deux journées ci-dessus servent de sondes. Une concordance sur les heures livrées ne suffit pas à identifier la cause d’une rupture de régime sur plusieurs mois. Les horaires manquants du canal et une éventuelle différence de filtrage doivent rester distingués.</p>
<h3>Décimales ambiguës et volumes du pilote initial</h3>
<p>Les 122 horodatages ambigus appariés sont uniques dans le pilote. Le test énumère les découpages de la ligne brute en quatre nombres avec partie entière ou décimale. <strong>{precision['ambiguous_exact_unique']} lignes</strong> ont un seul découpage égal terme à terme aux quatre valeurs JForex. {precision['ambiguous_unresolved']} ligne ne possède pas un découpage exact unique et {precision['ambiguous_without_reference']} autres lignes ambiguës de la journée restent sans référence. Les preuves par ligne et leur SHA-256 sont sauvegardées en local. Aucune écriture dans les originaux.</p>
<p>Sur les 36 écarts de volume, 32 touchent Bid et 4 Ask. Le volume local est inférieur dans 33 cas et supérieur dans 3. Les écarts absolus vont de 0,000002 à 0,000005. La troncature à cinq décimales reproduit {precision['volume_rounding_tests']['down']} valeurs locales et l’arrondi HALF_UP en reproduit {precision['volume_rounding_tests']['half_up']}. Ces résultats écartent une règle uniforme pour les 36 lignes. Les deux valeurs sont conservées dans le rapport de preuve local.</p>
<h3>Indicateurs qualité de la référence JForex</h3>{checks}
<h3>Comparaison entre canaux</h3>
{table(['Canal', 'État et comparaison possible'], [('JForex Java SDK DEMO','Lots partiels disponibles et rapprochés avec les CSV locaux ci-dessus'), ('Dukascopy-node / flux HTTP historique','Échecs 403/429 déjà documentés, aucun fichier comparable'), ('JForex 4 Desktop','Aucun export distinct disponible pour une comparaison canal à canal'), ('S3 Requester Pays','Préparé hors ligne, utilisateur n’a pas autorisé la requête')])}
<p>Le pilote S3 préparé cible un seul objet quotidien du 5 janvier 2021 à partir de la <a href="https://wiki.dukascopy.com/en/development/data-export/">convention Dukascopy</a>. Estimation <strong>0,009 USD</strong> hors taxes et avant franchise pour un HEAD et un GET Standard au plus 100 Mo. Aucun accès au bucket n’a eu lieu. <a href="https://aws.amazon.com/s3/pricing/">Tarifs AWS</a> · <a href="https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html">Requester Pays</a></p>
<h3>Provenance et décision Bronze</h3>{table(['Journée UTC','Export local','SHA-256'], summaries)}
<details><summary>Journal horaire des demandes officielles</summary>{table(['Jour UTC','Heure','Statut','Ticks'], hour_rows)}</details>
<p>Les lots utiles demeurent des références Bronze avec validation en attente. Le pilote d’une journée et l’absence de comparaison entre canaux ne suffisent pas à autoriser une collecte massive ou le remplacement des historiques existants.</p>
</section>'''


def main(root, baseline, output):
    followup = root / f'bronze_dukascopy/verifications_jforex_{LONG_DATE}'
    with (followup / 'heures.tsv').open(encoding='utf-8', newline='') as stream:
        journal = list(csv.DictReader(stream, delimiter='\t'))
    results = inspect_dates(root, journal)
    proof = json.loads((followup / 'preuves_decimales_2021-01-05.json').read_text(encoding='utf-8'))
    report = baseline.read_text(encoding='utf-8')
    assert SECTION not in report and '</main></body></html>' in report
    report = report.replace('<h1>Audit Bronze XAUUSD · 2004–2026</h1>',
                            '<h1>Audit Bronze XAUUSD · 2004–2026</h1>'
                            f'<p><a href="#{SECTION}">Voir les vérifications JForex et les statuts par canal</a>.</p>')
    report = report.replace('</main></body></html>', build_section(results, proof['counts']) + '</main></body></html>')
    output.write_text(report, encoding='utf-8')
    (followup / 'comparaisons_verification.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    for day, info in results.items():
        best = info['best_comparison']
        if best:
            best = {k: best[k] for k in ['offset_minutes_local_minus_utc', 'local_ticks', 'timestamp_matches',
                                        'standard_timestamp_price_matches', 'exact_timestamps_prices_volumes_by_local_volume_factor']}
        print(day, info['attempted_hours'], info['jforex_total_ticks'], info['jforex_hours'], best)


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:4]))
