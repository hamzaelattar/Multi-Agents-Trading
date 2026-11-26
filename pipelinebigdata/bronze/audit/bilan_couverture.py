"""Report annual coverage without adding observations from different sources."""

import argparse
import csv
from html import escape
import json
import re
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write_coverage(data):
    archive = data / 'recovered/autres_fournisseurs'
    histdata = archive / 'histdata'
    reports = data / 'reports'
    baseline_path = data / 'source_runs/audit_local/bilan_annuel.csv'
    with baseline_path.open(encoding='utf-8-sig', newline='') as stream:
        baseline = list(csv.DictReader(stream))

    output = reports / 'Couverture_annuelle_XAUUSD.csv'
    pending = output.with_suffix('.tmp')
    gains_path = reports / 'Apport_minutes_UTC_XAUUSD.json'
    gains = read_json(gains_path) if gains_path.is_file() else {}
    histdata_gains_path = reports / 'Apport_minutes_HistData_XAUUSD.json'
    historic_gains = read_json(histdata_gains_path) if histdata_gains_path.is_file() else {}
    table = []
    for row in baseline:
        year = row['année']
        monthly = [read_json(path) for path in sorted((histdata / year).glob('*/HISTDATA_*.audit.json'))]
        raw_gain = historic_gains.get(year, {})
        if raw_gain.get('mois_audites') != len(monthly):
            raw_gain = {}
        mirror_dir = archive / 'touhid_xauusd' / year
        mirror_path = mirror_dir / 'audit_annuel.json'
        if not mirror_path.exists():
            mirror_path = mirror_dir / 'semaines_publiques/audit_annuel.json'
        mirror = read_json(mirror_path) if mirror_path.is_file() else {}
        carlos_path = archive / 'carlos_silva_xauusd' / f'year={year}/audit_annuel.json'
        carlos = read_json(carlos_path) if carlos_path.is_file() else {}
        table.append({
            'annee': year,
            'minutes_utc_absentes_des_deux_hypotheses_locales': gains.get(year, {}).get('minutes_inedites_utc'),
            'ticks_dans_ces_minutes': gains.get(year, {}).get('ticks_dans_minutes_inedites'),
            'ticks_locaux': row['ticks locaux'],
            'jours_locaux': row['jours avec ticks'],
            'minutes_locales': row['minutes avec ticks'],
            'mois_histdata_audites': len(monthly),
            'liste_mois_histdata': ' '.join(month['month'] for month in monthly),
            'ticks_histdata_bruts': sum(month['ticks'] for month in monthly) if monthly else None,
            'minutes_histdata_absentes_sous_6h_7h_8h': raw_gain.get('minutes_sans_locale'),
            'ticks_histdata_dans_ces_minutes': raw_gain.get('ticks_dans_minutes_sans_locale'),
            'jours_histdata_bruts': sum(month['dates'] for month in monthly) if monthly else None,
            'minutes_histdata_brutes': sum(month['minutes'] for month in monthly) if monthly else None,
            'debut_histdata_brut': monthly[0]['first_raw'] if monthly else None,
            'fin_histdata_brut': monthly[-1]['last_raw'] if monthly else None,
            'ticks_archive_independante': mirror.get('ticks'),
            'debut_archive_utc': mirror.get('first_utc'),
            'fin_archive_utc': mirror.get('last_utc'),
            'ticks_archive_carlos_silva': carlos.get('ticks'),
            'debut_archive_carlos_utc': carlos.get('first_utc'),
            'fin_archive_carlos_utc': carlos.get('last_utc'),
            'dossier_histdata': (histdata / year).relative_to(data).as_posix() if monthly else None,
            'dossier_archive_independante': (mirror_path.parent / 'brut').relative_to(data).as_posix() if mirror else None,
            'dossier_archive_carlos': carlos_path.parent.relative_to(data).as_posix() if carlos else None,
        })

    with pending.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]), delimiter=';')
        writer.writeheader()
        writer.writerows(table)
    pending.replace(output)
    print(output)
    return table


def write_html(table, report):
    def cell(value):
        if isinstance(value, int):
            return f'{value:,}'.replace(',', ' ')
        return escape(str(value)) if value is not None else '—'

    recent = []
    for row in table:
        if int(row['annee']) < 2023:
            continue
        mirror = row['ticks_archive_independante'] is not None
        recent.append((row['annee'], int(row['ticks_locaux']),
                       row['ticks_archive_independante'] if mirror else row['ticks_archive_carlos_silva'],
                       'Touhid-HZ' if mirror else 'Carlos Silva',
                       row['fin_archive_utc'] if mirror else row['fin_archive_carlos_utc'],
                       row['minutes_utc_absentes_des_deux_hypotheses_locales']))
    rows = ''.join('<tr>' + ''.join(f'<td>{cell(value)}</td>' for value in row) + '</tr>' for row in recent)
    months = sum(row['mois_histdata_audites'] for row in table)
    section = f"""
<section id="couverture-archives-utc">
<h2>Historique complémentaire audité par année</h2>
<p>Les fichiers d’origine et les archives indépendantes sont disponibles dans <code>data</code>. Les comptes ci-dessous se chevauchent. Les archives sont conservées par provenance, avec leurs empreintes et leurs audits DuckDB.</p>
<div class="scroll"><table><thead><tr><th>Année</th><th>Ticks locaux</th><th>Ticks de l’archive UTC</th><th>Édition</th><th>Dernier tick UTC</th><th>Minutes absentes des deux hypothèses locales</th></tr></thead>
<tbody>{rows}</tbody></table></div>
<p>La comparaison des minutes considère les décalages locaux UTC+2 et UTC+3. Une minute est comptée si elle est absente dans les deux hypothèses. Le bilan <code>data/reports/Couverture_annuelle_XAUUSD.csv</code> indique les sources et les bornes par année.</p>
<p>HistData met à disposition des lots mensuels XAU/USD depuis mars 2009. {cell(months)} lots sont audités dans le projet et conservés avec leur horloge brute. Les volumes de cotation y sont indisponibles. Le 23 novembre 2026 est documenté par un lot JForex partiel de 500 254 ticks. Les huit heures ciblées lors du suivi sont restées vides.</p>
</section>"""
    before = report.read_text(encoding='utf-8')
    marker = r'<section id="couverture-archives-utc">.*?</section>'
    if re.search(marker, before, flags=re.S):
        after = re.sub(marker, lambda match: section.strip(), before, count=1, flags=re.S)
    else:
        closing = '</main></body></html>'
        if before.count(closing) != 1:
            raise ValueError(f'Structure HTML inattendue : {report}')
        after = before.replace(closing, section + closing)
    report.write_text(after, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--html', type=Path)
    args = parser.parse_args()
    rows = write_coverage(args.data)
    if args.html:
        write_html(rows, args.html)
