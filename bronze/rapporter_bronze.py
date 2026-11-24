"""Build the Bronze baseline from minute aggregates; provider-dependent fields stay unverified."""

import argparse
import base64
import csv
from datetime import date, timedelta
from html import escape
import io
import json
from pathlib import Path
import statistics
import sys
from tempfile import TemporaryDirectory

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from profiler_minutes_bronze import literal, query


START = date(2004, 1, 1)
END = date(2026, 11, 23)
NEXT_DAY = END + timedelta(days=1)


def local_aggregates(exe, audit):
    files = [audit / f'minutes_{kind}.parquet' for kind in ('ancien', 'recent')]
    paths = ', '.join(literal(p.as_posix()) for p in files)
    start = literal(START.isoformat())
    finish = literal(NEXT_DAY.isoformat())
    statements = f"""
SET threads=4;
SET memory_limit='4GB';
CREATE VIEW minutes AS SELECT * FROM read_parquet([{paths}]);
CREATE TEMP TABLE heures AS
    SELECT date_trunc('hour', minute_exportee) AS heure, sum(ticks)::BIGINT AS ticks,
           count(*) AS minutes_observees, sum(largeurs_atypiques)::BIGINT AS largeurs_atypiques
    FROM minutes GROUP BY 1;
"""
    exports = {
        'heures_observees': 'SELECT * FROM heures ORDER BY heure',
        'heures_sans_ticks': f"""SELECT h AS heure
            FROM generate_series(TIMESTAMP {start}, TIMESTAMP {finish} - INTERVAL 1 HOUR, INTERVAL 1 HOUR) AS s(h)
            ANTI JOIN heures ON heures.heure = s.h ORDER BY h""",
        'jours_observes': """SELECT cast(minute_exportee AS DATE) AS jour, count(*) AS minutes_observees,
            count(DISTINCT date_trunc('hour', minute_exportee)) AS heures_observees,
            sum(ticks)::BIGINT AS ticks, sum(largeurs_atypiques)::BIGINT AS largeurs_atypiques,
            min(minute_exportee) AS premiere_minute, max(minute_exportee) AS derniere_minute,
            date_diff('minute', min(minute_exportee), max(minute_exportee)) + 1 - count(*) AS minutes_vides_dans_amplitude
            FROM minutes GROUP BY 1 ORDER BY 1""",
        'jours_sans_ticks': f"""SELECT cast(d AS DATE) AS jour
            FROM generate_series(DATE {start}, DATE {finish} - INTERVAL 1 DAY, INTERVAL 1 DAY) AS s(d)
            ANTI JOIN (SELECT DISTINCT cast(heure AS DATE) AS jour FROM heures) observed
            ON observed.jour = cast(s.d AS DATE) ORDER BY s.d""",
        'intervalles_sans_ticks': f"""WITH punctuated AS (
            SELECT minute_exportee AS t FROM minutes UNION ALL
            SELECT TIMESTAMP {start} - INTERVAL 1 MINUTE UNION ALL SELECT TIMESTAMP {finish}
        ), previous AS (SELECT t, lag(t) OVER (ORDER BY t) AS p FROM punctuated)
        SELECT p + INTERVAL 1 MINUTE AS premiere_minute_vide, t AS fin_exclusive,
               date_diff('minute',p,t)-1 AS minutes_vides
        FROM previous WHERE t > p + INTERVAL 1 MINUTE ORDER BY premiere_minute_vide""",
    }
    with TemporaryDirectory(prefix='resume_bronze_', dir=audit) as temp:
        scratch = Path(temp)
        for name, sql in exports.items():
            target = scratch / f'{name}.parquet'
            statements += f'COPY ({sql}) TO {literal(target.as_posix())} (FORMAT PARQUET, COMPRESSION zstd);\n'
        query(exe, statements, scratch)
        for name in exports:
            (scratch / f'{name}.parquet').replace(audit / f'{name}.parquet')
        def rows(sql):
            return list(csv.DictReader(io.StringIO(query(exe, sql, scratch))))
        daily = rows(f"SELECT * FROM read_parquet({literal((audit / 'jours_observes.parquet').as_posix())}) ORDER BY jour;")
        biggest = rows(f"SELECT * FROM read_parquet({literal((audit / 'intervalles_sans_ticks.parquet').as_posix())}) ORDER BY minutes_vides DESC, premiere_minute_vide LIMIT 30;")
    return daily, biggest


def yearly(daily):
    records = []
    for year in range(START.year, END.year + 1):
        active = [row for row in daily if row['jour'][:4] == str(year)]
        first = date(year, 1, 1)
        stop = min(date(year + 1, 1, 1), NEXT_DAY)
        days = (stop - first).days
        hours = sum(int(row['heures_observees']) for row in active)
        minutes = sum(int(row['minutes_observees']) for row in active)
        records.append({'année': year, 'ticks locaux': sum(int(row['ticks']) for row in active),
                        'largeur atypique': sum(int(row['largeurs_atypiques']) for row in active),
                        'jours avec ticks': len(active), 'jours sans observation': days - len(active),
                        'heures avec ticks': hours, 'heures sans observation': 24 * days - hours,
                        'minutes avec ticks': minutes, 'minutes internes vides': sum(int(row['minutes_vides_dans_amplitude']) for row in active),
                        'jours avec minutes internes vides': sum(int(row['minutes_vides_dans_amplitude']) > 0 for row in active),
                        'ticks téléchargés': 0, 'nouveaux ticks récupérés': 0, 'ticks identiques fournisseur': 'N/D',
                        'divergences fournisseur': 'N/D', 'lignes ambiguës réparées': 0,
                        'jours manquants confirmés avant': 'N/D', 'jours manquants confirmés après': 'N/D',
                        'heures manquantes confirmées avant': 'N/D', 'heures manquantes confirmées après': 'N/D',
                        'couverture sessions avant': 'N/D', 'couverture sessions après': 'N/D',
                        'période fournisseur vérifiée': 'N/D'})
    assert sum(row['ticks locaux'] for row in records) == 505_295_814
    return records


def low_frequency(daily):
    records, examples = [], []
    for year in (2007, 2008, 2009):
        active = [row for row in daily if row['jour'][:4] == str(year) and int(row['heures_observees']) >= 18]
        rates = sorted((int(row['ticks']) / int(row['heures_observees']), row['jour'], int(row['ticks']),
                        int(row['heures_observees'])) for row in active)
        decile = rates[(len(rates) - 1) // 10][0]
        records.append((year, len(rates), round(decile, 1), round(statistics.median(row[0] for row in rates), 1),
                        sum(row[0] <= decile for row in rates)))
        examples.extend((year, date, count, hours, round(rate, 1)) for rate, date, count, hours in rates[:10])
    return records, examples


def number(value):
    return f'{value:,}'.replace(',', '\u202f') if isinstance(value, int) else str(value)


def figure(records, daily):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.5, 4.4), constrained_layout=True)
    years = [row['année'] for row in records]
    left.bar(years, [row['minutes avec ticks'] / 1000 for row in records], color='#176970')
    left.set(title='Minutes locales avec cotation', ylabel='Milliers de minutes', xticks=years[::3])
    left.text(.01, .98, 'Après récupération : non mesuré', transform=left.transAxes,
              va='top', fontsize=9, color='#536575')
    buckets = {}
    for row in daily:
        year = int(row['jour'][:4])
        if 2007 <= year <= 2009 and int(row['heures_observees']) >= 18:
            buckets.setdefault(row['jour'][:7], []).append(int(row['ticks']) / int(row['heures_observees']))
    months = sorted(buckets)
    right.plot(months, [statistics.median(buckets[m]) for m in months], color='#176970', linewidth=2)
    right.set(title='Fréquence médiane locale en 2007–2009', ylabel='Ticks par heure observée')
    right.set_xticks(range(0, len(months), 6), [m for m in months[::6]], rotation=25)
    for axis in (left, right):
        axis.grid(True, axis='y', alpha=.22)
        axis.set_axisbelow(True)
    stream = io.BytesIO()
    fig.savefig(stream, format='png', dpi=140, facecolor='white')
    plt.close(fig)
    return base64.b64encode(stream.getvalue()).decode()


def table(headers, rows):
    head = ''.join(f'<th scope="col">{escape(str(value))}</th>' for value in headers)
    body = ''.join('<tr>' + ''.join(f'<td>{escape(number(value))}</td>' for value in row) + '</tr>' for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def build_report(daily, biggest, records, attempts, origins):
    low, samples = low_frequency(daily)
    early_2007 = [statistics.median(int(row['ticks']) / int(row['heures_observees'])
                                   for row in daily if row['jour'].startswith(month)
                                   and int(row['heures_observees']) >= 18)
                  for month in ('2007-03', '2007-04')]
    sources = table(['Fichier', 'Octets', 'SHA-256'], [(r['file'], r['bytes'], r['sha256']) for r in origins])
    network = table(['Canal', 'Période testée', 'Résultat'], [(r['channel'], r['period'], r['status']) for r in attempts])
    annual = table(list(records[0]), [list(row.values()) for row in records])
    largest = table(['Première minute vide', 'Fin exclusive', 'Minutes sans tick'], [list(row.values()) for row in biggest[:12]])
    low_table = table(['Année', 'Jours ≥ 18 h observées', 'P10 ticks/h', 'Médiane ticks/h', 'Jours ≤ P10'], low)
    sample_table = table(['Année', 'Jour', 'Ticks', 'Heures avec ticks', 'Ticks/h'], samples)
    image = figure(records, daily)
    covered_months = {row['jour'][:7] for row in daily}
    absent_months = ', '.join(f'{y}-{m:02d}' for y in range(START.year, END.year + 1)
                              for m in range(1, 13) if date(y, m, 1) <= END
                              and f'{y}-{m:02d}' not in covered_months)
    return f'''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Audit Bronze XAUUSD · 2004–2026</title>
<style>
body{{font:16px/1.55 system-ui,sans-serif;color:#18333c;max-width:1180px;margin:48px auto;padding:0 24px}}
h1{{font-size:2.45rem;line-height:1.1}} h2{{margin-top:2rem}} .note{{padding:16px;background:#e9f3f3;border-left:4px solid #176970}}
.scroll{{overflow:auto}} table{{border-collapse:collapse;min-width:65%;font-size:.88rem}} td,th{{padding:7px 9px;border-bottom:1px solid #dbe6e8;text-align:left;white-space:nowrap}} th{{background:#e9f3f3;white-space:normal;min-width:78px}}
img{{max-width:100%}} a{{color:#176970}} code{{overflow-wrap:anywhere}}
</style><body><main>
<h1>Audit Bronze XAUUSD · 2004–2026</h1>
<p>Fenêtre demandée du 1er janvier 2004 au 23 novembre 2026 inclus. Audit local du 24 novembre 2026.</p>
<p class="note"><strong>505 295 814 ticks locaux distincts par rapport au chevauchement déjà vérifié.</strong> Aucun tick fournisseur n’a été récupéré lors de cette exécution. Le rapport distingue les absences observées localement des pertes et couvertures qui exigent une comparaison fournisseur.</p>
<h2>Collecte et preuves d’accès</h2>{network}
<p>L’essai API a été effectué avec dukascopy-node 1.51.1 sur XAUUSD tick, du 5 au 6 janvier 2021 UTC (fin exclusive), avec cache. La page d’export s’ouvre mais ses demandes de catalogue Jetta échouent dans le navigateur. Le canal S3 documenté par Dukascopy exige des identifiants AWS et une facturation au demandeur.</p>
<p><a href="https://www.dukascopy-node.app/config/node">Paramètres dukascopy-node</a> · <a href="https://www.dukascopy.com/swiss/english/marketwatch/historical/">Export officiel</a> · <a href="https://wiki.dukascopy.com/en/development/data-export/">Export S3 Requester Pays</a></p>
<h2>Couverture locale avant récupération</h2>
<img src="data:image/png;base64,{image}" alt="Minutes avec ticks par année et fréquence médiane mensuelle entre 2007 et 2009">
<p>Les jours et heures sans observation utilisent le calendrier civil comme repère et comprennent les fermetures possibles. Les minutes internes vides sont des minutes sans tick entre la première et la dernière minute observées d’un jour. Ces mesures permettent de repérer des journées à examiner. Les métriques de perte et de couverture par session restent N/D. Le profil annuel exclut la partie commune de l’ancien fichier selon le seuil déjà vérifié de 2021.01.04 01:00:00.413. Les deux CSV d’origine restent séparés.</p>
<p>Mois civils entièrement sans observation locale dans la fenêtre : {escape(absent_months)}.</p>
<p>Les heures sont laissées dans le repère local étiqueté EET. L’offset et son éventuelle variation saisonnière restent à établir par comparaison de ticks hivernaux, estivaux et autour des changements d’heure. Les unités de volume des CSV d’origine restent à rapprocher des unités de la source.</p>
{annual}
<h2>Intervalles locaux sans observation</h2>{largest}
<p>La fin est exclusive et correspond à la minute de reprise, ou à la fin de la fenêtre demandée pour la bordure finale. Les intervalles sont calculés sur les minutes des exports, sans conversion UTC présumée.</p>
<h2>Fréquence en 2007–2009</h2>{low_table}
<p>La médiane quotidienne du taux horaire passe de {number(round(early_2007[0]))} ticks en mars 2007 à {number(round(early_2007[1]))} en avril 2007 parmi les jours avec au moins 18 heures observées. Ce changement doit être comparé au flux source.</p>
<p>Chaque ligne ci-dessous appartient à l’un des dix jours de fréquence la plus faible de son année parmi les jours avec au moins 18 heures observées. Une fréquence faible constitue un signal de revue, sans établir une perte.</p>{sample_table}
<h2>Provenance et vérifications locales</h2>{sources}
<p>Le dossier local <code>bronze_dukascopy/audit_local</code> contient les agrégats des minutes, heures et jours observés, les jours et heures sans observation et les intervalles représentant toutes les minutes sans observation. Les largeurs atypiques sont comptées par le nombre de séparateurs et les lignes originales restent intactes. Aucune valeur ambiguë n’a été réparée sur une hypothèse.</p>
<h2>Points restant à valider avant Silver</h2><ul>
<li>Valider la journée pilote auprès de la source, puis comparer des jours d’hiver, d’été et de changement d’heure pour établir le repère UTC et les facteurs de prix et de volume.</li>
<li>Comparer les collections de lignes complètes, avec multiplicité. Conserver les cotations distinctes partageant une milliseconde. Attribuer une correction uniquement à une observation source identifiable.</li>
<li>Établir les calendriers de session par période et les exceptions de jours fériés avant de calculer des taux de couverture. La <a href="https://www.dukascopy.com/swiss/english/forex/forex-trading-accounts/link/">grille actuelle</a> fournit un repère. L’<a href="https://www.dukascopy.com/swiss/english/about/ournews/-dbl194921/">annonce de 2011</a> illustre que le calendrier actuel ne peut pas être appliqué à toute la série.</li>
<li>Après validation de l’accès, contrôler aussi 2004–2022. Prioriser les coupures 2023–2026, les journées à faible fréquence et les lignes de largeur atypique, puis confirmer le reste auprès de la source.</li>
</ul></main></body></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', required=True, type=Path)
    parser.add_argument('--duckdb', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    audit = args.audit.resolve(strict=True)
    daily, biggest = local_aggregates(args.duckdb.resolve(strict=True), audit)
    annual = yearly(daily)
    with (audit / 'bilan_annuel.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annual[0]))
        writer.writeheader()
        writer.writerows(annual)
    origins = json.loads((audit / 'origines_sha256.json').read_text(encoding='utf-8'))
    attempts = json.loads((audit / 'acces_fournisseur.json').read_text(encoding='utf-8'))
    args.output.write_text(build_report(daily, biggest, annual, attempts, origins), encoding='utf-8')
    print(f'{args.output.resolve()} | {len(annual)} years, {len(daily)} observed days')


if __name__ == '__main__':
    sys.exit(main())
