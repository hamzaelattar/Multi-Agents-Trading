"""Append the separate-provider reconciliation and verified recovery counts to the Bronze HTML."""

import base64
import csv
from html import escape
import io
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rapporter_verifications_bronze import number, table


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def chart(august, before_jan, after_hist, nov_hours):
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    hours = list(range(24))
    axes[0].bar(hours, [after_hist.get(str(h), 0) for h in hours], color='#a8cbd0', label='HistData / local')
    axes[0].bar(hours, [before_jan.get(str(h), 0) for h in hours], color='#176970', label='JForex')
    axes[0].set_title('5 janvier 2021 UTC')
    axes[0].legend(fontsize=8, loc='upper left')
    axes[1].bar(hours, [august.get(str(h), 0) for h in hours], color='#176970')
    axes[1].set_title('5 août 2025 UTC · JForex')
    axes[2].bar(hours, [nov_hours.get(h, 0) for h in hours], color='#176970')
    axes[2].set_title('23 novembre 2026 UTC · JForex')
    for axis in axes:
        axis.set_xlabel('Heure UTC')
        axis.set_xticks([0, 6, 12, 18, 23])
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Ticks par heure')
    content = io.BytesIO()
    fig.savefig(content, format='png', dpi=140)
    plt.close(fig)
    return base64.b64encode(content.getvalue()).decode('ascii')


def main(root, source, output):
    repository = root / 'data'
    bronze = root / 'data/source_runs'
    hist = repository / 'recovered/autres_fournisseurs/histdata'
    compare = load(repository / 'reports/comparaisons_verification.json')
    winter = load(hist / '2021/01/comparaison_JForex_2021-01-05.json')
    summer = load(hist / '2025/08/comparaison_JForex_2025-08-05.json')
    august_audit = load(hist / '2025/08/audit_et_comparaison_2025-08.json')['month'][0]
    august_candidate = load(hist / '2025/08/XAUUSD_2025-08_HistData_prix_candidats.json')
    winter_local = load(hist / '2021/01/comparaison_local_2021-01-05.json')
    november = load(hist / '2026/11/inventaire_horloge_brute_2026-11.json')
    late = load(repository / 'manifests/JForex_2026-11-23.audit.json')
    late_hours = {h['hour']: h['ticks'] for h in late['requested_hours']}
    late_ticks = sum(late_hours.values())
    late_positive = sum(v > 0 for v in late_hours.values())
    late_minutes = late.get('summary', {}).get('minutes_with_ticks', 0)
    august = compare['2025-08-05']['jforex_total_ticks']
    august_minutes = compare['2025-08-05']['jforex_minutes']
    previous = 505_295_814
    assert winter['strongest']['timestamp_ask_bid_matches'] == winter['jforex_ticks'] == 146982
    assert summer['strongest']['timestamp_ask_bid_matches'] == summer['jforex_ticks'] == august
    assert winter['strongest']['ticks'] == 183840
    assert winter['strongest']['utc_minus_raw_hours'] == 5 and summer['strongest']['utc_minus_raw_hours'] == 4
    encoded = chart(compare['2025-08-05']['jforex_hours_counts'], compare['2021-01-05']['jforex_hours_counts'],
                    winter['strongest']['tick_hours'], late_hours)
    march = compare['2007-03-20']['jforex_hours_counts']
    april = compare['2007-04-10']['jforex_hours_counts']
    shared_hours = march.keys() & april.keys()
    a, b = [sum(counts[h] for h in shared_hours) for counts in (march, april)]
    archive_rows = [
        ('HistData 2021-01', 'Archive locale', 'Hiver : +5 h sur la journée du 5 janvier'),
        ('HistData 2025-08', number(august_audit['ticks']), 'Été : +4 h sur la journée du 5 août'),
        ('HistData 2026-11', number(november['ticks']), f'Dernière étiquette brute {november["last_raw"]}. Jour du 23 absent'),
        ('HistData 2007-03 et 2007-04', 'Aucune archive livrée', 'Formulaire public HTTP 200 sans ZIP'),
    ]
    with (bronze / 'audit_local/bilan_annuel.csv').open(encoding='utf-8', newline='') as stream:
        before = {int(row['année']): int(row['ticks locaux']) for row in csv.DictReader(stream)}
    assert sum(before.values()) == previous
    coverage = [(2025, before[2025], august, before[2025] + august, f'Août : 1 jour / {august_minutes} minutes UTC'),
                (2026, before[2026], late_ticks, before[2026] + late_ticks,
                 f'23 novembre : {late_positive} heures / {late_minutes} minutes UTC')]
    day_rows = [(f'{h:02d} h', compare['2021-01-05']['jforex_hours_counts'].get(str(h), 0),
                 winter['strongest']['tick_hours'].get(str(h), 0),
                 compare['2025-08-05']['jforex_hours_counts'].get(str(h), 0),
                 late_hours.get(h, 0)) for h in range(24)]
    section = f'''
<section id="reprise-multicanal">
<h2>Récupération par lots et rapprochement entre canaux</h2>
<p class="note"><strong>{number(august + late_ticks)} ticks Dukascopy supplémentaires</strong> ont été obtenus pour deux dates sans chevauchement avec les originaux. Le total des observations disponibles dans les lots séparés est de <strong>{number(previous + august + late_ticks)}</strong>. Il s’agit d’un total de disponibilité, avec validation des nouveaux lots en attente.</p>
{table(['Année','Base locale','Nouveaux ticks Dukascopy','Total disponible','Couverture ajoutée'], coverage)}
<p>Le tableau annuel initial conserve les effectifs des autres années. Les mesures par mois, jour et heure ci-dessous correspondent aux lots effectivement contrôlés.</p>
<img src="data:image/png;base64,{encoded}" alt="Couverture par heure UTC des lots du pilote et des deux jours récupérés">
{table(['Heure UTC','Jan. 2021 JForex','Jan. 2021 HistData / local','Août 2025 JForex','Nov. 2026 JForex'], day_rows)}
<h3>Canal public HistData et dérive de l’horloge</h3>
{table(['Archive testée','Observations ou état','Périmètre de validation'], archive_rows)}
<p>Le canal public <a href="https://www.histdata.com/download-free-forex-data/">HistData</a> fournit des cotations Bid/Ask à la milliseconde. Les données tiennent dans des archives mensuelles conservées à part. Le champ volume vaut zéro dans les contrôles effectués. Les deux volumes de cotation restent nuls au sens SQL dans le CSV comparatif. <a href="https://www.histdata.com/f-a-q/data-files-detailed-specification/">Format documenté</a></p>
<p>Le <strong>5 janvier 2021</strong>, ajouter 5 h à l’horloge brute reproduit exactement les 146 982 timestamps et paires Ask/Bid de JForex. HistData compte 183 840 paires au total et couvre les six heures absentes du pilote. Les originaux ont déjà des cotations pour ces heures. Le <strong>5 août 2025</strong>, ajouter 4 h reproduit exactement les 156 842 paires JForex. Avec 5 h, aucune paire n’est exactement appariée. Le fichier initial calculé avec 5 h est isolé dans <code>quarantine</code>. Aucun décalage annuel uniforme n’a été déduit de ces deux sondes.</p>
<p>Les six heures absentes du pilote JForex initial ont {number(winter_local['missing_jforex_hour_timestamps'])} timestamps concordants avec HistData et {number(winter_local['missing_jforex_hour_readable_prices'])} paires Ask/Bid locales lisibles concordantes. Les volumes ne sont pas vérifiables par HistData. Un lot séparé rassemble les 121 reconstructions décimales exactes, avec une preuve par ligne d’origine. Ces lignes existaient déjà et leur total reste compté une seule fois.</p>
<p>Le Parquet HistData d’août contient <strong>{number(august_candidate['ticks'])} cotations de prix candidates</strong> sur {august_candidate['days_with_ticks']} dates UTC et {august_candidate['hours_with_ticks']} heures. Il applique la calibration estivale comme hypothèse à l’archive, puis garde les dates UTC d’août. Sa validation reste en attente pour les jours hors de la sonde du 5 août. Les volumes demeurent NULL.</p>
<p>La concordance des prix ne démontre pas que les flux en amont sont indépendants. Les archives HistData sont conservées séparément, avec leur horloge et leur propre provenance. Le lot HistData d’août n’est pas compté une deuxième fois dans le total Dukascopy.</p>
<h3>Rupture en 2007</h3>
<p>Sur les {len(shared_hours)} heures UTC couvertes par JForex les deux jours, le 20 mars contient {number(a)} ticks et le 10 avril {number(b)}. Les prix lisibles appariés sont exacts après alignement du fuseau. La baisse est aussi visible chez JForex sur ce périmètre. La cause historique reste indéterminée.</p>
<h3>Accès et couverture encore ouverts</h3>
<p>Les retours JForex vides font partie du journal, distincts des fermetures confirmées. Le 23 novembre 2026 a livré {number(late_ticks)} ticks dans {late_positive} heures. Les périodes 2004–2026 hors des lots récupérés restent à couvrir. Darwinex annonce XAUUSD à partir d’octobre 2017 via un compte et un accès FTP. FXCM demande une prise de contact pour ses données tick. <a href="https://www.darwinex.com/it/algorithmic-trading/tick-data">Darwinex</a> · <a href="https://github.com/fxcm/MarketData#tickdata">FXCM</a></p>
<p>S3 Requester Pays est préparé hors ligne. L’utilisateur a refusé l’autorisation pour l’instant. Aucun accès au bucket ni aucune dépense. Les acquisitions et preuves restent dans les répertoires <code>recovered</code>, <code>manifests</code>, <code>reports</code> et <code>logs</code> du dossier de données. Aucun fichier source n’a été fusionné ni remplacé.</p>
</section>'''
    original = source.read_text(encoding='utf-8')
    assert '<section id="verifications-jforex-2026-11-24">' in original
    assert 'id="reprise-multicanal"' not in original
    original = original.replace('<h1>Audit Bronze XAUUSD · 2004–2026</h1>',
                                '<h1>Audit Bronze XAUUSD · 2004–2026</h1><p><a href="#reprise-multicanal">Voir les nouveaux lots récupérés</a>.</p>')
    output.write_text(original.replace('</main></body></html>', section + '</main></body></html>'), encoding='utf-8')
    print(f'Base={previous} additional={august + late_ticks} available={previous + august + late_ticks} late={late_ticks}')


if __name__ == '__main__':
    main(*map(Path, sys.argv[1:4]))
