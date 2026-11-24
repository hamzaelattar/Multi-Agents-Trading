# Reprise Bronze XAUUSD · 24 novembre 2026

Le [rapport Bronze](Rapport_Bronze_XAUUSD.html) distingue le profil initial, les vérifications JForex et la comparaison HistData. Le dépôt contient le code et le rapport agrégé. Le collecteur HTTP Dukascopy reste suspendu. Les sources complémentaires sont conservées séparément.

## Dossier local `data`

- Les deux CSV historiques sont à la racine de `data`. Les archives ZIP et RAR d’origine sont dans `data/archives`. Le contenu de chaque archive correspond à l’un des CSV par empreinte SHA-256.
- Les lots JForex et les fichiers HistData sont dans `data/recovered`. Les archives HistData de janvier 2021, août 2025 et novembre 2026 sont extraites dans les sous-dossiers mensuels `brut`. Les horodatages y suivent l’horloge brute du fournisseur. Novembre 2026 s’arrête le 13 à 16 h 59 min 59 s selon cette horloge.
- Les manifestes `archives_historiques.json` et `extractions_histdata.json` sont dans `data/manifests`. Les audits, les rapports et la quarantaine restent sous `data`.

## Lots officiels JForex

`jforex/VerifierJForexBronze.java` utilise le SDK officiel 3.6.51, l’API 2.13.99 et un compte démo. Il interroge une heure à la fois et consigne chaque réponse. Le mode sans date ne couvre que les cinq sondes de cette exécution. Le second argument limite la requête à un jour UTC terminé. Avec ce jour explicite, le troisième argument peut limiter les requêtes à des heures UTC séparées par des virgules. Choisir un nouveau dossier local pour chaque tentative. Le SDK gère son cache de session. Les fichiers partiels et les heures vides restent des preuves à examiner.

`jforex/auditer_verifications_jforex.cjs` audite les CSV finalisés avec DuckDB et vérifie leurs effectifs par rapport au journal. Depuis le dépôt, avec les dépendances installées dans `bronze`, l’ordre des arguments est le chemin de `bronze/collecter_dukascopy.cjs`, le dossier de la session JForex puis le chemin de `duckdb.exe`. `jforex/etudier_ambiguities_jforex.py` prend le dossier de projet et un chemin JSON de sortie. Il rapproche le premier pilote avec les deux originaux. `archiver_verifications_bronze.py` prend le dossier de projet, vérifie les SHA-256 et copie les lots audités dans `recovered/dukascopy`, avec journaux et manifestes séparés.

## Canal public HistData

- `histdata/collecter_histdata_pilote.py --month AAAA-MM --out <dossier>` soumet le formulaire public d’un seul mois. Le ZIP, son statut et son empreinte sont conservés. Le service FTP payant reste exclu.
- `histdata/auditer_histdata_pilote.py` contrôle le mois d’août 2025 avec DuckDB. Avec `--jforex`, le CSV d’une journée suit d’abord l’horloge documentée et doit être traité comme non validé. Avec `--candidat-mensuel`, le script exige la preuve du décalage sur le 5 août avant de créer le Parquet candidat. L’offset du reste du mois demeure une hypothèse.
- `histdata/verifier_horloge_histdata.py` compare les deux offsets candidats pour un jour UTC désigné. Un fichier `--candidate` est créé uniquement si le meilleur offset apparie la totalité de la référence JForex. Les volumes Ask et Bid restent NULL. L’archive brute demeure conservée.

## Récupération mensuelle et bilan annuel

Le catalogue [HistData XAU/USD](https://www.histdata.com/download-free-forex-data/?/ascii/tick-data-quotes/xauusd) propose des archives par mois depuis mars 2009. Le collecteur de reprise utilise le formulaire public mois par mois et réutilise les archives dont les empreintes correspondent. Il interrompt le lot en cas de refus d’accès ou d’échec réseau et consigne le mois dans le journal. Les deux fichiers extraits sont conservés dans le sous-dossier brut du mois. DuckDB contrôle les prix, les horodatages, les volumes bruts et la couverture de chaque archive.

Les horodatages HistData sont conservés dans l’horloge du fichier. Les volumes de cotation Bid et Ask sont indisponibles sur ce canal. L’exactitude de la conversion UTC doit être vérifiée avant tout rapprochement hors des jours déjà testés. [Spécification HistData](https://www.histdata.com/f-a-q/data-files-detailed-specification/)

Des éditions publiques indépendantes compilées par Touhid-HZ annoncent des ticks Dukascopy en UTC avec les deux volumes. Le fichier hebdomadaire pilote couvrant le 5 août 2025 concorde sur ses 156 842 lignes avec le lot officiel JForex pour les cinq champs. Ce contrôle porte sur une journée. La provenance, le ZIP original, les fichiers hebdomadaires et les audits DuckDB sont conservés par année. [2024](https://github.com/Touhid-HZ/xauusd-tick-data-2024) · [2025](https://github.com/Touhid-HZ/xauusd-tick-data-2025) · [2026](https://github.com/Touhid-HZ/xauusd-tick-data-2026)

Le jeu [CarlosSilva1/xauusd-ticks](https://huggingface.co/datasets/CarlosSilva1/xauusd-ticks) (CC BY 4.0 selon son éditeur) fournit les ticks en Parquet. Les partitions récupérées de mai à décembre 2021, de 2022 et de 2023 sont auditables directement dans `data/recovered/autres_fournisseurs/carlos_silva_xauusd`. Les fichiers d’origine contiennent davantage de ticks sur l’année 2022 et restent conservés. Le lot pilote de sa révision `d14fc64dc3b0452732e60662d94b500a6bb3d534` concorde également avec les 156 842 ticks JForex du 5 août 2025. Le dépôt source décrit un flux issu de données de courtage.

Le pilote officiel du 5 janvier 2004 a retrouvé 18 277 horodatages déjà présents dans le CSV local sous UTC+2. Le CSV contient 20 535 ticks sur la même journée UTC et le SDK a laissé cinq heures vides à vérifier. La preuve est dans `data/source_runs/jforex_pilote_2004-01-05`. Des sondes binaires conservées dans `data/quarantine/esmaeil_ticks_pilote` retrouvent exactement les horodatages des jours testés de 2004 à février 2009 et les prix locaux lisibles. Le 20 mars 2007 comporte 50 156 ticks, contre 11 371 le 10 avril, dans le CSV et dans la sonde binaire. Le catalogue HistData commence en mars 2009 pour cet instrument.

Depuis le dépôt, avec les variables PowerShell projet et DuckDB de la section principale, lancer une plage avec le collecteur mensuel. L’option offline audite les archives disponibles sur la plage sans télécharger. Le bilan annuel CSV tient les comptes par origine et ne les additionne pas.

```powershell
$data = Join-Path $projet 'data'
$histdata = Join-Path $data 'recovered/autres_fournisseurs/histdata'
python bronze/histdata/collecter_histdata_mensuel.py --out $histdata --duckdb $duckdb --from 2023-01 --through 2026-11 --pause 10
python bronze/mesurer_apport_utc.py --data $data --duckdb $duckdb 2021 2022 2023 2024 2025 2026
python bronze/mesurer_apport_utc.py --data $data --duckdb $duckdb --histdata 2009 2013
python bronze/bilan_couverture.py --data $data --html (Join-Path $data 'reports/Rapport_Bronze_XAUUSD.html')
```



La collecte HistData a produit 194 archives mensuelles extraites et auditées, dont les 166 mois de mars 2009 à décembre 2022. Les archives représentent 472 336 326 lignes de prix brutes. L’analyse HistData teste des déplacements de 6 h, 7 h et 8 h entre l’horloge brute et le CSV local. Elle signale les minutes qui n’apparaissent dans aucune hypothèse. Les données restent en attente de rapprochement au tick. Pour 2013, 1 616 minutes et 64 210 lignes de prix ont été signalées par ce contrôle.

### Résultats de la couverture récente

| Année | Ticks des CSV historiques | Ticks audités dans la source complémentaire | Source |
| --- | ---: | ---: | --- |
| 2023 | 7 205 128 | 36 414 968 | Carlos Silva, partitions Parquet |
| 2024 | 1 596 736 | 56 217 145 | Touhid-HZ, 53 CSV compressés |
| 2025 | 314 006 | 71 403 972 | Touhid-HZ, 53 CSV compressés |
| 2026 | 22 690 | 84 371 369 | Touhid-HZ, 47 CSV compressés jusqu’au 22 novembre UTC |

Les quatre archives UTC contiennent 248 407 454 ticks. Parmi eux, 238 893 396 sont dans des minutes absentes des CSV locaux avec les deux décalages possibles UTC+2 ou UTC+3. Le lot JForex du 23 novembre 2026 contient 500 254 ticks sur 15 heures. Les huit heures ciblées dans une seconde session restent sans réponse exploitable. Le bilan CSV indique le dossier de chaque source, les bornes observées et les mois HistData audités. Les contrôles d’intégrité et les comparaisons sur certaines journées donnent deux niveaux de preuve distincts. Les volumes, les prix et les horodatages concordent sur la journée officielle du 5 août 2025 pour les deux éditions testées.


## S3 Requester Pays

`pilote_s3_requester_pays.py` présente par défaut un plan hors ligne pour un seul objet du 5 janvier 2021. L’utilisateur a décliné l’autorisation pour l’instant. Aucun appel n’a été exécuté. Le scénario prévoit un HEAD, puis au maximum un GET Standard et 100 Mo. Le coût indicatif est de 0,009 USD avant taxes et franchise. [Format Dukascopy S3](https://wiki.dukascopy.com/en/development/data-export/) · [Tarification AWS](https://aws.amazon.com/s3/pricing/)

## Régénérer le rapport

Générer d’abord la version initiale dans un fichier temporaire avec `rapporter_bronze.py`. `rapporter_verifications_bronze.py` prend le dossier de projet, le HTML initial et une sortie intermédiaire. Après les audits et le classement local, `rapport_reprise_bronze.py` prend le dossier de projet, le HTML intermédiaire et la sortie finale. Les fichiers indiqués dans le rapport et les scripts sont conservés dans le dossier de données et les dossiers de session locaux.
