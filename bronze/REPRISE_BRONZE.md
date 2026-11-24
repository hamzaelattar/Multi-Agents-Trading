# Reprise Bronze XAUUSD · 24 novembre 2026

Le [rapport Bronze](Rapport_Bronze_XAUUSD.html) distingue le profil initial, les vérifications JForex et la comparaison HistData. Le dépôt contient le code et le rapport agrégé. La collecte massive reste suspendue tant que les heures JForex manquantes et les limites des autres sources ne sont pas résolues.

## Dossier local `data`

- Les deux CSV historiques sont à la racine de `data`. Les archives ZIP et RAR d’origine sont dans `data/archives`. Le contenu de chaque archive correspond à l’un des CSV par empreinte SHA-256.
- Les lots JForex et les fichiers HistData sont dans `data/recovered`. Les archives HistData de janvier 2021, août 2025 et novembre 2026 sont extraites dans les sous-dossiers mensuels `brut`. Les horodatages y suivent l’horloge brute du fournisseur. Novembre 2026 s’arrête le 13 à 16 h 59 min 59 s selon cette horloge.
- Les manifestes `archives_historiques.json` et `extractions_histdata.json` sont dans `data/manifests`. Les audits, les rapports et la quarantaine restent sous `data`.

## Lots officiels JForex

`jforex/VerifierJForexBronze.java` utilise le SDK officiel 3.6.51, l’API 2.13.99 et un compte démo. Il interroge une heure à la fois et consigne chaque réponse. Le mode sans date ne couvre que les cinq sondes de cette exécution. Le second argument limite la requête à un jour UTC terminé. Choisir un nouveau dossier local pour chaque tentative. Le SDK gère son cache de session. Les fichiers partiels et les heures vides restent des preuves à examiner.

`jforex/auditer_verifications_jforex.cjs` audite les CSV finalisés avec DuckDB et vérifie leurs effectifs par rapport au journal. Depuis le dépôt, avec les dépendances installées dans `bronze`, l’ordre des arguments est le chemin de `bronze/collecter_dukascopy.cjs`, le dossier de la session JForex puis le chemin de `duckdb.exe`. `jforex/etudier_ambiguities_jforex.py` prend le dossier de projet et un chemin JSON de sortie. Il rapproche le premier pilote avec les deux originaux. `archiver_verifications_bronze.py` prend le dossier de projet, vérifie les SHA-256 et copie les lots audités dans `recovered/dukascopy`, avec journaux et manifestes séparés.

## Canal public HistData

- `histdata/collecter_histdata_pilote.py --month AAAA-MM --out <dossier>` soumet le formulaire public d’un seul mois. Le ZIP, son statut et son empreinte sont conservés. Le service FTP payant reste exclu.
- `histdata/auditer_histdata_pilote.py` contrôle le mois d’août 2025 avec DuckDB. Avec `--jforex`, le CSV d’une journée suit d’abord l’horloge documentée et doit être traité comme non validé. Avec `--candidat-mensuel`, le script exige la preuve du décalage sur le 5 août avant de créer le Parquet candidat. L’offset du reste du mois demeure une hypothèse.
- `histdata/verifier_horloge_histdata.py` compare les deux offsets candidats pour un jour UTC désigné. Un fichier `--candidate` est créé uniquement si le meilleur offset apparie la totalité de la référence JForex. Les volumes Ask et Bid restent NULL. L’archive brute demeure conservée.

## S3 Requester Pays

`pilote_s3_requester_pays.py` présente par défaut un plan hors ligne pour un seul objet du 5 janvier 2021. L’utilisateur a décliné l’autorisation pour l’instant. Aucun appel n’a été exécuté. Le scénario prévoit un HEAD, puis au maximum un GET Standard et 100 Mo. Le coût indicatif est de 0,009 USD avant taxes et franchise. [Format Dukascopy S3](https://wiki.dukascopy.com/en/development/data-export/) · [Tarification AWS](https://aws.amazon.com/s3/pricing/)

## Régénérer le rapport

Générer d’abord la version initiale dans un fichier temporaire avec `rapporter_bronze.py`. `rapporter_verifications_bronze.py` prend le dossier de projet, le HTML initial et une sortie intermédiaire. Après les audits et le classement local, `rapport_reprise_bronze.py` prend le dossier de projet, le HTML intermédiaire et la sortie finale. Les fichiers indiqués dans le rapport et les scripts sont conservés dans le dossier de données et les dossiers de session locaux.
