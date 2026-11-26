# Pipeline Big Data XAUUSD

## Ordre de lecture

| Étape | Entrée | Résultat |
| --- | --- | --- |
| 1. Collecte Bronze | CSV historiques et exports fournisseur | Sources et preuves conservées dans `data/`. Voir [les outils Bronze](bronze/). |
| 2. Profilage DuckDB | CSV historiques | Couverture, chevauchement, décimales, anomalies et métriques. Ouvrir [le notebook de profilage](notebooks/Profilage_donnees_XAUUSD.ipynb). |
| 3. Nettoyage Silver | Lots admis par le catalogue | Ticks normalisés en UTC, dédupliqués et validés. Les lignes non résolues rejoignent la quarantaine. |
| 4. Enrichissement Gold | Ticks Silver | Bougies multi-échelles et variables calculées avec l’information disponible à chaque instant. |
| 5. Préparation des datasets | Variables Gold | Parquet partitionné en `train`, `validation` et `test` dans l’ordre chronologique. Le scaler est ajusté sur le train. |

Les étapes 3 à 5 sont dans le [notebook PySpark](notebooks/Pipeline_BigData_XAUUSD.ipynb). Ses cellules se suivent de la configuration au rapport final.

## Outils Bronze

- [acquisition/](bronze/acquisition/) contient les collecteurs et le pilote S3.
- [audit/](bronze/audit/) contient les contrôles DuckDB, le rapprochement, la fusion brute historique et les générateurs de rapports.
- [jforex/](bronze/jforex/) et [histdata/](bronze/histdata/) regroupent les outils propres à chaque fournisseur.
- [preuves/](bronze/preuves/) conserve des résultats, empreintes et rapports de référence versionnés.

Le mode PySpark `SNAPSHOT` est le mode par défaut. Il consulte les résultats présents sans calcul Spark. `TEST` et `PILOT` écrivent respectivement sous `data/pipeline_runs/test/` et `data/pipeline_runs/pilot_<debut>_<fin>/`. `FULL` écrit sous `data/pipeline_runs/full/`. Les modes de calcul requièrent le noyau Linux ou WSL2 avec PySpark et Java.

`XAUUSD_PROJECT_DIR` permet d’indiquer la racine du projet. `XAUUSD_DATA_DIR` permet de pointer les notebooks vers un autre emplacement de données.

## Préparation Binance XAUUSDT

[binance/calculer_indicateurs.py](binance/calculer_indicateurs.py) prépare les bougies et les indicateurs du CSV destiné aux modèles actuels. Il lit et écrit sous `data/binance/` par défaut. Les [commandes d’entraînement](../training/) décrivent le fichier attendu.
