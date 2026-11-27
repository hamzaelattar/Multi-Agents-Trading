# Pipeline Big Data XAUUSD

## Ordre d’exécution

| Notebook | Entrée | Sortie |
| --- | --- | --- |
| [01 Bronze](notebooks/01_Bronze.ipynb) | Originaux et lots récupérés | Catalogue, profil Spark par mois et par source, empreintes et métriques. |
| [02 Silver](notebooks/02_Silver.ipynb) | Catalogue et audit Bronze compatibles | Réparations prouvées, ticks en UTC, quarantaine et neuf séries OHLCV. |
| [03 Gold](notebooks/03_Gold.ipynb) | Ticks validés et bougies Silver | Variables causales, catalogue des features et profil numérique. |
| [04 Dataset Builder](notebooks/04_Dataset_Builder.ipynb) | Features et catalogue Gold | Jeux Parquet Train, Validation et Test, fenêtres, paramètres du scaler et bilan. |

Chaque notebook initialise sa propre session Spark. Exécuter les quatre avec le même périmètre et le même `XAUUSD_RUN_LABEL`. Les manifestes et marqueurs vérifient les dépendances. Pour lire les sorties d’un FULL sans recalcul, utiliser `XAUUSD_MODE=SNAPSHOT` et `XAUUSD_RESUME_MODE=FULL`. Sous Linux ou Ubuntu WSL2, le noyau utilise Python, Java 17, PySpark 4.0.1, PyArrow, Psutil et Matplotlib.

`XAUUSD_DATASET_OPTIONS` accepte un objet JSON. Exemples de réglages : `features` peut être `"all"` ou une liste de colonnes, `families` filtre selon le catalogue Gold, `lookback` fixe la longueur de fenêtre, et `scaler` accepte `standard`, `minmax`, `robust` et `none`. Les options personnalisées ont une sortie identifiée par leur empreinte. Les seuils de disponibilité et les valeurs manquantes peuvent exclure une observation. Consulter les effectifs et la couverture avant d’utiliser un dataset.

## Outils Bronze

- [acquisition/](bronze/acquisition/) contient les collecteurs et le pilote S3.
- [audit/](bronze/audit/) contient les contrôles DuckDB, le rapprochement, la fusion brute historique et les générateurs de rapports.
- [jforex/](bronze/jforex/) et [histdata/](bronze/histdata/) regroupent les outils propres à chaque fournisseur.
- [preuves/](bronze/preuves/) conserve des résultats, empreintes et rapports de référence versionnés.

Le mode PySpark `SNAPSHOT` est le mode par défaut. Il consulte les résultats présents sans calcul Spark. `TEST` et `PILOT` écrivent respectivement sous `data/pipeline_runs/test/` et `data/pipeline_runs/pilot_<debut>_<fin>/`. `FULL` écrit sous `data/pipeline_runs/full/`. Les profils et résultats de pilotage sont séparés des sorties FULL. Les métriques non exposées par Spark UI sont marquées indisponibles.

`XAUUSD_PROJECT_DIR` permet d’indiquer la racine du projet. `XAUUSD_DATA_DIR` permet de pointer les notebooks vers un autre emplacement de données.

## Préparation Binance XAUUSDT

[binance/calculer_indicateurs.py](binance/calculer_indicateurs.py) prépare les bougies et les indicateurs du CSV destiné aux modèles actuels. Il lit et écrit sous `data/binance/` par défaut. Les [commandes d’entraînement](../training/) décrivent le fichier attendu.
