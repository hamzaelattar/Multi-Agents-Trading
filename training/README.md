# Entraînement des modèles

| Fichier | Rôle |
| --- | --- |
| [train_drl_dem.py](train_drl_dem.py) | Entraîne le modèle DRL-DEM. |
| [train_finrl_ensemble.py](train_finrl_ensemble.py) | Entraîne l’ensemble FinRL. |
| [train_stockformer.py](train_stockformer.py) | Entraîne Stockformer. |
| [training.py](training.py) | Regroupe les composants PPO. |

Les trois scripts `train_*.py` lisent le format Binance **XAUUSDT M1**, avec ses 35 colonnes. Les datasets Parquet **XAUUSD** issus de PySpark suivent un autre schéma. L’intégration XAUUSD devra valider les variables d’entrée, le contrat des modèles et les découpages chronologiques.

La préparation des bougies Binance s’effectue avec [le script de calcul des indicateurs](../pipelinebigdata/binance/calculer_indicateurs.py). Placer le CSV Binance sous `data/binance/` ou transmettre son chemin explicitement. Exécuter les commandes depuis la racine.

```powershell
python pipelinebigdata/binance/calculer_indicateurs.py --input data/binance/XAUUSDT_1min_binance.csv
python training/train_drl_dem.py --csv data/binance/XAUUSDT_1min_binance_indicateurs.csv
```

Les métriques et les poids sont écrits par défaut dans `data/training_runs/<modele>/`. L’option `--out` accepte un autre dossier.
