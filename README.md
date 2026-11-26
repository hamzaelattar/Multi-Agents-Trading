# Multi-Agents-Trading

## Description du projet

Multi-Agents-Trading est un projet de trading automatisé conçu pour intervenir sur les marchés financiers en conditions réelles. Il repose sur des agents spécialisés qui analysent deux sources complémentaires. L’analyse des actualités économiques et financières sert à repérer les événements susceptibles d’influencer les actifs suivis. L’analyse technique exploite les prix, les volumes et les indicateurs pour évaluer la tendance, le momentum et la volatilité.

Le système combine ces analyses pour décider quand ouvrir, ajuster ou fermer une position. En conditions réelles, un module d’exécution transmet les ordres au courtier.


## Organisation du projet

Le dossier `Multi-Agents-Trading` contient le dépôt Git `github/` et le dossier local `data/`. Ouvrir le dossier parent dans VS Code permet de voir les deux dans le même explorateur.

| Emplacement | Rôle |
| --- | --- |
| [Profilage_donnees_XAUUSD.ipynb](Profilage_donnees_XAUUSD.ipynb) | Audit des exports XAUUSD, de leur couverture, de leurs anomalies et de leur chevauchement. |
| [pipeline bigdata/](pipeline%20bigdata/) | Notebook PySpark qui enchaîne le catalogue Bronze, le nettoyage et la réconciliation Silver, puis les bougies et les variables Gold. |
| [bronze/](bronze/) | Outils de collecte et de validation, pilotes par fournisseur et preuves d’audit. |
| [fusionner_ticks.py](fusionner_ticks.py) | Fusion brute des exports historiques. Le comparateur du pilote JForex réutilise ses fonctions. |
| [training/](training/) | Entraînement des modèles DRL-DEM, FinRL Ensemble et Stockformer. |

### Données locales

Les notebooks et les outils partagent les entrées de `data/` et y écrivent leurs résultats.

| Emplacement dans `data/` | Contenu |
| --- | --- |
| Racine | Deux exports CSV historiques XAUUSD. |
| `archives/` | Archives ZIP et RAR des exports historiques. |
| `recovered/` | Extractions complémentaires classées par fournisseur. |
| `source_runs/` | Fichiers et artefacts des collectes, pilotes et audits par source. |
| `manifests/` | Inventaires, provenance et résultats d’audit par lot. |
| `quarantine/` | Extractions ou lignes isolées pendant leur validation. |
| `reports/` et `logs/` | Bilans d’audit et journaux d’exécution. |
| `pipeline_runs/` | Sorties PySpark classées par exécution, dont les pilotes. |

### Entraînement sur les bougies Binance

Les trois scripts `train_*.py` attendent un CSV Binance XAUUSDT en bougies d’une minute avec les 35 colonnes prévues. L’historique XAUUSD du notebook PySpark suit son propre schéma. Dans `training/`, `calculer_indicateurs.py` prépare les indicateurs Binance et `training.py` contient des composants d’apprentissage PPO.

Depuis `github/`, utiliser l’environnement Python du projet. L’option `--out` place les résultats dans le dossier de données local.

```powershell
python training/train_drl_dem.py --csv "C:/chemin/vers/XAUUSDT_1min_binance_indicateurs.csv" --out "../data/training_runs/drl_dem"
```

Les deux autres points d’entrée sont `training/train_finrl_ensemble.py` et `training/train_stockformer.py`. Chaque script expose ses paramètres avec `--help`.

## 1. Données de marché obligatoires

Ces données brutes servent à calculer les indicateurs techniques.

| Variable | Description |
| --- | --- |
| `timestamp` | Date et heure |
| `symbol` | Actif financier |
| `open` | Prix d'ouverture |
| `high` | Prix maximum |
| `low` | Prix minimum |
| `close` | Prix de clôture |
| `volume` | Volume échangé |

## 2. Indicateurs sélectionnés

| Catégorie | Indicateur | Paramètres proposés |
| --- | --- | --- |
| Tendance | SMA | 5, 10, 20, 50 |
| Tendance | EMA | 5, 10, 20, 50 |
| Momentum | RSI | 7, 14 |
| Momentum | MACD | (5, 13, 4) et (5, 26, 4) |
| Momentum | Stochastic | (9, 3) et (14, 3) |
| Momentum | CCI | 14 |
| Volatilité | ATR | 7, 14 |
| Volatilité | Bollinger Bands | 20, 2 |
| Volume | OBV | Variation sur 10 et 20 bougies |
