# Données et artefacts locaux

| Emplacement | Contenu |
| --- | --- |
| CSV à la racine | Deux exports historiques XAUUSD aux chemins d’origine. |
| `archives/` | Archives ZIP et RAR des exports historiques. |
| `recovered/` | Extractions complémentaires classées par fournisseur. |
| `source_runs/` | Données et artefacts de collecte et d’audit par source. |
| `manifests/` | Provenance, inventaires et audits des lots. |
| `quarantine/` | Lignes ou lots isolés pendant la validation. |
| `reports/` et `logs/` | Rapports et journaux locaux. |
| `pipeline_runs/<execution>/` | Sorties PySpark par exécution. Le contenu peut inclure Silver, Gold, les datasets Parquet et leurs rapports. |
| `binance/` | Emplacement prévu pour les bougies XAUUSDT M1 et le CSV d’indicateurs. |
| `training_runs/<modele>/` | Emplacement prévu pour les métriques et les poids. |

Ce guide est versionné. Les datasets et artefacts de ce dossier sont locaux.
