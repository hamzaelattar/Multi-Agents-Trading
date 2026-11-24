# Collecte Bronze des ticks XAUUSD

Le rapport du 24 novembre 2026 présente les résultats locaux minute par minute. L’essai Dukascopy du 5 janvier 2021 a reçu HTTP 403 sur `dukascopy-node` et HTTP 429 sur le flux historique. Les colonnes qui exigent une comparaison fournisseur restent non déterminées dans le rapport.

## Journée pilote et reprise

Depuis la racine du dépôt, avec Node.js et le binaire DuckDB installé :

```powershell
npm.cmd ci --prefix bronze
$projet = Join-Path ([Environment]::GetFolderPath('DesktopDirectory')) 'Multi-Agents-Trading'
$donnees = Join-Path $projet 'data_trading_agent'
$bronze = Join-Path $projet 'bronze_dukascopy'
$duckdb = Join-Path $env:LOCALAPPDATA 'duckdb/cli/2.0.1/duckdb.exe'
node bronze/collecter_dukascopy.cjs --from 2021-01-05 --through 2021-01-05 --out $bronze --duckdb $duckdb
```

`--through` inclut le jour indiqué. Après validation du pilote, employer le même dossier et traiter les mois par plages successives. Le collecteur écrit un CSV UTC par journée, organisé par année et par mois, avec un manifeste JSON de provenance, de contrôles DuckDB et d’empreinte SHA-256. Au redémarrage, il vérifie l’empreinte des journées déjà terminées et réutilise le cache des autres journées. Une réponse vide est consignée comme non vérifiée. Un refus d’accès arrête la collecte.

Les volumes fournisseur sont demandés en `millions`, selon les unités de dukascopy-node. L’équivalence avec les CSV historiques locaux reste à vérifier. Les fichiers Bronze et le cache se trouvent à côté du dépôt Git. [Paramètres dukascopy-node](https://www.dukascopy-node.app/config/node) · [Export S3 Requester Pays de Dukascopy](https://wiki.dukascopy.com/en/development/data-export/)

## Reproduire l’audit local

Les deux premières commandes produisent des agrégats Parquet sous `audit_local`. Les preuves de cette exécution sont disponibles dans `bronze/preuves`. Le rapport HTML nécessite Python 3 et Matplotlib.

```powershell
$audit = Join-Path $bronze 'audit_local'
python bronze/profiler_minutes_bronze.py --data $donnees --out (Join-Path $audit 'minutes_ancien.parquet') --duckdb $duckdb --export ancien
python bronze/profiler_minutes_bronze.py --data $donnees --out (Join-Path $audit 'minutes_recent.parquet') --duckdb $duckdb --export recent
Copy-Item bronze/preuves/origines_sha256.json,bronze/preuves/acces_fournisseur.json $audit
python bronze/rapporter_bronze.py --audit $audit --duckdb $duckdb --output (Join-Path $bronze 'Rapport_Bronze_XAUUSD.html')
```

Le décompte local utilise l’heure étiquetée EET, sans présumer de conversion UTC. L’ancien export est tronqué dans le calcul des effectifs à partir du premier tick commun déjà vérifié. Pour distinguer une fermeture normale d’une perte, le prochain audit doit comparer le pilote, vérifier la convention de fuseau et les unités, puis prendre en compte le calendrier des sessions par période.
