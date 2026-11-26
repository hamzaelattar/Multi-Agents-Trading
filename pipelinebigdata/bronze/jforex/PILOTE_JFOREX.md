# Pilote XAUUSD du 5 janvier 2021 UTC

*Mise à jour du 24 novembre 2026.* La session DEMO a produit 146 982 ticks sur 17 heures. Les deux exports locaux en contiennent chacun 183 840 avec le décalage UTC+2. Les 36 858 ticks d’écart sont concentrés sur six heures. La reprise ciblée est restée vide. Le lot est audité et la validation du pilote reste en attente. [Résultats chiffrés du pilote](Rapport_Pilote_JForex_2021-01-05.txt).

Le pilote vise l’intervalle `2021-01-05T00:00:00.000Z` inclus à `2021-01-06T00:00:00.000Z` exclu. Il repose sur une session **DEMO** authentifiée. Le descripteur officiel de la session DEMO a répondu HTTP 200. Aucun CSV fournisseur n’a été obtenu sans cette session et l’historique local n’a pas été modifié.

## JForex 4 Desktop

Dans l’application officielle, ouvrir une session **DEMO**. Dans **Historical Data Manager**, sélectionner **XAU/USD** et demander les **ticks** de la seule journée pilote, avec millisecondes, Ask, Bid et les deux volumes. Utiliser UTC/GMT si l’interface permet de le choisir et noter le fuseau et l’unité affichés dans tous les cas. Conserver l’export brut dans `data/source_runs/pilote_jforex`. Les identifiants de démonstration se saisissent dans l’application ou le terminal local. La création d’un compte, si nécessaire, se fait sur la page officielle. [JForex 4](https://www.dukascopy.com/land/trading/bank/platforms/jforex4/) · [Manuel de l’espace de travail](https://wiki.dukascopy.com/en/manuals/jforex4-desktop/workspace/)

## JForex Java SDK

`PiloteXauusdJForex.java` utilise uniquement `IContext.getHistory()` et `IHistory.getTicks()` pour la lecture. Les appels horaires couvrent `start` à `end - 1 ms` parce que `getTicks` inclut ses deux bornes. Les ticks de même timestamp restent distincts. Les valeurs `ITick` sont exportées telles que retournées, sans facteur de volume supposé. Le code a compilé avec l’API officielle 2.13.99. Son exécution sur le serveur attend l’authentification démo. [Historique des ticks](https://wiki.dukascopy.com/en/development/strategy-api/historical-data/history-ticks/) · [Vue d’ensemble](https://wiki.dukascopy.com/en/development/strategy-api/historical-data/overview-historical-data/) · [SDK officiel](https://wiki.dukascopy.com/en/development/get-started-api/use-jforex-sdk/download-jforex-sdk) · [Client démo](https://www.dukascopy.com/wiki/en/development/get-started-api/use-jforex-sdk/sdk-client/)

Le fichier peut être chargé dans l’éditeur d’une session démo. Pour le SDK autonome, ajouter la classe au projet officiel et exécuter sa méthode `main` dans un terminal avec comme unique argument le dossier `data/source_runs/pilote_jforex`. Le programme demande le mot de passe au terminal et limite la requête à la journée pilote. L’export normalisé porte le nom `XAUUSD_2021-01-05_JForex_UTC.csv`.

## Comparaison avec les originaux

Depuis la racine du dépôt :

```powershell
$projet = Join-Path ([Environment]::GetFolderPath('DesktopDirectory')) 'Multi-Agents-Trading'
$data = Join-Path $projet 'data'
$pilote = Join-Path $projet 'data/source_runs/pilote_jforex'
python pipelinebigdata/bronze/jforex/comparer_pilote_jforex.py `
  --local (Join-Path $data 'XAUUSD_Ticks_2004.01.01_2021.10.15.csv') `
  --local (Join-Path $data 'XAUUSD_Ticks_2021.01.04_2026.11.20.csv') `
  --reference (Join-Path $pilote 'XAUUSD_2021-01-05_JForex_UTC.csv') `
  --output (Join-Path $pilote 'comparaison_2021-01-05.json')
```

Le comparateur attend le format UTC du script Java. Conserver un export du gestionnaire au format natif et vérifier son fuseau, son séparateur et les volumes avant de l’adapter. Le rapport distingue les effectifs par milliseconde, les prix exactement égaux et les volumes exactement égaux sous plusieurs facteurs candidats. Les paires sont comparées avec leur multiplicité. Les lignes au découpage ambigu contribuent au comptage des ticks et des timestamps. Un seul jour d’hiver permet de tester l’offset de cette journée. Le changement d’heure et l’équivalence des volumes exigent des comparaisons supplémentaires.

Les deux extraits locaux ont, chacun, **183 682 ticks** pour l’hypothèse `UTC+0`, **183 840** pour `UTC+2` et **184 349** pour `UTC+3`. Ce sont des fenêtres candidates. Le JSON local joint conserve les autres hypothèses. La référence officielle est nécessaire pour départager ces valeurs.

## Estimation S3 Requester Pays

Le canal officiel se trouve en `eu-west-1`. AWS facture au demandeur les requêtes et le transfert. Le fichier quotidien XAUUSD et sa taille n’ont pas été vérifiés par une requête S3. Dukascopy indique **0,0004 USD par 1 000 GET** et cite **0,02 USD/Go** pour son estimation. Pour les scénarios vers ce poste, le calcul ci-dessous retient le tarif public AWS de **0,09 USD/Go** pour une sortie internet depuis l’Irlande, avant application d’une éventuelle franchise ou de crédits. [Export S3 Dukascopy](https://wiki.dukascopy.com/en/development/data-export/) · [Tarifs S3 AWS](https://aws.amazon.com/s3/pricing/) · [Règles Requester Pays](https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html)

| Scénario illustratif | Hypothèse | Estimation hors taxes et frais additionnels |
|---|---|---:|
| Pilote d’un jour | 1 GET, 1 HEAD et 100 Mo transférés | ≈ 0,009 USD |
| Fenêtre 2004–2026 | Jusqu’à 8 363 GET si un objet par jour civil, 10 Go transférés | ≈ 0,90 USD |
| Fenêtre 2004–2026 | Même nombre de GET, 50 Go transférés | ≈ 4,50 USD |

Ces scénarios servent à préparer un budget. Le coût réel dépend notamment des tailles, de la classe de stockage, des requêtes de découverte, des relances et des conditions du compte. Toute requête payante attend l’autorisation. Le téléchargement massif attend la validation du pilote.

## Statuts Bronze

- `downloaded` indique un CSV enregistré avec un compte attendu et une empreinte SHA-256. Une nouvelle tentative d’audit relit ce fichier local.
- `audited` indique un rapport DuckDB enregistré avec la même empreinte. `internal_checks` distingue un audit sans anomalie bloquante d’un audit ayant trouvé des problèmes. Les doublons exacts, les spreads nuls et les intervalles longs sont signalés pour revue.
- `validated` exige un rapprochement documenté avec la source, des conventions d’heure et d’unité établies pour la fenêtre, une explication des divergences importantes et une validation explicite. Le collecteur HTTP n’attribue plus ce statut sur la base de ses propres contrôles. Le comparateur pilote s’arrête à `compared_pending_validation` lorsqu'une référence est fournie.
