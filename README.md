# Multi-Agent-Tradign

## Description du projet

Multi-Agents-Trading est un projet de trading automatisé conçu pour intervenir sur les marchés financiers en conditions réelles. Il repose sur des agents spécialisés qui analysent deux sources complémentaires. L’analyse des actualités économiques et financières sert à repérer les événements susceptibles d’influencer les actifs suivis. L’analyse technique exploite les prix, les volumes et les indicateurs pour évaluer la tendance, le momentum et la volatilité.

Le système combine ces analyses pour décider quand ouvrir, ajuster ou fermer une position. En conditions réelles, un module d’exécution transmet les ordres au courtier.


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
