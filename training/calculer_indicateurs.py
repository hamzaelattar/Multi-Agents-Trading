"""Génère un CSV de bougies et d'indicateurs techniques avec pandas et TA-Lib."""

import argparse
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import talib


DATA_DIR = Path(__file__).resolve().parents[2] / "data_binance"
DEFAULT_INPUT = DATA_DIR / "XAUUSDT_1min_binance.csv"
PRICE_VOLUME_COLUMNS = ["open", "high", "low", "close", "volume"]
MARKET_COLUMNS = ["timestamp", "symbol", *PRICE_VOLUME_COLUMNS]
MA_PERIODS = (5, 10, 20, 50)


def read_candles(source: Path, fallback_symbol: str) -> pd.DataFrame:
    candles = pd.read_csv(
        source,
        usecols=lambda name: name in MARKET_COLUMNS,
        dtype={"timestamp": "string", "symbol": "string"},
        float_precision="round_trip",
    )
    if "symbol" not in candles.columns:
        candles["symbol"] = fallback_symbol

    missing = set(MARKET_COLUMNS) - set(candles.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV : {', '.join(sorted(missing))}")
    if candles.empty:
        raise ValueError("Le CSV source ne contient aucune bougie.")

    candles["symbol"] = candles["symbol"].str.strip()
    if candles["symbol"].isna().any() or candles["symbol"].eq("").any():
        raise ValueError("Chaque bougie doit avoir un symbole.")

    candles[PRICE_VOLUME_COLUMNS] = candles[PRICE_VOLUME_COLUMNS].astype("float64")
    if not np.isfinite(candles[PRICE_VOLUME_COLUMNS].to_numpy()).all():
        raise ValueError("Le CSV contient des prix ou des volumes manquants ou non finis.")

    candles["_time"] = pd.to_datetime(candles["timestamp"], format="ISO8601", utc=True)
    if candles["_time"].isna().any():
        raise ValueError("Le CSV contient des horodatages manquants.")
    if candles.duplicated(["symbol", "_time"]).any():
        raise ValueError("Le CSV contient plusieurs bougies pour un même symbole et horodatage.")

    candles = candles.sort_values(["symbol", "_time"], ignore_index=True)
    return candles[MARKET_COLUMNS]


def calculate_indicators(candles: pd.DataFrame) -> pd.DataFrame:
    """Calcule les indicateurs sur les bougies triées d'un seul symbole."""
    high = candles["high"].to_numpy()
    low = candles["low"].to_numpy()
    close = candles["close"].to_numpy()
    volume = candles["volume"].to_numpy()

    indicators = {}
    for name, average in (("sma", talib.SMA), ("ema", talib.EMA)):
        for period in MA_PERIODS:
            indicators[f"{name}_{period}"] = average(close, timeperiod=period)

    for period in (7, 14):
        indicators[f"rsi_{period}"] = talib.RSI(close, timeperiod=period)

    for slow_period in (13, 26):
        macd, signal, histogram = talib.MACD(
            close, fastperiod=5, slowperiod=slow_period, signalperiod=4
        )
        suffix = f"5_{slow_period}_4"
        indicators[f"macd_{suffix}"] = macd
        indicators[f"macd_signal_{suffix}"] = signal
        indicators[f"macd_hist_{suffix}"] = histogram

    for period in (9, 14):
        percent_k, percent_d = talib.STOCHF(
            high, low, close, fastk_period=period, fastd_period=3, fastd_matype=0
        )
        indicators[f"stoch_k_{period}_3"] = percent_k
        indicators[f"stoch_d_{period}_3"] = percent_d

    indicators["cci_14"] = talib.CCI(high, low, close, timeperiod=14)
    for period in (7, 14):
        indicators[f"atr_{period}"] = talib.ATR(high, low, close, timeperiod=period)

    upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    indicators["bb_upper_20_2"] = upper
    indicators["bb_middle_20_2"] = middle
    indicators["bb_lower_20_2"] = lower

    obv = pd.Series(talib.OBV(close, volume), index=candles.index)
    for period in (10, 20):
        indicators[f"obv_diff_{period}"] = obv.diff(period)

    return candles.join(pd.DataFrame(indicators, index=candles.index))


def write_csv(candles: pd.DataFrame, destination: Path) -> None:
    # Le fichier final est remplacé seulement après une écriture complète.
    with NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=destination.parent,
        suffix=".tmp", delete=False,
    ) as temp_file:
        temp_path = Path(temp_file.name)
        try:
            candles.to_csv(temp_file, index=False, lineterminator="\n")
            temp_file.close()
            temp_path.replace(destination)
        finally:
            temp_file.close()
            temp_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="CSV source")
    parser.add_argument("--output", type=Path, help="Chemin du CSV généré")
    parser.add_argument("--symbol", help="Symbole à utiliser si la colonne symbol est absente")
    args = parser.parse_args()

    source = args.input.resolve()
    destination = args.output or source.with_name(f"{source.stem}_indicateurs.csv")
    destination = destination.resolve()
    if destination == source:
        parser.error("Le CSV généré doit être différent du CSV source.")

    fallback_symbol = args.symbol or source.stem.split("_", maxsplit=1)[0]
    try:
        candles = read_candles(source, fallback_symbol)
        result = pd.concat(
            [calculate_indicators(group) for _, group in candles.groupby("symbol", sort=False)],
            ignore_index=True,
        )
        write_csv(result, destination)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Erreur : {error}\n")

    print(f"{len(result):,} bougies et {len(result.columns)} colonnes exportées dans {destination}")


if __name__ == "__main__":
    main()
