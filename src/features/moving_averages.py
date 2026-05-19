"""Moving-average features.

Each MA is expressed as a *deviation from current price* rather than
the raw level. This makes features stationary-ish across regimes — a
50-day SMA of $100 in 2010 and $50k in 2024 are not comparable, but
"close is 3% below 50-day SMA" is.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_WINDOWS = (5, 10, 20, 50)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def build(df: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_WINDOWS) -> pd.DataFrame:
    """SMA, EMA, and crossover features."""
    out = pd.DataFrame(index=df.index)
    close = df["close"]
    for w in windows:
        out[f"sma_{w}"] = close.rolling(w).mean() / close - 1
        out[f"ema_{w}"] = _ema(close, w) / close - 1

    # Common crossover spreads
    if 5 in windows and 20 in windows:
        out["ma_cross_5_20"] = out["sma_5"] - out["sma_20"]
    if 20 in windows and 50 in windows:
        out["ma_cross_20_50"] = out["sma_20"] - out["sma_50"]
    return out
