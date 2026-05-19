"""Volume-based features.

Price tells you what; volume tells you how much conviction was behind
it. The features below capture three different angles on volume: scale
(z-score), economic size (dollar volume), and accumulation/distribution
(OBV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def volume_zscore(volume: pd.Series, window: int = 20) -> pd.Series:
    mu = volume.rolling(window).mean()
    sigma = volume.rolling(window).std()
    return (volume - mu) / sigma


def on_balance_volume(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV: cumulative volume signed by direction of close."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["volume_z_20"] = volume_zscore(df["volume"], 20)
    out["dollar_volume"] = df["close"] * df["volume"]
    out["obv"] = on_balance_volume(df["close"], df["volume"])
    return out
