"""Volatility features.

Three different estimators because each picks up something different:
    - Rolling std of log returns: classical, noisy on small windows.
    - ATR: range-based, handles gaps and noisy ticks.
    - Parkinson: high-low-only, ~5x more efficient than close-to-close.
    - Garman-Klass: uses OHLC, even more efficient under some assumptions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_VOL_WINDOWS = (5, 20, 60)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.rolling(period).mean()


def _parkinson(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    return (
        np.log(high / low).pow(2).rolling(window).mean() / (4 * np.log(2))
    ).pow(0.5)


def _garman_klass(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Garman-Klass volatility — assumes no drift, no jumps.

    Variance estimator:
        0.5 * ln(H/L)^2 - (2*ln(2) - 1) * ln(C/O)^2
    """
    log_hl = np.log(df["high"] / df["low"]).pow(2)
    log_co = np.log(df["close"] / df["open"]).pow(2)
    var = 0.5 * log_hl - (2 * np.log(2) - 1) * log_co
    return var.rolling(window).mean().clip(lower=0).pow(0.5)


def build(df: pd.DataFrame, windows: tuple[int, ...] = DEFAULT_VOL_WINDOWS) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))

    for w in windows:
        out[f"vol_{w}"] = log_ret.rolling(w).std()

    out["atr_14"] = _atr(df["high"], df["low"], df["close"], 14)
    out["parkinson_20"] = _parkinson(df["high"], df["low"], 20)
    out["garman_klass_20"] = _garman_klass(df, 20)
    return out
