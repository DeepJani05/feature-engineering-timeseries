"""Momentum indicators.

Classical TA — chosen because (a) they're well-understood, (b) every
analyst recognizes them, and (c) they have decades of literature on
how they break.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index.

    Range [0, 100]; conventionally >70 = overbought, <30 = oversold.
    """
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD line, signal line, histogram (line - signal)."""
    ema12 = _ema(close, 12)
    ema26 = _ema(close, 26)
    line = ema12 - ema26
    signal = _ema(line, 9)
    return line, signal, line - signal


def stochastic_k(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Stochastic %K: where close sits in the period's range."""
    low_n = df["low"].rolling(period).min()
    high_n = df["high"].rolling(period).max()
    return 100 * (df["close"] - low_n) / (high_n - low_n)


def build(df: pd.DataFrame, roc_windows: tuple[int, ...] = (10, 20, 60)) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["rsi_14"] = rsi(df["close"], 14)

    line, signal, hist = macd(df["close"])
    out["macd"] = line
    out["macd_signal"] = signal
    out["macd_hist"] = hist

    for w in roc_windows:
        out[f"roc_{w}"] = df["close"].pct_change(w)

    out["stoch_k_14"] = stochastic_k(df, 14)
    return out
