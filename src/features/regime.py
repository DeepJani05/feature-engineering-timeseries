"""Regime features.

Stationarity is a polite fiction in financial data. The volatility you
saw in calm markets isn't the volatility you'll see in a crisis. These
features try to *flag* which regime you're in so a downstream model
can adapt rather than averaging across regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build(
    df: pd.DataFrame,
    short_vol_window: int = 20,
    lookback: int = 250,
) -> pd.DataFrame:
    """Volatility-percentile regime feature.

    For each bar, compute the rolling std of returns over the last
    `short_vol_window` bars. Then compare that to the trailing
    `lookback` bars and emit its percentile rank in [0, 1].

        0.0  = quietest regime in recent history
        0.5  = median
        1.0  = most volatile in recent history

    Bars where we don't have enough history to compute the percentile
    are emitted as NaN (caller drops them).
    """
    out = pd.DataFrame(index=df.index)
    log_ret = np.log(df["close"] / df["close"].shift(1))
    short_vol = log_ret.rolling(short_vol_window).std()

    # Percentile rank against the trailing window
    out["vol_regime"] = short_vol.rolling(lookback).apply(
        lambda x: x.rank(pct=True).iloc[-1], raw=False
    )
    return out
