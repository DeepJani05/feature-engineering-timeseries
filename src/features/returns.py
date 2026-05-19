"""Multi-horizon log returns.

Why log returns? Additivity across time: log(r_5) = sum(log(r_1) for 5 bars).
Makes rolling aggregations correct without compounding errors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 5, 15, 60)


def build(df: pd.DataFrame, horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """Compute log returns at the requested horizons.

    Parameters
    ----------
    df : pd.DataFrame
        Validated OHLCV frame.
    horizons : tuple[int, ...]
        Lookback windows in bars.

    Returns
    -------
    pd.DataFrame
        One column per horizon: `ret_{h}`. The 1-bar return is the
        single-step log return; longer horizons are sums of 1-bar
        returns over the trailing window (equivalent to log(p_t / p_{t-h})).
    """
    log_ret = np.log(df["close"] / df["close"].shift(1))
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        if h == 1:
            out["ret_1"] = log_ret
        else:
            out[f"ret_{h}"] = log_ret.rolling(h).sum()
    return out
