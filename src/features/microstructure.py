"""Microstructure features.

Bar-shape features that don't depend on a window: how the bar looked
in isolation. Surprisingly predictive on intraday data — a wide bar
that closed near the high carries different information than a wide
bar that closed mid-range.
"""
from __future__ import annotations

import pandas as pd


def build(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    # High-low range normalized by close
    out["hl_range"] = (df["high"] - df["low"]) / df["close"]

    # Where in the bar did the close land? 0 = at the low, 1 = at the high.
    # NaN-safe: bars where high == low produce NaN, which is the right answer.
    spread = (df["high"] - df["low"]).replace(0, pd.NA)
    out["close_loc"] = (df["close"] - df["low"]) / spread

    # Overnight gap: open vs previous close
    prev_close = df["close"].shift(1)
    out["gap"] = (df["open"] - prev_close) / prev_close

    return out
