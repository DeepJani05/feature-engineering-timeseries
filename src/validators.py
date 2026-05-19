"""Input validation for OHLCV data.

Cheap to run, expensive to skip. Bad OHLCV data silently produces bad
features, which silently train bad models, which lose money or make bad
business decisions. Every assumption the feature builders make is
verified here, up front, with a clear error message when violated.
"""
from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class OHLCVValidationError(ValueError):
    """Raised when input data violates the OHLCV contract."""


def validate_ohlcv(df: pd.DataFrame, *, max_gap_bars: int = 5) -> pd.DataFrame:
    """Run every input check and return a cleaned, normalized frame.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns open/high/low/close/volume (case-insensitive)
        and a DatetimeIndex.
    max_gap_bars : int
        Maximum number of consecutive missing bars allowed before we
        raise. Set conservatively — large gaps in financial data often
        signal data-vendor issues, holidays handled wrong, or splits
        applied incorrectly.

    Returns
    -------
    pd.DataFrame
        Lowercase columns, sorted by index, with the typed columns
        guaranteed to be float64.
    """
    if not isinstance(df, pd.DataFrame):
        raise OHLCVValidationError(f"expected DataFrame, got {type(df).__name__}")

    if df.empty:
        raise OHLCVValidationError("input is empty")

    df = df.rename(columns=str.lower).copy()

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise OHLCVValidationError(f"missing required columns: {sorted(missing)}")

    if not isinstance(df.index, pd.DatetimeIndex):
        raise OHLCVValidationError(
            f"index must be DatetimeIndex, got {type(df.index).__name__}"
        )

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

    if df.index.has_duplicates:
        dup_count = df.index.duplicated().sum()
        raise OHLCVValidationError(f"index has {dup_count} duplicate timestamps")

    # Type coercion
    for c in REQUIRED_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        bad = df[list(REQUIRED_COLUMNS)].isna().sum()
        raise OHLCVValidationError(f"OHLCV contains NaN values:\n{bad}")

    # Sanity ranges
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise OHLCVValidationError("prices must be strictly positive")
    if (df["volume"] < 0).any():
        raise OHLCVValidationError("volume cannot be negative")
    if (df["high"] < df["low"]).any():
        raise OHLCVValidationError("found bars where high < low")
    if (df["high"] < df[["open", "close"]].max(axis=1)).any():
        raise OHLCVValidationError("high is below open or close on some bars")
    if (df["low"] > df[["open", "close"]].min(axis=1)).any():
        raise OHLCVValidationError("low is above open or close on some bars")

    # Gap detection (warn loudly for big gaps)
    inferred = pd.infer_freq(df.index)
    if inferred is not None:
        full = pd.date_range(df.index[0], df.index[-1], freq=inferred)
        missing_bars = full.difference(df.index)
        if len(missing_bars) > max_gap_bars:
            raise OHLCVValidationError(
                f"detected {len(missing_bars)} missing bars (cap {max_gap_bars}); "
                f"first 5 missing: {list(missing_bars[:5])}"
            )

    return df[list(REQUIRED_COLUMNS)]
