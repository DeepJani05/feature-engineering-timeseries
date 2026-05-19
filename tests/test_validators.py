"""Tests for input validation."""
from __future__ import annotations

import pandas as pd
import pytest

from src.validators import OHLCVValidationError, validate_ohlcv


def test_clean_input_passes(synthetic_ohlcv):
    out = validate_ohlcv(synthetic_ohlcv)
    assert set(out.columns) == {"open", "high", "low", "close", "volume"}
    assert out.index.is_monotonic_increasing


def test_uppercase_columns_normalized(synthetic_ohlcv):
    df = synthetic_ohlcv.rename(columns=str.upper)
    out = validate_ohlcv(df)
    assert "close" in out.columns


def test_missing_column_raises(synthetic_ohlcv):
    with pytest.raises(OHLCVValidationError, match="missing required columns"):
        validate_ohlcv(synthetic_ohlcv.drop(columns="volume"))


def test_non_datetime_index_raises(synthetic_ohlcv):
    df = synthetic_ohlcv.reset_index(drop=True)
    with pytest.raises(OHLCVValidationError, match="DatetimeIndex"):
        validate_ohlcv(df)


def test_negative_price_raises(synthetic_ohlcv):
    bad = synthetic_ohlcv.copy()
    bad.iloc[5, bad.columns.get_loc("close")] = -1.0
    with pytest.raises(OHLCVValidationError, match="strictly positive"):
        validate_ohlcv(bad)


def test_high_below_low_raises(synthetic_ohlcv):
    bad = synthetic_ohlcv.copy()
    bad.iloc[3, bad.columns.get_loc("high")] = 1.0
    bad.iloc[3, bad.columns.get_loc("low")] = 100.0
    with pytest.raises(OHLCVValidationError, match="high < low"):
        validate_ohlcv(bad)


def test_duplicate_timestamps_raises(synthetic_ohlcv):
    dup = pd.concat([synthetic_ohlcv, synthetic_ohlcv.iloc[[10]]])
    with pytest.raises(OHLCVValidationError, match="duplicate"):
        validate_ohlcv(dup)


def test_empty_input_raises():
    with pytest.raises(OHLCVValidationError, match="empty"):
        validate_ohlcv(pd.DataFrame())
