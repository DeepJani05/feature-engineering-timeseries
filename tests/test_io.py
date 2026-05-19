"""Tests for the SQL loader (long-format conversion only — no DB needed)."""
from __future__ import annotations

import pandas as pd

from src.io.azure_sql import to_long_format


def test_long_format_basic():
    wide = pd.DataFrame(
        {"ret_1": [0.01, 0.02], "rsi_14": [55.0, 60.0]},
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )
    long_df = to_long_format(wide, asset_id="BTC")
    assert set(long_df.columns) == {"asset_id", "timestamp", "feature_name", "value"}
    assert len(long_df) == 4  # 2 rows x 2 features
    assert (long_df["asset_id"] == "BTC").all()


def test_long_format_drops_nans():
    wide = pd.DataFrame(
        {"ret_1": [0.01, None], "rsi_14": [55.0, 60.0]},
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )
    long_df = to_long_format(wide, asset_id="BTC")
    assert len(long_df) == 3  # one NaN dropped
    assert not long_df["value"].isna().any()


def test_long_format_preserves_ordering():
    wide = pd.DataFrame(
        {"a": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    )
    long_df = to_long_format(wide, asset_id="X")
    assert list(long_df["value"]) == [1.0, 2.0, 3.0]
