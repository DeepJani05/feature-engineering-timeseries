"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """Deterministic geometric Brownian-ish OHLCV for tests."""
    rng = np.random.default_rng(42)
    n = 500
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rets = rng.normal(0, 0.002, size=n)
    close = 100 * np.exp(rets.cumsum())
    high = close * (1 + np.abs(rng.normal(0, 0.001, size=n)))
    low = close * (1 - np.abs(rng.normal(0, 0.001, size=n)))
    open_ = close * (1 + rng.normal(0, 0.0005, size=n))
    # Clamp open within [low, high] so validators don't complain
    open_ = np.clip(open_, low, high)
    volume = rng.integers(1_000, 10_000, size=n).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=idx,
    )
