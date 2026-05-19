"""Tests for the feature library.

The key test here is ``test_no_lookahead_*``: it verifies the central
correctness property the project claims — that appending future rows
never changes the features for past rows. A feature that fails this
isn't a feature, it's a bug.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features import FAMILY_BUILDERS, build_all
from src.validators import validate_ohlcv


@pytest.fixture
def clean(synthetic_ohlcv):
    return validate_ohlcv(synthetic_ohlcv)


# ---------------------------------------------------------------- shape


def test_all_families_produce_features(clean):
    feats = build_all(clean)
    assert feats.shape[0] == len(clean)
    assert feats.shape[1] >= 30  # we claim 40+, allow some headroom


def test_unknown_family_raises(clean):
    with pytest.raises(ValueError, match="unknown feature families"):
        build_all(clean, families=("returns", "not_a_real_family"))


def test_individual_families_run(clean):
    for name, fn in FAMILY_BUILDERS.items():
        out = fn(clean)
        assert isinstance(out, pd.DataFrame), name
        assert len(out) == len(clean), name


# ------------------------------------------------------ the lookahead test


@pytest.mark.parametrize("family", list(FAMILY_BUILDERS.keys()))
def test_no_lookahead(family, clean):
    """For every feature family, adding future rows must not change past values.

    We compute features on the full frame, then on a truncated frame, and
    assert the truncated values equal the corresponding rows of the full
    computation. A failure means the feature is peeking at the future.
    """
    fn = FAMILY_BUILDERS[family]
    full = fn(clean)

    cutoff = len(clean) - 50
    truncated = fn(clean.iloc[:cutoff])

    common = full.iloc[:cutoff].dropna(how="all").index
    # Compare element-wise on overlapping rows
    pd.testing.assert_frame_equal(
        full.loc[common], truncated.loc[common], check_names=False, check_dtype=False
    )


def test_build_all_no_lookahead(clean):
    """Same property at the composed-pipeline level."""
    full = build_all(clean)
    cutoff = len(clean) - 50
    truncated = build_all(clean.iloc[:cutoff])
    common = full.iloc[:cutoff].dropna(how="all").index
    pd.testing.assert_frame_equal(
        full.loc[common], truncated.loc[common], check_names=False, check_dtype=False
    )


# ------------------------------------------------------------ smoke values


def test_rsi_in_range(clean):
    from src.features.momentum import rsi

    r = rsi(clean["close"]).dropna()
    assert (r >= 0).all() and (r <= 100).all()


def test_vol_regime_in_unit_interval(clean):
    from src.features.regime import build

    vr = build(clean)["vol_regime"].dropna()
    assert (vr >= 0).all() and (vr <= 1).all()


def test_returns_sum_relationship(clean):
    from src.features.returns import build

    r = build(clean, horizons=(1, 5))
    # ret_5 at time t equals sum of ret_1 over (t-4..t)
    expected = r["ret_1"].rolling(5).sum()
    pd.testing.assert_series_equal(
        r["ret_5"].dropna(), expected.dropna(), check_names=False
    )
