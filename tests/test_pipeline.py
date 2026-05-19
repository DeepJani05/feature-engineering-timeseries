"""End-to-end pipeline test."""
from __future__ import annotations

from src.pipeline import run


def test_pipeline_runs_end_to_end(synthetic_ohlcv):
    result = run(synthetic_ohlcv)
    assert result.n_input_rows == len(synthetic_ohlcv)
    assert result.n_output_rows > 0
    assert result.n_output_rows < result.n_input_rows  # warmup dropped
    assert len(result.feature_columns) >= 30
    # No NaNs in the final matrix (warmup dropped by default)
    assert not result.features.isna().any().any()


def test_pipeline_with_family_subset(synthetic_ohlcv):
    result = run(synthetic_ohlcv, families=("returns", "volatility"))
    # All columns belong to the selected families
    for col in result.feature_columns:
        assert col.startswith(("ret_", "vol_", "atr_", "parkinson", "garman"))


def test_pipeline_with_warmup_kept(synthetic_ohlcv):
    full = run(synthetic_ohlcv, drop_warmup=True)
    kept = run(synthetic_ohlcv, drop_warmup=False)
    assert kept.n_output_rows == kept.n_input_rows
    assert kept.n_output_rows > full.n_output_rows
