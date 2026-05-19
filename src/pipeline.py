"""End-to-end pipeline orchestrator.

Read raw OHLCV -> validate -> build features -> drop warmup rows ->
return a clean feature matrix. Pure function; no side effects. The
CLI module wraps this with I/O.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from src.features import build_all
from src.validators import validate_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    features: pd.DataFrame
    n_input_rows: int
    n_output_rows: int
    n_dropped_warmup: int
    feature_columns: list[str]


def run(
    ohlcv: pd.DataFrame,
    *,
    families: tuple[str, ...] | None = None,
    drop_warmup: bool = True,
) -> PipelineResult:
    """Run the full feature pipeline on a single asset's OHLCV.

    Parameters
    ----------
    ohlcv : pd.DataFrame
        Raw OHLCV. Will be validated and normalized.
    families : tuple[str, ...] | None
        Subset of feature families to build. Defaults to all.
    drop_warmup : bool
        If True, drop the leading rows that contain NaNs from rolling
        windows. Almost always what you want — models can't train on
        NaN-laden rows anyway.
    """
    n_in = len(ohlcv)
    clean = validate_ohlcv(ohlcv)
    features = build_all(clean, families=families)

    n_dropped = 0
    if drop_warmup:
        before = len(features)
        features = features.dropna()
        n_dropped = before - len(features)

    logger.info(
        "pipeline.complete",
        extra={
            "n_input": n_in,
            "n_output": len(features),
            "n_dropped": n_dropped,
            "n_features": features.shape[1],
        },
    )

    return PipelineResult(
        features=features,
        n_input_rows=n_in,
        n_output_rows=len(features),
        n_dropped_warmup=n_dropped,
        feature_columns=list(features.columns),
    )
