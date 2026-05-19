"""Feature builders.

Each submodule exposes a ``build(df: pd.DataFrame) -> pd.DataFrame``
function that returns features aligned to the input index. The
``build_all`` helper here composes them into the canonical matrix
used by the pipeline.
"""
from __future__ import annotations

import pandas as pd

from . import (
    microstructure,
    momentum,
    moving_averages,
    regime,
    returns,
    volatility,
    volume,
)

FAMILY_BUILDERS = {
    "returns": returns.build,
    "moving_averages": moving_averages.build,
    "volatility": volatility.build,
    "momentum": momentum.build,
    "volume": volume.build,
    "microstructure": microstructure.build,
    "regime": regime.build,
}


def build_all(
    df: pd.DataFrame,
    *,
    families: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Build every (or selected) feature family and concatenate the result."""
    families = families or tuple(FAMILY_BUILDERS.keys())
    unknown = set(families) - set(FAMILY_BUILDERS.keys())
    if unknown:
        raise ValueError(f"unknown feature families: {sorted(unknown)}")

    parts = [FAMILY_BUILDERS[name](df) for name in families]
    return pd.concat(parts, axis=1)


__all__ = ["build_all", "FAMILY_BUILDERS"]
