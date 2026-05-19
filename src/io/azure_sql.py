"""Azure SQL loader.

Lands the wide feature matrix into a long-format table that Power BI
likes:

    asset_id | timestamp | feature_name | value

Why long format? One slicer (`feature_name`) drives every visual.
Without it, an analyst building a "compare RSI across 50 assets"
report has to add 50 columns to the model. Long format also keeps the
schema stable when we add or remove features.

For raw-matrix queries, `dbo.v_features_wide` pivots on demand.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def to_long_format(features: pd.DataFrame, asset_id: str) -> pd.DataFrame:
    """Convert a wide feature matrix to the long-format SQL schema."""
    long_df = features.copy()
    long_df.index.name = "timestamp"
    long_df = long_df.reset_index().melt(
        id_vars=["timestamp"], var_name="feature_name", value_name="value"
    )
    long_df.insert(0, "asset_id", asset_id)
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)
    return long_df


def load_to_sql(
    features: pd.DataFrame,
    asset_id: str,
    *,
    connection_string: str,
    table: str = "dbo.features_daily",
    chunksize: int = 10_000,
    if_exists: str = "append",
) -> int:
    """Load a feature matrix into Azure SQL in long format.

    Returns
    -------
    int
        Number of rows written.
    """
    import sqlalchemy as sa  # lazy

    long_df = to_long_format(features, asset_id)
    if long_df.empty:
        logger.warning("nothing to load — long-format frame is empty")
        return 0

    engine = sa.create_engine(connection_string, fast_executemany=True)
    with engine.begin() as conn:
        long_df.to_sql(
            table.split(".")[-1],
            con=conn,
            schema=table.split(".")[0] if "." in table else None,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
            method=None,
        )
    logger.info("loaded %d rows to %s (asset=%s)", len(long_df), table, asset_id)
    return len(long_df)


CREATE_TABLE_DDL = """
CREATE TABLE dbo.features_daily (
    asset_id      NVARCHAR(64)  NOT NULL,
    [timestamp]   DATETIME2     NOT NULL,
    feature_name  NVARCHAR(64)  NOT NULL,
    value         FLOAT         NOT NULL,
    CONSTRAINT PK_features_daily PRIMARY KEY (asset_id, [timestamp], feature_name)
);

CREATE INDEX IX_features_daily_feature ON dbo.features_daily (feature_name);
CREATE INDEX IX_features_daily_ts      ON dbo.features_daily ([timestamp]);

-- Wide view for analysts who want the matrix shape
CREATE VIEW dbo.v_features_wide AS
SELECT *
FROM (
    SELECT asset_id, [timestamp], feature_name, value
    FROM dbo.features_daily
) AS src
PIVOT (
    MAX(value) FOR feature_name IN (
        [ret_1], [ret_5], [rsi_14], [vol_20], [macd], [vol_regime] -- extend as needed
    )
) AS p;
"""
