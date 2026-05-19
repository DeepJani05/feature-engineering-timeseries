"""Command-line interface.

Subcommands:
    build       — read OHLCV, produce a feature matrix, write parquet
    validate    — run walk-forward CV on a feature matrix + labels
    load-sql    — push a feature matrix into Azure SQL (long format)

Examples
--------
    python -m src.cli build --input data/btc.csv --output data/features.parquet
    python -m src.cli validate --input data/features.parquet --label-horizon 5
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

from src.io.azure_blob import read_parquet, write_parquet
from src.pipeline import run as run_pipeline
from src.validation.walk_forward import (
    directional_accuracy,
    hit_rate,
    information_coefficient,
    walk_forward_cv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------------------------------------------- build


def _cmd_build(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.input, parse_dates=[args.timestamp_col], index_col=args.timestamp_col)
    df = df.sort_index()
    result = run_pipeline(df, families=tuple(args.families) if args.families else None)

    write_parquet(
        result.features,
        args.output,
        storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    )
    print(
        f"Built {result.n_output_rows} feature rows "
        f"({result.features.shape[1]} columns) -> {args.output}"
    )
    print(f"Dropped {result.n_dropped_warmup} warmup rows.")
    return 0


# -------------------------------------------------------------- validate


def _cmd_validate(args: argparse.Namespace) -> int:
    features = read_parquet(
        args.input,
        storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    )

    # Labels: forward log return at the requested horizon, sign-thresholded
    # to {-1, 0, 1} when --classify is set.
    if "close" in features.columns:
        close = features["close"]
    elif args.price_csv:
        close = pd.read_csv(
            args.price_csv, parse_dates=["timestamp"], index_col="timestamp"
        )["close"]
    else:
        raise SystemExit("need either 'close' in features or --price-csv")

    fwd_ret = np.log(close.shift(-args.label_horizon) / close)
    if args.classify:
        labels = np.sign(fwd_ret)
    else:
        labels = fwd_ret

    common = features.index.intersection(labels.dropna().index)
    X = features.loc[common].select_dtypes(include="number").dropna(axis=1, how="any")
    y = labels.loc[common].dropna()
    common = X.index.intersection(y.index)
    X, y = X.loc[common], y.loc[common]

    from xgboost import XGBRegressor

    def _train(X_tr, y_tr):
        m = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        m.fit(X_tr, y_tr)
        return m

    def _predict(model, X_te):
        return model.predict(X_te)

    metrics = {
        "hit_rate": hit_rate,
        "directional_accuracy": directional_accuracy,
        "ic": information_coefficient,
    }

    result = walk_forward_cv(
        X, y, _train, _predict, metrics,
        train_window=args.train_window,
        test_window=args.test_window,
    )

    summary = result.summary()
    print("\nPer-fold metrics:")
    print(summary.to_string(index=False))
    print("\nAggregate (mean across folds):")
    for k, v in result.aggregate_metrics().items():
        print(f"  {k:25s} {v:+.4f}")

    if result.feature_importance is not None:
        print("\nTop 15 features by importance:")
        print(result.feature_importance.head(15).to_string())
    return 0


# --------------------------------------------------------------- load-sql


def _cmd_load_sql(args: argparse.Namespace) -> int:
    from src.io.azure_sql import load_to_sql

    features = read_parquet(
        args.input,
        storage_connection_string=os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    )
    conn = os.getenv("AZURE_SQL_CONNECTION_STRING")
    if not conn:
        raise SystemExit("AZURE_SQL_CONNECTION_STRING is not set")

    n = load_to_sql(
        features, args.asset_id,
        connection_string=conn, table=args.table, if_exists=args.if_exists,
    )
    print(f"Loaded {n} rows to {args.table} (asset_id={args.asset_id}).")
    return 0


# ---------------------------------------------------------------- main


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="feature-pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)

    pb = sub.add_parser("build", help="Build a feature matrix from raw OHLCV.")
    pb.add_argument("--input", required=True)
    pb.add_argument("--output", required=True)
    pb.add_argument("--timestamp-col", default="timestamp")
    pb.add_argument("--families", nargs="*", default=None)
    pb.set_defaults(func=_cmd_build)

    pv = sub.add_parser("validate", help="Run walk-forward CV over a feature matrix.")
    pv.add_argument("--input", required=True)
    pv.add_argument("--price-csv", help="OHLCV CSV with a 'close' column (if not embedded).")
    pv.add_argument("--label-horizon", type=int, default=5)
    pv.add_argument("--train-window", type=int, default=504)
    pv.add_argument("--test-window", type=int, default=21)
    pv.add_argument("--classify", action="store_true", help="Sign-threshold the label.")
    pv.set_defaults(func=_cmd_validate)

    pl = sub.add_parser("load-sql", help="Load a feature matrix into Azure SQL.")
    pl.add_argument("--input", required=True)
    pl.add_argument("--asset-id", required=True)
    pl.add_argument("--table", default="dbo.features_daily")
    pl.add_argument("--if-exists", default="append", choices=["append", "replace"])
    pl.set_defaults(func=_cmd_load_sql)

    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
