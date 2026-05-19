# Feature Engineering Pipeline for Financial Time-Series Forecasting

> A production-grade, model-agnostic feature pipeline that transforms raw OHLCV data into **40+ engineered features** — rolling statistics, momentum, lags, volatility regimes — validated with walk-forward back-testing, persisted to Azure, and served to Power BI for analyst self-service.

[![CI](https://github.com/<your-handle>/feature-engineering-timeseries/actions/workflows/ci.yml/badge.svg)](./.github/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## 1. The Business Problem

Every quant team and FP&A group I've seen runs into the same wall: **the model isn't the bottleneck, the features are.**

- **Inconsistent definitions.** Three analysts compute "20-day momentum" three different ways. None of them match the back-test.
- **Feature drift goes unnoticed.** The pipeline that worked in January silently breaks in March when a vendor changes the timestamp format.
- **No reuse.** Each new forecasting project re-implements RSI, ATR, rolling vol from scratch — sometimes correctly, sometimes not.
- **Analysts can't self-serve.** Business users want to slice "momentum by sector" in Power BI but the features live in a Jupyter notebook on someone's laptop.

The cost: forecasting projects take 6 weeks instead of 2; back-tests are quietly optimistic because of look-ahead bugs; and Power BI dashboards lag because there's no governed feature store behind them.

## 2. Why I Built This

I picked this project to **separate the engineering of features from the modelling of them**. In my trading bot work I noticed the same feature code being copy-pasted across notebooks, with subtle drift. I wanted one library that:

1. Has a **single source of truth** for every feature definition.
2. Is **paranoid about look-ahead** — every feature passes a "future data can't change past values" test.
3. Is **deployable**, not just runnable. It reads from and writes to Azure, runs in CI, and surfaces metrics into Power BI.
4. Is **model-agnostic**. The feature matrix is the deliverable; what you fit on top — linear regression, XGBoost, LSTM, Prophet — is your problem.

## 3. What It Does

```
       ┌────────────────────┐
       │  Raw OHLCV         │   CSV / Parquet / Azure Blob
       │  (any frequency)   │
       └────────┬───────────┘
                │
                ▼
       ┌────────────────────┐
       │  Validation Layer  │   schema, monotonic index, no gaps,
       │                    │   no NaNs in OHLCV, sane ranges
       └────────┬───────────┘
                │
                ▼
       ┌────────────────────┐
       │  Feature Builder   │   40+ features, deterministic,
       │  (composable)      │   look-ahead-safe by construction
       └────────┬───────────┘
                │
                ▼
       ┌────────────────────┐
       │  Walk-Forward      │   out-of-sample CV, per-fold
       │  Validator         │   metrics, feature stability
       └────────┬───────────┘
                │
        ┌───────┴────────┐
        ▼                ▼
   ┌──────────┐    ┌──────────────┐
   │ Parquet  │    │ Power BI via │
   │ + Lake   │    │ Azure SQL    │
   └──────────┘    └──────────────┘
```

### Key results
- **40+ engineered features** across 7 families: returns, moving averages, volatility, momentum, volume, microstructure, regime
- **Look-ahead-safe by construction** — every feature has a unit test asserting that adding future rows doesn't change past values
- **Walk-forward CV** built in; outputs per-fold Sharpe, hit rate, drawdown, feature importance
- **Power BI-ready** — features land in a star schema in Azure SQL, with a daily refresh job

## 4. What's Different from a Notebook

| Notebook approach | This pipeline |
|---|---|
| Feature defined inline, re-derived per project | Defined once in `src/features/`, imported everywhere |
| Look-ahead bugs found by accident in production | Caught by `test_no_lookahead` on every PR |
| Train/test split is `iloc[:0.8*N]` | Walk-forward with rolling windows |
| Results live in `output.csv` | Lake parquet + SQL + Power BI |
| No reproducibility | Deterministic, seeded, versioned |

## 5. Repo Layout

```
feature-engineering-timeseries/
├── src/
│   ├── __init__.py
│   ├── pipeline.py             # end-to-end orchestrator
│   ├── validators.py           # input validation
│   ├── features/
│   │   ├── __init__.py
│   │   ├── returns.py
│   │   ├── moving_averages.py
│   │   ├── volatility.py
│   │   ├── momentum.py
│   │   ├── volume.py
│   │   ├── microstructure.py
│   │   └── regime.py
│   ├── validation/
│   │   └── walk_forward.py     # CV engine
│   ├── io/
│   │   ├── azure_blob.py       # read/write parquet to Blob
│   │   └── azure_sql.py        # load to star schema
│   └── cli.py                  # `python -m src.cli build --config ...`
├── notebooks/
│   └── 01_exploration.ipynb    # walk through a single asset
├── tests/
├── .github/workflows/ci.yml
├── config.yaml
├── requirements.txt
├── pyproject.toml
├── Dockerfile
└── README.md
```

## 6. Quickstart

```bash
git clone https://github.com/<your-handle>/feature-engineering-timeseries.git
cd feature-engineering-timeseries

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Build features for a CSV of OHLCV data
python -m src.cli build \
  --input data/sample_ohlcv.csv \
  --output data/features.parquet

# Run walk-forward validation
python -m src.cli validate \
  --input data/features.parquet \
  --label-horizon 5 \
  --train-window 504 \
  --test-window 21

# Push features into Azure SQL for Power BI
python -m src.cli load-sql \
  --input data/features.parquet \
  --table dbo.features_daily

pytest -v
```

## 7. Feature Catalog

| Family | Features | Why it matters |
|---|---|---|
| **Returns** | `ret_1`, `ret_5`, `ret_15`, `ret_60` | Multi-horizon momentum signal |
| **Moving averages** | `sma_{5,10,20,50}`, `ema_{5,10,20,50}`, crossovers | Trend detection |
| **Volatility** | `vol_{5,20,60}`, `atr_14`, Parkinson, Garman-Klass | Regime + position sizing |
| **Momentum** | `rsi_14`, MACD (line/signal/hist), `roc_{10,20,60}`, Stochastic %K | Overbought/oversold signals |
| **Volume** | `volume_z_20`, `dollar_volume`, OBV | Conviction behind a move |
| **Microstructure** | `hl_range`, `close_loc`, `gap` | Intraday character |
| **Regime** | `vol_regime` (percentile of rolling vol) | Stationarity proxy |

Each family is a separate module so you can import only what you need and unit-test families in isolation.

## 8. Design Principles

**One file per feature family.** Easy to find, easy to test, easy to deprecate. No 800-line `utils.py`.

**Pure functions only.** Every feature is `f(ohlcv: DataFrame) -> DataFrame`. No global state, no hidden side effects. Trivially composable.

**Look-ahead is a unit test, not a hope.** Each feature module has a test that appends future rows and asserts the past doesn't change. If you can't pass that test, your feature isn't a feature — it's a bug.

**The output is the contract.** A parquet file with a documented schema. Whatever consumes it (XGBoost notebook, Power BI report, R script) doesn't need to know how it was made.

**Walk-forward, never random shuffle.** Time series have temporal structure. Random splits leak future info backwards and inflate every metric you care about.

## 9. Power BI Integration

Features land in `dbo.features_daily` with this shape:

```
asset_id  | date       | feature_name | value
----------+------------+--------------+--------
BTC/USDT  | 2025-09-15 | ret_5        | 0.0231
BTC/USDT  | 2025-09-15 | rsi_14       | 67.4
BTC/USDT  | 2025-09-15 | vol_20       | 0.0182
...
```

The long format makes it trivial to build a Power BI slicer on `feature_name` and reuse one visual across all 40 features. A view `dbo.v_features_wide` pivots on demand for analysts who want the matrix shape.

## 10. Roadmap
- [ ] Add cross-sectional features (rank within universe, sector-relative momentum)
- [ ] Plug into Azure Data Factory for scheduled refresh
- [ ] Feature store integration (Feathr or Tecton)
- [ ] Online/streaming feature computation for live trading systems

## 11. License
MIT
