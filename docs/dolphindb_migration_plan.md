# DolphinDB Migration Plan

This plan keeps research CSVs as inputs, but stores maintainable engineering tables in DolphinDB.

## Table Boundaries

Do not import the current wide CSVs as single production tables. Files such as `roe.csv` repeat the full market panel plus one financial field, which makes updates and lineage hard to maintain.

Use these layers instead:

| Layer | Tables | Purpose |
| --- | --- | --- |
| Market facts | `core_market_bar_daily`, `core_market_status_daily`, `core_market_valuation_daily`, `core_industry_sw_l1_daily` | Daily OHLC, trading status, valuation and industry facts. |
| Point-in-time features | `feature_financial_daily` | Daily features visible at `trade_date`, preserving `ann_date` and `end_date` to audit lookahead risk. |
| Factor outputs | `factor_daily`, `factor_meta`, `factor_run_log` | Factor values and run metadata. |

Raw vendor files can be added later as `raw_*` tables if replay/audit requirements grow. The first milestone focuses on the tables needed by factor calculation and validation.

## Partitioning

Large daily tables use composite partitioning:

- `month`: monthly value partition derived from `trade_date`.
- `ts_code`: hash partition with 64 buckets.

This supports both common access paths: full-market daily cross sections and per-stock time series.

## Local DolphinDB

This machine has DolphinDB under `C:\Users\63585\DolphinDB` and a desktop starter:

```powershell
& "C:\Users\63585\Desktop\Start DolphinDB.bat"
```

The local single node is configured as `localhost:8848:local8848` with default login `admin/123456`.

## Schema Setup

Create or verify the database schema with the existing Node runner:

```powershell
cd C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem
node C:\Users\63585\DolphinDB\ddb-runner.mjs --file dolphindb_scripts\schema.dos
```

Run a DolphinDB-side smoke import of 10,000 market rows:

```powershell
node C:\Users\63585\DolphinDB\ddb-runner.mjs --file dolphindb_scripts\import_sample.dos
```

## Chunked Python Sync

Install the DolphinDB Python API if it is not already installed:

```powershell
pip install -r requirements.txt
```

Smoke test imports without loading the full multi-GB files:

```powershell
python run_dolphindb_sync.py --sample-rows 10000 market
python run_dolphindb_sync.py --sample-rows 10000 features
```

Full market and financial feature sync:

```powershell
python run_dolphindb_sync.py market
python run_dolphindb_sync.py features
```

Sync an existing factor CSV:

```powershell
python run_dolphindb_sync.py factor --factor-name alpha_60 --factor-path Factors\alpha_60.csv --factor-version v1
```

If `--factor-path` is omitted, the script uses `Factors/<factor_name>.csv`.

Calculate `alpha_60` directly from DolphinDB market tables and write to `factor_daily` without materializing a CSV:

```powershell
python run_alpha60_dolphindb.py --start-month 2017-10 --end-month 2025-12 --replace --factor-version v1
```

Calculate `alpha_57` directly from DolphinDB market tables and write to `factor_daily`:

```powershell
python run_alpha57_dolphindb.py --replace --factor-version v1
```

## Incremental Engineering Path

1. Run `schema` once per DolphinDB environment.
2. Import market facts from `回测数据集/20170930-20251231_pipe.csv`.
3. Import daily point-in-time financial features from `roe.csv`, `归母股东权益.csv` and `环比购买固定资产支出增长率(TTM).csv`.
4. Change factor scripts to read from DolphinDB instead of local CSV.
5. After each factor calculation, append results to `factor_daily`.
6. Change validation modules to read `factor_daily` by `factor_name`, `factor_version` and date window.

## Data Quality Checks To Add Next

- Row uniqueness by `(trade_date, ts_code)` in market fact tables.
- Point-in-time invariant: `end_date <= ann_date <= trade_date` for `feature_financial_daily`.
- Factor run completeness by daily stock count and null ratio.
- Stable `data_version` labels for every import batch.
