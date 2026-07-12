# Raw Data Processing Lineage

This document covers only raw data preprocessing. Factor calculation, factor neutralization, validation and backtest steps are intentionally excluded.

## Current Raw Inputs

The current research dataset is built mainly from files under `回测数据集/`:

| Input | Main fields | Role |
| --- | --- | --- |
| `20170930-20251231_ori.csv` | `ts_code`, `trade_date`, OHLC, `vol`, `amount` | Base daily market panel. |
| `adj_factor_ori.csv` | `ts_code`, `trade_date`, `adj_factor` | Price and volume forward-adjustment. |
| `ST-Stocks_ori.csv` | `ts_code`, `trade_date`, `type` | Remove ST risk-warning rows. |
| `Suspension_data17-25_ori.csv` | `ts_code`, `trade_date`, `suspend_timing`, `suspend_type` | Mark suspended and resumed days. |
| `SWlevel1_sorted_ori.csv` | `trade_date`, `stock_code`, `industry_name` | Daily SW Level-1 industry mapping. |
| `daily_basic_ori.csv` | valuation, shares, market cap columns | Join daily valuation and market-cap fields. |
| `交易日历.csv`, `rf.csv` | `trade_date`, `rf` | Risk-free rate calendar alignment for later regression/backtest modules. |

Financial disclosure-derived inputs are stored in domain folders:

| Input | Main fields | Role |
| --- | --- | --- |
| `A股上市财务指标数据/*.csv` | `ts_code`, `ann_date`, `end_date`, `roe` | ROE disclosure source files. |
| `A股上市资产负债表数据/*.csv` | `ts_code`, `f_ann_date`, `end_date`, `total_hldr_eqy_exc_min_int` | Parent-company shareholder equity source files. |
| `A股上市现金流量表数据/构建固定资产等支付的现金_clean.csv` | `ts_code`, `f_ann_date`, `end_date`, `c_pay_acq_const_fiolta` | Capex cashflow source for TTM/QoQ. |

## Existing Processing Chain

### 1. OHLC Aggregation

`A股日度OHLC数据/历史OHLC数据合并.py` merges many `daily_stock_YYYYMMDD.csv` files into a date-range OHLC file.

Current issue: it is interactive, uses a hard-coded path, writes to the current working directory, and is not called by the main data pipeline.

### 2. Market Panel Cleaning

`数据获取与数据处理/DataCleanPipeline.py` builds `回测数据集/20170930-20251231_pipe.csv`:

1. Read raw OHLC, adjustment factors, ST rows, suspension rows, industry mapping and daily basic data.
2. Apply forward adjustment to OHLC and inverse-adjust `vol`; keep `amount` unchanged.
3. Remove rows marked `type == 'ST'`.
4. Remove first 120 trading days for stocks first appearing after the sample start.
5. Compute `dret` from adjusted `close` by stock.
6. Merge suspension data and set `is_trading`: `1` normal, `-1` full-day suspension, `-2` one-word limit day.
7. Cap abnormal `dret` by board rule: main board 10%, STAR/ChiNext 20%, BSE/other 30%.
8. Merge SW Level-1 industry by `trade_date` and `ts_code`; missing industry becomes `未分类`.
9. Forward-fill `daily_basic` fields within each stock and left-join them into the market panel.
10. Sort by `ts_code`, `trade_date` and write `20170930-20251231_pipe.csv`.

### 3. Disclosure Data Cleaning

Separate scripts clean quarterly/annual disclosure source files:

- `A股上市财务指标数据/数据合并与清洗.py` -> intended output `roe_data_clean.csv`.
- `A股上市资产负债表数据/数据清洗与合并.py` -> intended output `归母股东权益_clean.csv`.
- `A股上市现金流量表数据/数据合并与清洗.py` -> intended output is inconsistent with the later TTM script.

Common logic:

1. Glob CSV files in the source directory.
2. Keep `ts_code`, announcement date, `end_date`, and one target value column.
3. Normalize date fields to `YYYYMMDD` strings.
4. Drop duplicates and non-A-share-like rows where `ts_code` starts with `A`.
5. Within each stock, fill only the first missing value after a valid value.

### 4. Cashflow TTM/QoQ

`A股上市现金流量表数据/(TTM处理)环比增长率计算.py` builds `qoq.csv`:

1. Deduplicate by `ts_code`, `end_date`, keeping the latest `f_ann_date`.
2. Infer quarter from `end_date`.
3. Calculate TTM cashflow using cumulative financial report fields.
4. Calculate QoQ growth from consecutive TTM values.
5. Build a complete quarterly grid and fill only the first missing `ttm`/`qoq` value per stock.

### 5. Disclosure-to-Market Alignment

The three `pipe数据与...合并.py` scripts align disclosure data to each market day and write wide panels:

- `回测数据集/roe.csv`
- `回测数据集/归母股东权益.csv`
- `回测数据集/环比购买固定资产支出增长率(TTM).csv`

The intended rule is sound: for each stock and trade date, use only disclosure rows where both announcement date and report period end date are no later than the trade date. Then choose the latest announcement date.

## Hard Issues Found

1. **Intermediate data is not reproducible in the current checkout.** `roe.csv`, `归母股东权益.csv` and `环比购买固定资产支出增长率(TTM).csv` exist, but their documented cleaned inputs such as `roe_data_clean.csv` and `归母股东权益_clean.csv` are currently missing.

2. **`glob('*.csv')` reads outputs as inputs.** The disclosure cleaning scripts scan every CSV in the same folder where cleaned/intermediate outputs are also written. Re-running can accidentally ingest `qoq.csv`, `*_clean.csv` or other generated files.

3. **Hard-coded absolute paths make the flow non-portable.** Most scripts point to `C:\Users\63585\...` directly instead of resolving paths from the project root.

4. **One branch has an uninitialized variable bug.** In the old disclosure alignment scripts, if no valid disclosure match exists for a stock, `empty_fin` is created but `final_fin_data` is not assigned before later use.

5. **Return capping loses sign in the old code.** The old abnormal return adjustment returns `limit` when `abs(dret) > limit`; a -40% return becomes +10% on main board. The organized implementation preserves sign with `sign(dret) * limit`.

6. **Disclosure cleaning fills values before point-in-time alignment.** Filling missing quarterly disclosure values can be acceptable as a data vendor repair rule, but it should be documented and audited because it creates synthetic disclosure values.

7. **Several scripts mix exploration, plotting, processing and output.** This makes automated reruns brittle and hard to test.

## Organized Flow Added

New files:

- `data_pipeline/paths.py`: project-root-relative data paths.
- `data_pipeline/io.py`: checked CSV reads and lightweight audit output.
- `data_pipeline/market.py`: market panel preprocessing functions.
- `data_pipeline/financial.py`: disclosure cleaning, TTM/QoQ and point-in-time alignment functions.
- `run_data_pipeline.py`: command-line entry for non-factor data processing.

Smoke test without overwriting full outputs:

```powershell
python -B run_data_pipeline.py --steps market --sample-rows 5000
python -B run_data_pipeline.py --steps align-cashflow-qoq --sample-rows 5000
```

Full market panel rebuild:

```powershell
python -B run_data_pipeline.py --steps market
```

Full disclosure alignment steps require their cleaned disclosure inputs to exist:

```powershell
python -B run_data_pipeline.py --steps cashflow-qoq align-cashflow-qoq
python -B run_data_pipeline.py --steps align-roe align-equity
```

The new entry intentionally excludes factor calculation.
