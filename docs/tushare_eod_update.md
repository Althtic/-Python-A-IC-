# Tushare Post-Market Data Update

## Configuration

The project-local `.env` must contain `TUSHARE_TOKEN`. The token is read only
by `data_pipeline.tushare_update` and is never written to logs or DolphinDB.

## Data Layers

Tushare source records are written to `raw_tushare_*` tables with a run ID and
ingestion timestamp. They are the audit and replay layer.

The market update also writes the existing compatibility tables and the new
`core_market_bar_daily_adjusted` table. The new table stores, per stock-day:

- Raw OHLC, volume, amount and `adj_factor`.
- `qfq_*` front-adjusted OHLC and volume.
- `hfq_*` back-adjusted OHLC and volume.
- `qfq_factor`, `hfq_factor`, and `hfq_anchor_date`.

The formulas are:

```text
qfq_factor = adj_factor / latest_adj_factor
hfq_factor = adj_factor / first_adj_factor
adjusted_OHLC = raw_OHLC * adjustment_factor
adjusted_volume = raw_volume / adjustment_factor
```

The front-adjustment anchor is stored in `etl_watermark` as
`market_price_anchor`. When the latest factor changes, DolphinDB rescales
existing front-adjusted values before the current date window is replaced.

## Commands

Normal post-market market update with a five-trading-day revision lookback:

```powershell
python -B run_tushare_update.py market
```

Refresh the SW2021 Level-1 membership snapshot and materialize the requested
market dates:

```powershell
python -B run_tushare_update.py market --refresh-industry
```

Refresh completed financial report periods. The selector excludes a quarter
until its statutory disclosure deadline has passed:

```powershell
python -B run_tushare_update.py financial --financial-periods 12
```

For a full historical market backfill, use controlled non-overlapping ranges.
Each trade date issues four Tushare requests, so start with a small window and
inspect `etl_run_log` and `data_quality_log` before continuing:

```powershell
python -B run_tushare_update.py market --start-date 20140101 --end-date 20140131 --refresh-industry
```

After raw adjustment factors have been backfilled for the full desired range,
rerun the same market ranges to ensure every `hfq_anchor_date` points to the
first available source factor. Do not use a later date as an artificial back
adjustment base.

## Schedule

`scripts/install_tushare_eod_tasks.ps1` registers three current-user tasks:

- Weekdays 18:30: market data and five-day revision lookback.
- Fridays 20:00: market update plus SW2021 industry refresh.
- Saturdays 10:00: twelve completed financial report periods.

Tasks use `StartWhenAvailable`; they run after the next login if the machine
was unavailable at the scheduled time.

## Quality Gates

The market writer rejects a batch before replacing data when it has fewer than
4,000 daily stocks, duplicate stock-day keys, or incomplete same-day factor
coverage. It then verifies post-write row counts. Run metadata, watermarks and
quality results are persisted in `etl_run_log`, `etl_watermark` and
`data_quality_log`.

## Industry and Factor Follow-up

`python -B run_tushare_update.py industry --start-date 20240101 --end-date 20251231`
rebuilds the existing post-2024 SW2021 daily mappings month by month. The
command requires at least 95% classified rows and records the coverage in the
run log.

The market task accepts `--update-factors`. It checks that the latest 120 open
market dates are present before invoking the 27 formulas that completed the
full-history validation. Each affected month is calculated first, then its old
factor rows are deleted and replaced. Known failed formulas are excluded until
fixed and independently validated.