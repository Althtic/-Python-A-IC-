"""Tushare-backed, auditable end-of-day data updates for QuantSystem."""

from __future__ import annotations

import datetime as dt
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from data_pipeline.dolphindb_sync import (
    DEFAULT_DB_PATH,
    MARKET_BAR_COLUMNS,
    MARKET_STATUS_COLUMNS,
    MARKET_VALUATION_COLUMNS,
    connect,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOKEN_PLACEHOLDER = "YOUR_TUSHARE_TOKEN"
SW_SOURCE = "TUSHARE_SW2021"


@dataclass(frozen=True)
class UpdateWindow:
    start_date: str
    end_date: str
    trade_dates: list[str]
    adjustment_anchor_date: str


@dataclass
class UpdateReport:
    run_id: str
    trade_dates: list[str]
    raw_rows: dict[str, int]
    core_rows: dict[str, int]
    quality_checks: list[dict[str, Any]]


def _date_literal(value: str | pd.Timestamp | dt.date) -> str:
    return pd.Timestamp(value).strftime("%Y.%m.%d")


def _date_string(value: Any) -> str:
    return pd.Timestamp(value).strftime("%Y%m%d")


def _to_datetime(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce").dt.normalize()
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return pd.to_datetime(numeric.astype(str), format="%Y%m%d", errors="coerce")


def _market_name(ts_code: pd.Series) -> pd.Series:
    code = ts_code.astype(str).str.split(".").str[0]
    return np.select(
        [
            code.str.startswith(("8", "9")),
            code.str.startswith(("688", "689")),
            code.str.startswith(("300", "301", "302")),
            code.str.startswith(("60", "000", "001", "002", "003")),
        ],
        ["北交所", "科创板", "创业板", "沪深主板"],
        default="其他",
    )


def load_tushare_token(project_root: Path = PROJECT_ROOT) -> str:
    env_path = project_root / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"Missing {env_path}. Add TUSHARE_TOKEN to the local .env file.")

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")

    token = values.get("TUSHARE_TOKEN", "")
    if not token or token == TOKEN_PLACEHOLDER:
        raise ValueError("TUSHARE_TOKEN is not configured in .env.")
    return token


class TushareClient:
    """Small retrying wrapper around the Tushare Pro client."""

    def __init__(self, token: str, retries: int = 3, retry_delay_seconds: float = 1.0):
        try:
            import tushare as ts
        except ModuleNotFoundError as exc:
            raise RuntimeError("Missing tushare. Install it with `pip install tushare`.") from exc
        self.pro = ts.pro_api(token)
        self.retries = retries
        self.retry_delay_seconds = retry_delay_seconds

    def request(self, method: str, **kwargs: Any) -> pd.DataFrame:
        func: Callable[..., pd.DataFrame] = getattr(self.pro, method)
        error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                frame = func(**kwargs)
                if frame is None:
                    raise RuntimeError(f"Tushare {method} returned None")
                return frame
            except Exception as exc:  # Provider exceptions are not stable across tushare versions.
                error = exc
                if attempt == self.retries:
                    break
                time.sleep(self.retry_delay_seconds * attempt)
        raise RuntimeError(f"Tushare {method} failed after {self.retries} attempts: {error}") from error

    def open_dates(self, start_date: str, end_date: str) -> list[str]:
        calendar = self.request(
            "trade_cal",
            exchange="",
            start_date=start_date,
            end_date=end_date,
            fields="cal_date,is_open",
        )
        if calendar.empty:
            return []
        return sorted(calendar.loc[calendar["is_open"].astype(str) == "1", "cal_date"].astype(str).tolist())

    def latest_open_date(self, end_date: str) -> str:
        dates = self.open_dates(_date_string(pd.Timestamp(end_date) - pd.Timedelta(days=31)), end_date)
        if not dates:
            raise RuntimeError(f"No open trading date found on or before {end_date}.")
        return dates[-1]

    def market_day(self, trade_date: str) -> dict[str, pd.DataFrame]:
        frames = {
            "daily": self.request("daily", trade_date=trade_date),
            "adj_factor": self.request("adj_factor", trade_date=trade_date),
            "daily_basic": self.request("daily_basic", trade_date=trade_date),
            "suspend": self.request("suspend_d", trade_date=trade_date),
        }
        if frames["daily"].empty:
            raise RuntimeError(f"Tushare daily returned no rows for the open date {trade_date}.")
        return frames

    def sw_l1_membership(self) -> pd.DataFrame:
        classes = self.request("index_classify", level="L1", src="SW2021")
        if classes.empty:
            raise RuntimeError("Tushare index_classify returned no SW2021 L1 industries.")
        parts: list[pd.DataFrame] = []
        for row in classes.itertuples(index=False):
            members = self.request("index_member_all", l1_code=row.index_code, is_new="Y")
            if members.empty:
                continue
            members = members[["ts_code", "in_date", "out_date"]].copy()
            members["industry_code"] = row.index_code
            members["industry_name"] = row.industry_name
            parts.append(members)
        if not parts:
            raise RuntimeError("Tushare returned no SW2021 L1 members.")
        result = pd.concat(parts, ignore_index=True).drop_duplicates(
            ["ts_code", "industry_code", "in_date", "out_date"]
        )
        result["source"] = SW_SOURCE
        return result

    def financial_period(self, period: str) -> dict[str, pd.DataFrame]:
        return {
            "fina_indicator": self.request(
                "fina_indicator_vip", period=period, fields="ts_code,ann_date,end_date,roe"
            ),
            "balancesheet": self.request(
                "balancesheet_vip",
                period=period,
                fields="ts_code,ann_date,f_ann_date,end_date,total_hldr_eqy_exc_min_int",
            ),
            "cashflow": self.request(
                "cashflow_vip",
                period=period,
                fields="ts_code,ann_date,f_ann_date,end_date,c_pay_acq_const_fiolta",
            ),
            "income": self.request(
                "income_vip",
                period=period,
                fields="ts_code,ann_date,f_ann_date,end_date,revenue,total_revenue",
            ),
        }


UPDATE_SCHEMA = r'''
dbPath = "{db_path}";
if (!existsDatabase(dbPath)) throw "Missing DolphinDB database: " + dbPath;
db = database(dbPath);
def createPartitionedIfAbsent(db, dbPath, tableName, schema, partitionColumns) {{
    if (!existsTable(dbPath, tableName)) db.createPartitionedTable(schema, tableName, partitionColumns);
}}
def createDimensionIfAbsent(db, dbPath, tableName, schema) {{
    if (!existsTable(dbPath, tableName)) db.createTable(schema, tableName);
}}
rawDaily = table(1:0, `trade_date`month`ts_code`open`high`low`close`pre_close`change`pct_chg`vol`amount`run_id`ingested_at,
    [DATE, MONTH, SYMBOL, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, SYMBOL, TIMESTAMP]);
createPartitionedIfAbsent(db, dbPath, `raw_tushare_daily, rawDaily, `month`ts_code);
rawAdj = table(1:0, `trade_date`month`ts_code`adj_factor`run_id`ingested_at,
    [DATE, MONTH, SYMBOL, DOUBLE, SYMBOL, TIMESTAMP]);
createPartitionedIfAbsent(db, dbPath, `raw_tushare_adj_factor, rawAdj, `month`ts_code);
rawBasic = table(1:0, `trade_date`month`ts_code`turnover_rate`turnover_rate_f`volume_ratio`pe`pe_ttm`pb`ps`ps_ttm`dv_ratio`dv_ttm`total_share`float_share`free_share`total_mv`circ_mv`run_id`ingested_at,
    [DATE, MONTH, SYMBOL, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, SYMBOL, TIMESTAMP]);
createPartitionedIfAbsent(db, dbPath, `raw_tushare_daily_basic, rawBasic, `month`ts_code);
rawSuspend = table(1:0, `trade_date`month`ts_code`suspend_timing`suspend_type`run_id`ingested_at,
    [DATE, MONTH, SYMBOL, SYMBOL, SYMBOL, SYMBOL, TIMESTAMP]);
createPartitionedIfAbsent(db, dbPath, `raw_tushare_suspend, rawSuspend, `month`ts_code);
rawIndustry = table(1:0, `ts_code`industry_code`industry_name`in_date`out_date`source`run_id`ingested_at,
    [SYMBOL, SYMBOL, SYMBOL, DATE, DATE, SYMBOL, SYMBOL, TIMESTAMP]);
createDimensionIfAbsent(db, dbPath, `raw_tushare_sw_l1_member, rawIndustry);
rawFina = table(1:0, `ts_code`ann_date`end_date`roe`run_id`ingested_at,
    [SYMBOL, DATE, DATE, DOUBLE, SYMBOL, TIMESTAMP]);
createDimensionIfAbsent(db, dbPath, `raw_tushare_fina_indicator, rawFina);
rawBalance = table(1:0, `ts_code`ann_date`end_date`total_hldr_eqy_exc_min_int`run_id`ingested_at,
    [SYMBOL, DATE, DATE, DOUBLE, SYMBOL, TIMESTAMP]);
createDimensionIfAbsent(db, dbPath, `raw_tushare_balancesheet, rawBalance);
rawCash = table(1:0, `ts_code`ann_date`end_date`c_pay_acq_const_fiolta`run_id`ingested_at,
    [SYMBOL, DATE, DATE, DOUBLE, SYMBOL, TIMESTAMP]);
createDimensionIfAbsent(db, dbPath, `raw_tushare_cashflow, rawCash);
rawIncome = table(1:0, `ts_code`ann_date`end_date`revenue`total_revenue`run_id`ingested_at,
    [SYMBOL, DATE, DATE, DOUBLE, DOUBLE, SYMBOL, TIMESTAMP]);
createDimensionIfAbsent(db, dbPath, `raw_tushare_income, rawIncome);
priceBar = table(1:0, `trade_date`month`ts_code`open`high`low`close`pre_close`vol`amount`adj_factor`qfq_factor`hfq_factor`hfq_anchor_date`qfq_open`qfq_high`qfq_low`qfq_close`qfq_pre_close`qfq_vol`hfq_open`hfq_high`hfq_low`hfq_close`hfq_pre_close`hfq_vol,
    [DATE, MONTH, SYMBOL, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DATE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE, DOUBLE]);
createPartitionedIfAbsent(db, dbPath, `core_market_bar_daily_adjusted, priceBar, `month`ts_code);
watermark = table(1:0, `data_domain`last_successful_trade_date`updated_at`run_id`message,
    [SYMBOL, DATE, TIMESTAMP, SYMBOL, STRING]);
createDimensionIfAbsent(db, dbPath, `etl_watermark, watermark);
runLog = table(1:0, `run_id`job_name`started_at`finished_at`status`message,
    [SYMBOL, SYMBOL, TIMESTAMP, TIMESTAMP, SYMBOL, STRING]);
createDimensionIfAbsent(db, dbPath, `etl_run_log, runLog);
qualityLog = table(1:0, `run_id`data_domain`check_name`passed`observed_value`expected_value`created_at,
    [SYMBOL, SYMBOL, SYMBOL, BOOL, STRING, STRING, TIMESTAMP]);
createDimensionIfAbsent(db, dbPath, `data_quality_log, qualityLog);
'''


class DolphinDBWriter:
    def __init__(self, session: Any, db_path: str):
        self.session = session
        self.db_path = db_path

    def ensure_schema(self) -> None:
        self.session.run(UPDATE_SCHEMA.format(db_path=self.db_path))

    def replace_date_window(self, table_name: str, frame: pd.DataFrame, expression: str, start_date: str, end_date: str) -> int:
        if frame.empty:
            raise ValueError(f"Refusing to replace {table_name} with an empty frame.")
        self.session.upload({"qsUpdateTmp": frame})
        self.session.run(
            f"""
            target = loadTable('{self.db_path}', `{table_name});
            delete from target where trade_date between {_date_literal(start_date)} : {_date_literal(end_date)};
            target.append!(select {expression} from qsUpdateTmp);
            undef(`qsUpdateTmp, VAR);
            """
        )
        return len(frame)

    def replace_report_periods(self, table_name: str, frame: pd.DataFrame, expression: str, periods: Iterable[str]) -> int:
        if frame.empty:
            return 0
        target = f"loadTable('{self.db_path}', `{table_name})"
        deletes = "\n".join(f"delete from {target} where end_date = {_date_literal(period)};" for period in sorted(set(periods)))
        self.session.upload({"qsUpdateTmp": frame})
        self.session.run(f"{deletes}\ntarget = {target}; target.append!(select {expression} from qsUpdateTmp); undef(`qsUpdateTmp, VAR);")
        return len(frame)

    def replace_industry_members(self, frame: pd.DataFrame) -> int:
        if frame.empty:
            raise ValueError("Refusing to replace SW membership with an empty frame.")
        self.session.upload({"qsUpdateTmp": frame})
        self.session.run(
            f"""
            target = loadTable('{self.db_path}', `raw_tushare_sw_l1_member);
            delete from target where source = `{SW_SOURCE};
            target.append!(select symbol(ts_code) as ts_code, symbol(industry_code) as industry_code,
                symbol(industry_name) as industry_name, date(in_date) as in_date, date(out_date) as out_date,
                symbol(source) as source, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at from qsUpdateTmp);
            undef(`qsUpdateTmp, VAR);
            """
        )
        return len(frame)
    def write_watermark(self, data_domain: str, date_value: str, run_id: str, message: str) -> None:
        frame = pd.DataFrame([{
            "data_domain": data_domain,
            "last_successful_trade_date": pd.Timestamp(date_value),
            "updated_at": pd.Timestamp.now(),
            "run_id": run_id,
            "message": message,
        }])
        self.session.upload({"qsWatermarkTmp": frame})
        self.session.run(
            f"""
            target = loadTable('{self.db_path}', `etl_watermark);
            delete from target where data_domain = `{data_domain};
            target.append!(select symbol(data_domain) as data_domain, date(last_successful_trade_date) as last_successful_trade_date,
                timestamp(updated_at) as updated_at, symbol(run_id) as run_id, message from qsWatermarkTmp);
            undef(`qsWatermarkTmp, VAR);
            """
        )

    def write_run_log(self, run_id: str, status: str, started_at: pd.Timestamp, message: str) -> None:
        frame = pd.DataFrame([{
            "run_id": run_id, "job_name": "tushare_eod_update", "started_at": started_at,
            "finished_at": pd.Timestamp.now(), "status": status, "message": message,
        }])
        self.session.upload({"qsRunLogTmp": frame})
        self.session.run(
            f"""
            target = loadTable('{self.db_path}', `etl_run_log);
            target.append!(select symbol(run_id) as run_id, symbol(job_name) as job_name, timestamp(started_at) as started_at,
                timestamp(finished_at) as finished_at, symbol(status) as status, message from qsRunLogTmp);
            undef(`qsRunLogTmp, VAR);
            """
        )

    def write_quality_checks(self, run_id: str, checks: list[dict[str, Any]]) -> None:
        if not checks:
            return
        frame = pd.DataFrame(checks)
        frame["run_id"] = run_id
        frame["created_at"] = pd.Timestamp.now()
        self.session.upload({"qsQualityTmp": frame})
        self.session.run(
            f"""
            target = loadTable('{self.db_path}', `data_quality_log);
            target.append!(select symbol(run_id) as run_id, symbol(data_domain) as data_domain,
                symbol(check_name) as check_name, passed, observed_value, expected_value,
                timestamp(created_at) as created_at from qsQualityTmp);
            undef(`qsQualityTmp, VAR);
            """
        )

    def load_watermark(self, data_domain: str) -> str | None:
        frame = self.session.run(f"t = loadTable('{self.db_path}', `etl_watermark); select last_successful_trade_date from t where data_domain = `{data_domain}")
        return None if frame is None or frame.empty else _date_string(frame.iloc[0, 0])

    def table_row_count(self, table_name: str, trade_date: str) -> int:
        frame = self.session.run(f"t = loadTable('{self.db_path}', `{table_name}); select count(*) as rows from t where trade_date = {_date_literal(trade_date)}")
        return int(frame.iloc[0]["rows"])

    def current_industry_mapping(self) -> pd.DataFrame:
        frame = self.session.run(f"t = loadTable('{self.db_path}', `core_industry_sw_l1_daily); lastDate = exec max(trade_date) from t; select ts_code, industry_name from t where trade_date = lastDate")
        return frame if frame is not None else pd.DataFrame(columns=["ts_code", "industry_name"])


def _append_metadata(frame: pd.DataFrame, run_id: str) -> pd.DataFrame:
    result = frame.copy()
    result["run_id"] = run_id
    result["ingested_at"] = pd.Timestamp.now()
    return result


def _prepare_market_frames(
    trade_date: str,
    source: dict[str, pd.DataFrame],
    anchor_factors: pd.DataFrame,
    run_id: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    daily = source["daily"].copy()
    adj = source["adj_factor"].copy()
    basic = source["daily_basic"].copy()
    suspension = source["suspend"].copy()
    required_daily = {"ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "vol", "amount"}
    missing_daily = required_daily.difference(daily.columns)
    if missing_daily:
        raise ValueError(f"Tushare daily missing columns: {sorted(missing_daily)}")
    if daily.duplicated(["ts_code", "trade_date"]).any():
        raise ValueError(f"Tushare daily has duplicate stock-date rows for {trade_date}.")
    daily["trade_date"] = _to_datetime(daily["trade_date"])
    adj["trade_date"] = _to_datetime(adj["trade_date"])
    basic["trade_date"] = _to_datetime(basic["trade_date"])
    if suspension.empty:
        suspension = pd.DataFrame(columns=["ts_code", "trade_date", "suspend_timing", "suspend_type"])
    else:
        suspension["trade_date"] = _to_datetime(suspension["trade_date"])

    anchor = anchor_factors[["ts_code", "adj_factor"]].rename(columns={"adj_factor": "anchor_adj_factor"})
    joined = daily.merge(adj[["ts_code", "adj_factor"]], on="ts_code", how="left").merge(anchor, on="ts_code", how="left")
    if joined["adj_factor"].isna().any():
        raise ValueError(f"Missing same-day adj_factor for {int(joined['adj_factor'].isna().sum())} market rows.")
    joined["anchor_adj_factor"] = joined["anchor_adj_factor"].fillna(joined["adj_factor"])
    joined["forward_factor"] = joined["adj_factor"] / joined["anchor_adj_factor"]
    core_bar = joined[["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "vol", "amount", "adj_factor"]].copy()
    for column in ["open", "high", "low", "close", "pre_close"]:
        core_bar[column] = (pd.to_numeric(core_bar[column], errors="coerce") * joined["forward_factor"]).round(2)
    core_bar["vol"] = (pd.to_numeric(core_bar["vol"], errors="coerce") / joined["forward_factor"]).round(2)
    core_bar["dret"] = ((core_bar["close"] / core_bar["pre_close"]) - 1.0).round(5)
    market = _market_name(core_bar["ts_code"])
    limits = pd.Series(market).map({"北交所": 0.30, "科创板": 0.20, "创业板": 0.20, "沪深主板": 0.10, "其他": 0.30}).to_numpy()
    core_bar["dret"] = np.where(core_bar["dret"].abs() > limits, np.sign(core_bar["dret"]) * limits, core_bar["dret"])

    core_status = core_bar[["trade_date", "ts_code"]].merge(suspension[["ts_code", "trade_date", "suspend_timing", "suspend_type"]], on=["ts_code", "trade_date"], how="left")
    core_status["market"] = market
    is_suspended = core_status["suspend_timing"].isna() & (core_status["suspend_type"] == "S")
    is_one_word_limit = (core_bar["open"] == core_bar["close"]) & (core_bar["high"] == core_bar["low"]) & (core_bar["open"] == core_bar["high"])
    core_status["is_trading"] = np.select([is_suspended, is_one_word_limit], [-1, -2], default=1)

    missing_valuation = set(MARKET_VALUATION_COLUMNS).difference(basic.columns)
    if missing_valuation:
        raise ValueError(f"Tushare daily_basic missing columns: {sorted(missing_valuation)}")
    core_valuation = core_bar[["trade_date", "ts_code"]].merge(basic[MARKET_VALUATION_COLUMNS], on=["trade_date", "ts_code"], how="left")
    raw_frames = {
        "raw_tushare_daily": _append_metadata(daily[["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "change", "pct_chg", "vol", "amount"]], run_id),
        "raw_tushare_adj_factor": _append_metadata(adj[["trade_date", "ts_code", "adj_factor"]], run_id),
        "raw_tushare_daily_basic": _append_metadata(basic[MARKET_VALUATION_COLUMNS], run_id),
        "raw_tushare_suspend": _append_metadata(suspension[["trade_date", "ts_code", "suspend_timing", "suspend_type"]], run_id),
    }
    core_frames = {
        "core_market_bar_daily": core_bar[MARKET_BAR_COLUMNS],
        "core_market_status_daily": core_status[MARKET_STATUS_COLUMNS],
        "core_market_valuation_daily": core_valuation[MARKET_VALUATION_COLUMNS],
    }
    return raw_frames, core_frames


def _quality_check(data_domain: str, check_name: str, passed: bool, observed: Any, expected: Any) -> dict[str, Any]:
    return {"data_domain": data_domain, "check_name": check_name, "passed": bool(passed), "observed_value": str(observed), "expected_value": str(expected)}

class TushareEodUpdater:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8848,
        user: str = "admin",
        password: str = "123456",
        db_path: str = DEFAULT_DB_PATH,
        token: str | None = None,
    ):
        self.client = TushareClient(token or load_tushare_token())
        self.session = connect(host, port, user, password)
        self.writer = DolphinDBWriter(self.session, db_path)
        self.writer.ensure_schema()

    def resolve_window(
        self,
        end_date: str | None = None,
        start_date: str | None = None,
        lookback_trading_days: int = 5,
        max_trading_days: int | None = None,
    ) -> UpdateWindow:
        requested_end = end_date or dt.date.today().strftime("%Y%m%d")
        latest_open = self.client.latest_open_date(requested_end)
        if start_date:
            candidate_start = start_date
        else:
            watermark = self.writer.load_watermark("market_daily")
            candidate_start = _date_string(pd.Timestamp(watermark or latest_open) - pd.Timedelta(days=31))
        dates = self.client.open_dates(candidate_start, latest_open)
        if lookback_trading_days and not start_date:
            dates = dates[-max(lookback_trading_days, 1):]
        if max_trading_days is not None:
            dates = dates[:max_trading_days]
        if not dates:
            raise RuntimeError("No trading days selected for update.")
        return UpdateWindow(dates[0], dates[-1], dates, latest_open)

    def update_market(self, window: UpdateWindow, refresh_industry: bool = False) -> UpdateReport:
        run_id = f"tushare_eod_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        started_at = pd.Timestamp.now()
        raw_parts: dict[str, list[pd.DataFrame]] = {}
        core_parts: dict[str, list[pd.DataFrame]] = {}
        checks: list[dict[str, Any]] = []
        try:
            anchor = self.client.request("adj_factor", trade_date=window.adjustment_anchor_date)
            if anchor.empty:
                raise RuntimeError(f"No adjustment anchor returned for {window.adjustment_anchor_date}.")
            anchor = anchor[["ts_code", "adj_factor"]].drop_duplicates("ts_code")
            previous_anchor_date = self.writer.load_watermark("market_price_anchor")
            if previous_anchor_date is None:
                previous_anchor_date = _latest_core_date_before(self.writer, window.start_date)
            rebase_scales = pd.DataFrame(columns=["ts_code", "scale"])
            if previous_anchor_date and previous_anchor_date != window.adjustment_anchor_date:
                previous_anchor = self.client.request("adj_factor", trade_date=previous_anchor_date)
                previous_anchor = previous_anchor[["ts_code", "adj_factor"]].rename(columns={"adj_factor": "previous_adj_factor"})
                comparison = previous_anchor.merge(anchor, on="ts_code", how="inner")
                comparison = comparison[(comparison["previous_adj_factor"] > 0) & (comparison["adj_factor"] > 0)]
                comparison["scale"] = comparison["previous_adj_factor"] / comparison["adj_factor"]
                rebase_scales = comparison.loc[~np.isclose(comparison["scale"], 1.0), ["ts_code", "scale"]]
            for trade_date in window.trade_dates:
                source = self.client.market_day(trade_date)
                raw_frames, core_frames = _prepare_market_frames(trade_date, source, anchor, run_id)
                for name, frame in raw_frames.items():
                    raw_parts.setdefault(name, []).append(frame)
                for name, frame in core_frames.items():
                    core_parts.setdefault(name, []).append(frame)
                daily_rows = len(source["daily"])
                coverage = source["daily"]["ts_code"].isin(source["adj_factor"]["ts_code"]).mean()
                checks.extend([
                    _quality_check("market_daily", f"{trade_date}_minimum_stock_count", daily_rows >= 4000, daily_rows, ">=4000"),
                    _quality_check("market_daily", f"{trade_date}_unique_stock_date", not source["daily"].duplicated(["ts_code", "trade_date"]).any(), 0, "0 duplicates"),
                    _quality_check("market_daily", f"{trade_date}_adj_coverage", coverage == 1.0, coverage, "1.0"),
                ])
            if not all(check["passed"] for check in checks):
                failed = [check["check_name"] for check in checks if not check["passed"]]
                raise RuntimeError(f"Data quality checks failed: {failed}")

            rebased_symbols = _rebase_core_market_prices(self.writer, rebase_scales)
            rebased_qfq_symbols = _rebase_qfq_prices(self.writer, rebase_scales)
            checks.append(_quality_check("market_daily", "forward_adjustment_rebase", True, rebased_symbols, "legacy symbols with changed adjustment anchor"))
            checks.append(_quality_check("market_daily", "qfq_adjustment_rebase", True, rebased_qfq_symbols, "adjusted-table symbols with changed adjustment anchor"))
            raw_rows = self._write_raw_market(raw_parts, window)
            core_rows = self._write_core_market(core_parts, window)
            core_rows["core_market_bar_daily_adjusted"] = _write_adjusted_prices(self.writer, raw_parts, anchor, window)
            if refresh_industry:
                raw_industry, core_industry = self.refresh_industry(window, run_id, core_parts["core_market_bar_daily"])
                raw_rows["raw_tushare_sw_l1_member"] = raw_industry
                core_rows["core_industry_sw_l1_daily"] = core_industry
            else:
                core_rows["core_industry_sw_l1_daily"] = self._carry_forward_industry(window, core_parts["core_market_bar_daily"])

            for table_name, expected in core_rows.items():
                if table_name == "core_industry_sw_l1_daily":
                    continue
                observed = self.writer.table_row_count(table_name, window.end_date)
                checks.append(_quality_check("market_daily", f"{table_name}_latest_row_count", observed >= min(expected, 4000), observed, f">={min(expected, 4000)}"))
            if not all(check["passed"] for check in checks):
                failed = [check["check_name"] for check in checks if not check["passed"]]
                raise RuntimeError(f"Post-write quality checks failed: {failed}")

            self.writer.write_watermark("market_daily", window.end_date, run_id, f"Updated {len(window.trade_dates)} open dates")
            self.writer.write_watermark("market_price_anchor", window.adjustment_anchor_date, run_id, "Forward-adjustment anchor")
            self.writer.write_quality_checks(run_id, checks)
            self.writer.write_run_log(run_id, "success", started_at, f"market dates={window.start_date}..{window.end_date}")
            return UpdateReport(run_id, window.trade_dates, raw_rows, core_rows, checks)
        except Exception as exc:
            self.writer.write_quality_checks(run_id, checks)
            self.writer.write_run_log(run_id, "failed", started_at, str(exc))
            raise

    def _write_raw_market(self, raw_parts: dict[str, list[pd.DataFrame]], window: UpdateWindow) -> dict[str, int]:
        expressions = {
            "raw_tushare_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, open, high, low, close, pre_close, change, pct_chg, vol, amount, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at",
            "raw_tushare_adj_factor": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, adj_factor, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at",
            "raw_tushare_daily_basic": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at",
            "raw_tushare_suspend": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, symbol(string(suspend_timing)) as suspend_timing, symbol(string(suspend_type)) as suspend_type, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at",
        }
        return {
            table: self.writer.replace_date_window(table, pd.concat(parts, ignore_index=True), expressions[table], window.start_date, window.end_date)
            for table, parts in raw_parts.items()
        }

    def _write_core_market(self, core_parts: dict[str, list[pd.DataFrame]], window: UpdateWindow) -> dict[str, int]:
        expressions = {
            "core_market_bar_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, open, high, low, close, pre_close, vol, amount, adj_factor, dret",
            "core_market_status_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, is_trading, symbol(string(suspend_timing)) as suspend_timing, symbol(string(suspend_type)) as suspend_type, symbol(market) as market",
            "core_market_valuation_daily": "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, turnover_rate, turnover_rate_f, volume_ratio, pe, pe_ttm, pb, ps, ps_ttm, dv_ratio, dv_ttm, total_share, float_share, free_share, total_mv, circ_mv",
        }
        return {
            table: self.writer.replace_date_window(table, pd.concat(parts, ignore_index=True), expressions[table], window.start_date, window.end_date)
            for table, parts in core_parts.items()
        }
    def refresh_industry(self, window: UpdateWindow, run_id: str, market_parts: list[pd.DataFrame]) -> tuple[int, int]:
        members = self.client.sw_l1_membership()
        members["in_date"] = _to_datetime(members["in_date"])
        members["out_date"] = _to_datetime(members["out_date"])
        members = _append_metadata(members, run_id)
        raw_rows = self.writer.replace_industry_members(members)
        industry = self._materialise_industry(window.trade_dates, pd.concat(market_parts, ignore_index=True), members)
        core_rows = self.writer.replace_date_window(
            "core_industry_sw_l1_daily",
            industry,
            "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, symbol(industry_name) as industry_name, symbol(source) as source",
            window.start_date,
            window.end_date,
        )
        self.writer.write_watermark("sw_l1_industry", window.end_date, run_id, f"Refreshed {raw_rows} membership rows")
        return raw_rows, core_rows

    @staticmethod
    def _materialise_industry(trade_dates: list[str], market: pd.DataFrame, members: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        for value in trade_dates:
            trade_date = pd.Timestamp(value)
            codes = market.loc[market["trade_date"] == trade_date, ["trade_date", "ts_code"]].copy()
            active = members[(members["in_date"] <= trade_date) & (members["out_date"].isna() | (members["out_date"] >= trade_date))]
            mapping = active.sort_values(["ts_code", "in_date"]).drop_duplicates("ts_code", keep="last")
            part = codes.merge(mapping[["ts_code", "industry_name"]], on="ts_code", how="left")
            part["industry_name"] = part["industry_name"].fillna("未分类")
            part["source"] = SW_SOURCE
            parts.append(part)
        return pd.concat(parts, ignore_index=True)

    def refresh_industry_history(self, start_date: str, end_date: str) -> dict[str, Any]:
        """Rebuild SW L1 daily mappings over an existing market-data range."""
        run_id = f"tushare_industry_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        started_at = pd.Timestamp.now()
        try:
            members = self.client.sw_l1_membership()
            members["in_date"] = _to_datetime(members["in_date"])
            members["out_date"] = _to_datetime(members["out_date"])
            members = _append_metadata(members, run_id)
            raw_rows = self.writer.replace_industry_members(members)
            total_rows = 0
            total_classified = 0
            for month in pd.period_range(start=start_date, end=end_date, freq="M"):
                month_start = month.start_time.strftime("%Y%m%d")
                month_end = month.end_time.strftime("%Y%m%d")
                market = self.session.run(
                    f"""
                    t = loadTable('{self.writer.db_path}', `core_market_bar_daily);
                    select trade_date, ts_code from t
                    where trade_date between {_date_literal(month_start)} : {_date_literal(month_end)}
                    """
                )
                if market.empty:
                    continue
                market["trade_date"] = pd.to_datetime(market["trade_date"])
                trade_dates = sorted(market["trade_date"].dt.strftime("%Y%m%d").unique().tolist())
                industry = self._materialise_industry(trade_dates, market, members)
                total_rows += self.writer.replace_date_window(
                    "core_industry_sw_l1_daily",
                    industry,
                    "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, symbol(industry_name) as industry_name, symbol(source) as source",
                    month_start,
                    month_end,
                )
                total_classified += int((industry["industry_name"] != "未分类").sum())
            coverage = total_classified / total_rows if total_rows else 0.0
            if total_rows == 0 or coverage < 0.95:
                raise RuntimeError(f"SW L1 historical mapping coverage is insufficient: {coverage:.2%}")
            self.writer.write_watermark("sw_l1_industry", end_date, run_id, f"Historical rebuild rows={total_rows}, coverage={coverage:.2%}")
            self.writer.write_run_log(run_id, "success", started_at, f"industry dates={start_date}..{end_date}; coverage={coverage:.2%}")
            return {"run_id": run_id, "raw_members": raw_rows, "daily_rows": total_rows, "classified_rows": total_classified, "coverage": coverage}
        except Exception as exc:
            self.writer.write_run_log(run_id, "failed", started_at, str(exc))
            raise
    def _carry_forward_industry(self, window: UpdateWindow, market_parts: list[pd.DataFrame]) -> int:
        market = pd.concat(market_parts, ignore_index=True)[["trade_date", "ts_code"]]
        mapping = self.writer.current_industry_mapping()
        if mapping.empty:
            market["industry_name"] = "未分类"
        else:
            market = market.merge(mapping, on="ts_code", how="left")
            market["industry_name"] = market["industry_name"].fillna("未分类")
        market["source"] = "SW_L1_CARRY_FORWARD"
        return self.writer.replace_date_window(
            "core_industry_sw_l1_daily",
            market,
            "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, symbol(industry_name) as industry_name, symbol(source) as source",
            window.start_date,
            window.end_date,
        )

    def update_financial_raw(self, reference_date: str | None = None, periods: int = 12) -> dict[str, int]:
        run_id = f"tushare_financial_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
        started_at = pd.Timestamp.now()
        reference = reference_date or dt.date.today().strftime("%Y%m%d")
        report_periods = _quarter_periods(reference, periods)
        collected: dict[str, list[pd.DataFrame]] = {"fina_indicator": [], "balancesheet": [], "cashflow": [], "income": []}
        try:
            for period in report_periods:
                for name, frame in self.client.financial_period(period).items():
                    if not frame.empty:
                        collected[name].append(frame)
            specs = {
                "fina_indicator": ("raw_tushare_fina_indicator", ["roe"], "ts_code, date(ann_date) as ann_date, date(end_date) as end_date, roe, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at"),
                "balancesheet": ("raw_tushare_balancesheet", ["total_hldr_eqy_exc_min_int"], "ts_code, date(ann_date) as ann_date, date(end_date) as end_date, total_hldr_eqy_exc_min_int, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at"),
                "cashflow": ("raw_tushare_cashflow", ["c_pay_acq_const_fiolta"], "ts_code, date(ann_date) as ann_date, date(end_date) as end_date, c_pay_acq_const_fiolta, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at"),
                "income": ("raw_tushare_income", ["revenue", "total_revenue"], "ts_code, date(ann_date) as ann_date, date(end_date) as end_date, revenue, total_revenue, symbol(run_id) as run_id, timestamp(ingested_at) as ingested_at"),
            }
            counts: dict[str, int] = {}
            for name, (table, values, expression) in specs.items():
                if not collected[name]:
                    raise RuntimeError(f"Tushare returned no {name} records for the selected periods.")
                normalised = _normalise_financial_frame(pd.concat(collected[name], ignore_index=True), values, run_id)
                counts[table] = self.writer.replace_report_periods(table, normalised, expression, report_periods)
            self.writer.write_watermark("financial_raw", max(report_periods), run_id, f"Refreshed {len(report_periods)} report periods")
            self.writer.write_run_log(run_id, "success", started_at, f"financial periods={','.join(report_periods)}")
            return counts
        except Exception as exc:
            self.writer.write_run_log(run_id, "failed", started_at, str(exc))
            raise


def _latest_core_date_before(writer: DolphinDBWriter, before_date: str) -> str | None:
    frame = writer.session.run(
        f"t=loadTable('{writer.db_path}', `core_market_bar_daily); select max(trade_date) as trade_date from t where trade_date < {_date_literal(before_date)}"
    )
    value = frame.iloc[0, 0] if frame is not None and not frame.empty else None
    return None if pd.isna(value) else _date_string(value)


def _rebase_core_market_prices(writer: DolphinDBWriter, scales: pd.DataFrame) -> int:
    """Rescale existing forward-adjusted OHLC when the adjustment anchor changes."""
    if scales.empty:
        return 0
    writer.session.upload({"qsRebaseTmp": scales[["ts_code", "scale"]]})
    writer.session.run(
        f"""
        target = loadTable('{writer.db_path}', `core_market_bar_daily);
        scaleTable = select symbol(ts_code) as ts_code, scale from qsRebaseTmp;
        update target set open=round(open*scale, 2), high=round(high*scale, 2),
            low=round(low*scale, 2), close=round(close*scale, 2),
            pre_close=round(pre_close*scale, 2), vol=round(vol/scale, 2)
        from ej(target, scaleTable, `ts_code);
        undef(`qsRebaseTmp, VAR);
        """
    )
    return len(scales)

def _quarter_periods(reference_date: str, count: int) -> list[str]:
    """Return report periods whose statutory disclosure window has completed."""
    reference = pd.Timestamp(reference_date)
    candidates: list[pd.Timestamp] = []
    years_to_scan = max(6, (count + 3) // 4 + 2)
    for year in range(reference.year, reference.year - years_to_scan, -1):
        for month_day, available_after in (
            ("0331", pd.Timestamp(f"{year}-05-01")),
            ("0630", pd.Timestamp(f"{year}-09-01")),
            ("0930", pd.Timestamp(f"{year}-11-01")),
            ("1231", pd.Timestamp(f"{year + 1}-05-01")),
        ):
            if reference >= available_after:
                candidates.append(pd.Timestamp(f"{year}{month_day}"))
    return [_date_string(value) for value in sorted(candidates, reverse=True)[:count]]


def _normalise_financial_frame(frame: pd.DataFrame, value_columns: list[str], run_id: str) -> pd.DataFrame:
    result = frame.copy()
    if "f_ann_date" in result.columns:
        result["ann_date"] = result["f_ann_date"].fillna(result.get("ann_date"))
    result["end_date"] = _to_datetime(result["end_date"])
    result["ann_date"] = _to_datetime(result["ann_date"])

    # Tushare occasionally omits disclosure dates. Use the same conservative
    # statutory deadlines as the legacy pipeline, never a date before end_date.
    missing = result["ann_date"].isna() & result["end_date"].notna()
    end_dates = result.loc[missing, "end_date"]
    inferred = pd.Series(pd.NaT, index=end_dates.index, dtype="datetime64[ns]")
    suffix = end_dates.dt.strftime("%m%d")
    inferred.loc[suffix == "0331"] = pd.to_datetime(end_dates.loc[suffix == "0331"].dt.year.astype(str) + "-04-30")
    inferred.loc[suffix == "0630"] = pd.to_datetime(end_dates.loc[suffix == "0630"].dt.year.astype(str) + "-08-31")
    inferred.loc[suffix == "0930"] = pd.to_datetime(end_dates.loc[suffix == "0930"].dt.year.astype(str) + "-10-31")
    december = suffix == "1231"
    inferred.loc[december] = pd.to_datetime((end_dates.loc[december].dt.year + 1).astype(str) + "-04-30")
    result.loc[missing, "ann_date"] = inferred

    result = result.dropna(subset=["ts_code", "ann_date", "end_date"])
    result = result.sort_values(["ts_code", "end_date", "ann_date"]).drop_duplicates(["ts_code", "end_date"], keep="last")
    return _append_metadata(result[["ts_code", "ann_date", "end_date", *value_columns]], run_id)

def _first_adjustment_factors(writer: DolphinDBWriter) -> pd.DataFrame:
    frame = writer.session.run(
        f"""
        t = loadTable('{writer.db_path}', `raw_tushare_adj_factor);
        firstDates = select ts_code, min(trade_date) as trade_date from t group by ts_code;
        select ts_code, adj_factor as first_adj_factor, trade_date as hfq_anchor_date
        from ej(firstDates, t, `ts_code`trade_date)
        """
    )
    return frame if frame is not None else pd.DataFrame(columns=["ts_code", "first_adj_factor", "hfq_anchor_date"])


def _build_adjusted_price_frame(
    daily: pd.DataFrame,
    adj: pd.DataFrame,
    latest_factors: pd.DataFrame,
    first_factors: pd.DataFrame,
) -> pd.DataFrame:
    prices = daily[["trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "vol", "amount"]].copy()
    prices["trade_date"] = _to_datetime(prices["trade_date"])
    factors = adj[["ts_code", "adj_factor"]].copy()
    latest = latest_factors[["ts_code", "adj_factor"]].rename(columns={"adj_factor": "latest_adj_factor"})
    result = prices.merge(factors, on="ts_code", how="left").merge(latest, on="ts_code", how="left").merge(first_factors, on="ts_code", how="left")
    if result["adj_factor"].isna().any():
        raise ValueError("Cannot build adjusted prices with missing same-day adjustment factors.")
    result["latest_adj_factor"] = result["latest_adj_factor"].fillna(result["adj_factor"])
    result["first_adj_factor"] = result["first_adj_factor"].fillna(result["adj_factor"])
    result["hfq_anchor_date"] = pd.to_datetime(result["hfq_anchor_date"].fillna(result["trade_date"]))
    result["qfq_factor"] = result["adj_factor"] / result["latest_adj_factor"]
    result["hfq_factor"] = result["adj_factor"] / result["first_adj_factor"]
    for field in ["open", "high", "low", "close", "pre_close"]:
        value = pd.to_numeric(result[field], errors="coerce")
        result[f"qfq_{field}"] = (value * result["qfq_factor"]).round(2)
        result[f"hfq_{field}"] = (value * result["hfq_factor"]).round(2)
    result["qfq_vol"] = (pd.to_numeric(result["vol"], errors="coerce") / result["qfq_factor"]).round(2)
    result["hfq_vol"] = (pd.to_numeric(result["vol"], errors="coerce") / result["hfq_factor"]).round(2)
    columns = [
        "trade_date", "ts_code", "open", "high", "low", "close", "pre_close", "vol", "amount", "adj_factor",
        "qfq_factor", "hfq_factor", "hfq_anchor_date", "qfq_open", "qfq_high", "qfq_low", "qfq_close", "qfq_pre_close", "qfq_vol",
        "hfq_open", "hfq_high", "hfq_low", "hfq_close", "hfq_pre_close", "hfq_vol",
    ]
    return result[columns]


def _write_adjusted_prices(
    writer: DolphinDBWriter,
    raw_parts: dict[str, list[pd.DataFrame]],
    latest_factors: pd.DataFrame,
    window: UpdateWindow,
) -> int:
    first_factors = _first_adjustment_factors(writer)
    frame = _build_adjusted_price_frame(
        pd.concat(raw_parts["raw_tushare_daily"], ignore_index=True),
        pd.concat(raw_parts["raw_tushare_adj_factor"], ignore_index=True),
        latest_factors,
        first_factors,
    )
    expression = "date(trade_date) as trade_date, month(date(trade_date)) as month, symbol(ts_code) as ts_code, open, high, low, close, pre_close, vol, amount, adj_factor, qfq_factor, hfq_factor, date(hfq_anchor_date) as hfq_anchor_date, qfq_open, qfq_high, qfq_low, qfq_close, qfq_pre_close, qfq_vol, hfq_open, hfq_high, hfq_low, hfq_close, hfq_pre_close, hfq_vol"
    return writer.replace_date_window("core_market_bar_daily_adjusted", frame, expression, window.start_date, window.end_date)


def _rebase_qfq_prices(writer: DolphinDBWriter, scales: pd.DataFrame) -> int:
    if scales.empty:
        return 0
    writer.session.upload({"qsQfqRebaseTmp": scales[["ts_code", "scale"]]})
    writer.session.run(
        f"""
        target = loadTable('{writer.db_path}', `core_market_bar_daily_adjusted);
        scaleTable = select symbol(ts_code) as ts_code, scale from qsQfqRebaseTmp;
        update target set qfq_factor=qfq_factor*scale, qfq_open=round(qfq_open*scale, 2),
            qfq_high=round(qfq_high*scale, 2), qfq_low=round(qfq_low*scale, 2),
            qfq_close=round(qfq_close*scale, 2), qfq_pre_close=round(qfq_pre_close*scale, 2),
            qfq_vol=round(qfq_vol/scale, 2)
        from ej(target, scaleTable, `ts_code);
        undef(`qsQfqRebaseTmp, VAR);
        """
    )
    return len(scales)