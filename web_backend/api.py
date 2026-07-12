"""HTTP API for the QuantSystem factor research workbench."""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_DIR = PROJECT_ROOT / "功能模块"
FRONTEND_DIR = PROJECT_ROOT / "web_frontend"

for path in (PROJECT_ROOT, MODULE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from data_pipeline.dolphindb_analysis_loader import load_ic_from_dolphindb, load_quantile_from_dolphindb  # noqa: E402
from data_pipeline.dolphindb_factor_loader import load_factor_panel_from_dolphindb  # noqa: E402
from data_pipeline.dolphindb_sync import DEFAULT_DB_PATH, connect  # noqa: E402
from ic_drawdown import calculate_cumulative_ic_drawdown  # noqa: E402
from ValidationTest import IC_calculate, data_preprocessing as validation_preprocessing, factor_cumuret_rank, ic_ttest_sample  # noqa: E402
import QuantileSpreadTest as qst  # noqa: E402

DB_HOST = "127.0.0.1"
DB_PORT = 8848
DB_USER = "admin"
DB_PASSWORD = "123456"

TABLES = [
    "core_market_bar_daily",
    "core_market_status_daily",
    "core_market_valuation_daily",
    "core_industry_sw_l1_daily",
    "feature_financial_daily",
    "factor_daily",
    "market_factor_daily",
    "macro_rate_daily",
]

# Formula text is kept next to the API contract so the dashboard never has to
# infer mathematics from a factor name. Operators such as TsRank and DecayLinear
# match the implementations in Factor_Calculate.
FACTOR_FORMULAS = {
    "alpha_01": r"\operatorname{rank}(\operatorname{TsMax}_{5}(\operatorname{SignedPower}(x,2)))-0.5",
    "alpha_02": r"-\operatorname{Corr}_{6}(\operatorname{rank}(\Delta_2\log(\operatorname{Vol})),\operatorname{rank}((C-O)/O))",
    "alpha_03": r"-\operatorname{Corr}_{10}(\operatorname{rank}(O),\operatorname{rank}(\operatorname{Vol}))",
    "alpha_04": r"-\operatorname{TsRank}_{9}(\operatorname{rank}(L))",
    "alpha_05": r"\operatorname{rank}(O)\times(-|\operatorname{rank}(C)|)",
    "alpha_06": r"-\operatorname{Corr}_{10}(O,\operatorname{Vol})",
    "alpha_07": r"\begin{cases}-\operatorname{TsRank}_{60}(|\Delta_7 C|)\operatorname{sign}(\Delta_7 C),&\operatorname{Vol}>\operatorname{Adv}_{20}\\-1,&\text{otherwise}\end{cases}",
    "alpha_08": r"-\operatorname{rank}(\operatorname{Sum}_5(O)\operatorname{Sum}_5(R)-\operatorname{Delay}_{10}(\operatorname{Sum}_5(O)\operatorname{Sum}_5(R)))",
    "alpha_09": r"\begin{cases}\Delta C,&\operatorname{TsMin}_5(\Delta C)>0\\\Delta C,&\operatorname{TsMax}_5(\Delta C)<0\\-\Delta C,&\text{otherwise}\end{cases}",
    "alpha_11": r"(\operatorname{rank}(\operatorname{TsMax}_3(VWAP-C))+\operatorname{rank}(\operatorname{TsMin}_3(VWAP-C)))\operatorname{rank}(\Delta_3\operatorname{Vol})",
    "alpha_12": r"\operatorname{sign}(\Delta_1\operatorname{Vol})(-\Delta_1 C)",
    "alpha_13": r"-\operatorname{rank}(\operatorname{Cov}_5(\operatorname{rank}(C),\operatorname{rank}(\operatorname{Vol})))",
    "alpha_14": r"-\operatorname{rank}(\Delta_3 R)\times\operatorname{Corr}_{10}(O,\operatorname{Vol})",
    "alpha_15": r"-\operatorname{Sum}_3(\operatorname{rank}(\operatorname{Corr}_3(\operatorname{rank}(H),\operatorname{rank}(\operatorname{Vol}))))",
    "alpha_16": r"-\operatorname{rank}(\operatorname{Cov}_5(\operatorname{rank}(H),\operatorname{rank}(\operatorname{Vol})))",
    "alpha_17": r"\operatorname{TsRank}_{10}(C)\operatorname{rank}(\Delta_2 C)\operatorname{rank}(\operatorname{TsRank}_5(\operatorname{Adv}_{20}))",
    "alpha_18": r"-\operatorname{rank}(\operatorname{Std}_5(|C-O|)+(C-O)+\operatorname{Corr}_{10}(C,O))",
    "alpha_23": r"\begin{cases}-\Delta_2 H,&\operatorname{Mean}_{20}(H)<H\\-1,&\text{otherwise}\end{cases}",
    "alpha_25": r"\operatorname{rank}((-R)\operatorname{Adv}_{20}\operatorname{VWAP}(H-C))",
    "alpha_26": r"-\operatorname{TsMax}_{3}(\operatorname{Corr}_{5}(\operatorname{TsRank}_5(\operatorname{Vol}),\operatorname{TsRank}_5(H)))",
    "alpha_27": r"\operatorname{rank}(\operatorname{Mean}_{6}(\operatorname{Corr}_{2}(\operatorname{rank}(\operatorname{Vol}),\operatorname{rank}(\operatorname{VWAP}))))",
    "alpha_28": r"\operatorname{rank}(\operatorname{Corr}_{5}(\operatorname{Adv}_{20},L)+(H+L)/2-C)",
    "alpha_30": r"(\operatorname{sign}(\Delta_1 C)+\operatorname{sign}(\Delta_1 C_{-1})+\operatorname{sign}(\Delta_1 C_{-2}))\frac{\operatorname{Sum}_5(\operatorname{Vol})}{\operatorname{Sum}_{20}(\operatorname{Vol})}",
    "alpha_32": r"\operatorname{Scale}(\operatorname{Mean}_7(C)-C)+20\operatorname{Scale}(\operatorname{Corr}_{252}(VWAP,\operatorname{Delay}_5(C)))",
    "alpha_33": r"\operatorname{rank}(-[1-O/C])",
    "alpha_34": r"\operatorname{rank}(1-\operatorname{rank}(\operatorname{Std}_2(R)/\operatorname{Std}_5(R))+1-\operatorname{rank}(\Delta_1 C))",
    "alpha_35": r"\operatorname{TsRank}_{32}(\operatorname{Vol})[1-\operatorname{TsRank}_{16}(C+H-L)][1-\operatorname{TsRank}_{32}(R)]",
    "alpha_37": r"\operatorname{rank}(\operatorname{Corr}_{200}(\operatorname{Delay}_1(O-C),C))+\operatorname{rank}(O-C)",
    "alpha_45": r"-\operatorname{rank}(\operatorname{MidTermMomentum})\times\operatorname{Corr}(\operatorname{Price},\operatorname{Volume})\times\operatorname{TrendRank}",
    "alpha_49": r"\begin{cases}-(\operatorname{Delay}_{20}(C)-\operatorname{Delay}_{10}(C)),&\operatorname{Delay}_{10}(C)-C< -0.1\\-1,&\text{otherwise}\end{cases}",
    "alpha_50": r"-\operatorname{TsMax}_{5}(\operatorname{rank}(\operatorname{Corr}_5(\operatorname{rank}(\operatorname{Vol}),\operatorname{rank}(VWAP))))",
    "alpha_57": r"-\frac{C-VWAP}{\operatorname{DecayLinear}_{2}(\operatorname{rank}(\operatorname{TsArgMax}_{21}(C)))}",
    "alpha_60": r"2\operatorname{Scale}(\operatorname{rank}(\operatorname{Vol}\times((C-L)-(H-L))/(H-L)))-\operatorname{Scale}(\operatorname{rank}(\operatorname{TsRank}_{10}(C)))",
    "alpha_61": r"2\operatorname{Scale}(\operatorname{rank}(\operatorname{Vol}\times((C-L)-(H-L))/(H-L)))-\operatorname{Scale}(\operatorname{rank}(\operatorname{TsRank}_{10}(C)))",
}
app = FastAPI(title="Quant Factor Research System", version="0.2.0")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


def _session():
    return connect(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD)


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, pd.Period):
        return str(value)
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).strftime("%Y%m%d")
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if pd.isna(value):
        return None
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = pd.to_datetime(out[col]).dt.strftime("%Y%m%d")
    return [{str(key): _clean(value) for key, value in row.items()} for row in out.to_dict("records")]


def _clean_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _clean(value) for key, value in values.items()}


def _effective_window(df: pd.DataFrame, start_date: str, end_date: str) -> tuple[str, str]:
    dates = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d")
    start = max(pd.to_datetime(start_date, format="%Y%m%d"), dates.min())
    end = min(pd.to_datetime(end_date, format="%Y%m%d"), dates.max())
    if start > end:
        raise ValueError(f"Window {start_date}-{end_date} has no overlap with loaded data.")
    return start.strftime("%Y%m%d"), end.strftime("%Y%m%d")


def _load_panel(factor_name: str, factor_version: str, start_date: str, end_date: str) -> pd.DataFrame:
    return load_factor_panel_from_dolphindb(
        factor_name=factor_name,
        factor_version=factor_version,
        start_date=start_date,
        end_date=end_date,
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
    )


def _industry_quantile_results(
    processed: pd.DataFrame, factor_name: str, layers: int
) -> dict[str, Any]:
    columns = ["trade_date", "ts_code", factor_name, "holding_lndret", "circ_mv", "industry_name"]
    industry_input = processed[columns].dropna(subset=[factor_name, "industry_name"])
    industry_input = industry_input.groupby(["trade_date", "industry_name"], group_keys=False).filter(
        lambda group: len(group) >= layers
    )
    if industry_input.empty:
        raise ValueError("No industry-date group has enough securities for the requested number of layers.")

    layered = (
        industry_input.groupby(["trade_date", "industry_name"], group_keys=False)
        .apply(lambda group: qst.process_group_by_date(group, layers, factor_name))
        .reset_index(drop=True)
    )
    series: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for industry_name, industry_df in layered.groupby("industry_name", sort=True):
        result, _, cumulative = qst.spread_ret_cumsum_calculate(industry_df, layers, "mean_lndret", "_ew")
        result.insert(0, "industry_name", industry_name)
        series.extend(_records(result))
        summary.append(
            {
                "industry_name": industry_name,
                "observations": len(industry_df),
                "ls_final": _clean(cumulative.get("L-S")),
            }
        )
    return {"enabled": True, "series": series, "summary": summary}


def _ic_result_from_daily(daily: pd.DataFrame, ic_ma_window: int, holding_period: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    result = daily.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"].astype(str), format="%Y%m%d")
    result = result.sort_values("trade_date").reset_index(drop=True)
    rank_ic = result["Rank_IC"]
    drawdown_result = calculate_cumulative_ic_drawdown(rank_ic)
    cumulative = drawdown_result["cumulative_ic"]
    max_dd = drawdown_result["max_drawdown"]
    max_dd_index = drawdown_result["max_drawdown_date"]
    max_dd_date = result.loc[max_dd_index, "trade_date"] if pd.notna(max_dd_index) else pd.NaT
    max_dd_ratio = drawdown_result["max_drawdown_ratio"]
    max_dd_peak_index = drawdown_result["peak_date"]
    max_dd_recovery_index = drawdown_result["recovery_date"]
    max_dd_peak_date = result.loc[max_dd_peak_index, "trade_date"] if pd.notna(max_dd_peak_index) and max_dd_peak_index in result.index else pd.NaT
    max_dd_recovery_date = result.loc[max_dd_recovery_index, "trade_date"] if pd.notna(max_dd_recovery_index) and max_dd_recovery_index in result.index else pd.NaT
    max_dd_recovery_days = drawdown_result["recovery_days"]
    max_dd_recovery_status = drawdown_result["recovery_status"]
    ic_std = rank_ic.std()
    icir = rank_ic.mean() / ic_std if ic_std else np.nan
    result["Rank_IC_Std"] = ic_std
    result["Cumulative_IC"] = cumulative.reindex(result.index).ffill()
    result["Cumulative_IC_MaxDD"] = max_dd
    result["MaxDD_Occur_Date"] = max_dd_date
    result["Cumulative_IC_MaxDD_Ratio"] = max_dd_ratio
    result["MaxDD_Peak_Date"] = max_dd_peak_date
    result["MaxDD_Recovery_Date"] = max_dd_recovery_date
    result["MaxDD_Recovery_Days"] = max_dd_recovery_days
    result["MaxDD_Recovery_Status"] = max_dd_recovery_status
    result["IC_MA"] = rank_ic.rolling(window=ic_ma_window, min_periods=max(1, int(holding_period * 0.6))).mean()
    result["ICIR"] = icir
    result["IC_Win_Rate"] = (rank_ic > 0).mean()
    return result, ic_ttest_sample(rank_ic)


def _quantile_frames_from_daily(daily: pd.DataFrame, layers: int) -> tuple[pd.DataFrame, pd.Series, dict[Any, float], pd.DataFrame, pd.Series, dict[Any, float]]:
    def build(value_column: str, suffix: str) -> tuple[pd.DataFrame, pd.Series, dict[Any, float]]:
        frame = daily.pivot(index="trade_date", columns="quantile", values=value_column).sort_index().reset_index()
        missing = [index for index in range(layers) if index not in frame.columns]
        if missing:
            raise ValueError(f"DolphinDB aggregation is missing quantile groups: {missing}")
        spread = frame[0] - frame[layers - 1]
        frame["L-S"] = spread
        cumulative: dict[Any, float] = {}
        for index in range(layers):
            column = f"sum_ret_{index}{suffix}"
            frame[column] = frame[index].cumsum()
            cumulative[index] = float(frame[column].iloc[-1])
        frame[f"sum_ret_L-S{suffix}"] = spread.cumsum()
        cumulative["L-S"] = float(frame[f"sum_ret_L-S{suffix}"].iloc[-1])
        return frame, spread, cumulative

    ew, spread_ew, cum_ew = build("mean_lndret", "_ew")
    vw, spread_vw, cum_vw = build("mean_lndret_vw", "_vw")
    return ew, spread_ew, cum_ew, vw, spread_vw, cum_vw

@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health():
    try:
        session = _session()
        session.run("1+1")
    except Exception as exc:  # pragma: no cover - environment dependent
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ok", "database": DEFAULT_DB_PATH}


@app.get("/api/tables/status")
def table_status():
    session = _session()
    rows = []
    for table in TABLES:
        try:
            df = session.run(
                f"""
                t = loadTable('{DEFAULT_DB_PATH}', `{table})
                select count(*) as rows, min(trade_date) as start_date, max(trade_date) as end_date from t
                """
            )
            record = _records(df)[0]
            record["table"] = table
            rows.append(record)
        except Exception as exc:
            rows.append({"table": table, "error": str(exc)})
    return {"tables": rows}


@app.get("/api/factors")
def factors():
    session = _session()
    df = session.run(
        f"""
        f = loadTable('{DEFAULT_DB_PATH}', `factor_daily)
        select count(*) as rows,
               min(trade_date) as start_date,
               max(trade_date) as end_date,
               min(created_at) as first_created_at,
               max(created_at) as last_created_at
        from f
        group by factor_name, factor_version
        order by factor_name, factor_version
        """
    )
    records = _records(df)
    for record in records:
        formula = FACTOR_FORMULAS.get(record.get("factor_name"))
        record["formula_latex"] = formula or r"\operatorname{DefinedInSource}\left(" + str(record.get("factor_name")) + r"\right)"
        record["formula_source"] = "Factor_Calculate"
        record["formula_verified"] = formula is not None
    return {"factors": records}


@app.get("/api/market/coverage")
def market_coverage(
    start_date: str = Query("20190101", pattern=r"^\d{8}$"),
    end_date: str = Query("20190331", pattern=r"^\d{8}$"),
):
    session = _session()
    start = pd.to_datetime(start_date, format="%Y%m%d").strftime("%Y.%m.%d")
    end = pd.to_datetime(end_date, format="%Y%m%d").strftime("%Y.%m.%d")
    df = session.run(
        f"""
        bar = loadTable('{DEFAULT_DB_PATH}', `core_market_bar_daily)
        select count(*) as bars
        from bar
        where trade_date between {start} : {end}
        group by trade_date
        order by trade_date
        """
    )
    return {"start_date": start_date, "end_date": end_date, "series": _records(df)}


@app.get("/api/factors/{factor_name}/ic")
def factor_ic(
    factor_name: str,
    factor_version: str = "v1",
    start_date: str = Query("20190101", pattern=r"^\d{8}$"),
    end_date: str = Query("20190331", pattern=r"^\d{8}$"),
    holding_period: int = Query(5, ge=1, le=60),
    ic_ma_window: int = Query(20, ge=1, le=252),
    analysis_engine: str = Query("dolphindb"),
):
    if analysis_engine not in {"dolphindb", "python"}:
        raise HTTPException(status_code=422, detail="analysis_engine must be dolphindb or python")
    try:
        if analysis_engine == "dolphindb":
            daily, ic_decay = load_ic_from_dolphindb(
                factor_name, factor_version, start_date, end_date, holding_period,
                DB_HOST, DB_PORT, DB_USER, DB_PASSWORD,
            )
            result_df, ttest_result = _ic_result_from_daily(daily, ic_ma_window, holding_period)
            loaded_rows = None
            rows_after_preprocess = None
            effective_start = daily["trade_date"].min()
            effective_end = daily["trade_date"].max()
        else:
            panel = _load_panel(factor_name, factor_version, start_date, end_date)
            effective_start, effective_end = _effective_window(panel, start_date, end_date)
            prepared = validation_preprocessing(panel, effective_start, effective_end)
            ranked = factor_cumuret_rank(prepared, factor_name, holding_period)
            result_df, ic_decay, ttest_result = IC_calculate(ranked, ic_ma_window, holding_period)
            loaded_rows = len(panel)
            rows_after_preprocess = len(ranked)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    result_df["trade_date"] = pd.to_datetime(result_df["trade_date"])
    monthly = result_df.groupby(result_df["trade_date"].dt.to_period("M"))["Rank_IC"].agg(rank_ic_mean="mean", rank_ic_std="std").reset_index(names="year_month")
    monthly["icir"] = monthly["rank_ic_mean"] / monthly["rank_ic_std"]
    yearly = result_df.groupby(result_df["trade_date"].dt.to_period("Y"))["Rank_IC"].agg(rank_ic_mean="mean", rank_ic_std="std").reset_index(names="year")
    yearly["icir"] = yearly["rank_ic_mean"] / yearly["rank_ic_std"]
    series_cols = ["trade_date", "Rank_IC", "Cumulative_IC", "IC_MA"]
    return {
        "factor": factor_name, "version": factor_version, "analysis_engine": analysis_engine,
        "requested_window": {"start_date": start_date, "end_date": end_date},
        "effective_window": {"start_date": _clean(effective_start), "end_date": _clean(effective_end)},
        "summary": {
            "loaded_rows": loaded_rows, "rows_after_preprocess": rows_after_preprocess,
            "rank_ic_mean": _clean(result_df["Rank_IC"].mean()), "rank_ic_std": _clean(result_df["Rank_IC_Std"].iloc[0]),
            "icir": _clean(result_df["ICIR"].iloc[0]), "ic_win_rate": _clean(result_df["IC_Win_Rate"].iloc[0]),
            "cumulative_ic_final": _clean(result_df["Cumulative_IC"].iloc[-1]),
            "cumulative_ic_max_drawdown": _clean(result_df["Cumulative_IC_MaxDD"].iloc[0]),
            "cumulative_ic_max_drawdown_ratio": _clean(result_df["Cumulative_IC_MaxDD_Ratio"].iloc[0]),
            "max_drawdown_date": _clean(result_df["MaxDD_Occur_Date"].iloc[0]),
            "max_drawdown_peak_date": _clean(result_df["MaxDD_Peak_Date"].iloc[0]),
            "max_drawdown_recovery_date": _clean(result_df["MaxDD_Recovery_Date"].iloc[0]),
            "max_drawdown_recovery_days": _clean(result_df["MaxDD_Recovery_Days"].iloc[0]),
            "max_drawdown_recovery_status": result_df["MaxDD_Recovery_Status"].iloc[0],
            "holding_period": holding_period, "ic_ma_window": ic_ma_window,
        },
        "newey_west": _clean_mapping(ttest_result), "series": _records(result_df[series_cols]),
        "ic_decay": [{"period": index + 1, "rank_ic_mean": _clean(value)} for index, value in enumerate(ic_decay)],
        "monthly": _records(monthly), "yearly": _records(yearly),
    }

@app.get("/api/factors/{factor_name}/quantile")
def factor_quantile(
    factor_name: str,
    factor_version: str = "v1",
    start_date: str = Query("20190101", pattern=r"^\d{8}$"),
    end_date: str = Query("20190331", pattern=r"^\d{8}$"),
    layers: int = Query(10, ge=2, le=20),
    holding_period: int = Query(5, ge=1, le=60),
    industry_grouping: bool = False,
    analysis_engine: str = Query("dolphindb"),
):
    if analysis_engine not in {"dolphindb", "python"}:
        raise HTTPException(status_code=422, detail="analysis_engine must be dolphindb or python")
    try:
        use_dolphindb = analysis_engine == "dolphindb" and not industry_grouping
        if use_dolphindb:
            daily, memberships = load_quantile_from_dolphindb(
                factor_name, factor_version, start_date, end_date, layers, holding_period,
                DB_HOST, DB_PORT, DB_USER, DB_PASSWORD,
            )
            df_ew, spread_ew, cum_ew, df_vw, spread_vw, cum_vw = _quantile_frames_from_daily(daily, layers)
            turnover_df, avg_ew, std_ew, ann_ew, avg_vw, std_vw, ann_vw, group_turnovers = qst.calculate_turnover_rate(
                memberships, layers, holding_period, already_rebalanced=True
            )
            loaded_rows = None
            rows_after_preprocess = None
            rows_after_layering = None
            effective_start = daily["trade_date"].min()
            effective_end = daily["trade_date"].max()
            industry = {"enabled": False, "series": [], "summary": []}
        else:
            panel = _load_panel(factor_name, factor_version, start_date, end_date)
            effective_start, effective_end = _effective_window(panel, start_date, end_date)
            processed = qst.data_preprocessing(panel, effective_start, effective_end, holding_period).dropna(subset=[factor_name])
            layer_cols = ["trade_date", "ts_code", factor_name, "holding_lndret", "circ_mv"]
            layered = processed[layer_cols].groupby("trade_date", group_keys=False).apply(lambda group: qst.process_group_by_date(group, layers, factor_name)).reset_index(drop=True)
            df_ew, spread_ew, cum_ew = qst.spread_ret_cumsum_calculate(layered, layers, "mean_lndret", "_ew")
            df_vw, spread_vw, cum_vw = qst.spread_ret_cumsum_calculate(layered, layers, "mean_lndret_vw", "_vw")
            turnover_df, avg_ew, std_ew, ann_ew, avg_vw, std_vw, ann_vw, group_turnovers = qst.calculate_turnover_rate(layered, layers, holding_period)
            loaded_rows = len(panel)
            rows_after_preprocess = len(processed)
            rows_after_layering = len(layered)
            industry = _industry_quantile_results(processed, factor_name, layers) if industry_grouping else {"enabled": False, "series": [], "summary": []}
        nw_ew = qst.t_test_spread_ret(spread_ew, run_label="equal-weight L-S")
        nw_vw = qst.t_test_spread_ret(spread_vw, run_label="value-weight L-S")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    ew_cols = ["trade_date"] + [f"sum_ret_{index}_ew" for index in range(layers)] + ["sum_ret_L-S_ew"]
    vw_cols = ["trade_date"] + [f"sum_ret_{index}_vw" for index in range(layers)] + ["sum_ret_L-S_vw"]
    turnover = turnover_df.rename(columns={"date": "trade_date"})
    turnover_cols = [column for column in ["trade_date", "turnover_ls_ew", "turnover_ls_vw"] if column in turnover.columns]
    return {
        "factor": factor_name, "version": factor_version,
        "analysis_engine": "dolphindb" if use_dolphindb else "python",
        "requested_window": {"start_date": start_date, "end_date": end_date},
        "effective_window": {"start_date": _clean(effective_start), "end_date": _clean(effective_end)},
        "summary": {
            "loaded_rows": loaded_rows, "rows_after_preprocess": rows_after_preprocess, "rows_after_layering": rows_after_layering,
            "layers": layers, "holding_period": holding_period,
            "ls_ew_final": _clean(cum_ew.get("L-S")), "ls_vw_final": _clean(cum_vw.get("L-S")),
            "avg_turnover_ew": _clean(avg_ew), "avg_turnover_vw": _clean(avg_vw),
            "ann_turnover_ew": _clean(ann_ew), "ann_turnover_vw": _clean(ann_vw),
        },
        "newey_west": {"equal_weight": _clean_mapping(nw_ew), "value_weight": _clean_mapping(nw_vw)},
        "equal_weight": _records(df_ew[[column for column in ew_cols if column in df_ew.columns]]),
        "value_weight": _records(df_vw[[column for column in vw_cols if column in df_vw.columns]]),
        "turnover": _records(turnover[turnover_cols]) if turnover_cols else [],
        "layer_turnover": [{"layer": int(key), "turnover": _clean(value)} for key, value in group_turnovers.items()],
        "final_cumulative": {"equal_weight": {str(key): _clean(value) for key, value in cum_ew.items()}, "value_weight": {str(key): _clean(value) for key, value in cum_vw.items()}},
        "industry": industry,
    }