from __future__ import annotations

import pandas as pd

from data_pipeline.dolphindb_sync import DEFAULT_DB_PATH, connect


def _date_literal(yyyymmdd: str) -> str:
    value = pd.to_datetime(str(yyyymmdd), format="%Y%m%d")
    return value.strftime("%Y.%m.%d")


def load_factor_panel_from_dolphindb(
    factor_name: str,
    factor_version: str = "v1",
    start_date: str = "20190101",
    end_date: str = "20251231",
    host: str = "127.0.0.1",
    port: int = 8848,
    user: str = "admin",
    password: str = "123456",
    db_path: str = DEFAULT_DB_PATH,
    include_raw_factor: bool = True,
) -> pd.DataFrame:
    session = connect(host, port, user, password)
    start = _date_literal(start_date)
    end = _date_literal(end_date)

    raw_expr = ", f.raw_value as raw_factor" if include_raw_factor else ""
    script = f"""
        f = loadTable('{db_path}', `factor_daily)
        bar = loadTable('{db_path}', `core_market_bar_daily)
        val = loadTable('{db_path}', `core_market_valuation_daily)
        ind = loadTable('{db_path}', `core_industry_sw_l1_daily)
        select
            f.ts_code as ts_code,
            f.trade_date as trade_date,
            bar.open as open,
            bar.high as high,
            bar.low as low,
            bar.close as close,
            bar.pre_close as pre_close,
            val.circ_mv as circ_mv,
            ind.industry_name as industry_name,
            f.factor_value as {factor_name}{raw_expr}
        from f
        left join bar on f.trade_date = bar.trade_date and f.ts_code = bar.ts_code
        left join val on f.trade_date = val.trade_date and f.ts_code = val.ts_code
        left join ind on f.trade_date = ind.trade_date and f.ts_code = ind.ts_code
        where f.factor_name = `{factor_name},
              f.factor_version = `{factor_version},
              f.trade_date between {start} : {end}
        order by f.trade_date, f.ts_code
    """
    df = session.run(script)
    if df.empty:
        raise ValueError(
            f"No factor rows found in DolphinDB for {factor_name}/{factor_version} "
            f"between {start_date} and {end_date}."
        )
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.strftime("%Y%m%d")
    return df
