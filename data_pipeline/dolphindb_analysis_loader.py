"""Server-side DolphinDB aggregations for interactive factor analysis."""
from __future__ import annotations

import pandas as pd

from data_pipeline.dolphindb_factor_loader import _date_literal
from data_pipeline.dolphindb_sync import DEFAULT_DB_PATH, connect


def _session(host: str, port: int, user: str, password: str):
    return connect(host, port, user, password)


def _factor_base_script(
    factor_name: str,
    factor_version: str,
    start_date: str,
    end_date: str,
    include_mv: bool = False,
    db_path: str = DEFAULT_DB_PATH,
) -> str:
    start = _date_literal(start_date)
    end = _date_literal(end_date)
    valuation = f"val = loadTable('{db_path}', `core_market_valuation_daily)" if include_mv else ""
    valuation_join = "left join val on f.trade_date = val.trade_date and f.ts_code = val.ts_code" if include_mv else ""
    circ_mv = ", val.circ_mv as circ_mv" if include_mv else ""
    return f"""
        f = loadTable('{db_path}', `factor_daily)
        bar = loadTable('{db_path}', `core_market_bar_daily)
        {valuation}
        base = select f.trade_date as trade_date,
                      f.ts_code as ts_code,
                      f.factor_value as factor,
                      bar.close as close,
                      bar.pre_close as pre_close{circ_mv}
               from f
               left join bar on f.trade_date = bar.trade_date and f.ts_code = bar.ts_code
               {valuation_join}
               where f.factor_name = `{factor_name},
                     f.factor_version = `{factor_version},
                     f.trade_date between {start} : {end}
    """


def _format_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"]).dt.strftime("%Y%m%d")
    return out


def load_ic_from_dolphindb(
    factor_name: str,
    factor_version: str,
    start_date: str,
    end_date: str,
    holding_period: int,
    host: str,
    port: int,
    user: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[pd.DataFrame, list[float]]:
    """Return daily Rank IC and IC decay using DolphinDB ranks and correlations."""
    session = _session(host, port, user, password)
    setup = _factor_base_script(factor_name, factor_version, start_date, end_date, db_path=db_path)
    session.run(
        f"""
        {setup}
        daily_return = select trade_date, ts_code, factor, close,
                              log(close / pre_close) as lndret
                       from base
                       where not isNull(factor), not isNull(close), not isNull(pre_close)
        next_day = select trade_date, ts_code, factor, close, lndret,
                          move(lndret, -1) as one_day_return
                   from daily_return
                   context by ts_code csort trade_date
        factor_rank_min = select trade_date, ts_code, factor, close, lndret,
                                 rank(factor) as factor_rank_min
                          from next_day
                          where not isNull(one_day_return)
                          context by trade_date
        factor_ranked = select trade_date, ts_code, factor, close, lndret,
                               factor_rank_min + (count(factor) - 1) / 2.0 as factor_rank
                        from factor_rank_min
                        context by trade_date, factor
        cumulative_return = select trade_date, ts_code, factor_rank, lndret,
                                   cumsum(lndret) as cumulative_lndret
                            from factor_ranked
                            context by ts_code csort trade_date
        """
    )

    def daily_ic(horizon: int) -> pd.DataFrame:
        return session.run(
            f"""
            future_return = select trade_date, ts_code, factor_rank,
                                   move(cumulative_lndret, -{horizon}) - cumulative_lndret as holding_return
                            from cumulative_return
                            context by ts_code csort trade_date
            return_rank_min = select trade_date, ts_code, factor_rank, holding_return,
                                     rank(holding_return) as return_rank_min
                              from future_return
                              where not isNull(holding_return)
                              context by trade_date
            return_ranked = select trade_date, ts_code, factor_rank,
                                   return_rank_min + (count(holding_return) - 1) / 2.0 as return_rank
                            from return_rank_min
                            context by trade_date, holding_return
            select corr(factor_rank, return_rank) as Rank_IC
            from return_ranked
            group by trade_date
            order by trade_date
            """
        )

    daily = _format_dates(daily_ic(holding_period))
    decay = []
    for horizon in range(1, holding_period + 1):
        values = daily if horizon == holding_period else _format_dates(daily_ic(horizon))
        decay.append(float(values["Rank_IC"].mean()) if not values.empty else float("nan"))
    return daily, decay


def load_quantile_from_dolphindb(
    factor_name: str,
    factor_version: str,
    start_date: str,
    end_date: str,
    layers: int,
    holding_period: int,
    host: str,
    port: int,
    user: str,
    password: str,
    db_path: str = DEFAULT_DB_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily group returns plus only the rebalancing-date memberships."""
    session = _session(host, port, user, password)
    setup = _factor_base_script(
        factor_name, factor_version, start_date, end_date, include_mv=True, db_path=db_path
    )
    daily, memberships = session.run(
        f"""
        {setup}
        future_return = select trade_date, ts_code, factor, circ_mv,
                               log(move(close, -{holding_period}) / close) / {holding_period} as holding_lndret
                        from base
                        where not isNull(factor), not isNull(close)
                        context by ts_code csort trade_date
        ranked = select trade_date, ts_code, circ_mv, holding_lndret,
                        int(rank(factor, false) * {layers} / count(factor)) as quantile
                 from future_return
                 where not isNull(holding_lndret)
                 context by trade_date
        daily = select avg(holding_lndret) as mean_lndret,
                       wavg(holding_lndret, circ_mv) as mean_lndret_vw
                from ranked
                group by trade_date, quantile
                order by trade_date, quantile
        dates = exec distinct trade_date from daily order by trade_date
        rebalance_dates = dates[((0..(size(dates)-1)) % {holding_period}) == 0]
        memberships = select trade_date, ts_code, circ_mv, quantile
                      from ranked
                      where trade_date in rebalance_dates
                      order by trade_date, ts_code
        [daily, memberships]
        """
    )
    return _format_dates(daily), _format_dates(memberships)