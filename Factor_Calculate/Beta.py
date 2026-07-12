import logging
import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot

'''
Fama-French MKT 因子
按照市值权重构建市场组合与无风险利率做差
'''

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)  # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)  # 列宽无限制（防止单元格内容被截断）

def calculate_alpha(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True]).reset_index(drop=True)
    # 合并无风险利率数据（Shibor 3M）
    data_rf = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\rf.csv',
                            usecols=['trade_date', 'rf'])
    data_c = data_c.merge(data_rf, how='inner', on='trade_date')
    data_c['excess_dret'] = data_c['dret'] - data_c['rf']
    # 按流通市值构造市场组合权重
    data_c['mv_weight'] = data_c.groupby('trade_date')['circ_mv'].transform(
        lambda x: x / x.sum()
    )

    data_c['contrib'] = data_c['excess_dret'] * data_c['mv_weight']
    # 市场组合每日超额收益为各股超额收益按市值加权
    market_group_dret = data_c.groupby('trade_date')['contrib'].sum()
    data_c['market_group_dret'] = data_c['trade_date'].map(market_group_dret)

    data_beta = data_c[['trade_date', 'ts_code', 'excess_dret', 'market_group_dret', 'mv_weight']]
    data_beta = data_beta.dropna()

    # beta = （X^T · X）^-1 * X^T * Y
    def calculate_rolling_beta(group, window=120):
        min_window = int(2/3 * int(window))
        if len(group) < window:
            return pd.Series(np.nan, index=group.index)
        try:
            y = group['excess_dret'].astype(float)
            x = group['market_group_dret'].astype(float)
        except KeyError:
            raise KeyError("请确保 dataframe 中包含 'excess_dret' 和 'market_group_dret' 列")
        if y.isna().all() or x.isna().all():
            return pd.Series(np.nan, index=group.index)
        roll_cov = x.rolling(window=window, min_periods=min_window).cov(y)
        roll_var = x.rolling(window=window, min_periods=min_window).var()
        beta = roll_cov / roll_var
        return beta.reindex(group.index)

    result_series = data_beta.groupby('ts_code', group_keys=False).apply(
        lambda g: calculate_rolling_beta(g, window=21)
    )

    factor_name = 'mkt'
    return_data = result_series.reset_index()
    print(return_data)



    try:
        return_data = return_data.dropna(subset=factor_name)
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e
    return return_data

if __name__ == "__main__":
    logger.info("--- 开始执行 Beta 计算流程 ---")
    try:
        columns_needed = [
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'dret',
            'industry_name',
            'circ_mv',
            'suspend_type'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_pipe.csv',
            usecols=columns_needed,
            low_memory=False)
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")
        logger.info(f"开始计算特征")
        # print(history_data.head())
        processed_data = calculate_alpha(history_data)
        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")
        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        logger.info("正在保存数据...")
        save_data(processed_data, "mkt.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")