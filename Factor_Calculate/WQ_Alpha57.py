import logging
import time
import pandas as pd
import numpy as np
import warnings
from save_csv import save_data
from factor_mad import mad_outlier_remove
from factor_winsorize import winsorize_factor
from factor_suspension_processing import remove_resume_window_data
from factor_neutralization import neutralize_factor
from factor_zscore_standardization import zscore_transform
from factor_distribution_plot import distribution_plot
from calculate_ts_rank import calc_grouped_rolling_percentile_rank
from calculate_linear_decay import linear_decay_peaks
'''
收盘价与VWAP差异的线性衰减因子
Alpha_57 = -1 * ( (close-vwap) / (decay_linear(rank(ts_argmax(close,21)), 2) ) )
捕捉价格偏离后的反转潜力
'''
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）


    
def calculate_alpha(data) -> pd.DataFrame:
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True]).reset_index(drop=True)

    # part 1 收盘价与VWAP的差异
    try:
        data_c['vwap'] = 10 * data_c['amount'] / data_c['vol']
        data_c['close_vwap_diff'] = data_c['close'] - data_c['vwap']
    except Exception as e:
        logger.info(f"Error in Part 1: {e}")
        raise e

    # part 2 近期价格峰值与相对位置(线性衰减)
    try:
        codes = data_c['ts_code'].values
        # 找到变化的索引位置
        change_indices = np.where(codes[:-1] != codes[1:])[0] + 1 # 各自删除第一个与最后一个数据，错位比较
        group_starts = np.concatenate(([0], change_indices)) # 变化开始的索引位置
        group_lengths = np.diff(np.concatenate((group_starts, [len(codes)]))) # 每次变化后股票数据的长度
        
        def run_grouped_calc(col_name, window):
            values = data_c[col_name].values.astype(np.float64)
            ranks = calc_grouped_rolling_percentile_rank(values, group_starts, group_lengths, window)
            return ranks
        ts_argmax_close_30 = run_grouped_calc('close', 21)
        data_c['ts_argmax_close_30'] = ts_argmax_close_30
        print(data_c[['ts_code', 'trade_date', 'close', 'ts_argmax_close_30']])
        data_c['rank_ts_argmax_close_30'] = data_c.groupby('trade_date')['ts_argmax_close_30'].rank(pct=True)

        decay_linear_rank_close_30 = linear_decay_peaks(data_c['rank_ts_argmax_close_30'], codes, 2)
        data_c['decay_linear_rank_close_30'] = decay_linear_rank_close_30
        print(data_c[['ts_code', 'trade_date', 'close', 'ts_argmax_close_30','decay_linear_rank_close_30']])

    except Exception as e:
        logger.info(f"Error in Part 2: {e}")
        raise e

    # part 3 最终因子
    try:
        data_c['alpha_57'] = -1 * (data_c['close_vwap_diff'] / data_c['decay_linear_rank_close_30'])
        factor_name = data_c.columns[-1]

    except Exception as e:
        logger.info(f"Error in Part 3: {e}")
        raise e

    try:
        data_c[factor_name] = data_c[factor_name].replace([np.inf, -np.inf], np.nan)
        data_c = data_c.dropna(subset=[factor_name])
        # 删除掉预热期数据（每只股票前20个交易日）
        # data_c = data_c.apply(lambda x: x.iloc[20:]).reset_index(level=0, drop=True)
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=21)
        # 去除极端值（MAD）
        # data_c = mad_outlier_remove(data_c)
        # 去除极端值（分位数截断法）
        data_c = winsorize_factor(data_c)
        # 中性化处理因子值（市值 & 行业：回归残差法）
        data_c = neutralize_factor(data_c, target_factor=factor_name)
        # 删除Nan值
        data_c = data_c.dropna(subset=[factor_name])
        # 截面 z-score 标准化
        data_c = zscore_transform(data_c)
        # 截面 rank 标准化
        # data_c[factor_name] = data_c.groupby('trade_date')[factor_name].rank(pct=True)
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'raw_factor', factor_name]
    available_columns = [col for col in return_columns if col in data_c.columns]
    return_data = data_c[available_columns]

    end_time = time.time()
    elapsed_time = end_time - start_time

    # 打印耗时信息
    logger.info(f"因子值计算完成！")
    logger.info(f"   - 返回数据行数：{len(return_data)}")
    logger.info(f"   - 总耗时：{elapsed_time:.4f} 秒")

    return return_data


if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha#57 计算流程 ---")
    try:
        columns_needed = [
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'pre_close',
            'dret',
            'vol',
            'amount',
            'industry_name',
            'suspend_type',
            'circ_mv'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_pipe.csv',
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
        save_data(processed_data, "alpha_57.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")