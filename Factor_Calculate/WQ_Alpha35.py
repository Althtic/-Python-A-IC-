import logging
import time
import pandas as pd
import numpy as np
from save_csv import save_data
from factor_mad import mad_outlier_remove
from factor_winsorize import winsorize_factor
from factor_suspension_processing import remove_resume_window_data
from factor_neutralization import neutralize_factor
from factor_zscore_standardization import zscore_transform
from factor_distribution_plot import distribution_plot
from calculate_ts_rank import calc_grouped_rolling_percentile_rank

'''     
成交量与价格综合因子
Alpha_35 = ( ts_rank(vol,32) * (1-ts_rank((close + high - low),16)) * (1-ts_rank(returns,32) )
寻找交易量较大，价格动量或波动相对较小，且近期回报率相对较低的资产
'''

# --- 配置日志 ---
# level=logging.INFO 表示记录 INFO 及以上级别的信息
# format 定义了日志消息的格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）


def calculate_alpha(data):
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True]).reset_index(drop=True)

    data_c['CHL'] = data_c['close'] + data_c['high'] - data_c['low']
    # 准备分组信息 (避免 groupby 迭代)
    # 获取每只股票的起始索引和长度,例如在[0,2,5]索引处被分段（按照股票排序）
    # 方法：利用 ts_code 的变化点
    codes = data_c['ts_code'].values
    # 找到变化的索引位置
    change_indices = np.where(codes[:-1] != codes[1:])[0] + 1 # 各自删除第一个与最后一个数据，错位比较
    group_starts = np.concatenate(([0], change_indices)) # 变化开始的索引位置
    group_lengths = np.diff(np.concatenate((group_starts, [len(codes)]))) # 每次变化后股票数据的长度
    
    # 定义一个 helper 来运行计算
    def run_grouped_calc(col_name, window):
        values = data_c[col_name].values.astype(np.float64) # 确保类型一致
        ranks = calc_grouped_rolling_percentile_rank(values, group_starts, group_lengths, window)
        return ranks
    # 并行或串行计算 (Numba 内部已经很快，通常不需要多进程，除非数据极大)
    rank_vol_32 = run_grouped_calc('vol', 32)
    rank_dret_32 = run_grouped_calc('dret', 32)
    rank_dret_16 = run_grouped_calc('dret', 16) 

    data_c['rank_vol_32'] = rank_vol_32
    data_c['rank_dret_32'] = 1 - rank_dret_32
    data_c['CHL_dret_16'] = 1 - rank_dret_16
    data_c['alpha_35'] = data_c['rank_vol_32'] * data_c['rank_dret_32'] * data_c['CHL_dret_16']
    factor_name = data_c.columns[-1]
    
    try:
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=32)
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
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'raw_factor', 'alpha_35']
    available_columns = [col for col in return_columns if col in data_c.columns]
    return_data = data_c[available_columns]

    end_time = time.time()
    elapsed_time = end_time - start_time
    # 打印耗时信息
    logger.info(f"因子值计算完成！")
    logger.info(f"   - 返回数据行数：{len(return_data)}")
    logger.info(f"   - 总耗时：{elapsed_time:.4f} 秒 ({elapsed_time*1000:.2f} 毫秒)")

    return return_data


if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha#35 计算流程 ---")
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
            'industry_name',
            'suspend_type'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
                                   usecols=columns_needed,
                                   low_memory=False)
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha(history_data)
        # logger.info(processed_data[processed_data['ts_code'] == '000001.SZ'])

        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")
        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_35.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")
