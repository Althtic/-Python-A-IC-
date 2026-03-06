import logging
import pandas as pd
import numpy as np
import numba as nb
from save_csv import save_data  # 确保函数名正确
from factor_distribution_plot import distribution_plot

'''
Alpha01 修改版：使用 ts_max 代替 ts_argmax
原始公式: (rank(ts_argmax(signedpower((($returns < 0) ? stddev($returns, 20) : $close), 2), 5)) - 0.5)
修改后: 使用滚动窗口最大值代替最大值位置
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

# --- Numba 和辅助函数定义 ---
@nb.jit(nopython=True, parallel=False)
def ts_max_numba(series_values, window):
    n = len(series_values)
    result = np.full(n, np.nan, dtype=np.float64)

    if n < window:
        return result
    for i in range(window - 1, n):
        window_data_slice = series_values[i - window + 1: i + 1]
        max_val_in_window = np.max(window_data_slice)
        result[i] = float(max_val_in_window)
    return result

def calculate_max_for_series(group_series, window_size):
    series_values = group_series.values
    result_array = ts_max_numba(series_values, window=window_size)
    return pd.Series(result_array, index=group_series.index)

def process_alpha_01_features(history_data, window_size=5):
    logger.info(f"开始计算 Alpha#01 特征，窗口大小: {window_size}")
    history_data = history_data.copy()
    history_data = history_data.sort_values(by=['ts_code','trade_date'], ascending=[True, True])
    # 提取负收益
    history_data['neg_returns'] = np.where(history_data['dret'] < 0, history_data['dret'], 0)
    # 平方保留负号，放大尾部风险
    history_data['squared_neg'] = np.sign(history_data['neg_returns']) * (np.abs(history_data['neg_returns']) ** 2)
    # 使用 groupby.transform 配合 lambda 调用辅助函数
    # transform 会自动将结果广播回原始 DataFrame 的形状
    # 注意：lambda 需要捕获 window_size 变量
    history_data['max_val'] = history_data.groupby('ts_code')['squared_neg'].transform(
        lambda x: calculate_max_for_series(x, window_size)
    )

    initial_len = len(history_data)
    history_data = history_data.dropna()
    final_len = len(history_data)
    logger.info(f"删除了 {initial_len - final_len} 行包含 NaN 的数据。")
    # 由于现在返回的是最大值而不是位置，直接使用最大值作为因子值
    # 注意：原始Alpha01因子是基于位置进行排名的，这里简化处理

    history_data['alpha_01'] = (history_data.groupby('trade_date')['max_val'].rank(pct=True) - 0.5).round(7)

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_01']
    return_data = history_data[return_columns]
    return return_data

# --- 主执行块 ---
if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha#01 计算流程 ---")
    try:
        columns_needed = [ # Alpha #1 计算所需数据列
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'pre_close',
            'industry_name',
            'dret'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
            usecols=columns_needed
        )
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")

        # 可以在此更改窗口大小
        WINDOW_SIZE = 5

        logger.info(f"正在计算特征 (窗口大小: {WINDOW_SIZE})...")
        processed_data = process_alpha_01_features(history_data, window_size=WINDOW_SIZE)
        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")

        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)

        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_01.csv")
        logger.info("数据已保存。")

    except FileNotFoundError as e:
        logger.error(f"错误: 找不到输入文件")
    except Exception as e:
        logger.error(f"执行过程中发生未知错误: {e}")
    finally:
        logger.info("--- 执行流程结束 ---")