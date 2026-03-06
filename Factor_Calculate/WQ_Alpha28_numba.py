import logging
import time
import pandas as pd
import numpy as np
from numba import njit
from save_csv import save_data
from factor_mad import mad_outlier_remove
from factor_winsorize import winsorize_factor
from factor_suspension_processing import remove_resume_window_data
from factor_neutralization import neutralize_factor
from factor_zscore_standardization import zscore_transform
from factor_distribution_plot import distribution_plot

'''      
成交量与最低价的相关性，结合价格的中枢偏离度
Alpha_28 = rank((correlation(adv20,low,5) + ((high+low)/2)-close)
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

@njit
def rolling_corr(x, y, window):
    """滚动相关系数（numba加速）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        x_win = x[i-window+1:i+1]
        y_win = y[i-window+1:i+1]
        if np.any(np.isnan(x_win)) or np.any(np.isnan(y_win)):
            continue
        x_mean = np.mean(x_win)
        y_mean = np.mean(y_win)
        x_std = np.std(x_win)
        y_std = np.std(y_win)
        if x_std > 0 and y_std > 0:
            result[i] = np.mean((x_win - x_mean) * (y_win - y_mean)) / (x_std * y_std)
    return result

def calculate_alpha(data) -> pd.DataFrame:
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True]).reset_index(drop=True)
    data_c['adv20'] = (
        data_c.groupby('ts_code')['vol'].rolling(window=20).mean().reset_index(level=0, drop=True)
    )
    data_c['mid_price'] = (data_c['high'] - data_c['low']) / 2

    corr_adv20_low_5 = []
    for _, group in data_c.groupby('ts_code', sort=False):
        vol_series = group['vol'].values
        adv20_series = group['adv20'].values
        corr = rolling_corr(vol_series, adv20_series, 5)
        corr_adv20_low_5.extend(corr)
    data_c['corr_adv20_low_5'] = corr_adv20_low_5
    data_c = data_c.dropna(subset=['corr_adv20_low_5','mid_price','close'])
    # 价格偏离度
    data_c['alpha_28'] = data_c['corr_adv20_low_5'] + data_c['mid_price'] - data_c['close']
    factor_name = data_c.columns[-1]

    try:
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=20)
        # 去除极端值（MAD）(可选项 1)
        data_c = mad_outlier_remove(data_c)
        # 去除极端值（分位数截断法）(可选项 1)
        # data_c = winsorize_factor(data_c)
        # 中性化处理因子值（市值 & 行业：回归残差法）
        data_c = neutralize_factor(data_c, target_factor=factor_name)
        # 删除Nan值
        data_c = data_c.dropna(subset=[factor_name])
        # 截面 z-score 标准化 (可选项 2)
        # data_c = zscore_transform(data_c)
        # 截面 rank 排序 (可选项 2)
        data_c['alpha_28'] = data_c.groupby('trade_date')['alpha_28'].rank(pct=True)
    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'raw_factor', 'alpha_28']
    return_data = data_c[return_columns]

    end_time = time.time()
    elapsed_time = end_time - start_time
    # 打印耗时信息
    logger.info(f"因子值计算完成！")
    logger.info(f"   - 返回数据行数：{len(return_data)}")
    logger.info(f"   - 总耗时：{elapsed_time:.4f} 秒 ({elapsed_time*1000:.2f} 毫秒)")

    return return_data


if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha 计算流程 ---")
    try:
        columns_needed = [
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'pre_close',
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

        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")
        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_28.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")
