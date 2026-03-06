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
from calculate_rolling_corr import rolling_corr_numba

'''     
复合动量与相关性策略
Alpha_45 = -1 * (rank(Mid-tern Momentum)) * (Price-Volumn Correlation) * (Trend Consistency Rank)
结合多维市场信号，识别强势资产
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

def calculate_alpha(data) -> pd.DataFrame:
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True]).reset_index(drop=True)

    try:
        # Part A 中期动量排名
        data_c['delay_c5'] = data_c['close'].shift(5)
        data_c['mean_sum_d_c5'] = data_c['delay_c5'].rolling(window=15).mean()
        data_c['rank_mid_momentum'] = data_c.groupby('trade_date')['mean_sum_d_c5'].rank(pct=True)
    except Exception as e:
        logger.info(f"Error in Part A (Momentum): {e}")
        raise e

    try:
        # Part B 量价相关性
        def compute_corr(group):
            series_1 = group['close'].values
            series_2 = group['vol'].values
            
            n = len(series_2)
            if n < 10: return pd.Series([np.nan] * n, index=group.index)
            # 调用相关系数计算函数
            corr_res = rolling_corr_numba(series_1, series_2, 5) 
            # 必须返回一个 Series，且 index 要与 group 的 index 一致，这样 apply 才能自动对齐
            return pd.Series(corr_res, index=group.index)
        data_c['corr_close_vol_2'] = data_c.groupby('ts_code', group_keys=False).apply(compute_corr)
    except Exception as e:
        logger.info(f"Error in Part B (Vol-Price Corr): {e}")
        raise e

    try:
        # Part C 趋势一致性排名
        def compute_corr(group):
            series_1 = group['sum_c5'].values
            series_2 = group['sum_c20'].values
            n = len(series_2)
            if n < 10: return pd.Series([np.nan] * n, index=group.index)
            corr_res = rolling_corr_numba(series_1, series_2, 5) 
            return pd.Series(corr_res, index=group.index)
        
        data_c['sum_c5'] = data_c['close'].rolling(window=5).sum()
        data_c['sum_c20'] = data_c['close'].rolling(window=20).sum()
  
        data_c['corr_sumc5_sumc20_2'] = data_c.groupby('ts_code', group_keys=False).apply(compute_corr)
        data_c['rank_corr_sumc5_sumc20'] = data_c.groupby('trade_date')['corr_sumc5_sumc20_2'].rank(pct=True)

    except Exception as e:
        logger.info(f"Error in Part C (Consistency): {e}")
        raise e

    try:
        # Part D 计算最终 Alpha
        data_c['alpha_45'] = -1 * data_c['rank_mid_momentum'] * data_c['corr_close_vol_2'] * data_c['rank_corr_sumc5_sumc20']
        factor_name = data_c.columns[-1]
    except Exception as e:
        logger.info(f"Error in Part D (Final): {e}")
        raise e

    try:
        # inf 替换为 Nan
        data_c['alpha_45'] = data_c['alpha_45'].replace([np.inf, -np.inf], np.nan)
        # 删除掉预热期数据（每只股票前20个交易日）
        data_c = data_c.apply(lambda x: x.iloc[25:]).reset_index(level=0, drop=True)
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=25)
        # 去除极端值（MAD）
        data_c = mad_outlier_remove(data_c)
        # 去除极端值（分位数截断法）
        # data_c = winsorize_factor(data_c)
        # 中性化处理因子值（市值 & 行业：回归残差法）
        data_c = neutralize_factor(data_c, target_factor=factor_name)
        # 删除Nan值
        data_c = data_c.dropna(subset=[factor_name])
        # 截面 z-score 标准化
        data_c = zscore_transform(data_c)

    except Exception as e:
        logger.info(f"Error in Final Calculation or Post-processing: {e}")
        raise e

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'raw_factor', 'alpha_45']
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
    logger.info("--- 开始执行 Alpha#45 计算流程 ---")
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
            'dret',
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
        save_data(processed_data, "alpha_45.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")
