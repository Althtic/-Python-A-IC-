import logging
import time
import pandas as pd
import numpy as np
from save_csv import save_data
from factor_winsorize import winsorize_factor
from factor_mad import mad_outlier_remove
from factor_suspension_processing import remove_resume_window_data
from factor_neutralization import neutralize_factor
from factor_zscore_standardization import zscore_transform
from factor_distribution_plot import distribution_plot
'''     
价格下跌加速指示器
Term1 = (delay(close,20) - delay(close,10)) / 10
Term2 = (delay(close,10) - close) / 10
            1, if Term1 < Term2 < -0.1
Alpha_49 = 
            delay(close, 1) - close, else
捕捉和量化股价在下跌趋势中加速的信号
'''
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_columns', None) # 显示所有列
# pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
# pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

logger = logging.getLogger(__name__)  # 获取一个命名的 logger 实例

def calculate_alpha(data) -> pd.DataFrame:
    start_time = time.time()
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True]).reset_index(drop=True)

    try:
        data_c['delay_close5'] = data_c['close'].shift(20)
        data_c['delay_close10'] = data_c['close'].shift(10)
        # Term 1
        data_c['term_1'] = (data_c['delay_close10'] - data_c['delay_close5']) / 10
        # Term 2
        data_c['term_2'] = (data_c['delay_close10'] - data_c['close']) / 10
    except Exception as e:
        logger.info(f"Error in Term Calculating: {e}")
        raise e

    try:
        data_c['alpha_49'] = np.where(
            data_c['term_1'] - data_c['term_2'] < -0.1,
            1,
            (data_c['pre_close'] - data_c['close']).round(7)
        )
        factor_name = data_c.columns[-1]
    except Exception as e:
        logger.info(f"Error in Alpha49 Calculating: {e}")
        raise e

    try:
        # 删除掉预热期数据（每只股票前20个交易日）
        data_c = data_c.apply(lambda x: x.iloc[20:]).reset_index(level=0, drop=True)
        # 删除复牌日股票在.shift操作中可能产生的错误计算
        data_c = remove_resume_window_data(data_c, window=20)
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

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'raw_factor', 'alpha_49']
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
    logger.info("--- 开始执行 Alpha#49 计算流程 ---")
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
            'industry_name',
            'suspend_type'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
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
        save_data(processed_data, "alpha_49.csv")
        logger.info("数据已保存。")
    except Exception as e:
        logger.info(f"执行过程中发生未知错误: {e}")