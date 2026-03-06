import logging
import pandas as pd
import numpy as np
from save_csv import save_data  # 确保函数名正确
from factor_distribution_plot import distribution_plot # 初步检查数据分布
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
量价背离
Alpha#2 = (-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
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




def calculate_alpha_02(df: pd.DataFrame) -> pd.DataFrame:
    history_data = df.copy()
    history_data = history_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True])
    history_data['intrady_ret'] = (history_data['close'] - history_data['open']) / history_data['open']
    # 对数成交量及其二阶差分
    history_data['log_vol'] = np.log(history_data['vol'])
    history_data['delta_vol_diff'] = history_data.groupby('ts_code')['log_vol'].diff(2)
    history_data['percentile_voldiff'] = history_data.groupby('trade_date')['delta_vol_diff'].rank(pct=True)
    history_data['percentile_intradydret'] = history_data.groupby('trade_date')['intrady_ret'].rank(pct=True)
    # 计算滚动相关系数 (按股票分组) 使用 apply + lambda 的方式
    history_data['corr_dert_voldiff'] = history_data.groupby('ts_code', group_keys=False).apply(
        lambda x: x['percentile_voldiff'].rolling(window=6).corr(x['percentile_intradydret'])
    )
    history_data['corr_dert_voldiff'] = history_data['corr_dert_voldiff'].round(6)
    # 计算 alpha_02 (相关系数的负值)
    history_data['alpha_02'] = (history_data['corr_dert_voldiff'] * -1).round(7)
    history_data = history_data.dropna(subset=['alpha_02'])

    history_data = remove_resume_window_data(history_data, window=6) # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_02']
    return_data = history_data[return_columns]
    return return_data

# --- 主执行块 ---
if __name__ == "__main__":
    logger.info("--- 开始执行 因子 计算流程 ---")
    try:
        columns_needed = [ # Alpha #1 计算所需数据列
            'ts_code',
            'trade_date',
            'open',
            'high',
            'low',
            'close',
            'pre_close',
            'vol',
            'industry_name',
            'dret',
            'suspend_type' # rolling计算的误差，删除掉 R 日之后window长度的因子数据
        ]
        logger.info("正在读取数据...")
        # history_data = pd.read_csv('../daily_stock_20200101_20250101.csv')
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
            usecols = columns_needed
        )
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")

        # print(history_data.head(20))
        # print('===========================')

        # Alpha02计算函数
        processed_data = calculate_alpha_02(history_data)
        logger.info(f"特征计算完成，共 {len(processed_data)} 行。")

        logger.info(f"正在生成特征分布图。")
        distribution_plot(processed_data)

        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_02.csv")
        logger.info("数据已保存。")

    except FileNotFoundError as e:
        logger.error(f"错误: 找不到输入文件")
    except Exception as e:
        logger.error(f"执行过程中发生未知错误: {e}")
    finally:
        logger.info("--- 执行流程结束 ---")

