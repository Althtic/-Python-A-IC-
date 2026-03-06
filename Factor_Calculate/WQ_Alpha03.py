import logging
import pandas as pd
import numpy as np
from save_csv import save_data  # 确保函数名正确
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
量价背离
Alpha#3 = −1×correlation(rank(Open),rank(Volume),10)
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

def calculate_alpha_03(df):
    df_c = df.copy()
    df_c = df_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True])
    df_c['open_percentile_rank'] = df_c.groupby('trade_date')['open'].rank(pct=True)
    df_c['vol_percentile_rank'] = df_c.groupby('trade_date')['vol'].rank(pct=True)

    df_c['corr_open_vol_percentile'] = df_c.groupby('ts_code', group_keys=False).apply(
        lambda x: x['open_percentile_rank'].rolling(window=10).corr(x['vol_percentile_rank'])
    ).replace([np.inf, -np.inf], np.nan)
    # 对相关系数取负后结果保留6位小数
    df_c['alpha_03'] = -1 * df_c['corr_open_vol_percentile'].round(6)

    # 检查无穷值
    inf_count = np.isinf(df_c['alpha_03']).sum()
    pos_inf_count = (df_c['alpha_03'] == np.inf).sum()
    neg_inf_count = (df_c['alpha_03'] == -np.inf).sum()

    print(f"\n⚠ 无穷值检查:")
    print(f"   无穷值总数：{inf_count:,}")
    print(f"   - 正无穷 (+inf): {pos_inf_count:,}")
    print(f"   - 负无穷 (-inf): {neg_inf_count:,}")


    df_c = df_c.dropna(subset=['alpha_03'])


    df_c = remove_resume_window_data(df_c, window=10) # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_03']
    return_data = df_c[return_columns]
    return return_data


if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha#03 计算流程 ---")
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
            'suspend_type'
        ]
        logger.info("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
            usecols=columns_needed
        )
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")

        processed_data = calculate_alpha_03(history_data)
        logger.info(f"Alpha#03 特征计算完成，共 {len(processed_data)} 行。")


        logger.info(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)

        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_03.csv")
        logger.info("数据已保存。")

    except FileNotFoundError as e:
        logger.error(f"错误: 找不到输入文件")
    except Exception as e:
        logger.error(f"执行过程中发生未知错误: {e}")
    finally:
        logger.info("--- 执行流程结束 ---")