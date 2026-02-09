import logging
import pandas as pd
from save_csv import save_data  # 确保函数名正确

'''
量价背离
Alpha#3 = scale(rank(−1×correlation(rank(Open),rank(Volume),10)))
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
    df_c['SE'] = df['ts_code'].str.slice(-2)
    Stock_Exchange = df_c['SE'].unique()

    all_exchange_data = []
    for ex in Stock_Exchange:

        exchange_mask = df_c['SE'] == ex
        exchange_data = df_c.loc[exchange_mask].copy()
        exchange_data['open_percentile_rank'] = df_c.groupby('trade_date')['open'].rank(pct=True)
        exchange_data['vol_percentile_rank'] = df_c.groupby('trade_date')['vol'].rank(pct=True)

        # exchange_data = exchange_data.sort_values(['ts_code','trade_date'], ascending=[True, True]).reset_index(drop=True)
        exchange_data['corr_open_vol_percentile'] = exchange_data.groupby('ts_code', group_keys=False).apply(
            lambda x: x['open_percentile_rank'].rolling(window=10).corr(x['vol_percentile_rank'])
        )

        # 对相关系数结果保留6位小数
        exchange_data['corr_open_vol_percentile'] = exchange_data['corr_open_vol_percentile'].round(6)
        exchange_data['alpha_03'] = exchange_data['corr_open_vol_percentile'] * -1
        exchange_data = exchange_data.dropna()
        # print(exchange_data)
        all_exchange_data.append(exchange_data)

    final_result_df = pd.concat(all_exchange_data, ignore_index=True)
    final_result_df = final_result_df.dropna()

    columns_to_drop = [
        'vol_percentile_rank',
        'open_percentile_rank',
        'corr_open_vol_percentile'
    ]
    final_result_df = final_result_df.drop(columns=columns_to_drop)

    return final_result_df

if __name__ == "__main__":
    logger.info("--- 开始执行 Alpha#03 计算流程 ---")
    try:
        logger.info("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20200101-20250101.csv')
        logger.info(f"数据读取完成，共 {len(history_data)} 行。")

        processed_data = calculate_alpha_03(history_data)
        logger.info(f"Alpha#02 特征计算完成，共 {len(processed_data)} 行。")

        logger.info("正在保存数据...")
        save_data(processed_data, "alpha_03.csv")
        logger.info("数据已保存。")

    except FileNotFoundError as e:
        logger.error(f"错误: 找不到输入文件 '20200101_20250101.csv': {e}")
    except Exception as e:
        logger.error(f"执行过程中发生未知错误: {e}")
    finally:
        logger.info("--- 执行流程结束 ---")