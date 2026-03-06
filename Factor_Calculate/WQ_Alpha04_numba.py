import pandas as pd
import numba as nb
import numpy as np
from save_csv import save_data  # 确保函数名正确
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
均值回归
Alpha#4 = -1 * TsRank(rank(low),9)
'''
@nb.jit(nopython=True, parallel=False)
def count_comparison(window):
    # window 包含了当前值和过去9天的值，共10个数据
    current_value = window[-1]  # 取出最后一个，也就是当前值
    past_values = window[:-1]   # 取出前面9个，也就是过去9天的值
    # 统计过去9天里有多少比当前值小，多少比当前值大
    count_smaller = (past_values < current_value).sum()
    alpha_04 = -1 * np.round(count_smaller / (len(window) - 1), 4)
    return alpha_04

def calculate_alpha_04(df) -> pd.DataFrame:
    df_c = df.copy()
    df_c = df_c.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True])

    df_c['rank_low'] = df_c.groupby('trade_date')['low'].rank(pct=True)
    df_c['alpha_04'] = (df_c.groupby('ts_code')['rank_low'].transform(
        lambda x: x.rolling(10).apply(count_comparison, raw=True)
    )).round(7)
    df_c = df_c.dropna(subset=['alpha_04'])

    df_c = remove_resume_window_data(df_c, window=10) # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_04']
    return_data = df_c[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#04 计算流程 ---")
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
            'suspend_type'
        ]
        print("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
            usecols=columns_needed
        )
        print(f"数据读取完成，共 {len(history_data)} 行。")
        processed_data = calculate_alpha_04(history_data)
        # print(processed_data.head())
        print(f"Alpha#04 特征计算完成，共 {len(processed_data)} 行。")


        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)


        print("正在保存数据...")
        save_data(processed_data, "alpha_04.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")