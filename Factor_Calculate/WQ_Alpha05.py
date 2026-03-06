import pandas as pd
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
Alpha_05 = rank_open * (-1 * abs(rank_close))
'''

def calculate_alpha_05(df: pd.DataFrame) -> pd.DataFrame:
    history_data = df.copy()
    history_data = history_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True,True])
    history_data['vwap'] = 10 * history_data['amount'] / history_data['vol']

    history_data['average10d_vwap'] = history_data.groupby('ts_code')['vwap'].transform(
        lambda x: x.rolling(window=10).mean()
    )
    # close与open相较vawp偏离度
    diff_open = history_data['open'] - history_data['average10d_vwap']
    diff_close = history_data['close'] - history_data['average10d_vwap']
    # 偏离度排序
    rank_open = diff_open.groupby(history_data['trade_date']).rank(pct=True)
    rank_close = diff_close.groupby(history_data['trade_date']).rank(pct=True)

    alpha_05 = rank_open * (-1 * rank_close.abs())
    history_data['alpha_05'] = alpha_05.round(7)
    history_data['alpha_05'] = history_data['alpha_05'].round(7)
    processed_data = history_data.dropna(subset=['alpha_05'])

    processed_data = remove_resume_window_data(processed_data, window=10) # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_05']
    return_data = processed_data[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#05 计算流程 ---")
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
            'amount',
            'industry_name',
            'suspend_type'
        ]
        print("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
            usecols=columns_needed
        )
        print(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha_05(history_data)
        print(f"Alpha#05 特征计算完成，共 {len(processed_data)} 行。")

        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)

        print("正在保存数据...")
        save_data(processed_data, "alpha_05.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")