import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
放量上涨但价格下跌，做多；放量上涨且价格上涨，做空
            -1 * ts_rank(|delta(close,7)|,60) * sign(delta(close,7)), if vol > adv20
alpha_07 =
            -1, otherwise
            
ts_rank(|delta(close,7)|,60) = "今天的7日价格波动幅度，在过去60天里属于什么水平？"
用来加权交易信号：波动越剧烈，反转信号越强
'''


# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）
# --- 设置结束 ---

def calculate_alpha_07(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'], ascending=[True,True])
    # adv20 (20-day average volume)
    data_c['vol_average_20d'] = data_c.groupby('ts_code')['vol'].transform(
        lambda x: x.rolling(window=20).mean()
    )
    # delta(close, 7)
    data_c['delta_close_7'] = data_c.groupby('ts_code')['close'].transform(
        lambda x: x - x.shift(7)
    )
    # sign of delta(close, 7)
    data_c['sign_delta_close_7'] = np.sign(data_c['delta_close_7'])
    # ts_rank(abs(delta(close, 7)), 60)
    data_c['rank_delta_close'] = data_c.groupby('ts_code')['delta_close_7'].transform(
        lambda x: abs(x).rolling(window=60,min_periods=20).rank(pct=True)
    )
    # alpha07
    data_c['alpha_07'] = np.where(
        data_c['vol'] > data_c['vol_average_20d'], # 条件
        -1 * data_c['rank_delta_close'] * data_c['sign_delta_close_7'], # 条件成立时候的操作
        -1 # 条件不成立时的操作
    )

    data_c = remove_resume_window_data(data_c, window=60) # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code','trade_date','open','high','low','close','pre_close','industry_name','alpha_07']
    data_c = data_c.dropna()
    return_data = data_c[return_columns]
    return return_data


if __name__ == "__main__":
    print("--- 开始执行 Alpha#07 计算流程 ---")
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
        print("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv',
            usecols=columns_needed
        )
        print(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha_07(history_data)
        # print(processed_data.head())
        print(f"Alpha#07 特征计算完成，共 {len(processed_data)} 行。")

        print("正在生成特征分布图。")
        distribution_plot(processed_data)

        print("正在保存数据...")
        save_data(processed_data, "alpha_07.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")
