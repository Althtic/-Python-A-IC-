import pandas as pd
import numpy as np
from save_csv import save_data
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据


'''
            delta(close,1)    if 0 < ts_min(delta(close,1),5)  追涨
Alpha_09 =  delta(close,1)    if ts_max(delta(close,1),5) < 0  杀跌
            -1 * delta(close,1)    otherwise  有涨有跌，反转信号
            
原因子分布太混乱（没有考虑量纲的影响）
调整为收益率的一期差分
'''


# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）


def calculate_alpha_09(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True])

    data_c['delta_dret1d'] = data_c.groupby('ts_code')['dret'].transform(
        lambda x: x - x.shift(1)
    )
    data_c = data_c.dropna().reset_index(drop=True)
    data_c['min_dret_5d'] = data_c.groupby('ts_code')['delta_dret1d'].transform(
        lambda x: x.rolling(5).min()
    )
    data_c['max_dret_5d'] = data_c.groupby('ts_code')['delta_dret1d'].transform(
        lambda x: x.rolling(5).max()
    )
    conditions = [
        data_c['min_dret_5d'] > 0,
        data_c['max_dret_5d'] < 0,
    ]
    choices = [
        data_c['delta_dret1d'],
        data_c['delta_dret1d']
    ]
    # data_c = data_c.dropna(subset=['min_dret_5d','max_dret_5d'])
    data_c['alpha_09'] = np.select(conditions, choices, default=-1 * data_c['delta_dret1d'])


    data_c = remove_resume_window_data(data_c, window=5)  # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_09']
    return_data = data_c[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#09 计算流程 ---")
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
        print("正在读取数据...")
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv',
            usecols=columns_needed
        )
        print(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha_09(history_data)
        # print(processed_data.head(3))

        print(f"Alpha#09 特征计算完成，共 {len(processed_data)} 行。")
        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)

        print("正在保存数据...")
        save_data(processed_data, "alpha_09.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")