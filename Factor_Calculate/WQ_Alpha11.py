import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
结合价格偏离极值与成交量变动的综合排名因子
Alpha_11 = ((rank(ts_max((vwap-close),3)) + rank(ts_min((vwap-close),3))) * rank(delta(volumn,3))) 
'''

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

def calculate_alpha_11(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True])
    data_c['vwap'] = 10 * data_c['amount'] / data_c['vol']
    data_c['diff_vwap_close'] = (data_c['vwap'] - data_c['close'])
    data_c['ts_max_diff_vc'] = data_c.groupby('ts_code')['diff_vwap_close'].transform(
        lambda x: x.rolling(window=3).max()
    )
    data_c['ts_min_diff_vc'] = data_c.groupby('ts_code')['diff_vwap_close'].transform(
        lambda x: x.rolling(window=3).min()
    )
    data_c['diff_vol_3'] = data_c.groupby('ts_code')['vol'].transform(
        lambda x: x - x.shift(3)
    )
    data_c = data_c.dropna(subset=['ts_max_diff_vc','ts_min_diff_vc','diff_vol_3'])
    data_c['rank_tsmax_vc'] = data_c.groupby('trade_date')['ts_max_diff_vc'].rank(ascending=True,pct=True)
    data_c['rank_tsmin_vc'] = data_c.groupby('trade_date')['ts_min_diff_vc'].rank(ascending=True,pct=True)
    data_c['rank_vol_3'] = data_c.groupby('trade_date')['diff_vol_3'].rank(ascending=True,pct=True)

    data_c['alpha_11'] = (data_c['rank_tsmax_vc'] + data_c['rank_tsmin_vc']) * data_c['rank_vol_3']
    # data_c['alpha_11'] = -1 * (data_c['rank_tsmax_vc'] + data_c['rank_tsmin_vc']) * data_c['rank_vol_3']
    # print(data_c.head())

    data_c = remove_resume_window_data(data_c, window=3)  # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_11']
    return_data = data_c[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#11 计算流程 ---")
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
            'amount',
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
        processed_data = calculate_alpha_11(history_data)
        print(processed_data.head(3))

        print(f"Alpha#11 特征计算完成，共 {len(processed_data)} 行。")
        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        print("正在保存数据...")
        save_data(processed_data, "alpha_11.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")