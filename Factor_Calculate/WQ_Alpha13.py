import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
反转因子
Alpha_13 = -1 * rank(covariance(rank(close),rank(volumn),5))
'''

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

def calculate_alpha_13(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True])

    data_c['rank_close'] = data_c.groupby('trade_date')['close'].rank(ascending=True, pct=True)
    data_c['rank_vol'] = data_c.groupby('trade_date')['vol'].rank(ascending=True, pct=True)
    '''
    .cov()：计算这个 5天窗口内rank_close和rank_vol之间的协方差矩阵。.unstack()['rank_close']['rank_vol']
    从计算出的协方差矩阵中，提取rank_close和rank_vol之间的协方差值。.unstack()重新组织了MultiIndex，
    使得可以通过['rank_close']['rank_vol']来索引。
    '''
    data_c['cov_5d'] = data_c.groupby('ts_code').apply(
        lambda df: df[['rank_close', 'rank_vol']].rolling(window=5).cov().unstack()['rank_close']['rank_vol']
    ).reset_index(level=0, drop=True)
    data_c = data_c.dropna(subset=['cov_5d'])
    data_c['rank_cov_5d'] = data_c.groupby('trade_date')['cov_5d'].rank(ascending=True, pct=True)
    data_c['alpha_13'] = -1 * data_c['rank_cov_5d']
    # print(data_c.head())
    data_c = remove_resume_window_data(data_c, window=5)  # 删除复牌日后窗口期内的异常因子值
    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_13']
    return_data = data_c[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#13 计算流程 ---")
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
            'industry_name',
            'suspend_type'
        ]
        print("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv',
                                   usecols=columns_needed)
        print(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha_13(history_data)

        print(f"Alpha#13 特征计算完成，共 {len(processed_data)} 行。")
        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        print("正在保存数据...")
        save_data(processed_data, "alpha_13.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")