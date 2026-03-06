import pandas as pd
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据


'''
“量价背离”与“量价共振”后可能发生的均值回归
Alpha_15 = -1 * sum(rank(correlation(rank(high),rank(volumn),3)),3))
'''

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

def calculate_alpha_15(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True])

    data_c['rank_high'] = data_c.groupby('trade_date')['high'].rank(pct=True)
    data_c['rank_vol'] = data_c.groupby('trade_date')['vol'].rank(pct=True)
    data_c['corr_high_vol_3'] = data_c.groupby('ts_code').apply(
        lambda x: x['high'].rolling(3).corr(x['vol'])
    ).droplevel(level=0).sort_index()
    data_c['rank_corr_h_v'] = data_c.groupby('trade_date')['corr_high_vol_3'].rank(pct=True)
    data_c['sum_rank_corr_3d'] = data_c.groupby('ts_code')['corr_high_vol_3'].rolling(3).sum().reset_index(level=0,drop=True)
    data_c = data_c.dropna(subset=['sum_rank_corr_3d'])
    data_c['alpha_15'] = -data_c['sum_rank_corr_3d']

    data_c = remove_resume_window_data(data_c, window=3)  # 删除复牌日后窗口期内的异常因子值
    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_15']
    return_data = data_c[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#15 计算流程 ---")
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
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv')
        print(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha_15(history_data)
        print(processed_data.head(3))

        print(f"Alpha#15 特征计算完成，共 {len(processed_data)} 行。")
        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        print("正在保存数据...")
        save_data(processed_data, "alpha_15.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")