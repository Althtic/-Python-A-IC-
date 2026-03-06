import pandas as pd
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据


'''
Alpha_08 = -1 * rank((sum(open,5) * sum(returns,5) - delay((sum(open,5) * sum(returns,5)),10))
'''

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

def calculate_alpha_08(data) -> pd.DataFrame:
    data_c = data.sort_values(by=['ts_code', 'trade_date']).reset_index(drop=True)
    data_c['sum_open_5'] = data_c.groupby('ts_code')['open'].transform(
        lambda x: x.rolling(window=5).sum()
    )
    data_c['sum_return_5'] = data_c.groupby('ts_code')['dret'].transform(
        lambda x: x.rolling(window=5).sum()
    )
    data_c['open_return_mutiple'] = data_c['sum_open_5'] * data_c['sum_return_5']
    data_c['diff10_open_return_mutiple'] = data_c.groupby('ts_code')['open_return_mutiple'].diff(10)
    data_c.dropna(subset=['diff10_open_return_mutiple'], inplace=True)
    data_c['alpha_08'] = data_c.groupby('trade_date')['diff10_open_return_mutiple'].transform(
        lambda x: -1 * x.rank(pct=True, method='average')
    )

    data_c = remove_resume_window_data(data_c, window=10)  # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_08']
    return_data = data_c[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#08 计算流程 ---")
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
        processed_data = calculate_alpha_08(history_data)
        # print(processed_data.head(3))
        print(f"Alpha#08 特征计算完成，共 {len(processed_data)} 行。")

        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)

        print("正在保存数据...")
        save_data(processed_data, "alpha_08.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")