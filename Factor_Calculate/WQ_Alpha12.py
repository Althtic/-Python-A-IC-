import pandas as pd
import numpy as np
from save_csv import save_data
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
量价背离反转
Alpha_12 = sign(delta_volumn_1) * (-1 * delta_close_1)
修改：delta_close修改为delta_dret,后取截面排名
'''

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

def calculate_alpha_12(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True])
    data_c['delta_dret'] = data_c.groupby('ts_code')['dret'].diff(1)
    data_c['delta_vol'] = data_c.groupby('ts_code')['vol'].diff(1)
    print(data_c)

    data_c['sign_delta_vol'] = np.sign(data_c['delta_vol'])
    data_c['alpha_12'] = data_c['sign_delta_vol'] * (-1 * data_c['delta_dret'])

    data_c = data_c.dropna(subset=['alpha_12'])

    data_c = remove_resume_window_data(data_c, window=1)  # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_12']
    return_data = data_c[return_columns]
    return return_data

if __name__ == "__main__":
    print("--- 开始执行 Alpha#12 计算流程 ---")
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
        processed_data = calculate_alpha_12(history_data)

        print(f"Alpha#12 特征计算完成，共 {len(processed_data)} 行。")
        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        print("正在保存数据...")
        save_data(processed_data, "alpha_12.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")