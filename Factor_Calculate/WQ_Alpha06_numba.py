import pandas as pd
import numpy as np
import numba as nb
from save_csv import save_data  # 确保函数名正确
from factor_distribution_plot import distribution_plot
from factor_suspension_processing import remove_resume_window_data # 处理复牌后window内的异常数据

'''
Alpha_06 = -1 * rolling_corr(open_price, volume, window=10)
'''

# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）
# --- 设置结束 ---

@nb.jit(nopython=True, parallel=False)
def rolling_corr_numba(target_array, feature_array, window):
    """
    计算滚动相关系数的底层函数
    target_array: 主序列 (open)
    feature_array: 副序列 (vol)
    window: 窗口大小
    """
    n = len(target_array)
    result = np.full(n, np.nan)  # 初始化结果数组

    for i in range(window, n):  # 从第 window 个元素开始计算
        # 提取窗口内的子数组
        tgt_win = target_array[i - window:i]
        ftr_win = feature_array[i - window:i]

        # 计算均值
        mean_tgt = np.mean(tgt_win)
        mean_ftr = np.mean(ftr_win)

        # 计算协方差和方差 (手动计算以避免 np.cov 的开销)
        cov = 0.0
        var_tgt = 0.0
        var_ftr = 0.0
        for j in range(window):
            cov += (tgt_win[j] - mean_tgt) * (ftr_win[j] - mean_ftr)
            var_tgt += (tgt_win[j] - mean_tgt) ** 2
            var_ftr += (ftr_win[j] - mean_ftr) ** 2

        # 防止除以0
        if var_tgt > 1e-8 and var_ftr > 1e-8:
            corr = cov / (np.sqrt(var_tgt) * np.sqrt(var_ftr))
        else:
            corr = np.nan

        result[i] = corr

    return result


def calculate_alpha_06(history_data: pd.DataFrame) -> pd.DataFrame:
    data_c = history_data.copy()
    data_c = data_c.sort_values(by=['ts_code', 'trade_date'])

    # 获取唯一的时间点和股票列表 (假设数据是平衡面板)
    dates = data_c['trade_date'].unique()
    codes = data_c['ts_code'].unique()

    # 创建一个多索引 DataFrame 以便快速赋值
    data_c.set_index(['ts_code', 'trade_date'], inplace=True)
    data_c['alpha_06'] = np.nan

    # 遍历每只股票 (groupby 在 numba 场景下往往不如直接循环快)
    for code in codes:
        if code not in data_c.index:
            continue
        stock_data = data_c.loc[code]
        # 确保是时间序列连续的 (如果日期不连续，需要处理 reindex)
        opens = stock_data['open'].values
        vols = stock_data['vol'].values

        if len(opens) < 10:
            continue

        # 调用 numba 加速函数
        corr_values = rolling_corr_numba(opens, vols, 10)

        # 写回结果 (注意：corr_values 是 -1 * corr)
        data_c.loc[code, 'alpha_06'] = -corr_values

    data_c = data_c.reset_index()
    data_c = data_c.dropna(subset=['alpha_06'])

    data_c = remove_resume_window_data(data_c, window=10) # 删除复牌日后窗口期内的异常因子值

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_06']
    return_data = data_c[return_columns]
    return return_data

# 主执行块
if __name__ == "__main__":
    print("--- 开始执行 Alpha#06 计算流程 ---")
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
        history_data = pd.read_csv(
            r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv',
            usecols=columns_needed
        )
        print(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head(3))
        processed_data = calculate_alpha_06(history_data)
        print(f"Alpha#06特征计算完成，共 {len(processed_data)} 行。")


        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)

        '''检查市场截面排序结果'''
        # print(processed_data[processed_data['trade_date'] == 20200103])
        '''检查个股时序计算结果'''
        # print(processed_data[processed_data['ts_code'] == '000011.SZ'])

        print("正在保存数据...")
        save_data(processed_data, "alpha_06.csv")
        print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")