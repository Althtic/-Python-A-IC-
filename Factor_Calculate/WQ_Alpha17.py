import pandas as pd
from save_csv import save_data
from factor_distribution_plot import distribution_plot

'''
Alpha_17 = ts_rank(close,10) * rank(delta(delta(close,1))) * rank(ts_rank(adv20,5))
'''

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）


def calculate_alpha_17(data) -> pd.DataFrame:
    data_c = data.copy()
    data_c = data_c.sort_values(by=['ts_code','trade_date'],ascending=[True,True]).reset_index(drop=True)
    # ========================================
    # 优化:pivot 转宽表向量化计算
    # ========================================
    # 创建透视表（日期×股票）
    close_pivot = data_c.pivot(index='trade_date', columns='ts_code', values='close')
    vol_pivot = data_c.pivot(index='trade_date', columns='ts_code', values='vol')

    # Component A: 价格动量反转
    ts_rank_close_10d = close_pivot.rolling(window=10, min_periods=10).rank(pct=True)
    component_A_pivot = -ts_rank_close_10d
    # Component B: 价格加速度
    acce_delta_close = close_pivot.diff(periods=1, axis=0).diff(periods=1, axis=0)
    component_B_pivot = acce_delta_close.rank(pct=True, axis=1)  # axis=1 横截面排名
    # print(component_B_pivot)
    # Component C: 成交量活跃度
    adv20 = vol_pivot.rolling(window=20, min_periods=10).mean()
    ts_rank_adv20_5d = adv20.rolling(window=5, min_periods=5).rank(pct=True)
    component_C_pivot = ts_rank_adv20_5d.rank(pct=True, axis=1)  # axis=1 横截面排名
    # 因子计算
    alpha_17_pivot = component_A_pivot * component_B_pivot * component_C_pivot
    # 还原为长格式(stack.处理Nan)
    alpha_17_long = alpha_17_pivot.stack(dropna=True).reset_index(name='alpha_17')
    alpha_17_long.columns = ['trade_date', 'ts_code', 'alpha_17']
    # 合并回原数据
    data_c = data_c.merge(alpha_17_long, on=['ts_code', 'trade_date'], how='inner')

    return_columns = ['ts_code', 'trade_date', 'open', 'high', 'low', 'close', 'pre_close', 'industry_name', 'alpha_17']
    available_columns = [col for col in return_columns if col in data_c.columns]
    return_data = data_c[available_columns]
    return return_data


if __name__ == "__main__":
    print("--- 开始执行 Alpha#17 计算流程 ---")
    try:
        print("正在读取数据...")
        history_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv')
        print(f"数据读取完成，共 {len(history_data)} 行。")
        # print(history_data.head())
        processed_data = calculate_alpha_17(history_data)
        print(processed_data.head(3))

        print(f"Alpha#17 特征计算完成，共 {len(processed_data)} 行。")
        print(f"正在生成特征分布直方图。")
        distribution_plot(processed_data)
        # print("正在保存数据...")
        # save_data(processed_data, "alpha_17.csv")
        # print("数据已保存。")
    except Exception as e:
        print(f"执行过程中发生未知错误: {e}")