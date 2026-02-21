import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
from scipy import stats
from factor_validation_config_loader import traget_factor, layers, test_window_start, test_window_end

# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

# 加载数据
def data_loading(traget_factor):
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df_loading = pd.read_csv(Input_path)
    return df_loading
# 筛选时间窗口
def cut_time_window(df, start_time, end_time):
    try:
        # 预处理原始数据的日期列
        if df['trade_date'].dtype in ['int64', 'int32']:
            df_date_for_check = pd.to_datetime(df['trade_date'].astype(str), format='%Y%m%d')
        elif df['trade_date'].dtype == 'object':
            df_date_for_check = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        else:
            df_date_for_check = df['trade_date']

        # 获取原始数据的日期范围
        original_min_date = df_date_for_check.min()
        original_max_date = df_date_for_check.max()
        print(f"原始数据的时间范围: {original_min_date.strftime('%Y%m%d')} 到 {original_max_date.strftime('%Y%m%d')}")

        # 将输入的时间也转换为 datetime 进行比较
        input_start_dt = pd.to_datetime(str(start_time), format='%Y%m%d')
        input_end_dt = pd.to_datetime(str(end_time), format='%Y%m%d')

        # 判断输入窗口是否超出原始数据范围 ---
        if input_start_dt > original_max_date or input_end_dt < original_min_date:
            print(f"错误: 请求的时间窗口 [{start_time}, {end_time}] 完全超出了原始数据的日期范围 [{original_min_date.strftime('%Y%m%d')}, {original_max_date.strftime('%Y%m%d')}]。")
            print("--- 程序终止 ---")

            return None # 返回 None 表示操作失败

        if input_start_dt < original_min_date or input_end_dt > original_max_date:
             # 检查是否有部分超出，如果有，也视为超出范围
             effective_start = max(input_start_dt, original_min_date)
             effective_end = min(input_end_dt, original_max_date)
             print(f"警告: 请求的时间窗口 [{start_time}, {end_time}] 部分超出了原始数据范围。")
             print(f"     建议使用有效范围: [{effective_start.strftime('%Y%m%d')}, {effective_end.strftime('%Y%m%d')}]")

             return None # 返回 None 表示操作失败

        # 如果时间窗口在范围内，进行筛选
        # 原始数据 'trade_date' 列转化为字符串 string (用于后续筛选)
        df_copy = df.copy()
        if df_copy['trade_date'].dtype in ['int64', 'int32']:
            df_copy['trade_date'] = df_copy['trade_date'].astype(str)

        # 创建布尔掩码并筛选
        mask = (df_copy['trade_date'] >= start_time) & (df_copy['trade_date'] <= end_time)
        df_in_window = df_copy.loc[mask]

        return df_in_window

    except Exception as e:
        print(f"处理时间窗口时发生错误: {e}")
        print("--- 程序终止 ---")
        return None # 返回 None 表示操作失败
# 持有期内累计对数收益率计算
def data_preprocessing(df,test_window_start,test_window_end, holding_period=5):
    df = df.copy()
    # 截取回测期间的历史数据
    df_preprocess = cut_time_window(df, test_window_start, test_window_end)
    # 今日对数收益率计算
    df_preprocess['lndret'] = np.log(df_preprocess['close'] / df_preprocess['pre_close'])

    df_preprocess['holding_lndret'] = (
        df_preprocess.groupby('ts_code')['close']
        .transform(lambda x: np.log(x.shift(-holding_period) / x))
    )
    df_preprocess = df_preprocess.dropna(subset=['holding_lndret'])

    return df_preprocess
# 计算每层收益均值
def process_group_by_date(group, target_factor, layers):
    # 先按目标因子值对当前分组进行排序(从小到大升序排列，负向信号在前group_0,正向信号在后group_5)
    sorted_group = group.sort_values(by=target_factor, ascending=False).reset_index(drop=True)

    n = len(sorted_group)
    # 计算基础组大小和余数（按索引分组，防止.qcut()面对大量重复值时的无效分组问题）
    base_size = n // layers  # 每组的基本大小
    remainder = n % layers  # 无法整除后剩下的样本数

    quantiles = np.empty(n, dtype=int)
    current_idx = 0
    for i in range(layers):
        size_for_this_group = base_size + (1 if i < remainder else 0)
        quantiles[current_idx: current_idx + size_for_this_group] = i
        current_idx += size_for_this_group

    # 计算出的分组结果赋值给 DataFrame
    sorted_group['quantile'] = quantiles
    # 计算每组的平均收益率
    sorted_group['mean_lndret'] = sorted_group.groupby('quantile')['holding_lndret'].transform('mean')

    return sorted_group
# group累计对数收益计算
def spread_ret_cumsum_calculate(data, layers):
    # pivot_table数据透视
    Spread_ret = data.pivot_table(
        index='trade_date',
        columns='quantile',
        values='mean_lndret',
        aggfunc='mean' # 数值都一样，取第一个就好
    ).reset_index()
    # 多空收益计算
    Spread_ret['L-S'] = Spread_ret.iloc[:, 1] - Spread_ret.iloc[:, -1]  # group_0 - group_last
    # L-S多空收益率序列,用于后续的t检验
    t_test_series = np.array(Spread_ret['L-S'])
    # 计算各组累计收益 (只计算存在的列)
    quantiles = list(range(layers)) # e.g  layers=5,quantiles=[0,1,2,3,4]
    for q in quantiles:
        if q in Spread_ret.columns:  # 检查列是否存在
            Spread_ret[f'sum_ret_{q}'] = Spread_ret[q].cumsum()
            # print(f"已计算分位数 {q} 的累计收益")
        else:
            print(f"警告: 数据中缺少分位数 {q} 的数据")
    # 分层累计收益
    Spread_ret['sum_ret_L-S'] = Spread_ret['L-S'].cumsum()
    # 提取各组最后一天的总累计收益，存入字典
    final_cumulative_returns = {}
    for q in quantiles:
        sum_ret_col_name = f'sum_ret_{q}'
        if sum_ret_col_name in Spread_ret.columns:
            # 取该列的最后一行的值，即最后一天的累计收益
            final_cumulative_returns[q] = Spread_ret[sum_ret_col_name].iloc[-1]
        else:
            print(f"警告: 未能找到累计收益列 '{sum_ret_col_name}' 来提取最终收益。")

    return Spread_ret, t_test_series, final_cumulative_returns
# L-S收益率的t检验
def t_test_spread_ret(Series):
    # 只关心是否大于0，应是单尾检验
    t_stat, two_tailed_p_value = stats.ttest_1samp(Series, 0.0)
    ''' 
    --- 单尾检验逻辑 ---
    我们的原假设 H0: mean <= 0, 备择假设 H1: mean > 0
    1. 如果 t_stat < 0，说明样本均值小于0，肯定不支持 H1。
        此时 p-value 应为 1 - (two_tailed_p_value / 2)，肯定远大于 0.05。
    2. 如果 t_stat > 0，说明样本均值大于0，支持 H1 的方向。
        此时 p-value 应为 two_tailed_p_value / 2。
    只关心 t_stat > 0 且 p-value < 0.05 的情况。
    '''
    if t_stat > 0:
        one_tailed_p_value = two_tailed_p_value / 2
    else:
        # t_stat <= 0 时，我们不拒绝 H0 (mean <= 0)，即认为不显著大于0
        one_tailed_p_value = 1.0  # 可以写成 > 0.05 的任意值

    # 打印 t 统计量和 p-value
    print(f"T-statistic: {t_stat:.4f}")
    print(f"P-value (one-tailed): {one_tailed_p_value:.4f}")
    # 根据 p-value 和 t-stat 判断并打印结论
    alpha = 0.05
    if one_tailed_p_value < alpha and t_stat > 0:
        conclusion = "多空组合收益率序列显著为正"
    else:
        conclusion = "多空组合收益率序列不显著为正"
    print(f"结论: {conclusion}")
    # 返回检验结果
    return one_tailed_p_value < 0.05
# 分层回测结果绘图
def plot_multiple_return_metrics(dataframe, cumulative_returns, layers, target_factor):
    # 设置中文字体以支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 优先使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    # 创建2*1的布局，即上下两部分
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9))
    # 格式转换YYYY-MM-DD
    if not np.issubdtype(dataframe['trade_date'].dtype, np.datetime64):
        date_col = pd.to_datetime(dataframe['trade_date'].astype(str), format='%Y%m%d')
    else:
        date_col = dataframe['trade_date']

    colors = ['#228B22', '#32CD32', '#7CFC00', '#ADFF2F', '#FFFF00', '#FFD700', '#FFA500', '#FF8C00', '#FF6A6A','#DC143C']
    # 上方柱状图
    labels = []
    bars = []
    for idx, (key, value) in enumerate(cumulative_returns.items()):
        if key == 'L-S':  # 跳过多空组合
            continue
        labels.append(f'group_{key}')
        color_idx = int(key) % len(colors)
        bar = ax1.bar(labels[-1], value, color=colors[color_idx])
        bars.append(bar)
        # 在柱状图上方添加文本注释显示累计收益值
        ax1.text(bar[0].get_x() + bar[0].get_width() / 2,
                 bar[0].get_height(),
                 f'{value:.4f}',  # 格式化为小数点后两位
                 ha='center', va='bottom')  # 水平居中对齐，垂直底部对齐
    ax1.set_title('各组最终累计收益', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Cumulative Return', fontsize=12)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    ax1.grid(True, axis='y', alpha=0.3)  # 添加横向grid

    # 下方折线图
    n = len(dataframe) - 1  # 减1是为了作为索引
    indices = np.linspace(0, n, 8, dtype=int)
    xtick_labels = date_col.iloc[indices].dt.strftime('%Y%m%d')
    xtick_positions = indices  # x轴位置对应这些索引
    # 动态绘制每条线
    for q in range(layers):
        col_name = f'sum_ret_{q}'
        if col_name in dataframe.columns:
            color_idx = q % len(colors)
            ax2.plot(dataframe.index, dataframe[col_name], label=f'group_{q}', color=colors[color_idx], linewidth=1)
        else:
            print(f"警告: 数据中不存在列 '{col_name}'")
    # 单独画多空收益曲线
    excess_col = 'sum_ret_L-S'
    if excess_col in dataframe.columns:
        ax2.plot(dataframe.index, dataframe[excess_col], label='L-S', color='blue', linestyle='--', linewidth=2)
    else:
        print(f"警告: 数据中不存在列 '{excess_col}'")

    ax2.set_title(f"{target_factor}因子分组多空回测结果图", fontsize=14, fontweight='bold')  # 添加标题
    ax2.set_ylabel('Cumulative Return', fontsize=12)
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(xtick_positions)
    ax2.set_xticklabels(xtick_labels, rotation=45)

    plt.tight_layout()
    plt.show()

def run():
    """主执行函数"""
    print(f"开始分析因子: {traget_factor}")
    '''数据加载与处理'''
    df = data_loading(traget_factor)
    df = data_preprocessing(df, test_window_start, test_window_end)
    '''按日期分组处理'''
    df_layers_processed = df.groupby('trade_date')[['trade_date', 'ts_code', traget_factor, 'holding_lndret']].apply(
        lambda group: process_group_by_date(group, traget_factor, layers)
    ).reset_index(drop=True)
    '''分层累计收益计算'''
    df_layers_cumulative_ret, spread_ret_series, final_cumulative_returns = spread_ret_cumsum_calculate(
        df_layers_processed, layers)
    '''多空收益的t-检验'''
    t_test_spread_ret(spread_ret_series)
    '''结果绘图'''
    plot_multiple_return_metrics(df_layers_cumulative_ret, final_cumulative_returns, layers, traget_factor)

if __name__ == "__main__":
    run()










