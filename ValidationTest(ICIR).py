import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端，matplotlib得以在Pycharm中正常显示
import matplotlib.pyplot as plt

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

# 加载数据
def data_loading(traget_factor):
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df_loading = pd.read_csv(Input_path)
    return df_loading
# 截取时间窗
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
        df_in_window = df_in_window.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
        return df_in_window

    except Exception as e:
        print(f"处理时间窗口时发生错误: {e}")
        print("--- 程序终止 ---")
        return None # 返回 None 表示操作失败
# 数据处理
def data_preprocessing(df,test_window_start,test_window_end):
    df = df.copy()
    # 截取回测期间的历史数据
    df_preprocess = cut_time_window(df, test_window_start, test_window_end)
    # 今日对数收益率计算
    df_preprocess['lndret'] = np.log(df_preprocess['close'] / df_preprocess['pre_close'])
    # 持有至第二个交易日的对数收益率holding1D_lndret
    df_preprocess['holding1D_lndret'] = df_preprocess.groupby('ts_code')['lndret'].shift(-1)
    df_preprocess = df_preprocess.dropna()
    return df_preprocess
# 因子与未来累计收益排序
def factor_cumuret_rank(df, target_factor, test_period):
    # 1.根据因子值大小排序(升序)，用于计算斯皮尔曼相关系数(RankIC)
    df_factor_ranking = df.copy()
    df_factor_ranking['factor_rank'] = df_factor_ranking.groupby('trade_date')[target_factor].rank(pct=True)
    # 2.未来累计收益排序（升序），用于计算斯皮尔曼相关系数（RankIC）'''
    # rolling()的计算只能面向“过去”，所以需要提前翻转数据（日期翻转），使得rolling()函数可以直接计算未来收益
    df_factor_ranking = df_factor_ranking.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
    # 计算未来test_period个交易日的收益累计和
    df_factor_ranking['period_cumu_lndret'] = df_factor_ranking.groupby('ts_code')['lndret'].transform(
        lambda x: x.shift(1).rolling(window = test_period).sum()
    )
    # drop掉NaN值
    df_cumu_dret_rank = df_factor_ranking.dropna(subset=['period_cumu_lndret'])
    # 数据顺序恢复原状(升序)
    df_cumu_dret_rank = df_cumu_dret_rank.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
    # 针对累计对数收益率rank排序
    df_cumu_dret_rank['cumu_lndret_rank'] = df_cumu_dret_rank.groupby('trade_date')['period_cumu_lndret'].rank(pct=True)
    df_factor_cumuret_rank = df_cumu_dret_rank
    return df_factor_cumuret_rank
# IC及相关评价指标计算
def IC_calculate(df, ic_ma_period):
    df_ic_processing = df.copy()
    # 斯皮尔曼相关系数(Rank_IC)值计算
    rank_ic_series = df_ic_processing.groupby('trade_date').apply(
        lambda x: x['factor_rank'].corr(x['cumu_lndret_rank'])
    )
    # IC累计和序列
    cumulative_ic_series = rank_ic_series.cumsum()
    # IC均值计算(rolling)
    ic_ma_period = rank_ic_series.rolling(window=ic_ma_period, min_periods=1).mean()
    # ICIR = IC.mean / IC.std
    ICIR = rank_ic_series.mean() / rank_ic_series.std()
    return rank_ic_series, cumulative_ic_series, ic_ma_period, ICIR
# 可视化绘图
def plot_validation_analysis(rank_ic_series, cumulative_ic_series, ic_ma_period, icir):
    # 创建一个新的figure和axes对象
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # 第一个子图：rank_ic_series与ic_ma_period
    ax1.plot(rank_ic_series.index, rank_ic_series.values, label='Rank IC', color='blue', linewidth=0.8)
    ax1.plot(rank_ic_series.index, ic_ma_period, label=f'Rank IC 21MA', color='red', linewidth=1.2)
    # 设置第一个子图的标题和标签
    ax1.set_title('IC(MA) Analysis')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('IC Value')
    ax1.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    ax1.legend()
    # 添加ICIR到图例中
    textstr = f'ICIR: {icir:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
             verticalalignment='top', bbox=props)
    # 横坐标处理
    date_col = pd.to_datetime(rank_ic_series.index)
    # 生成 8 个均匀分布的索引
    n = len(date_col) - 1  # 减1是为了作为索引
    indices = np.linspace(0, n, 8, dtype=int)
    # 根据索引获取对应日期标签 (格式化为 '20140215')
    # 修正：DatetimeIndex 可以直接索引，不需要 .iloc
    xtick_labels = date_col[indices].strftime('%Y%m%d')
    xtick_positions = indices  # x轴位置对应这些索引
    # 设置刻度位置和标签
    ax1.set_xticks(xtick_positions)
    ax1.set_xticklabels(xtick_labels, rotation=45, ha="right", fontsize=8)

    # 第二个子图：cumulative_ic_series
    ax2.plot(cumulative_ic_series.index, cumulative_ic_series.values, label='Cumulative IC', color='green',
             linewidth=1.0)
    # 设置第二个子图的标题和标签
    ax2.set_title('Cumulative IC Analysis')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Cumulative IC Value')
    ax2.grid(True, linestyle='--', alpha=0.6)  # 添加网格线
    ax2.legend(loc='best')  # 选择最佳位置放置图例
    # 设置横坐标显示
    date_col_cumsum = pd.to_datetime(cumulative_ic_series.index)
    n_cumsum = len(date_col_cumsum) - 1
    indices_cumsum = np.linspace(0, n_cumsum, 8, dtype=int)
    # 修正：DatetimeIndex 可以直接索引，不需要 .iloc
    xtick_labels_cumsum = date_col_cumsum[indices_cumsum].strftime('%Y%m%d')
    xtick_positions_cumsum = indices_cumsum
    # 设置刻度位置和标签
    ax2.set_xticks(xtick_positions_cumsum)
    ax2.set_xticklabels(xtick_labels_cumsum, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    plt.show()

'''
        ICIR 的绝对值大于 0.5 通常被视为具备一定预测能力，而大于 2 则被视为统计显著的优质因子
        IC_Mean 衡量的是因子的预测强度。但是，单独看 IC_Mean 是有误导性的，必须结合 ICIR 看。
        绝对值 > 0.05：通常被认为是有意义的。
        绝对值 > 0.1：非常强的预测能力（这种因子很少见，通常很快会被市场消化）。
'''

if __name__ == "__main__":

    traget_factor = 'alpha_07'  # 分层多空检验的目标因子名称
    window_start = '20200601'
    window_end = '20241201'
    test_period = 21  # 对未来多少个交易日的累计收益率进行IC检测   "1个月=21个交易日"
    ic_ma_period = 21  # IC_mean计算窗口长度

    '''数据加载与预处理'''
    df = data_loading(traget_factor)
    df_initial_preprocess = data_preprocessing(df, window_start, window_end)
    # print(df_initial_preprocess)
    '''计算因子与未来累计收益排序'''
    df_factor_rank_processed = factor_cumuret_rank(df_initial_preprocess, traget_factor, test_period)
    '''因子有效性相关评价指标'''
    rank_ic_series, cumulative_ic_series, ic_ma_period, icir = IC_calculate(df_factor_rank_processed, ic_ma_period)
    # print(rank_ic_series)
    '''绘图可视化'''
    plot_validation_analysis(rank_ic_series, cumulative_ic_series, ic_ma_period, icir)










