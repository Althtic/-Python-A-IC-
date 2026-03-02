import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from statsmodels.stats.sandwich_covariance import cov_hac
from config_loader import traget_factor, test_window_start, test_window_end, test_period, ic_ma_period

# --- 设置 Pandas 显示选项 ---
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

# ─────────────────────────────────────────────────────────────────────────────
# 加载数据
# ─────────────────────────────────────────────────────────────────────────────
def data_loading(traget_factor):
    print(f"目标检测因子：{traget_factor}")
    base_directory = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101'
    alpha = traget_factor
    filename = f"{alpha}.csv"
    Input_path = base_directory + '\\' + filename
    df_loading = pd.read_csv(Input_path)
    cv_df = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv',
                        usecols=['trade_date', 'ts_code', 'circ_mv'])
    df_merge = df_loading.merge(cv_df,on=['trade_date','ts_code'], how='left')
    return df_merge
# ─────────────────────────────────────────────────────────────────────────────
# 截取时间窗
# ─────────────────────────────────────────────────────────────────────────────
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

        return None 
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
# 持有期对数收益率计算, 后续将分别按照t期因子值与t+1期收益率进行排序
# ─────────────────────────────────────────────────────────────────────────────
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
# ─────────────────────────────────────────────────────────────────────────────
# 中性化处理（市值 & 行业）
# ─────────────────────────────────────────────────────────────────────────────
def neutralize_factor_by_date(group, factor_col=traget_factor, cap_col='circ_mv', industry_col='industry_name'):
    """
    Factor ~ ln(MarketCap) + Industry_Dummies，返回残差作为中性化后的因子值
    """
    group = group.replace([np.inf, -np.inf], np.nan)
    valid_mask = group[factor_col].notna() & group[cap_col].notna() & group[industry_col].notna()

    if valid_mask.sum() < 10:
        group['factor_neutralized'] = np.nan
        return group
    data_valid = group[valid_mask].copy()

    y = data_valid[factor_col].values
    log_cap = np.log(data_valid[cap_col]).values.reshape(-1, 1)

    # 处理自变量 X2: 行业哑变量
    industry_series = data_valid[industry_col].fillna('Unknown')
    # 生成哑变量矩阵（True & False）
    industry_dummies = pd.get_dummies(industry_series, prefix='ind', dummy_na=False)
    # 合并 X (市值 + 行业)将 log_cap 和 dummies 的 values 拼在一起,使用 np.hstack 效率更高，
    X = np.hstack([log_cap, industry_dummies.values])

    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions
    # 提取残差
    group['factor_neutralized'] = np.nan
    group.loc[valid_mask, 'factor_neutralized'] = residuals
    return group
# ─────────────────────────────────────────────────────────────────────────────
# 因子与未来累计收益排序
# ─────────────────────────────────────────────────────────────────────────────
def factor_cumuret_rank(df, test_period):
    # 根据因子值大小排序(升序)，用于计算斯皮尔曼相关系数(RankIC)
    df_factor_ranking = df
    # 按照过滤后得到的中性化因子值进行截面排序
    df_factor_ranking['factor_rank'] = df_factor_ranking.groupby('trade_date')['factor_neutralized'].rank(pct=True)
    # 未来累计收益排序（升序），用于计算斯皮尔曼相关系数（RankIC）
    # rolling()的计算只能面向“过去”，所以需要提前翻转数据（日期翻转），使得rolling()函数可以直接计算未来收益
    df_factor_ranking = df_factor_ranking.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
    # 计算未来test_period个交易日的收益累计和
    df_factor_ranking['period_cumu_lndret'] = df_factor_ranking.groupby('ts_code')['lndret'].transform(
        lambda x: x.shift(1).rolling(window = test_period).sum()
    )
    df_cumu_dret_rank = df_factor_ranking.dropna(subset=['period_cumu_lndret'])
    # 数据顺序恢复原状(升序)
    df_cumu_dret_rank = df_cumu_dret_rank.sort_values(by=['trade_date', 'ts_code'], ascending=[True, True])
    # 针对累计对数收益率rank排序
    df_cumu_dret_rank['cumu_lndret_rank'] = df_cumu_dret_rank.groupby('trade_date')['period_cumu_lndret'].rank(pct=True)
    df_factor_cumuret_rank = df_cumu_dret_rank
    return df_factor_cumuret_rank
# ─────────────────────────────────────────────────────────────────────────────
# RankIC, ICmean, ICIR及相关评价指标计算
# ─────────────────────────────────────────────────────────────────────────────
def IC_calculate(df, ic_ma_period):
    df_ic_processing = df
    df_ic_processing['trade_date'] = pd.to_datetime(df_ic_processing['trade_date'])
    # 斯皮尔曼相关系数(Rank_IC)值计算
    rank_ic_series = df_ic_processing.groupby('trade_date').apply(
        lambda x: x['factor_rank'].corr(x['cumu_lndret_rank'])
    )
    # rank_ic有效性检验（单尾 > 0.05）
    ic_ttest_sample(rank_ic_series, threshold=0.05)
    # IC累计和序列
    cumulative_ic_series = rank_ic_series.cumsum()
    # IC均值计算(rolling)
    ic_ma_series = rank_ic_series.rolling(window=ic_ma_period, min_periods=1).mean()
    # ICIR = IC.mean / IC.std
    ICIR_value = rank_ic_series.mean() / rank_ic_series.std()

    # 创建临时DataFrame存储计算结果
    temp_df = pd.DataFrame({
        'trade_date': rank_ic_series.index,
        'Rank_IC': rank_ic_series.values,
        'Cumulative_IC': cumulative_ic_series.values,
        'IC_MA': ic_ma_series.values,
        'ICIR': [ICIR_value] * len(rank_ic_series),
    })
    # 计算每月的 Rank_IC 均值
    temp_df['year_month'] = temp_df['trade_date'].dt.to_period('M')  # e.g., '2024-08'
    monthly_means = temp_df.groupby('year_month')['Rank_IC'].mean()
    monthly_ICIR = temp_df.groupby('year_month').apply(
        lambda x: x['Rank_IC'].mean()/x['Rank_IC'].std()
    )
    # 使用 map 函数，将每一行的 'year_month' 映射到对应的月度均值
    temp_df['Monthly_Rank_IC_Mean'] = temp_df['year_month'].map(monthly_means)
    temp_df['Monthly_ICIR'] = temp_df['year_month'].map(monthly_ICIR)

    # 计算每年的 Rank_IC 均值，原理同上
    temp_df['year'] = temp_df['trade_date'].dt.to_period('Y')  # e.g., '2024'
    yearly_means = temp_df.groupby('year')['Rank_IC'].mean()
    yearly_ICIR = temp_df.groupby('year').apply(
        lambda x: x['Rank_IC'].mean()/x['Rank_IC'].std()
    )
    temp_df['Yearly_Rank_IC_Mean'] = temp_df['year'].map(yearly_means)
    temp_df['Yearly_ICIR'] = temp_df['year'].map(yearly_ICIR)
    # final_df = temp_df.drop(columns=['year_month','year'])
    # return final_df
    return temp_df
# ─────────────────────────────────────────────────────────────────────────────
# IC序列的t检验（双尾检验), 需同样进行Newey-West调整自相关
# ─────────────────────────────────────────────────────────────────────────────
def ic_ttest_sample(rank_ic_series, threshold=0.05, alpha=0.05):
    print("=" * 70)
    print("IC 统计显著性检验结果 (Newey-West调整)")
    print("=" * 70)
    
    # 数据预处理
    if hasattr(rank_ic_series, 'values'):
        rank_ic_series = rank_ic_series.values
    rank_ic_series = np.array(rank_ic_series, dtype=float)
    rank_ic_series = rank_ic_series[~np.isnan(rank_ic_series)]
    n = len(rank_ic_series)
    
    if n < 5:
        print("样本量不足 (N<5)，无法进行可靠的 Newey-West 调整")
        return None

    # ==================== 使用 Newey-West 计算 SE 和 T 统计量 ====================
    # 构建截距模型 (Y = mean + error)
    X = np.ones((n, 1))
    model = sm.OLS(rank_ic_series, X)
    results = model.fit()
    # 计算 HAC (Newey-West) 协方差矩阵
    cov_matrix = cov_hac(results, nlags=None)
    # 提取 NW 标准误 (SE)
    se_nw = np.sqrt(cov_matrix[0, 0])
    # 计算均值和 IR
    sample_mean = results.params[0] # 等同于 np.mean()
    sample_std = rank_ic_series.std(ddof=1) # 描述性统计仍用普通标准差
    # 计算 NW 调整后的 T 统计量 (针对 H0: mean = 0)
    if se_nw < 1e-12:
        t_stat_0 = 0.0
    else:
        t_stat_0 = sample_mean / se_nw
        
    # 计算双尾 P 值 (基于 t 分布，自由度 n-1)
    # 虽然 NW 是渐近正态的，但在小样本下用 t 分布更保守
    p_value_0 = 2 * (1 - stats.t.cdf(abs(t_stat_0), df=n-1))
    
    # 估算使用的 Lags 用于打印
    lags_used = int(np.floor(4 * (n / 100)**(2/9)))
    lags_used = max(1, min(lags_used, n-2))

    # ==================== 描述性统计 ====================
    # 打印 NW 调整后的 SE，以反映真实波动
    print(f"【1. 描述性统计】")
    print(f"   样本均值 (Mean IC):   {sample_mean:.4f}")
    print(f"   样本标准差 (Std IC):  {sample_std:.4f}")
    print(f"   标准误 (SE, NW 调整):  {se_nw:.4f}  (Lags={lags_used})")
    print(f"   观测样本数 (N):       {n}")

    # ==================== 检验 IC 是否显著不为 0 ====================
    # 使用上面计算的 t_stat_0 和 p_value_0
    t_critical_0 = stats.t.ppf(1 - alpha / 2, df=n - 1)
    significant_0 = abs(t_stat_0) > t_critical_0

    print(f"【2. IC 均值是否显著不为 0? ]")
    print(f"   原假设 H0:  mean = 0")
    print(f"   备择假设 H1:  mean ≠ 0")
    print(f"   t-statistic (NW):     {t_stat_0:.4f}")
    print(f"   |t|:                  {abs(t_stat_0):.4f}")
    print(f"   t 临界值 (α={alpha}):    ±{t_critical_0:.4f}")
    print(f"   p-value (双尾):       {p_value_0:.6f}")
    print("-" * 70)

    if significant_0:
        if t_stat_0 > 0:
            print(f"   |t| = {abs(t_stat_0):.4f} > {t_critical_0:.4f}")
            print(f"   结论：IC 均值显著为正（因子正向有效）")
            direction_0 = "positive"
        else:
            print(f"   |t| = {abs(t_stat_0):.4f} > {t_critical_0:.4f}")
            print(f"   结论：IC 均值显著为负（因子反向有效）")
            direction_0 = "negative"
    else:
        print(f"   |t| = {abs(t_stat_0):.4f} < {t_critical_0:.4f}")
        print(f"   结论：IC 均值不显著异于 0（因子无效）")
        direction_0 = "not_significant"

    # ==================== 检验 IC 是否显著大于 threshold 或小于 -threshold ====================
    # T 统计量 = (Sample_Mean - Hypothesized_Mean) / SE_NW
    # SE_NW 是在 H0: mean=0 下估计的长程方差，通常也用于局部假设检验
    
    # 3.1 检验是否显著大于 threshold (H0: mean <= threshold)
    # t_stat = (mean - threshold) / se_nw
    if se_nw < 1e-12:
        t_stat_upper = 0.0
    else:
        t_stat_upper = (sample_mean - threshold) / se_nw
        
    # 计算双尾 p 值以便转换
    _, p_value_upper_two_dummy = stats.ttest_1samp(rank_ic_series, threshold) # 这个 p 值不准，仅为了占位或不用
    # 手动计算基于 NW SE 的双尾 p 值
    p_val_two_tail_upper = 2 * (1 - stats.t.cdf(abs(t_stat_upper), df=n-1))
    
    # 单尾检验 (H1: mean > threshold)
    if t_stat_upper > 0:
        p_value_upper = p_val_two_tail_upper / 2
    else:
        p_value_upper = 1.0
    significant_upper = p_value_upper < alpha

    # 3.2 检验是否显著小于 -threshold (H0: mean >= -threshold)
    if se_nw < 1e-12:
        t_stat_lower = 0.0
    else:
        t_stat_lower = (sample_mean - (-threshold)) / se_nw
        
    p_val_two_tail_lower = 2 * (1 - stats.t.cdf(abs(t_stat_lower), df=n-1))
    
    # 单尾检验 (H1: mean < -threshold)
    if t_stat_lower < 0:
        p_value_lower = p_val_two_tail_lower / 2
    else:
        p_value_lower = 1.0
    significant_lower = p_value_lower < alpha

    print(f"【3. IC 均值是否显著大于 {threshold} 或小于 {-threshold}？(NW 调整)]")
    print(f"   阈值范围：[-{threshold}, {threshold}]")
    print("-" * 70)

    # 检验是否显著大于 threshold
    print(f"   (a) 检验是否显著大于 {threshold}:")
    print(f"       原假设 H0:  mean ≤ {threshold}")
    print(f"       备择假设 H1:  mean > {threshold}")
    print(f"       t-statistic (NW): {t_stat_upper:.4f}")
    print(f"       p-value (单尾):   {p_value_upper:.6f}")
    if significant_upper:
        print(f"       结论：IC 均值显著大于 {threshold}")
    else:
        print(f"       结论：IC 均值不显著大于 {threshold}")

    # 检验是否显著小于 -threshold
    print(f"   (b) 检验是否显著小于 {-threshold}:")
    print(f"       原假设 H0:  mean ≥ {-threshold}")
    print(f"       备择假设 H1:  mean < {-threshold}")
    print(f"       t-statistic (NW): {t_stat_lower:.4f}")
    print(f"       p-value (单尾):   {p_value_lower:.6f}")
    if significant_lower:
        print(f"       结论：IC 均值显著小于 {-threshold}")
    else:
        print(f"       结论：IC 均值不显著小于 {-threshold}")

    # 最终结论
    print("-" * 70)
    print(f"   【综合结论】")
    if significant_upper:
        print(f"   IC 均值显著大于 {threshold}（强因子）")
        threshold_conclusion = "greater_than_threshold"
    elif significant_lower:
        print(f"   IC 均值显著小于 {-threshold}（强反向因子）")
        threshold_conclusion = "less_than_negative_threshold"
    else:
        print(f"   IC 均值在 [{-threshold}, {threshold}] 范围内（普通因子或无效）")
        threshold_conclusion = "within_threshold"
    print("=" * 70)
# ─────────────────────────────────────────────────────────────────────────────
# 嵌套字典，各月份对应的各年度指标，单独独立出来的，便于可视化的前置处理
# ─────────────────────────────────────────────────────────────────────────────
def monthly_processing(df):
    # df = df.copy()
    # 需要处理的列
    target_columns = ['year_month', 'year', 'Monthly_Rank_IC_Mean','Monthly_ICIR']
    df = df[target_columns]
    # 确保 'year_month' 列是字符串类型
    df['year_month'] = df['year_month'].astype(str)
    df['month'] = df['year_month'].str.split('-').str[1].astype(int)

    # 构建嵌套字典
    result_nested_dict_mean = {}
    # 构建 Monthly_ICIR 的嵌套字典
    result_nested_dict_icir = {}

    for index, row in df.iterrows():
        month = row['month']  # month 已经是 int
        year = row['year']    # year 已经是 int
        mean_value = row['Monthly_Rank_IC_Mean']
        icir_value = row['Monthly_ICIR']

        # 为 Monthly_Rank_IC_Mean 构建字典
        if month not in result_nested_dict_mean:
            result_nested_dict_mean[month] = {}
        result_nested_dict_mean[month][year] = mean_value

        # 为 Monthly_ICIR 构建字典
        if month not in result_nested_dict_icir:
            result_nested_dict_icir[month] = {}
        result_nested_dict_icir[month][year] = icir_value

    return result_nested_dict_mean, result_nested_dict_icir
# ─────────────────────────────────────────────────────────────────────────────
# 可视化Ⅰ:IC & IC_mean & cumulative_IC
# ─────────────────────────────────────────────────────────────────────────────
def plot_validation_analysis(df):
    # 提取数据系列
    rank_ic_series = df.set_index('trade_date')['Rank_IC']
    cumulative_ic_series = df.set_index('trade_date')['Cumulative_IC']
    ic_ma_period = df.set_index('trade_date')['IC_MA']
    icir = df['ICIR'].iloc[0] # 提取 ICIR 标量值

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True) # sharex 共享x轴，简化处理

    # 第一个子图
    ax1.plot(rank_ic_series.index, rank_ic_series.values, label='Rank IC', color='blue', linewidth=0.8)
    ax1.plot(rank_ic_series.index, ic_ma_period, label='Rank IC 21MA', color='red', linewidth=1.2)
    ax1.set_title(f'IC(MA) Analysis (ICIR: {icir:.2f})') # 将ICIR放在标题中，更简洁
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # 第二个子图
    ax2.plot(cumulative_ic_series.index, cumulative_ic_series.values, label='Cumulative IC', color='green', linewidth=1.0)
    ax2.set_title('Cumulative IC')
    ax2.set_xlabel('Date') # X轴标签只需在最下方设置一次
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    # 自动格式化日期横坐标，无需手动设置刻度
    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# 可视化Ⅱ:Yearly IC_mean & ICIR
# ─────────────────────────────────────────────────────────────────────────────
def plot_validation_yearly_series_bar(df):
    yearly_data = df.sort_values(by='year').copy()
    # 提取 'year' 列的唯一值，并转换为字符串列表
    years = yearly_data['year'].unique().astype(str).tolist()
    # 提取 Yearly_ICIR 和 Rank_IC_Mean 值
    icir_values = yearly_data['Yearly_ICIR'].unique().tolist()
    rank_ic_mean_values = yearly_data['Yearly_Rank_IC_Mean'].unique().tolist()
    
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    # --- 左图：ICIR ---
    bars0 = axs[0].bar(years, icir_values)
    axs[0].set_title('ICIR')
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel('ICIR')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].grid(axis='y', alpha=0.5)
    # 添加数值标签 (保留2位小数)
    for bar in bars0:
        height = bar.get_height()
        axs[0].text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}', 
                    ha='center', va='bottom', fontsize=10)

    # --- 右图：IC Mean ---
    bars1 = axs[1].bar(years, rank_ic_mean_values)
    axs[1].set_title('IC Mean')
    axs[1].set_xlabel('Year')
    axs[1].set_ylabel('IC Mean')
    axs[1].tick_params(axis='x', rotation=45)
    axs[1].grid(axis='y', alpha=0.5)
    # 添加数值标签 (保留4位小数，因为IC通常较小)
    for bar in bars1:
        height = bar.get_height()
        axs[1].text(bar.get_x() + bar.get_width()/2, height, f'{height:.4f}', 
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# 可视化Ⅲ:Monthly & IC_mean & ICIR
# ─────────────────────────────────────────────────────────────────────────────
def plot_validation_monthly_series_bar(dict1, dict2):
    result_nested_dict_mean = dict1
    result_nested_dict_icir = dict2
    months = list(range(1, 13))
    years = set()

    for month_year_values in [result_nested_dict_mean, result_nested_dict_icir]:
        for month, year_values in month_year_values.items():
            years.update(year_values.keys())

    years = sorted(years)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    for idx, (title, data_dict) in enumerate(
            [("Rank IC Mean", result_nested_dict_mean), ("ICIR", result_nested_dict_icir)]):
        ax = axs[idx]
        for i, year in enumerate(years):
            values = [data_dict.get(month, {}).get(year, 0) for month in months]
            ax.bar([m + i * 0.1 for m in months], values, width=0.1, label=str(year))
        ax.set_xlabel('Month')
        ax.set_ylabel('Value')
        ax.set_title(title)
        ax.set_xticks([m + 0.1 * (len(years) - 1) / 2 for m in months])
        ax.set_xticklabels(months)
        ax.legend()
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
# ─────────────────────────────────────────────────────────────────────────────
# 主运行函数
# ─────────────────────────────────────────────────────────────────────────────
def run():

    df = data_loading(traget_factor)
    '''数据加载与预处理'''
    df_initial_preprocess = data_preprocessing(df, test_window_start, test_window_end)
    '''因子中性化处理'''
    df_neutralized_preprocess = df_initial_preprocess.groupby('trade_date', group_keys=False).apply(neutralize_factor_by_date, factor_col=traget_factor)
    '''计算因子与未来累计收益排序'''
    df_factor_rank_processed = factor_cumuret_rank(df_neutralized_preprocess, traget_factor, test_period)
    '''因子有效性相关评价指标'''
    df_validation_features = IC_calculate(df_factor_rank_processed, ic_ma_period)
    # print(df_validation_features.head(3))
    '''返回嵌套字典，Rank IC mean & ICIR mean (monthly)可视化的前置操作，单独分出来'''
    result_nested_dict_mean, result_nested_dict_icir = monthly_processing(df_validation_features)
    '''可视化：Rank IC 21mean & Cumulative Rank IC'''
    plot_validation_analysis(df_validation_features)
    '''可视化：Rank IC mean & ICIR mean (yearly)'''
    plot_validation_yearly_series_bar(df_validation_features)
    '''可视化：Rank IC mean & ICIR mean (monthly)'''
    plot_validation_monthly_series_bar(result_nested_dict_mean, result_nested_dict_icir)

if __name__ == "__main__":
    '''
            IC_Mean 衡量的是因子的预测强度。但是，单独看 IC_Mean 是有误导性的，必须结合 ICIR 看。
            绝对值 > 0.05：通常被认为是有意义的。
            绝对值 > 0.1：非常强的预测能力（这种因子很少见，通常很快会被市场消化）。
            ICIR 的绝对值大于 0.5 通常被视为具备一定预测能力，而大于 2 则被视为统计显著的优质因子
    '''
    run()











