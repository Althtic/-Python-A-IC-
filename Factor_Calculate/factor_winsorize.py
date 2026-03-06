import logging
import pandas as pd
from factor_distribution_plot import distribution_plot

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def winsorize_factor(df, lower_quantile=0.005, upper_quantile=0.995, inplace=False):
    """
    对因子值进行分位数截断（Winsorization）处理。

    参数:
    -----------
    df : pd.DataFrame
        输入数据框，必须包含 'ts_code', 'trade_date' 列，且因子值在最后一列。
        trade_date 格式示例：'20241220' (字符串或整数均可)
    lower_quantile : float
        下分位数阈值，默认 0.01 (即剔除最小的 1%)
    upper_quantile : float
        上分位数阈值，默认 0.99 (即剔除最大的 1%)
    inplace : bool
        是否原地修改，默认 False (返回新 DataFrame)

    返回:
    -----------
    pd.DataFrame
        处理后的数据框，因子列的值已被截断。
    """

    # 1. 校验必要列
    required_cols = ['ts_code', 'trade_date']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("输入 DataFrame 必须包含 'ts_code' 和 'trade_date' 列")

    # 2. 确定因子列 (最后一列)
    factor_col = df.columns[-1]
    logger.info(f"正在对因子列 '{factor_col}' 进行分位数缩尾处理...")

    # 3. 定义单组处理函数
    def _apply_winsorize(group):
        factor_series = group[factor_col]

        # 如果该截面有效数据少于 2 个，无法计算分位数，直接返回
        if factor_series.count() < 2:
            return group

        # 计算上下界
        # dropna() 确保计算分位数时不包含 NaN
        lower_bound = factor_series.quantile(lower_quantile)
        upper_bound = factor_series.quantile(upper_quantile)

        # 边界检查：防止因数据完全相同导致 lower > upper (极少见但可能)
        if lower_bound > upper_bound:
            lower_bound, upper_bound = upper_bound, lower_bound

        # 执行截断 (clip)
        # NaN 值会被 clip 自动保留为 NaN
        group[factor_col] = factor_series.clip(lower=lower_bound, upper=upper_bound)

        return group

    # 4. 执行处理
    if inplace:
        target_df = df
    else:
        target_df = df.copy()

    # 按 trade_date 分组应用
    # sort=False 保持原始日期顺序，提高效率
    result_df = target_df.groupby('trade_date', sort=False, group_keys=False).apply(_apply_winsorize)
    logger.info(f"处理完成。截断范围：[{lower_quantile * 100:.1f}%, {upper_quantile * 100:.1f}%]")
    return result_df


# ==========================================
# 使用示例 (你可以直接复制下方代码测试)
# ==========================================
if __name__ == "__main__":
    # 如果要使用测试案例，记得将distribution_plot.py文件中 last_col = df.columns[-1] 修改为 last_col = df.columns[-2]
    df_test = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101\alpha_45.csv')
    logger.info(df_test.head())
    distribution_plot(df_test)
    df_cleaned = winsorize_factor(df_test, lower_quantile=0.005, upper_quantile=0.995)
    distribution_plot(df_cleaned)
    logger.info("\n--- 处理后数据 (极端值已被拉回边界) ---")
    logger.info(df_cleaned)

    # 你可以将 df_cleaned 用于后续计算
    # result_df = calculate_alpha(df_cleaned)