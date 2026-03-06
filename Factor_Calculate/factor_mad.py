import logging
import pandas as pd
import numpy as np
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


def mad_outlier_remove(df, n_sigma=5.0, inplace=False):

    # 校验必要列
    required_cols = ['ts_code', 'trade_date']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("输入 DataFrame 必须包含 'ts_code' 和 'trade_date' 列")

    # 确定因子列 (最后一列)
    factor_col = df.columns[-1]
    logger.info(f"正在对因子列 '{factor_col}' 进行 MAD 去极值处理 (阈值: {n_sigma} * MAD)...")

    # 定义单组处理函数
    def _apply_mad(group):
        factor_series = group[factor_col]

        # 获取非 NaN 值用于计算统计量
        valid_series = factor_series.dropna()

        # 如果该截面有效数据少于 2 个，无法计算 MAD，直接返回
        if len(valid_series) < 2:
            return group

        # 计算中位数
        median_val = valid_series.median()

        # 计算绝对中位差 (MAD)
        # MAD = Median(|X_i - Median(X)|)
        mad_val = (valid_series - median_val).abs().median()

        # 边界检查：防止 MAD 为 0 (当所有有效值都相同时)
        # 如果 MAD 为 0，说明数据没有离散度，无需处理，直接返回
        if mad_val == 0:
            logger.debug(f"日期 {group['trade_date'].iloc[0]} 的 MAD 为 0，跳过截断。")
            return group

        # 计算上下界
        lower_bound = median_val - n_sigma * mad_val
        upper_bound = median_val + n_sigma * mad_val

        # 执行截断 (clip)
        # NaN 值会被 clip 自动保留为 NaN
        group[factor_col] = factor_series.clip(lower=lower_bound, upper=upper_bound)

        return group

    # 执行处理
    if inplace:
        target_df = df
    else:
        target_df = df.copy()

    # 按 trade_date 分组应用
    # sort=False 保持原始日期顺序，提高效率
    result_df = target_df.groupby('trade_date', sort=False, group_keys=False).apply(_apply_mad)

    logger.info(f"MAD 处理完成。阈值倍数: {n_sigma}, 基于中位数和绝对中位差动态调整边界")
    return result_df

# ==========================================
# 使用示例 (你可以直接复制下方代码测试)
# ==========================================
if __name__ == "__main__":
    # 路径需根据实际情况修改
    file_path = r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\WorldQuant_Alpha101\alpha_45.csv'

    try:
        df_test = pd.read_csv(file_path)
        logger.info("--- 原始数据预览 ---")
        logger.info(df_test.head())

        # 绘制处理前分布
        distribution_plot(df_test)

        # 执行 MAD 去极值 (默认 3 倍 MAD)
        # 你也可以尝试修改 n_sigma=5.0 来看看更宽松的效果
        df_cleaned = mad_outlier_remove(df_test, n_sigma=3.0)

        # 绘制处理后分布
        distribution_plot(df_cleaned)

        logger.info("\n--- 处理后数据 (极端值已被拉回 MAD 边界) ---")
        logger.info(df_cleaned)

        # 你可以将 df_cleaned 用于后续计算
        # result_df = calculate_alpha(df_cleaned)

    except FileNotFoundError:
        logger.error(f"未找到文件: {file_path}, 请检查路径。")
    except Exception as e:
        logger.error(f"发生错误: {e}")