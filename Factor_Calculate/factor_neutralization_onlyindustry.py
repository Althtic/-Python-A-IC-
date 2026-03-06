import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)


def neutralize_factor_by_industry(df,
                                  target_factor,
                                  industry_col='industry_name',
                                  date_col='trade_date',
                                  id_col='ts_code'):
    """
    因子行业中性化函数（已移除市值中性化部分）：
    1. 按日期 (trade_date) 分组
    2. 对每个截面进行回归: Factor ~ Industry_Dummies
    3. 返回残差作为中性化后的因子值

    参数:
        df: 输入 DataFrame，必须包含 target_factor, industry_col, date_col, id_col
        target_factor: 需要中性化的因子列名
        industry_col: 行业列名
        date_col: 日期列名
        id_col: 股票代码列名

    返回:
        包含原始数据及新增中性化因子列的 DataFrame
        (原因子列重命名为 'raw_factor', 中性化后列名为 target_factor)
    """

    # 截面中性化逻辑函数
    def process_single_date(group):
        # 初始化结果列为 NaN
        group['factor_neutralized'] = np.nan
        group_clean = group.copy()

        # 找出所有数值类型的列 (用于处理 inf)
        numeric_cols = group_clean.select_dtypes(include=[np.number]).columns

        # 数据清洗：替换 inf 为 NaN
        if len(numeric_cols) > 0:
            group_clean[numeric_cols] = group_clean[numeric_cols].mask(
                np.isinf(group_clean[numeric_cols]),
                np.nan
            )

        # 构建有效掩码：因子和行业都不能为空
        # [修改点] 移除了对市值列的非空检查
        valid_mask = (
                group_clean[target_factor].notna() &
                group_clean[industry_col].notna()
        )

        # 如果有效样本太少，跳过回归，直接返回全 NaN
        if valid_mask.sum() < 10:
            logger.warning(f"日期 {group[date_col].iloc[0]} 有效样本不足 10 个，跳过中性化")
            return group

        data_valid = group_clean[valid_mask].copy()

        # 准备 Y (因子值)
        y = data_valid[target_factor].values

        # 准备 X: 仅包含行业哑变量
        # fillna('Unknown') 防止行业列为空导致 get_dummies 出错
        industry_series = data_valid[industry_col].fillna('Unknown')

        # 生成哑变量 (drop_first=False 保留所有行业，让截距项吸收整体均值，或者 drop_first=True 避免多重共线性)
        # 在纯行业中性化中，通常 drop_first=False 配合 fit_intercept=True (默认) 是安全的
        industry_dummies = pd.get_dummies(industry_series, prefix='ind', drop_first=False)

        # 构建回归矩阵 X
        # [修改点] 不再 hstack 市值数据，直接使用行业哑变量
        X = industry_dummies.values

        # 执行线性回归
        model = LinearRegression()
        model.fit(X, y)

        # 计算残差 (Residuals = Y - Predicted_Y)
        predictions = model.predict(X)
        residuals = y - predictions

        # 将残差填回原 group 的对应位置
        group.loc[valid_mask, 'factor_neutralized'] = residuals

        return group

    # --- 步骤：按日期分组应用 ---
    logger.info(f"正在进行纯行业截面中性化处理 (目标因子：{target_factor})...")

    # groupby.apply 处理每个日期的子 DataFrame
    # 注意：df_work 应该是传入的 df，这里修正变量名错误 (原代码中使用了未定义的 df_work)
    result_df = df.groupby(date_col, group_keys=False).apply(process_single_date)

    # 保留初始因子值，重命名为 'raw_factor'
    # 检查列是否存在以防重复运行报错
    if target_factor in result_df.columns:
        result_df.rename(columns={target_factor: 'raw_factor'}, inplace=True)
        result_df['raw_factor'] = result_df['raw_factor'].round(6)

    # 将残差列重命名为原因子名，作为最终使用的中性化因子
    if 'factor_neutralized' in result_df.columns:
        result_df.rename(columns={'factor_neutralized': target_factor}, inplace=True)

    return result_df


# ==========================================
# 使用示例
# ==========================================
if __name__ == "__main__":
    # 请确保路径正确
    file_path = r'/Factors/smb.csv'

    try:
        data = pd.read_csv(file_path)

        # 调用函数 (仅行业中性化)
        final_data = neutralize_factor_by_industry(data, target_factor='smb')

        # 查看结果
        print(final_data.head())
        print(f"数据形状：{final_data.shape}")
        print(f"列名：{final_data.columns.tolist()}")

    except FileNotFoundError:
        logger.error(f"文件未找到：{file_path}")
    except Exception as e:
        logger.error(f"发生错误：{e}")