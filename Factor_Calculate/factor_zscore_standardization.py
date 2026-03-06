import pandas as pd

def zscore_transform(df):
    factor_col = df.columns[-1]

    # 2. 定义标准化函数 (您提供的开头)
    def zscore_transform(group):
        mean_val = group.mean()
        std_val = group.std(ddof=0)  # ddof=0 表示总体标准差，量化截面常用；若需样本标准差改为 1

        # 处理标准差为 0 的情况（避免除以零错误）
        if std_val == 0:
            return pd.Series(0.0, index=group.index)

        return (group - mean_val) / std_val

    # 3. 执行分组、计算并拼接
    # transform 会自动对齐索引，将结果返回为与原 df 长度一致的 Series
    new_col_name = 'alpha_zscore'
    df[new_col_name] = df.groupby('trade_date')[factor_col].transform(zscore_transform)
    # 删除掉原始列
    df.drop(columns=[factor_col], inplace=True)
    # 修改列名并传回
    df.rename(columns={'alpha_zscore': factor_col}, inplace=True)
    df[factor_col] = df[factor_col].round(6)

    return df

