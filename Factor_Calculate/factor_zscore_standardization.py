import numpy as np
import pandas as pd

def zscore_transform(df):
    factor_col = df.columns[-1]

    def zscore_transform(group):
        valid = np.isfinite(group)
        if not valid.any():
            return pd.Series(np.nan, index=group.index)
        mean_val = group.loc[valid].mean()
        std_val = group.loc[valid].std(ddof=0)
        if std_val == 0 or not np.isfinite(std_val):
            return pd.Series(0.0, index=group.index)
        out = pd.Series(0.0, index=group.index)
        out.loc[valid] = (group.loc[valid] - mean_val) / std_val
        out.loc[~valid] = np.nan
        return out

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

