import os
import pandas as pd
from functools import reduce

BASE_PATH = r"C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors"

def load_factors(factor_names, base_path=BASE_PATH):
    dfs = []
    for name in factor_names:
        file_path = os.path.join(base_path, f"{name}.csv")
        df = pd.read_csv(file_path)
        value_col = [c for c in df.columns if c != 'trade_date'][0]
        df = df[['trade_date', value_col]].rename(columns={value_col: name})
        dfs.append(df)
    merged = reduce(lambda left, right: pd.merge(left, right, on='trade_date', how='inner'), dfs)
    return merged

# 示例
factor_df = load_factors(['mkt', 'hml', 'smb'])
print(factor_df.head())