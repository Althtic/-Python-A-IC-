import pandas as pd

df = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv')



df['chg'] = df.groupby('ts_code')['close'].transform(
    lambda x: (x - x.shift(1)) / x.shift(1)
)



# df.loc[(df['chg'] < -0.3) | (df['chg'] > 0.3), 'chg'] = 0
print(df.head())







col = 'chg'

print("=" * 60)
print(f"第 {col} 列数据详细信息")
print("=" * 60)

# ==================== 1. 基本统计信息 ====================
print("\n【1. 基本统计信息】")
print("-" * 60)
stats = df[col].describe()
print(stats)

# 补充统计信息
print(f"\n数据总数: {len(df[col])}")
print(f"非空值数量: {df[col].count()}")
print(f"缺失值数量: {df[col].isna().sum()}")
print(f"缺失值比例: {df[col].isna().sum() / len(df[col]) * 100:.2f}%")
print(f"唯一值数量: {df[col].nunique()}")

# ==================== 2. 最大值 Top 10 ====================
print("\n【2. 最大值 Top 10】")
print("-" * 60)
top10_max = df.nlargest(10, col)[[col]]
print(top10_max)
print(f"\n最大值: {df[col].max()}")
print(f"最大值位置(索引): {df[col].idxmax()}")

# ==================== 3. 最小值 Top 10 ====================
print("\n【3. 最小值 Top 10】")
print("-" * 60)
top10_min = df.nsmallest(10, col)[[col]]
print(top10_min)
print(f"\n最小值: {df[col].min()}")
print(f"最小值位置(索引): {df[col].idxmin()}")
print(df.iloc[865283])
print(df.iloc[865284])
print(df.iloc[865285])