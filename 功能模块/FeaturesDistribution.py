import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt

# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

df = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\factor_set.csv')
print(df.head())
distribution_hist = df.columns[2:]


# --- 关键修改：临时创建干净的数据副本用于绘图 ---
# 只替换要绘制的列，不修改原始df
df_for_plot = df[distribution_hist].replace([np.inf, -np.inf], np.nan)

# pandas的hist方法会自动跳过NaN值
df_for_plot.hist(bins=30, figsize=(18, 9),
                 layout=(4, 5),
                 edgecolor='black',
                 alpha=0.7,
                 grid=True)
plt.tight_layout()
plt.show()

# --- 验证原始数据未被修改 ---
print("\n原始df中的NaN数量（应保持不变）：")
print(df[distribution_hist].isna().sum())


