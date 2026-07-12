import pandas as pd
import numpy as np
from scipy import stats


# --- 设置 Pandas 显示选项 ---
# pd.set_option('display.max_rows', None)    # 显示所有行
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）


# 读取数据
df = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv')


# 1. 确保日期列为int型并排序
df['trade_date'] = df['trade_date'].astype(int)

# 2. 找出每只股票的首个交易日
first_trade = df.groupby('ts_code')['trade_date'].min().reset_index()
first_trade.columns = ['ts_code', 'first_date']

# 3. 合并到原数据
df = df.merge(first_trade, on='ts_code', how='left')
print(df)

# 4. 标记是否为需要清洗的新股（首个交易日在20171010-20251231之间）
df['is_new_stock'] = (df['first_date'] >= 20171010) & (df['first_date'] <= 20251231)
print(df)
# 5. 标记每只股票的第几个交易日
df['trade_day_num'] = df.groupby('ts_code').cumcount() + 1
print(df)
# 6. 执行删除逻辑
# - 老股票（is_new_stock=False）：全部保留
# - 新股（is_new_stock=True）：只保留第20个交易日之后
df_clean = df[(df['is_new_stock'] == False) | (df['trade_day_num'] > 20)].copy()
print(df)
# 7. 清理辅助列
df_clean = df_clean.drop(columns=['first_date', 'is_new_stock', 'trade_day_num'])
df_clean = df_clean.reset_index(drop=True)
print(df_clean)
# 8. 验证结果
print(f"原始数据量: {len(df):,}")
print(f"清洗后数据量: {len(df_clean):,}")
print(f"删除数据量: {len(df) - len(df_clean):,}")
print(f"删除比例: {(len(df) - len(df_clean)) / len(df) * 100:.2f}%")

# df_clean.to_csv('test',index=False)