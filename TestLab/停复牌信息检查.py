import pandas as pd

# pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

df1 = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\TestLab\ts_ssss')

df = df1[df1['suspend_type'].isin(['S'])]
df = df.sort_values(by=['ts_code','trade_date'],ascending=[True,True])
print(df[df['trade_date'] == 20251230])