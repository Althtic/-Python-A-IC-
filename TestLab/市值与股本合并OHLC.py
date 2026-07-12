import pandas as pd

pd.set_option('display.max_columns', None) # 显示所有列
pd.set_option('display.width', None)       # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None)# 列宽无限制（防止单元格内容被截断）

df1 = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\TestLab\suspension_data_1.csv')
df2 = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\TestLab\suspension_data_2.csv')
df3 = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\TestLab\suspension_data_3.csv')

df = pd.concat([df1, df2, df3])
df.to_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\TestLab\suspension_data17-25.csv', index=False)