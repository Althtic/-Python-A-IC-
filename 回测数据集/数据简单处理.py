import pandas as pd

df_rf = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\rf.csv')
df_calender = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\交易日历.csv')
df_calender.columns = ['trade_date']

rf_merge = pd.merge(df_calender, df_rf, on=['trade_date'], how='left')
rf_merge   = rf_merge.sort_values(by=['trade_date'], ascending=True)
print(rf_merge[rf_merge['rf'].isna()])


rf_merge['rf'] = rf_merge['rf'].ffill()

print(rf_merge[rf_merge['trade_date'] == 20210630])
print(rf_merge[rf_merge['trade_date'] == 20210629])
print(rf_merge[rf_merge['trade_date'] == 20210628])

rf_merge.to_csv('rf.csv', index=False)

