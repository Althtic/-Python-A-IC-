import tushare as ts
import pandas as pd

pro = ts.pro_api()
# 按日期获取
df = pro.query('trade_cal', start_date='20170101', end_date='20170901')
date = df['cal_date']

res = []

for day in date:
    df = pro.stock_st(trade_date=day)
    res.append(df)
    print(df)

res = pd.concat(res)
res.to_csv('ST_stock_1.csv',index=False)
print(res)