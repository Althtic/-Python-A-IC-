import pandas as pd
import tushare as ts
pro = ts.pro_api()

cal_date = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\calender_date.csv')
cal_date = cal_date['cal_date'][:800]
# cal_date = cal_date['cal_date'][800:1600]
# cal_date = cal_date['cal_date'][1600:]
print(cal_date)

suspension_data = []

for date in cal_date:
    df = pro.suspend_d(trade_date=date)
    print(df.head(3))
    suspension_data.append(df)

data = pd.concat(suspension_data, ignore_index=True)
data.to_csv('suspension_data_1.csv', index=False)
print(data)
