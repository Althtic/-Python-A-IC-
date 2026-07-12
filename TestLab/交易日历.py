import tushare as ts

pro = ts.pro_api()

df = pro.trade_cal(exchange='', start_date='20170101', end_date='20251231')
df = df[df['is_open'] == 1]
df = df.sort_values(by=['cal_date'])['cal_date']
df.to_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\calender_date.csv', index=False)

