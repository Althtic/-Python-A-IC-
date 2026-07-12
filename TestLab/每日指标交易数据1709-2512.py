import pandas as pd
import tushare as ts
import time

pro = ts.pro_api()

tool_man = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\20170930-20251231.csv')

trade_date_series = tool_man['trade_date'].unique()
print(len(trade_date_series[:500]))
# trade_date_series = trade_date_series[:500]
# trade_date_series = trade_date_series[500:1000]
# trade_date_series = trade_date_series[1000:1500]
# trade_date_series = trade_date_series[1500:1750]
trade_date_series = trade_date_series[1750:]


res_list = []  # 创建一个列表来存储每个DataFrame
for day in trade_date_series:
    time.sleep(0.3)
    df = pro.daily_basic(ts_code='', trade_date=int(day), fields='ts_code,trade_date,turnover_rate,turnover_rate_f,volume_ratio,pe,pb,ps,dv_ratio,total_share,float_share,free_share,total_mv,circ_mv')
    res_list.append(df)  # 将每个DataFrame添加到列表中
    print(day)
# 使用concat合并所有DataFrame
res = pd.concat(res_list, ignore_index=True)

res.to_csv('daily_basic_1709-2512_5', index=False)
