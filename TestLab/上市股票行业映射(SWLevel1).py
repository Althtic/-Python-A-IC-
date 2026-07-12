import pandas as pd
import numpy as np

stock_daily_data = pd.read_csv(r'/pythonProject/QuantSystem/20170930_20191231_ori.csv')
sorted_result = pd.read_csv(r'/pythonProject/QuantSystem/SWlevel1_sorted_ori.csv', index_col=0)
sorted_result.columns = ['trade_date','ts_code','industry_name']
trade_date_list = stock_daily_data['trade_date'].unique()
print(trade_date_list)

all_result = []
for date in trade_date_list:
    intrady_stock_data = stock_daily_data[stock_daily_data['trade_date'] == date]
    intrady_sorted_result = sorted_result[sorted_result['trade_date'] == date]
    industry_mapping = intrady_sorted_result[['ts_code', 'industry_name']]
    intrady_stock_data = intrady_stock_data.merge(
        industry_mapping,
        on='ts_code',
        how='left'  # 保留所有股票，未匹配的行业设为NaN
    )
    print(f'正在处理{date}数据')
    intrady_stock_data['industry_name'] = intrady_stock_data['industry_name'].fillna('Unknown')
    all_result.append(intrady_stock_data)
all_result_data = pd.concat(all_result)
print(all_result_data)
all_result_data = all_result_data.sort_values(by=['trade_date'])
print(len(all_result_data[all_result_data['industry_name']=='Unknown']))
all_result_data.to_csv('test.csv')
