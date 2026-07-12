import pandas as pd


financial_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\financial_metrics_1709-2512.csv')
daily_data = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231.csv')

merge = daily_data.merge(financial_data, how='outer', on=['trade_date','ts_code'])