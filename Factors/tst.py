import pandas as pd

smb = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors\smb.csv')
cma = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors\cma.csv')
hml = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors\hml.csv')
rmw = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors\rmw.csv')
mkt = pd.read_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors\mkt.csv')

ff5 = smb.merge(cma, on='trade_date').merge(hml, on='trade_date').merge(rmw, on='trade_date').merge(mkt, on='trade_date')

ff5.to_csv(r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Factors\FF5.csv', index=False)
