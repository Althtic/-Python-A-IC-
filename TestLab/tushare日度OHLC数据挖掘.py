import time
import tushare as ts
import os

save_directory = r"C:\Users\63585\Desktop\PycharmProjects\pythonProject\StockDailyData"

if not os.path.exists(save_directory):
    os.makedirs(save_directory)

ts.set_token('247bfc2db00340dbec50a647c3219aca2230d2e8058a1b2c53fc655d')
pro = ts.pro_api()
df = pro.daily(ts_code='000001.SZ')
trade_Date = df['trade_date'].unique() # 获取目标日期序列

for i in trade_Date:
    try:
        allstock = pro.daily(trade_date=i)
        filename = f"daily_stock_{i}.csv"
        filepath = os.path.join(save_directory, filename)

        allstock.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"✅ 成功保存 {i} 的数据，共 {len(allstock)} 行记录")

        time.sleep(0.15)

    except Exception as e:
        # 如果某一天的数据出错了（比如节假日休市），打印错误但不停止程序
        print(f"❌ 获取 {i} 数据失败：{e}")
        time.sleep(1)  # 出错时多休息一下
