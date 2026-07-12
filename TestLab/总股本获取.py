import tushare as ts

# 1. 设置你的 token (请替换成你自己的)
ts.set_token('247bfc2db00340dbec50a647c3219aca2230d2e8058a1b2c53fc655d')
pro = ts.pro_api()

# 2. 调用 daily_basic 接口
# ts_code: 股票代码，例如 '000001.SZ'
# trade_date: 查询日期，格式 YYYYMMDD，例如 '20240102'
# start_date & end_date: 也可以用这两个参数查询一个日期范围内的数据

stock_code = '000001.SZ'  # 替换为你想查询的股票代码
query_date = '20240102'  # 替换为你想查询的日期

try:
    df = pro.daily_basic(ts_code=stock_code, trade_date=query_date)

    if not df.empty:
        # 获取流通股本 (单位：万股)
        float_share = df.iloc[0]['float_share']

        # 获取总股本 (单位：万股)
        total_share = df.iloc[0]['total_share']

        # 获取当日收盘价 (可选，用于计算)
        close_price = df.iloc[0]['close']

        print(f"股票代码: {stock_code}")
        print(f"查询日期: {query_date}")
        print(f"总股本: {total_share} 万股")
        print(f"流通股本: {float_share} 万股")
        print(f"当日收盘价: {close_price}")

        # 如果你想得到流通股本的股数（单位：股），需要乘以 10000
        float_share_count = float_share * 10000
        print(f"流通股本数: {float_share_count} 股")

    else:
        print(f"未找到股票 {stock_code} 在 {query_date} 的数据。")

except Exception as e:
    print(f"调用API失败: {e}")