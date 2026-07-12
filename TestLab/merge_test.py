import pandas as pd
import numpy as np

# 设置随机种子以获得可重现的结果
np.random.seed(42)

# 生成示例行情数据
# 股票代码列表
stock_codes = ['000001.SZ', '000002.SZ', '600000.SH']

# 交易日期列表 (2024年1月的几个工作日)
trade_dates = [
    20240101, 20240102, 20240103, 20240104, 20240105,
    20240108, 20240109, 20240110, 20240111, 20240112,
    20240115, 20240116, 20240117, 20240118, 20240119,
    20240122, 20240123, 20240124, 20240125, 20240126,
    20240129, 20240130, 20240131
]

# 创建行情数据框
market_data_rows = []
for code in stock_codes:
    for date in trade_dates:
        # 生成模拟的行情数据
        close_price = round(np.random.uniform(10, 100), 2)
        volume = int(np.random.uniform(100000, 1000000))
        market_data_rows.append({
            'ts_code': code,
            'trade_date': date,
            'close': close_price,
            'volume': volume
        })

market_data = pd.DataFrame(market_data_rows)
print("--- 日度行情数据 (market_data) ---")
print(market_data)
print("\n")


# 生成示例财务数据
# 财务数据包含公告日期(ann_date)和报告期(end_date)
financial_data_rows = [
    # 000001.SZ 的财务报告
    {'ts_code': '000001.SZ', 'ann_date': 20240115, 'end_date': 20231231, 'total_share': 1000000, 'bvps': 5.5},
    {'ts_code': '000001.SZ', 'ann_date': 20240120, 'end_date': 20230930, 'total_share': 990000, 'bvps': 4.2}, # 假设930报告在1231年报之后公告
    {'ts_code': '000001.SZ', 'ann_date': 20240105, 'end_date': 20230630, 'total_share': 980000, 'bvps': 3.1},
    {'ts_code': '000001.SZ', 'ann_date': 20240110, 'end_date': 20230331, 'total_share': 970000, 'bvps': 2.0},

    # 000002.SZ 的财务报告
    {'ts_code': '000002.SZ', 'ann_date': 20240118, 'end_date': 20231231, 'total_share': 2000000, 'bvps': 8.0},
    {'ts_code': '000002.SZ', 'ann_date': 20240108, 'end_date': 20230930, 'total_share': 1950000, 'bvps': 6.5},
    {'ts_code': '000002.SZ', 'ann_date': 20240112, 'end_date': 20230630, 'total_share': 1900000, 'bvps': 5.0},

    # 600000.SH 的财务报告
    {'ts_code': '600000.SH', 'ann_date': 20240125, 'end_date': 20231231, 'total_share': 3000000, 'bvps': 10.0},
    {'ts_code': '600000.SH', 'ann_date': 20240110, 'end_date': 20230930, 'total_share': 2900000, 'bvps': 7.5},
    {'ts_code': '600000.SH', 'ann_date': 20240115, 'end_date': 20230630, 'total_share': 2800000, 'bvps': 5.0},
]

financial_data = pd.DataFrame(financial_data_rows)
print("--- 财务数据 (financial_data) ---")
print(financial_data)
print("\n")


def merge_market_with_financial(market_df, financial_df):
    """
    合并日度行情数据和财务数据，确保不使用未来数据。
    """
    # 1. 确保日期列为 int 类型 (因为我们生成的就是 int)
    # 这里为了排序和比较，临时转换为 str 再转回 int，以确保排序逻辑正确
    # 但更推荐的做法是将 int 日期转换为 datetime，计算后再转回 int
    # 为了简化演示，我们直接用 int 进行比较，因为 YYYYMMDD 格式的 int 比较逻辑与日期一致
    # 例如 20240102 > 20240101 是成立的

    # 2. 进行笛卡尔积式的合并，然后筛选 ann_date <= trade_date 的记录
    merged_df = market_df.merge(financial_df, on='ts_code', how='left')
    # 确保 ann_date 是 int 型
    merged_df['ann_date'] = merged_df['ann_date'].astype(
        'Int64')  # 使用 nullable integer type to handle potential NaN from merge
    valid_records = merged_df[merged_df['ann_date'] <= merged_df['trade_date']]

    # 3. 排序：按 ts_code, trade_date 分组，然后按 end_date 降序排列
    # 注意：如果 ann_date 或 end_date 中有 NaN，它们会被排到最后
    valid_records_sorted = valid_records.sort_values(
        by=['ts_code', 'trade_date', 'end_date'],
        ascending=[True, True, False],
        na_position='last'  # 确保 NaN 排在最后，不影响正常排序
    )

    # 4. 去重：保留每个 ts_code-trade_date 组合的第一个记录（即 end_date 最晚的那个）
    final_result = valid_records_sorted.drop_duplicates(
        subset=['ts_code', 'trade_date'],
        keep='first'
    ).reset_index(drop=True)

    # 5. 清理：删除不再需要的 ann_date 和 end_date 列（可选）
    # final_result = final_result.drop(columns=['ann_date', 'end_date'])

    return final_result


# 执行合并
merged_data = merge_market_with_financial(market_data, financial_data)

print("--- 合并后的数据 (merged_data) ---")
print(merged_data)