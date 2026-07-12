import pandas as pd
import numpy as np


def merge_market_with_financial_debug(market_df, financial_df):
    print("=" * 60)
    print("【步骤 1: 数据预处理与排序】")
    fin_sorted = financial_df.sort_values(['ts_code', 'ann_date']).reset_index(drop=True)
    mkt_sorted = market_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    # 打印排序后的财务数据，方便对照
    print("排序后的财务数据 (按 ann_date):")
    print(fin_sorted[['ts_code', 'ann_date', 'end_date', 'report_period']].to_string(index=False))
    print("-" * 60)

    results = []
    stocks = mkt_sorted['ts_code'].unique()

    for stock in stocks:
        print(f"\n>>> 正在处理股票: {stock}")

        f_mask = fin_sorted['ts_code'] == stock
        m_mask = mkt_sorted['ts_code'] == stock
        current_fin = fin_sorted[f_mask]
        current_mkt = mkt_sorted[m_mask]

        if current_fin.empty or current_mkt.empty:
            continue

        # 提取数组
        mkt_dates = current_mkt['trade_date'].values
        fin_ann = current_fin['ann_date'].values
        fin_end = current_fin['end_date'].values
        fin_reports = current_fin['report_period'].values  # 用于打印展示

        M, N = len(mkt_dates), len(fin_ann)

        print(f"   交易日数量: {M}, 财务记录数量: {N}")

        # --- 核心逻辑可视化 ---
        # 1. 构建掩码矩阵
        # reshape 是为了广播：(M, 1) 和 (1, N) 比较 -> 得到 (M, N) 矩阵
        mask_ann = fin_ann.reshape(1, -1) <= mkt_dates.reshape(-1, 1)
        mask_end = fin_end.reshape(1, -1) <= mkt_dates.reshape(-1, 1)
        valid_mask = mask_ann & mask_end

        print("\n   [调试信息] 每日匹配矩阵 (行=交易日, 列=财务记录):")
        print("   列索引对应财务记录:", list(range(N)), "-> 对应报告期:", fin_reports)

        # 逐行打印每个交易日的筛选过程
        for i, date in enumerate(mkt_dates):
            row_mask = valid_mask[i]
            row_ann = fin_ann
            row_rep = fin_reports

            # 找出符合条件的索引
            valid_indices = np.where(row_mask)[0]

            status = ""
            selected_report = "无 (NA)"

            if len(valid_indices) > 0:
                # 在符合条件的里面找 ann_date 最大的
                # 我们只取这些有效索引对应的 ann_date
                valid_anns = row_ann[valid_indices]
                best_local_idx = np.argmax(valid_anns)  # 在有效列表中的位置
                global_best_idx = valid_indices[best_local_idx]  # 映射回原始索引

                selected_report = row_rep[global_best_idx]
                status = f"选中第 {global_best_idx} 条记录 ({selected_report})"
            else:
                status = "无匹配记录 (全 NA)"

            # 打印可读性强的日志
            print(f"   - 交易日 {date}: 符合条件索引={valid_indices.tolist()}, {status}")

        # --- 实际计算 (向量化) ---
        if not np.any(valid_mask):
            final_fin_data = pd.DataFrame(pd.NA, index=range(M), columns=current_fin.columns)
        else:
            ann_values = np.where(valid_mask, fin_ann, -1)
            best_idx = np.argmax(ann_values, axis=1)
            has_match = np.max(ann_values, axis=1) > -1

            selected_rows = current_fin.iloc[best_idx].reset_index(drop=True)
            final_fin_data = selected_rows.copy()

            if not np.all(has_match):
                cols_to_nan = [c for c in final_fin_data.columns if c != 'ts_code']
                final_fin_data.loc[~has_match, cols_to_nan] = pd.NA

        # 拼接
        clean_fin = final_fin_data.drop(columns=['ts_code', 'trade_date'], errors='ignore')
        clean_fin = clean_fin.reset_index(drop=True)
        reset_mkt = current_mkt.reset_index(drop=True)
        combined = pd.concat([reset_mkt, clean_fin], axis=1)
        results.append(combined)

        print(f"\n   >>> {stock} 处理完成，当日合并结果预览:")
        # 只打印关键列
        display_cols = [c for c in combined.columns if
                        c in ['trade_date', 'price', 'report_period', 'ann_date', 'end_date']]
        print(combined[display_cols].to_string(index=False))
        print("-" * 60)

    return pd.concat(results, ignore_index=True)


# ================= 构造小样本数据 =================

# 1. 构造财务数据 (Financial Data)
# 场景设计：
# R1 (Q1): 结束于 0331, 公告于 0410 (正常)
# R2 (Q2): 结束于 0630, 公告于 0715 (正常)
# R3 (Q3): 结束于 0930, 公告于 1020 (正常)
# 注意：故意设置一个 trade_date 在 0405，此时 R1 还没公告，应该为 NA
# 注意：设置一个 trade_date 在 0710，此时 R2 还没公告，虽然 R2 的 end_date(0630) 已过，但不能用，只能用 R1
data_fin = {
    'ts_code': ['000001.SZ', '000001.SZ', '000001.SZ'],
    'ann_date': [20230410, 20230715, 20231020],  # 公告日
    'end_date': [20230331, 20230630, 20230930],  # 财报截止日
    'report_period': ['2023Q1', '2023Q2', '2023Q3'],  # 辅助列，方便看结果
    'net_profit': [100, 200, 300]  # 辅助列，模拟数值
}
df_fin = pd.DataFrame(data_fin)

# 2. 构造行情数据 (Market Data)
# 覆盖几个关键时间点：
# T1 (0405): 在 Q1 公告前 -> 应 NA
# T2 (0415): 在 Q1 公告后，Q2 结束前 -> 应 Q1
# T3 (0710): 在 Q2 结束後，但 Q2 公告前 -> 应 Q1 (关键点：end_date 过了但 ann_date 没过，不能用 Q2)
# T4 (0720): 在 Q2 公告后 -> 应 Q2
# T5 (1025): 在 Q3 公告后 -> 应 Q3
data_mkt = {
    'ts_code': ['000001.SZ'] * 5,
    'trade_date': [20230405, 20230415, 20230710, 20230720, 20231025],
    'price': [10.5, 10.8, 11.2, 11.5, 12.0]
}
df_mkt = pd.DataFrame(data_mkt)

print("=== 原始输入数据 ===")
print("财务数据:")
print(df_fin.to_string(index=False))
print("\n行情数据:")
print(df_mkt.to_string(index=False))
print("\n")

# ================= 运行函数 =================
result_df = merge_market_with_financial_debug(df_mkt, df_fin)

print("\n=== 最终完整结果 ===")
print(result_df.to_string(index=False))