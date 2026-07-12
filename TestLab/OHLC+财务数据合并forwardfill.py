import pandas as pd
import numpy as np


def merge_market_with_financial(market_df, financial_df):
    print("开始快速合并...")

    # 1. 预处理：排序
    fin_sorted = financial_df.sort_values(['ts_code', 'f_ann_date']).reset_index(drop=True)
    mkt_sorted = market_df.sort_values(['ts_code', 'trade_date']).reset_index(drop=True)

    results = []
    stocks = mkt_sorted['ts_code'].unique()
    total_stocks = len(stocks)
    print(f"共需处理 {total_stocks} 只股票...")

    for i, stock in enumerate(stocks):
        if (i + 1) % 100 == 0:
            print(f"进度：{i + 1}/{total_stocks}")

        # 提取单只股票数据
        f_mask = fin_sorted['ts_code'] == stock
        m_mask = mkt_sorted['ts_code'] == stock
        current_fin = fin_sorted[f_mask]
        current_mkt = mkt_sorted[m_mask]

        if current_fin.empty or current_mkt.empty:
            continue

        # 转为 Numpy 数组以加速计算
        mkt_dates = current_mkt['trade_date'].values
        fin_ann = current_fin['ann_date'].values
        fin_end = current_fin['end_date'].values

        M, N = len(mkt_dates), len(fin_ann)

        # 向量化广播计算 (M, 1) vs (1, N)
        # 掩码：同时满足公告日<=交易日 且 财报截止日<=交易日
        valid_mask = (fin_ann.reshape(1, -1) <= mkt_dates.reshape(-1, 1)) & \
                     (fin_end.reshape(1, -1) <= mkt_dates.reshape(-1, 1))

        # 若无任何匹配，直接生成全 NA 的财务表
        if not np.any(valid_mask):
            empty_fin = pd.DataFrame(pd.NA, index=range(M), columns=current_fin.columns)
        else:
            # 将无效位置的 ann_date 设为 -1，以便 argmax 找到有效最大值
            ann_values = np.where(valid_mask, fin_ann, -1)
            best_idx = np.argmax(ann_values, axis=1)
            has_match = np.max(ann_values, axis=1) > -1

            # 选取对应的财务行
            selected_rows = current_fin.iloc[best_idx].reset_index(drop=True)

            # 构建最终财务表：先复制选中的行，再将无匹配的行设为 NA
            final_fin_data = selected_rows.copy()
            if not np.all(has_match):
                cols_to_nan = [c for c in final_fin_data.columns if c != 'ts_code']
                final_fin_data.loc[~has_match, cols_to_nan] = pd.NA

        # 清理重叠列 (避免 concat 冲突)，保留行情表的 ts_code 和 trade_date
        clean_fin = final_fin_data.drop(columns=['ts_code', 'trade_date'], errors='ignore')
        clean_fin = clean_fin.reset_index(drop=True)
        reset_mkt = current_mkt.reset_index(drop=True)

        # 关键优化：使用 pd.concat 一次性拼接，避免 DataFrame 碎片化
        combined = pd.concat([reset_mkt, clean_fin], axis=1)
        results.append(combined)

    print("合并完成，正在拼接总表...")
    return pd.concat(results, ignore_index=True)


# ================= 主程序 =================
if __name__ == '__main__':
    # 读取数据
    market_df = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\回测数据集\20170930-20251231_pipe.csv')
    financial_df = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\A股上市资产负债表数据\merged_balance_sheet.csv')

    print(f"原始行情股票数: {len(market_df['ts_code'].unique())}")
    print(f"原始财务股票数: {len(financial_df['ts_code'].unique())}")

    # 获取交集
    common_codes = pd.Index(market_df['ts_code'].unique()).intersection(pd.Index(financial_df['ts_code'].unique()))
    # 如需全量运行，请注释掉下一行的切片操作
    # common_codes = common_codes[:100]

    filtered_mkt = market_df[market_df['ts_code'].isin(common_codes)]
    filtered_fin = financial_df[financial_df['ts_code'].isin(common_codes)]

    # 执行合并
    merged_df = merge_market_with_financial(filtered_mkt, filtered_fin)

    # 日期列类型转换 (兼容 NA 的 Int64)
    date_cols = ['ann_date', 'end_date', 'trade_date']
    for col in date_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype('Int64')

    # 保存结果
    output_file = '../回测数据集/20170930-20251231_balance_sheet.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"成功保存至 {output_file}")