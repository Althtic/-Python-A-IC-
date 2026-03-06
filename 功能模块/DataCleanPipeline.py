import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('TkAgg')  # 设置后端
import matplotlib.pyplot as plt
from typing import Callable, List, Dict, Any
from datetime import datetime


# ─────────────────────────────────────────────────────────────────────────────
# Pandas 显示设置
# ─────────────────────────────────────────────────────────────────────────────
# pd.set_option('display.max_rows', None)   # 显示所有行
pd.set_option('display.max_columns', None)  # 显示所有列
pd.set_option('display.width', None)        # 取消换行（字符宽度限制）
pd.set_option('display.max_colwidth', None) # 列宽无限制（防止单元格内容被截断）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ─────────────────────────────────────────────────────────────────────────────
# 数据处理函数
# ─────────────────────────────────────────────────────────────────────────────

# *ST & ST股票合并
def ST_stock_id(hist_data, st_data):

    st_data = st_data.sort_values(by=['ts_code', 'trade_date'], ascending=[True, True])
    st_data['ST_type'] = st_data['name'].str[:-2]
    drop_columns = ['name', 'type', 'type_name']
    st_data = st_data.drop(columns=drop_columns)
    ST_processing_data = hist_data.merge(st_data, how='left', on=['ts_code', 'trade_date'])

    return ST_processing_data

# 可成交日处理
def is_trading_processing(hist_data, sus_data):

    data = hist_data.merge(sus_data, how='left', on=['ts_code', 'trade_date'])
    # 条件1：停牌（suspend_timing有值 且 suspend_type=='S'）
    is_suspended = (
            data['suspend_timing'].notna() &
            (data['suspend_type'] == 'S')
    )

    # 条件2：复牌日且无成交（suspend_type=='R' 且 OHLC价格相等）
    is_resume_no_trade = (
            (data['suspend_type'] == 'R') &
            (data['open'] == data['close']) &
            (data['high'] == data['low']) &
            (data['open'] == data['high'])
    )

    # 条件3：全天无成交（流动性枯竭）
    is_no_volume = (
            (data['vol'] == 0) &
            (data['open'] == data['close']) &
            (data['high'] == data['low'])
    )

    # 条件4：数据缺失（NaN）
    is_data_missing = data['open'].isna()

    # 不可成交赋 0，可成交赋值 1
    data['is_trading'] = np.where(is_suspended | is_resume_no_trade, 0, 1)

    return data

# 每日指标数据合并
def merge_daily_basic(hist_data, daily_data):
    daily_data = hist_data.merge(daily_data, how='left', on=['ts_code', 'trade_date'])
    return daily_data


# 日度收益率计算
def calculate_dret(history_data):

    # 日度收益率序列计算
    history_data['dret'] = history_data.groupby('ts_code')['close'].transform(
        lambda x: ((x - x.shift(1)) / x.shift(1)).round(5)
    )
    # 剔除第一行因为没有前一天close导致dret为nan的数据
    history_data = history_data.dropna(subset=['dret'])
    return history_data
# 剔除次新股上市后15个交易日数据（波动太大，不作为投资标的）
def remove_new_stock_initial_days(hist_data):
    df = hist_data.copy()
    # 1. 确保日期列为int型并排序
    df['trade_date'] = df['trade_date'].astype(int)
    # 2. 找出每只股票的首个交易日
    first_trade = df.groupby('ts_code')['trade_date'].min().reset_index()
    first_trade.columns = ['ts_code', 'first_date']
    # 3. 合并到原数据
    df = df.merge(first_trade, on='ts_code', how='left')
    # 4. 标记是否为需要清洗的新股（首个交易日在20171010-20251231之间）
    df['is_new_stock'] = (df['first_date'] >= 20171010) & (df['first_date'] <= 20251231)
    # 5. 标记每只股票的第几个交易日
    df['trade_day_num'] = df.groupby('ts_code').cumcount() + 1
    # 6. 执行删除逻辑
    # - 老股票（is_new_stock=False）：全部保留
    # - 新股（is_new_stock=True）：只保留第15个交易日之后
    df_clean = df[(df['is_new_stock'] == False) | (df['trade_day_num'] > 15)].copy()
    # 7. 清理辅助列
    df_clean = df_clean.drop(columns=['first_date', 'is_new_stock', 'trade_day_num'])
    df_clean = df_clean.reset_index(drop=True)
    print(df_clean)
    # 8. 验证结果
    print(f"原始数据量: {len(df):,}")
    print(f"清洗后数据量: {len(df_clean):,}")
    print(f"删除数据量: {len(df) - len(df_clean):,}")
    print(f"删除比例: {(len(df) - len(df_clean)) / len(df) * 100:.2f}%")
    return df_clean
# 截断停复牌异常收益率(针对复牌日的异常收益率赋 0)
def return_adjustment_trading_suspensions(data):
    df = data.copy()
    # 定义各市场涨跌幅阈值
    chg_limits = {
        '北交所': 0.30,
        '科创板': 0.20,
        '创业板': 0.20,
        '沪深主板': 0.10,
        '其他': 0.30  # 默认30%
    }
    # 市场状态识别函数
    def identify_market(ts_code, trade_date=None):
        ts_code = str(ts_code)
        if ts_code.startswith('8') or ts_code.startswith('9'):
            return '北交所'
        elif ts_code.startswith('688') or ts_code.startswith('689'):
            return '科创板'
        elif ts_code.startswith('300') or ts_code.startswith('301') or ts_code.startswith('302'):
            return '创业板'
        elif ts_code.startswith('60') or ts_code.startswith('000') or ts_code.startswith('001') or ts_code.startswith(
                '002') or ts_code.startswith('003'):
            return '沪深主板'
        else:
            return '其他'
    # 异常值赋NaN值，后续继续做填充处理
    def truncate_chg(row):
        market = row['market']
        chg = row['dret']
        limit = chg_limits.get(market, 0.30)

        if abs(chg) > limit:
            return 0  # 复牌日收益率填'0'，（现实意义：肯定买不进去）
        return chg

    df['market'] = df['ts_code'].astype(str).apply(identify_market)
    df['dret'] = df.apply(truncate_chg, axis=1)
    return df
# 收益率直方图，检查是否还有离群值（由停复牌后.shift计算导致）
def distribution_plot(df, value_col='dret', market_col='market', save_path='markets_dret_distribution_plot.png'):
    markets = df[market_col].unique()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, market in enumerate(markets):
        # 提取当前市场数据，替换无穷值为NaN
        data = df[df[market_col] == market][value_col].replace([np.inf, -np.inf], np.nan)
        axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
        axes[i].set_title(f'{market}\n(n={len(data):,})', fontsize=12)
        axes[i].set_xlabel(value_col)
        axes[i].set_ylabel('frequency')
        axes[i].grid(True, alpha=0.3)
        axes[i].axvline(0, color='red', linestyle='--', linewidth=1)

    for j in range(len(markets), 4):
        axes[j].set_visible(False)

    plt.suptitle(f'{value_col} distribution - by markets', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 图表已保存至: {os.path.abspath(save_path)}")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline 类封装
# ─────────────────────────────────────────────────────────────────────────────
class DataPipeline:

    def __init__(self, initial_data: pd.DataFrame, name: str = "StockDataPipeline"):
        self.data = initial_data.copy()
        self.name = name
        self.steps: List[Dict] = []
        self.start_time = datetime.now()

    def add_step(self, func: Callable, name: str = None, verbose: bool = True, **kwargs) -> 'DataPipeline':
        """添加处理步骤"""
        step_name = name or func.__name__
        prev_rows = len(self.data)

        try:
            self.data = func(self.data, **kwargs)
            curr_rows = len(self.data)

            self.steps.append({
                'step': step_name,
                'status': 'success',
                'prev_rows': prev_rows,
                'curr_rows': curr_rows,
                'delta': curr_rows - prev_rows
            })

            if verbose:
                delta_str = f"{curr_rows - prev_rows:+,}"
                print(f"✅ {step_name}: {prev_rows:,} → {curr_rows:,} ({delta_str})")

        except Exception as e:
            self.steps.append({
                'step': step_name,
                'status': 'failed',
                'error': str(e)
            })
            print(f"❌ {step_name} 失败：{e}")
            raise

        return self

    def summary(self) -> None:
        """打印执行摘要"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()

        print("\n" + "=" * 70)
        print(f"📊 Pipeline 执行摘要：{self.name}")
        print("=" * 70)
        print(f"{'步骤':<25} {'状态':<8} {'行数变化':>15}")
        print("-" * 70)

        for step in self.steps:
            if step['status'] == 'success':
                delta_str = f"{step['delta']:+,}"
                print(f"✅ {step['step']:<22} {'成功':<8} {delta_str:>15}")
            else:
                print(f"❌ {step['step']:<22} {'失败':<8} {step.get('error', ''):>15}")

        print("=" * 70)
        print(f"最终数据量：{len(self.data):,} 行")
        print(f"总耗时：{duration:.2f} 秒")
        print("=" * 70 + "\n")

    def get_data(self) -> pd.DataFrame:
        """获取处理后的数据"""
        return self.data

    def save(self, path: str, **kwargs) -> 'DataPipeline':
        """保存数据"""
        self.data.to_csv(path, index=False, **kwargs)
        print(f"💾 数据已保存至：{os.path.abspath(path)}")
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. 加载数据
    print("=" * 70)
    print("🚀 开始数据处理 Pipeline")
    print("=" * 70)

    history_data = pd.read_csv(
        r'/pythonProject/QuantSystem/20170930-20251231_OHLC.csv')
    suspension_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\Suspension_data17-25.csv')
    st_stock_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\ST-Stocks.csv')
    daily_basic_data = pd.read_csv(
        r'C:\Users\63585\Desktop\PycharmProjects\pythonProject\QuantSystem\daily_basic_1709-2512.csv'
    )

    print(f"📁 历史数据：{len(history_data):,} 行")
    print(f"📁 停牌数据：{len(suspension_data):,} 行")
    print(f"📁 ST 数据：{len(st_stock_data):,} 行")
    print(f"📁 每日指标数据：{len(daily_basic_data):,} 行")
    print("=" * 70 + "\n")

    # 2. 构建 Pipeline
    pipeline = (DataPipeline(history_data, name="股票数据清洗 Pipeline")

                .add_step(ST_stock_id, "ST 股票标记", st_data=st_stock_data)
                .add_step(is_trading_processing, "可交易状态标记", sus_data=suspension_data)
                .add_step(merge_daily_basic, "每日指标数据合并", daily_data=daily_basic_data)
                .add_step(calculate_dret, "日度收益率计算")
                .add_step(remove_new_stock_initial_days, "剔除新股前 15 日")
                .add_step(return_adjustment_trading_suspensions, "停牌收益率调整")
                )

    # 3. 执行摘要
    pipeline.summary()

    # 4. 可视化
    print("📈 生成收益率分布图...")
    distribution_plot(pipeline.get_data())

    # 5. 保存结果
    pipeline.save('20170930-20251231.csv')

    # 6. 最终验证
    df_final = pipeline.get_data()
    print("\n📋 最终数据概览:")
    print(f"   总行数：{len(df_final):,}")
    print(f"   总列数：{len(df_final.columns)}")
    print(f"   列名：{df_final.columns.tolist()}")



