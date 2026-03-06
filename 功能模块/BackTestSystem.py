import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')


class QuantBacktester:
    """
    完整的量化回测框架
    支持全市场股票日度数据的回测分析
    """

    def __init__(self):
        self.data = None
        self.signals = None
        self.portfolio_history = None
        self.metrics = {}

    def load_data(self, data: pd.DataFrame) -> None:
        """
        加载股票日度数据
        数据格式要求: ['date', 'stock_code', 'open', 'high', 'low', 'close', 'volume', 'adj_close']
        """
        self.data = data.copy()
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data.sort_values(['stock_code', 'date'], inplace=True)
        print(f"数据加载完成，共{len(self.data)}条记录")

    def generate_signals(self, strategy_func) -> pd.DataFrame:
        """
        生成交易信号
        strategy_func: 接收DataFrame返回信号DataFrame的函数
        返回: 包含['date', 'stock_code', 'signal', 'weight']的DataFrame
        """
        signals = strategy_func(self.data)
        self.signals = signals
        return signals

    def backtest(
            self,
            initial_capital: float = 1000000,
            commission_rate: float = 0.001,
            rebalance_freq: str = 'M',
            start_date: str = None,
            end_date: str = None
    ) -> Dict:
        """
        执行回测

        Parameters:
        - initial_capital: 初始资金
        - commission_rate: 手续费率
        - rebalance_freq: 调仓频率 ('D', 'W', 'M')
        - start_date, end_date: 回测时间范围
        """
        if start_date:
            start_date = pd.to_datetime(start_date)
        else:
            start_date = self.data['date'].min()

        if end_date:
            end_date = pd.to_datetime(end_date)
        else:
            end_date = self.data['date'].max()

        # 过滤日期范围
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        unique_dates = sorted(set(date_range) & set(self.data['date']))

        # 获取调仓日期
        rebalance_dates = self._get_rebalance_dates(unique_dates, rebalance_freq)

        portfolio_history = []
        current_holdings = {}
        cash = initial_capital

        for i, date in enumerate(unique_dates):
            # 获取当天价格数据
            daily_data = self.data[self.data['date'] == date]
            price_dict = dict(zip(daily_data['stock_code'], daily_data['close']))

            # 调仓日执行交易
            if date in rebalance_dates and date in self.signals['date'].values:
                # 获取新信号
                signal_data = self.signals[self.signals['date'] == date]

                # 计算新权重
                total_value = cash
                for stock, weight in current_holdings.items():
                    if stock in price_dict:
                        total_value += current_holdings[stock] * price_dict[stock]

                new_holdings = {}
                commission_cost = 0

                for _, row in signal_data.iterrows():
                    stock_code = row['stock_code']
                    target_weight = row.get('weight', row['signal'])  # 兼容不同列名

                    if target_weight > 0:  # 买入信号
                        target_value = total_value * target_weight
                        if stock_code in price_dict:
                            shares = int(target_value / price_dict[stock_code])
                            if shares > 0:
                                # 计算交易成本
                                trade_value = shares * price_dict[stock_code]
                                commission = trade_value * commission_rate

                                if cash >= trade_value + commission:
                                    new_holdings[stock_code] = shares
                                    cash -= (trade_value + commission)
                                    commission_cost += commission

                current_holdings = new_holdings

            # 计算当前组合价值
            portfolio_value = cash
            for stock, shares in current_holdings.items():
                if stock in price_dict:
                    portfolio_value += shares * price_dict[stock]

            # 记录组合状态
            portfolio_history.append({
                'date': date,
                'cash': cash,
                'portfolio_value': portfolio_value,
                'commission_cost': commission_cost
            })

        self.portfolio_history = pd.DataFrame(portfolio_history)
        self.portfolio_history.set_index('date', inplace=True)

        # 计算收益率序列
        self.portfolio_history['daily_return'] = self.portfolio_history['portfolio_value'].pct_change()
        self.portfolio_history['cumulative_return'] = (1 + self.portfolio_history['daily_return']).cumprod() - 1

        # 计算回测指标
        self.metrics = self._calculate_metrics(initial_capital)

        return self.metrics

    def _get_rebalance_dates(self, all_dates: List, freq: str) -> List:
        """获取调仓日期"""
        if freq == 'D':
            return all_dates
        elif freq == 'W':
            df = pd.DataFrame({'date': all_dates})
            df['week'] = df['date'].dt.isocalendar().week
            df['year'] = df['date'].dt.year
            rebalance_dates = df.groupby(['year', 'week']).first()['date'].tolist()
        elif freq == 'M':
            df = pd.DataFrame({'date': all_dates})
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            rebalance_dates = df.groupby(['year', 'month']).first()['date'].tolist()
        else:
            raise ValueError("freq must be 'D', 'W', or 'M'")

        return rebalance_dates

    def _calculate_metrics(self, initial_capital: float) -> Dict:
        """计算回测指标"""
        returns = self.portfolio_history['daily_return'].dropna()

        # 年化收益率
        total_return = self.portfolio_history['cumulative_return'].iloc[-1]
        trading_days = len(returns)
        years = trading_days / 252  # 假设每年252个交易日
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

        # 波动率
        volatility = returns.std() * np.sqrt(252)

        # 夏普比率
        risk_free_rate = 0.03  # 假设无风险利率3%
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility != 0 else 0

        # 最大回撤
        cumulative = self.portfolio_history['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # 卡玛比率
        calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0

        # 胜率
        positive_returns = (returns > 0).sum()
        win_rate = positive_returns / len(returns) if len(returns) > 0 else 0

        # 最大连续上涨/下跌天数
        up_streaks = []
        down_streaks = []
        current_up = 0
        current_down = 0

        for ret in returns:
            if ret > 0:
                current_up += 1
                if current_down > 0:
                    down_streaks.append(current_down)
                    current_down = 0
            elif ret < 0:
                current_down += 1
                if current_up > 0:
                    up_streaks.append(current_up)
                    current_up = 0
            else:
                if current_up > 0:
                    up_streaks.append(current_up)
                    current_up = 0
                if current_down > 0:
                    down_streaks.append(current_down)
                    current_down = 0

        max_up_streak = max(up_streaks) if up_streaks else 0
        max_down_streak = max(down_streaks) if down_streaks else 0

        metrics = {
            '总收益率': f"{total_return:.4f} ({total_return * 100:.2f}%)",
            '年化收益率': f"{annual_return:.4f} ({annual_return * 100:.2f}%)",
            '年化波动率': f"{volatility:.4f} ({volatility * 100:.2f}%)",
            '夏普比率': f"{sharpe_ratio:.4f}",
            '最大回撤': f"{max_drawdown:.4f} ({max_drawdown * 100:.2f}%)",
            '卡玛比率': f"{calmar_ratio:.4f}",
            '胜率': f"{win_rate:.4f} ({win_rate * 100:.2f}%)",
            '最大连续上涨天数': max_up_streak,
            '最大连续下跌天数': max_down_streak,
            '回测天数': trading_days,
            '最终净值': self.portfolio_history['portfolio_value'].iloc[-1],
            '初始资金': initial_capital
        }

        return metrics

    def plot_performance(self) -> None:
        """绘制回测结果图表"""
        if self.portfolio_history is None:
            print("请先执行回测")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 净值曲线
        axes[0, 0].plot(self.portfolio_history.index,
                        self.portfolio_history['portfolio_value'] / 1e6,
                        label='Portfolio Value (Million)', linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Value (Million)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()

        # 累计收益率曲线
        axes[0, 1].plot(self.portfolio_history.index,
                        self.portfolio_history['cumulative_return'] * 100,
                        label='Cumulative Return', color='red', linewidth=2)
        axes[0, 1].set_title('Cumulative Return Over Time')
        axes[0, 1].set_ylabel('Return (%)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0, 1].legend()

        # 日收益率分布
        returns = self.portfolio_history['daily_return'].dropna()
        axes[1, 0].hist(returns, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Daily Return Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].grid(True, alpha=0.3)

        # 回撤曲线
        cumulative = self.portfolio_history['cumulative_return']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        axes[1, 1].fill_between(self.portfolio_history.index,
                                drawdown * 100,
                                0,
                                alpha=0.7,
                                color='lightcoral',
                                label='Drawdown')
        axes[1, 1].set_title('Drawdown Over Time')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()

        plt.tight_layout()
        plt.show()

    def print_metrics(self) -> None:
        """打印回测指标"""
        if not self.metrics:
            print("请先执行回测")
            return

        print("=" * 50)
        print("量化策略回测报告")
        print("=" * 50)

        for key, value in self.metrics.items():
            print(f"{key:.<20} {value}")

        print("=" * 50)


# 示例使用
def sample_strategy(data: pd.DataFrame) -> pd.DataFrame:
    """
    示例策略：简单的动量策略
    基于过去20日收益率排序，买入前10%的股票
    """
    # 计算过去20日收益率
    data_with_returns = data.copy()
    data_with_returns['return_20d'] = data_with_returns.groupby('stock_code')['close'].pct_change(periods=20)

    # 获取每日最新的收益率数据
    latest_returns = data_with_returns.dropna(subset=['return_20d']).groupby('date').apply(
        lambda x: x.nlargest(int(len(x) * 0.1), 'return_20d')[['stock_code', 'return_20d']]
    ).reset_index(level=1, drop=True).reset_index()

    # 生成等权配置
    latest_returns['weight'] = 1.0 / len(latest_returns.groupby('date'))

    return latest_returns[['date', 'stock_code', 'weight']]


# 创建示例数据
def create_sample_data():
    """创建示例股票数据"""
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    dates = dates[dates.weekday < 5]  # 只保留工作日

    stocks = [f'STOCK_{i:03d}' for i in range(1, 101)]  # 100只股票

    data_list = []
    for stock in stocks:
        # 生成模拟股价
        base_price = np.random.uniform(10, 100)
        returns = np.random.normal(0.0005, 0.02, len(dates))  # 日收益率均值0.05%，标准差2%
        prices = [base_price]

        for r in returns[1:]:
            prices.append(prices[-1] * (1 + r))

        stock_data = pd.DataFrame({
            'date': dates,
            'stock_code': stock,
            'open': [p * np.random.uniform(0.98, 1.02) for p in prices],
            'high': [p * np.random.uniform(1.0, 1.05) for p in prices],
            'low': [p * np.random.uniform(0.95, 1.0) for p in prices],
            'close': prices,
            'volume': np.random.randint(100000, 1000000, len(dates)),
            'adj_close': prices
        })
        data_list.append(stock_data)

    return pd.concat(data_list, ignore_index=True)


# 使用示例
if __name__ == "__main__":
    # 创建回测器实例
    backtester = QuantBacktester()

    # 创建示例数据
    print("正在生成示例数据...")
    sample_data = create_sample_data()
    # print(sample_data.head())

    # 加载数据
    backtester.load_data(sample_data)

    # 生成信号
    print("正在生成交易信号...")
    signals = backtester.generate_signals(sample_strategy)

    # 执行回测
    print("开始回测...")
    metrics = backtester.backtest(
        initial_capital=1000000,
        commission_rate=0.001,
        rebalance_freq='M',
        start_date='2021-01-01',
        end_date='2023-12-31'
    )

    # 打印指标
    backtester.print_metrics()

    # 绘制图表
    backtester.plot_performance()
