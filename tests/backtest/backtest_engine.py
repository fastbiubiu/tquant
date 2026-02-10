"""
回测引擎模块
提供策略回测的核心功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

import tqsdk

from tquant.utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """回测结果数据类"""
    initial_balance: float
    final_balance: float
    total_return: float
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    max_drawdown: float
    sharpe_ratio: float
    annual_return: float
    trades: List[Dict[str, Any]]
    equity_curve: pd.Series


class BacktestEngine:
    """回测引擎类"""

    def __init__(self,
                 strategy_func: Callable,
                 initial_balance: float = 100000.0,
                 start_dt: datetime = None,
                 end_dt: datetime = None,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.0):
        """
        初始化回测引擎

        Args:
            strategy_func: 策略函数，接收数据并返回交易信号
            initial_balance: 初始资金
            start_dt: 回测开始时间
            end_dt: 回测结束时间
            commission_rate: 交易手续费率
            slippage: 滑点设置
        """
        self.strategy_func = strategy_func
        self.initial_balance = initial_balance
        self.start_dt = start_dt or datetime(2020, 1, 1)
        self.end_dt = end_dt or datetime(2023, 12, 31)
        self.commission_rate = commission_rate
        self.slippage = slippage
        self.indicators = TechnicalIndicators()

        # 回测状态
        self.account_balance = initial_balance
        self.position = 0
        self.trades = []
        self.equity_curve = []
        self.current_date = None

    def run(self, symbol: str, period: str = "1d") -> BacktestResult:
        """
        运行回测

        Args:
            symbol: 交易品种
            period: K线周期

        Returns:
            BacktestResult: 回测结果
        """
        try:
            logger.info(f"🚀 开始回测: {symbol} 从 {self.start_dt} 到 {self.end_dt}")

            # 创建回测 API
            auth_obj = tqsdk.auth.TqAuth("demo", "demo")
            api = tqsdk.TqApi(backtest=tqsdk.TqBacktest(start_dt=self.start_dt, end_dt=self.end_dt), auth=auth_obj)

            # 获取K线数据
            klines = api.get_kline_serial(symbol, period, 10000)

            # 初始化账户信息
            account = api.get_account()

            # 逐天回测
            for i in range(len(klines)):
                kline = klines.iloc[i]
                self.current_date = datetime.fromtimestamp(kline.datetime / 1000000000)

                # 计算技术指标
                ohlc = pd.DataFrame({
                    'open': kline.open.values,
                    'high': kline.high.values,
                    'low': kline.low.values,
                    'close': kline.close.values,
                    'volume': kline.volume.values
                }, index=[self.current_date])

                indicators = self.indicators.calculate_all(ohlc)

                # 获取价格
                current_price = kline.close[-1]
                high_price = kline.high[-1]
                low_price = kline.low[-1]

                # 执行策略信号
                signals = self.strategy_func(indicators, current_price, self.current_date)

                # 处理交易信号
                for signal in signals:
                    self._process_signal(signal, current_price, account)

                # 更新账户信息
                self._update_account(account, current_price)

                # 记录权益曲线
                self.equity_curve.append(self.account_balance)

                # 每10天保存一次进度
                if i % 10 == 0:
                    logger.info(f"进度: {i}/{len(klines)} 天 - 权益: {self.account_balance:.2f}")

            # 关闭API
            api.close()

            logger.info(f"✅ 回测完成: 最终权益 {self.account_balance:.2f}")

            # 生成回测结果
            return self._generate_result()

        except Exception as e:
            logger.error(f"❌ 回测失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_signal(self, signal: Dict, price: float, account):
        """处理交易信号"""
        symbol = signal.get('symbol', '')
        action = signal.get('action', '')  # 'BUY', 'SELL', 'CLOSE'
        volume = signal.get('volume', 0)

        if not volume or volume <= 0:
            return

        # 计算手续费
        commission = price * volume * self.commission_rate

        if action == 'BUY':
            # 买入
            cost = price * volume + commission
            if cost <= self.account_balance:
                self.account_balance -= cost
                self.position += volume
                self.trades.append({
                    'date': self.current_date,
                    'action': 'BUY',
                    'symbol': symbol,
                    'price': price,
                    'volume': volume,
                    'commission': commission,
                    'balance': self.account_balance
                })
                logger.info(f"买入 {symbol} {volume} @ {price:.2f} (剩余: {self.account_balance:.2f})")

        elif action == 'SELL':
            # 卖出
            revenue = price * volume - commission
            self.account_balance += revenue
            self.position -= volume
            self.trades.append({
                'date': self.current_date,
                'action': 'SELL',
                'symbol': symbol,
                'price': price,
                'volume': volume,
                'commission': commission,
                'balance': self.account_balance
            })
            logger.info(f"卖出 {symbol} {volume} @ {price:.2f} (剩余: {self.account_balance:.2f})")

        elif action == 'CLOSE':
            # 平仓（全部平仓）
            if self.position > 0:
                revenue = price * self.position - commission
                self.account_balance += revenue
                self.trades.append({
                    'date': self.current_date,
                    'action': 'CLOSE',
                    'symbol': symbol,
                    'price': price,
                    'volume': self.position,
                    'commission': commission,
                    'balance': self.account_balance
                })
                logger.info(f"平仓 {symbol} {self.position} @ {price:.2f} (剩余: {self.account_balance:.2f})")
                self.position = 0

    def _update_account(self, account, price):
        """更新账户信息（持仓盈亏）"""
        # 计算浮动盈亏
        unrealized_pnl = (price - price) * self.position  # 简化计算
        self.account_balance += unrealized_pnl

        # 如果有持仓，记录当前价值
        if self.position > 0:
            position_value = price * self.position
            realized_pnl = position_value - (price * self.position)  # 重置
            self.account_balance += realized_pnl

    def _generate_result(self) -> BacktestResult:
        """生成回测结果"""
        df_trades = pd.DataFrame(self.trades)

        # 计算交易统计
        total_trades = len(df_trades)
        win_trades = len(df_trades[df_trades['action'] == 'SELL'])
        loss_trades = total_trades - win_trades
        win_rate = win_trades / total_trades if total_trades > 0 else 0

        # 计算总收益
        total_return = (self.account_balance - self.initial_balance) / self.initial_balance * 100

        # 计算最大回撤
        equity_series = pd.Series(self.equity_curve)
        cummax = equity_series.cummax()
        drawdown = (equity_series - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # 计算夏普比率
        returns = equity_series.pct_change().dropna()
        sharpe_ratio = (returns.mean() / returns.std() * np.sqrt(252)) if len(returns) > 1 else 0

        # 计算年化收益
        years = (self.end_dt - self.start_dt).days / 365
        annual_return = total_return / years if years > 0 else 0

        result = BacktestResult(
            initial_balance=self.initial_balance,
            final_balance=self.account_balance,
            total_return=total_return,
            total_trades=total_trades,
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=win_rate,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            annual_return=annual_return,
            trades=self.trades,
            equity_curve=pd.Series(self.equity_curve)
        )

        return result
