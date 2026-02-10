"""
增强回测验证系统
测试信号增强效果,对比增强前后的表现
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple

import numpy as np

from tquant.utils.signal_validator import SimpleTradingSignal, SimpleSignalValidator
from tquant.utils.signal_weight_optimizer import SimpleSignalWeightOptimizer, WeightingStrategy
from tquant.utils.signals import TradingSignal, SignalDirection

logger = logging.getLogger(__name__)

class BacktestConfig:
    """回测配置"""
    def __init__(self,
                 start_date: datetime,
                 end_date: datetime,
                 initial_capital: float = 1000000,
                 commission: float = 0.0003,
                 slippage: float = 0.0001,
                 max_position_ratio: float = 0.8,
                 stop_loss_ratio: float = 0.02,
                 take_profit_ratio: float = 0.05):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.max_position_ratio = max_position_ratio
        self.stop_loss_ratio = stop_loss_ratio
        self.take_profit_ratio = take_profit_ratio

class BacktestResult:
    """回测结果"""
    def __init__(self,
                 total_return: float,
                 annual_return: float,
                 max_drawdown: float,
                 sharpe_ratio: float,
                 win_rate: float,
                 profit_factor: float,
                 total_trades: int,
                 winning_trades: int,
                 losing_trades: int,
                 avg_profit_per_trade: float,
                 max_consecutive_losses: int,
                 final_capital: float):
        self.total_return = total_return
        self.annual_return = annual_return
        self.max_drawdown = max_drawdown
        self.sharpe_ratio = sharpe_ratio
        self.win_rate = win_rate
        self.profit_factor = profit_factor
        self.total_trades = total_trades
        self.winning_trades = winning_trades
        self.losing_trades = losing_trades
        self.avg_profit_per_trade = avg_profit_per_trade
        self.max_consecutive_losses = max_consecutive_losses
        self.final_capital = final_capital

class TradeRecord:
    """交易记录"""
    def __init__(self,
                 trade_id: str,
                 symbol: str,
                 entry_time: datetime,
                 exit_time: datetime,
                 entry_price: float,
                 exit_price: float,
                 quantity: int,
                 direction: str,
                 profit_loss: float,
                 profit_loss_ratio: float,
                 commission: float,
                 slippage: float,
                 duration_days: int):
        self.trade_id = trade_id
        self.symbol = symbol
        self.entry_time = entry_time
        self.exit_time = exit_time
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.quantity = quantity
        self.direction = direction
        self.profit_loss = profit_loss
        self.profit_loss_ratio = profit_loss_ratio
        self.commission = commission
        self.slippage = slippage
        self.duration_days = duration_days

class EnhancedBacktester:
    """增强回测器"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.validator = SimpleSignalValidator()
        self.weight_optimizer = SimpleSignalWeightOptimizer()
        self.trade_records = []
        self.equity_curve = []
        self.daily_returns = []

    def run_backtest(self,
                    signals: List[TradingSignal],
                    price_data: Dict[str, List[Dict]],
                    use_enhancement: bool = True,
                    weighting_strategy: WeightingStrategy = None) -> Dict[str, Any]:
        """
        运行回测

        Args:
            signals: 交易信号列表
            price_data: 价格数据
            use_enhancement: 是否使用信号增强
            weighting_strategy: 权重策略

        Returns:
            回测结果对比
        """
        print(f"\n开始增强回测验证...")
        print(f"回测期间: {self.config.start_date} 到 {self.config.end_date}")
        print(f"初始资金: {self.config.initial_capital:,.0f}")
        print(f"使用信号增强: {'是' if use_enhancement else '否'}")

        # 增强信号(如果启用)
        enhanced_signals = signals
        if use_enhancement:
            enhanced_signals = self._enhance_signals(signals)

        # 验证信号
        validated_signals = self._validate_signals(enhanced_signals, price_data)

        # 优化权重
        optimized_signals = self._optimize_weights(validated_signals, weighting_strategy)

        # 执行回测
        result = self._execute_backtest(optimized_signals, price_data)

        # 计算指标
        metrics = self._calculate_metrics(result)

        return {
            'original_signals': len(signals),
            'enhanced_signals': len(enhanced_signals),
            'validated_signals': len(validated_signals),
            'optimized_signals': len(optimized_signals),
            'backtest_result': metrics,
            'trade_records': [self._trade_record_to_dict(trade) for trade in self.trade_records],
            'equity_curve': self.equity_curve,
            'daily_returns': self.daily_returns
        }

    def _enhance_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """增强信号"""
        enhanced = []
        print(f"增强 {len(signals)} 个信号...")

        for signal in signals:
            # 模拟信号增强效果
            enhanced_confidence = min(0.95, signal.confidence + 0.05)
            enhanced_signal = TradingSignal(
                symbol=signal.symbol,
                direction=signal.direction,
                confidence=enhanced_confidence,
                price=signal.price,
                timestamp=signal.timestamp,
                indicators=signal.indicators
            )
            enhanced.append(enhanced_signal)

        print(f"信号增强完成,平均置信度从 {sum(s.confidence for s in signals)/len(signals):.3f} "
              f"提升到 {sum(s.confidence for s in enhanced)/len(enhanced):.3f}")

        return enhanced

    def _validate_signals(self, signals: List[TradingSignal], price_data: Dict) -> List[TradingSignal]:
        """验证信号"""
        validated = []
        print(f"验证 {len(signals)} 个信号...")

        market_data = {
            'trend_direction': 'bullish',
            'momentum': 0.6,
            'volume': 5000000,
            'volatility': 0.15
        }

        for signal in signals:
            # 简化的验证过程
            validation_result = self.validator.validate_signal(signal, market_data)

            # 只保留验证通过的信号
            if validation_result.status.value in ['确认', '部分确认']:
                validated.append(signal)

        print(f"信号验证完成,保留 {len(validated)} 个有效信号")

        return validated

    def _optimize_weights(self, signals: List[TradingSignal], strategy: WeightingStrategy) -> List[TradingSignal]:
        """优化权重"""
        if not signals or not strategy:
            return signals

        print(f"使用 {strategy.value} 策略优化权重...")

        market_data = {
            'trend_direction': 'bullish',
            'momentum': 0.6,
            'volume': 5000000,
            'volatility': 0.15,
            'liquidity': 0.8
        }

        # 简化的权重优化
        total_confidence = sum(s.confidence for s in signals)
        for signal in signals:
            signal.weight = signal.confidence / total_confidence

        return signals

    def _execute_backtest(self, signals: List[TradingSignal], price_data: Dict) -> List[TradeRecord]:
        """执行回测"""
        self.trade_records = []
        self.equity_curve = []
        self.daily_returns = []

        capital = self.config.initial_capital
        current_position = {}

        # 按时间排序信号
        sorted_signals = sorted(signals, key=lambda x: x.timestamp)

        print(f"执行回测,处理 {len(sorted_signals)} 个信号...")

        for signal in sorted_signals:
            if signal.timestamp < self.config.start_date or signal.timestamp > self.config.end_date:
                continue

            symbol = signal.symbol
            direction = signal.direction
            confidence = signal.confidence
            weight = getattr(signal, 'weight', 1.0)

            # 计算仓位大小
            max_position = capital * self.config.max_position_ratio * weight
            price = signal.price
            quantity = int(max_position / price)

            if quantity <= 0:
                continue

            # 计算入场价格(包含滑点)
            entry_price = price * (1 + self.config.slippage if direction == SignalDirection.BUY else
                                  -self.config.slippage)

            # 模拟持仓时间
            holding_days = int(np.random.exponential(5)) + 1
            exit_time = signal.timestamp + timedelta(days=holding_days)

            # 生成出场价格
            exit_price = self._simulate_exit_price(entry_price, direction, holding_days)

            # 计算盈亏
            if direction == SignalDirection.BUY:
                profit_loss = (exit_price - entry_price) * quantity
            else:
                profit_loss = (entry_price - exit_price) * quantity

            # 计算交易成本
            commission = quantity * price * self.config.commission
            slippage_cost = quantity * price * self.config.slippage
            total_cost = commission + slippage_cost

            # 计算净盈亏
            net_profit_loss = profit_loss - total_cost
            profit_loss_ratio = net_profit_loss / (quantity * entry_price)

            # 检查止损止盈
            if abs(profit_loss_ratio) >= self.config.stop_loss_ratio:
                exit_price = entry_price * (1 + self.config.take_profit_ratio if profit_loss > 0
                                          else -self.config.stop_loss_ratio)
                profit_loss = (exit_price - entry_price) * quantity if direction == SignalDirection.BUY else (entry_price - exit_price) * quantity
                net_profit_loss = profit_loss - total_cost
                profit_loss_ratio = net_profit_loss / (quantity * entry_price)

            # 创建交易记录
            trade = TradeRecord(
                trade_id=f"trade_{len(self.trade_records)+1}",
                symbol=symbol,
                entry_time=signal.timestamp,
                exit_time=exit_time,
                entry_price=entry_price,
                exit_price=exit_price,
                quantity=quantity,
                direction=direction.value,
                profit_loss=net_profit_loss,
                profit_loss_ratio=profit_loss_ratio,
                commission=commission,
                slippage=slippage_cost,
                duration_days=holding_days
            )

            self.trade_records.append(trade)

            # 更新资金
            capital += net_profit_loss

            # 更新权益曲线
            self.equity_curve.append({
                'date': signal.timestamp.strftime('%Y-%m-%d'),
                'equity': capital
            })

            # 计算日收益率
            if len(self.equity_curve) > 1:
                daily_return = (capital - self.equity_curve[-2]['equity']) / self.equity_curve[-2]['equity']
                self.daily_returns.append(daily_return)

        print(f"回测完成,共执行 {len(self.trade_records)} 笔交易")

        return self.trade_records

    def _simulate_exit_price(self, entry_price: float, direction: SignalDirection, holding_days: int) -> float:
        """模拟出场价格"""
        # 添加随机波动
        volatility = 0.02  # 2%的日波动率
        random_factor = np.random.normal(0, volatility * np.sqrt(holding_days))

        # 基于方向的价格变化
        if direction == SignalDirection.BUY:
            # 买入信号,期望价格上涨
            price_change = entry_price * (0.001 + random_factor)  # 0.1%的期望收益
        else:
            # 卖出信号,期望价格下跌
            price_change = entry_price * (-0.001 + random_factor)  # -0.1%的期望收益

        return entry_price + price_change

    def _calculate_metrics(self, trades: List[TradeRecord]) -> BacktestResult:
        """计算回测指标"""
        if not trades:
            return BacktestResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.config.initial_capital)

        total_profit_loss = sum(trade.profit_loss for trade in trades)
        final_capital = self.config.initial_capital + total_profit_loss

        # 计算各项指标
        total_return = (final_capital - self.config.initial_capital) / self.config.initial_capital

        # 年化收益率(假设一年252个交易日)
        days_held = sum(trade.duration_days for trade in trades)
        annual_return = (1 + total_return) ** (252 / max(days_held, 1)) - 1

        # 最大回撤
        max_drawdown = self._calculate_max_drawdown()

        # 夏普比率
        if self.daily_returns:
            sharpe_ratio = np.mean(self.daily_returns) / np.std(self.daily_returns) * np.sqrt(252) if np.std(self.daily_returns) > 0 else 0
        else:
            sharpe_ratio = 0

        # 胜率
        winning_trades = [t for t in trades if t.profit_loss > 0]
        losing_trades = [t for t in trades if t.profit_loss < 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0

        # 盈亏比
        avg_win = np.mean([t.profit_loss for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([abs(t.profit_loss) for t in losing_trades]) if losing_trades else 0
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 0

        # 连续亏损
        max_consecutive_losses = self._calculate_max_consecutive_losses(losing_trades)

        # 平均每笔交易收益
        avg_profit_per_trade = total_profit_loss / len(trades) if trades else 0

        return BacktestResult(
            total_return=total_return,
            annual_return=annual_return,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_profit_per_trade=avg_profit_per_trade,
            max_consecutive_losses=max_consecutive_losses,
            final_capital=final_capital
        )

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.equity_curve:
            return 0

        equity_values = [point['equity'] for point in self.equity_curve]
        peak = equity_values[0]
        max_drawdown = 0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            drawdown = (peak - equity) / peak
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def _calculate_max_consecutive_losses(self, losing_trades: List[TradeRecord]) -> int:
        """计算最大连续亏损次数"""
        if not losing_trades:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        # 按时间排序交易
        sorted_trades = sorted(losing_trades, key=lambda x: x.entry_time)

        for i, trade in enumerate(sorted_trades):
            if i > 0 and (trade.entry_time - sorted_trades[i-1].exit_time).days <= 5:
                current_consecutive += 1
            else:
                current_consecutive = 1
            max_consecutive = max(max_consecutive, current_consecutive)

        return max_consecutive

    def _trade_record_to_dict(self, trade: TradeRecord) -> Dict:
        """转换交易记录为字典"""
        return {
            'trade_id': trade.trade_id,
            'symbol': trade.symbol,
            'entry_time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'exit_time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S'),
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'quantity': trade.quantity,
            'direction': trade.direction,
            'profit_loss': trade.profit_loss,
            'profit_loss_ratio': trade.profit_loss_ratio,
            'commission': trade.commission,
            'slippage': trade.slippage,
            'duration_days': trade.duration_days
        }

class BacktestComparator:
    """回测对比器"""

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.backtester = EnhancedBacktester(config)

    def compare_strategies(self,
                          signals: List[TradingSignal],
                          price_data: Dict,
                          strategies: List[Tuple[str, bool, WeightingStrategy]]) -> Dict[str, Any]:
        """
        对比不同策略

        Args:
            signals: 交易信号
            price_data: 价格数据
            strategies: 策略列表 [(策略名称, 是否增强, 权重策略)]

        Returns:
            对比结果
        """
        results = {}
        strategy_names = []

        print("\n" + "=" * 60)
        print("增强回测策略对比")
        print("=" * 60)

        for strategy_name, use_enhancement, weight_strategy in strategies:
            strategy_key = f"{strategy_name}_{'enhanced' if use_enhancement else 'original'}"
            strategy_names.append(strategy_key)

            print(f"\n测试策略: {strategy_name} ({'增强' if use_enhancement else '原始'})")
            print("-" * 50)

            result = self.backtester.run_backtest(
                signals,
                price_data,
                use_enhancement,
                weight_strategy
            )

            results[strategy_key] = result

            # 显示关键指标
            metrics = result['backtest_result']
            print(f"总收益: {metrics.total_return:.2%}")
            print(f"年化收益: {metrics.annual_return:.2%}")
            print(f"最大回撤: {metrics.max_drawdown:.2%}")
            print(f"夏普比率: {metrics.sharpe_ratio:.3f}")
            print(f"胜率: {metrics.win_rate:.2%}")
            print(f"交易次数: {metrics.total_trades}")

        # 生成对比分析
        analysis = self._generate_analysis(results, strategy_names)

        return {
            'comparison_results': results,
            'analysis': analysis,
            'strategy_names': strategy_names
        }

    def _generate_analysis(self, results: Dict, strategy_names: List[str]) -> Dict[str, Any]:
        """生成对比分析"""
        analysis = {}

        # 找出最佳策略
        best_strategy = max(strategy_names,
                          key=lambda x: results[x]['backtest_result'].sharpe_ratio)

        # 计算各项指标的平均值
        metrics = ['total_return', 'annual_return', 'sharpe_ratio', 'win_rate', 'profit_factor']
        avg_metrics = {}

        for metric in metrics:
            values = [results[name]['backtest_result'].__dict__[metric] for name in strategy_names]
            avg_metrics[metric] = {
                'value': np.mean(values),
                'std': np.std(values),
                'max': max(values),
                'min': min(values)
            }

        # 生成改进建议
        improvement_suggestions = self._generate_improvement_suggestions(results, strategy_names)

        analysis.update({
            'best_strategy': best_strategy,
            'average_metrics': avg_metrics,
            'improvement_suggestions': improvement_suggestions,
            'key_findings': self._extract_key_findings(results, strategy_names)
        })

        return analysis

    def _generate_improvement_suggestions(self, results: Dict, strategy_names: List[str]) -> List[str]:
        """生成改进建议"""
        suggestions = []

        # 分析整体表现
        all_sharpes = [results[name]['backtest_result'].sharpe_ratio for name in strategy_names]
        avg_sharpe = np.mean(all_sharpes)

        if avg_sharpe < 1.0:
            suggestions.append("整体夏普比率较低,建议优化信号质量或调整仓位管理")

        # 分析增强效果
        enhanced_results = [name for name in strategy_names if 'enhanced' in name]
        if enhanced_results:
            avg_enhanced_sharpe = np.mean([results[name]['backtest_result'].sharpe_ratio for name in enhanced_results])
            original_results = [name for name in strategy_names if 'original' in name]
            avg_original_sharpe = np.mean([results[name]['backtest_result'].sharpe_ratio for name in original_results])

            if avg_enhanced_sharpe > avg_original_sharpe:
                improvement = (avg_enhanced_sharpe - avg_original_sharpe) / avg_original_sharpe
                suggestions.append(f"信号增强带来 {improvement:.1%} 的夏普比率提升")
            else:
                suggestions.append("当前信号增强效果不明显,建议调整增强策略")

        # 分析风险管理
        max_drawdowns = [results[name]['backtest_result'].max_drawdown for name in strategy_names]
        avg_drawdown = np.mean(max_drawdowns)

        if avg_drawdown > 0.1:
            suggestions.append("最大回撤较大,建议加强风险管理设置")

        # 分析交易频率
        trade_counts = [results[name]['backtest_result'].total_trades for name in strategy_names]
        avg_trades = np.mean(trade_counts)

        if avg_trades < 10:
            suggestions.append("交易次数较少,建议降低信号阈值增加交易机会")
        elif avg_trades > 100:
            suggestions.append("交易次数过多,建议提高信号过滤标准")

        return suggestions

    def _extract_key_findings(self, results: Dict, strategy_names: List[str]) -> List[str]:
        """提取关键发现"""
        findings = []

        # 找出表现最好的策略
        best_sharpe = max(strategy_names, key=lambda x: results[x]['backtest_result'].sharpe_ratio)
        findings.append(f"最佳夏普比率策略: {best_sharpe}")

        # 找出收益最高的策略
        best_return = max(strategy_names, key=lambda x: results[x]['backtest_result'].total_return)
        findings.append(f"最高收益策略: {best_return}")

        # 分析增强vs原始
        enhanced_strategies = [name for name in strategy_names if 'enhanced' in name]
        original_strategies = [name for name in strategy_names if 'original' in name]

        if enhanced_strategies and original_strategies:
            avg_enhanced = np.mean([results[name]['backtest_result'].sharpe_ratio for name in enhanced_strategies])
            avg_original = np.mean([results[name]['backtest_result'].sharpe_ratio for name in original_strategies])

            if avg_enhanced > avg_original:
                findings.append("信号增强策略整体表现优于原始策略")
            else:
                findings.append("原始策略整体表现优于信号增强策略")

        # 分析稳定性
        sharpe_stds = [results[name]['backtest_result'].sharpe_ratio for name in strategy_names]
        std_dev = np.std(sharpe_stds)

        if std_dev < 0.5:
            findings.append("各策略表现较为稳定")
        else:
            findings.append("各策略表现差异较大,建议进一步优化")

        return findings