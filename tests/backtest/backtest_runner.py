"""
回测运行器模块
提供策略回测的完整流程管理
"""

import pandas as pd
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta

from tests.backtest.backtest_engine import BacktestEngine, BacktestResult
from tests.backtest.backtest_validator import BacktestValidator

logger = logging.getLogger(__name__)


class BacktestRunner:
    """回测运行器类"""

    def __init__(self,
                 initial_balance: float = 100000.0,
                 start_dt: datetime = None,
                 end_dt: datetime = None,
                 commission_rate: float = 0.0003,
                 slippage: float = 0.0):
        """
        初始化回测运行器

        Args:
            initial_balance: 初始资金
            start_dt: 回测开始时间
            end_dt: 回测结束时间
            commission_rate: 交易手续费率
            slippage: 滑点设置
        """
        self.initial_balance = initial_balance
        self.start_dt = start_dt or datetime(2020, 1, 1)
        self.end_dt = end_dt or datetime(2023, 12, 31)
        self.commission_rate = commission_rate
        self.slippage = slippage

        self.validator = BacktestValidator()

    def run_strategy(self,
                     strategy_func: Callable,
                     symbol: str,
                     period: str = "1d",
                     verbose: bool = True) -> Optional[BacktestResult]:
        """
        运行单个策略回测

        Args:
            strategy_func: 策略函数
            symbol: 交易品种
            period: K线周期
            verbose: 是否输出详细日志

        Returns:
            BacktestResult: 回测结果
        """
        if verbose:
            logger.info(f"🚀 开始回测: {symbol}")
            logger.info(f"   时间范围: {self.start_dt} 到 {self.end_dt}")
            logger.info(f"   初始资金: {self.initial_balance:,.2f}")

        # 创建回测引擎
        engine = BacktestEngine(
            strategy_func=strategy_func,
            initial_balance=self.initial_balance,
            start_dt=self.start_dt,
            end_dt=self.end_dt,
            commission_rate=self.commission_rate,
            slippage=self.slippage
        )

        # 运行回测
        result = engine.run(symbol, period)

        if result:
            # 验证回测结果
            validations = self.validator.validate(result)
            result.validations = validations

            if verbose:
                logger.info(f"✅ 回测完成: 最终权益 {result.final_balance:,.2f} (收益 {result.total_return:.2f}%)")

        return result

    def run_multiple_strategies(self,
                               strategy_dict: Dict[str, Callable],
                               symbols: List[str] = None,
                               periods: List[str] = None,
                               verbose: bool = True) -> pd.DataFrame:
        """
        运行多个策略回测

        Args:
            strategy_dict: 策略字典 {策略名: 策略函数}
            symbols: 交易品种列表（可选）
            periods: K线周期列表（可选）
            verbose: 是否输出详细日志

        Returns:
            pd.DataFrame: 多策略比较结果
        """
        if verbose:
            logger.info(f"🚀 开始多策略回测: {len(strategy_dict)} 个策略")

        results = []
        strategy_names = list(strategy_dict.keys())
        symbols = symbols or ['SHFE.rb']  # 默认使用螺纹钢
        periods = periods or ['1d', '4h']

        for strategy_name, strategy_func in strategy_dict.items():
            if verbose:
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 策略: {strategy_name}")
                logger.info(f"{'='*60}")

            for symbol in symbols:
                for period in periods:
                    result = self.run_strategy(
                        strategy_func,
                        symbol,
                        period,
                        verbose=verbose
                    )

                    if result:
                        results.append({
                            'strategy': strategy_name,
                            'symbol': symbol,
                            'period': period,
                            'result': result
                        })

        # 生成比较报告
        return self._generate_comparison_report(results)

    def _generate_comparison_report(self, results: List[Dict]) -> pd.DataFrame:
        """
        生成多策略比较报告

        Args:
            results: 回测结果列表

        Returns:
            pd.DataFrame: 比较报告
        """
        data = []

        for item in results:
            result = item['result']
            data.append({
                '策略': item['strategy'],
                '品种': item['symbol'],
                '周期': item['period'],
                '初始资金': result.initial_balance,
                '最终权益': result.final_balance,
                '总收益率': result.total_return,
                '总交易次数': result.total_trades,
                '胜率': result.win_rate,
                '最大回撤': result.max_drawdown,
                '夏普比率': result.sharpe_ratio,
                '年化收益': result.annual_return,
                '验证通过': result.validations.get('overall', False)
            })

        df = pd.DataFrame(data)

        # 按收益率排序
        df = df.sort_values('总收益率', ascending=False)

        return df

    def run_walk_forward(self,
                        strategy_func: Callable,
                        symbol: str,
                        periods: List[str] = None,
                        windows: List[int] = None,
                        steps: List[int] = None) -> Dict[str, Any]:
        """
        运行滚动窗口回测（滚动优化）

        Args:
            strategy_func: 策略函数
            symbol: 交易品种
            periods: K线周期列表
            windows: 回测窗口大小（天数）
            steps: 步长大小（天数）

        Returns:
            Dict[str, Any]: 滚动窗口回测结果
        """
        if verbose:
            logger.info(f"🚀 开始滚动窗口回测: {symbol}")

        windows = windows or [90]  # 默认90天窗口
        steps = steps or [7]  # 默认7天步长
        periods = periods or ['1d']

        results = []

        for window in windows:
            for step in steps:
                for period in periods:
                    logger.info(f"\n窗口: {window}天, 步长: {step}天, 周期: {period}")

                    # 计算时间范围
                    start_dt = self.start_dt
                    end_dt = min(self.end_dt, start_dt + timedelta(days=window))

                    while end_dt <= self.end_dt:
                        # 运行回测
                        engine = BacktestEngine(
                            strategy_func=strategy_func,
                            initial_balance=self.initial_balance,
                            start_dt=start_dt,
                            end_dt=end_dt,
                            commission_rate=self.commission_rate,
                            slippage=self.slippage
                        )

                        result = engine.run(symbol, period)

                        if result:
                            results.append({
                                'window': window,
                                'step': step,
                                'period': period,
                                'start_dt': start_dt,
                                'end_dt': end_dt,
                                'result': result
                            })

                        # 移动窗口
                        start_dt += timedelta(days=step)
                        end_dt += timedelta(days=step)

        # 分析结果
        return self._analyze_walk_forward(results)

    def _analyze_walk_forward(self, results: List[Dict]) -> Dict[str, Any]:
        """
        分析滚动窗口结果

        Args:
            results: 滚动窗口回测结果列表

        Returns:
            Dict[str, Any]: 分析结果
        """
        logger.info("\n" + "="*60)
        logger.info("📈 滚动窗口分析结果")
        logger.info("="*60)

        if not results:
            logger.warning("没有找到回测结果")
            return {}

        # 统计所有策略的性能
        all_results = [r['result'] for r in results]
        avg_return = np.mean([r.total_return for r in all_results])
        avg_sharpe = np.mean([r.sharpe_ratio for r in all_results])
        avg_max_dd = np.mean([r.max_drawdown for r in all_results])

        logger.info(f"平均总收益率: {avg_return:.2f}%")
        logger.info(f"平均夏普比率: {avg_sharpe:.2f}")
        logger.info(f"平均最大回撤: {avg_max_dd:.2f}%")
        logger.info(f"总回测次数: {len(results)}")

        # 识别最佳窗口
        best_result = max(all_results, key=lambda x: x.total_return)

        logger.info(f"\n最佳策略结果:")
        logger.info(f"  周期: {best_result.equity_curve.index[0]} 到 {best_result.equity_curve.index[-1]}")
        logger.info(f"  总收益率: {best_result.total_return:.2f}%")
        logger.info(f"  夏普比率: {best_result.sharpe_ratio:.2f}")
        logger.info(f"  最大回撤: {best_result.max_drawdown:.2f}%")
        logger.info(f"  胜率: {best_result.win_rate:.2%}")

        return {
            'results': results,
            'average_return': avg_return,
            'average_sharpe': avg_sharpe,
            'average_max_drawdown': avg_max_dd,
            'best_result': best_result
        }

    def save_results(self, result: BacktestResult, filename: str = None):
        """
        保存回测结果到文件

        Args:
            result: 回测结果
            filename: 文件名（可选，默认自动生成）
        """
        if not filename:
            filename = f"backtest_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        # 转换为 DataFrame
        df_trades = pd.DataFrame(result.trades)
        df_trades['date'] = pd.to_datetime(df_trades['date'])

        # 保存交易记录
        df_trades.to_csv(f"{filename}_trades.csv", index=False)
        logger.info(f"✅ 交易记录已保存: {filename}_trades.csv")

        # 保存权益曲线
        df_equity = pd.DataFrame({
            'date': result.equity_curve.index,
            'balance': result.equity_curve.values
        })
        df_equity.to_csv(f"{filename}_equity.csv", index=False)
        logger.info(f"✅ 权益曲线已保存: {filename}_equity.csv")

        # 保存结果摘要
        summary = {
            'initial_balance': result.initial_balance,
            'final_balance': result.final_balance,
            'total_return': result.total_return,
            'total_trades': result.total_trades,
            'win_trades': result.win_trades,
            'loss_trades': result.loss_trades,
            'win_rate': result.win_rate,
            'max_drawdown': result.max_drawdown,
            'sharpe_ratio': result.sharpe_ratio,
            'annual_return': result.annual_return,
            'validations': result.validations
        }

        import json
        with open(f"{filename}_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"✅ 结果摘要已保存: {filename}_summary.json")
