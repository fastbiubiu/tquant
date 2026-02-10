"""
回测验证器模块
提供策略性能评估和验证功能
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

from tests.backtest.backtest_engine import BacktestResult
from tquant.utils.indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class BacktestValidator:
    """回测验证器类"""

    def __init__(self, threshold: Dict[str, float] = None):
        """
        初始化回测验证器

        Args:
            threshold: 验证阈值配置
                {
                    'min_total_return': 0,        # 最小总收益率
                    'max_drawdown': 20,           # 最大回撤
                    'min_sharpe_ratio': 1.0,      # 最小夏普比率
                    'min_win_rate': 0.4,          # 最小胜率
                    'min_annual_return': 10.0     # 最小年化收益
                }
        """
        self.threshold = threshold or {
            'min_total_return': 0,
            'max_drawdown': 20,
            'min_sharpe_ratio': 1.0,
            'min_win_rate': 0.4,
            'min_annual_return': 10.0
        }

        self.indicators = TechnicalIndicators()

    def validate(self, result: BacktestResult) -> Dict[str, bool]:
        """
        验证回测结果

        Args:
            result: 回测结果

        Returns:
            Dict[str, bool]: 验证结果
                {
                    'total_return': bool,       # 总收益率验证
                    'max_drawdown': bool,       # 最大回撤验证
                    'sharpe_ratio': bool,       # 夏普比率验证
                    'win_rate': bool,           # 胜率验证
                    'annual_return': bool,      # 年化收益验证
                    'overall': bool             # 整体验证
                }
        """
        if not result:
            logger.error("❌ 回测结果为空")
            return {'overall': False}

        validations = {}

        # 总收益率验证
        validations['total_return'] = result.total_return >= self.threshold['min_total_return']
        logger.info(f"📊 总收益率: {result.total_return:.2f}% (阈值: {self.threshold['min_total_return']}%) - {'✅' if validations['total_return'] else '❌'}")

        # 最大回撤验证
        validations['max_drawdown'] = result.max_drawdown <= self.threshold['max_drawdown']
        logger.info(f"📉 最大回撤: {result.max_drawdown:.2f}% (阈值: {self.threshold['max_drawdown']}%) - {'✅' if validations['max_drawdown'] else '❌'}")

        # 夏普比率验证
        validations['sharpe_ratio'] = result.sharpe_ratio >= self.threshold['min_sharpe_ratio']
        logger.info(f"📈 夏普比率: {result.sharpe_ratio:.2f} (阈值: {self.threshold['min_sharpe_ratio']}) - {'✅' if validations['sharpe_ratio'] else '❌'}")

        # 胜率验证
        validations['win_rate'] = result.win_rate >= self.threshold['min_win_rate']
        logger.info(f"🎯 胜率: {result.win_rate:.2%} (阈值: {self.threshold['min_win_rate']:.0%}) - {'✅' if validations['win_rate'] else '❌'}")

        # 年化收益验证
        validations['annual_return'] = result.annual_return >= self.threshold['min_annual_return']
        logger.info(f"💰 年化收益: {result.annual_return:.2f}% (阈值: {self.threshold['min_annual_return']}%) - {'✅' if validations['annual_return'] else '❌'}")

        # 整体验证
        validations['overall'] = all(validations.values())

        # 生成详细报告
        self._generate_report(result, validations)

        return validations

    def _generate_report(self, result: BacktestResult, validations: Dict[str, bool]):
        """生成详细报告"""
        logger.info("\n" + "="*60)
        logger.info("📋 回测验证报告")
        logger.info("="*60)

        # 基本统计
        logger.info(f"\n📊 基本统计:")
        logger.info(f"  - 初始资金: {result.initial_balance:,.2f}")
        logger.info(f"  - 最终权益: {result.final_balance:,.2f}")
        logger.info(f"  - 总收益: {result.total_return:.2f}%")
        logger.info(f"  - 总交易次数: {result.total_trades}")
        logger.info(f"  - 盈利交易: {result.win_trades}")
        logger.info(f"  - 亏损交易: {result.loss_trades}")
        logger.info(f"  - 胜率: {result.win_rate:.2%}")

        # 风险指标
        logger.info(f"\n📈 风险指标:")
        logger.info(f"  - 最大回撤: {result.max_drawdown:.2f}%")
        logger.info(f"  - 夏普比率: {result.sharpe_ratio:.2f}")
        logger.info(f"  - 年化收益: {result.annual_return:.2f}%")

        # 验证结果
        logger.info(f"\n✅ 验证结果:")
        for key, value in validations.items():
            status = "✅ 通过" if value else "❌ 失败"
            logger.info(f"  - {key}: {status}")

        logger.info("="*60 + "\n")

    def get_performance_metrics(self, result: BacktestResult) -> Dict[str, Any]:
        """
        获取性能指标

        Args:
            result: 回测结果

        Returns:
            Dict[str, Any]: 性能指标字典
        """
        if not result:
            return {}

        # 计算各项指标
        metrics = {
            'trades': {
                'total': result.total_trades,
                'win_rate': result.win_rate,
                'profit_factor': self._calculate_profit_factor(result),
                'avg_win': self._calculate_avg_win(result),
                'avg_loss': self._calculate_avg_loss(result),
            },
            'returns': {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'max_drawdown': result.max_drawdown,
                'sharpe_ratio': result.sharpe_ratio,
            },
            'risk_metrics': {
                'max_drawdown': result.max_drawdown,
                'volatility': self._calculate_volatility(result),
                'beta': self._calculate_beta(result),
            }
        }

        return metrics

    def _calculate_profit_factor(self, result: BacktestResult) -> float:
        """计算盈利因子（总盈利 / 总亏损）"""
        sell_trades = [t for t in result.trades if t['action'] in ['SELL', 'CLOSE']]
        if not sell_trades:
            return 0

        total_profit = sum([
            t['price'] * t['volume'] - t['commission']
            for t in sell_trades if t['price'] * t['volume'] - t['commission'] > 0
        ])

        total_loss = abs(sum([
            t['price'] * t['volume'] - t['commission']
            for t in sell_trades if t['price'] * t['volume'] - t['commission'] < 0
        ]))

        return total_profit / total_loss if total_loss > 0 else 0

    def _calculate_avg_win(self, result: BacktestResult) -> float:
        """计算平均盈利金额"""
        sell_trades = [t for t in result.trades if t['action'] in ['SELL', 'CLOSE']]
        profits = [
            t['price'] * t['volume'] - t['commission']
            for t in sell_trades if t['price'] * t['volume'] - t['commission'] > 0
        ]

        return sum(profits) / len(profits) if profits else 0

    def _calculate_avg_loss(self, result: BacktestResult) -> float:
        """计算平均亏损金额"""
        sell_trades = [t for t in result.trades if t['action'] in ['SELL', 'CLOSE']]
        losses = [
            t['price'] * t['volume'] - t['commission']
            for t in sell_trades if t['price'] * t['volume'] - t['commission'] < 0
        ]

        return sum(abs(losses)) / len(losses) if losses else 0

    def _calculate_volatility(self, result: BacktestResult) -> float:
        """计算波动率"""
        returns = result.equity_curve.pct_change().dropna()

        if len(returns) < 2:
            return 0

        return returns.std() * np.sqrt(252) * 100  # 年化波动率

    def _calculate_beta(self, result: BacktestResult) -> float:
        """计算Beta值（相对于市场）"""
        # 简化计算：使用市场波动率作为基准
        market_returns = result.equity_curve.pct_change().dropna()
        strategy_returns = result.equity_curve.pct_change().dropna()

        if len(market_returns) < 2 or len(strategy_returns) < 2:
            return 0

        covariance = strategy_returns.cov(market_returns)
        market_variance = market_returns.var()

        return covariance / market_variance if market_variance > 0 else 0

    def generate_visualizations(self, result: BacktestResult, output_path: str = None):
        """
        生成可视化图表

        Args:
            result: 回测结果
            output_path: 输出路径（可选）
        """
        try:
            import matplotlib.pyplot as plt

            # 创建图表
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))

            # 1. 权益曲线
            axes[0].plot(result.equity_curve.index, result.equity_curve.values, label='权益曲线', linewidth=2)
            axes[0].axhline(y=result.initial_balance, color='r', linestyle='--', label='初始资金')
            axes[0].set_title('权益曲线')
            axes[0].set_ylabel('资金 (¥)')
            axes[0].legend()
            axes[0].grid(True)

            # 2. 回撤图
            equity = result.equity_curve.values
            cumulative_max = np.maximum.accumulate(equity)
            drawdown = (equity - cumulative_max) / cumulative_max * 100
            axes[1].fill_between(result.equity_curve.index, drawdown, 0, color='red', alpha=0.3)
            axes[1].set_title(f'最大回撤: {result.max_drawdown:.2f}%')
            axes[1].set_ylabel('回撤 (%)')
            axes[1].set_xlabel('日期')
            axes[1].grid(True)

            # 3. 交易统计
            sell_trades = [t for t in result.trades if t['action'] in ['SELL', 'CLOSE']]
            profits = [
                t['price'] * t['volume'] - t['commission']
                for t in sell_trades if t['price'] * t['volume'] - t['commission'] > 0
            ]
            losses = [
                t['price'] * t['volume'] - t['commission']
                for t in sell_trades if t['price'] * t['volume'] - t['commission'] < 0
            ]

            bars = axes[2].bar(
                ['盈利交易', '亏损交易'],
                [len(profits), len(losses)],
                color=['green', 'red']
            )
            axes[2].set_title('交易统计')
            axes[2].set_ylabel('交易次数')
            axes[2].grid(True, axis='y')

            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                axes[2].text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom')

            plt.tight_layout()

            # 保存或显示
            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches='tight')
                logger.info(f"📊 图表已保存到: {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            logger.warning("matplotlib未安装，无法生成可视化图表")

    def compare_strategies(self, results: List[BacktestResult]) -> pd.DataFrame:
        """
        比较多个策略的性能

        Args:
            results: 多个回测结果列表

        Returns:
            pd.DataFrame: 性能比较表
        """
        data = []

        for result in results:
            data.append({
                '初始资金': result.initial_balance,
                '最终权益': result.final_balance,
                '总收益率': f"{result.total_return:.2f}%",
                '总交易次数': result.total_trades,
                '胜率': f"{result.win_rate:.2%}",
                '最大回撤': f"{result.max_drawdown:.2f}%",
                '夏普比率': f"{result.sharpe_ratio:.2f}",
                '年化收益': f"{result.annual_return:.2f}%"
            })

        df = pd.DataFrame(data)

        # 按收益率排序
        df = df.sort_values('总收益率', key=lambda x: x.str.rstrip('%').astype(float), ascending=False)

        return df
