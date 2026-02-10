"""
回测系统模块
提供完整的策略回测和验证功能
"""

from .backtest_engine import BacktestEngine, BacktestResult
from .backtest_validator import BacktestValidator
from .backtest_runner import BacktestRunner
from .example_strategies import STRATEGIES, ma_crossover_strategy, rsi_strategy, macd_strategy, bollinger_strategy, combined_strategy

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'BacktestValidator',
    'BacktestRunner',
    'STRATEGIES',
    'ma_crossover_strategy',
    'rsi_strategy',
    'macd_strategy',
    'bollinger_strategy',
    'combined_strategy'
]
