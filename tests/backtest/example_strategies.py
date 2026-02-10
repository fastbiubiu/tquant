"""
示例策略模块
提供各种回测策略示例
"""

import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime

from tquant.utils.indicators import TechnicalIndicators


def ma_crossover_strategy(indicators: TechnicalIndicators, current_price: float, date: datetime) -> List[Dict]:
    """
    均线交叉策略（金叉买入，死叉卖出）

    Args:
        indicators: 技术指标
        current_price: 当前价格
        date: 当前日期

    Returns:
        List[Dict]: 交易信号列表
    """
    signals = []

    # 获取MA指标
    ma_short = indicators.get_ma('ma_20', current_price)
    ma_long = indicators.get_ma('ma_60', current_price)

    if ma_short and ma_long:
        # 金叉：短期均线上穿长期均线
        if ma_short > ma_long:
            # 检查是否已经持仓（这里简化，假设每次买入都成功）
            signals.append({
                'symbol': 'SHFE.rb',  # 螺纹钢
                'action': 'BUY',
                'volume': 1,
                'price': current_price
            })

        # 死叉：短期均线下穿长期均线
        elif ma_short < ma_long:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'SELL',
                'volume': 1,
                'price': current_price
            })

    return signals


def rsi_strategy(indicators: TechnicalIndicators, current_price: float, date: datetime) -> List[Dict]:
    """
    RSI策略（超买超卖）

    Args:
        indicators: 技术指标
        current_price: 当前价格
        date: 当前日期

    Returns:
        List[Dict]: 交易信号列表
    """
    signals = []

    # 获取RSI指标
    rsi = indicators.get_rsi('rsi_14')

    if rsi:
        # RSI超过70超买，卖出
        if rsi > 70:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'SELL',
                'volume': 1,
                'price': current_price
            })

        # RSI低于30超卖，买入
        elif rsi < 30:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'BUY',
                'volume': 1,
                'price': current_price
            })

    return signals


def macd_strategy(indicators: TechnicalIndicators, current_price: float, date: datetime) -> List[Dict]:
    """
    MACD策略（金叉死叉）

    Args:
        indicators: 技术指标
        current_price: 当前价格
        date: 当前日期

    Returns:
        List[Dict]: 交易信号列表
    """
    signals = []

    # 获取MACD指标
    macd_diff = indicators.get_macd_diff()

    if macd_diff and len(macd_diff) >= 2:
        # 金叉：MACD线上穿信号线
        if macd_diff[-1] > macd_diff[-2] and macd_diff[-1] > 0:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'BUY',
                'volume': 1,
                'price': current_price
            })

        # 死叉：MACD线下穿信号线
        elif macd_diff[-1] < macd_diff[-2] and macd_diff[-1] < 0:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'SELL',
                'volume': 1,
                'price': current_price
            })

    return signals


def bollinger_strategy(indicators: TechnicalIndicators, current_price: float, date: datetime) -> List[Dict]:
    """
    布林带策略（突破）

    Args:
        indicators: 技术指标
        current_price: 当前价格
        date: 当前日期

    Returns:
        List[Dict]: 交易信号列表
    """
    signals = []

    # 获取布林带指标
    bollinger_upper = indicators.get_bollinger_upper()
    bollinger_lower = indicators.get_bollinger_lower()

    if bollinger_upper and bollinger_lower:
        # 上轨突破，卖出
        if current_price > bollinger_upper:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'SELL',
                'volume': 1,
                'price': current_price
            })

        # 下轨突破，买入
        elif current_price < bollinger_lower:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'BUY',
                'volume': 1,
                'price': current_price
            })

    return signals


def combined_strategy(indicators: TechnicalIndicators, current_price: float, date: datetime) -> List[Dict]:
    """
    组合策略（MA + RSI + MACD）

    Args:
        indicators: 技术指标
        current_price: 当前价格
        date: 当前日期

    Returns:
        List[Dict]: 交易信号列表
    """
    signals = []

    # 获取所有指标
    ma_short = indicators.get_ma('ma_20', current_price)
    ma_long = indicators.get_ma('ma_60', current_price)
    rsi = indicators.get_rsi('rsi_14')
    macd_diff = indicators.get_macd_diff()

    # 金叉信号
    if ma_short and ma_long and ma_short > ma_long:
        # 只有在RSI不超过70时才买入
        if not rsi or rsi < 70:
            signals.append({
                'symbol': 'SHFE.rb',
                'action': 'BUY',
                'volume': 1,
                'price': current_price
            })

    # 死叉信号
    if ma_short and ma_long and ma_short < ma_long:
        signals.append({
            'symbol': 'SHFE.rb',
            'action': 'SELL',
            'volume': 1,
            'price': current_price
        })

    # RSI超卖信号
    if rsi and rsi < 30:
        signals.append({
            'symbol': 'SHFE.rb',
            'action': 'BUY',
            'volume': 1,
            'price': current_price
        })

    # MACD金叉信号
    if macd_diff and len(macd_diff) >= 2 and macd_diff[-1] > macd_diff[-2] and macd_diff[-1] > 0:
        signals.append({
            'symbol': 'SHFE.rb',
            'action': 'BUY',
            'volume': 1,
            'price': current_price
        })

    # MACD死叉信号
    if macd_diff and len(macd_diff) >= 2 and macd_diff[-1] < macd_diff[-2] and macd_diff[-1] < 0:
        signals.append({
            'symbol': 'SHFE.rb',
            'action': 'SELL',
            'volume': 1,
            'price': current_price
        })

    return signals


# 策略字典，方便批量回测
STRATEGIES = {
    'MA_Crossover': ma_crossover_strategy,
    'RSI': rsi_strategy,
    'MACD': macd_strategy,
    'Bollinger': bollinger_strategy,
    'Combined': combined_strategy
}


def run_all_strategies():
    """运行所有策略的示例"""
    from tests.backtest.backtest_runner import BacktestRunner

    # 创建回测运行器
    runner = BacktestRunner(
        initial_balance=100000.0,
        start_dt=datetime(2020, 1, 1),
        end_dt=datetime(2023, 12, 31),
        commission_rate=0.0003
    )

    # 运行所有策略
    results_df = runner.run_multiple_strategies(
        strategy_dict=STRATEGIES,
        symbols=['SHFE.rb', 'SHFE.ag'],  # 螺纹钢、白银
        periods=['1d', '4h'],
        verbose=True
    )

    # 打印比较报告
    print("\n" + "="*80)
    print("📊 策略比较报告")
    print("="*80)
    print(results_df)

    return results_df


if __name__ == "__main__":
    # 运行所有策略示例
    run_all_strategies()
