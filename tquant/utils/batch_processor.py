"""
批量处理器
用于批量处理多个信号和交易,优化系统性能
"""

import logging
from datetime import datetime
from typing import List, Dict, Union

import numpy as np

from tquant.config import Config
from tquant.utils.cache_manager import CacheManager
from tquant.utils.signals import TradingSignal

logger = logging.getLogger(__name__)


class SignalBatchProcessor:
    """信号批量处理器"""

    def __init__(self, config: Union[Dict, Config], cache_manager: CacheManager = None):
        """
        初始化信号批量处理器

        Args:
            config: 配置(字典或 Config 对象)
            cache_manager: 缓存管理器
        """
        self.config = config
        self.cache_manager = cache_manager

        # 批量处理配置
        if isinstance(config, Config):
            # 从 Config 对象获取配置
            self.batch_size = 10  # 默认值
            self.min_confidence = 0.5  # 默认值
            self.max_signals_per_batch = 50  # 默认值
            self.aggregation_method = 'average'  # 默认值
        else:
            # 从字典获取配置
            self.batch_size = config.get('batch_size', 10)
            self.min_confidence = config.get('min_confidence', 0.5)
            self.max_signals_per_batch = config.get('max_signals_per_batch', 50)
            self.aggregation_method = config.get('aggregation_method', 'average')

        logger.info(f"信号批量处理器初始化完成 (批次大小: {self.batch_size}, 最小置信度: {self.min_confidence})")

    def batch_analyze_signals(
        self,
        signals: List[TradingSignal],
        batch_size: int = None
    ) -> List[List[TradingSignal]]:
        """
        将信号批量分组

        Args:
            signals: 信号列表
            batch_size: 批次大小

        Returns:
            批次列表
        """
        batch_size = batch_size or self.batch_size
        return [signals[i:i + batch_size] for i in range(0, len(signals), batch_size)]

    def aggregate_signals(
        self,
        signals: List[TradingSignal],
        method: str = None
    ) -> TradingSignal:
        """
        聚合多个信号

        Args:
            signals: 信号列表
            method: 聚合方法

        Returns:
            聚合后的信号
        """
        method = method or self.aggregation_method

        if not signals:
            raise ValueError("信号列表不能为空")

        # 创建聚合信号
        aggregated_signal = TradingSignal(
            symbol=signals[0].symbol,
            price=sum(s.price for s in signals) / len(signals),
            confidence=sum(s.confidence for s in signals) / len(signals),
            signal_type=self._aggregate_signal_type(signals, method),
            direction=self._aggregate_direction(signals),
            indicators=self._aggregate_indicators(signals),
            reasoning=self._aggregate_reasoning(signals),
            thresholds=signals[0].thresholds if hasattr(signals[0], 'thresholds') else None,
            action_required=self._calculate_action_required(signals)
        )

        return aggregated_signal

    def _aggregate_signal_type(self, signals: List[TradingSignal], method: str) -> str:
        """
        聚合信号类型

        Args:
            signals: 信号列表
            method: 聚合方法

        Returns:
            聚合后的信号类型
        """
        # 统计各信号类型的数量
        counts = {'STRONG_BUY': 0, 'BUY': 0, 'HOLD': 0, 'SELL': 0, 'STRONG_SELL': 0}

        for signal in signals:
            if hasattr(signal, 'signal_type') and signal.signal_type:
                counts[signal.signal_type.value] += 1

        # 根据聚合方法确定结果
        if method == 'average':
            # 取数量最多的信号类型
            max_count = max(counts.values())
            max_types = [k for k, v in counts.items() if v == max_count]
            if len(max_types) == 1:
                return max_types[0]
            else:
                # 如果有多个相同计数,取置信度最高的
                return max(
                    max_types,
                    key=lambda t: max([s.confidence for s in signals if hasattr(s, 'signal_type') and s.signal_type and s.signal_type.value == t])
                )

        elif method == 'max':
            # 取置信度最高的信号类型
            return max(counts.keys(), key=lambda t: counts[t])

        else:
            # 默认：取数量最多的
            return max(counts.keys(), key=lambda t: counts[t])

    def _aggregate_direction(self, signals: List[TradingSignal]) -> str:
        """
        聚合交易方向

        Args:
            signals: 信号列表

        Returns:
            聚合后的方向
        """
        buy_signals = sum(1 for s in signals if hasattr(s, 'direction') and s.direction in ['LONG', 'BUY'])
        sell_signals = sum(1 for s in signals if hasattr(s, 'direction') and s.direction in ['SHORT', 'SELL'])

        if buy_signals > sell_signals:
            return 'LONG'
        elif sell_signals > buy_signals:
            return 'SHORT'
        else:
            return 'NEUTRAL'

    def _aggregate_indicators(self, signals: List[TradingSignal]) -> Dict:
        """
        聚合技术指标

        Args:
            signals: 信号列表

        Returns:
            聚合后的指标
        """
        if not signals:
            return {}

        # 收集所有指标
        indicators_dict = {}
        for signal in signals:
            if hasattr(signal, 'indicators') and signal.indicators:
                for key, value in signal.indicators.items():
                    if key not in indicators_dict:
                        indicators_dict[key] = []
                    indicators_dict[key].append(value)

        # 计算平均指标
        aggregated = {}
        for key, values in indicators_dict.items():
            if isinstance(values[0], (int, float)):
                aggregated[key] = np.mean(values)
            elif isinstance(values[0], str):
                aggregated[key] = values[0]  # 保持字符串指标不变
            else:
                aggregated[key] = values

        return aggregated

    def _aggregate_reasoning(self, signals: List[TradingSignal]) -> str:
        """
        聚合分析理由

        Args:
            signals: 信号列表

        Returns:
            聚合后的理由
        """
        reasons = []
        for signal in signals:
            if hasattr(signal, 'reasoning') and signal.reasoning:
                reasons.append(signal.reasoning)

        if not reasons:
            return "基于多个信号的综合分析"

        return " | ".join(reasons)

    def _calculate_action_required(self, signals: List[TradingSignal]) -> bool:
        """
        计算是否需要执行操作

        Args:
            signals: 信号列表

        Returns:
            是否需要执行
        """
        # 如果所有信号都是 HOLD,则不需要执行
        hold_count = sum(1 for s in signals if hasattr(s, 'signal_type') and s.signal_type and s.signal_type.value == 'HOLD')

        return hold_count < len(signals) * 0.5  # 如果超过 50% 是 HOLD,则不需要执行

    def filter_low_confidence_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        过滤低置信度信号

        Args:
            signals: 信号列表

        Returns:
            过滤后的信号列表
        """
        min_conf = self.min_confidence

        filtered = [
            s for s in signals
            if s.confidence >= min_conf
        ]

        logger.info(f"信号过滤: {len(signals)} -> {len(filtered)} (置信度 >= {min_conf})")

        return filtered

    def sort_signals_by_confidence(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        按置信度排序信号

        Args:
            signals: 信号列表

        Returns:
            排序后的信号列表
        """
        return sorted(signals, key=lambda s: s.confidence, reverse=True)


class TradeBatchProcessor:
    """交易批量处理器"""

    def __init__(self, config: Dict):
        """
        初始化交易批量处理器

        Args:
            config: 配置
        """
        self.config = config

        # 批量处理配置
        self.max_trades_per_batch = config.get('max_trades_per_batch', 10)
        self.max_daily_trades = config.get('max_daily_trades', 20)

        # 交易聚合配置
        self.aggregation_method = config.get('aggregation_method', 'average')  # 'average', 'conservative', 'aggressive'

        logger.info(f"交易批量处理器初始化完成 (每批最大交易数: {self.max_trades_per_batch})")

    def batch_execute_trades(
        self,
        trade_results: List[Dict],
        batch_size: int = None
    ) -> List[List[Dict]]:
        """
        将交易结果批量分组

        Args:
            trade_results: 交易结果列表
            batch_size: 批次大小

        Returns:
            批次列表
        """
        batch_size = batch_size or self.max_trades_per_batch
        return [trade_results[i:i + batch_size] for i in range(0, len(trade_results), batch_size)]

    def aggregate_trades(
        self,
        trade_results: List[Dict],
        method: str = None
    ) -> Dict:
        """
        聚合交易结果

        Args:
            trade_results: 交易结果列表
            method: 聚合方法

        Returns:
            聚合后的结果
        """
        method = method or self.aggregation_method

        if not trade_results:
            return {
                'total_trades': 0,
                'successful_trades': 0,
                'failed_trades': 0,
                'total_profit': 0.0,
                'win_rate': 0.0
            }

        # 统计结果
        successful_trades = [t for t in trade_results if t.get('success', False)]
        failed_trades = [t for t in trade_results if not t.get('success', False)]

        # 计算总利润(如果有利润字段)
        total_profit = sum(
            t.get('profit', 0) for t in successful_trades
            if 'profit' in t and isinstance(t['profit'], (int, float))
        )

        # 计算胜率
        win_rate = len(successful_trades) / len(trade_results) * 100 if trade_results else 0

        # 根据聚合方法返回不同信息
        if method == 'conservative':
            return {
                'status': 'CAUTIOUS',
                'message': '交易结果良好,建议保持当前策略',
                'total_trades': len(trade_results),
                'successful_trades': len(successful_trades),
                'failed_trades': len(failed_trades),
                'total_profit': total_profit,
                'win_rate': win_rate
            }

        elif method == 'aggressive':
            return {
                'status': 'CONFIDENT',
                'message': '交易表现强劲,可考虑增加仓位',
                'total_trades': len(trade_results),
                'successful_trades': len(successful_trades),
                'failed_trades': len(failed_trades),
                'total_profit': total_profit,
                'win_rate': win_rate
            }

        else:  # average
            return {
                'status': 'MODERATE',
                'message': '交易表现平稳',
                'total_trades': len(trade_results),
                'successful_trades': len(successful_trades),
                'failed_trades': len(failed_trades),
                'total_profit': total_profit,
                'win_rate': win_rate
            }

    def validate_trade_batch(self, trade_results: List[Dict]) -> Dict:
        """
        验证交易批次

        Args:
            trade_results: 交易结果列表

        Returns:
            验证结果
        """
        errors = []
        warnings = []
        success_count = sum(1 for t in trade_results if t.get('success', False))
        fail_count = len(trade_results) - success_count

        # 检查错误
        if fail_count > 0:
            errors.append(f"交易批次中有 {fail_count} 个失败")

        # 检查警告
        if fail_count > len(trade_results) * 0.3:
            warnings.append("失败率超过 30%,建议检查系统状态")

        if fail_count > len(trade_results) * 0.5:
            errors.append("失败率超过 50%,系统可能存在问题")

        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'success_rate': success_count / len(trade_results) * 100 if trade_results else 0
        }

    def sort_trades_by_time(self, trade_results: List[Dict]) -> List[Dict]:
        """
        按时间排序交易结果

        Args:
            trade_results: 交易结果列表

        Returns:
            排序后的交易结果
        """
        return sorted(
            trade_results,
            key=lambda t: t.get('timestamp', datetime.now()),
            reverse=True
        )

    def filter_successful_trades(self, trade_results: List[Dict]) -> List[Dict]:
        """
        过滤成功的交易

        Args:
            trade_results: 交易结果列表

        Returns:
            成功的交易列表
        """
        return [t for t in trade_results if t.get('success', False)]
