"""
交易信号定义模块
定义各种交易信号类型和格式
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


class SignalType(Enum):
    """信号类型枚举"""
    STRONG_BUY = "强烈买入"
    BUY = "买入"
    HOLD = "持有"
    SELL = "卖出"
    STRONG_SELL = "强烈卖出"

class Direction(Enum):
    """交易方向枚举"""
    LONG = "多头"
    SHORT = "空头"
    NEUTRAL = "中性"

@dataclass
class IndicatorSignal:
    """单个指标信号"""
    name: str  # 指标名称
    value: float  # 指标值
    signal_type: SignalType  # 信号类型
    confidence: float  # 信心度 (0-1)
    timestamp: datetime  # 时间戳

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'value': self.value,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }

@dataclass
class TradingSignal:
    """综合交易信号"""
    symbol: str  # 交易品种
    direction: Direction  # 交易方向
    signal_type: SignalType  # 信号类型
    confidence: float  # 信心度 (0-1)
    indicators: List[IndicatorSignal]  # 各指标信号
    price: float  # 当前价格
    timestamp: datetime  # 时间戳
    reasoning: str  # 推理过程
    action_required: bool  # 是否需要执行操作

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'symbol': self.symbol,
            'direction': self.direction.value,
            'signal_type': self.signal_type.value,
            'confidence': self.confidence,
            'indicators': [ind.to_dict() for ind in self.indicators],
            'price': self.price,
            'timestamp': self.timestamp.isoformat(),
            'reasoning': self.reasoning,
            'action_required': self.action_required
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def get_action(self) -> Optional[str]:
        """获取建议操作"""
        if not self.action_required:
            return None

        if self.signal_type == SignalType.STRONG_BUY:
            return "买入"
        elif self.signal_type == SignalType.BUY:
            return "买入"
        elif self.signal_type == SignalType.SELL:
            return "卖出"
        elif self.signal_type == SignalType.STRONG_SELL:
            return "卖出"
        else:
            return "持有"

class SignalGenerator:
    """信号生成器基类"""

    @staticmethod
    def calculate_signal_strength(indicators: List[IndicatorSignal]) -> float:
        """计算信号强度"""
        if not indicators:
            return 0.0

        total_confidence = sum(ind.confidence for ind in indicators)
        avg_confidence = total_confidence / len(indicators)

        # 计算信号方向一致性
        positive_signals = sum(1 for ind in indicators
                             if ind.signal_type in [SignalType.STRONG_BUY, SignalType.BUY])
        negative_signals = sum(1 for ind in indicators
                             if ind.signal_type in [SignalType.STRONG_SELL, SignalType.SELL])

        direction_factor = 0
        if positive_signals > negative_signals:
            direction_factor = (positive_signals - negative_signals) / len(indicators)
        elif negative_signals > positive_signals:
            direction_factor = -(negative_signals - positive_signals) / len(indicators)

        # 综合信号强度
        signal_strength = avg_confidence * direction_factor
        return max(-1.0, min(1.0, signal_strength))

    @staticmethod
    def determine_signal_type(strength: float, thresholds: Dict = None) -> SignalType:
        """根据强度确定信号类型"""
        if thresholds is None:
            thresholds = {
                'strong_buy': 0.7,
                'buy': 0.3,
                'sell': -0.3,
                'strong_sell': -0.7
            }

        if strength >= thresholds['strong_buy']:
            return SignalType.STRONG_BUY
        elif strength >= thresholds['buy']:
            return SignalType.BUY
        elif strength >= thresholds['sell']:
            return SignalType.SELL
        elif strength >= thresholds['strong_sell']:
            return SignalType.STRONG_SELL
        else:
            return SignalType.HOLD

    @staticmethod
    def create_trading_signal(
        symbol: str,
        price: float,
        indicators: List[IndicatorSignal],
        reasoning: str = "",
        thresholds: Dict = None
    ) -> TradingSignal:
        """创建综合交易信号"""
        # 计算信号强度
        strength = SignalGenerator.calculate_signal_strength(indicators)

        # 确定信号类型
        signal_type = SignalGenerator.determine_signal_type(strength, thresholds)

        # 确定交易方向
        if signal_type in [SignalType.STRONG_BUY, SignalType.BUY]:
            direction = Direction.LONG
        elif signal_type in [SignalType.STRONG_SELL, SignalType.SELL]:
            direction = Direction.SHORT
        else:
            direction = Direction.NEUTRAL

        # 确定是否需要操作
        action_required = signal_type != SignalType.HOLD

        return TradingSignal(
            symbol=symbol,
            direction=direction,
            signal_type=signal_type,
            confidence=abs(strength),
            indicators=indicators,
            price=price,
            timestamp=datetime.now(),
            reasoning=reasoning,
            action_required=action_required
        )

# 信号工具函数
def signal_summary(signals: List[TradingSignal]) -> str:
    """生成信号摘要"""
    if not signals:
        return "暂无交易信号"

    summary = f"\n=== 交易信号总结 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n"

    for signal in signals:
        summary += f"\n品种: {signal.symbol}\n"
        summary += f"信号: {signal.signal_type.value} ({signal.direction.value})\n"
        summary += f"信心度: {signal.confidence:.2f}\n"
        summary += f"当前价格: {signal.price:.2f}\n"
        summary += f"操作建议: {signal.get_action() or '等待'}\n"
        summary += f"推理: {signal.reasoning}\n"

    return summary