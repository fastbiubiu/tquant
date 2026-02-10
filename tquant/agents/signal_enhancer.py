"""
信号增强器
使用 LLM 增强技术信号的质量和可信度
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Any, Optional

from tquant.utils.signals import TradingSignal, SignalType

logger = logging.getLogger(__name__)

@dataclass
class EnhancedSignal:
    """增强的信号"""
    original_signal: TradingSignal
    enhanced_signal: TradingSignal
    confidence_boost: float  # 置信度提升
    reasoning: str
    risk_level: str
    market_context: str
    signal_strength: str  # 强/中/弱

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'symbol': self.original_signal.symbol,
            'original_direction': self.original_signal.direction,
            'original_confidence': self.original_signal.confidence,
            'enhanced_direction': self.enhanced_signal.direction,
            'enhanced_confidence': self.enhanced_signal.confidence,
            'confidence_boost': self.confidence_boost,
            'reasoning': self.reasoning,
            'risk_level': self.risk_level,
            'market_context': self.market_context,
            'signal_strength': self.signal_strength,
            'timestamp': datetime.now().isoformat()
        }

class SignalEnhancer:
    """信号增强器 - 使用 LLM 增强交易信号"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化信号增强器

        Args:
            config_path: 配置文件路径
        """
        from config import get_config, Config
        self.config: Config = get_config()

        # 获取 LLM 配置
        self.llm_config = self.config.get('llm', {})
        self.deep_model = self.llm_config.get('gpt4o', {})
        self.quick_model = self.llm_config.get('gpt4o_mini', {})

        # 信号增强配置
        self.enhancement_config = self.config.get('analysis', {}).get('signal_generation', {})
        self.min_confidence = self.enhancement_config.get('min_confidence', 0.5)
        self.max_signals_per_symbol = self.enhancement_config.get('max_signals_per_symbol', 3)

        # 系统提示词
        self.system_prompt = self._get_system_prompt()

        logger.info("信号增强器初始化完成")

    def _get_system_prompt(self) -> str:
        """获取系统提示词"""
        return """你是一个专业的技术信号增强专家,专注于金融市场分析。

你的任务：
1. 评估技术分析信号的可信度
2. 结合市场上下文增强信号质量
3. 识别潜在的假信号和反转机会
4. 提供详细的分析理由

分析原则：
- 基于技术指标,但不要过度依赖单一指标
- 考虑市场整体趋势和成交量
- 注意支撑位和阻力位的重要性
- 关注新闻和事件影响(如果有)
- 保持客观,避免主观臆断

输出格式：
{
    "signals": {
        "BUY": 0.85,
        "SELL": 0.15,
        "HOLD": 0.30
    },
    "reasoning": "详细的分析理由...",
    "risk_level": "低/中/高",
    "market_context": "市场上下文分析...",
    "signal_strength": "强/中/弱"
}

说明：
- signals: 三个方向的概率值,总和应为1.0
- reasoning: 至少50字的详细分析
- risk_level: 基于波动率和市场条件评估
- market_context: 当前市场环境分析
- signal_strength: 基于技术分析结果评估"""

    def enhance_signals(self, signals: List[TradingSignal],
                       market_data: Optional[Dict] = None) -> List[EnhancedSignal]:
        """
        增强交易信号

        Args:
            signals: 原始交易信号列表
            market_data: 市场数据(可选)

        Returns:
            增强后的信号列表
        """
        if not signals:
            return []

        enhanced_signals = []

        for signal in signals:
            try:
                # 检查是否需要增强
                if signal.confidence >= 0.9:
                    # 高置信度信号,直接使用
                    enhanced_signal = self._create_enhanced_signal(
                        signal, signal.confidence,
                        "信号置信度已很高,无需额外增强",
                        "低", "强"
                    )
                    enhanced_signals.append(enhanced_signal)
                    continue

                # 调用 LLM 进行增强
                enhanced_signal = self._llm_enhance_signal(signal, market_data)
                enhanced_signals.append(enhanced_signal)

            except Exception as e:
                logger.error(f"增强信号失败 {symbol}: {e}")
                # 使用原始信号作为后备
                enhanced_signal = self._create_enhanced_signal(
                    signal, signal.confidence,
                    "信号增强失败,使用原始信号",
                    "中", "中"
                )
                enhanced_signals.append(enhanced_signal)

        # 按置信度排序
        enhanced_signals.sort(key=lambda x: x.enhanced_signal.confidence, reverse=True)

        # 限制每个品种的最大信号数
        final_signals = self._filter_signals(enhanced_signals)

        logger.info(f"成功增强 {len(final_signals)} 个信号")
        return final_signals

    def _llm_enhance_signal(self, signal: TradingSignal,
                           market_data: Optional[Dict] = None) -> EnhancedSignal:
        """
        使用 LLM 增强单个信号

        Args:
            signal: 原始信号
            market_data: 市场数据

        Returns:
            增强后的信号
        """
        from utils.llm_client import LLMClient

        # 构建提示词
        prompt = self._build_enhancement_prompt(signal, market_data)

        # 选择模型
        if signal.confidence < 0.6:
            # 低置信度使用深度分析
            model_config = self.deep_model
        else:
            # 中高置信度使用快速分析
            model_config = self.quick_model

        # 创建 LLM 客户端
        llm_client = LLMClient(
            model=model_config['model'],
            api_key=model_config['api_key'],
            temperature=model_config['temperature'],
            max_tokens=model_config['max_tokens']
        )

        # 调用 LLM
        response = llm_client.invoke([
            ("system", self.system_prompt),
            ("user", prompt)
        ])

        # 解析响应
        enhanced_data = self._parse_enhancement_response(response.content)

        # 创建增强信号
        enhanced_signal = TradingSignal(
            symbol=signal.symbol,
            direction=self._get_signal_direction(enhanced_data['signals']),
            signal_type=self._map_direction_to_signal_type(enhanced_data['signals']),
            confidence=enhanced_data['signals'][self._get_signal_direction(enhanced_data['signals'])],
            indicators=signal.indicators,
            price=signal.price,
            timestamp=datetime.now(),
            reasoning=enhanced_data['reasoning'],
            action_required=True
        )

        return EnhancedSignal(
            original_signal=signal,
            enhanced_signal=enhanced_signal,
            confidence_boost=enhanced_signal.confidence - signal.confidence,
            reasoning=enhanced_data['reasoning'],
            risk_level=enhanced_data['risk_level'],
            market_context=enhanced_data['market_context'],
            signal_strength=enhanced_data['signal_strength']
        )

    def _build_enhancement_prompt(self, signal: TradingSignal,
                               market_data: Optional[Dict] = None) -> str:
        """构建信号增强提示词"""

        # 汇总技术指标
        indicators_summary = self._summarize_indicators(signal.indicators)

        prompt = f"""
请增强以下交易信号：

品种：{signal.symbol}
当前价格：{signal.price}
原始信号方向：{signal.direction}
原始信号置信度：{signal.confidence:.2%}
技术指标分析：
{indicators_summary}

"""

        if market_data:
            prompt += f"市场数据：\n{json.dumps(market_data, ensure_ascii=False, indent=2)}"

        prompt += """Please provide enhanced analysis based on the above information."""

        return prompt

    def _summarize_indicators(self, indicators: List[Dict]) -> str:
        """汇总技术指标"""
        if not indicators:
            return "无技术指标数据"

        summary = []
        for indicator in indicators[:10]:  # 只取前10个指标
            name = indicator.get('name', 'Unknown')
            value = indicator.get('value', 0)
            signal_type = indicator.get('signal_type', 'Unknown')

            if isinstance(value, (int, float)):
                summary.append(f"- {name}: {value:.2f} ({signal_type})")
            else:
                summary.append(f"- {name}: {signal_type}")

        return "\n".join(summary)

    def _parse_enhancement_response(self, response: str) -> Dict[str, Any]:
        """解析增强响应"""
        try:
            # 尝试提取 JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)

            # 如果没有找到 JSON,进行文本解析
            return self._parse_text_response(response)

        except json.JSONDecodeError:
            logger.warning("LLM 返回的 JSON 格式无效,使用文本解析")
            return self._parse_text_response(response)

    def _parse_text_response(self, response: str) -> Dict[str, Any]:
        """解析文本响应"""
        # 默认值
        default_response = {
            "signals": {"BUY": 0.33, "SELL": 0.33, "HOLD": 0.34},
            "reasoning": "无法解析 LLM 响应,使用默认值",
            "risk_level": "中",
            "market_context": "市场上下文分析失败",
            "signal_strength": "中"
        }

        # 尝试提取信号概率
        import re

        # 查找 BUY/SELL/HOLD 的概率
        buy_match = re.search(r'["\']BUY["\']:\s*(\d+\.?\d*)', response)
        sell_match = re.search(r'["\']SELL["\']:\s*(\d+\.?\d*)', response)
        hold_match = re.search(r'["\']HOLD["\']:\s*(\d+\.?\d*)', response)

        if buy_match and sell_match and hold_match:
            buy_prob = float(buy_match.group(1))
            sell_prob = float(sell_match.group(1))
            hold_prob = float(hold_match.group(1))

            # 归一化
            total = buy_prob + sell_prob + hold_prob
            if total > 0:
                default_response["signals"] = {
                    "BUY": buy_prob / total,
                    "SELL": sell_prob / total,
                    "HOLD": hold_prob / total
                }

        # 提取风险等级
        risk_match = re.search(r'["\']risk_level["\']:\s*["\']([^"\']+)["\']', response)
        if risk_match:
            default_response["risk_level"] = risk_match.group(1)

        # 提取信号强度
        strength_match = re.search(r'["\']signal_strength["\']:\s*["\']([^"\']+)["\']', response)
        if strength_match:
            default_response["signal_strength"] = strength_match.group(1)

        return default_response

    def _get_signal_direction(self, signals: Dict[str, float]) -> str:
        """获取信号方向"""
        return max(signals, key=signals.get)

    def _map_direction_to_signal_type(self, signals: Dict[str, float]) -> SignalType:
        """将方向映射到信号类型"""
        direction = self._get_signal_direction(signals)

        if direction == "BUY":
            return SignalType.BUY
        elif direction == "SELL":
            return SignalType.SELL
        else:
            return SignalType.HOLD

    def _create_enhanced_signal(self, original_signal: TradingSignal,
                               confidence: float, reasoning: str,
                               risk_level: str, signal_strength: str) -> EnhancedSignal:
        """创建增强信号(fallback方案)"""

        # 保持原始信号方向
        direction = original_signal.direction

        # 创建增强信号
        enhanced_signal = TradingSignal(
            symbol=original_signal.symbol,
            direction=direction,
            signal_type=SignalType.BUY if direction == "买入" else SignalType.SELL if direction == "卖出" else SignalType.HOLD,
            confidence=confidence,
            indicators=original_signal.indicators,
            price=original_signal.price,
            timestamp=datetime.now(),
            reasoning=reasoning,
            action_required=True
        )

        return EnhancedSignal(
            original_signal=original_signal,
            enhanced_signal=enhanced_signal,
            confidence_boost=confidence - original_signal.confidence,
            reasoning=reasoning,
            risk_level=risk_level,
            market_context="无市场上下文",
            signal_strength=signal_strength
        )

    def _filter_signals(self, enhanced_signals: List[EnhancedSignal]) -> List[EnhancedSignal]:
        """过滤信号,确保质量"""
        final_signals = []

        # 按品种分组
        symbol_signals = {}
        for enhanced_signal in enhanced_signals:
            symbol = enhanced_signal.original_signal.symbol
            if symbol not in symbol_signals:
                symbol_signals[symbol] = []
            symbol_signals[symbol].append(enhanced_signal)

        # 每个品种只保留最佳信号
        for symbol, signals in symbol_signals.items():
            # 按置信度排序
            signals.sort(key=lambda x: x.enhanced_signal.confidence, reverse=True)

            # 取前 N 个信号
            selected = signals[:self.max_signals_per_symbol]

            # 过滤低置信度信号
            selected = [s for s in selected if s.enhanced_signal.confidence >= self.min_confidence]

            final_signals.extend(selected)

        return final_signals

    def get_enhancement_statistics(self, enhanced_signals: List[EnhancedSignal]) -> Dict[str, Any]:
        """获取增强统计信息"""
        if not enhanced_signals:
            return {}

        total_signals = len(enhanced_signals)
        enhanced_count = sum(1 for s in enhanced_signals if s.confidence_boost > 0)
        weakened_count = sum(1 for s in enhanced_signals if s.confidence_boost < 0)

        avg_original_confidence = sum(s.original_signal.confidence for s in enhanced_signals) / total_signals
        avg_enhanced_confidence = sum(s.enhanced_signal.confidence for s in enhanced_signals) / total_signals
        avg_boost = sum(s.confidence_boost for s in enhanced_signals) / total_signals

        # 信号强度分布
        strength_dist = {"强": 0, "中": 0, "弱": 0}
        for s in enhanced_signals:
            strength_dist[s.signal_strength] += 1

        # 风险等级分布
        risk_dist = {"低": 0, "中": 0, "高": 0}
        for s in enhanced_signals:
            risk_dist[s.risk_level] += 1

        return {
            "total_signals": total_signals,
            "enhanced_count": enhanced_count,
            "weakened_count": weakened_count,
            "avg_original_confidence": avg_original_confidence,
            "avg_enhanced_confidence": avg_enhanced_confidence,
            "avg_confidence_boost": avg_boost,
            "improvement_rate": enhanced_count / total_signals,
            "strength_distribution": strength_dist,
            "risk_distribution": risk_dist
        }