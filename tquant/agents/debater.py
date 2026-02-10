"""
辩论 Agent 模块
实现多头/空头辩论机制，通过辩论产生最终决策
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from tquant.config import get_config, Config
from tquant.utils.signals import TradingSignal, SignalType, Direction

logger = logging.getLogger(__name__)


class DebatePhase(Enum):
    """辩论阶段"""
    INITIAL_ANALYSIS = "initial_analysis"
    BULL_ARGUMENT = "bull_argument"
    BEAR_ARGUMENT = "bear_argument"
    MODERATION = "moderation"
    FINAL_DECISION = "final_decision"


@dataclass
class DebateState:
    """辩论状态"""
    symbol: str
    initial_signal: TradingSignal
    bull_arguments: List[Dict[str, Any]]
    bear_arguments: List[Dict[str, Any]]
    moderator_decision: Optional[Dict[str, Any]] = None
    iteration_count: int = 0
    max_iterations: int = 3
    phase: DebatePhase = DebatePhase.INITIAL_ANALYSIS
    timestamp: str = ""
    messages: List[str] = None

    def __post_init__(self):
        if self.messages is None:
            self.messages = []


class DebaterAgent:
    """辩论 Agent - 实现多头/空头辩论机制"""

    def __init__(self, config_path: str = None):
        """初始化辩论 Agent"""
        self.config: Config = get_config()

        # 辩论配置（使用属性访问）
        self.max_iterations = getattr(self.config, 'debate', {}).get('max_iterations', 3) if hasattr(self.config, 'debate') else 3
        self.min_confidence = getattr(self.config, 'debate', {}).get('min_confidence', 0.5) if hasattr(self.config, 'debate') else 0.5
        self.risk_tolerance = getattr(self.config, 'debate', {}).get('risk_tolerance', 'medium') if hasattr(self.config, 'debate') else 'medium'

        logger.info(f"辩论 Agent 初始化完成，最大迭代次数: {self.max_iterations}")

    def create_bull_researcher(self) -> callable:
        """创建多头研究员节点"""
        def bull_researcher_node(state: DebateState) -> DebateState:
            """多头研究员"""
            logger.info("=== 多头研究员分析 ===")

            # 获取技术指标
            indicators = state.initial_signal.indicators

            # 构建多头分析提示词
            prompt = self._build_bull_prompt(indicators, state.initial_signal)

            # 使用 LLM 生成多头论点
            bull_argument = self._generate_argument(prompt, direction="BUY")

            state.bull_arguments.append(bull_argument)
            state.phase = DebatePhase.BULL_ARGUMENT
            state.messages.append(f"多头研究员分析了 {state.symbol} 的看涨因素")

            logger.info(f"多头研究员完成: {bull_argument.get('summary', 'N/A')}")
            return state

        return bull_researcher_node

    def create_bear_researcher(self) -> callable:
        """创建空头研究员节点"""
        def bear_researcher_node(state: DebateState) -> DebateState:
            """空头研究员"""
            logger.info("=== 空头研究员分析 ===")

            # 获取技术指标
            indicators = state.initial_signal.indicators

            # 构建空头分析提示词
            prompt = self._build_bear_prompt(indicators, state.initial_signal)

            # 使用 LLM 生成空头论点
            bear_argument = self._generate_argument(prompt, direction="SELL")

            state.bear_arguments.append(bear_argument)
            state.phase = DebatePhase.BEAR_ARGUMENT
            state.messages.append(f"空头研究员分析了 {state.symbol} 的看跌因素")

            logger.info(f"空头研究员完成: {bear_argument.get('summary', 'N/A')}")
            return state

        return bear_researcher_node

    def create_moderator(self) -> callable:
        """创建 Moderator 节点"""
        def moderator_node(state: DebateState) -> DebateState:
            """Moderator 进行综合决策"""
            logger.info("=== Moderator 综合决策 ===")

            bull_summary = self._summarize_arguments(state.bull_arguments)
            bear_summary = self._summarize_arguments(state.bear_arguments)

            # 构建 Moderator 提示词
            prompt = self._build_moderator_prompt(
                state.initial_signal,
                bull_summary,
                bear_summary
            )

            # 使用 LLM 生成 Moderator 决策
            moderator_decision = self._generate_moderator_decision(prompt)

            state.moderator_decision = moderator_decision
            state.phase = DebatePhase.MODERATION
            state.messages.append(f"Moderator 综合多头和空头观点: {moderator_decision.get('decision', 'N/A')}")

            logger.info(f"Moderator 决策: {moderator_decision.get('decision', 'N/A')}")
            return state

        return moderator_node

    def _build_bull_prompt(self, indicators: List[Dict], signal: TradingSignal) -> str:
        """构建多头分析提示词"""
        indicators_summary = self._summarize_indicators(indicators)

        prompt = f"""
你是一个专业的多头研究员，专注于金融市场分析。

当前分析品种：{signal.symbol}
当前价格：{signal.price}
初始信号方向：{signal.direction}
初始信号置信度：{signal.confidence:.2%}
技术指标分析：
{indicators_summary}

请分析该品种的看涨因素：
1. 技术指标是否支持看涨（如 MA 指金叉、MACD 指金叉、RSI 处于超卖反弹等）
2. 市场趋势是否向上
3. 成交量是否放大
4. 支撑位情况
5. 其他看涨因素

请提供详细的多头论点，包括：
- 看涨因素分析
- 潜在风险
- 建议操作

输出格式（JSON）：
{{
    "summary": "多头论点摘要(50字以内)",
    "factors": ["看涨因素1", "看涨因素2", ...],
    "confidence": 0.85,
    "risk_level": "低/中/高",
    "suggestion": "具体建议(如买入价格、止损、目标价位)"
}}
"""
        return prompt

    def _build_bear_prompt(self, indicators: List[Dict], signal: TradingSignal) -> str:
        """构建空头分析提示词"""
        indicators_summary = self._summarize_indicators(indicators)

        prompt = f"""
你是一个专业的空头研究员，专注于金融市场分析。

当前分析品种：{signal.symbol}
当前价格：{signal.price}
初始信号方向：{signal.direction}
初始信号置信度：{signal.confidence:.2%}
技术指标分析：
{indicators_summary}

请分析该品种的看跌因素：
1. 技术指标是否支持看跌（如 MA 指死叉、MACD 指死叉、RSI 处于超买回落等）
2. 市场趋势是否向下
3. 成交量是否放大
4. 阻力位情况
5. 其他看跌因素

请提供详细的空头论点，包括：
- 看跌因素分析
- 潜在风险
- 建议操作

输出格式（JSON）：
{{
    "summary": "空头论点摘要(50字以内)",
    "factors": ["看跌因素1", "看跌因素2", ...],
    "confidence": 0.85,
    "risk_level": "低/中/高",
    "suggestion": "具体建议(如卖出价格、止损、目标价位)"
}}
"""
        return prompt

    def _build_moderator_prompt(
        self,
        initial_signal: TradingSignal,
        bull_summary: Dict,
        bear_summary: Dict
    ) -> str:
        """构建 Moderator 提示词"""
        prompt = f"""
你是一个专业的辩论 Moderator，负责综合多头和空头观点，做出最终决策。

当前分析品种：{initial_signal.symbol}
当前价格：{initial_signal.price}
初始信号方向：{initial_signal.direction}
初始信号置信度：{initial_signal.confidence:.2%}

多头研究员观点：
{json.dumps(bull_summary, ensure_ascii=False, indent=2)}

空头研究员观点：
{json.dumps(bear_summary, ensure_ascii=False, indent=2)}

请综合以上观点，做出最终决策。考虑因素：
1. 两个观点的权重和可信度
2. 市场趋势和趋势转折
3. 风险控制
4. 交易建议

输出格式（JSON）：
{{
    "decision": "买入/卖出/持有",
    "confidence": 0.85,
    "confidence_based_on": "多头观点(0.85) vs 空头观点(0.15)",
    "reasoning": "决策理由(至少50字)",
    "action": "具体操作建议",
    "risk_level": "低/中/高",
    "expected_pnl": "预期盈亏(百分比)",
    "stop_loss": "止损价格",
    "take_profit": "止盈价格"
}}
"""
        return prompt

    def _generate_argument(self, prompt: str, direction: str) -> Dict[str, Any]:
        """生成论点（简化版，实际应该调用 LLM）"""
        # 这是一个简化实现，实际应该调用 LLM
        if direction == "BUY":
            return {
                "summary": f"{direction} 分析 - 支持买入的多个技术指标",
                "factors": [
                    "MACD 指金叉",
                    "MA 指向上",
                    "RSI 处于超卖反弹区域",
                    "成交量放大",
                    "支撑位在上方"
                ],
                "confidence": 0.75,
                "risk_level": "中",
                "suggestion": f"在当前价格 {0.98 * self._get_current_price()} 附近买入，止损在 {0.96 * self._get_current_price()}，目标价位在 {1.05 * self._get_current_price()}"
            }
        else:
            return {
                "summary": f"{direction} 分析 - 支持卖出的多个技术指标",
                "factors": [
                    "MACD 指死叉",
                    "MA 指向下",
                    "RSI 处于超买回落区域",
                    "成交量放大",
                    "阻力位在下方"
                ],
                "confidence": 0.75,
                "risk_level": "中",
                "suggestion": f"在当前价格 {1.02 * self._get_current_price()} 附近卖出，止损在 {1.05 * self._get_current_price()}，目标价位在 {0.95 * self._get_current_price()}"
            }

    def _get_current_price(self) -> float:
        """获取当前价格（简化实现）"""
        # 实际应该从市场数据获取
        return 3500.0

    def _summarize_indicators(self, indicators: List[Dict]) -> str:
        """汇总技术指标"""
        if not indicators:
            return "无技术指标数据"

        summary = []
        for indicator in indicators[:10]:
            name = indicator.get('name', 'Unknown')
            value = indicator.get('value', 0)
            signal_type = indicator.get('signal_type', 'Unknown')

            summary.append(f"- {name}: {value:.2f} ({signal_type})")

        return "\n".join(summary)

    def _summarize_arguments(self, arguments: List[Dict]) -> Dict[str, Any]:
        """汇总论点"""
        if not arguments:
            return {}

        summary = {
            "summary": "",
            "factors": [],
            "confidence": 0.0,
            "risk_level": ""
        }

        for arg in arguments:
            if arg.get("summary"):
                summary["summary"] = arg.get("summary")
            if arg.get("factors"):
                summary["factors"].extend(arg.get("factors", []))
            summary["confidence"] = max(summary["confidence"], arg.get("confidence", 0.0))
            summary["risk_level"] = arg.get("risk_level", "")

        return summary

    def _generate_moderator_decision(self, prompt: str) -> Dict[str, Any]:
        """生成 Moderator 决策（简化版，实际应该调用 LLM）"""
        # 这是一个简化实现，实际应该调用 LLM
        return {
            "decision": "买入",
            "confidence": 0.85,
            "confidence_based_on": "多头观点占主导",
            "reasoning": "技术指标显示多头优势，MACD 金叉，RSI 处于超卖反弹区域，成交量放大",
            "action": "在当前价格附近买入",
            "risk_level": "中",
            "expected_pnl": "+2.5%",
            "stop_loss": "当前价格 -2%",
            "take_profit": "当前价格 +5%"
        }

    def run_full_debate(
        self,
        initial_signal: TradingSignal
    ) -> DebateState:
        """运行完整的辩论流程"""
        logger.info(f"开始运行辩论: {initial_signal.symbol}")

        # 初始化状态
        state = DebateState(
            symbol=initial_signal.symbol,
            initial_signal=initial_signal,
            bull_arguments=[],
            bear_arguments=[],
            iteration_count=0,
            phase=DebatePhase.INITIAL_ANALYSIS,
            timestamp=datetime.now().isoformat()
        )

        # 创建辩论节点
        bull_researcher = self.create_bull_researcher()
        bear_researcher = self.create_bear_researcher()
        moderator = self.create_moderator()

        # 运行辩论流程
        state = bull_researcher(state)
        state = bear_researcher(state)
        state = moderator(state)

        logger.info(f"辩论完成: 最终决策 = {state.moderator_decision.get('decision', 'N/A')}")
        return state

    def convert_debate_decision_to_signal(
        self,
        debate_state: DebateState
    ) -> TradingSignal:
        """将辩论决策转换为交易信号"""
        if not debate_state.moderator_decision:
            return debate_state.initial_signal

        decision = debate_state.moderator_decision.get('decision', 'HOLD')
        confidence = debate_state.moderator_decision.get('confidence', 0.5)

        # 映射决策到信号类型，保持与初始信号的枚举类型一致
        SignalTypeEnum = debate_state.initial_signal.signal_type.__class__
        mapped_signal_type = self._map_decision_to_signal_type(decision)
        # 使用相同名称从初始信号的枚举类型中取值，避免不同导入路径导致的不等问题
        try:
            signal_type = SignalTypeEnum[mapped_signal_type.name]
        except Exception:
            signal_type = mapped_signal_type

        # 使用初始信号的 Direction 枚举类型，避免因不同导入路径导致的枚举实例不等问题
        DirectionEnum = debate_state.initial_signal.direction.__class__
        if decision == "买入":
            direction = DirectionEnum.LONG
        elif decision == "卖出":
            direction = DirectionEnum.SHORT
        else:
            direction = DirectionEnum.NEUTRAL

        return TradingSignal(
            symbol=debate_state.symbol,
            direction=direction,
            signal_type=signal_type,
            confidence=confidence,
            indicators=debate_state.initial_signal.indicators,
            price=debate_state.initial_signal.price,
            timestamp=datetime.now(),
            reasoning=debate_state.moderator_decision.get('reasoning', ''),
            action_required=True
        )

    def _map_decision_to_signal_type(self, decision: str) -> SignalType:
        """映射决策到信号类型"""
        decision = decision.upper()
        if decision == "买入":
            return SignalType.BUY
        elif decision == "卖出":
            return SignalType.SELL
        else:
            return SignalType.HOLD


class DebateOrchestrator:
    """辩论协调器 - 管理多轮辩论"""

    def __init__(self, config_path: str = None):
        """初始化辩论协调器"""
        self.debater = DebaterAgent(config_path)

        # 获取辩论配置
        self.max_iterations = self.config.get('debate', {}).get('max_iterations', 3)

        logger.info(f"辩论协调器初始化完成")

    def orchestrate_debate(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """协调多个信号的辩论"""
        enhanced_signals = []

        for signal in signals:
            logger.info(f"开始为 {signal.symbol} 进行辩论...")

            # 运行辩论
            debate_state = self.debater.run_full_debate(signal)

            # 转换为交易信号
            enhanced_signal = self.debater.convert_debate_decision_to_signal(debate_state)

            enhanced_signals.append(enhanced_signal)

            logger.info(f"{signal.symbol} 辩论完成，最终决策: {enhanced_signal.signal_type.value}")

        return enhanced_signals
