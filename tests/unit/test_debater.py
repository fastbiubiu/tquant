"""
Debater Agent 集成测试
测试辩论机制的完整流程
"""

import unittest
from datetime import datetime
from agents.debater import DebaterAgent, DebateState, DebatePhase
from utils.signals import TradingSignal, SignalType, Direction
import json


class TestDebaterAgent(unittest.TestCase):
    """Debater Agent 测试"""

    def setUp(self):
        """测试前准备"""
        self.debater = DebaterAgent()

    def test_create_bull_researcher(self):
        """测试创建多头研究员"""
        bull_researcher = self.debater.create_bull_researcher()

        # 创建测试状态
        test_signal = self._create_test_signal()
        state = DebateState(
            symbol="SHFE.rb2501",
            initial_signal=test_signal,
            bull_arguments=[],
            bear_arguments=[],
            phase=DebatePhase.INITIAL_ANALYSIS
        )

        # 运行多头研究员
        result = bull_researcher(state)

        # 验证结果
        self.assertEqual(result.phase, DebatePhase.BULL_ARGUMENT)
        self.assertEqual(len(result.bull_arguments), 1)
        self.assertIn("summary", result.bull_arguments[0])
        self.assertGreater(result.bull_arguments[0]["confidence"], 0.5)
        self.assertIn("factors", result.bull_arguments[0])
        self.assertGreater(len(result.bull_arguments[0]["factors"]), 0)

        print(f"\n多头研究员结果:\n{json.dumps(result.bull_arguments[0], ensure_ascii=False, indent=2)}")

    def test_create_bear_researcher(self):
        """测试创建空头研究员"""
        bear_researcher = self.debater.create_bear_researcher()

        # 创建测试状态
        test_signal = self._create_test_signal()
        state = DebateState(
            symbol="SHFE.rb2501",
            initial_signal=test_signal,
            bull_arguments=[],
            bear_arguments=[],
            phase=DebatePhase.INITIAL_ANALYSIS
        )

        # 运行空头研究员
        result = bear_researcher(state)

        # 验证结果
        self.assertEqual(result.phase, DebatePhase.BEAR_ARGUMENT)
        self.assertEqual(len(result.bear_arguments), 1)
        self.assertIn("summary", result.bear_arguments[0])
        self.assertGreater(result.bear_arguments[0]["confidence"], 0.5)

        print(f"\n空头研究员结果:\n{json.dumps(result.bear_arguments[0], ensure_ascii=False, indent=2)}")

    def test_create_moderator(self):
        """测试创建 Moderator"""
        moderator = self.debater.create_moderator()

        # 创建测试状态
        test_signal = self._create_test_signal()
        state = DebateState(
            symbol="SHFE.rb2501",
            initial_signal=test_signal,
            bull_arguments=[{"summary": "多头观点", "confidence": 0.8, "factors": ["因素1"]}],
            bear_arguments=[{"summary": "空头观点", "confidence": 0.2, "factors": ["因素2"]}],
            phase=DebatePhase.BEAR_ARGUMENT
        )

        # 运行 Moderator
        result = moderator(state)

        # 验证结果
        self.assertEqual(result.phase, DebatePhase.MODERATION)
        self.assertIsNotNone(result.moderator_decision)
        self.assertIn("decision", result.moderator_decision)
        self.assertIn("confidence", result.moderator_decision)
        self.assertIn("reasoning", result.moderator_decision)

        print(f"\nModerator 决策:\n{json.dumps(result.moderator_decision, ensure_ascii=False, indent=2)}")

    def test_run_full_debate(self):
        """测试运行完整辩论流程"""
        test_signal = self._create_test_signal()

        debate_state = self.debater.run_full_debate(test_signal)

        # 验证辩论状态
        self.assertEqual(debate_state.phase, DebatePhase.MODERATION)
        self.assertGreater(len(debate_state.bull_arguments), 0)
        self.assertGreater(len(debate_state.bear_arguments), 0)
        self.assertIsNotNone(debate_state.moderator_decision)
        self.assertIn("decision", debate_state.moderator_decision)

        print(f"\n完整辩论结果:\n")
        print(f"多头论点数: {len(debate_state.bull_arguments)}")
        print(f"空头论点数: {len(debate_state.bear_arguments)}")
        print(f"最终决策: {debate_state.moderator_decision.get('decision', 'N/A')}")
        print(f"置信度: {debate_state.moderator_decision.get('confidence', 0):.2%}")
        print(f"理由: {debate_state.moderator_decision.get('reasoning', 'N/A')}")
        print(f"风险等级: {debate_state.moderator_decision.get('risk_level', 'N/A')}")

    def test_convert_debate_decision_to_signal(self):
        """测试辩论决策转换为交易信号"""
        test_signal = self._create_test_signal()

        # 创建辩论状态
        debate_state = DebateState(
            symbol=test_signal.symbol,
            initial_signal=test_signal,
            bull_arguments=[],
            bear_arguments=[],
            moderator_decision={
                "decision": "买入",
                "confidence": 0.85,
                "reasoning": "技术指标显示多头优势",
                "action": "买入",
                "risk_level": "中",
                "expected_pnl": "+2.5%",
                "stop_loss": "当前价格 -2%",
                "take_profit": "当前价格 +5%"
            },
            phase=DebatePhase.MODERATION
        )

        # 转换为交易信号
        enhanced_signal = self.debater.convert_debate_decision_to_signal(debate_state)

        # 验证交易信号
        self.assertEqual(enhanced_signal.symbol, test_signal.symbol)
        self.assertEqual(enhanced_signal.direction, Direction.LONG)
        self.assertEqual(enhanced_signal.signal_type, SignalType.BUY)
        self.assertGreater(enhanced_signal.confidence, 0.7)
        self.assertTrue(enhanced_signal.action_required)

        print(f"\n转换后的交易信号:")
        print(f"  Symbol: {enhanced_signal.symbol}")
        print(f"  Direction: {enhanced_signal.direction.value}")
        print(f"  Signal Type: {enhanced_signal.signal_type.value}")
        print(f"  Confidence: {enhanced_signal.confidence:.2f}")
        print(f"  Reasoning: {enhanced_signal.reasoning}")

    def test_debate_orchestrator(self):
        """测试辩论协调器"""
        orchestrator = DebaterAgent()

        # 创建测试信号
        test_signals = [
            self._create_test_signal("SHFE.rb2501"),
            self._create_test_signal("SHFE.ag2501")
        ]

        # 为每个信号运行辩论
        enhanced_signals = []
        for signal in test_signals:
            debate_state = orchestrator.run_full_debate(signal)
            enhanced_signal = orchestrator.convert_debate_decision_to_signal(debate_state)
            enhanced_signals.append(enhanced_signal)

        # 验证结果
        self.assertEqual(len(enhanced_signals), len(test_signals))

        for i, signal in enumerate(enhanced_signals):
            print(f"\n{i+1}. {signal.symbol}: {signal.signal_type.value} (置信度: {signal.confidence:.2f})")
            self.assertIsNotNone(signal)
            self.assertTrue(signal.action_required)

    def test_bull_vs_bear_differentiation(self):
        """测试多头和空头论点的区分"""
        test_signal = self._create_test_signal()

        # 运行辩论
        debate_state = self.debater.run_full_debate(test_signal)

        # 验证多头和空头论点
        bull_summary = self.debater._summarize_arguments(debate_state.bull_arguments)
        bear_summary = self.debater._summarize_arguments(debate_state.bear_arguments)

        print(f"\n多头论点摘要: {bull_summary.get('summary', 'N/A')}")
        print(f"多头论点因素: {bull_summary.get('factors', [])}")
        print(f"多头论点置信度: {bull_summary.get('confidence', 0):.2f}")

        print(f"\n空头论点摘要: {bear_summary.get('summary', 'N/A')}")
        print(f"空头论点因素: {bear_summary.get('factors', [])}")
        print(f"空头论点置信度: {bear_summary.get('confidence', 0):.2f}")

        # 验证论点摘要不相同
        self.assertNotEqual(bull_summary.get("summary", ""), bear_summary.get("summary", ""))
        # 多头和空头应该有不同的方向标识
        self.assertNotEqual(
            "BUY" if "BUY" in bull_summary.get("summary", "") else "LONG" if "多头" in bull_summary.get("summary", "") else "neutral",
            "SELL" if "SELL" in bear_summary.get("summary", "") else "SHORT" if "空头" in bear_summary.get("summary", "") else "neutral"
        )

        # 多头论点应该包含 BUY 或多头相关词汇
        self.assertTrue(
            "BUY" in bull_summary.get("summary", "") or "多头" in bull_summary.get("summary", ""),
            f"多头论点摘要不包含 BUY 或多头: {bull_summary.get('summary', '')}"
        )
        # 空头论点应该包含 SELL 或空头相关词汇
        self.assertTrue(
            "SELL" in bear_summary.get("summary", "") or "空头" in bear_summary.get("summary", ""),
            f"空头论点摘要不包含 SELL 或空头: {bear_summary.get('summary', '')}"
        )

    def _create_test_signal(self, symbol: str = "SHFE.rb2501") -> TradingSignal:
        """创建测试交易信号"""
        return TradingSignal(
            symbol=symbol,
            direction=Direction.LONG,
            signal_type=SignalType.BUY,
            confidence=0.75,
            indicators=[
                {
                    "name": "MACD",
                    "value": 0.5,
                    "signal_type": " bullish crossover"
                },
                {
                    "name": "RSI",
                    "value": 45.0,
                    "signal_type": " neutral"
                },
                {
                    "name": "MA",
                    "value": 3520.0,
                    "signal_type": " above MA"
                }
            ],
            price=3500.0,
            timestamp=datetime.now(),
            reasoning="技术指标显示多头优势",
            action_required=True
        )


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
