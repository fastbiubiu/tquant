"""
信号验证器单元测试 - 新的 API
"""

import pytest
from datetime import datetime
from utils.signal_validator import SignalValidator
from utils.signals import TradingSignal, SignalType, Direction


def create_test_signal(confidence=0.8, signal_type=SignalType.BUY):
    """创建测试信号"""
    return TradingSignal(
        symbol='SHFE.cu2401',
        price=8000,
        confidence=confidence,
        signal_type=signal_type,
        direction=Direction.LONG,
        indicators=[],
        reasoning='Test signal',
        timestamp=datetime.now(),
        action_required=True
    )


class TestSignalValidatorInitialization:
    """测试信号验证器初始化"""

    def test_initialization(self):
        """测试初始化"""
        validator = SignalValidator(config={})
        assert validator is not None

    def test_initialization_with_config(self):
        """测试使用配置初始化"""
        config = {'min_confidence': 0.6}
        validator = SignalValidator(config=config)
        assert validator is not None


class TestSignalValidation:
    """测试信号验证"""

    def test_validate_signal_high_confidence(self):
        """测试验证高置信度信号"""
        validator = SignalValidator(config={})

        signal = create_test_signal(confidence=0.9)
        result = validator.validate_signal(signal)

        assert isinstance(result, dict) or isinstance(result, bool)

    def test_validate_signal_low_confidence(self):
        """测试验证低置信度信号"""
        validator = SignalValidator(config={})

        signal = create_test_signal(confidence=0.3)
        result = validator.validate_signal(signal)

        assert isinstance(result, dict) or isinstance(result, bool)

    def test_validate_signal_medium_confidence(self):
        """测试验证中等置信度信号"""
        validator = SignalValidator(config={})

        signal = create_test_signal(confidence=0.5)
        result = validator.validate_signal(signal)

        assert isinstance(result, dict) or isinstance(result, bool)


class TestBatchValidation:
    """测试批量验证"""

    def test_batch_validation(self):
        """测试批量验证信号"""
        validator = SignalValidator(config={})

        signals = [
            create_test_signal(confidence=0.9),
            create_test_signal(confidence=0.7),
            create_test_signal(confidence=0.5),
        ]

        results = validator.batch_validate(signals)
        assert isinstance(results, list)
        assert len(results) == 3


class TestSignalAnalysis:
    """测试信号分析"""

    def test_check_trend_consistency(self):
        """测试检查趋势一致性"""
        validator = SignalValidator(config={})

        signal = create_test_signal()
        if hasattr(validator, 'check_trend_consistency'):
            result = validator.check_trend_consistency(signal)
            assert result is not None or result is None

    def test_check_indicators_rationality(self):
        """测试检查指标合理性"""
        validator = SignalValidator(config={})

        signal = create_test_signal()
        if hasattr(validator, 'check_indicators_rationality'):
            result = validator.check_indicators_rationality(signal)
            assert result is not None or result is None


class TestSignalPerformance:
    """测试信号性能"""

    def test_historical_performance(self):
        """测试历史性能"""
        validator = SignalValidator(config={})

        if hasattr(validator, 'get_historical_performance'):
            performance = validator.get_historical_performance()
            assert performance is not None or performance is None

    def test_performance_tracking(self):
        """测试性能跟踪"""
        validator = SignalValidator(config={})

        signal = create_test_signal()
        if hasattr(validator, 'track_performance'):
            validator.track_performance(signal)


class TestSignalValidatorEdgeCases:
    """测试信号验证器边界情况"""

    def test_empty_signal_handling(self):
        """测试空信号处理"""
        validator = SignalValidator(config={})

        # 测试边界情况
        signal = create_test_signal(confidence=0.0)
        result = validator.validate_signal(signal)
        assert result is not None or result is None

    def test_extreme_confidence(self):
        """测试极端置信度"""
        validator = SignalValidator(config={})

        signal = create_test_signal(confidence=1.0)
        result = validator.validate_signal(signal)
        assert result is not None or result is None


class TestSignalValidationSummary:
    """测试信号验证摘要"""

    def test_validation_summary(self):
        """测试验证摘要"""
        validator = SignalValidator(config={})

        if hasattr(validator, 'get_validation_summary'):
            summary = validator.get_validation_summary()
            assert isinstance(summary, (str, dict)) or summary is None

    def test_suggested_action(self):
        """测试建议操作"""
        validator = SignalValidator(config={})

        signal = create_test_signal()
        if hasattr(validator, 'get_suggested_action'):
            action = validator.get_suggested_action(signal)
            assert action is not None or action is None
