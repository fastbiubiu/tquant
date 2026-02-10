"""
批量处理器单元测试 - 新的 SignalBatchProcessor API
"""

import pytest
from datetime import datetime
from utils.batch_processor import SignalBatchProcessor
from utils.signals import TradingSignal, SignalType, Direction


def create_test_signal(symbol='SHFE.cu2401', price=8000, confidence=0.7, index=0):
    """创建测试信号的辅助函数"""
    return TradingSignal(
        symbol=symbol,
        price=price + index * 10,
        confidence=confidence + index * 0.01,
        signal_type=SignalType.BUY,
        direction=Direction.LONG,
        indicators=[],
        reasoning=f'Test signal {index}',
        timestamp=datetime.now(),
        action_required=True
    )


class TestSignalBatchProcessorInitialization:
    """测试 SignalBatchProcessor 初始化"""

    def test_initialization_with_empty_config(self):
        """测试使用空配置初始化"""
        processor = SignalBatchProcessor(config={})
        assert processor.batch_size == 10  # 默认值
        assert processor.min_confidence == 0.5  # 默认值
        assert processor.max_signals_per_batch == 50  # 默认值
        assert processor.aggregation_method == 'average'  # 默认值

    def test_initialization_with_custom_config(self):
        """测试使用自定义配置初始化"""
        config = {
            'batch_size': 20,
            'min_confidence': 0.6,
            'max_signals_per_batch': 100,
            'aggregation_method': 'weighted'
        }
        processor = SignalBatchProcessor(config=config)
        assert processor.batch_size == 20
        assert processor.min_confidence == 0.6
        assert processor.max_signals_per_batch == 100
        assert processor.aggregation_method == 'weighted'

    def test_initialization_with_partial_config(self):
        """测试使用部分配置初始化"""
        config = {'batch_size': 15}
        processor = SignalBatchProcessor(config=config)
        assert processor.batch_size == 15
        assert processor.min_confidence == 0.5  # 使用默认值


class TestSignalBatching:
    """测试信号批处理"""

    def test_batch_analyze_signals_basic(self):
        """测试基础信号批处理"""
        processor = SignalBatchProcessor(config={'batch_size': 3})

        # 创建测试信号
        signals = [create_test_signal(index=i) for i in range(10)]

        batches = processor.batch_analyze_signals(signals)
        assert len(batches) == 4  # 10 signals / 3 per batch = 4 batches
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_batch_analyze_signals_custom_batch_size(self):
        """测试自定义批次大小"""
        processor = SignalBatchProcessor(config={'batch_size': 5})

        signals = [create_test_signal(index=i) for i in range(12)]

        batches = processor.batch_analyze_signals(signals, batch_size=4)
        assert len(batches) == 3  # 12 signals / 4 per batch = 3 batches

    def test_batch_analyze_signals_empty(self):
        """测试空信号列表"""
        processor = SignalBatchProcessor(config={})
        batches = processor.batch_analyze_signals([])
        assert len(batches) == 0


class TestSignalAggregation:
    """测试信号聚合"""

    def test_aggregate_signals_empty_raises_error(self):
        """测试空信号列表抛出错误"""
        processor = SignalBatchProcessor(config={})

        with pytest.raises(ValueError, match="信号列表不能为空"):
            processor.aggregate_signals([])


class TestBatchProcessorConfiguration:
    """测试批处理器配置"""

    def test_config_dict_initialization(self):
        """测试字典配置初始化"""
        processor = SignalBatchProcessor(config={'batch_size': 15})
        assert processor.batch_size == 15

    def test_cache_manager_integration(self):
        """测试缓存管理器集成"""
        from utils.cache_manager import CacheManager

        cache_manager = CacheManager(config={})
        processor = SignalBatchProcessor(
            config={'batch_size': 10},
            cache_manager=cache_manager
        )

        assert processor.cache_manager is cache_manager


class TestBatchProcessorEdgeCases:
    """测试批处理器边界情况"""

    def test_batch_size_larger_than_signals(self):
        """测试批次大小大于信号数"""
        processor = SignalBatchProcessor(config={'batch_size': 100})

        signals = [create_test_signal(index=i) for i in range(5)]

        batches = processor.batch_analyze_signals(signals)
        assert len(batches) == 1
        assert len(batches[0]) == 5

    def test_batch_size_one(self):
        """测试批次大小为1"""
        processor = SignalBatchProcessor(config={'batch_size': 1})

        signals = [create_test_signal(index=i) for i in range(5)]

        batches = processor.batch_analyze_signals(signals)
        assert len(batches) == 5
        for batch in batches:
            assert len(batch) == 1
