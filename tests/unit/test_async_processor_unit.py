"""
异步处理器单元测试
"""

import asyncio
import pytest
import time
from datetime import datetime

import sys
sys.path.insert(0, '/Users/dunkebiao/Working/Coding/Python/tquant')

from utils.async_processor import AsyncProcessor, AsyncTask


class TestAsyncProcessorInitialization:
    """测试异步处理器初始化"""

    def test_initialization_default(self):
        """测试默认初始化"""
        processor = AsyncProcessor()
        assert processor.max_workers == 5
        assert processor.max_queue_size == 1000
        assert len(processor.tasks) == 0
        assert len(processor.completed_tasks) == 0
        assert len(processor.failed_tasks) == 0
        processor.close()

    def test_initialization_custom(self):
        """测试自定义初始化"""
        processor = AsyncProcessor(max_workers=10, max_queue_size=500)
        assert processor.max_workers == 10
        assert processor.max_queue_size == 500
        processor.close()

    def test_stats_initialization(self):
        """测试统计信息初始化"""
        processor = AsyncProcessor()
        stats = processor.get_stats()
        assert stats['total_tasks'] == 0
        assert stats['completed_tasks'] == 0
        assert stats['failed_tasks'] == 0
        processor.close()


class TestAsyncSignalProcessing:
    """测试异步信号处理"""

    @pytest.mark.asyncio
    async def test_process_signals_basic(self):
        """测试基础信号处理"""
        processor = AsyncProcessor()
        signals = [
            {'id': f'signal_{i}', 'type': 'BUY', 'confidence': 0.8}
            for i in range(5)
        ]

        results = await processor.process_signals(signals)

        assert len(results) == len(signals)
        assert processor.stats['completed_tasks'] == len(signals)
        processor.close()

    @pytest.mark.asyncio
    async def test_process_signals_concurrent(self):
        """测试并发信号处理"""
        processor = AsyncProcessor(max_workers=5)
        signals = [
            {'id': f'signal_{i}', 'type': 'BUY' if i % 2 == 0 else 'SELL'}
            for i in range(20)
        ]

        start_time = time.time()
        results = await processor.process_signals(signals)
        duration = time.time() - start_time

        assert len(results) == len(signals)
        assert processor.stats['completed_tasks'] == len(signals)
        # 并发处理应该比顺序处理快
        assert duration < 1.0
        processor.close()

    @pytest.mark.asyncio
    async def test_process_signals_empty(self):
        """测试空信号列表"""
        processor = AsyncProcessor()
        results = await processor.process_signals([])

        assert len(results) == 0
        assert processor.stats['completed_tasks'] == 0
        processor.close()

    @pytest.mark.asyncio
    async def test_process_signals_error_handling(self):
        """测试错误处理"""
        processor = AsyncProcessor()
        signals = [
            {'id': 'signal_1', 'type': 'BUY'},
            {'id': 'signal_2', 'type': 'INVALID'},  # 无效信号
            {'id': 'signal_3', 'type': 'SELL'},
        ]

        results = await processor.process_signals(signals)

        assert len(results) == len(signals)
        # 应该有错误处理
        processor.close()


class TestAsyncTradeProcessing:
    """测试异步交易处理"""

    @pytest.mark.asyncio
    async def test_process_trades_basic(self):
        """测试基础交易处理"""
        processor = AsyncProcessor()
        trades = [
            {'id': f'trade_{i}', 'action': 'BUY', 'volume': 10}
            for i in range(5)
        ]

        results = await processor.process_trades(trades)

        assert len(results) == len(trades)
        assert processor.stats['completed_tasks'] == len(trades)
        processor.close()

    @pytest.mark.asyncio
    async def test_process_trades_concurrent(self):
        """测试并发交易处理"""
        processor = AsyncProcessor(max_workers=5)
        trades = [
            {'id': f'trade_{i}', 'action': 'BUY' if i % 2 == 0 else 'SELL', 'volume': 10}
            for i in range(20)
        ]

        start_time = time.time()
        results = await processor.process_trades(trades)
        duration = time.time() - start_time

        assert len(results) == len(trades)
        assert processor.stats['completed_tasks'] == len(trades)
        assert duration < 1.0
        processor.close()

    @pytest.mark.asyncio
    async def test_process_trades_empty(self):
        """测试空交易列表"""
        processor = AsyncProcessor()
        results = await processor.process_trades([])

        assert len(results) == 0
        assert processor.stats['completed_tasks'] == 0
        processor.close()

    @pytest.mark.asyncio
    async def test_process_trades_error_handling(self):
        """测试交易错误处理"""
        processor = AsyncProcessor()
        trades = [
            {'id': 'trade_1', 'action': 'BUY', 'volume': 10},
            {'id': 'trade_2', 'action': 'INVALID', 'volume': 10},
            {'id': 'trade_3', 'action': 'SELL', 'volume': 10},
        ]

        results = await processor.process_trades(trades)

        assert len(results) == len(trades)
        processor.close()


class TestTaskQueueManagement:
    """测试任务队列管理"""

    @pytest.mark.asyncio
    async def test_task_queue_basic(self):
        """测试基础任务队列"""
        processor = AsyncProcessor(max_queue_size=10)

        # 添加任务到队列
        for i in range(5):
            await processor.task_queue.put({'id': f'task_{i}'})

        assert processor.task_queue.qsize() == 5
        processor.close()

    @pytest.mark.asyncio
    async def test_task_queue_full(self):
        """测试队列满的情况"""
        processor = AsyncProcessor(max_queue_size=5)

        # 填满队列
        for i in range(5):
            await processor.task_queue.put({'id': f'task_{i}'})

        assert processor.task_queue.qsize() == 5
        processor.close()


class TestConcurrentLimit:
    """测试并发限制"""

    @pytest.mark.asyncio
    async def test_concurrent_limit_enforcement(self):
        """测试并发限制执行"""
        processor = AsyncProcessor(max_workers=3)

        signals = [
            {'id': f'signal_{i}', 'type': 'BUY'}
            for i in range(10)
        ]

        results = await processor.process_signals(signals)

        assert len(results) == len(signals)
        # 应该使用最多 3 个工作线程
        assert processor.max_workers == 3
        processor.close()

    @pytest.mark.asyncio
    async def test_concurrent_limit_scaling(self):
        """测试并发限制扩展"""
        processor = AsyncProcessor(max_workers=5)

        # 处理大量任务
        signals = [
            {'id': f'signal_{i}', 'type': 'BUY'}
            for i in range(100)
        ]

        results = await processor.process_signals(signals)

        assert len(results) == len(signals)
        processor.close()


class TestTimeoutHandling:
    """测试超时处理"""

    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """测试超时处理"""
        processor = AsyncProcessor()

        signals = [
            {'id': f'signal_{i}', 'type': 'BUY'}
            for i in range(5)
        ]

        # 应该在合理时间内完成
        start_time = time.time()
        results = await processor.process_signals(signals)
        duration = time.time() - start_time

        assert len(results) == len(signals)
        assert duration < 5.0  # 应该在 5 秒内完成
        processor.close()


class TestStatsTracking:
    """测试统计跟踪"""

    @pytest.mark.asyncio
    async def test_stats_tracking_basic(self):
        """测试基础统计跟踪"""
        processor = AsyncProcessor()

        signals = [
            {'id': f'signal_{i}', 'type': 'BUY'}
            for i in range(10)
        ]

        await processor.process_signals(signals)

        stats = processor.get_stats()
        assert stats['completed_tasks'] == len(signals)
        assert stats['total_duration'] > 0
        assert stats['average_duration'] > 0
        processor.close()

    @pytest.mark.asyncio
    async def test_stats_tracking_multiple_calls(self):
        """测试多次调用的统计跟踪"""
        processor = AsyncProcessor()

        # 第一次调用
        signals1 = [{'id': f'signal_{i}', 'type': 'BUY'} for i in range(5)]
        await processor.process_signals(signals1)

        # 第二次调用
        signals2 = [{'id': f'signal_{i}', 'type': 'SELL'} for i in range(5)]
        await processor.process_signals(signals2)

        stats = processor.get_stats()
        assert stats['completed_tasks'] == 10
        assert stats['average_duration'] > 0
        processor.close()

    @pytest.mark.asyncio
    async def test_stats_duration_tracking(self):
        """测试持续时间跟踪"""
        processor = AsyncProcessor()

        signals = [
            {'id': f'signal_{i}', 'type': 'BUY'}
            for i in range(5)
        ]

        await processor.process_signals(signals)

        stats = processor.get_stats()
        assert stats['max_duration'] > 0
        assert stats['min_duration'] > 0
        assert stats['max_duration'] >= stats['min_duration']
        processor.close()


class TestAsyncProcessorIntegration:
    """异步处理器集成测试"""

    @pytest.mark.asyncio
    async def test_mixed_signals_and_trades(self):
        """测试混合信号和交易处理"""
        processor = AsyncProcessor()

        signals = [{'id': f'signal_{i}', 'type': 'BUY'} for i in range(5)]
        trades = [{'id': f'trade_{i}', 'action': 'BUY'} for i in range(5)]

        signal_results = await processor.process_signals(signals)
        trade_results = await processor.process_trades(trades)

        assert len(signal_results) == len(signals)
        assert len(trade_results) == len(trades)
        assert processor.stats['completed_tasks'] == 10
        processor.close()

    @pytest.mark.asyncio
    async def test_processor_lifecycle(self):
        """测试处理器生命周期"""
        processor = AsyncProcessor()

        # 初始化
        assert processor.max_workers == 5

        # 处理
        signals = [{'id': f'signal_{i}', 'type': 'BUY'} for i in range(5)]
        results = await processor.process_signals(signals)
        assert len(results) == len(signals)

        # 关闭
        processor.close()
        # 应该能够安全关闭


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
