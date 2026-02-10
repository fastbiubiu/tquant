"""
异步处理器模块
用于处理异步 Agent 执行、并发信号处理和异步 I/O 操作
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class AsyncTask:
    """异步任务"""
    task_id: str
    task_type: str
    data: Any
    created_at: datetime
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[Any] = None
    error: Optional[str] = None
    duration: float = 0.0


class AsyncProcessor:
    """
    异步处理器

    功能:
    - 异步 Agent 执行
    - 并发信号处理
    - 异步 I/O 操作
    - 任务队列管理
    - 并发控制
    """

    def __init__(self, max_workers: int = 5, max_queue_size: int = 1000):
        """
        初始化异步处理器

        Args:
            max_workers: 最大工作线程数
            max_queue_size: 最大队列大小
        """
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # 任务队列和管理
        self.task_queue = asyncio.Queue(maxsize=max_queue_size)
        self.tasks: Dict[str, AsyncTask] = {}
        self.completed_tasks: List[AsyncTask] = []
        self.failed_tasks: List[AsyncTask] = []

        # 统计信息
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'total_duration': 0.0,
            'average_duration': 0.0,
            'max_duration': 0.0,
            'min_duration': float('inf')
        }

        logger.info(f"异步处理器初始化完成: max_workers={max_workers}")

    async def process_signals(self, signals: List[Dict]) -> List[Dict]:
        """
        异步处理信号

        Args:
            signals: 信号列表

        Returns:
            处理结果列表
        """
        if not signals:
            return []

        logger.info(f"开始异步处理 {len(signals)} 个信号")

        start_time = time.time()
        tasks = []
        for signal in signals:
            task = asyncio.create_task(self._process_signal(signal))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"信号处理异常: {result}")
                processed_results.append({'error': str(result)})
            else:
                processed_results.append(result)

        # 更新统计
        duration = time.time() - start_time
        self.stats['completed_tasks'] += len(signals)
        self.stats['total_duration'] += duration
        self.stats['average_duration'] = (
            self.stats['total_duration'] / max(self.stats['completed_tasks'], 1)
        )
        self.stats['max_duration'] = max(self.stats['max_duration'], duration)
        if self.stats['min_duration'] == float('inf'):
            self.stats['min_duration'] = duration
        else:
            self.stats['min_duration'] = min(self.stats['min_duration'], duration)

        logger.info(f"完成异步处理 {len(signals)} 个信号, 耗时 {duration:.2f}s")
        return processed_results

    async def process_trades(self, trades: List[Dict]) -> List[Dict]:
        """
        异步处理交易

        Args:
            trades: 交易列表

        Returns:
            处理结果列表
        """
        if not trades:
            return []

        logger.info(f"开始异步处理 {len(trades)} 笔交易")

        start_time = time.time()
        tasks = []
        for trade in trades:
            task = asyncio.create_task(self._process_trade(trade))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"交易处理异常: {result}")
                processed_results.append({'error': str(result)})
            else:
                processed_results.append(result)

        # 更新统计
        duration = time.time() - start_time
        self.stats['completed_tasks'] += len(trades)
        self.stats['total_duration'] += duration
        self.stats['average_duration'] = (
            self.stats['total_duration'] / max(self.stats['completed_tasks'], 1)
        )
        self.stats['max_duration'] = max(self.stats['max_duration'], duration)
        if self.stats['min_duration'] == float('inf'):
            self.stats['min_duration'] = duration
        else:
            self.stats['min_duration'] = min(self.stats['min_duration'], duration)

        logger.info(f"完成异步处理 {len(trades)} 笔交易, 耗时 {duration:.2f}s")
        return processed_results

    async def process_batch(self, batch_data: Dict, processor_func: Callable) -> Dict:
        """
        异步处理批次

        Args:
            batch_data: 批次数据
            processor_func: 处理函数

        Returns:
            处理结果
        """
        task_id = f"batch_{int(time.time() * 1000)}"

        try:
            # 创建任务
            task = AsyncTask(
                task_id=task_id,
                task_type='batch',
                data=batch_data,
                created_at=datetime.now(),
                status='running'
            )
            self.tasks[task_id] = task
            self.stats['total_tasks'] += 1

            # 执行处理
            start_time = time.time()
            result = await asyncio.to_thread(processor_func, batch_data)
            duration = time.time() - start_time

            # 更新任务状态
            task.status = 'completed'
            task.result = result
            task.duration = duration

            # 更新统计
            self._update_stats(task)
            self.completed_tasks.append(task)

            logger.info(f"批次处理完成: {task_id}, 耗时 {duration:.2f}s")
            return result

        except Exception as e:
            logger.error(f"批次处理失败: {task_id}, 错误: {e}")
            task = self.tasks[task_id]
            task.status = 'failed'
            task.error = str(e)
            self.failed_tasks.append(task)
            raise

    async def submit_task(self, task_type: str, data: Any,
                         processor_func: Callable) -> str:
        """
        提交异步任务

        Args:
            task_type: 任务类型
            data: 任务数据
            processor_func: 处理函数

        Returns:
            任务 ID
        """
        task_id = f"{task_type}_{int(time.time() * 1000)}"

        task = AsyncTask(
            task_id=task_id,
            task_type=task_type,
            data=data,
            created_at=datetime.now()
        )

        self.tasks[task_id] = task
        self.stats['total_tasks'] += 1

        # 提交到队列
        try:
            await self.task_queue.put((task_id, processor_func, data))
            logger.info(f"任务已提交: {task_id}")
        except asyncio.QueueFull:
            logger.error(f"任务队列已满: {task_id}")
            task.status = 'failed'
            task.error = 'Queue full'
            self.failed_tasks.append(task)
            raise

        return task_id

    async def get_task_result(self, task_id: str, timeout: float = 30.0) -> Optional[Any]:
        """
        获取任务结果

        Args:
            task_id: 任务 ID
            timeout: 超时时间(秒)

        Returns:
            任务结果
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if task_id not in self.tasks:
                logger.warning(f"任务不存在: {task_id}")
                return None

            task = self.tasks[task_id]

            if task.status == 'completed':
                return task.result
            elif task.status == 'failed':
                raise Exception(f"任务失败: {task.error}")

            # 等待一段时间后重试
            await asyncio.sleep(0.1)

        raise TimeoutError(f"任务超时: {task_id}")

    async def process_concurrent(self, items: List[Any],
                                processor_func: Callable,
                                max_concurrent: int = 5) -> List[Any]:
        """
        并发处理项目

        Args:
            items: 项目列表
            processor_func: 处理函数
            max_concurrent: 最大并发数

        Returns:
            处理结果列表
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_process(item):
            async with semaphore:
                return await asyncio.to_thread(processor_func, item)

        tasks = [bounded_process(item) for item in items]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        return results

    def _update_stats(self, task: AsyncTask):
        """更新统计信息"""
        if task.status == 'completed':
            self.stats['completed_tasks'] += 1
            self.stats['total_duration'] += task.duration
            self.stats['average_duration'] = (
                self.stats['total_duration'] / self.stats['completed_tasks']
            )
            self.stats['max_duration'] = max(
                self.stats['max_duration'], task.duration
            )
            self.stats['min_duration'] = min(
                self.stats['min_duration'], task.duration
            )
        elif task.status == 'failed':
            self.stats['failed_tasks'] += 1

    async def _process_signal(self, signal: Dict) -> Dict:
        """处理单个信号"""
        try:
            # 模拟信号处理
            await asyncio.sleep(0.01)
            return {
                'signal_id': signal.get('id'),
                'status': 'processed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"信号处理失败: {e}")
            raise

    async def _process_trade(self, trade: Dict) -> Dict:
        """处理单个交易"""
        try:
            # 模拟交易处理
            await asyncio.sleep(0.01)
            return {
                'trade_id': trade.get('id'),
                'status': 'processed',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"交易处理失败: {e}")
            raise

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'pending_tasks': len([t for t in self.tasks.values() if t.status == 'pending']),
            'running_tasks': len([t for t in self.tasks.values() if t.status == 'running']),
            'queue_size': self.task_queue.qsize()
        }

    def get_summary(self) -> str:
        """获取摘要"""
        stats = self.get_stats()
        return f"""
=== 异步处理器摘要 ===
总任务数: {stats['total_tasks']}
已完成: {stats['completed_tasks']}
已失败: {stats['failed_tasks']}
待处理: {stats['pending_tasks']}
运行中: {stats['running_tasks']}
队列大小: {stats['queue_size']}

性能指标:
  平均耗时: {stats['average_duration']:.3f}s
  最大耗时: {stats['max_duration']:.3f}s
  最小耗时: {stats['min_duration']:.3f}s
  总耗时: {stats['total_duration']:.2f}s
"""

    def close(self):
        """关闭处理器"""
        self.executor.shutdown(wait=True)
        logger.info("异步处理器已关闭")


# 全局异步处理器实例
_async_processor: Optional[AsyncProcessor] = None


def get_async_processor(max_workers: int = 5) -> AsyncProcessor:
    """获取全局异步处理器实例"""
    global _async_processor
    if _async_processor is None:
        _async_processor = AsyncProcessor(max_workers=max_workers)
    return _async_processor


async def process_signals_async(signals: List[Dict]) -> List[Dict]:
    """异步处理信号(便捷函数)"""
    processor = get_async_processor()
    return await processor.process_signals(signals)


async def process_trades_async(trades: List[Dict]) -> List[Dict]:
    """异步处理交易(便捷函数)"""
    processor = get_async_processor()
    return await processor.process_trades(trades)
