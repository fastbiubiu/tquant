"""
错误处理单元测试
"""

import unittest
import os
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
import json

from utils.error_handler import (
    TradingError, APIConnectionError, MarketDataError, SignalGenerationError,
    TradeExecutionError, WorkflowError, ErrorHandler, ErrorRecovery
)

class TestErrorClasses(unittest.TestCase):
    """错误类测试"""

    def test_trading_error(self):
        """测试基础交易错误"""
        error = TradingError("测试交易错误", "TRADING_001")
        self.assertEqual(error.message, "测试交易错误")
        self.assertEqual(error.error_code, "TRADING_001")
        self.assertEqual(error.details, {})
        self.assertIsInstance(error.timestamp, datetime)

    def test_api_connection_error(self):
        """测试API连接错误"""
        error = APIConnectionError("API连接失败", "tqsdk")
        self.assertEqual(error.message, "API连接失败")
        self.assertEqual(error.error_code, "API_CONNECTION_ERROR")
        self.assertEqual(error.details.get("api_type"), "tqsdk")

    def test_market_data_error(self):
        """测试市场数据错误"""
        error = MarketDataError("获取市场数据失败", "SHFE.cu2401")
        self.assertEqual(error.message, "获取市场数据失败")
        self.assertEqual(error.error_code, "MARKET_DATA_ERROR")
        self.assertEqual(error.details.get("symbol"), "SHFE.cu2401")

    def test_signal_generation_error(self):
        """测试信号生成错误"""
        error = SignalGenerationError("生成信号失败", "DCE.i2403")
        self.assertEqual(error.message, "生成信号失败")
        self.assertEqual(error.error_code, "SIGNAL_GENERATION_ERROR")
        self.assertEqual(error.details.get("symbol"), "DCE.i2403")

    def test_trade_execution_error(self):
        """测试交易执行错误"""
        error = TradeExecutionError("订单执行失败", "CZCE.MA401", "order_123")
        self.assertEqual(error.message, "订单执行失败")
        self.assertEqual(error.error_code, "TRADE_EXECUTION_ERROR")
        self.assertEqual(error.details.get("symbol"), "CZCE.MA401")
        self.assertEqual(error.details.get("order_id"), "order_123")

    def test_workflow_error(self):
        """测试工作流错误"""
        error = WorkflowError("工作流失败", "analyze_market")
        self.assertEqual(error.message, "工作流失败")
        self.assertEqual(error.error_code, "WORKFLOW_ERROR")
        self.assertEqual(error.details.get("workflow_node"), "analyze_market")

class TestErrorHandler(unittest.TestCase):
    """错误处理器测试"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.log_dir = os.path.join(self.temp_dir, 'logs')
        self.error_handler = ErrorHandler(self.log_dir)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.temp_dir)

    def test_error_handler_initialization(self):
        """测试错误处理器初始化"""
        self.assertIsNotNone(self.error_handler.logger)
        self.assertEqual(self.error_handler.log_dir, Path(self.log_dir))
        self.assertEqual(self.error_handler.error_counts, {})

    def test_log_error(self):
        """测试错误记录"""
        error = TradingError("测试错误", "TEST_001")
        self.error_handler.log_error(error, {"test": "context"})

        # 检查错误计数
        self.assertEqual(self.error_handler.error_counts["TradingError_TEST_001"], 1)

        # 检查错误历史
        self.assertEqual(len(self.error_handler.error_history), 1)
        self.assertEqual(self.error_handler.error_history[0]['error_code'], "TEST_001")

    def test_save_error_report(self):
        """测试错误报告保存"""
        error = TradingError("测试错误", "TEST_001")
        report_path = os.path.join(self.temp_dir, 'error_report.json')
        self.error_handler.save_error_report(error, report_path)

        # 检查文件是否存在
        self.assertTrue(os.path.exists(report_path))

        # 读取并验证报告内容
        with open(report_path, 'r') as f:
            report = json.load(f)

        self.assertEqual(report['error_code'], "TEST_001")
        self.assertIn('timestamp', report)
        self.assertIn('error_counts', report)

    def test_get_error_statistics(self):
        """测试错误统计"""
        # 记录一些错误
        error1 = TradingError("错误1", "ERROR_001")
        error2 = APIConnectionError("错误2", "tqsdk")
        error3 = TradingError("错误3", "ERROR_001")

        self.error_handler.log_error(error1)
        self.error_handler.log_error(error2)
        self.error_handler.log_error(error3)

        # 获取统计
        stats = self.error_handler.get_error_statistics()

        # 验证统计
        self.assertEqual(stats['total_errors'], 3)
        self.assertEqual(stats['unique_error_types'], 2)
        self.assertEqual(stats['error_by_type']['TradingError'], 2)
        self.assertEqual(stats['error_by_type']['APIConnectionError'], 1)

class TestErrorRecovery(unittest.TestCase):
    """错误恢复策略测试"""

    def test_recover_from_api_error(self):
        """测试API错误恢复"""
        error = APIConnectionError("API连接失败", "tqsdk")

        # 测试有重试次数的情况
        result = ErrorRecovery.recover_from_api_error(error, retry_count=3, delay=1.0)
        self.assertEqual(result['success'], False)
        self.assertEqual(result['action_taken'], 'retry')
        self.assertEqual(result['remaining_retries'], 2)
        self.assertIn('将在 1.0 秒后重试', result['message'])

        # 测试无重试次数的情况
        result = ErrorRecovery.recover_from_api_error(error, retry_count=0, delay=1.0)
        self.assertEqual(result['success'], False)
        self.assertEqual(result['action_taken'], 'stop')
        self.assertIn('已达到最大重试次数', result['message'])

    def test_recover_from_market_data_error(self):
        """测试市场数据错误恢复"""
        error = MarketDataError("获取市场数据失败", "SHFE.cu2401")

        result = ErrorRecovery.recover_from_market_data_error(error)
        self.assertEqual(result['success'], False)
        self.assertEqual(result['action_taken'], 'skip_symbol')
        self.assertIn('跳过品种 SHFE.cu2401', result['message'])

    def test_recover_from_signal_error(self):
        """测试信号错误恢复"""
        error = SignalGenerationError("生成信号失败", "DCE.i2403")

        result = ErrorRecovery.recover_from_signal_error(error)
        self.assertEqual(result['success'], False)
        self.assertEqual(result['action_taken'], 'skip_symbol')
        self.assertIn('跳过 DCE.i2403 的信号生成', result['message'])

    def test_recover_from_trade_error(self):
        """测试交易错误恢复"""
        error = TradeExecutionError("交易执行失败", "CZCE.MA401", "order_123")

        result = ErrorRecovery.recover_from_trade_error(error)
        self.assertEqual(result['success'], False)
        self.assertEqual(result['action_taken'], 'cancel_order')
        self.assertIn('取消该订单', result['message'])

    def test_recover_from_trade_error_without_order_id(self):
        """测试没有订单ID的交易错误恢复"""
        error = TradeExecutionError("交易执行失败", "CZCE.MA401")

        result = ErrorRecovery.recover_from_trade_error(error)
        self.assertEqual(result['success'], False)
        self.assertEqual(result['action_taken'], 'reduce_position')
        self.assertIn('降低仓位', result['message'])

if __name__ == '__main__':
    unittest.main()