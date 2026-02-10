"""
错误处理模块
提供统一的错误处理和日志记录功能
"""

import json
import logging
import sys
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Any, Optional, Callable


# 定义错误类型
class TradingError(Exception):
    """交易系统基础错误"""
    def __init__(self, message: str, error_code: str = "TRADING_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = datetime.now()

class MarketDataError(TradingError):
    """市场数据错误"""
    def __init__(self, message: str, symbol: str = None, details: Optional[Dict] = None):
        super().__init__(message, "MARKET_DATA_ERROR", {"symbol": symbol, **(details or {})})

class APIConnectionError(TradingError):
    """API连接错误"""
    def __init__(self, message: str, api_type: str = None, details: Optional[Dict] = None):
        super().__init__(message, "API_CONNECTION_ERROR", {"api_type": api_type, **(details or {})})

class SignalGenerationError(TradingError):
    """信号生成错误"""
    def __init__(self, message: str, symbol: str = None, details: Optional[Dict] = None):
        super().__init__(message, "SIGNAL_GENERATION_ERROR", {"symbol": symbol, **(details or {})})

class TradeExecutionError(TradingError):
    """交易执行错误"""
    def __init__(self, message: str, symbol: str = None, order_id: str = None, details: Optional[Dict] = None):
        super().__init__(message, "TRADE_EXECUTION_ERROR", {
            "symbol": symbol,
            "order_id": order_id,
            **(details or {})
        })

class ConfigurationError(TradingError):
    """配置错误"""
    def __init__(self, message: str, config_key: str = None, details: Optional[Dict] = None):
        super().__init__(message, "CONFIGURATION_ERROR", {"config_key": config_key, **(details or {})})

class BacktestError(TradingError):
    """回测错误"""
    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(message, "BACKTEST_ERROR", details or {})

class WorkflowError(TradingError):
    """工作流错误"""
    def __init__(self, message: str, workflow_node: str = None, details: Optional[Dict] = None):
        super().__init__(message, "WORKFLOW_ERROR", {"workflow_node": workflow_node, **(details or {})})

class ErrorHandler:
    """错误处理器"""

    def __init__(self, log_dir: str = "logs"):
        """
        初始化错误处理器

        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        # 设置日志
        self._setup_logging()

        # 错误计数器
        self.error_counts = {}
        self.error_history = []

    def _setup_logging(self):
        """设置日志"""
        # 创建日志文件名
        log_file = self.log_dir / f"error_handler_{datetime.now().strftime('%Y%m%d')}.log"

        # 配置日志格式
        logging.basicConfig(
            level=logging.ERROR,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )

        self.logger = logging.getLogger("ErrorHandler")

    def log_error(self, error: Exception, context: Optional[Dict] = None):
        """
        记录错误

        Args:
            error: 异常对象
            context: 上下文信息
        """
        error_type = type(error).__name__
        error_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # 增加错误计数
        error_key = f"{error_type}_{error.__class__.__name__}"
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

        # 添加到错误历史
        error_record = {
            'timestamp': error_time,
            'error_type': error_type,
            'error_code': getattr(error, 'error_code', 'UNKNOWN'),
            'message': str(error),
            'traceback': traceback.format_exc(),
            'context': context or {},
            'count': self.error_counts[error_key]
        }

        self.error_history.append(error_record)

        # 只保留最近1000条错误记录
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

        # 记录到日志
        self.logger.error(f"错误类型: {error_type}")
        self.logger.error(f"错误代码: {getattr(error, 'error_code', 'UNKNOWN')}")
        self.logger.error(f"错误信息: {str(error)}")

        if hasattr(error, 'details') and error.details:
            self.logger.error(f"错误详情: {json.dumps(error.details, ensure_ascii=False)}")

        if context:
            self.logger.error(f"上下文信息: {json.dumps(context, ensure_ascii=False)}")

    def save_error_report(self, error: Exception, report_path: Optional[str] = None):
        """
        保存错误报告

        Args:
            error: 异常对象
            report_path: 报告保存路径(如果为None则使用默认路径)
        """
        if report_path is None:
            report_path = self.log_dir / f"error_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        error_report = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_code': getattr(error, 'error_code', 'UNKNOWN'),
            'message': str(error),
            'details': getattr(error, 'details', {}),
            'traceback': traceback.format_exc(),
            'error_counts': self.error_counts,
            'recent_errors': self.error_history[-10:]  # 最近10条错误
        }

        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(error_report, f, ensure_ascii=False, indent=2)
            self.logger.info(f"错误报告已保存: {report_path}")
        except Exception as e:
            self.logger.error(f"保存错误报告失败: {e}")

    def get_error_statistics(self) -> Dict[str, Any]:
        """
        获取错误统计信息

        Returns:
            错误统计信息
        """
        total_errors = sum(self.error_counts.values())
        unique_errors = len(self.error_counts)

        # 按错误类型统计
        error_by_type = {}
        for error_record in self.error_history:
            error_type = error_record['error_type']
            error_by_type[error_type] = error_by_type.get(error_type, 0) + 1

        # 最近一小时错误数量
        from datetime import timedelta
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = sum(1 for record in self.error_history
                         if datetime.strptime(record['timestamp'], '%Y-%m-%d %H:%M:%S') > one_hour_ago)

        return {
            'total_errors': total_errors,
            'unique_error_types': unique_errors,
            'error_by_type': error_by_type,
            'recent_errors_1h': recent_errors,
            'error_counts': self.error_counts
        }

    def handle_critical_error(self, error: Exception, context: Optional[Dict] = None):
        """
        处理严重错误

        Args:
            error: 异常对象
            context: 上下文信息
        """
        self.log_error(error, context)
        self.save_error_report(error)

        # 输出严重错误到stderr
        print(f"\n严重错误发生！", file=sys.stderr)
        print(f"错误类型: {type(error).__name__}", file=sys.stderr)
        print(f"错误信息: {str(error)}", file=sys.stderr)

        # 如果是交易相关的错误,提供恢复建议
        if isinstance(error, TradingError):
            print(f"\n错误代码: {error.error_code}", file=sys.stderr)
            if error.details:
                print(f"详细信息: {json.dumps(error.details, ensure_ascii=False, indent=2)}", file=sys.stderr)

        print("\n请检查错误报告获取详细信息。\n", file=sys.stderr)

def handle_errors(func: Callable) -> Callable:
    """
    错误处理装饰器

    Args:
        func: 要装饰的函数

    Returns:
        装饰后的函数
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        error_handler = ErrorHandler()
        try:
            return func(*args, **kwargs)
        except TradingError as te:
            error_handler.handle_critical_error(te, {'function': func.__name__})
            raise  # 重新抛出交易错误
        except Exception as e:
            error_handler.handle_critical_error(e, {'function': func.__name__})
            raise TradingError(
                f"执行 {func.__name__} 时发生未预期的错误: {str(e)}",
                "UNEXPECTED_ERROR",
                {"original_error": str(e), "function": func.__name__}
            )
    return wrapper

class ErrorRecovery:
    """错误恢复策略"""

    @staticmethod
    def recover_from_api_error(error: APIConnectionError, retry_count: int = 3, delay: float = 1.0):
        """
        API错误恢复策略

        Args:
            error: API连接错误
            retry_count: 重试次数
            delay: 重试延迟(秒)

        Returns:
            恢复结果
        """
        recovery_result = {
            'success': False,
            'message': '',
            'action_taken': None,
            'remaining_retries': retry_count
        }

        if retry_count > 0:
            recovery_result['message'] = f"将在 {delay} 秒后重试(剩余 {retry_count} 次)"
            recovery_result['action_taken'] = 'retry'
            recovery_result['remaining_retries'] = retry_count - 1
        else:
            recovery_result['message'] = "API连接失败,已达到最大重试次数"
            recovery_result['action_taken'] = 'stop'

        return recovery_result

    @staticmethod
    def recover_from_market_data_error(error: MarketDataError):
        """
        市场数据错误恢复策略

        Args:
            error: 市场数据错误

        Returns:
            恢复结果
        """
        recovery_result = {
            'success': False,
            'message': '',
            'action_taken': None
        }

        if error.symbol:
            recovery_result['message'] = f"跳过品种 {error.symbol},使用历史缓存数据"
            recovery_result['action_taken'] = 'skip_symbol'
        else:
            recovery_result['message'] = "市场数据获取失败,系统将退出"
            recovery_result['action_taken'] = 'stop'

        return recovery_result

    @staticmethod
    def recover_from_signal_error(error: SignalGenerationError):
        """
        信号生成错误恢复策略

        Args:
            error: 信号生成错误

        Returns:
            恢复结果
        """
        recovery_result = {
            'success': False,
            'message': '',
            'action_taken': None
        }

        if error.symbol:
            recovery_result['message'] = f"跳过 {error.symbol} 的信号生成,继续其他品种"
            recovery_result['action_taken'] = 'skip_symbol'
        else:
            recovery_result['message'] = "信号生成失败,使用保守策略"
            recovery_result['action_taken'] = 'conservative'

        return recovery_result

    @staticmethod
    def recover_from_trade_error(error: TradeExecutionError):
        """
        交易执行错误恢复策略

        Args:
            error: 交易执行错误

        Returns:
            恢复结果
        """
        recovery_result = {
            'success': False,
            'message': '',
            'action_taken': None
        }

        if error.order_id:
            recovery_result['message'] = f"订单 {error.order_id} 执行失败,将取消该订单"
            recovery_result['action_taken'] = 'cancel_order'
        else:
            recovery_result['message'] = "交易执行失败,降低仓位"
            recovery_result['action_taken'] = 'reduce_position'

        return recovery_result

# 全局错误处理器实例
global_error_handler: Optional[ErrorHandler] = None

def get_global_error_handler() -> ErrorHandler:
    """获取全局错误处理器实例"""
    global global_error_handler
    if global_error_handler is None:
        global_error_handler = ErrorHandler()
    return global_error_handler

def set_global_error_handler(error_handler: ErrorHandler):
    """设置全局错误处理器实例"""
    global global_error_handler
    global_error_handler = error_handler