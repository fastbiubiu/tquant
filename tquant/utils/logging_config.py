"""
日志配置模块
提供统一的日志配置和管理功能
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ColoredFormatter(logging.Formatter):
    """彩色日志格式化器"""

    # ANSI颜色代码
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[35m',  # 紫色
        'RESET': '\033[0m'       # 重置
    }

    def format(self, record):
        # 添加颜色
        if record.levelname in self.COLORS:
            record.levelname = f"{self.COLORS[record.levelname]}{record.levelname}{self.COLORS['RESET']}"

        return super().format(record)

class RotatingFileHandler(logging.handlers.RotatingFileHandler):
    """自定义的日志轮转处理器"""

    def __init__(self, filename, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8', **kwargs):
        """
        初始化

        Args:
            filename: 日志文件名
            maxBytes: 最大字节数
            backupCount: 备份文件数量
            encoding: 编码
        """
        # 确保目录存在
        log_dir = Path(filename).parent
        log_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(filename, maxBytes=maxBytes, backupCount=backupCount, encoding=encoding, **kwargs)

class LoggingConfigManager:
    """日志配置管理器"""

    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化日志配置管理器

        Args:
            config: 配置字典
        """
        self.config = config or self._get_default_config()
        self.loggers = {}
        self.handlers = {}

        # 创建日志目录
        log_dir = Path(self.config.get('log_dir', 'logs'))
        log_dir.mkdir(exist_ok=True)

    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'log_dir': 'logs',
            'log_level': 'INFO',
            'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'max_file_size': 10 * 1024 * 1024,  # 10MB
            'backup_count': 5,
            'enable_console': True,
            'enable_file': True,
            'enable_json': False,
            'colored_output': True,
            'log_rotation': True,
            'log_compression': True
        }

    def configure_root_logger(self) -> logging.Logger:
        """
        配置根日志记录器

        Returns:
            配置好的根日志记录器
        """
        root_logger = logging.getLogger()
        root_logger.setLevel(self._get_log_level())

        # 清除现有的处理器
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # 创建控制台处理器
        if self.config.get('enable_console', True):
            console_handler = self._create_console_handler()
            root_logger.addHandler(console_handler)

        # 创建文件处理器
        if self.config.get('enable_file', True):
            file_handler = self._create_file_handler('trading_system.log')
            root_logger.addHandler(file_handler)

        # 创建JSON日志处理器
        if self.config.get('enable_json', False):
            json_handler = self._create_json_handler('trading_system.json')
            root_logger.addHandler(json_handler)

        return root_logger

    def _get_log_level(self) -> int:
        """获取日志级别"""
        level_str = self.config.get('log_level', 'INFO').upper()
        return getattr(logging, level_str, logging.INFO)

    def _create_console_handler(self) -> logging.Handler:
        """创建控制台处理器"""
        console_handler = logging.StreamHandler(sys.stdout)

        if self.config.get('colored_output', True):
            formatter = ColoredFormatter(
                self.config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )
        else:
            formatter = logging.Formatter(
                self.config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            )

        console_handler.setFormatter(formatter)
        console_handler.setLevel(self._get_log_level())

        return console_handler

    def _create_file_handler(self, filename: str) -> logging.Handler:
        """创建文件处理器"""
        file_path = Path(self.config.get('log_dir', 'logs')) / filename

        if self.config.get('log_rotation', True):
            file_handler = RotatingFileHandler(
                file_path,
                maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
                backupCount=self.config.get('backup_count', 5),
                encoding='utf-8'
            )
        else:
            file_handler = logging.FileHandler(file_path, encoding='utf-8')

        formatter = logging.Formatter(
            self.config.get('log_format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(self._get_log_level())

        return file_handler

    def _create_json_handler(self, filename: str) -> logging.Handler:
        """创建JSON日志处理器"""
        file_path = Path(self.config.get('log_dir', 'logs')) / filename

        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_data = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'logger': record.name,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line': record.lineno
                }

                if record.exc_info:
                    log_data['exception'] = self.formatException(record.exc_info)

                return json.dumps(log_data, ensure_ascii=False)

        json_handler = RotatingFileHandler(
            file_path,
            maxBytes=self.config.get('max_file_size', 10 * 1024 * 1024),
            backupCount=self.config.get('backup_count', 5),
            encoding='utf-8'
        )

        json_handler.setFormatter(JSONFormatter())
        json_handler.setLevel(self._get_log_level())

        return json_handler

    def get_logger(self, name: str) -> logging.Logger:
        """
        获取或创建日志记录器

        Args:
            name: 日志记录器名称

        Returns:
            日志记录器
        """
        if name not in self.loggers:
            logger = logging.getLogger(name)
            logger.setLevel(self._get_log_level())
            self.loggers[name] = logger

        return self.loggers[name]

    def setup_module_loggers(self) -> Dict[str, logging.Logger]:
        """
        设置模块特定的日志记录器

        Returns:
            模块日志记录器字典
        """
        module_loggers = {}

        # 交易模块
        module_loggers['trading'] = self.get_logger('trading')
        module_loggers['trading'].info("交易模块日志记录器已初始化")

        # 市场分析模块
        module_loggers['market_analyst'] = self.get_logger('market_analyst')
        module_loggers['market_analyst'].info("市场分析模块日志记录器已初始化")

        # 交易执行模块
        module_loggers['trader'] = self.get_logger('trader')
        module_loggers['trader'].info("交易执行模块日志记录器已初始化")

        # 工作流模块
        module_loggers['workflow'] = self.get_logger('workflow')
        module_loggers['workflow'].info("工作流模块日志记录器已初始化")

        # 回测模块
        module_loggers['backtest'] = self.get_logger('backtest')
        module_loggers['backtest'].info("回测模块日志记录器已初始化")

        # 配置模块
        module_loggers['config'] = self.get_logger('config')
        module_loggers['config'].info("配置模块日志记录器已初始化")

        # 工具模块
        module_loggers['utils'] = self.get_logger('utils')
        module_loggers['utils'].info("工具模块日志记录器已初始化")

        return module_loggers

    def create_performance_logger(self) -> logging.Logger:
        """
        创建性能监控日志记录器

        Returns:
            性能日志记录器
        """
        perf_logger = self.get_logger('performance')

        # 添加性能专用的文件处理器
        perf_handler = self._create_file_handler('performance.log')
        perf_logger.addHandler(perf_handler)

        return perf_logger

    def create_audit_logger(self) -> logging.Logger:
        """
        创建审计日志记录器

        Returns:
            审计日志记录器
        """
        audit_logger = self.get_logger('audit')

        # 添加审计专用的文件处理器
        audit_handler = self._create_file_handler('audit.log')
        audit_logger.addHandler(audit_handler)

        # 审计日志级别设置为INFO
        audit_logger.setLevel(logging.INFO)

        # 设置审计日志格式(更简洁)
        audit_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        audit_handler.setFormatter(audit_formatter)

        return audit_logger

    def get_log_statistics(self) -> Dict[str, Any]:
        """
        获取日志统计信息

        Returns:
            日志统计信息
        """
        stats = {
            'log_dir': self.config.get('log_dir', 'logs'),
            'log_level': self.config.get('log_level', 'INFO'),
            'configured_loggers': len(self.loggers),
            'log_files': []
        }

        # 获取日志文件信息
        log_dir = Path(self.config.get('log_dir', 'logs'))
        if log_dir.exists():
            for log_file in log_dir.glob('*.log'):
                file_size = log_file.stat().st_size
                file_size_mb = file_size / (1024 * 1024)

                stats['log_files'].append({
                    'filename': log_file.name,
                    'size_mb': round(file_size_mb, 2),
                    'last_modified': datetime.fromtimestamp(log_file.stat().st_mtime).isoformat()
                })

        # 按文件大小排序
        stats['log_files'].sort(key=lambda x: x['size_mb'], reverse=True)

        return stats

    def cleanup_old_logs(self, days: int = 30):
        """
        清理旧的日志文件

        Args:
            days: 保留天数
        """
        log_dir = Path(self.config.get('log_dir', 'logs'))
        cutoff_time = datetime.now().timestamp() - (days * 24 * 60 * 60)

        if log_dir.exists():
            for log_file in log_dir.glob('*.log*'):
                if log_file.stat().st_mtime < cutoff_time:
                    try:
                        log_file.unlink()
                        logging.info(f"已删除旧日志文件: {log_file}")
                    except Exception as e:
                        logging.error(f"删除日志文件失败 {log_file}: {e}")

# 全局日志配置管理器实例
logging_manager: Optional[LoggingConfigManager] = None

def setup_logging(config: Dict[str, Any] = None) -> LoggingConfigManager:
    """
    设置日志系统

    Args:
        config: 日志配置字典

    Returns:
        日志配置管理器
    """
    global logging_manager
    logging_manager = LoggingConfigManager(config)
    logging_manager.configure_root_logger()
    return logging_manager

def get_logger(name: str) -> logging.Logger:
    """
    获取日志记录器

    Args:
        name: 日志记录器名称

    Returns:
        日志记录器
    """
    global logging_manager
    if logging_manager is None:
        logging_manager = setup_logging()
    return logging_manager.get_logger(name)

def get_audit_logger() -> logging.Logger:
    """
    获取审计日志记录器

    Returns:
        审计日志记录器
    """
    global logging_manager
    if logging_manager is None:
        logging_manager = setup_logging()
    return logging_manager.create_audit_logger()

def get_performance_logger() -> logging.Logger:
    """
    获取性能日志记录器

    Returns:
        性能日志记录器
    """
    global logging_manager
    if logging_manager is None:
        logging_manager = setup_logging()
    return logging_manager.create_performance_logger()