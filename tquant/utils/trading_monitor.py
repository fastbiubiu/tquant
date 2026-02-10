"""
交易监控系统
实时监控交易状态、异常告警和日志记录
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """告警数据类"""
    alert_id: str
    alert_type: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime
    metadata: Dict = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class TradingMonitor:
    """交易监控系统"""

    def __init__(self, config: Dict):
        """
        初始化监控系统

        Args:
            config: 监控配置
        """
        self.config = config

        # 监控配置
        self.alert_thresholds = config.get('alert_thresholds', {
            'max_positions': 5,
            'risk_ratio': 0.8,
            'daily_loss': 0.05,
            'order_failure_rate': 0.3
        })

        self.monitoring_interval = config.get('monitoring_interval', 60)

        # 监控数据
        self.alerts: List[Alert] = []
        self.tracking_data: Dict[str, Any] = {}

        # 统计数据
        self.stats = {
            'alerts_generated': 0,
            'alerts_resolved': 0,
            'monitoring_runs': 0
        }

        # 异步监控任务
        self.monitoring_task = None
        self.is_monitoring = False

        logger.info(f"交易监控系统初始化完成,间隔: {self.monitoring_interval}秒")

    def initialize_tracking(self, trader_data: Dict):
        """
        初始化跟踪数据

        Args:
            trader_data: 交易器数据
        """
        self.tracking_data = {
            'symbol_activity': {},
            'order_status': {},
            'trade_performance': {},
            'portfolio_value': {},
            'risk_metrics': {},
            'timestamps': {}
        }

        # 初始化跟踪数据
        for symbol in trader_data.get('symbols', []):
            self.tracking_data['symbol_activity'][symbol] = {'buys': 0, 'sells': 0, 'last_trade': None}
            self.tracking_data['order_status'][symbol] = {}
            self.tracking_data['trade_performance'][symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}

        logger.info(f"初始化跟踪数据: {len(tracker_data.get('symbols', []))} 个品种")

    def record_alert(self, alert_type: str = None, level: str = None, message: str = None, type: str = None, metadata: Dict = None, **kwargs):
        """
        记录告警

        Args:
            alert_type: 告警类型
            level: 告警级别
            message: 告警消息
            type: 告警类型(备选参数)
            metadata: 告警元数据
        """
        # 处理参数兼容性
        if type and not alert_type:
            alert_type = type
        if not alert_type:
            alert_type = kwargs.get('type', 'UNKNOWN')
        if not level:
            level = kwargs.get('level', 'INFO')
        if not message:
            message = kwargs.get('message', '')

        alert = Alert(
            alert_id=str(int(time.time() * 1000)),
            alert_type=alert_type,
            level=level,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        self.alerts.append(alert)
        self.stats['alerts_generated'] += 1

        logger.warning(f"[{level}] {alert_type}: {message}")
        self._notify_alert(alert)

    def _notify_alert(self, alert: Alert):
        """通知告警"""
        # 在实际应用中,这里可以发送到各种通知渠道
        # Telegram, 微信, 邮件等
        pass

    def monitor_positions(self, positions: Dict, account_info: Dict):
        """
        监控持仓

        Args:
            positions: 持仓信息
            account_info: 账户信息
        """
        max_positions = self.alert_thresholds['max_positions']
        current_positions = len(positions)

        # 检查持仓数量
        if current_positions > max_positions:
            self.record_alert(
                alert_type='POSITION_LIMIT',
                level='WARNING',
                message=f'持仓数量达到限制: {current_positions}/{max_positions}',
                metadata={'current': current_positions, 'max': max_positions}
            )

        # 检查风险比率
        risk_ratio = account_info.get('risk_ratio', 0)
        risk_threshold = self.alert_thresholds['risk_ratio']

        if risk_ratio > risk_threshold:
            self.record_alert(
                alert_type='RISK_HIGH',
                level='WARNING',
                message=f'风险比率过高: {risk_ratio:.1%}',
                metadata={'risk_ratio': risk_ratio, 'threshold': risk_threshold}
            )

    def monitor_trades(self, trade_results: List[Dict], trader_data: Dict):
        """
        监控交易结果

        Args:
            trade_results: 交易结果列表
            trader_data: 交易器数据
        """
        successful_trades = sum(1 for r in trade_results if r.get('success', False))
        failed_trades = len(trade_results) - successful_trades
        failure_rate = successful_trades / len(trade_results) if trade_results else 0

        failure_threshold = self.alert_thresholds['order_failure_rate']

        # 检查失败率
        if failure_rate > failure_threshold:
            self.record_alert(
                alert_type='TRADE_FAILURE_HIGH',
                level='WARNING',
                message=f'交易失败率过高: {failure_rate:.1%}',
                metadata={'failure_rate': failure_rate, 'threshold': failure_threshold}
            )

        # 记录交易活动
        for result in trade_results:
            symbol = result.get('symbol')
            if symbol:
                if symbol not in self.tracking_data['symbol_activity']:
                    self.tracking_data['symbol_activity'][symbol] = {'buys': 0, 'sells': 0, 'last_trade': None}

                if result.get('action') in ['BUY', 'OPEN']:
                    self.tracking_data['symbol_activity'][symbol]['buys'] += 1
                    self.tracking_data['symbol_activity'][symbol]['last_trade'] = datetime.now()

                elif result.get('action') in ['SELL', 'CLOSE']:
                    self.tracking_data['symbol_activity'][symbol]['sells'] += 1
                    self.tracking_data['symbol_activity'][symbol]['last_trade'] = datetime.now()

                # 记录交易性能
                if symbol not in self.tracking_data['trade_performance']:
                    self.tracking_data['trade_performance'][symbol] = {'wins': 0, 'losses': 0, 'pnl': 0}

                if result.get('success', False):
                    profit_loss = result.get('profit_loss', 0)
                    self.tracking_data['trade_performance'][symbol]['pnl'] += profit_loss
                    if profit_loss > 0:
                        self.tracking_data['trade_performance'][symbol]['wins'] += 1
                    else:
                        self.tracking_data['trade_performance'][symbol]['losses'] += 1

    def generate_daily_report(self) -> Dict:
        """
        生成日报

        Returns:
            日报数据
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'alerts_count': len(self.alerts),
            'alerts_by_level': self._get_alerts_by_level(),
            'tracking_data': self.tracking_data,
            'stats': self.stats
        }

        return report

    def _get_alerts_by_level(self) -> Dict[str, int]:
        """按级别统计告警"""
        alerts_by_level = {'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}

        for alert in self.alerts:
            level = alert.level
            if level in alerts_by_level:
                alerts_by_level[level] += 1

        return alerts_by_level

    def get_active_alerts(self, level: str = None) -> List[Alert]:
        """
        获取活跃告警

        Args:
            level: 告警级别(None 表示所有级别)

        Returns:
            活跃告警列表
        """
        alerts = self.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts

    def resolve_alert(self, alert_id: str):
        """
        解决告警

        Args:
            alert_id: 告警 ID
        """
        alert = next((a for a in self.alerts if a.alert_id == alert_id), None)

        if alert:
            self.alerts.remove(alert)
            self.stats['alerts_resolved'] += 1
            logger.info(f"告警已解决: {alert_id}")

    def get_monitoring_summary(self) -> str:
        """
        获取监控摘要

        Returns:
            摘要文本
        """
        summary = f"\n=== 交易监控摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"

        # 告警统计
        alerts = self.get_active_alerts()
        summary += f"告警统计:\n"
        summary += f"• 当前告警: {len(alerts)}\n"
        summary += f"• 总告警数: {self.stats['alerts_generated']}\n"
        summary += f"• 已解决: {self.stats['alerts_resolved']}\n\n"

        # 告警按级别
        alerts_by_level = self._get_alerts_by_level()
        summary += f"告警级别:\n"
        for level, count in alerts_by_level.items():
            summary += f"  • {level}: {count}\n"
        summary += "\n"

        # 活跃告警
        if alerts:
            summary += "活跃告警:\n"
            for alert in alerts[:5]:  # 最多显示 5 个
                summary += f"  [{alert.level}] {alert.alert_type}: {alert.message}\n"
            summary += "\n"

        return summary

    async def start_monitoring(self, trader_data: Dict):
        """
        启动异步监控

        Args:
            trader_data: 交易器数据
        """
        if self.is_monitoring:
            logger.warning("监控已经在运行")
            return

        self.is_monitoring = True
        self.initialize_tracking(trader_data)

        logger.info("启动监控任务")

        while self.is_monitoring:
            try:
                # 等待间隔时间
                await asyncio.sleep(self.monitoring_interval)

                # 更新监控统计
                self.stats['monitoring_runs'] += 1

                # 执行监控检查
                self._perform_monitoring_checks()

            except asyncio.CancelledError:
                logger.info("监控任务被取消")
                break
            except Exception as e:
                logger.error(f"监控任务出错: {e}")

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        logger.info("监控任务已停止")

    def close(self):
        """关闭监控系统"""
        self.stop_monitoring()
        logger.info("监控系统已关闭")

    def _perform_monitoring_checks(self):
        """执行监控检查"""
        # 在实际应用中,这里会调用 trader 实例的监控方法
        # 这里只是模拟
        pass

    def save_report(self, filename: str = None):
        """
        保存监控报告

        Args:
            filename: 文件名(None 表示自动生成)
        """
        if filename is None:
            filename = f"logs/trading_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            report = self.generate_daily_report()
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, default=str)

            logger.info(f"监控报告已保存: {filename}")
        except Exception as e:
            logger.error(f"保存监控报告失败: {e}")


class LogRecorder:
    """日志记录器"""

    def __init__(self, config: Dict):
        """
        初始化日志记录器

        Args:
            config: 日志配置
        """
        self.config = config
        self.log_file = config.get('log_file', 'logs/trading.log')
        self.max_log_size = config.get('max_log_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = config.get('backup_count', 5)

        # 日志缓存
        self.log_buffer = []
        self.buffer_size = config.get('buffer_size', 100)

    def log(self, level: str, message: str, metadata: Dict = None):
        """
        记录日志

        Args:
            level: 日志级别
            message: 日志消息
            metadata: 元数据
        """
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'metadata': metadata or {}
        }

        self.log_buffer.append(log_entry)

        # 写入文件
        self._write_log(log_entry)

        # 检查缓冲区大小
        if len(self.log_buffer) >= self.buffer_size:
            self._flush_buffer()

    def _write_log(self, log_entry: Dict):
        """写入日志文件"""
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            logger.error(f"写入日志失败: {e}")

    def _flush_buffer(self):
        """刷新缓冲区"""
        if self.log_buffer:
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    for entry in self.log_buffer:
                        f.write(json.dumps(entry) + '\n')
                self.log_buffer = []
            except Exception as e:
                logger.error(f"刷新缓冲区失败: {e}")

    def get_recent_logs(self, limit: int = 50) -> List[Dict]:
        """
        获取最近的日志

        Args:
            limit: 限制数量

        Returns:
            日志列表
        """
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                logs = [json.loads(line) for line in f]
                return logs[-limit:] if logs else []
        except Exception as e:
            logger.error(f"读取日志失败: {e}")
            return []

    def close(self):
        """关闭日志记录器"""
        self._flush_buffer()
        logger.info("日志记录器已关闭")
