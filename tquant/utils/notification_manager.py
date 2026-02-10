"""
通知系统
支持 Telegram、微信和邮件通知
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


@dataclass
class Notification:
    """通知数据类"""
    type: str  # telegram, wechat, email
    recipient: str
    title: str
    content: str
    level: str  # INFO, WARNING, ERROR, SUCCESS
    timestamp: datetime
    metadata: Dict = None

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class NotificationManager:
    """通知管理器"""

    def __init__(self, config: Dict):
        """
        初始化通知管理器

        Args:
            config: 通知配置
        """
        self.config = config

        # 通知配置
        self.notifications_enabled = config.get('enabled', False)
        self.notification_types = config.get('notification_types', ['telegram'])

        # Telegram 配置
        self.telegram_config = config.get('telegram', {})
        self.telegram_enabled = self.notifications_enabled and self.telegram_config.get('enabled', False)

        # 微信配置
        self.wechat_config = config.get('wechat', {})
        self.wechat_enabled = self.notifications_enabled and self.wechat_config.get('enabled', False)

        # 邮件配置
        self.email_config = config.get('email', {})
        self.email_enabled = self.notifications_enabled and self.email_config.get('enabled', False)

        # 通知历史
        self.notification_history: List[Notification] = []
        self.max_history = config.get('max_history', 1000)

        # 通知队列
        self.notification_queue = []
        self.is_running = False

        # 异步发送任务
        self.sender_task = None

        logger.info(f"通知管理器初始化完成: {len(self.notification_types)} 种通知类型")

    async def send_notification(
        self,
        type: str,
        recipient: str,
        title: str,
        content: str,
        level: str = 'INFO',
        metadata: Dict = None
    ) -> bool:
        """
        发送通知

        Args:
            type: 通知类型
            recipient: 接收者
            title: 标题
            content: 内容
            level: 级别
            metadata: 元数据

        Returns:
            是否成功
        """
        if not self.notifications_enabled:
            return False

        # 检查通知类型是否启用
        if type not in self.notification_types:
            return False

        notification = Notification(
            type=type,
            recipient=recipient,
            title=title,
            content=content,
            level=level,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )

        # 添加到队列
        self.notification_queue.append(notification)

        # 如果发送任务没有运行,启动它
        if not self.is_running:
            self._start_sender()

        return True

    def _start_sender(self):
        """启动异步发送任务"""
        if self.is_running:
            return

        self.is_running = True
        self.sender_task = asyncio.create_task(self._sender_loop())

        logger.info("启动通知发送任务")

    async def _sender_loop(self):
        """异步发送循环"""
        while self.is_running:
            try:
                await asyncio.sleep(1)  # 每秒处理一次

                # 处理队列中的通知
                while self.notification_queue:
                    notification = self.notification_queue.pop(0)

                    # 异步发送通知
                    asyncio.create_task(self._send_single_notification(notification))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"通知发送错误: {e}")

    async def _send_single_notification(self, notification: Notification):
        """
        发送单个通知

        Args:
            notification: 通知对象
        """
        try:
            logger.info(f"[{notification.type.upper()}] 发送通知到 {notification.recipient}")

            if notification.type == 'telegram':
                await self._send_telegram(notification)
            elif notification.type == 'wechat':
                await self._send_wechat(notification)
            elif notification.type == 'email':
                await self._send_email(notification)

            # 记录到历史
            self.notification_history.append(notification)
            if len(self.notification_history) > self.max_history:
                self.notification_history = self.notification_history[-self.max_history:]

        except Exception as e:
            logger.error(f"发送 {notification.type} 通知失败: {e}")

    async def _send_telegram(self, notification: Notification):
        """
        发送 Telegram 通知

        Args:
            notification: 通知对象
        """
        if not self.telegram_enabled:
            return

        try:
            import requests

            # Telegram API
            api_token = self.telegram_config.get('api_token')
            chat_id = self.telegram_config.get('chat_id')

            if not api_token or not chat_id:
                logger.warning("Telegram 配置不完整")
                return

            # 构建消息
            message = f"<b>{notification.title}</b>\n\n{notification.content}"

            # 发送请求
            url = f"https://api.telegram.org/bot{api_token}/sendMessage"
            response = requests.post(
                url,
                json={
                    'chat_id': chat_id,
                    'text': message,
                    'parse_mode': 'HTML',
                    'disable_web_page_preview': True
                },
                timeout=30
            )

            if response.status_code != 200:
                logger.error(f"Telegram 发送失败: {response.text}")

        except Exception as e:
            logger.error(f"Telegram 通知发送错误: {e}")

    async def _send_wechat(self, notification: Notification):
        """
        发送微信通知

        Args:
            notification: 通知对象
        """
        if not self.wechat_enabled:
            return

        try:
            # 微信通知实现(这里只是框架,实际需要企业微信 API)
            logger.info(f"[微信] 发送通知到 {notification.recipient}")

            # 这里可以实现企业微信 webhook 通知
            # webhook_url = self.wechat_config.get('webhook_url')
            # requests.post(webhook_url, json={'text': notification.content})

        except Exception as e:
            logger.error(f"微信通知发送错误: {e}")

    async def _send_email(self, notification: Notification):
        """
        发送邮件通知

        Args:
            notification: 通知对象
        """
        if not self.email_enabled:
            return

        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # 邮件配置
            smtp_server = self.email_config.get('smtp_server')
            smtp_port = self.email_config.get('smtp_port', 587)
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            sender_email = self.email_config.get('sender_email', username)
            recipients = self.email_config.get('recipients', [notification.recipient])

            if not smtp_server or not username:
                logger.warning("邮件配置不完整")
                return

            # 创建邮件
            msg = MIMEMultipart()
            msg['From'] = sender_email
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = notification.title
            msg['Date'] = datetime.now()

            # 邮件内容
            body = f"""
            <h1>{notification.title}</h1>
            <p><strong>级别:</strong> {notification.level}</p>
            <p><strong>时间:</strong> {notification.timestamp}</p>
            <hr>
            <p>{notification.content}</p>
            """
            msg.attach(MIMEText(body, 'html'))

            # 发送邮件
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)

            logger.info(f"邮件发送成功到 {recipients}")

        except Exception as e:
            logger.error(f"邮件通知发送错误: {e}")

    def send_alert_notification(self, alert: Dict) -> bool:
        """
        发送告警通知

        Args:
            alert: 告警数据

        Returns:
            是否成功
        """
        return self.send_notification(
            type='telegram',
            recipient=alert.get('recipient', 'admin'),
            title=f"🚨 {alert.get('type', 'Alert')}",
            content=alert.get('message', ''),
            level=alert.get('level', 'WARNING'),
            metadata=alert
        )

    def send_trade_notification(self, trade: Dict) -> bool:
        """
        发送交易通知

        Args:
            trade: 交易数据

        Returns:
            是否成功
        """
        if trade.get('success', False):
            return self.send_notification(
                type='telegram',
                recipient=trade.get('recipient', 'admin'),
                title=f"✅ {trade.get('symbol', 'Trade')} 交易成功",
                content=f"操作: {trade.get('action', '')}\n数量: {trade.get('volume', 0)}\n价格: {trade.get('price', 0)}",
                level='SUCCESS',
                metadata=trade
            )
        else:
            return self.send_notification(
                type='telegram',
                recipient=trade.get('recipient', 'admin'),
                title=f"❌ {trade.get('symbol', 'Trade')} 交易失败",
                content=trade.get('message', '未知错误'),
                level='ERROR',
                metadata=trade
            )

    def get_notification_history(self, limit: int = 100) -> List[Notification]:
        """
        获取通知历史

        Args:
            limit: 限制数量

        Returns:
            通知列表
        """
        return self.notification_history[-limit:]

    def get_notification_summary(self) -> str:
        """
        获取通知摘要

        Returns:
            摘要文本
        """
        summary = f"\n=== 通知摘要 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n"

        # 通知统计
        total = len(self.notification_history)
        by_type = {}
        by_level = {}

        for notification in self.notification_history:
            # 按类型统计
            if notification.type not in by_type:
                by_type[notification.type] = 0
            by_type[notification.type] += 1

            # 按级别统计
            if notification.level not in by_level:
                by_level[notification.level] = 0
            by_level[notification.level] += 1

        summary += f"通知统计:\n"
        summary += f"• 总通知数: {total}\n\n"

        summary += "按类型:\n"
        for type_name, count in by_type.items():
            summary += f"  • {type_name}: {count}\n"
        summary += "\n"

        summary += "按级别:\n"
        for level, count in by_level.items():
            summary += f"  • {level}: {count}\n"
        summary += "\n"

        # 最近的通知
        recent = self.get_notification_history(10)
        if recent:
            summary += "最近通知:\n"
            for notification in recent:
                summary += f"  [{notification.timestamp}] [{notification.level}] {notification.title}\n"

        return summary

    def close(self):
        """关闭通知管理器"""
        self.is_running = False

        if self.sender_task:
            self.sender_task.cancel()

        logger.info("通知管理器已关闭")


class NotificationTemplate:
    """通知模板"""

    @staticmethod
    def format_trade_notification(trade: Dict) -> Dict:
        """
        格式化交易通知

        Args:
            trade: 交易数据

        Returns:
            通知配置
        """
        return {
            'recipient': trade.get('recipient', 'admin'),
            'title': f"{'✅' if trade.get('success') else '❌'} {trade.get('symbol')} {trade.get('action')}",
            'content': f"""
            {trade.get('symbol', 'Trade')} {trade.get('action', '')} 交易
            - 数量: {trade.get('volume', 0)}
            - 价格: {trade.get('price', 0)}
            - 状态: {'成功' if trade.get('success') else '失败'}
            - 订单ID: {trade.get('order_id', 'N/A')}
            """,
            'level': 'SUCCESS' if trade.get('success') else 'ERROR',
            'metadata': trade
        }

    @staticmethod
    def format_risk_alert(risk_data: Dict) -> Dict:
        """
        格式化风险告警通知

        Args:
            risk_data: 风险数据

        Returns:
            通知配置
        """
        return {
            'recipient': risk_data.get('recipient', 'admin'),
            'title': f"🚨 风险告警: {risk_data.get('type', 'Risk')}",
            'content': f"""
            风险类型: {risk_data.get('type', 'Unknown')}
            当前值: {risk_data.get('value', 'N/A')}
            阈值: {risk_data.get('threshold', 'N/A')}
            消息: {risk_data.get('message', '')}
            """,
            'level': 'WARNING',
            'metadata': risk_data
        }

    @staticmethod
    def format_system_alert(alert_data: Dict) -> Dict:
        """
        格式化系统告警通知

        Args:
            alert_data: 告警数据

        Returns:
            通知配置
        """
        return {
            'recipient': alert_data.get('recipient', 'admin'),
            'title': f"⚠️ 系统告警: {alert_data.get('type', 'System Alert')}",
            'content': f"""
            告警类型: {alert_data.get('type', 'Unknown')}
            详细信息: {alert_data.get('message', '')}
            时间: {alert_data.get('timestamp', 'N/A')}
            """,
            'level': 'ERROR',
            'metadata': alert_data
        }
