"""
结果记录器
记录交易结果、统计分析、可视化和数据导出
"""

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """信号类型枚举"""
    STRONG_BUY = 'STRONG_BUY'
    BUY = 'BUY'
    HOLD = 'HOLD'
    SELL = 'SELL'
    STRONG_SELL = 'STRONG_SELL'


@dataclass
class TradeRecord:
    """交易记录数据类"""
    trade_id: str
    symbol: str
    action: str  # BUY, SELL, CLOSE
    direction: str  # LONG, SHORT
    volume: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    profit_loss: float
    profit_loss_percent: float
    signal_type: str
    confidence: float
    order_id: Optional[str] = None
    commission: float = 0.0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    total_trades: int
    win_trades: int
    loss_trades: int
    win_rate: float
    total_profit: float
    total_loss: float
    net_profit: float
    profit_factor: float
    max_drawdown: float
    average_profit: float
    average_loss: float
    sharpe_ratio: Optional[float] = None
    max_trades_in_day: int = 0

    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class ResultRecorder:
    """结果记录器"""

    def __init__(self, config: Dict):
        """
        初始化结果记录器

        Args:
            config: 记录器配置
        """
        self.config = config
        self.records: List[TradeRecord] = []
        self.records_file = config.get('records_file', 'data/trading_records.json')
        self.daily_file = config.get('daily_file', 'data/trading_daily.json')
        self.daily_summary_file = config.get('daily_summary_file', 'data/trading_daily_summary.json')
        self.history_file = config.get('history_file', 'data/trading_history.json')

        logger.info("结果记录器初始化完成")

    def record_trade(self, trade_data: Dict) -> TradeRecord:
        """
        记录交易

        Args:
            trade_data: 交易数据

        Returns:
            交易记录
        """
        trade_id = f"{trade_data['symbol']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        trade = TradeRecord(
            trade_id=trade_id,
            symbol=trade_data['symbol'],
            action=trade_data.get('action', ''),
            direction=trade_data.get('direction', ''),
            volume=trade_data.get('volume', 0),
            entry_price=trade_data.get('entry_price', 0),
            exit_price=trade_data.get('exit_price', 0),
            entry_time=trade_data.get('entry_time', datetime.now()),
            exit_time=trade_data.get('exit_time', datetime.now()),
            profit_loss=trade_data.get('profit_loss', 0),
            profit_loss_percent=trade_data.get('profit_loss_percent', 0),
            signal_type=trade_data.get('signal_type', ''),
            confidence=trade_data.get('confidence', 0),
            order_id=trade_data.get('order_id'),
            commission=trade_data.get('commission', 0)
        )

        self.records.append(trade)
        self._save_to_file()

        logger.info(f"交易记录已保存: {trade_id}")

        return trade

    def get_records(self, symbol: str = None, limit: int = None) -> List[TradeRecord]:
        """
        获取交易记录

        Args:
            symbol: 品种(None 表示全部)
            limit: 限制数量

        Returns:
            交易记录列表
        """
        records = self.records

        if symbol:
            records = [r for r in records if r.symbol == symbol]

        if limit:
            records = records[-limit:]

        return records

    def get_performance_metrics(self) -> PerformanceMetrics:
        """
        获取性能指标

        Returns:
            性能指标
        """
        if not self.records:
            return PerformanceMetrics(
                total_trades=0,
                win_trades=0,
                loss_trades=0,
                win_rate=0,
                total_profit=0,
                total_loss=0,
                net_profit=0,
                profit_factor=0,
                max_drawdown=0,
                average_profit=0,
                average_loss=0
            )

        # 基本统计
        total_profit = sum(r.profit_loss for r in self.records if r.profit_loss > 0)
        total_loss = abs(sum(r.profit_loss for r in self.records if r.profit_loss < 0))
        net_profit = total_profit - total_loss

        # 交易统计
        win_trades = sum(1 for r in self.records if r.profit_loss > 0)
        loss_trades = sum(1 for r in self.records if r.profit_loss < 0)
        win_rate = win_trades / len(self.records) * 100 if self.records else 0

        # 盈亏比
        profit_factor = total_loss / total_profit if total_profit > 0 else 0

        # 平均盈亏
        average_profit = total_profit / win_trades if win_trades > 0 else 0
        average_loss = total_loss / loss_trades if loss_trades > 0 else 0

        # 最大回撤(简化计算)
        max_drawdown = self._calculate_max_drawdown()

        # Sharpe 比率(简化计算)
        sharpe_ratio = self._calculate_sharpe_ratio()

        # 每日最大交易数
        max_trades_in_day = self._get_max_trades_in_day()

        return PerformanceMetrics(
            total_trades=len(self.records),
            win_trades=win_trades,
            loss_trades=loss_trades,
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            max_drawdown=max_drawdown,
            average_profit=average_profit,
            average_loss=average_loss,
            sharpe_ratio=sharpe_ratio,
            max_trades_in_day=max_trades_in_day
        )

    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if not self.records:
            return 0

        # 计算累计净值
        cumulative_pnl = 0
        max_pnl = 0
        max_drawdown = 0

        for record in self.records:
            cumulative_pnl += record.profit_loss
            max_pnl = max(max_pnl, cumulative_pnl)
            drawdown = (max_pnl - cumulative_pnl) / max_pnl if max_pnl > 0 else 0
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown * 100

    def _calculate_sharpe_ratio(self) -> Optional[float]:
        """计算 Sharpe 比率(简化版)"""
        if len(self.records) < 2:
            return None

        # 计算收益率
        returns = []
        for i in range(1, len(self.records)):
            prev_record = self.records[i-1]
            curr_record = self.records[i]

            prev_value = prev_record.profit_loss
            curr_value = curr_record.profit_loss
            change = curr_value - prev_value
            prev_return = prev_value / abs(prev_value) if prev_value != 0 else 0
            change_return = change / abs(prev_value) if prev_value != 0 else 0
            returns.append(change_return)

        if not returns:
            return None

        # 计算收益率统计
        avg_return = sum(returns) / len(returns)
        variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
        std_dev = variance ** 0.5

        if std_dev == 0:
            return None

        # Sharpe 比率(假设无风险利率为 0)
        sharpe = avg_return / std_dev * (252 ** 0.5)

        return sharpe

    def _get_max_trades_in_day(self) -> int:
        """获取每日最大交易数"""
        if not self.records:
            return 0

        trades_per_day = {}
        for record in self.records:
            day_key = record.entry_time.strftime('%Y-%m-%d')
            trades_per_day[day_key] = trades_per_day.get(day_key, 0) + 1

        return max(trades_per_day.values()) if trades_per_day else 0

    def generate_daily_summary(self) -> Dict:
        """
        生成日报摘要

        Returns:
            日报摘要
        """
        if not self.records:
            return {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'total_trades': 0,
                'total_profit': 0,
                'win_rate': 0
            }

        # 按日期分组
        trades_by_day = {}
        for record in self.records:
            day_key = record.entry_time.strftime('%Y-%m-%d')
            if day_key not in trades_by_day:
                trades_by_day[day_key] = []
            trades_by_day[day_key].append(record)

        daily_summaries = []

        for day_key, trades in trades_by_day.items():
            total_profit = sum(t.profit_loss for t in trades)
            win_trades = sum(1 for t in trades if t.profit_loss > 0)
            win_rate = win_trades / len(trades) * 100 if trades else 0

            daily_summaries.append({
                'date': day_key,
                'trades': len(trades),
                'wins': win_trades,
                'losses': len(trades) - win_trades,
                'total_profit': total_profit,
                'win_rate': win_rate
            })

        return {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'daily_summaries': daily_summaries,
            'overall_metrics': self.get_performance_metrics().to_dict()
        }

    def export_to_csv(self, filename: str = None):
        """
        导出为 CSV 文件

        Args:
            filename: 文件名(None 表示自动生成)
        """
        if not self.records:
            logger.warning("没有交易记录可导出")
            return

        if filename is None:
            filename = f"data/trading_records_{datetime.now().strftime('%Y%m%d')}.csv"

        try:
            data = []
            for record in self.records:
                data.append(asdict(record))

            df = pd.DataFrame(data)
            df.to_csv(filename, index=False, encoding='utf-8-sig')
            logger.info(f"CSV 导出成功: {filename}")
        except Exception as e:
            logger.error(f"CSV 导出失败: {e}")

    def export_to_json(self, filename: str = None):
        """
        导出为 JSON 文件

        Args:
            filename: 文件名(None 表示自动生成)
        """
        if not self.records:
            logger.warning("没有交易记录可导出")
            return

        if filename is None:
            filename = f"data/trading_records_{datetime.now().strftime('%Y%m%d')}.json"

        try:
            data = [asdict(record) for record in self.records]
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"JSON 导出成功: {filename}")
        except Exception as e:
            logger.error(f"JSON 导出失败: {e}")

    def _save_to_file(self):
        """保存到文件"""
        self.export_to_json(self.records_file)

    def get_recent_performance(self, limit: int = 20) -> Dict:
        """
        获取最近性能

        Args:
            limit: 限制数量

        Returns:
            性能数据
        """
        records = self.records[-limit:]

        total_profit = sum(r.profit_loss for r in records)
        win_trades = sum(1 for r in records if r.profit_loss > 0)

        return {
            'total_trades': len(records),
            'win_trades': win_trades,
            'loss_trades': len(records) - win_trades,
            'win_rate': win_trades / len(records) * 100 if records else 0,
            'total_profit': total_profit,
            'average_profit': total_profit / len(records) if records else 0
        }

    def get_symbol_performance(self, symbol: str) -> Dict:
        """
        获取特定品种性能

        Args:
            symbol: 品种

        Returns:
            性能数据
        """
        symbol_records = [r for r in self.records if r.symbol == symbol]

        if not symbol_records:
            return {
                'symbol': symbol,
                'trades': 0,
                'win_rate': 0,
                'total_profit': 0
            }

        win_trades = sum(1 for r in symbol_records if r.profit_loss > 0)

        return {
            'symbol': symbol,
            'trades': len(symbol_records),
            'win_rate': win_trades / len(symbol_records) * 100 if symbol_records else 0,
            'total_profit': sum(r.profit_loss for r in symbol_records),
            'average_profit': sum(r.profit_loss for r in symbol_records) / len(symbol_records) if symbol_records else 0
        }

    def close(self):
        """关闭记录器"""
        self._save_to_file()
        logger.info("结果记录器已关闭")

    def generate_performance_report(self) -> str:
        """
        生成性能报告

        Returns:
            报告文本
        """
        metrics = self.get_performance_metrics()
        sharpe_str = f"{metrics.sharpe_ratio:.2f}" if metrics.sharpe_ratio else "N/A"

        report = f"""
=== 交易性能报告 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ===

基本统计:
  • 总交易数: {metrics.total_trades}
  • 盈利交易: {metrics.win_trades}
  • 亏损交易: {metrics.loss_trades}
  • 胜率: {metrics.win_rate:.2f}%

盈亏统计:
  • 总盈利: {metrics.total_profit:.2f}
  • 总亏损: {metrics.total_loss:.2f}
  • 净盈利: {metrics.net_profit:.2f}
  • 盈亏比: {metrics.profit_factor:.2f}

风险指标:
  • 最大回撤: {metrics.max_drawdown:.2f}%
  • Sharpe 比率: {sharpe_str}
  • 每日最大交易数: {metrics.max_trades_in_day}

平均数据:
  • 平均盈利: {metrics.average_profit:.2f}
  • 平均亏损: {metrics.average_loss:.2f}

报告生成时间: {datetime.now().isoformat()}
"""
        return report.strip()
