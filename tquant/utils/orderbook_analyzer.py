"""
订单簿分析系统
分析市场微观结构,包括深度分析、价格发现和市场压力评估
"""

import logging
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class OrderBookLevel(Enum):
    """订单簿层级"""
    BEST_BID = "最佳买价"
    BEST_ASK = "最佳卖价"
    SPREAD = "买卖价差"
    DEPTH_5 = "深度5档"
    DEPTH_10 = "深度10档"
    DEPTH_20 = "深度20档"

class MarketPressure(Enum):
    """市场压力类型"""
    BUY_SIDE = "买方压力"
    SELL_SIDE = "卖方压力"
    BALANCED = "平衡市场"
    EXTREME_BUY = "极端买方"
    EXTREME_SELL = "极端卖方"

@dataclass
class OrderBookData:
    """订单簿数据"""
    timestamp: datetime
    symbol: str
    bids: List[Tuple[float, int]]  # (价格, 数量)
    asks: List[Tuple[float, int]]  # (价格, 数量)
    last_price: float
    volume: int
    turnover: float

@dataclass
class OrderBookSnapshot:
    """订单簿快照"""
    timestamp: datetime
    symbol: str
    bid_levels: Dict[float, int]  # 价格到数量的映射
    ask_levels: Dict[float, int]
    bid_volume: int
    ask_volume: int
    spread: float
    mid_price: float
    imbalance_ratio: float
    total_depth: int

@dataclass
class LiquidityMetrics:
    """流动性指标"""
    bid_ask_spread: float
    spread_percentage: float
    depth_at_spread: int
    total_depth: int
    market_imbalance: float
    price_impact: float
    liquidity_score: float
    trading_activity: float

class OrderBookAnalyzer:
    """订单簿分析器"""

    def __init__(self, window_size: int = 100, depth_levels: int = 10):
        """
        初始化订单簿分析器

        Args:
            window_size: 分析窗口大小
            depth_levels: 分析深度档位数
        """
        self.window_size = window_size
        self.depth_levels = depth_levels
        self.orderbook_history = deque(maxlen=window_size)
        self.price_history = deque(maxlen=window_size)
        self.trade_history = deque(maxlen=window_size)
        self.snapshot_history = deque(maxlen=window_size)

        # 统计数据
        self.total_analyzed = 0
        self.spread_statistics = {
            'mean': 0,
            'std': 0,
            'min': float('inf'),
            'max': 0,
            'median': 0
        }

    def analyze_orderbook(self, orderbook_data: OrderBookData) -> Dict[str, Any]:
        """
        分析订单簿数据

        Args:
            orderbook_data: 订单簿数据

        Returns:
            分析结果
        """
        try:
            # 保存到历史
            self.orderbook_history.append(orderbook_data)
            self.price_history.append(orderbook_data.last_price)

            # 创建快照
            snapshot = self._create_orderbook_snapshot(orderbook_data)
            self.snapshot_history.append(snapshot)

            # 计算流动性指标
            liquidity_metrics = self._calculate_liquidity_metrics(snapshot)

            # 计算市场压力
            market_pressure = self._assess_market_pressure(snapshot)

            # 计算价格发现
            price_discovery = self._analyze_price_discovery(orderbook_data)

            # 检测异常模式
            anomalies = self._detect_anomalies(snapshot)

            # 计算市场深度
            depth_analysis = self._analyze_depth_profile(snapshot)

            # 生成交易信号
            trading_signals = self._generate_trading_signals(snapshot, liquidity_metrics)

            # 创建分析报告
            analysis_report = {
                'timestamp': orderbook_data.timestamp.isoformat(),
                'symbol': orderbook_data.symbol,
                'last_price': orderbook_data.last_price,
                'liquidity_metrics': asdict(liquidity_metrics),
                'market_pressure': market_pressure.value,
                'price_discovery': price_discovery,
                'anomalies': anomalies,
                'depth_analysis': depth_analysis,
                'trading_signals': trading_signals,
                'snapshot': asdict(snapshot)
            }

            # 更新统计
            self._update_statistics(snapshot)

            self.total_analyzed += 1

            return analysis_report

        except Exception as e:
            logger.error(f"订单簿分析失败: {e}")
            return {
                'timestamp': orderbook_data.timestamp.isoformat(),
                'symbol': orderbook_data.symbol,
                'error': str(e)
            }

    def _create_orderbook_snapshot(self, orderbook_data: OrderBookData) -> OrderBookSnapshot:
        """创建订单簿快照"""
        # 获取N档深度
        bid_levels = {}
        ask_levels = {}

        bid_prices = [price for price, _ in orderbook_data.bids[:self.depth_levels]]
        ask_prices = [price for price, _ in orderbook_data.asks[:self.depth_levels]]

        # 计算各档深度
        for price, quantity in orderbook_data.bids[:self.depth_levels]:
            if price in bid_prices:
                bid_levels[price] = bid_levels.get(price, 0) + quantity

        for price, quantity in orderbook_data.asks[:self.depth_levels]:
            if price in ask_prices:
                ask_levels[price] = ask_levels.get(price, 0) + quantity

        # 计算基本指标
        if bid_levels and ask_levels:
            best_bid = max(bid_levels.keys())
            best_ask = min(ask_levels.keys())
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            bid_volume = sum(bid_levels.values())
            ask_volume = sum(ask_levels.values())
            imbalance_ratio = (ask_volume - bid_volume) / (ask_volume + bid_volume) if (ask_volume + bid_volume) > 0 else 0
            total_depth = bid_volume + ask_volume
        else:
            best_bid = 0
            best_ask = 0
            spread = 0
            mid_price = orderbook_data.last_price
            bid_volume = 0
            ask_volume = 0
            imbalance_ratio = 0
            total_depth = 0

        return OrderBookSnapshot(
            timestamp=orderbook_data.timestamp,
            symbol=orderbook_data.symbol,
            bid_levels=bid_levels,
            ask_levels=ask_levels,
            bid_volume=bid_volume,
            ask_volume=ask_volume,
            spread=spread,
            mid_price=mid_price,
            imbalance_ratio=imbalance_ratio,
            total_depth=total_depth
        )

    def _calculate_liquidity_metrics(self, snapshot: OrderBookSnapshot) -> LiquidityMetrics:
        """计算流动性指标"""
        # 买卖价差
        spread = snapshot.spread
        spread_percentage = (spread / snapshot.mid_price) * 100 if snapshot.mid_price > 0 else 0

        # 价差处深度
        depth_at_spread = (snapshot.ask_levels.get(snapshot.bid_levels.keys()[0] if snapshot.bid_levels else 0, 0) +
                          snapshot.bid_levels.get(snapshot.ask_levels.keys()[0] if snapshot.ask_levels else 0, 0))

        # 总深度
        total_depth = snapshot.total_depth

        # 市场不平衡
        market_imbalance = snapshot.imbalance_ratio

        # 价格冲击(模拟)
        price_impact = spread / (total_depth + 1) * 10000  # 简化的价格冲击计算

        # 流动性评分 (0-100)
        liquidity_score = 100
        liquidity_score -= spread_percentage * 10  # 价差越小,流动性越好
        liquidity_score -= abs(market_imbalance) * 20  # 不平衡越小,流动性越好
        liquidity_score -= price_impact * 0.1
        liquidity_score = max(0, min(100, liquidity_score))

        # 交易活跃度
        trading_activity = (total_depth / 1000) * 100  # 假设1000为基准

        return LiquidityMetrics(
            bid_ask_spread=spread,
            spread_percentage=spread_percentage,
            depth_at_spread=depth_at_spread,
            total_depth=total_depth,
            market_imbalance=market_imbalance,
            price_impact=price_impact,
            liquidity_score=liquidity_score,
            trading_activity=trading_activity
        )

    def _assess_market_pressure(self, snapshot: OrderBookSnapshot) -> MarketPressure:
        """评估市场压力"""
        imbalance = snapshot.imbalance_ratio

        if imbalance > 0.3:
            return MarketPressure.EXTREME_SELL
        elif imbalance < -0.3:
            return MarketPressure.EXTREME_BUY
        elif imbalance > 0.1:
            return MarketPressure.SELL_SIDE
        elif imbalance < -0.1:
            return MarketPressure.BUY_SIDE
        else:
            return MarketPressure.BALANCED

    def _analyze_price_discovery(self, orderbook_data: OrderBookData) -> Dict[str, Any]:
        """分析价格发现"""
        if len(self.price_history) < 2:
            return {
                'price_efficiency': 0,
                'price_discovery_rate': 0,
                'convergence_speed': 0
            }

        # 计算价格效率(价格调整速度)
        price_changes = []
        for i in range(1, len(self.price_history)):
            change = (self.price_history[i] - self.price_history[i-1]) / self.price_history[i-1]
            price_changes.append(abs(change))

        price_efficiency = np.mean(price_changes) if price_changes else 0

        # 价格发现率(信息融入速度)
        if len(orderbook_data.bids) > 0 and len(orderbook_data.asks) > 0:
            best_bid = max(orderbook_data.bids, key=lambda x: x[0])[0]
            best_ask = min(orderbook_data.asks, key=lambda x: x[0])[0]
            mid_price = (best_bid + best_ask) / 2

            if len(self.price_history) > 0:
                price_diff = abs(orderbook_data.last_price - mid_price) / mid_price
                price_discovery_rate = 1 - price_diff
            else:
                price_discovery_rate = 0
        else:
            price_discovery_rate = 0

        # 收敛速度(价差收敛速度)
        recent_spreads = [snapshot.spread for snapshot in list(self.snapshot_history)[-10:]]
        if len(recent_spreads) > 1:
            spread_changes = [abs(recent_spreads[i] - recent_spreads[i-1]) for i in range(1, len(recent_spreads))]
            convergence_speed = 1 - np.mean(spread_changes) if spread_changes else 0
        else:
            convergence_speed = 0

        return {
            'price_efficiency': price_efficiency,
            'price_discovery_rate': price_discovery_rate,
            'convergence_speed': convergence_speed
        }

    def _detect_anomalies(self, snapshot: OrderBookSnapshot) -> List[Dict[str, Any]]:
        """检测异常模式"""
        anomalies = []

        # 检测异常价差
        if snapshot.spread > 0.01 * snapshot.mid_price:  # 价差超过1%
            anomalies.append({
                'type': '异常价差',
                'severity': '高',
                'description': f'买卖价差过大: {snapshot.spread:.2f} ({snapshot.spread_percentage:.2f}%)',
                'timestamp': snapshot.timestamp.isoformat()
            })

        # 检测异常不平衡
        if abs(snapshot.imbalance_ratio) > 0.5:
            anomalies.append({
                'type': '市场不平衡',
                'severity': '高',
                'description': f'市场严重不平衡: {snapshot.imbalance_ratio:.3f}',
                'timestamp': snapshot.timestamp.isoformat()
            })

        # 检测深度突变
        if len(self.snapshot_history) > 0:
            prev_snapshot = list(self.snapshot_history)[-1]
            depth_change = abs(snapshot.total_depth - prev_snapshot.total_depth) / prev_snapshot.total_depth
            if depth_change > 0.5:  # 深度变化超过50%
                anomalies.append({
                    'type': '深度突变',
                    'severity': '中',
                    'description': f'订单簿深度突变: {depth_change:.2%}',
                    'timestamp': snapshot.timestamp.isoformat()
                })

        # 检测异常挂单
        for price, quantity in snapshot.bid_levels.items():
            if quantity > 10000:  # 单个挂单超过1万手
                anomalies.append({
                    'type': '大额挂单',
                    'severity': '中',
                    'description': f'检测到大额买单: {price:.2f} x {quantity}',
                    'timestamp': snapshot.timestamp.isoformat()
                })

        for price, quantity in snapshot.ask_levels.items():
            if quantity > 10000:  # 单个挂单超过1万手
                anomalies.append({
                    'type': '大额挂单',
                    'severity': '中',
                    'description': f'检测到大额卖单: {price:.2f} x {quantity}',
                    'timestamp': snapshot.timestamp.isoformat()
                })

        return anomalies

    def _analyze_depth_profile(self, snapshot: OrderBookSnapshot) -> Dict[str, Any]:
        """分析深度分布"""
        bid_prices = list(snapshot.bid_levels.keys())
        ask_prices = list(snapshot.ask_levels.keys())

        # 计算深度集中度
        if len(bid_prices) > 0:
            max_bid_depth = max(snapshot.bid_levels.values())
            total_bid_depth = sum(snapshot.bid_levels.values())
            bid_concentration = max_bid_depth / total_bid_depth if total_bid_depth > 0 else 0
        else:
            bid_concentration = 0

        if len(ask_prices) > 0:
            max_ask_depth = max(snapshot.ask_levels.values())
            total_ask_depth = sum(snapshot.ask_levels.values())
            ask_concentration = max_ask_depth / total_ask_depth if total_ask_depth > 0 else 0
        else:
            ask_concentration = 0

        # 计算深度分布斜度
        bid_depths = list(snapshot.bid_levels.values())
        ask_depths = list(snapshot.bid_levels.values())

        bid_skew = self._calculate_skewness(bid_depths)
        ask_skew = self._calculate_skewness(ask_depths)

        # 计算深度衰减率
        if len(bid_prices) > 1:
            bid_depths_sorted = sorted(snapshot.bid_levels.values(), reverse=True)
            bid_decay_rate = self._calculate_decay_rate(bid_depths_sorted)
        else:
            bid_decay_rate = 0

        if len(ask_prices) > 1:
            ask_depths_sorted = sorted(snapshot.ask_levels.values(), reverse=True)
            ask_decay_rate = self._calculate_decay_rate(ask_depths_sorted)
        else:
            ask_decay_rate = 0

        return {
            'bid_concentration': bid_concentration,
            'ask_concentration': ask_concentration,
            'bid_skewness': bid_skew,
            'ask_skewness': ask_skew,
            'bid_decay_rate': bid_decay_rate,
            'ask_decay_rate': ask_decay_rate,
            'depth_balance': (sum(bid_depths) + 1) / (sum(ask_depths) + 1)
        }

    def _calculate_skewness(self, data: List[float]) -> float:
        """计算偏度"""
        if len(data) < 3:
            return 0

        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0

        skewness = np.mean([((x - mean) / std) ** 3 for x in data])
        return skewness

    def _calculate_decay_rate(self, depths: List[float]) -> float:
        """计算深度衰减率"""
        if len(depths) < 2:
            return 0

        # 使用指数衰减模型
        x = np.arange(len(depths))
        y = np.array(depths)

        # 简化的衰减率计算
        if y[0] > 0:
            decay_rate = (y[0] - y[-1]) / y[0]
            return decay_rate
        return 0

    def _generate_trading_signals(self, snapshot: OrderBookSnapshot, liquidity_metrics: LiquidityMetrics) -> List[Dict[str, Any]]:
        """生成交易信号"""
        signals = []

        # 基于流动性评分的信号
        if liquidity_metrics.liquidity_score < 30:
            signals.append({
                'signal_type': 'LIQUIDITY_WARNING',
                'strength': 'STRONG',
                'description': '市场流动性极低,谨慎交易',
                'recommendation': '等待流动性改善'
            })

        # 基于市场压力的信号
        if snapshot.imbalance_ratio < -0.2:
            signals.append({
                'signal_type': 'BUY_PRESSURE',
                'strength': 'MEDIUM',
                'description': '买方压力强劲,考虑做多',
                'recommendation': '关注买入机会'
            })
        elif snapshot.imbalance_ratio > 0.2:
            signals.append({
                'signal_type': 'SELL_PRESSURE',
                'strength': 'MEDIUM',
                'description': '卖方压力强劲,考虑做空',
                'recommendation': '关注卖出机会'
            })

        # 基于价差的信号
        if liquidity_metrics.spread_percentage > 0.5:
            signals.append({
                'signal_type': 'WIDE_SPREAD',
                'strength': 'MEDIUM',
                'description': '买卖价差过大',
                'recommendation': '等待价差收敛'
            })

        # 基于深度的信号
        if snapshot.total_depth < 1000:  # 假设1000为基准
            signals.append({
                'signal_type': 'SHALLOW_DEPTH',
                'strength': 'MEDIUM',
                'description': '市场深度较浅',
                'recommendation': '注意价格冲击'
            })

        return signals

    def _update_statistics(self, snapshot: OrderBookSnapshot):
        """更新统计信息"""
        if snapshot.spread > 0:
            spreads = [s.spread for s in self.snapshot_history]
            if spreads:
                self.spread_statistics['mean'] = np.mean(spreads)
                self.spread_statistics['std'] = np.std(spreads)
                self.spread_statistics['min'] = min(spreads)
                self.spread_statistics['max'] = max(spreads)
                self.spread_statistics['median'] = np.median(spreads)

    def get_market_microstructure_summary(self) -> Dict[str, Any]:
        """获取市场微观结构摘要"""
        if not self.snapshot_history:
            return {}

        latest_snapshot = list(self.snapshot_history)[-1]
        liquidity_metrics = self._calculate_liquidity_metrics(latest_snapshot)

        return {
            'symbol': latest_snapshot.symbol,
            'timestamp': latest_snapshot.timestamp.isoformat(),
            'last_price': latest_snapshot.mid_price,
            'spread': latest_snapshot.spread,
            'spread_percentage': liquidity_metrics.spread_percentage,
            'market_imbalance': latest_snapshot.imbalance_ratio,
            'liquidity_score': liquidity_metrics.liquidity_score,
            'total_depth': latest_snapshot.total_depth,
            'analysis_count': self.total_analyzed,
            'statistics': self.spread_statistics
        }

    def simulate_orderbook_data(self, symbol: str) -> OrderBookData:
        """模拟订单簿数据"""
        base_price = {
            "SHFE.cu2401": 68000,
            "DCE.i2403": 920,
            "CZCE.MA401": 2850,
            "SHFE.al2401": 18500,
            "ZCE.CF401": 12500
        }.get(symbol, 10000)

        # 模拟买卖盘
        bid_levels = []
        ask_levels = []

        # 生成买盘(从低到高)
        for i in range(10):
            price = base_price * (1 - 0.001 * (i + 1))  # 逐档降低0.1%
            quantity = random.randint(100, 1000)
            bid_levels.append((price, quantity))

        # 生成卖盘(从高到低)
        for i in range(10):
            price = base_price * (1 + 0.001 * (i + 1))  # 逐档提高0.1%
            quantity = random.randint(100, 1000)
            ask_levels.append((price, quantity))

        # 添加一些大单
        if random.random() > 0.7:  # 30%概率出现大单
            big_price = base_price * (1 + random.choice([-0.005, 0.005]))
            big_quantity = random.randint(5000, 20000)
            if random.random() > 0.5:
                bid_levels.append((big_price, big_quantity))
            else:
                ask_levels.append((big_price, big_quantity))

        last_price = base_price * (1 + random.uniform(-0.001, 0.001))

        return OrderBookData(
            timestamp=datetime.now(),
            symbol=symbol,
            bids=sorted(bid_levels, key=lambda x: x[0], reverse=True)[:20],  # 取前20档
            asks=sorted(ask_levels, key=lambda x: x[0])[:20],  # 取前20档
            last_price=last_price,
            volume=random.randint(10000, 100000),
            turnover=last_price * random.randint(10000, 100000)
        )