"""
流动性分析系统
分析市场流动性,包括成交量分析、价格冲击、深度分析等
"""

import logging
from collections import deque, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class LiquidityType(Enum):
    """流动性类型"""
    MARKET_DEPTH = "市场深度"
    ORDER_FLOW = "订单流"
    PRICE_IMPACT = "价格冲击"
    VOLUME_PROFILE = "成交量分布"
    TICK_SIZE = "买卖价差"
    SPREAD_ANALYSIS = "价差分析"
    TIME_WEIGHTED = "时间加权流动性"
    ADJUSTED_SPREAD = "调整后价差"

class LiquidityLevel(Enum):
    """流动性等级"""
    VERY_HIGH = "极高"
    HIGH = "高"
    MEDIUM = "中等"
    LOW = "低"
    VERY_LOW = "极低"

@dataclass
class MarketDataPoint:
    """市场数据点"""
    timestamp: datetime
    symbol: str
    price: float
    volume: int
    turnover: float
    bid_size: int
    ask_size: int
    bid_price: float
    ask_price: float
    trade_count: int

@dataclass
class LiquidityMetrics:
    """流动性指标"""
    time_weighted_spread: float
    price_impact: float
    market_depth: float
    volume_profile: Dict[str, float]
    liquidity_score: float
    liquidity_level: LiquidityLevel
    turnover_ratio: float
    volume_concentration: float
    depth_concentration: float
    spread_volatility: float

class LiquidityAnalyzer:
    """流动性分析器"""

    def __init__(self, window_size: int = 100, lookback_period: int = 20):
        """
        初始化流动性分析器

        Args:
            window_size: 分析窗口大小
            lookback_period: 回看周期
        """
        self.window_size = window_size
        self.lookback_period = lookback_period
        self.market_data_history = deque(maxlen=window_size)
        self.volume_history = deque(maxlen=window_size)
        self.price_history = deque(maxlen=window_size)
        self.spread_history = deque(maxlen=window_size)
        self.trade_count_history = deque(maxlen=window_size)

        # 成交量分布
        self.volume_profile = defaultdict(int)
        self.price_levels = {}

        # 深度分析
        self.depth_levels = {}
        self.order_flow = deque(maxlen=window_size)

        # 统计数据
        self.total_analyzed = 0
        self.liquidity_statistics = {
            'avg_spread': 0,
            'avg_depth': 0,
            'avg_impact': 0,
            'avg_score': 0
        }

    def analyze_market_data(self, market_data: MarketDataPoint) -> Dict[str, Any]:
        """
        分析市场数据

        Args:
            market_data: 市场数据点

        Returns:
            分析结果
        """
        try:
            # 保存到历史
            self.market_data_history.append(market_data)
            self.volume_history.append(market_data.volume)
            self.price_history.append(market_data.price)

            # 计算价差
            spread = market_data.ask_price - market_data.bid_price
            self.spread_history.append(spread)

            # 记录成交量分布
            price_level = int(market_data.price / 10) * 10  # 按档位分组
            self.volume_profile[price_level] += market_data.volume

            # 更新价格深度
            if market_data.symbol not in self.depth_levels:
                self.depth_levels[market_data.symbol] = {
                    'bid_depth': [],
                    'ask_depth': [],
                    'time_series': deque(maxlen=self.window_size)
                }

            # 计算流动性指标
            liquidity_metrics = self._calculate_liquidity_metrics(market_data)

            # 分析流动性特征
            liquidity_features = self._analyze_liquidity_features(market_data, liquidity_metrics)

            # 生成交易建议
            trading_recommendations = self._generate_trading_recommendations(
                liquidity_metrics, liquidity_features
            )

            # 检测流动性事件
            liquidity_events = self._detect_liquidity_events(market_data)

            # 创建分析报告
            analysis_report = {
                'timestamp': market_data.timestamp.isoformat(),
                'symbol': market_data.symbol,
                'current_price': market_data.price,
                'current_volume': market_data.volume,
                'current_turnover': market_data.turnover,
                'bid_ask_spread': spread,
                'bid_ask_spread_pct': (spread / market_data.price) * 100 if market_data.price > 0 else 0,
                'liquidity_metrics': asdict(liquidity_metrics),
                'liquidity_features': liquidity_features,
                'trading_recommendations': trading_recommendations,
                'liquidity_events': liquidity_events,
                'volume_profile': dict(self.volume_profile)
            }

            # 更新统计
            self._update_statistics(liquidity_metrics)
            self.total_analyzed += 1

            return analysis_report

        except Exception as e:
            logger.error(f"流动性分析失败: {e}")
            return {
                'timestamp': market_data.timestamp.isoformat(),
                'symbol': market_data.symbol,
                'error': str(e)
            }

    def _calculate_liquidity_metrics(self, market_data: MarketDataPoint) -> LiquidityMetrics:
        """计算流动性指标"""
        # 时间加权价差
        time_weighted_spread = self._calculate_time_weighted_spread()

        # 价格冲击
        price_impact = self._calculate_price_impact(market_data)

        # 市场深度
        market_depth = self._calculate_market_depth(market_data)

        # 成交量分布
        volume_profile = self._calculate_volume_profile()

        # 流动性评分 (0-100)
        liquidity_score = self._calculate_liquidity_score(
            time_weighted_spread, price_impact, market_depth
        )

        # 流动性等级
        liquidity_level = self._determine_liquidity_level(liquidity_score)

        # 换手率
        turnover_ratio = self._calculate_turnover_ratio(market_data)

        # 成交量集中度
        volume_concentration = self._calculate_volume_concentration()

        # 深度集中度
        depth_concentration = self._calculate_depth_concentration()

        # 价差波动率
        spread_volatility = self._calculate_spread_volatility()

        return LiquidityMetrics(
            time_weighted_spread=time_weighted_spread,
            price_impact=price_impact,
            market_depth=market_depth,
            volume_profile=volume_profile,
            liquidity_score=liquidity_score,
            liquidity_level=liquidity_level,
            turnover_ratio=turnover_ratio,
            volume_concentration=volume_concentration,
            depth_concentration=depth_concentration,
            spread_volatility=spread_volatility
        )

    def _calculate_time_weighted_spread(self) -> float:
        """计算时间加权价差"""
        if not self.spread_history or not self.market_data_history:
            return 0

        # 计算时间权重(越近的数据权重越高)
        total_weight = 0
        weighted_spread = 0

        for i, spread in enumerate(list(self.spread_history)):
            weight = (i + 1) / len(self.spread_history)  # 线性权重
            weighted_spread += spread * weight
            total_weight += weight

        return weighted_spread / total_weight if total_weight > 0 else 0

    def _calculate_price_impact(self, market_data: MarketDataPoint) -> float:
        """计算价格冲击"""
        if len(self.price_history) < 2:
            return 0

        # 计算成交量加权价格变化
        total_volume = sum(self.volume_history)
        if total_volume == 0:
            return 0

        price_change = abs(market_data.price - self.price_history[-1])
        volume_weighted_impact = price_change * market_data.volume / total_volume

        # 标准化价格冲击
        return volume_weighted_impact * 10000  # 转换为基点

    def _calculate_market_depth(self, market_data: MarketDataPoint) -> float:
        """计算市场深度"""
        bid_depth = market_data.bid_size
        ask_depth = market_data.ask_size
        total_depth = bid_depth + ask_depth

        # 计算深度相对价格
        if market_data.price > 0:
            depth_ratio = total_depth / market_data.price
        else:
            depth_ratio = 0

        return depth_ratio

    def _calculate_volume_profile(self) -> Dict[str, float]:
        """计算成交量分布"""
        if not self.volume_profile:
            return {}

        total_volume = sum(self.volume_profile.values())
        volume_profile_normalized = {}

        for price_level, volume in self.volume_profile.items():
            percentage = (volume / total_volume) * 100 if total_volume > 0 else 0
            volume_profile_normalized[str(price_level)] = percentage

        # 返回前10个最活跃的价格档位
        sorted_profile = sorted(volume_profile_normalized.items(),
                               key=lambda x: x[1], reverse=True)
        return dict(sorted_profile[:10])

    def _calculate_liquidity_score(self, spread: float, impact: float, depth: float) -> float:
        """计算流动性评分"""
        # 基础分100
        score = 100

        # 价差惩罚
        if spread > 0:
            spread_penalty = spread * 1000  # 价差越大,惩罚越大
            score -= spread_penalty

        # 价格冲击惩罚
        impact_penalty = impact * 0.5  # 价格冲击越大,惩罚越大
        score -= impact_penalty

        # 深度奖励
        depth_bonus = depth * 10  # 深度越大,奖励越大
        score += depth_bonus

        # 确保分数在0-100之间
        return max(0, min(100, score))

    def _determine_liquidity_level(self, score: float) -> LiquidityLevel:
        """确定流动性等级"""
        if score >= 80:
            return LiquidityLevel.VERY_HIGH
        elif score >= 60:
            return LiquidityLevel.HIGH
        elif score >= 40:
            return LiquidityLevel.MEDIUM
        elif score >= 20:
            return LiquidityLevel.LOW
        else:
            return LiquidityLevel.VERY_LOW

    def _calculate_turnover_ratio(self, market_data: MarketDataPoint) -> float:
        """计算换手率"""
        if not self.market_data_history:
            return 0

        # 计算平均成交量和平均价格
        avg_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else 0
        avg_price = sum(self.price_history) / len(self.price_history) if self.price_history else 0

        if avg_volume == 0 or avg_price == 0:
            return 0

        # 换手率 = 成交量 / 流通市值(假设流通市值)
        market_cap = avg_price * 1000000  # 假设流通100万股
        turnover_ratio = market_data.turnover / market_cap

        return turnover_ratio * 100  # 转换为百分比

    def _calculate_volume_concentration(self) -> float:
        """计算成交量集中度"""
        if not self.volume_profile:
            return 0

        # 计算HHI指数(赫芬达尔-赫希曼指数)
        total_volume = sum(self.volume_profile.values())
        if total_volume == 0:
            return 0

        hhi = sum((volume / total_volume) ** 2 for volume in self.volume_profile.values())
        return hhi * 100  # 转换为百分比

    def _calculate_depth_concentration(self) -> float:
        """计算深度集中度"""
        if not self.depth_levels:
            return 0

        # 简化计算：计算最大深度占比
        total_depth = sum(market_data.bid_size + market_data.ask_size
                        for market_data in self.market_data_history)

        if total_depth == 0:
            return 0

        max_depth = 0
        for market_data in self.market_data_history:
            max_depth = max(max_depth, market_data.bid_size, market_data.ask_size)

        return (max_depth / total_depth) * 100

    def _calculate_spread_volatility(self) -> float:
        """计算价差波动率"""
        if len(self.spread_history) < 2:
            return 0

        spreads = list(self.spread_history)
        avg_spread = sum(spreads) / len(spreads)

        # 计算标准差
        variance = sum((s - avg_spread) ** 2 for s in spreads) / len(spreads)
        std_dev = variance ** 0.5

        # 相对波动率
        if avg_spread > 0:
            return (std_dev / avg_spread) * 100
        else:
            return 0

    def _analyze_liquidity_features(self, market_data: MarketDataPoint,
                                  liquidity_metrics: LiquidityMetrics) -> Dict[str, Any]:
        """分析流动性特征"""
        features = {}

        # 流动性趋势
        if len(self.market_data_history) > 1:
            prev_spread = list(self.spread_history)[-2]
            current_spread = liquidity_metrics.time_weighted_spread
            features['spread_trend'] = 'widening' if current_spread > prev_spread else 'narrowing'
        else:
            features['spread_trend'] = 'stable'

        # 市场活跃度
        avg_trade_count = sum(self.trade_count_history) / len(self.trade_count_history) if self.trade_count_history else 0
        current_trade_count = market_data.trade_count
        features['activity_level'] = 'high' if current_trade_count > avg_trade_count * 1.2 else 'normal'

        # 深度变化
        if len(self.market_data_history) > 1:
            prev_depth = sum(prev_market_data.bid_size + prev_market_data.ask_size
                           for prev_market_data in list(self.market_data_history)[-2:])
            current_depth = market_data.bid_size + market_data.ask_size
            features['depth_change'] = 'increasing' if current_depth > prev_depth else 'decreasing'
        else:
            features['depth_change'] = 'stable'

        # 成交量模式
        features['volume_pattern'] = self._identify_volume_pattern()

        return features

    def _identify_volume_pattern(self) -> str:
        """识别成交量模式"""
        if len(self.volume_history) < 5:
            return 'insufficient_data'

        recent_volumes = list(self.volume_history)[-5:]
        avg_volume = sum(recent_volumes) / len(recent_volumes)

        # 检测成交量激增
        if recent_volumes[-1] > avg_volume * 2:
            return 'surge'
        # 检测成交量萎缩
        elif recent_volumes[-1] < avg_volume * 0.5:
            return 'decline'
        # 检测成交量递增
        elif all(recent_volumes[i] > recent_volumes[i-1] for i in range(1, len(recent_volumes))):
            return 'increasing'
        # 检测成交量递减
        elif all(recent_volumes[i] < recent_volumes[i-1] for i in range(1, len(recent_volumes))):
            return 'decreasing'
        else:
            return 'stable'

    def _generate_trading_recommendations(self, liquidity_metrics: LiquidityMetrics,
                                       liquidity_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成交易建议"""
        recommendations = []

        # 基于流动性评分的建议
        if liquidity_metrics.liquidity_score < 30:
            recommendations.append({
                'type': 'LIQUIDITY_WARNING',
                'severity': 'HIGH',
                'message': '市场流动性极低,建议谨慎交易',
                'action': '等待流动性改善'
            })
        elif liquidity_metrics.liquidity_score < 60:
            recommendations.append({
                'type': 'LIQUIDITY_CAUTION',
                'severity': 'MEDIUM',
                'message': '市场流动性较低,注意交易成本',
                'action': '调整交易策略'
            })

        # 基于价差的建议
        if liquidity_features['spread_trend'] == 'widening':
            recommendations.append({
                'type': 'SPREAD_WIDENING',
                'severity': 'MEDIUM',
                'message': '买卖价差正在扩大',
                'action': '关注价格滑点风险'
            })

        # 基于成交量的建议
        if liquidity_features['volume_pattern'] == 'surge':
            recommendations.append({
                'type': 'VOLUME_SURGE',
                'severity': 'LOW',
                'message': '成交量激增,市场活跃',
                'action': '关注趋势变化'
            })

        # 基于深度的建议
        if liquidity_features['depth_change'] == 'decreasing':
            recommendations.append({
                'type': 'DEPTH_DECREASING',
                'severity': 'MEDIUM',
                'message': '市场深度正在减少',
                'action': '注意大单冲击风险'
            })

        # 基于流动性的具体建议
        if liquidity_metrics.liquidity_level == LiquidityLevel.VERY_HIGH:
            recommendations.append({
                'type': 'HIGH_LIQUIDITY',
                'severity': 'LOW',
                'message': '市场流动性极佳',
                'action': '适合大额交易'
            })

        return recommendations

    def _detect_liquidity_events(self, market_data: MarketDataPoint) -> List[Dict[str, Any]]:
        """检测流动性事件"""
        events = []

        # 检测异常大的成交量
        if len(self.volume_history) > 0:
            avg_volume = sum(self.volume_history) / len(self.volume_history)
            if market_data.volume > avg_volume * 3:
                events.append({
                    'event_type': 'VOLUME_SPIKE',
                    'severity': 'HIGH',
                    'description': f'成交量异常放大: {market_data.volume:,} (平均: {avg_volume:,.0f})',
                    'timestamp': market_data.timestamp.isoformat()
                })

        # 检测异常大的订单
        if market_data.bid_size > 50000 or market_data.ask_size > 50000:
            events.append({
                'event_type': 'LARGE_ORDER',
                'severity': 'MEDIUM',
                'description': f'检测到大额订单: 买盘 {market_data.bid_size:,}, 卖盘 {market_data.ask_size:,}',
                'timestamp': market_data.timestamp.isoformat()
            })

        # 检测异常价差
        if len(self.spread_history) > 0:
            avg_spread = sum(self.spread_history) / len(self.spread_history)
            current_spread = market_data.ask_price - market_data.bid_price
            if current_spread > avg_spread * 2:
                events.append({
                    'event_type': 'WIDE_SPREAD',
                    'severity': 'MEDIUM',
                    'description': f'价差异常扩大: {current_spread:.2f} (平均: {avg_spread:.2f})',
                    'timestamp': market_data.timestamp.isoformat()
                })

        return events

    def _update_statistics(self, liquidity_metrics: LiquidityMetrics):
        """更新统计信息"""
        self.liquidity_statistics['avg_spread'] = (
            self.liquidity_statistics['avg_spread'] * 0.9 +
            liquidity_metrics.time_weighted_spread * 0.1
        )

        self.liquidity_statistics['avg_depth'] = (
            self.liquidity_statistics['avg_depth'] * 0.9 +
            liquidity_metrics.market_depth * 0.1
        )

        self.liquidity_statistics['avg_impact'] = (
            self.liquidity_statistics['avg_impact'] * 0.9 +
            liquidity_metrics.price_impact * 0.1
        )

        self.liquidity_statistics['avg_score'] = (
            self.liquidity_statistics['avg_score'] * 0.9 +
            liquidity_metrics.liquidity_score * 0.1
        )

    def get_liquidity_summary(self) -> Dict[str, Any]:
        """获取流动性摘要"""
        if not self.market_data_history:
            return {}

        latest_data = self.market_data_history[-1]

        # 计算各项指标的平均值
        avg_volume = sum(self.volume_history) / len(self.volume_history) if self.volume_history else 0
        avg_price = sum(self.price_history) / len(self.price_history) if self.price_history else 0
        avg_spread = sum(self.spread_history) / len(self.spread_history) if self.spread_history else 0

        return {
            'symbol': latest_data.symbol,
            'timestamp': latest_data.timestamp.isoformat(),
            'current_price': latest_data.price,
            'current_volume': latest_data.volume,
            'average_volume': avg_volume,
            'average_price': avg_price,
            'average_spread': avg_spread,
            'liquidity_score': self.liquidity_statistics['avg_score'],
            'total_analyzed': self.total_analyzed,
            'volume_levels': dict(list(self.volume_profile.items())[:10])
        }

    def simulate_market_data(self, symbol: str) -> MarketDataPoint:
        """模拟市场数据"""
        base_prices = {
            "SHFE.cu2401": 68000,
            "DCE.i2403": 920,
            "CZCE.MA401": 2850,
            "SHFE.al2401": 18500,
            "ZCE.CF401": 12500
        }

        base_price = base_prices.get(symbol, 10000)

        # 模拟价格变化
        price_change = random.uniform(-0.01, 0.01)
        current_price = base_price * (1 + price_change)

        # 模拟买卖盘
        bid_price = current_price * (1 - random.uniform(0.0005, 0.002))
        ask_price = current_price * (1 + random.uniform(0.0005, 0.002))

        # 模拟成交量和挂单量
        volume = random.randint(1000, 50000)
        bid_size = random.randint(1000, 20000)
        ask_size = random.randint(1000, 20000)

        # 模拟成交次数
        trade_count = random.randint(10, 100)

        return MarketDataPoint(
            timestamp=datetime.now(),
            symbol=symbol,
            price=current_price,
            volume=volume,
            turnover=current_price * volume,
            bid_size=bid_size,
            ask_size=ask_size,
            bid_price=bid_price,
            ask_price=ask_price,
            trade_count=trade_count
        )