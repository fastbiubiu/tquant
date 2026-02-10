"""
机构行为分析系统
分析机构行为,包括大单追踪、资金流向、持仓分析等
"""

import logging
from collections import deque, defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型"""
    MARKET_ORDER = "市价单"
    LIMIT_ORDER = "限价单"
    STOP_ORDER = "止损单"
    ICEBERG_ORDER = "冰山单"
    DARK_POOL = "暗池交易"
    ALGORITHM_ORDER = "算法交易"

class OrderSide(Enum):
    """订单方向"""
    BUY = "买入"
    SELL = "卖出"
    SHORT = "卖空"
    COVER = "平仓"

class InstitutionType(Enum):
    """机构类型"""
    BANK = "银行"
    SECURITY_FIRM = "券商"
    FUND = "基金"
    INSURANCE = "保险"
    QFII = "QFII"
    HEDGE_FUND = "对冲基金"
    INVESTMENT_TRUST = "投资信托"
    BROKER = "经纪商"

class TradeDirection(Enum):
    """交易方向"""
    BULLISH = "看涨"
    BEARISH = "看跌"
    NEUTRAL = "中性"
    MIXED = "混合"

@dataclass
class LargeOrder:
    """大单数据"""
    timestamp: datetime
    symbol: str
    order_id: str
    order_type: OrderType
    side: OrderSide
    price: float
    quantity: int
    value: float
    institution: str
    institution_type: InstitutionType
    algorithmic: bool
    iceberg: bool
    visible_quantity: int
    hidden_quantity: int

@dataclass
class TradeFlow:
    """资金流向"""
    timestamp: datetime
    symbol: str
    buy_volume: int
    sell_volume: int
    buy_value: float
    sell_value: float
    net_flow: float
    flow_ratio: float
    institutional_ratio: float
    retail_ratio: float
    block_trades: List[Dict[str, Any]]

@dataclass
class PositionChange:
    """持仓变化"""
    timestamp: datetime
    symbol: str
    institution: str
    position_change: int
    position_ratio: float
    cumulative_position: int
    market_value: float
    position_rank: int
    concentration_ratio: float

@dataclass
class InstitutionalBehavior:
    """机构行为"""
    timestamp: datetime
    symbol: str
    trade_direction: TradeDirection
    large_orders_count: int
    total_large_value: float
    net_institutional_flow: float
    institutional_activity: float
    order_imbalance: float
    concentration_ratio: float
    market_impact: float
    smart_money_flow: float
    institutional_sentiment: float
    key_institutions: List[str]
    trading_patterns: Dict[str, int]
    risk_signals: List[Dict[str, Any]]

class InstitutionalAnalyzer:
    """机构行为分析器"""

    def __init__(self, window_size: int = 100):
        """
        初始化机构行为分析器

        Args:
            window_size: 分析窗口大小
        """
        self.window_size = window_size
        self.large_orders = deque(maxlen=window_size)
        self.trade_flows = deque(maxlen=window_size)
        self.position_changes = deque(maxlen=window_size)

        # 机构列表
        self.institutions = [
            ('工商银行', InstitutionType.BANK),
            ('建设银行', InstitutionType.BANK),
            ('中国平安', InstitutionType.INSURANCE),
            ('易方达基金', InstitutionType.FUND),
            ('华夏基金', InstitutionType.FUND),
            ('南方基金', InstitutionType.FUND),
            ('高盛证券', InstitutionType.SECURITY_FIRM),
            ('摩根士丹利', InstitutionType.SECURITY_FIRM),
            ('瑞银集团', InstitutionType.BANK),
            ('淡马锡', InstitutionType.INVESTMENT_TRUST)
        ]

        # 大单阈值
        self.large_order_thresholds = {
            'SHFE.cu2401': 50000,  # 铜
            'DCE.i2403': 10000,   # 豆粕
            'CZCE.MA401': 5000,    # 甲醇
            'SHFE.al2401': 20000,  # 铝
            'ZCE.CF401': 3000      # 棉花
        }

        # 统计数据
        self.total_analyzed = 0
        self.behavior_statistics = {
            'avg_large_orders': 0,
            'avg_institutional_flow': 0,
            'avg_market_impact': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'neutral_count': 0
        }

    def analyze_large_orders(self, large_order: LargeOrder) -> Dict[str, Any]:
        """
        分析大单行为

        Args:
            large_order: 大单数据

        Returns:
            分析结果
        """
        try:
            # 保存到历史
            self.large_orders.append(large_order)

            # 分析订单特征
            order_analysis = self._analyze_order_features(large_order)

            # 分析机构行为
            institutional_analysis = self._analyze_institutional_behavior(large_order)

            # 计算市场冲击
            market_impact = self._calculate_market_impact(large_order)

            # 检测异常订单
            anomalies = self._detect_order_anomalies(large_order)

            return {
                'timestamp': large_order.timestamp.isoformat(),
                'symbol': large_order.symbol,
                'order_id': large_order.order_id,
                'order_type': large_order.order_type.value,
                'side': large_order.side.value,
                'price': large_order.price,
                'quantity': large_order.quantity,
                'value': large_order.value,
                'institution': large_order.institution,
                'institution_type': large_order.institution_type.value,
                'algorithmic': large_order.algorithmic,
                'iceberg': large_order.iceberg,
                'analysis': order_analysis,
                'institutional_analysis': institutional_analysis,
                'market_impact': market_impact,
                'anomalies': anomalies
            }

        except Exception as e:
            logger.error(f"大单分析失败: {e}")
            return {
                'timestamp': large_order.timestamp.isoformat(),
                'symbol': large_order.symbol,
                'error': str(e)
            }

    def analyze_trade_flow(self, trade_flow: TradeFlow) -> Dict[str, Any]:
        """
        分析资金流向

        Args:
            trade_flow: 资金流向数据

        Returns:
            分析结果
        """
        try:
            # 保存到历史
            self.trade_flows.append(trade_flow)

            # 分析资金流向特征
            flow_analysis = self._analyze_flow_characteristics(trade_flow)

            # 计算机构参与度
            institutional_participation = self._calculate_institutional_participation(trade_flow)

            # 分析净流入
            net_flow_analysis = self._analyze_net_flow(trade_flow)

            # 检测大额交易
            block_trades_analysis = self._analyze_block_trades(trade_flow)

            return {
                'timestamp': trade_flow.timestamp.isoformat(),
                'symbol': trade_flow.symbol,
                'buy_volume': trade_flow.buy_volume,
                'sell_volume': trade_flow.sell_volume,
                'buy_value': trade_flow.buy_value,
                'sell_value': trade_flow.sell_value,
                'net_flow': trade_flow.net_flow,
                'flow_ratio': trade_flow.flow_ratio,
                'institutional_ratio': trade_flow.institutional_ratio,
                'retail_ratio': trade_flow.retail_ratio,
                'flow_analysis': flow_analysis,
                'institutional_participation': institutional_participation,
                'net_flow_analysis': net_flow_analysis,
                'block_trades': block_trades_analysis
            }

        except Exception as e:
            logger.error(f"资金流向分析失败: {e}")
            return {
                'timestamp': trade_flow.timestamp.isoformat(),
                'symbol': trade_flow.symbol,
                'error': str(e)
            }

    def analyze_position_changes(self, position_change: PositionChange) -> Dict[str, Any]:
        """
        分析持仓变化

        Args:
            position_change: 持仓变化数据

        Returns:
            分析结果
        """
        try:
            # 保存到历史
            self.position_changes.append(position_change)

            # 分析持仓特征
            position_analysis = self._analyze_position_features(position_change)

            # 计算集中度
            concentration_analysis = self._analyze_concentration(position_change)

            # 分析持仓趋势
            trend_analysis = self._analyze_position_trend(position_change)

            return {
                'timestamp': position_change.timestamp.isoformat(),
                'symbol': position_change.symbol,
                'institution': position_change.institution,
                'position_change': position_change.position_change,
                'position_ratio': position_change.position_ratio,
                'cumulative_position': position_change.cumulative_position,
                'market_value': position_change.market_value,
                'position_rank': position_change.position_rank,
                'concentration_ratio': position_change.concentration_ratio,
                'position_analysis': position_analysis,
                'concentration_analysis': concentration_analysis,
                'trend_analysis': trend_analysis
            }

        except Exception as e:
            logger.error(f"持仓变化分析失败: {e}")
            return {
                'timestamp': position_change.timestamp.isoformat(),
                'symbol': position_change.symbol,
                'institution': position_change.institution,
                'error': str(e)
            }

    def analyze_institutional_behavior(self, timestamp: datetime, symbol: str) -> InstitutionalBehavior:
        """
        综合分析机构行为

        Args:
            timestamp: 时间戳
            symbol: 交易品种

        Returns:
            机构行为分析结果
        """
        try:
            # 计算大单统计
            large_orders_count = len(self.large_orders)
            total_large_value = sum(order.value for order in self.large_orders)

            # 计算资金流向
            net_institutional_flow = self._calculate_net_institutional_flow()
            institutional_activity = self._calculate_institutional_activity()

            # 计算订单不平衡
            order_imbalance = self._calculate_order_imbalance()

            # 计算集中度
            concentration_ratio = self._calculate_concentration_ratio()

            # 计算市场冲击
            market_impact = self._calculate_aggregated_market_impact()

            # 计算聪明钱流向
            smart_money_flow = self._calculate_smart_money_flow()

            # 计算机构情绪
            institutional_sentiment = self._calculate_institutional_sentiment()

            # 识别关键机构
            key_institutions = self._identify_key_institutions()

            # 分析交易模式
            trading_patterns = self._analyze_trading_patterns()

            # 检测风险信号
            risk_signals = self._detect_risk_signals()

            # 确定交易方向
            trade_direction = self._determine_trade_direction()

            return InstitutionalBehavior(
                timestamp=timestamp,
                symbol=symbol,
                trade_direction=trade_direction,
                large_orders_count=large_orders_count,
                total_large_value=total_large_value,
                net_institutional_flow=net_institutional_flow,
                institutional_activity=institutional_activity,
                order_imbalance=order_imbalance,
                concentration_ratio=concentration_ratio,
                market_impact=market_impact,
                smart_money_flow=smart_money_flow,
                institutional_sentiment=institutional_sentiment,
                key_institutions=key_institutions,
                trading_patterns=trading_patterns,
                risk_signals=risk_signals
            )

        except Exception as e:
            logger.error(f"机构行为分析失败: {e}")
            return InstitutionalBehavior(
                timestamp=timestamp,
                symbol=symbol,
                trade_direction=TradeDirection.NEUTRAL,
                large_orders_count=0,
                total_large_value=0,
                net_institutional_flow=0,
                institutional_activity=0,
                order_imbalance=0,
                concentration_ratio=0,
                market_impact=0,
                smart_money_flow=0,
                institutional_sentiment=0,
                key_institutions=[],
                trading_patterns={},
                risk_signals=[]
            )

    def _analyze_order_features(self, order: LargeOrder) -> Dict[str, Any]:
        """分析订单特征"""
        features = {}

        # 订单大小评估
        threshold = self.large_order_thresholds.get(order.symbol, 10000)
        if order.quantity > threshold * 2:
            features['size_category'] = 'extra_large'
        elif order.quantity > threshold:
            features['size_category'] = 'large'
        else:
            features['size_category'] = 'medium'

        # 订单类型分析
        if order.iceberg:
            features['strategy'] = 'stealth_trading'
        elif order.algorithmic:
            features['strategy'] = 'algorithmic_trading'
        else:
            features['strategy'] = 'manual_trading'

        # 订单方向分析
        if order.side == OrderSide.BUY:
            features['direction'] = 'bullish'
        elif order.side == OrderSide.SELL:
            features['direction'] = 'bearish'

        return features

    def _analyze_institutional_behavior(self, order: LargeOrder) -> Dict[str, Any]:
        """分析机构行为"""
        analysis = {}

        # 机构类型分析
        if order.institution_type == InstitutionType.FUND:
            analysis['type'] = 'institutional_investor'
        elif order.institution_type == InstitutionType.BANK:
            analysis['type'] = 'financial_institution'
        elif order.institution_type == InstitutionType.HEDGE_FUND:
            analysis['type'] = 'hedge_fund'
        else:
            analysis['type'] = 'other'

        # 机构规模评估
        if order.value > 10000000:  # 1000万以上
            analysis['scale'] = 'large'
        elif order.value > 1000000:  # 100万以上
            analysis['scale'] = 'medium'
        else:
            analysis['scale'] = 'small'

        return analysis

    def _calculate_market_impact(self, order: LargeOrder) -> float:
        """计算市场冲击"""
        # 基础冲击计算
        turnover_ratio = order.value / (order.price * order.quantity) if order.quantity > 0 else 0

        # 根据订单类型调整
        impact_multiplier = 1.0
        if order.iceberg:
            impact_multiplier = 0.5  # 冰山单冲击较小
        elif order.algorithmic:
            impact_multiplier = 0.7  # 算法单冲击中等
        else:
            impact_multiplier = 1.0  # 市价单冲击较大

        # 计算冲击比例
        market_impact = turnover_ratio * impact_multiplier * 100  # 转换为百分比

        return min(market_impact, 10)  # 限制最大冲击为10%

    def _detect_order_anomalies(self, order: LargeOrder) -> List[Dict[str, Any]]:
        """检测订单异常"""
        anomalies = []

        # 检测异常大单
        threshold = self.large_order_thresholds.get(order.symbol, 10000)
        if order.quantity > threshold * 5:
            anomalies.append({
                'type': 'extra_large_order',
                'severity': 'HIGH',
                'description': f'检测到异常大单: {order.quantity:,} 手'
            })

        # 检测异常时间
        current_time = order.timestamp.time()
        if current_time.hour < 9 or current_time.hour > 15:
            anomalies.append({
                'type': 'off_hours_trading',
                'severity': 'MEDIUM',
                'description': f'非交易时间大单: {current_time}'
            })

        # 检测异常价格
        if len(self.large_orders) > 0:
            avg_price = sum(o.price for o in self.large_orders) / len(self.large_orders)
            price_deviation = abs(order.price - avg_price) / avg_price
            if price_deviation > 0.05:  # 价格偏离超过5%
                anomalies.append({
                    'type': 'price_anomaly',
                    'severity': 'MEDIUM',
                    'description': f'价格异常偏离: {price_deviation:.2%}'
                })

        return anomalies

    def _analyze_flow_characteristics(self, flow: TradeFlow) -> Dict[str, Any]:
        """分析资金流向特征"""
        analysis = {}

        # 成交量对比
        volume_ratio = flow.buy_volume / (flow.sell_volume + 1)
        if volume_ratio > 1.5:
            analysis['volume_trend'] = 'buy_dominant'
        elif volume_ratio < 0.67:
            analysis['volume_trend'] = 'sell_dominant'
        else:
            analysis['volume_trend'] = 'balanced'

        # 金额对比
        value_ratio = flow.buy_value / (flow.sell_value + 1)
        if value_ratio > 1.5:
            analysis['value_trend'] = 'buy_dominant'
        elif value_ratio < 0.67:
            analysis['value_trend'] = 'sell_dominant'
        else:
            analysis['value_trend'] = 'balanced'

        return analysis

    def _calculate_institutional_participation(self, flow: TradeFlow) -> Dict[str, Any]:
        """计算机构参与度"""
        analysis = {}

        # 机构参与比例
        institutional_ratio = flow.institutional_ratio

        if institutional_ratio > 0.8:
            participation_level = 'very_high'
        elif institutional_ratio > 0.6:
            participation_level = 'high'
        elif institutional_ratio > 0.4:
            participation_level = 'medium'
        else:
            participation_level = 'low'

        analysis['participation_level'] = participation_level
        analysis['ratio'] = institutional_ratio

        # 机构活跃度评估
        total_volume = flow.buy_volume + flow.sell_volume
        if total_volume > 100000:
            activity_level = 'very_active'
        elif total_volume > 50000:
            activity_level = 'active'
        elif total_volume > 10000:
            activity_level = 'moderate'
        else:
            activity_level = 'low'

        analysis['activity_level'] = activity_level

        return analysis

    def _analyze_net_flow(self, flow: TradeFlow) -> Dict[str, Any]:
        """分析净流入"""
        analysis = {}

        # 净流入方向
        if flow.net_flow > 0:
            direction = 'inflow'
        elif flow.net_flow < 0:
            direction = 'outflow'
        else:
            direction = 'neutral'

        analysis['direction'] = direction
        analysis['amount'] = flow.net_flow

        # 净流入强度
        if abs(flow.net_flow) > flow.buy_value * 0.3:
            intensity = 'strong'
        elif abs(flow.net_flow) > flow.buy_value * 0.1:
            intensity = 'moderate'
        else:
            intensity = 'weak'

        analysis['intensity'] = intensity

        return analysis

    def _analyze_block_trades(self, flow: TradeFlow) -> List[Dict[str, Any]]:
        """分析大额交易"""
        block_trades = []

        for block_trade in flow.block_trades:
            # 分析单笔大额交易
            if block_trade['quantity'] > 1000:  # 超过1000手算大额交易
                block_trades.append({
                    'time': block_trade.get('time', ''),
                    'side': block_trade.get('side', ''),
                    'quantity': block_trade['quantity'],
                    'value': block_trade['value'],
                    'price': block_trade.get('price', 0),
                    'institution': block_trade.get('institution', '')
                })

        return block_trades[:10]  # 返回前10个大额交易

    def _analyze_position_features(self, position: PositionChange) -> Dict[str, Any]:
        """分析持仓特征"""
        analysis = {}

        # 持仓变化方向
        if position.position_change > 0:
            direction = 'increasing'
        elif position.position_change < 0:
            direction = 'decreasing'
        else:
            direction = 'stable'

        analysis['direction'] = direction
        analysis['magnitude'] = abs(position.position_change)

        # 持仓规模评估
        if position.market_value > 100000000:  # 1亿以上
            scale = 'very_large'
        elif position.market_value > 10000000:  # 1000万以上
            scale = 'large'
        elif position.market_value > 1000000:  # 100万以上
            scale = 'medium'
        else:
            scale = 'small'

        analysis['scale'] = scale

        return analysis

    def _analyze_concentration(self, position: PositionChange) -> Dict[str, Any]:
        """分析集中度"""
        analysis = {}

        # 集中度评估
        if position.concentration_ratio > 0.5:
            concentration_level = 'very_high'
        elif position.concentration_ratio > 0.3:
            concentration_level = 'high'
        elif position.concentration_ratio > 0.1:
            concentration_level = 'moderate'
        else:
            concentration_level = 'low'

        analysis['concentration_level'] = concentration_level
        analysis['ratio'] = position.concentration_ratio

        # 风险评估
        if concentration_level == 'very_high':
            risk_level = 'high'
        elif concentration_level == 'high':
            risk_level = 'medium'
        else:
            risk_level = 'low'

        analysis['risk_level'] = risk_level

        return analysis

    def _analyze_position_trend(self, position: PositionChange) -> Dict[str, Any]:
        """分析持仓趋势"""
        analysis = {}

        # 排名变化
        if position.position_rank <= 10:
            ranking = 'top_10'
        elif position.position_rank <= 50:
            ranking = 'top_50'
        elif position.position_rank <= 100:
            ranking = 'top_100'
        else:
            ranking = 'others'

        analysis['ranking'] = ranking
        analysis['rank'] = position.position_rank

        # 累计持仓评估
        if position.cumulative_position > position.position_change * 10:
            trend = 'accumulating'
        elif position.cumulative_position < position.position_change * 0.1:
            trend = 'reducing'
        else:
            trend = 'stable'

        analysis['trend'] = trend

        return analysis

    def _calculate_net_institutional_flow(self) -> float:
        """计算净机构资金流向"""
        if not self.trade_flows:
            return 0

        net_flow = sum(flow.net_flow for flow in self.trade_flows)
        return net_flow / len(self.trade_flows)

    def _calculate_institutional_activity(self) -> float:
        """计算机构活跃度"""
        if not self.trade_flows:
            return 0

        total_volume = sum(flow.buy_volume + flow.sell_volume for flow in self.trade_flows)
        return total_volume / len(self.trade_flows)

    def _calculate_order_imbalance(self) -> float:
        """计算订单不平衡"""
        if not self.large_orders:
            return 0

        buy_orders = sum(1 for order in self.large_orders if order.side == OrderSide.BUY)
        sell_orders = sum(1 for order in self.large_orders if order.side == OrderSide.SELL)
        total_orders = len(self.large_orders)

        if total_orders == 0:
            return 0

        return (buy_orders - sell_orders) / total_orders

    def _calculate_concentration_ratio(self) -> float:
        """计算集中度"""
        if not self.position_changes:
            return 0

        # 计算最大持仓占比
        positions = [p.position_ratio for p in self.position_changes]
        if sum(positions) > 0:
            max_ratio = max(positions)
            return max_ratio
        return 0

    def _calculate_aggregated_market_impact(self) -> float:
        """计算整体市场冲击"""
        if not self.large_orders:
            return 0

        total_impact = sum(order.value * 0.001 for order in self.large_orders)  # 基础冲击0.1%
        return min(total_impact / len(self.large_orders), 5)  # 限制最大冲击为5%

    def _calculate_smart_money_flow(self) -> float:
        """计算聪明钱流向"""
        # 基于机构大单的方向和规模
        if not self.large_orders:
            return 0

        smart_money_score = 0
        for order in self.large_orders:
            if order.institution_type in [InstitutionType.FUND, InstitutionType.BANK]:
                weight = 1.5 if order.institution_type == InstitutionType.FUND else 1.2
                if order.side == OrderSide.BUY:
                    smart_money_score += order.value * weight
                else:
                    smart_money_score -= order.value * weight

        # 标准化到-1到1之间
        total_value = sum(order.value for order in self.large_orders)
        if total_value > 0:
            return smart_money_score / total_value
        return 0

    def _calculate_institutional_sentiment(self) -> float:
        """计算机构情绪"""
        # 基于大单方向和资金流向
        order_score = self._calculate_order_imbalance()
        flow_score = self._calculate_net_institutional_flow() / 1000000  # 标准化

        # 综合评分
        sentiment = (order_score * 0.6 + flow_score * 0.4)
        return max(-1, min(1, sentiment))

    def _identify_key_institutions(self) -> List[str]:
        """识别关键机构"""
        institution_activity = defaultdict(int)

        for order in self.large_orders:
            institution_activity[order.institution] += order.value

        # 返回最活跃的5个机构
        sorted_institutions = sorted(institution_activity.items(),
                                   key=lambda x: x[1], reverse=True)
        return [inst[0] for inst in sorted_institutions[:5]]

    def _analyze_trading_patterns(self) -> Dict[str, int]:
        """分析交易模式"""
        patterns = defaultdict(int)

        for order in self.large_orders:
            if order.iceberg:
                patterns['iceberg_orders'] += 1
            elif order.algorithmic:
                patterns['algorithmic_orders'] += 1
            else:
                patterns['manual_orders'] += 1

            patterns[order.order_type.value] += 1
            patterns[order.side.value] += 1

        return dict(patterns)

    def _detect_risk_signals(self) -> List[Dict[str, Any]]:
        """检测风险信号"""
        risk_signals = []

        # 检测过度集中
        if self._calculate_concentration_ratio() > 0.5:
            risk_signals.append({
                'type': 'high_concentration',
                'severity': 'HIGH',
                'description': '机构持仓过度集中'
            })

        # 检测异常大单
        large_orders_count = sum(1 for order in self.large_orders
                               if order.quantity > self.large_order_thresholds.get(order.symbol, 10000) * 5)
        if large_orders_count > 5:
            risk_signals.append({
                'type': 'excessive_large_orders',
                'severity': 'MEDIUM',
                'description': f'检测到{large_orders_count}个异常大单'
            })

        # 检测单边市场
        if abs(self._calculate_order_imbalance()) > 0.8:
            risk_signals.append({
                'type': 'one_sided_market',
                'severity': 'HIGH',
                'description': '市场单边倾向明显'
            })

        return risk_signals

    def _determine_trade_direction(self) -> TradeDirection:
        """确定交易方向"""
        sentiment = self._calculate_institutional_sentiment()
        imbalance = self._calculate_order_imbalance()

        if sentiment > 0.5 and imbalance > 0.5:
            return TradeDirection.BULLISH
        elif sentiment < -0.5 and imbalance < -0.5:
            return TradeDirection.BEARISH
        elif abs(sentiment) < 0.2 and abs(imbalance) < 0.2:
            return TradeDirection.NEUTRAL
        else:
            return TradeDirection.MIXED

    def generate_trading_signals(self, institutional_behavior: InstitutionalBehavior) -> List[Dict[str, Any]]:
        """生成交易信号"""
        signals = []

        # 基于交易方向的信号
        if institutional_behavior.trade_direction == TradeDirection.BULLISH:
            signals.append({
                'signal_type': 'INSTITUTIONAL_BULLISH',
                'strength': 'STRONG',
                'message': '机构看涨意愿强烈',
                'recommendation': '关注做多机会',
                'smart_money_flow': institutional_behavior.smart_money_flow,
                'institutional_sentiment': institutional_behavior.institutional_sentiment
            })
        elif institutional_behavior.trade_direction == TradeDirection.BEARISH:
            signals.append({
                'signal_type': 'INSTITUTIONAL_BEARISH',
                'strength': 'STRONG',
                'message': '机构看跌意愿强烈',
                'recommendation': '谨慎做多或考虑做空',
                'smart_money_flow': institutional_behavior.smart_money_flow,
                'institutional_sentiment': institutional_behavior.institutional_sentiment
            })

        # 基于聪明钱流向的信号
        if institutional_behavior.smart_money_flow > 0.5:
            signals.append({
                'signal_type': 'SMART_MONEY_INFLOW',
                'strength': 'MEDIUM',
                'message': '聪明钱流入',
                'recommendation': '跟随主力资金',
                'smart_money_flow': institutional_behavior.smart_money_flow
            })
        elif institutional_behavior.smart_money_flow < -0.5:
            signals.append({
                'signal_type': 'SMART_MONEY_OUTFLOW',
                'strength': 'MEDIUM',
                'message': '聪明钱流出',
                'recommendation': '警惕资金撤离',
                'smart_money_flow': institutional_behavior.smart_money_flow
            })

        # 基于市场冲击的信号
        if institutional_behavior.market_impact > 3:
            signals.append({
                'signal_type': 'HIGH_MARKET_IMPACT',
                'strength': 'MEDIUM',
                'message': '市场冲击较大',
                'recommendation': '注意滑点风险',
                'market_impact': institutional_behavior.market_impact
            })

        # 基于集中度的信号
        if institutional_behavior.concentration_ratio > 0.5:
            signals.append({
                'signal_type': 'HIGH_CONCENTRATION',
                'strength': 'HIGH',
                'message': '机构持仓过度集中',
                'recommendation': '注意集中度风险',
                'concentration_ratio': institutional_behavior.concentration_ratio
            })

        # 基于风险信号的信号
        for risk in institutional_behavior.risk_signals:
            signals.append({
                'signal_type': 'RISK_SIGNAL',
                'strength': risk['severity'].lower(),
                'message': risk['description'],
                'recommendation': '注意风险控制',
                'risk_type': risk['type']
            })

        return signals

    def get_institutional_summary(self) -> Dict[str, Any]:
        """获取机构行为摘要"""
        if not self.large_orders:
            return {}

        latest_behavior = self.institutional_behavior if hasattr(self, 'institutional_behavior') else None

        return {
            'total_large_orders': len(self.large_orders),
            'total_large_value': sum(order.value for order in self.large_orders),
            'average_order_size': sum(order.quantity for order in self.large_orders) / len(self.large_orders),
            'buy_orders': sum(1 for order in self.large_orders if order.side == OrderSide.BUY),
            'sell_orders': sum(1 for order in self.large_orders if order.side == OrderSide.SELL),
            'institutional_orders': sum(1 for order in self.large_orders
                                        if order.institution_type in [InstitutionType.FUND, InstitutionType.BANK]),
            'algorithmic_orders': sum(1 for order in self.large_orders if order.algorithmic),
            'iceberg_orders': sum(1 for order in self.large_orders if order.iceberg),
            'key_institutions': self._identify_key_institutions() if latest_behavior else [],
            'total_analyzed': self.total_analyzed
        }

    def simulate_large_order(self) -> LargeOrder:
        """模拟大单数据"""
        import random

        symbols = ["SHFE.cu2401", "DCE.i2403", "CZCE.MA401", "SHFE.al2401", "ZCE.CF401"]
        symbol = random.choice(symbols)

        # 基准价格
        base_prices = {
            "SHFE.cu2401": 68000,
            "DCE.i2403": 920,
            "CZCE.MA401": 2850,
            "SHFE.al2401": 18500,
            "ZCE.CF401": 12500
        }

        base_price = base_prices.get(symbol, 10000)

        # 模拟价格变化
        price_change = random.uniform(-0.002, 0.002)
        price = base_price * (1 + price_change)

        # 模拟数量
        threshold = self.large_order_thresholds.get(symbol, 10000)
        quantity = random.randint(threshold, threshold * 5)

        # 模拟订单类型
        order_types = [
            (OrderType.MARKET_ORDER, 0.3),
            (OrderType.LIMIT_ORDER, 0.4),
            (OrderType.STOP_ORDER, 0.1),
            (OrderType.ICEBERG_ORDER, 0.1),
            (OrderType.DARK_POOL, 0.05),
            (OrderType.ALGORITHM_ORDER, 0.05)
        ]

        order_type = random.choices([ot[0] for ot in order_types],
                                  weights=[ot[1] for ot in order_types])[0]

        # 随机方向
        side = random.choice([OrderSide.BUY, OrderSide.SELL])

        # 选择机构
        institution, institution_type = random.choice(self.institutions)

        # 计算价值
        value = price * quantity

        # 冰山单特征
        iceberg = order_type == OrderType.ICEBERG_ORDER
        visible_quantity = quantity // 10 if iceberg else quantity
        hidden_quantity = quantity - visible_quantity

        # 算法订单特征
        algorithmic = order_type in [OrderType.ALGORITHM_ORDER, OrderType.DARK_POOL]

        return LargeOrder(
            timestamp=datetime.now(),
            symbol=symbol,
            order_id=f"ORD{random.randint(100000, 999999)}",
            order_type=order_type,
            side=side,
            price=price,
            quantity=quantity,
            value=value,
            institution=institution,
            institution_type=institution_type,
            algorithmic=algorithmic,
            iceberg=iceberg,
            visible_quantity=visible_quantity,
            hidden_quantity=hidden_quantity
        )