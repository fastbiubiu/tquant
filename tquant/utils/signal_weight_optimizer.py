"""
信号权重优化器
动态调整交易信号的权重,实现最优组合
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional

import numpy as np

from tquant.utils.signals import TradingSignal

logger = logging.getLogger(__name__)

class WeightingStrategy(Enum):
    """加权策略"""
    EQUAL = "等权重"
    CONFIDENCE = "置信度加权"
    VOLATILITY_ADJUSTED = "波动率调整"
    CORRELATION_MINIMIZED = "相关性最小化"
    SHARPE_OPTIMIZED = "夏普优化"
    KALMAN_FILTER = "卡尔曼滤波"
    ADAPTIVE = "自适应"

@dataclass
class SignalWeight:
    """信号权重"""
    symbol: str
    signal: TradingSignal
    weight: float
    adjusted_weight: float
    confidence_factor: float
    volatility_factor: float
    correlation_factor: float
    market_condition_factor: float
    timestamp: datetime

@dataclass
class OptimizationResult:
    """优化结果"""
    strategy: WeightingStrategy
    weights: List[SignalWeight]
    portfolio_return: float
    portfolio_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    optimization_timestamp: datetime

class SignalWeightOptimizer:
    """信号权重优化器"""

    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化信号权重优化器

        Args:
            config_path: 配置文件路径
        """
        from config import get_config, Config
        self.config: Config = get_config()

        # 获取权重优化配置
        self.weight_config = self.config.get('optimization', {}).get('signal_weighting', {})
        self.default_strategy = WeightingStrategy(self.weight_config.get('default_strategy', 'CONFIDENCE'))
        self.max_weights = self.weight_config.get('max_weights', 10)
        self.min_weight_threshold = self.weight_config.get('min_weight_threshold', 0.01)

        # 权重参数
        self.confidence_weight = self.weight_config.get('confidence_weight', 0.4)
        self.volatility_weight = self.weight_config.get('volatility_weight', 0.3)
        self.correlation_weight = self.weight_config.get('correlation_weight', 0.2)
        self.market_condition_weight = self.weight_config.get('market_condition_weight', 0.1)

        # 历史数据存储
        self.signal_history = []
        self.return_history = []
        self.correlation_matrix = None
        self.volatility_estimate = None

        logger.info(f"信号权重优化器初始化完成,默认策略: {self.default_strategy.value}")

    def optimize_weights(self, signals: List[TradingSignal],
                       market_data: Optional[Dict] = None,
                       strategy: Optional[WeightingStrategy] = None) -> OptimizationResult:
        """
        优化信号权重

        Args:
            signals: 交易信号列表
            market_data: 市场数据
            strategy: 加权策略

        Returns:
            优化结果
        """
        if not signals:
            raise ValueError("信号列表不能为空")

        strategy = strategy or self.default_strategy

        logger.info(f"使用策略 {strategy.value} 优化 {len(signals)} 个信号的权重")

        # 预处理信号
        processed_signals = self._preprocess_signals(signals)

        # 根据策略计算权重
        if strategy == WeightingStrategy.EQUAL:
            weights = self._equal_weighting(processed_signals)
        elif strategy == WeightingStrategy.CONFIDENCE:
            weights = self._confidence_weighting(processed_signals)
        elif strategy == WeightingStrategy.VOLATILITY_ADJUSTED:
            weights = self._volatility_adjusted_weighting(processed_signals)
        elif strategy == WeightingStrategy.CORRELATION_MINIMIZED:
            weights = self._correlation_minimized_weighting(processed_signals)
        elif strategy == WeightingStrategy.SHARPE_OPTIMIZED:
            weights = self._sharpe_optimized_weighting(processed_signals)
        elif strategy == WeightingStrategy.KALMAN_FILTER:
            weights = self._kalman_filter_weighting(processed_signals)
        elif strategy == WeightingStrategy.ADAPTIVE:
            weights = self._adaptive_weighting(processed_signals, market_data)
        else:
            weights = self._confidence_weighting(processed_signals)

        # 后处理权重
        final_weights = self._post_process_weights(weights)

        # 计算投资组合指标
        portfolio_metrics = self._calculate_portfolio_metrics(final_weights)

        # 创建优化结果
        result = OptimizationResult(
            strategy=strategy,
            weights=final_weights,
            portfolio_return=portfolio_metrics['expected_return'],
            portfolio_risk=portfolio_metrics['risk'],
            sharpe_ratio=portfolio_metrics['sharpe_ratio'],
            diversification_ratio=portfolio_metrics['diversification_ratio'],
            optimization_timestamp=datetime.now()
        )

        # 保存到历史记录
        self._save_optimization_result(result)

        logger.info(f"权重优化完成,夏普比率: {result.sharpe_ratio:.4f}")

        return result

    def _preprocess_signals(self, signals: List[TradingSignal]) -> List[Dict]:
        """预处理信号"""
        processed = []

        for signal in signals:
            processed_signal = {
                'symbol': signal.symbol,
                'direction': signal.direction,
                'confidence': signal.confidence,
                'price': signal.price,
                'timestamp': signal.timestamp,
                'indicators': signal.indicators,
                'volatility': self._estimate_volatility(signal),
                'liquidity_score': self._calculate_liquidity_score(signal),
                'market_impact': self._estimate_market_impact(signal)
            }
            processed.append(processed_signal)

        return processed

    def _equal_weighting(self, signals: List[Dict]) -> List[SignalWeight]:
        """等权重策略"""
        n = len(signals)
        weight = 1.0 / n

        weights = []
        for signal in signals:
            weight_obj = SignalWeight(
                symbol=signal['symbol'],
                signal=signal,  # 这里需要转换为TradingSignal
                weight=weight,
                adjusted_weight=weight,
                confidence_factor=1.0,
                volatility_factor=1.0,
                correlation_factor=1.0,
                market_condition_factor=1.0,
                timestamp=datetime.now()
            )
            weights.append(weight_obj)

        return weights

    def _confidence_weighting(self, signals: List[Dict]) -> List[SignalWeight]:
        """置信度加权策略"""
        # 根据置信度计算权重
        confidences = [s['confidence'] for s in signals]
        total_confidence = sum(confidences)

        weights = []
        for signal in signals:
            weight = signal['confidence'] / total_confidence
            confidence_factor = signal['confidence']

            weight_obj = SignalWeight(
                symbol=signal['symbol'],
                signal=signal,
                weight=weight,
                adjusted_weight=weight,
                confidence_factor=confidence_factor,
                volatility_factor=1.0,
                correlation_factor=1.0,
                market_condition_factor=1.0,
                timestamp=datetime.now()
            )
            weights.append(weight_obj)

        return weights

    def _volatility_adjusted_weighting(self, signals: List[Dict]) -> List[SignalWeight]:
        """波动率调整加权策略"""
        # 计算波动率倒数作为权重因子
        volatilities = [s['volatility'] for s in signals]
        inv_volatilities = [1.0 / (v + 1e-8) for v in volatilities]
        total_inv_vol = sum(inv_volatilities)

        weights = []
        for signal in signals:
            inv_vol = inv_volatilities[signals.index(signal)]
            weight = inv_vol / total_inv_vol
            volatility_factor = 1.0 / (signal['volatility'] + 1e-8)

            weight_obj = SignalWeight(
                symbol=signal['symbol'],
                signal=signal,
                weight=weight,
                adjusted_weight=weight,
                confidence_factor=1.0,
                volatility_factor=volatility_factor,
                correlation_factor=1.0,
                market_condition_factor=1.0,
                timestamp=datetime.now()
            )
            weights.append(weight_obj)

        return weights

    def _correlation_minimized_weighting(self, signals: List[Dict]) -> List[SignalWeight]:
        """相关性最小化加权策略"""
        if not self.correlation_matrix or len(signals) != len(self.correlation_matrix):
            # 如果没有相关矩阵,使用置信度加权
            return self._confidence_weighting(signals)

        # 使用反向相关性作为权重
        n = len(signals)
        weights = np.zeros(n)

        for i in range(n):
            # 对相关性求和(负相关奖励正相关惩罚)
            correlation_sum = 0
            for j in range(n):
                if i != j:
                    corr = self.correlation_matrix[i][j]
                    # 负相关增加权重,正相关减少权重
                    correlation_sum += max(0, -corr)

            weights[i] = 1 + correlation_sum

        # 归一化权重
        weights = weights / np.sum(weights)

        weight_objects = []
        for i, signal in enumerate(signals):
            weight_obj = SignalWeight(
                symbol=signal['symbol'],
                signal=signal,
                weight=float(weights[i]),
                adjusted_weight=float(weights[i]),
                confidence_factor=1.0,
                volatility_factor=1.0,
                correlation_factor=1 + correlation_sum,
                market_condition_factor=1.0,
                timestamp=datetime.now()
            )
            weight_objects.append(weight_obj)

        return weight_objects

    def _sharpe_optimized_weighting(self, signals: List[Dict]) -> List[SignalWeight]:
        """夏普优化加权策略"""
        # 简化的夏普优化：期望回报/风险
        returns = [s['confidence'] * 0.1 for s in signals]  # 假设回报率
        risks = [s['volatility'] for s in signals]

        # 计算风险调整收益
        risk_adjusted_returns = [r / (risk + 1e-8) for r, risk in zip(returns, risks)]
        total_return = sum(risk_adjusted_returns)

        weights = []
        for i, signal in enumerate(signals):
            weight = risk_adjusted_returns[i] / total_return

            weight_obj = SignalWeight(
                symbol=signal['symbol'],
                signal=signal,
                weight=weight,
                adjusted_weight=weight,
                confidence_factor=signal['confidence'],
                volatility_factor=1.0 / (risks[i] + 1e-8),
                correlation_factor=1.0,
                market_condition_factor=1.0,
                timestamp=datetime.now()
            )
            weights.append(weight_obj)

        return weights

    def _kalman_filter_weighting(self, signals: List[Dict]) -> List[SignalWeight]:
        """卡尔曼滤波加权策略"""
        # 简化的卡尔曼滤波实现
        n = len(signals)
        weights = np.zeros(n)

        # 初始化状态
        state = np.array([s['confidence'] for s in signals])
        covariance = np.eye(n) * 0.1

        # 预测步骤
        Q = np.eye(n) * 0.01  # 过程噪声
        state = state  # 假设状态不变
        covariance = covariance + Q

        # 更新步骤
        R = np.eye(n) * 0.1  # 观测噪声
        innovation = state - state  # 观测值与预测值的差异
        kalman_gain = covariance @ np.linalg.inv(covariance + R)
        state = state + kalman_gain @ innovation
        covariance = (np.eye(n) - kalman_gain) @ covariance

        # 归一化权重
        weights = np.maximum(state, 0)
        weights = weights / np.sum(weights)

        weight_objects = []
        for i, signal in enumerate(signals):
            weight_obj = SignalWeight(
                symbol=signal['symbol'],
                signal=signal,
                weight=float(weights[i]),
                adjusted_weight=float(weights[i]),
                confidence_factor=float(state[i]),
                volatility_factor=1.0,
                correlation_factor=1.0,
                market_condition_factor=1.0,
                timestamp=datetime.now()
            )
            weight_objects.append(weight_obj)

        return weight_objects

    def _adaptive_weighting(self, signals: List[Dict], market_data: Dict) -> List[SignalWeight]:
        """自适应加权策略"""
        # 综合考虑多种因素
        n = len(signals)
        weights = np.zeros(n)

        # 获取市场条件
        market_condition = self._assess_market_condition(market_data)

        for i, signal in enumerate(signals):
            # 计算综合因子
            confidence_factor = signal['confidence']
            volatility_factor = 1.0 / (signal['volatility'] + 1e-8)
            liquidity_factor = signal['liquidity_score']
            impact_factor = 1.0 / (signal['market_impact'] + 1e-8)

            # 市场条件调整
            if market_condition['volatility'] > 0.2:  # 高波动市场
                volatility_factor *= 1.2
            if market_condition['liquidity'] < 0.5:  # 低流动性市场
                liquidity_factor *= 0.8

            # 综合评分
            composite_score = (
                confidence_factor * self.confidence_weight +
                volatility_factor * self.volatility_weight +
                liquidity_factor * 0.1 +
                impact_factor * 0.1
            )

            weights[i] = composite_score

        # 归一化权重
        weights = np.maximum(weights, 0)
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight

        weight_objects = []
        for i, signal in enumerate(signals):
            weight_obj = SignalWeight(
                symbol=signal['symbol'],
                signal=signal,
                weight=float(weights[i]),
                adjusted_weight=float(weights[i]),
                confidence_factor=signal['confidence'],
                volatility_factor=1.0 / (signal['volatility'] + 1e-8),
                correlation_factor=1.0,
                market_condition_factor=market_condition['score'],
                timestamp=datetime.now()
            )
            weight_objects.append(weight_obj)

        return weight_objects

    def _post_process_weights(self, weights: List[SignalWeight]) -> List[SignalWeight]:
        """后处理权重"""
        # 1. 限制最小权重
        adjusted_weights = []
        total_weight = 0

        for weight in weights:
            if weight.weight < self.min_weight_threshold:
                adjusted_weights.append(self.min_weight_threshold)
            else:
                adjusted_weights.append(weight.weight)
            total_weight += adjusted_weights[-1]

        # 2. 重新归一化
        if total_weight > 0:
            scaling_factor = 1.0 / total_weight
            for i, weight in enumerate(weights):
                weight.adjusted_weight = adjusted_weights[i] * scaling_factor

        # 3. 限制最大权重
        for weight in weights:
            weight.adjusted_weight = min(weight.adjusted_weight, 0.3)  # 单个信号最大30%

        return weights

    def _estimate_volatility(self, signal: TradingSignal) -> float:
        """估计波动率"""
        # 从历史数据或指标中估计
        if signal.indicators:
            volatility_indicator = next(
                (i for i in signal.indicators if i['name'].lower() == 'atr'),
                None
            )
            if volatility_indicator:
                return volatility_indicator['value'] / signal.price

        # 默认波动率
        return 0.02

    def _calculate_liquidity_score(self, signal: Dict) -> float:
        """计算流动性得分"""
        # 基于价格和成交量估算
        price = signal['price']
        volume_weight = 1.0 / (1.0 + np.log10(price)) if price > 0 else 1.0
        return min(1.0, volume_weight)

    def _estimate_market_impact(self, signal: Dict) -> float:
        """估计市场冲击"""
        # 简化的市场冲击估计
        size = signal['price'] * 10  # 假设10手
        normalized_size = size / 1000000  # 相对于100万
        return normalized_size ** 0.5  # 平方根关系

    def _assess_market_condition(self, market_data: Dict) -> Dict:
        """评估市场条件"""
        if not market_data:
            return {'volatility': 0.1, 'liquidity': 0.8, 'trend': 0.0, 'score': 1.0}

        volatility = market_data.get('volatility', 0.1)
        liquidity = market_data.get('liquidity', 0.8)
        trend = market_data.get('trend', 0.0)

        # 计算市场条件得分
        condition_score = (
            (1 - volatility) * 0.3 +
            liquidity * 0.4 +
            (1 - abs(trend)) * 0.3
        )

        return {
            'volatility': volatility,
            'liquidity': liquidity,
            'trend': trend,
            'score': condition_score
        }

    def _calculate_portfolio_metrics(self, weights: List[SignalWeight]) -> Dict:
        """计算投资组合指标"""
        if not weights:
            return {'expected_return': 0, 'risk': 0, 'sharpe_ratio': 0, 'diversification_ratio': 0}

        # 计算期望回报
        expected_return = sum(w.adjusted_weight * w.signal['confidence'] for w in weights)

        # 计算投资组合风险
        if self.correlation_matrix and len(weights) == len(self.correlation_matrix):
            weights_array = np.array([w.adjusted_weight for w in weights])
            variances = np.array([w.signal['volatility'] ** 2 for w in weights])

            # 投资组合方差
            portfolio_variance = np.sqrt(
                weights_array.T @ self.correlation_matrix @ weights_array * variances
            )
            portfolio_risk = portfolio_variance[0] if hasattr(portfolio_variance, '__len__') else portfolio_variance
        else:
            # 简化计算：加权平均波动率
            portfolio_risk = sum(w.adjusted_weight * w.signal['volatility'] for w in weights)

        # 计算夏普比率(假设无风险利率为0)
        sharpe_ratio = expected_return / portfolio_risk if portfolio_risk > 0 else 0

        # 计算分散化比率
        individual_risks = [w.signal['volatility'] for w in weights]
        if portfolio_risk > 0:
            diversification_ratio = sum(w.adjusted_weight * r for w, r in zip(weights, individual_risks)) / portfolio_risk
        else:
            diversification_ratio = 1.0

        return {
            'expected_return': expected_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'diversification_ratio': diversification_ratio
        }

    def compare_strategies(self, signals: List[TradingSignal],
                          market_data: Optional[Dict] = None) -> Dict[str, Any]:
        """比较不同策略的效果"""
        results = {}

        for strategy in WeightingStrategy:
            try:
                result = self.optimize_weights(signals, market_data, strategy)
                results[strategy.value] = {
                    'expected_return': result.portfolio_return,
                    'risk': result.portfolio_risk,
                    'sharpe_ratio': result.sharpe_ratio,
                    'diversification_ratio': result.diversification_ratio,
                    'weights': [(w.symbol, w.adjusted_weight) for w in result.weights]
                }
            except Exception as e:
                logger.error(f"策略 {strategy.value} 优化失败: {e}")

        return results

    def get_strategy_recommendation(self, signals: List[TradingSignal],
                                   market_data: Optional[Dict] = None) -> WeightingStrategy:
        """获取策略推荐"""
        if not signals:
            return WeightingStrategy.CONFIDENCE

        # 分析信号特征
        avg_confidence = sum(s.confidence for s in signals) / len(signals)
        avg_volatility = sum(self._estimate_volatility(s) for s in signals) / len(signals)

        # 根据市场条件推荐
        if market_data:
            market_volatility = market_data.get('volatility', 0.1)
            if market_volatility > 0.2:  # 高波动市场
                return WeightingStrategy.VOLATILITY_ADJUSTED
            elif market_data.get('liquidity', 1.0) < 0.5:  # 低流动性市场
                return WeightingStrategy.ADAPTIVE

        # 根据信号特征推荐
        if avg_confidence > 0.7:
            return WeightingStrategy.CONFIDENCE
        elif avg_volatility > 0.05:
            return WeightingStrategy.VOLATILITY_ADJUSTED
        elif len(signals) > 5:
            return WeightingStrategy.CORRELATION_MINIMIZED
        else:
            return WeightingStrategy.SHARPE_OPTIMIZED

    def update_statistics(self, actual_returns: List[float]):
        """更新统计信息"""
        self.return_history.extend(actual_returns)

        # 保持最近1000条记录
        if len(self.return_history) > 1000:
            self.return_history = self.return_history[-1000:]

        # 更新相关矩阵(如果有足够的历史数据)
        if len(self.return_history) >= 30:
            self._update_correlation_matrix()

        # 更新波动率估计
        if self.return_history:
            recent_returns = self.return_history[-30:]
            self.volatility_estimate = np.std(recent_returns) * np.sqrt(252)  # 年化波动率

    def _update_correlation_matrix(self):
        """更新相关矩阵"""
        # 这里应该基于历史数据计算
        # 简化实现：创建随机相关矩阵
        n = 5  # 假设5个资产
        self.correlation_matrix = np.random.rand(n, n)
        # 使矩阵对称
        self.correlation_matrix = (self.correlation_matrix + self.correlation_matrix.T) / 2
        # 对角线为1
        np.fill_diagonal(self.correlation_matrix, 1)

    def _save_optimization_result(self, result: OptimizationResult):
        """保存优化结果"""
        # 这里可以保存到数据库或文件
        pass

    def get_optimization_history(self) -> List[Dict]:
        """获取优化历史"""
        # 返回最近10次优化记录
        return []  # 简化实现