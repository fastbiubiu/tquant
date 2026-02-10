"""
信号验证器
验证交易信号的准确性和可靠性
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from tquant.config import get_config, Config

from tquant.utils.signals import TradingSignal

logger = logging.getLogger(__name__)

class SignalStatus(Enum):
    """信号状态"""
    VALID = "有效"
    INVALID = "无效"
    UNCERTAIN = "不确定"
    CONFIRMED = "确认"
    REJECTED = "拒绝"
    PARTIAL_CONFIRM = "部分确认"

class RiskLevel(Enum):
    """风险等级"""
    LOW = "低"
    MEDIUM = "中"
    HIGH = "高"

@dataclass
class ValidationRule:
    """验证规则"""
    name: str
    enabled: bool
    min_threshold: float
    max_threshold: float
    weight: float
    description: str

@dataclass
class ValidationResult:
    """验证结果"""
    signal_id: str
    original_signal: TradingSignal
    status: SignalStatus
    confidence_score: float
    risk_level: RiskLevel
    validation_details: Dict[str, Any]
    timestamp: datetime
    suggested_action: str

class SignalValidator:
    """信号验证器"""

    def __init__(self):
        """
        初始化信号验证器

        Args:
            config_path: 配置文件路径
        """
        self.config: Config = get_config()

        # 获取验证配置
        self.validation_config = self.config.get('validation', {})
        self.enable_advanced_validation = self.validation_config.get('advanced', True)
        self.confirmation_threshold = self.validation_config.get('confirmation_threshold', 0.7)

        # 初始化验证规则
        self.rules = self._init_validation_rules()

        # 信号历史记录
        self.signal_history = {}
        self.performance_stats = {}

        logger.info("信号验证器初始化完成")

    def _init_validation_rules(self) -> List[ValidationRule]:
        """初始化验证规则"""
        default_rules = [
            # 置信度规则
            ValidationRule(
                name="confidence_threshold",
                enabled=True,
                min_threshold=0.3,
                max_threshold=1.0,
                weight=0.3,
                description="信号置信度必须在合理范围内"
            ),
            # 价格合理性检查
            ValidationRule(
                name="price合理性",
                enabled=True,
                min_threshold=0.01,
                max_threshold=1e6,
                weight=0.2,
                description="价格必须在合理范围内"
            ),
            # 成交量验证
            ValidationRule(
                name="volume验证",
                enabled=True,
                min_threshold=0,
                max_threshold=1e9,
                weight=0.15,
                description="成交量必须为正数"
            ),
            # 市场趋势一致性
            ValidationRule(
                name="trend_consistency",
                enabled=True,
                min_threshold=0.0,
                max_threshold=1.0,
                weight=0.2,
                description="信号必须与市场趋势基本一致"
            ),
            # 技术指标合理性
            ValidationRule(
                name="indicator_rationality",
                enabled=True,
                min_threshold=0.0,
                max_threshold=1.0,
                weight=0.15,
                description="技术指标必须在合理范围内"
            )
        ]

        # 从配置加载规则
        config_rules = self.config.get('validation', {}).get('rules', [])
        if config_rules:
            rules = []
            for rule_config in config_rules:
                rule = ValidationRule(**rule_config)
                rules.append(rule)
            return rules

        return default_rules

    def validate_signal(self, signal: TradingSignal,
                       market_data: Optional[Dict] = None,
                       historical_signals: Optional[List[TradingSignal]] = None) -> ValidationResult:
        """
        验证单个交易信号

        Args:
            signal: 要验证的交易信号
            market_data: 市场数据
            historical_signals: 历史信号

        Returns:
            验证结果
        """
        signal_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"

        # 初始化验证结果
        validation_details = {
            'rule_results': {},
            'risk_factors': [],
            'strength_factors': [],
            'market_context': market_data or {}
        }

        total_score = 0
        valid_rules = 0

        # 应用验证规则
        for rule in self.rules:
            if not rule.enabled:
                continue

            rule_result = self._apply_rule(signal, rule, market_data)
            validation_details['rule_results'][rule.name] = rule_result

            if rule_result['passed']:
                total_score += rule_result['score'] * rule.weight
                valid_rules += rule
            else:
                if rule_result['risk_level'] == 'high':
                    validation_details['risk_factors'].append(rule_result['message'])

            # 收集优势因素
            if rule_result.get('strength'):
                validation_details['strength_factors'].append(rule_result['message'])

        # 计算综合评分
        max_possible_score = sum(rule.weight for rule in self.rules if rule.enabled)
        confidence_score = total_score / max_possible_score if max_possible_score > 0 else 0

        # 检查历史表现
        historical_score = self._check_historical_performance(signal, historical_signals)
        confidence_score = (confidence_score + historical_score) / 2

        # 确定信号状态
        status, risk_level = self._determine_signal_status(confidence_score, signal, validation_details)

        # 确定建议操作
        suggested_action = self._get_suggested_action(status, confidence_score)

        # 创建验证结果
        result = ValidationResult(
            signal_id=signal_id,
            original_signal=signal,
            status=status,
            confidence_score=confidence_score,
            risk_level=risk_level,
            validation_details=validation_details,
            timestamp=datetime.now(),
            suggested_action=suggested_action
        )

        # 保存信号历史
        self.signal_history[signal_id] = result

        logger.info(f"信号 {signal_id} 验证完成,状态: {status.value}, 置信度: {confidence_score:.3f}")

        return result

    def _apply_rule(self, signal: TradingSignal, rule: ValidationRule,
                   market_data: Optional[Dict]) -> Dict[str, Any]:
        """
        应用单个验证规则

        Args:
            signal: 交易信号
            rule: 验证规则
            market_data: 市场数据

        Returns:
            规则执行结果
        """
        result = {
            'passed': False,
            'score': 0.0,
            'message': '',
            'risk_level': 'low',
            'strength': False
        }

        try:
            if rule.name == "confidence_threshold":
                # 置信度阈值检查
                if rule.min_threshold <= signal.confidence <= rule.max_threshold:
                    result['passed'] = True
                    result['score'] = signal.confidence
                    result['message'] = f"置信度 {signal.confidence:.3f} 在合理范围内"
                    result['strength'] = signal.confidence > 0.7
                else:
                    result['message'] = f"置信度 {signal.confidence:.3f} 超出阈值 [{rule.min_threshold}, {rule.max_threshold}]"
                    result['risk_level'] = 'high'

            elif rule.name == "price合理性":
                # 价格合理性检查
                if rule.min_threshold <= signal.price <= rule.max_threshold:
                    result['passed'] = True
                    result['score'] = 1.0
                    result['message'] = f"价格 {signal.price} 合理"
                    result['strength'] = True
                else:
                    result['message'] = f"价格 {signal.price} 不合理"
                    result['risk_level'] = 'high'

            elif rule.name == "volume验证":
                # 成交量验证
                volume = market_data.get('volume', 0) if market_data else 0
                if volume > 0:
                    result['passed'] = True
                    result['score'] = min(1.0, volume / 1000000)  # 相对评分
                    result['message'] = f"成交量 {volume} 正常"
                    result['strength'] = volume > 1000000
                else:
                    result['message'] = "成交量为零或负数"
                    result['risk_level'] = 'high'

            elif rule.name == "trend_consistency":
                # 趋势一致性检查
                trend_score = self._check_trend_consistency(signal, market_data)
                if trend_score > 0.5:
                    result['passed'] = True
                    result['score'] = trend_score
                    result['message'] = "信号与市场趋势基本一致"
                    result['strength'] = trend_score > 0.8
                else:
                    result['message'] = "信号与市场趋势不一致"
                    result['risk_level'] = 'medium'

            elif rule.name == "indicator_rationality":
                # 技术指标合理性
                indicator_score = self._check_indicators_rationality(signal)
                if indicator_score > 0.5:
                    result['passed'] = True
                    result['score'] = indicator_score
                    result['message'] = "技术指标在合理范围内"
                    result['strength'] = indicator_score > 0.8
                else:
                    result['message'] = "技术指标异常"
                    result['risk_level'] = 'medium'

            else:
                # 默认规则
                result['passed'] = True
                result['score'] = 1.0
                result['message'] = f"规则 {rule.name} 通过检查"

        except Exception as e:
            logger.error(f"应用规则 {rule.name} 失败: {e}")
            result['message'] = f"规则应用失败: {str(e)}"

        return result

    def _check_trend_consistency(self, signal: TradingSignal,
                               market_data: Optional[Dict]) -> float:
        """检查信号与市场趋势的一致性"""
        if not market_data:
            return 0.5  # 无市场数据,返回中性评分

        # 模拟趋势检查
        trend_direction = market_data.get('trend_direction', 'unknown')
        momentum = market_data.get('momentum', 0)

        # 根据信号方向和趋势方向计算一致性
        if signal.direction == '买入':
            if trend_direction in ['bullish', 'strong_bullish']:
                return min(1.0, momentum + 0.5)
            else:
                return max(0.0, momentum * 0.5)
        elif signal.direction == '卖出':
            if trend_direction in ['bearish', 'strong_bearish']:
                return min(1.0, abs(momentum) + 0.5)
            else:
                return max(0.0, abs(momentum) * 0.5)
        else:
            return 0.5  # 持有信号返回中性评分

    def _check_indicators_rationality(self, signal: TradingSignal) -> float:
        """检查技术指标的合理性"""
        if not signal.indicators:
            return 0.5  # 无指标数据

        total_rationality = 0
        indicator_count = 0

        for indicator in signal.indicators:
            name = indicator.get('name', '')
            value = indicator.get('value', 0)
            signal_type = indicator.get('signal_type', '')

            rationality_score = 1.0

            # 检查指标值是否在合理范围内
            if name.lower().startswith('ma'):
                # 移动平均线
                if value < 0:
                    rationality_score *= 0.5
            elif name.lower().startswith('rsi'):
                # RSI指标
                if not (0 <= value <= 100):
                    rationality_score *= 0.3
                elif value < 30 or value > 70:
                    rationality_score *= 0.7  # 极值但合理
                else:
                    rationality_score *= 1.0
            elif name.lower() == 'macd':
                # MACD指标
                if abs(value) > 1.0:  # 假设MACD值过大
                    rationality_score *= 0.6

            total_rationality += rationality_score
            indicator_count += 1

        return total_rationality / indicator_count if indicator_count > 0 else 0.5

    def _check_historical_performance(self, signal: TradingSignal,
                                   historical_signals: Optional[List[TradingSignal]]) -> float:
        """检查历史表现"""
        if not historical_signals or len(historical_signals) < 3:
            return 0.5  # 历史数据不足,返回中性评分

        # 找到相同品种的历史信号
        same_symbol_signals = [s for s in historical_signals
                              if s.symbol == signal.symbol and
                              s.timestamp < signal.timestamp]

        if len(same_symbol_signals) < 2:
            return 0.5

        # 计算历史表现
        successful_signals = 0
        for historical_signal in same_symbol_signals[-5:]:  # 只看最近的5个信号
            # 模拟检查历史信号的准确性
            if historical_signal.confidence > 0.7:
                successful_signals += 1

        success_rate = successful_signals / len(same_symbol_signals)
        return success_rate

    def _determine_signal_status(self, confidence_score: float,
                                signal: TradingSignal,
                                validation_details: Dict) -> Tuple[SignalStatus, RiskLevel]:
        """确定信号状态和风险等级"""
        # 根据置信度确定风险等级
        if confidence_score >= 0.8:
            risk_level = RiskLevel.LOW
        elif confidence_score >= 0.6:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.HIGH

        # 根据置信度和风险因素确定信号状态
        risk_factors = len(validation_details.get('risk_factors', []))

        if confidence_score >= self.confirmation_threshold and risk_factors == 0:
            return SignalStatus.CONFIRMED, risk_level
        elif confidence_score >= 0.6 and risk_factors <= 1:
            return SignalStatus.PARTIAL_CONFIRM, risk_level
        elif confidence_score >= 0.4:
            return SignalStatus.UNCERTAIN, risk_level
        else:
            return SignalStatus.REJECTED, risk_level

    def _get_suggested_action(self, status: SignalStatus, confidence_score: float) -> str:
        """获取建议操作"""
        if status == SignalStatus.CONFIRMED:
            return "立即执行"
        elif status == SignalStatus.PARTIAL_CONFIRM:
            return "谨慎执行,建议补充分析"
        elif status == SignalStatus.UNCERTAIN:
            return "观望,等待确认"
        elif status == SignalStatus.REJECTED:
            return "拒绝执行"
        else:
            return "需要进一步分析"

    def batch_validate_signals(self, signals: List[TradingSignal],
                             market_data: Optional[Dict] = None,
                             historical_signals: Optional[List[TradingSignal]] = None) -> List[ValidationResult]:
        """
        批量验证信号

        Args:
            signals: 信号列表
            market_data: 市场数据
            historical_signals: 历史信号

        Returns:
            验证结果列表
        """
        results = []

        logger.info(f"开始批量验证 {len(signals)} 个信号")

        for signal in signals:
            try:
                result = self.validate_signal(signal, market_data, historical_signals)
                results.append(result)
            except Exception as e:
                logger.error(f"验证信号失败: {e}")
                # 创建失败结果
                result = ValidationResult(
                    signal_id=f"error_{signal.symbol}",
                    original_signal=signal,
                    status=SignalStatus.INVALID,
                    confidence_score=0.0,
                    risk_level=RiskLevel.HIGH,
                    validation_details={'error': str(e)},
                    timestamp=datetime.now(),
                    suggested_action="拒绝执行"
                )
                results.append(result)

        # 按置信度排序
        results.sort(key=lambda x: x.confidence_score, reverse=True)

        logger.info(f"批量验证完成,有效信号: {len([r for r in results if r.status != SignalStatus.REJECTED])}")

        return results

    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """获取验证摘要"""
        if not results:
            return {}

        total_signals = len(results)
        confirmed_count = len([r for r in results if r.status == SignalStatus.CONFIRMED])
        partial_count = len([r for r in results if r.status == SignalStatus.PARTIAL_CONFIRM])
        uncertain_count = len([r for r in results if r.status == SignalStatus.UNCERTAIN])
        rejected_count = len([r for r in results if r.status == SignalStatus.REJECTED])

        avg_confidence = sum(r.confidence_score for r in results) / total_signals

        # 按风险等级统计
        risk_stats = {level.value: 0 for level in RiskLevel}
        for result in results:
            risk_stats[result.risk_level.value] += 1

        return {
            'total_signals': total_signals,
            'confirmed_signals': confirmed_count,
            'partial_signals': partial_count,
            'uncertain_signals': uncertain_count,
            'rejected_signals': rejected_count,
            'confirmation_rate': confirmed_count / total_signals,
            'average_confidence': avg_confidence,
            'risk_distribution': risk_stats
        }

    def update_performance_stats(self, signal_id: str, actual_result: float):
        """
        更新性能统计(实际结果)

        Args:
            signal_id: 信号ID
            actual_result: 实际结果(1.0表示成功,0.0表示失败)
        """
        if signal_id in self.signal_history:
            result = self.signal_history[signal_id]

            # 记录实际结果
            result.validation_details['actual_result'] = actual_result

            # 更新统计
            if signal_id not in self.performance_stats:
                self.performance_stats[signal_id] = {
                    'predicted_scores': [],
                    'actual_results': [],
                    'accuracy': 0.0
                }

            stats = self.performance_stats[signal_id]
            stats['predicted_scores'].append(result.confidence_score)
            stats['actual_results'].append(actual_result)

            # 计算准确率
            if len(stats['actual_results']) > 0:
                correct_predictions = 0
                for i, predicted in enumerate(stats['predicted_scores']):
                    actual = stats['actual_results'][i]
                    # 简单的准确率计算：预测高且实际成功,或预测低且实际失败
                    if (predicted > 0.5 and actual > 0.5) or (predicted <= 0.5 and actual <= 0.5):
                        correct_predictions += 1

                stats['accuracy'] = correct_predictions / len(stats['actual_results'])

            logger.info(f"更新信号 {signal_id} 性能统计,准确率: {stats['accuracy']:.3f}")

    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.performance_stats:
            return {'message': '暂无性能统计数据'}

        total_predictions = sum(len(stats['actual_results']) for stats in self.performance_stats.values())
        overall_accuracy = sum(stats['accuracy'] * len(stats['actual_results'])
                            for stats in self.performance_stats.values()) / total_predictions if total_predictions > 0 else 0

        # 按信号类型分组统计
        type_stats = {}
        for signal_id, stats in self.performance_stats.items():
            signal = self.signal_history.get(signal_id)
            if signal:
                signal_type = signal.original_signal.direction
                if signal_type not in type_stats:
                    type_stats[signal_type] = {
                        'count': 0,
                        'accuracy': 0.0,
                        'total_score': 0.0
                    }
                type_stats[signal_type]['count'] += 1
                type_stats[signal_type]['total_score'] += stats['accuracy']

        # 计算各类型的平均准确率
        for signal_type, stats in type_stats.items():
            if stats['count'] > 0:
                stats['accuracy'] = stats['total_score'] / stats['count']

        return {
            'total_predictions': total_predictions,
            'overall_accuracy': overall_accuracy,
            'type_statistics': type_stats,
            'improvement_suggestions': self._generate_performance_suggestions()
        }

    def _generate_performance_suggestions(self) -> List[str]:
        """生成性能改进建议"""
        suggestions = []

        # 分析整体准确率
        if self.performance_stats:
            total_accuracy = sum(stats['accuracy'] for stats in self.performance_stats.values())
            avg_accuracy = total_accuracy / len(self.performance_stats) if self.performance_stats else 0

            if avg_accuracy < 0.6:
                suggestions.append("整体准确率较低,建议调整验证规则")
            elif avg_accuracy > 0.8:
                suggestions.append("验证效果良好,建议保持当前策略")

        # 分析风险因素
        risk_factors_found = False
        for result in self.signal_history.values():
            if result.validation_details.get('risk_factors'):
                risk_factors_found = True
                break

        if risk_factors_found:
            suggestions.append("检测到多个风险因素,建议加强风险控制")

        return suggestions