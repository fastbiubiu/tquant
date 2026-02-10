"""
成本优化器模块
用于成本计算、成本优化和成本监控
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)


@dataclass
class CostRecord:
    """成本记录"""
    record_id: str
    timestamp: datetime
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    operation_type: str  # api_call, computation, storage
    metadata: Dict = field(default_factory=dict)


class CostOptimizer:
    """
    成本优化器

    功能:
    - 成本计算
    - 成本优化策略
    - 成本监控
    - 成本报告
    - 预算管理
    """

    def __init__(self, config: Dict):
        """
        初始化成本优化器

        Args:
            config: 配置字典
        """
        self.config = config
        self.model_costs = config.get('model_costs', {})
        self.monthly_budget = config.get('monthly_budget', 100.0)
        self.daily_budget = config.get('daily_budget', 5.0)
        self.warning_threshold = config.get('warning_threshold', 0.8)

        # 成本记录
        self.cost_records: List[CostRecord] = []
        self.daily_costs: Dict[str, float] = defaultdict(float)
        self.model_costs_breakdown: Dict[str, float] = defaultdict(float)

        # 统计信息
        self.stats = {
            'total_cost': 0.0,
            'total_api_calls': 0,
            'total_tokens': 0,
            'average_cost_per_call': 0.0,
            'daily_average': 0.0,
            'monthly_estimate': 0.0
        }

        # 优化策略
        self.optimization_strategies = {
            'use_cheaper_model': self._use_cheaper_model,
            'batch_requests': self._batch_requests,
            'cache_results': self._cache_results,
            'reduce_frequency': self._reduce_frequency
        }

        logger.info(f"成本优化器初始化完成: 月度预算=${self.monthly_budget}")

    def calculate_cost(self, model: str, input_tokens: int, output_tokens: int,
                      operation_type: str = 'api_call') -> float:
        """
        计算成本

        Args:
            model: 模型名称
            input_tokens: 输入 token 数
            output_tokens: 输出 token 数
            operation_type: 操作类型

        Returns:
            成本(美元)
        """
        if model not in self.model_costs:
            logger.warning(f"未知模型: {model}")
            return 0.0

        costs = self.model_costs[model]
        cost = (input_tokens * costs['input'] + output_tokens * costs['output']) / 1000

        # 记录成本
        record = CostRecord(
            record_id=f"cost_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
            operation_type=operation_type
        )
        self.cost_records.append(record)

        # 更新统计
        today = datetime.now().strftime('%Y-%m-%d')
        self.daily_costs[today] += cost
        self.model_costs_breakdown[model] += cost
        self.stats['total_cost'] += cost
        self.stats['total_api_calls'] += 1
        self.stats['total_tokens'] += input_tokens + output_tokens
        self.stats['average_cost_per_call'] = (
            self.stats['total_cost'] / self.stats['total_api_calls']
        )

        # 计算月度估计
        days_passed = len(self.daily_costs)
        if days_passed > 0:
            daily_average = self.stats['total_cost'] / days_passed
            self.stats['daily_average'] = daily_average
            self.stats['monthly_estimate'] = daily_average * 30

        # 检查预算
        self._check_budget()

        logger.debug(f"成本计算: {model}, 输入: {input_tokens}, 输出: {output_tokens}, 成本: ${cost:.4f}")

        return cost

    def _check_budget(self):
        """检查预算"""
        today = datetime.now().strftime('%Y-%m-%d')
        daily_cost = self.daily_costs[today]

        # 检查日预算
        if daily_cost > self.daily_budget:
            logger.warning(f"日预算超支: ${daily_cost:.2f} > ${self.daily_budget:.2f}")

        # 检查月度预算
        if self.stats['monthly_estimate'] > self.monthly_budget * self.warning_threshold:
            logger.warning(
                f"月度预算预警: ${self.stats['monthly_estimate']:.2f} > "
                f"${self.monthly_budget * self.warning_threshold:.2f}"
            )

    def get_cost_summary(self) -> Dict:
        """获取成本摘要"""
        today = datetime.now().strftime('%Y-%m-%d')
        daily_cost = self.daily_costs[today]

        return {
            'total_cost': self.stats['total_cost'],
            'daily_cost': daily_cost,
            'monthly_estimate': self.stats['monthly_estimate'],
            'monthly_budget': self.monthly_budget,
            'budget_remaining': self.monthly_budget - self.stats['monthly_estimate'],
            'budget_used_percentage': (
                self.stats['monthly_estimate'] / self.monthly_budget * 100
            ),
            'average_cost_per_call': self.stats['average_cost_per_call'],
            'total_api_calls': self.stats['total_api_calls'],
            'total_tokens': self.stats['total_tokens']
        }

    def get_model_breakdown(self) -> Dict[str, float]:
        """获取模型成本分解"""
        return dict(self.model_costs_breakdown)

    def get_daily_costs(self, days: int = 30) -> Dict[str, float]:
        """
        获取每日成本

        Args:
            days: 天数

        Returns:
            每日成本字典
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        result = {}

        for date_str, cost in self.daily_costs.items():
            date = datetime.strptime(date_str, '%Y-%m-%d')
            if date >= cutoff_date:
                result[date_str] = cost

        return result

    def recommend_optimizations(self) -> List[Dict]:
        """
        推荐优化策略

        Returns:
            优化建议列表
        """
        recommendations = []

        # 检查是否超预算
        if self.stats['monthly_estimate'] > self.monthly_budget:
            recommendations.append({
                'strategy': 'use_cheaper_model',
                'description': '使用更便宜的模型',
                'potential_savings': self._estimate_savings('use_cheaper_model'),
                'priority': 'high'
            })

        # 检查 API 调用频率
        if self.stats['total_api_calls'] > 100:
            recommendations.append({
                'strategy': 'batch_requests',
                'description': '批量处理请求',
                'potential_savings': self._estimate_savings('batch_requests'),
                'priority': 'high'
            })

        # 检查缓存使用
        recommendations.append({
            'strategy': 'cache_results',
            'description': '缓存结果以减少重复调用',
            'potential_savings': self._estimate_savings('cache_results'),
            'priority': 'medium'
        })

        # 检查调用频率
        if len(self.daily_costs) > 0:
            avg_daily = self.stats['total_cost'] / len(self.daily_costs)
            if avg_daily > self.daily_budget * 0.8:
                recommendations.append({
                    'strategy': 'reduce_frequency',
                    'description': '减少 API 调用频率',
                    'potential_savings': self._estimate_savings('reduce_frequency'),
                    'priority': 'medium'
                })

        return recommendations

    def _estimate_savings(self, strategy: str) -> float:
        """估计节省成本"""
        if strategy == 'use_cheaper_model':
            # 估计使用便宜模型可节省 30%
            return self.stats['total_cost'] * 0.3
        elif strategy == 'batch_requests':
            # 估计批量处理可节省 20%
            return self.stats['total_cost'] * 0.2
        elif strategy == 'cache_results':
            # 估计缓存可节省 40%
            return self.stats['total_cost'] * 0.4
        elif strategy == 'reduce_frequency':
            # 估计减少频率可节省 25%
            return self.stats['total_cost'] * 0.25
        return 0.0

    def _use_cheaper_model(self) -> Dict:
        """使用更便宜的模型"""
        return {
            'recommendation': '使用 GPT-4o-mini 或 Claude-3-haiku',
            'estimated_savings': self._estimate_savings('use_cheaper_model')
        }

    def _batch_requests(self) -> Dict:
        """批量处理请求"""
        return {
            'recommendation': '将多个请求合并为一个批量请求',
            'estimated_savings': self._estimate_savings('batch_requests')
        }

    def _cache_results(self) -> Dict:
        """缓存结果"""
        return {
            'recommendation': '实现结果缓存以减少重复调用',
            'estimated_savings': self._estimate_savings('cache_results')
        }

    def _reduce_frequency(self) -> Dict:
        """减少调用频率"""
        return {
            'recommendation': '减少 API 调用频率,增加调用间隔',
            'estimated_savings': self._estimate_savings('reduce_frequency')
        }

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            **self.stats,
            'model_breakdown': dict(self.model_costs_breakdown),
            'daily_costs': dict(self.daily_costs),
            'cost_records': len(self.cost_records)
        }

    def get_summary(self) -> str:
        """获取摘要"""
        summary = self.get_cost_summary()
        breakdown = self.get_model_breakdown()

        summary_str = f"""
=== 成本优化器摘要 ===
总成本: ${summary['total_cost']:.2f}
今日成本: ${summary['daily_cost']:.2f}
月度估计: ${summary['monthly_estimate']:.2f}
月度预算: ${summary['monthly_budget']:.2f}
剩余预算: ${summary['budget_remaining']:.2f}
预算使用率: {summary['budget_used_percentage']:.1f}%

API 统计:
  总调用数: {summary['total_api_calls']}
  总 Token 数: {summary['total_tokens']}
  平均成本/调用: ${summary['average_cost_per_call']:.4f}

模型成本分解:
"""
        for model, cost in breakdown.items():
            summary_str += f"  {model}: ${cost:.2f}\n"

        # 添加优化建议
        recommendations = self.recommend_optimizations()
        if recommendations:
            summary_str += "\n优化建议:\n"
            for rec in recommendations:
                summary_str += f"  • {rec['description']} (优先级: {rec['priority']})\n"
                summary_str += f"    预计节省: ${rec['potential_savings']:.2f}\n"

        return summary_str

    def export_report(self, filename: str = None) -> str:
        """
        导出成本报告

        Args:
            filename: 文件名

        Returns:
            报告内容
        """
        if filename is None:
            filename = f"cost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

        report = self.get_summary()

        try:
            with open(filename, 'w') as f:
                f.write(report)
            logger.info(f"成本报告已导出: {filename}")
        except Exception as e:
            logger.error(f"导出成本报告失败: {e}")

        return report

    def close(self):
        """关闭优化器"""
        logger.info("成本优化器已关闭")


# 全局成本优化器实例
_cost_optimizer: Optional[CostOptimizer] = None


def get_cost_optimizer(config: Dict) -> CostOptimizer:
    """获取全局成本优化器实例"""
    global _cost_optimizer
    if _cost_optimizer is None:
        _cost_optimizer = CostOptimizer(config)
    return _cost_optimizer
