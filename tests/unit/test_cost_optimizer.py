"""
成本优化器单元测试 - 新的 API
"""

import pytest
from utils.cost_optimizer import CostOptimizer


class TestCostOptimizerInitialization:
    """测试成本优化器初始化"""

    def test_initialization_with_empty_config(self):
        """测试使用空配置初始化"""
        optimizer = CostOptimizer(config={})
        assert optimizer is not None

    def test_initialization_with_custom_config(self):
        """测试使用自定义配置初始化"""
        config = {
            'budget': 1000,
            'alert_threshold': 0.8
        }
        optimizer = CostOptimizer(config=config)
        assert optimizer is not None


class TestCostCalculation:
    """测试成本计算"""

    def test_calculate_cost_gpt4o(self):
        """测试计算gpt-4o成本"""
        optimizer = CostOptimizer(config={})

        # 计算成本
        cost = optimizer.calculate_cost(
            model='gpt-4o',
            input_tokens=100,
            output_tokens=50
        )

        assert isinstance(cost, (int, float))
        assert cost >= 0

    def test_calculate_cost_gpt4o_mini(self):
        """测试计算gpt-4o-mini成本"""
        optimizer = CostOptimizer(config={})

        cost = optimizer.calculate_cost(
            model='gpt-4o-mini',
            input_tokens=100,
            output_tokens=50
        )

        assert isinstance(cost, (int, float))
        assert cost >= 0

    def test_calculate_cost_comparison(self):
        """测试成本比较"""
        optimizer = CostOptimizer(config={})

        cost_gpt4o = optimizer.calculate_cost(
            model='gpt-4o',
            input_tokens=100,
            output_tokens=50
        )

        cost_mini = optimizer.calculate_cost(
            model='gpt-4o-mini',
            input_tokens=100,
            output_tokens=50
        )

        # gpt-4o-mini应该更便宜
        assert cost_mini <= cost_gpt4o


class TestCostSummary:
    """测试成本摘要"""

    def test_get_cost_summary(self):
        """测试获取成本摘要"""
        optimizer = CostOptimizer(config={})

        # 计算一些成本
        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)
        optimizer.calculate_cost(model='gpt-4o', input_tokens=200, output_tokens=100)

        summary = optimizer.get_cost_summary()
        assert isinstance(summary, dict)

    def test_get_model_breakdown(self):
        """测试获取模型成本分解"""
        optimizer = CostOptimizer(config={})

        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)

        breakdown = optimizer.get_model_breakdown()
        assert isinstance(breakdown, dict)

    def test_get_daily_costs(self):
        """测试获取每日成本"""
        optimizer = CostOptimizer(config={})

        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)

        daily_costs = optimizer.get_daily_costs(days=30)
        assert isinstance(daily_costs, dict)


class TestOptimizationRecommendations:
    """测试优化建议"""

    def test_recommend_optimizations(self):
        """测试获取优化建议"""
        optimizer = CostOptimizer(config={})

        # 计算一些成本
        for _ in range(10):
            optimizer.calculate_cost(model='gpt-4o', input_tokens=1000, output_tokens=500)

        recommendations = optimizer.recommend_optimizations()
        assert isinstance(recommendations, list)

    def test_optimization_strategies(self):
        """测试优化策略"""
        optimizer = CostOptimizer(config={})

        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)

        recommendations = optimizer.recommend_optimizations()
        assert isinstance(recommendations, list)


class TestCostStats:
    """测试成本统计"""

    def test_get_stats(self):
        """测试获取统计信息"""
        optimizer = CostOptimizer(config={})

        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)

        stats = optimizer.get_stats()
        assert isinstance(stats, dict)

    def test_get_summary(self):
        """测试获取摘要"""
        optimizer = CostOptimizer(config={})

        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)

        summary = optimizer.get_summary()
        assert isinstance(summary, str)


class TestCostReporting:
    """测试成本报告"""

    def test_export_report(self):
        """测试导出报告"""
        optimizer = CostOptimizer(config={})

        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)

        report = optimizer.export_report()
        assert isinstance(report, str)

    def test_export_report_to_file(self):
        """测试导出报告到文件"""
        import tempfile
        import os

        optimizer = CostOptimizer(config={})

        optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, 'report.txt')
            report = optimizer.export_report(filename=filepath)
            assert isinstance(report, str)


class TestCostOptimizerEdgeCases:
    """测试成本优化器边界情况"""

    def test_calculate_cost_zero_tokens(self):
        """测试零token成本"""
        optimizer = CostOptimizer(config={})

        cost = optimizer.calculate_cost(
            model='gpt-4o',
            input_tokens=0,
            output_tokens=0
        )

        assert cost == 0

    def test_calculate_cost_large_tokens(self):
        """测试大量token成本"""
        optimizer = CostOptimizer(config={})

        cost = optimizer.calculate_cost(
            model='gpt-4o',
            input_tokens=100000,
            output_tokens=50000
        )

        assert isinstance(cost, (int, float))
        assert cost >= 0

    def test_multiple_models(self):
        """测试多个模型"""
        optimizer = CostOptimizer(config={})

        cost1 = optimizer.calculate_cost(model='gpt-4o', input_tokens=100, output_tokens=50)
        cost2 = optimizer.calculate_cost(model='gpt-4o-mini', input_tokens=100, output_tokens=50)

        assert isinstance(cost1, (int, float))
        assert isinstance(cost2, (int, float))
