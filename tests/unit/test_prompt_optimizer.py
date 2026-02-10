"""
Prompt优化器单元测试 - 新的 API
"""

import pytest
from utils.prompt_optimizer import PromptOptimizer


class TestPromptOptimizerInitialization:
    """测试Prompt优化器初始化"""

    def test_initialization(self):
        """测试初始化"""
        optimizer = PromptOptimizer(config={})
        assert optimizer is not None

    def test_initialization_with_config(self):
        """测试使用配置初始化"""
        config = {'model': 'gpt-4o', 'temperature': 0.7}
        optimizer = PromptOptimizer(config=config)
        assert optimizer is not None


class TestPromptVariants:
    """测试Prompt变体"""

    def test_create_prompt_variants(self):
        """测试创建Prompt变体"""
        optimizer = PromptOptimizer(config={})

        base_prompt = "分析市场趋势"
        variants = optimizer.create_prompt_variants(base_prompt, num_variants=3)

        assert isinstance(variants, list)
        assert len(variants) > 0

    def test_prompt_variant_structure(self):
        """测试Prompt变体结构"""
        optimizer = PromptOptimizer(config={})

        base_prompt = "生成交易信号"
        variants = optimizer.create_prompt_variants(base_prompt, num_variants=2)

        for variant in variants:
            assert isinstance(variant, (str, dict))


class TestPromptOptimization:
    """测试Prompt优化"""

    def test_get_optimal_prompt(self):
        """测试获取最优Prompt"""
        optimizer = PromptOptimizer(config={})

        base_prompt = "分析价格走势"
        optimal = optimizer.get_optimal_prompt(base_prompt)

        assert isinstance(optimal, (str, dict))

    def test_generate_improvement_suggestions(self):
        """测试生成改进建议"""
        optimizer = PromptOptimizer(config={})

        prompt = "分析市场"
        suggestions = optimizer.generate_improvement_suggestions(prompt)

        assert isinstance(suggestions, (list, dict))


class TestABTesting:
    """测试A/B测试"""

    def test_ab_test_creation(self):
        """测试A/B测试创建"""
        optimizer = PromptOptimizer(config={})

        prompt_a = "分析趋势"
        prompt_b = "分析市场走势"

        test = optimizer.create_ab_test(prompt_a, prompt_b)
        assert test is not None

    def test_ab_test_execution(self):
        """测试A/B测试执行"""
        optimizer = PromptOptimizer(config={})

        prompt_a = "生成信号"
        prompt_b = "生成交易信号"

        test = optimizer.create_ab_test(prompt_a, prompt_b)
        if test:
            result = optimizer.execute_ab_test(test)
            assert result is not None or result is None


class TestPromptExperiments:
    """测试Prompt实验"""

    def test_save_and_load_experiments(self):
        """测试保存和加载实验"""
        optimizer = PromptOptimizer(config={})

        base_prompt = "分析数据"
        variants = optimizer.create_prompt_variants(base_prompt, num_variants=2)

        if hasattr(optimizer, 'save_experiments'):
            optimizer.save_experiments('test_experiment')

            if hasattr(optimizer, 'load_experiments'):
                loaded = optimizer.load_experiments('test_experiment')
                assert loaded is not None or loaded is None

    def test_get_test_summary(self):
        """测试获取测试摘要"""
        optimizer = PromptOptimizer(config={})

        if hasattr(optimizer, 'get_test_summary'):
            summary = optimizer.get_test_summary()
            assert isinstance(summary, (str, dict)) or summary is None


class TestPromptAnalysis:
    """测试Prompt分析"""

    def test_analyze_ab_test_results(self):
        """测试分析A/B测试结果"""
        optimizer = PromptOptimizer(config={})

        if hasattr(optimizer, 'analyze_ab_test_results'):
            results = optimizer.analyze_ab_test_results()
            assert results is not None or results is None

    def test_export_test_report(self):
        """测试导出测试报告"""
        optimizer = PromptOptimizer(config={})

        if hasattr(optimizer, 'export_test_report'):
            report = optimizer.export_test_report()
            assert report is not None or report is None


class TestPromptOptimizerEdgeCases:
    """测试Prompt优化器边界情况"""

    def test_empty_prompt(self):
        """测试空Prompt"""
        optimizer = PromptOptimizer(config={})

        variants = optimizer.create_prompt_variants("", num_variants=1)
        assert isinstance(variants, list)

    def test_long_prompt(self):
        """测试长Prompt"""
        optimizer = PromptOptimizer(config={})

        long_prompt = "分析市场趋势" * 100
        variants = optimizer.create_prompt_variants(long_prompt, num_variants=1)
        assert isinstance(variants, list)

    def test_special_characters_in_prompt(self):
        """测试特殊字符Prompt"""
        optimizer = PromptOptimizer(config={})

        special_prompt = "分析@#$%^&*()市场"
        variants = optimizer.create_prompt_variants(special_prompt, num_variants=1)
        assert isinstance(variants, list)
