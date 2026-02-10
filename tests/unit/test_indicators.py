"""
技术指标单元测试 - 新的 API
"""

import pytest
import pandas as pd
import numpy as np
from utils.indicators import TechnicalIndicators


class TestTechnicalIndicatorsBasics:
    """测试技术指标基础功能"""

    @pytest.fixture
    def sample_data(self):
        """创建示例数据"""
        dates = pd.date_range('2024-01-01', periods=100)
        data = {
            'close': np.random.uniform(8000, 8500, 100),
            'high': np.random.uniform(8500, 9000, 100),
            'low': np.random.uniform(7500, 8000, 100),
            'volume': np.random.uniform(1000, 5000, 100)
        }
        df = pd.DataFrame(data, index=dates)
        return df

    def test_ma_calculation(self, sample_data):
        """测试移动平均线计算"""
        ma_result = TechnicalIndicators.ma(sample_data['close'], periods=[5, 10, 20])
        assert isinstance(ma_result, dict)
        assert 5 in ma_result
        assert 10 in ma_result
        assert 20 in ma_result

    def test_ema_calculation(self, sample_data):
        """测试指数移动平均线计算"""
        ema_result = TechnicalIndicators.ema(sample_data['close'], periods=[5, 10, 20])
        assert isinstance(ema_result, dict)
        assert 5 in ema_result
        assert 10 in ema_result
        assert 20 in ema_result

    def test_rsi_calculation(self, sample_data):
        """测试RSI计算"""
        rsi_result = TechnicalIndicators.rsi(sample_data['close'], period=14)
        assert isinstance(rsi_result, (int, float))
        assert 0 <= rsi_result <= 100

    def test_macd_calculation(self, sample_data):
        """测试MACD计算"""
        macd_result = TechnicalIndicators.macd(sample_data['close'])
        assert isinstance(macd_result, dict)
        assert 'macd' in macd_result
        assert 'signal' in macd_result
        assert 'histogram' in macd_result

    def test_bollinger_bands_calculation(self, sample_data):
        """测试布林带计算"""
        bb_result = TechnicalIndicators.bollinger_bands(sample_data['close'], period=20)
        assert isinstance(bb_result, dict)
        assert 'upper' in bb_result
        assert 'middle' in bb_result
        assert 'lower' in bb_result


class TestTechnicalIndicatorsEdgeCases:
    """测试技术指标边界情况"""

    def test_ma_with_small_data(self):
        """测试小数据集的MA"""
        data = pd.Series([100, 101, 102, 103, 104])
        ma_result = TechnicalIndicators.ma(data, periods=[2, 3])
        assert isinstance(ma_result, dict)

    def test_macd_with_small_data(self):
        """测试小数据集的MACD"""
        data = pd.Series(np.random.uniform(100, 110, 30))
        macd_result = TechnicalIndicators.macd(data)
        assert isinstance(macd_result, dict)


class TestTechnicalIndicatorsIntegration:
    """测试技术指标集成"""

    def test_volume_profile(self):
        """测试成交量分布"""
        dates = pd.date_range('2024-01-01', periods=50)
        df = pd.DataFrame({
            'close': np.random.uniform(8000, 8500, 50),
            'volume': np.random.uniform(1000, 5000, 50)
        }, index=dates)

        vp_result = TechnicalIndicators.volume_profile(df, price_col='close', volume_col='volume')
        assert isinstance(vp_result, dict)


class TestTechnicalIndicatorsPerformance:
    """测试技术指标性能"""

    def test_ma_performance_large_dataset(self):
        """测试大数据集的MA性能"""
        data = pd.Series(np.random.uniform(8000, 8500, 1000))
        ma_result = TechnicalIndicators.ma(data, periods=[5, 10, 20, 50])
        assert isinstance(ma_result, dict)
        assert len(ma_result) == 4

    def test_multiple_indicators(self):
        """测试多个指标"""
        dates = pd.date_range('2024-01-01', periods=100)
        df = pd.DataFrame({
            'close': np.random.uniform(8000, 8500, 100),
        }, index=dates)

        # 计算多个指标
        ma = TechnicalIndicators.ma(df['close'], periods=[5, 10])
        rsi = TechnicalIndicators.rsi(df['close'], period=14)
        macd = TechnicalIndicators.macd(df['close'])

        assert isinstance(ma, dict)
        assert isinstance(rsi, (int, float))
        assert isinstance(macd, dict)
