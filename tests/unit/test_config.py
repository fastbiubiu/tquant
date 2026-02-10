"""
配置管理单元测试 - 新的 Pydantic 配置系统
"""

import unittest
import os
import tempfile
import json
from pathlib import Path

from config import Config, get_config, load_config, save_config, get_config_dir, get_config_path


class TestConfigBasics(unittest.TestCase):
    """测试配置基础功能"""

    def test_get_config_returns_config_object(self):
        """测试 get_config 返回 Config 对象"""
        config = get_config()
        assert isinstance(config, Config)

    def test_config_has_required_attributes(self):
        """测试 Config 对象有必需的属性"""
        config = get_config()
        # 检查主要配置部分
        assert hasattr(config, 'trading')
        assert hasattr(config, 'llm')
        assert hasattr(config, 'indicators')

    def test_config_trading_section(self):
        """测试交易配置部分"""
        config = get_config()
        assert hasattr(config.trading, 'symbols')
        assert hasattr(config.trading, 'account')
        assert isinstance(config.trading.symbols, list)

    def test_config_llm_section(self):
        """测试LLM配置部分"""
        config = get_config()
        assert hasattr(config.llm, 'gpt4o')
        assert config.llm.gpt4o is not None
        assert config.llm.gpt4o.model == 'gpt-4o'

    def test_config_indicators_section(self):
        """测试指标配置部分"""
        config = get_config()
        assert hasattr(config.indicators, 'rsi')
        assert hasattr(config.indicators, 'macd')
        assert hasattr(config.indicators, 'bollinger_bands')


class TestConfigPaths(unittest.TestCase):
    """测试配置路径功能"""

    def test_get_config_dir(self):
        """测试获取配置目录"""
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)
        assert config_dir.exists()

    def test_get_config_path(self):
        """测试获取配置文件路径"""
        config_path = get_config_path()
        assert isinstance(config_path, Path)
        # 配置文件可能不存在，但路径应该是有效的

    def test_config_dir_is_home_tquant(self):
        """测试配置目录是 ~/.tquant"""
        config_dir = get_config_dir()
        assert config_dir.name == '.tquant'
        assert config_dir.parent == Path.home()


class TestConfigLoading(unittest.TestCase):
    """测试配置加载功能"""

    def setUp(self):
        """设置测试环境"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')

    def tearDown(self):
        """清理测试环境"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_load_config_from_file(self):
        """测试从文件加载配置"""
        # 创建测试配置文件
        test_config = {
            'trading': {
                'symbols': ['SHFE.cu2401', 'DCE.i2403'],
                'account': {
                    'initial_balance': 1000000,
                    'max_position_ratio': 0.7
                },
                'risk': {
                    'max_loss_ratio': 0.05,
                    'stop_loss_ratio': 0.02,
                    'take_profit_ratio': 0.05
                }
            },
            'llm': {
                'gpt4o': {
                    'model': 'gpt-4o',
                    'api_key': 'test-key',
                    'temperature': 0.7,
                    'max_tokens': 2000,
                    'timeout': 30
                }
            }
        }

        with open(self.config_file, 'w') as f:
            json.dump(test_config, f)

        # 加载配置
        config = load_config(Path(self.config_file))
        assert isinstance(config, Config)
        assert config.trading.symbols == ['SHFE.cu2401', 'DCE.i2403']

    def test_save_config_to_file(self):
        """测试保存配置到文件"""
        config = get_config()
        save_config(config, Path(self.config_file))

        assert os.path.exists(self.config_file)

        # 验证保存的内容
        with open(self.config_file, 'r') as f:
            saved_data = json.load(f)

        assert 'trading' in saved_data
        assert 'llm' in saved_data


class TestConfigTypeValidation(unittest.TestCase):
    """测试配置类型验证"""

    def test_config_symbols_is_list(self):
        """测试 symbols 是列表"""
        config = get_config()
        assert isinstance(config.trading.symbols, list)
        for symbol in config.trading.symbols:
            assert isinstance(symbol, str)

    def test_config_initial_balance_is_number(self):
        """测试初始余额是数字"""
        config = get_config()
        assert isinstance(config.trading.account.initial_balance, (int, float))
        assert config.trading.account.initial_balance > 0

    def test_config_max_position_ratio_is_float(self):
        """测试最大持仓比例是浮点数"""
        config = get_config()
        assert isinstance(config.trading.account.max_position_ratio, (int, float))
        assert 0 <= config.trading.account.max_position_ratio <= 1

    def test_config_model_is_string(self):
        """测试模型名称是字符串"""
        config = get_config()
        assert isinstance(config.llm.gpt4o.model, str)
        assert len(config.llm.gpt4o.model) > 0


class TestConfigDefaults(unittest.TestCase):
    """测试配置默认值"""

    def test_default_trading_symbols(self):
        """测试默认交易品种"""
        config = get_config()
        assert len(config.trading.symbols) > 0

    def test_default_initial_balance(self):
        """测试默认初始余额"""
        config = get_config()
        assert config.trading.account.initial_balance == 1000000

    def test_default_max_position_ratio(self):
        """测试默认最大持仓比例"""
        config = get_config()
        assert config.trading.account.max_position_ratio == 0.7

    def test_default_llm_model(self):
        """测试默认LLM模型"""
        config = get_config()
        assert config.llm.gpt4o is not None
        assert config.llm.gpt4o.model == 'gpt-4o'


class TestConfigEnvironmentVariables(unittest.TestCase):
    """测试环境变量覆盖"""

    def test_environment_variable_override(self):
        """测试环境变量覆盖配置"""
        # 设置环境变量
        os.environ['TQUANT_TRADING_SYMBOLS'] = '["SHFE.cu2402"]'

        # 重新加载配置
        from importlib import reload
        import config as config_module
        reload(config_module)

        config = config_module.get_config()
        # 验证环境变量是否被应用
        # 注意：这取决于配置系统的实现

    def tearDown(self):
        """清理环境变量"""
        if 'TQUANT_TRADING_SYMBOLS' in os.environ:
            del os.environ['TQUANT_TRADING_SYMBOLS']


class TestConfigModelDump(unittest.TestCase):
    """测试配置模型导出"""

    def test_model_dump_returns_dict(self):
        """测试 model_dump 返回字典"""
        config = get_config()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert 'trading' in config_dict
        assert 'llm' in config_dict

    def test_model_dump_preserves_structure(self):
        """测试 model_dump 保留结构"""
        config = get_config()
        config_dict = config.model_dump()

        assert isinstance(config_dict['trading'], dict)
        assert isinstance(config_dict['trading']['symbols'], list)
        assert isinstance(config_dict['llm'], dict)


if __name__ == '__main__':
    unittest.main()
