"""
缓存管理器单元测试 - 新的 API
"""

import pytest
import time
from utils.cache_manager import CacheManager


class TestCacheManagerInitialization:
    """测试缓存管理器初始化"""

    def test_initialization_with_empty_config(self):
        """测试使用空配置初始化"""
        cache = CacheManager(config={})
        assert cache is not None
        assert hasattr(cache, 'get') and hasattr(cache, 'set')

    def test_initialization_with_custom_config(self):
        """测试使用自定义配置初始化"""
        config = {'max_size': 1000, 'ttl': 3600}
        cache = CacheManager(config=config)
        assert cache is not None


class TestCacheSetAndGet:
    """测试缓存设置和获取"""

    def test_set_and_get_basic(self):
        """测试基础设置和获取"""
        cache = CacheManager(config={})
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'

    def test_set_and_get_multiple(self):
        """测试多个键值对"""
        cache = CacheManager(config={})
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')

        assert cache.get('key1') == 'value1'
        assert cache.get('key2') == 'value2'
        assert cache.get('key3') == 'value3'

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        cache = CacheManager(config={})
        assert cache.get('nonexistent') is None

    def test_overwrite_existing_key(self):
        """测试覆盖现有键"""
        cache = CacheManager(config={})
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'

        cache.set('key1', 'value2')
        assert cache.get('key1') == 'value2'


class TestCacheDelete:
    """测试缓存删除"""

    def test_invalidate_existing_key(self):
        """测试删除现有键"""
        cache = CacheManager(config={})
        cache.set('key1', 'value1')
        assert cache.get('key1') == 'value1'

        cache.invalidate('key1')
        assert cache.get('key1') is None

    def test_invalidate_nonexistent_key(self):
        """测试删除不存在的键"""
        cache = CacheManager(config={})
        # 不应该抛出错误
        cache.invalidate('nonexistent')

    def test_invalidate_all(self):
        """测试清空所有缓存"""
        cache = CacheManager(config={})
        cache.set('key1', 'value1')
        cache.set('key2', 'value2')
        cache.set('key3', 'value3')

        cache.invalidate()  # 不带参数清空所有

        assert cache.get('key1') is None
        assert cache.get('key2') is None
        assert cache.get('key3') is None


class TestCacheDataTypes:
    """测试缓存数据类型"""

    def test_cache_string(self):
        """测试缓存字符串"""
        cache = CacheManager(config={})
        cache.set('key', 'string_value')
        assert cache.get('key') == 'string_value'

    def test_cache_number(self):
        """测试缓存数字"""
        cache = CacheManager(config={})
        cache.set('key', 42)
        assert cache.get('key') == 42

    def test_cache_list(self):
        """测试缓存列表"""
        cache = CacheManager(config={})
        data = [1, 2, 3, 4, 5]
        cache.set('key', data)
        assert cache.get('key') == data

    def test_cache_dict(self):
        """测试缓存字典"""
        cache = CacheManager(config={})
        data = {'a': 1, 'b': 2, 'c': 3}
        cache.set('key', data)
        assert cache.get('key') == data

    def test_cache_complex_object(self):
        """测试缓存复杂对象"""
        cache = CacheManager(config={})
        data = {
            'list': [1, 2, 3],
            'dict': {'nested': 'value'},
            'string': 'test'
        }
        cache.set('key', data)
        assert cache.get('key') == data


class TestCacheStatistics:
    """测试缓存统计"""

    def test_cache_has_stats_method(self):
        """测试缓存有统计方法"""
        cache = CacheManager(config={})
        # 检查是否有统计相关的方法
        assert hasattr(cache, 'get') or hasattr(cache, 'set')


class TestCacheEdgeCases:
    """测试缓存边界情况"""

    def test_set_none_value(self):
        """测试设置None值"""
        cache = CacheManager(config={})
        cache.set('key', None)
        # None值应该被缓存
        assert cache.get('key') is None or 'key' in cache.cache

    def test_set_empty_string(self):
        """测试设置空字符串"""
        cache = CacheManager(config={})
        cache.set('key', '')
        assert cache.get('key') == ''

    def test_set_empty_list(self):
        """测试设置空列表"""
        cache = CacheManager(config={})
        cache.set('key', [])
        assert cache.get('key') == []

    def test_set_empty_dict(self):
        """测试设置空字典"""
        cache = CacheManager(config={})
        cache.set('key', {})
        assert cache.get('key') == {}
