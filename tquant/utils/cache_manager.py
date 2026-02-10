"""
缓存管理器
使用内存缓存和可选的 Redis 缓存来减少重复计算和 API 调用成本
"""

import hashlib
import json
import logging
import time
from functools import wraps
from typing import Dict, Optional, Any, Union

from tquant.config import Config

logger = logging.getLogger(__name__)


class CacheManager:
    """缓存管理器"""

    def __init__(self, config: Union[Dict, Config]):
        """
        初始化缓存管理器

        Args:
            config: 缓存配置(字典或 Config 对象)
        """
        self.config = config

        # 支持字典和 Config 对象
        if isinstance(config, Config):
            # 从 Config 对象获取配置(使用默认值)
            self.enabled = True
            self.ttl = 300
            self.max_size = 1000
            self.use_redis = False
            self.cache_type = 'memory'
            redis_host = 'localhost'
            redis_port = 6379
            redis_db = 0
            redis_password = None
        else:
            # 从字典获取配置
            self.enabled = config.get('enabled', True)
            self.ttl = config.get('ttl', 300)  # 默认 5 分钟
            self.max_size = config.get('max_size', 1000)  # 默认最多缓存 1000 条
            self.use_redis = config.get('use_redis', False)
            self.cache_type = config.get('cache_type', 'memory')  # 'memory' or 'redis'
            redis_host = config.get('redis_host', 'localhost')
            redis_port = config.get('redis_port', 6379)
            redis_db = config.get('redis_db', 0)
            redis_password = config.get('redis_password')

        # 内存缓存
        self.memory_cache: Dict[str, Any] = {}
        self.memory_cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }

        # Redis 连接(如果启用)
        self.redis_client = None
        if self.use_redis:
            try:
                import redis
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    password=redis_password,
                    decode_responses=True
                )
                # 测试连接
                self.redis_client.ping()
                logger.info("Redis 缓存连接成功")
            except Exception as e:
                logger.warning(f"Redis 连接失败: {e},使用内存缓存")
                self.use_redis = False

        if not self.use_redis:
            logger.info(f"内存缓存初始化完成 (TTL: {self.ttl}秒, 最大容量: {self.max_size})")

    def get_cache_key(self, data: Dict) -> str:
        """
        生成缓存键

        Args:
            data: 原始数据

        Returns:
            缓存键
        """
        # 将数据转换为字符串
        data_str = json.dumps(data, sort_keys=True)
        # 计算哈希值
        return hashlib.md5(data_str.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值

        Args:
            key: 缓存键

        Returns:
            缓存值或 None
        """
        if not self.enabled:
            return None

        start_time = time.time()

        # 尝试从 Redis 获取
        if self.use_redis and self.redis_client:
            try:
                cached_value = self.redis_client.get(key)
                if cached_value is not None:
                    self.memory_cache_stats['hits'] += 1
                    logger.debug(f"Redis 缓存命中: {key[:50]}...")
                    return json.loads(cached_value)
            except Exception as e:
                logger.warning(f"Redis 读取错误: {e}")

        # 尝试从内存缓存获取
        if key in self.memory_cache:
            cached_data = self.memory_cache[key]

            # 检查是否过期
            if time.time() - cached_data['timestamp'] < self.ttl:
                self.memory_cache_stats['hits'] += 1
                logger.debug(f"内存缓存命中: {key[:50]}...")
                return cached_data['value']
            else:
                # 已过期,删除
                del self.memory_cache[key]

        self.memory_cache_stats['misses'] += 1
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值

        Args:
            key: 缓存键
            value: 缓存值
            ttl: 过期时间(秒)
        """
        if not self.enabled:
            return

        # 检查缓存大小
        if len(self.memory_cache) >= self.max_size:
            # LRU 策略：删除最旧的条目
            oldest_key = min(self.memory_cache.keys(), key=lambda k: self.memory_cache[k]['timestamp'])
            del self.memory_cache[oldest_key]
            self.memory_cache_stats['evictions'] += 1
            logger.debug(f"缓存已满,淘汰条目: {oldest_key[:50]}...")

        # 设置 TTL
        expire_time = ttl if ttl is not None else self.ttl

        # 尝试设置到 Redis
        if self.use_redis and self.redis_client:
            try:
                self.redis_client.setex(key, expire_time, json.dumps(value))
                logger.debug(f"Redis 缓存写入: {key[:50]}...")
                return
            except Exception as e:
                logger.warning(f"Redis 写入错误: {e}")

        # 设置到内存缓存
        self.memory_cache[key] = {
            'value': value,
            'timestamp': time.time(),
            'expire_time': expire_time
        }
        logger.debug(f"内存缓存写入: {key[:50]}...")

    def invalidate(self, key: str = None) -> None:
        """
        使缓存失效

        Args:
            key: 缓存键(None 表示全部失效)
        """
        if key is None:
            # 清空所有缓存
            if self.use_redis and self.redis_client:
                self.redis_client.flushdb()
            self.memory_cache.clear()
            logger.info("缓存已清空")
        else:
            # 使特定条目失效
            if key in self.memory_cache:
                del self.memory_cache[key]
                logger.debug(f"缓存条目已失效: {key[:50]}...")

    def get_stats(self) -> Dict:
        """
        获取缓存统计信息

        Returns:
            统计信息
        """
        total_requests = self.memory_cache_stats['hits'] + self.memory_cache_stats['misses']
        hit_rate = self.memory_cache_stats['hits'] / total_requests * 100 if total_requests > 0 else 0

        return {
            'enabled': self.enabled,
            'use_redis': self.use_redis,
            'ttl': self.ttl,
            'max_size': self.max_size,
            'current_size': len(self.memory_cache),
            'hits': self.memory_cache_stats['hits'],
            'misses': self.memory_cache_stats['misses'],
            'evictions': self.memory_cache_stats['evictions'],
            'hit_rate': hit_rate,
            'requests': total_requests
        }

    def clear_stats(self) -> None:
        """清空统计信息"""
        self.memory_cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }


class CachedFunction:
    """带有缓存的装饰器"""

    def __init__(self, cache_manager: CacheManager, ttl: Optional[int] = None):
        """
        初始化缓存装饰器

        Args:
            cache_manager: 缓存管理器
            ttl: 自定义 TTL(秒)
        """
        self.cache_manager = cache_manager
        self.ttl = ttl

    def __call__(self, func):
        """应用装饰器"""

        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key_data = {
                'function': func.__name__,
                'args': args,
                'kwargs': kwargs
            }
            cache_key = self.cache_manager.get_cache_key(cache_key_data)

            # 尝试从缓存获取
            cached_value = self.cache_manager.get(cache_key)
            if cached_value is not None:
                return cached_value

            # 执行函数
            result = func(*args, **kwargs)

            # 设置缓存
            self.cache_manager.set(cache_key, result, ttl=self.ttl)

            return result

        return wrapper
