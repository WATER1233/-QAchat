import time
from unittest.mock import Mock, patch

import pytest

from tests import *  # noqa: F401, F403 — ensure sys.path
from api import LRUCache


class TestLRUCache:
    def test_put_and_get(self):
        cache = LRUCache(maxsize=10, ttl=300)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_get_nonexistent_key(self):
        cache = LRUCache(maxsize=10, ttl=300)
        assert cache.get("nonexistent") is None

    def test_get_expired_key(self):
        cache = LRUCache(maxsize=10, ttl=0.1)
        cache.put("key1", "value1")
        assert cache.get("key1") == "value1"
        time.sleep(0.15)
        assert cache.get("key1") is None

    def test_evicts_oldest_when_full(self):
        cache = LRUCache(maxsize=3, ttl=300)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")
        cache.put("d", "4")  # 应淘汰 a
        assert cache.get("a") is None
        assert cache.get("b") == "2"
        assert cache.get("c") == "3"
        assert cache.get("d") == "4"

    def test_access_refreshes_lru_order(self):
        cache = LRUCache(maxsize=3, ttl=300)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")
        # 访问 a，让 a 变成最近使用
        cache.get("a")
        cache.put("d", "4")  # 应淘汰 b（a 被刷新了）
        assert cache.get("a") == "1"
        assert cache.get("b") is None
        assert cache.get("d") == "4"

    def test_put_updates_existing_key(self):
        cache = LRUCache(maxsize=10, ttl=300)
        cache.put("key", "old")
        cache.put("key", "new")
        assert cache.get("key") == "new"

    def test_empty_cache_get_returns_none(self):
        cache = LRUCache(maxsize=10, ttl=300)
        assert cache.get("anything") is None

    def test_large_ttl_no_expiry(self):
        cache = LRUCache(maxsize=10, ttl=9999)
        cache.put("key", "value")
        assert cache.get("key") == "value"
