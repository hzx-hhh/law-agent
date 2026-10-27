"""Redis adapter with an explicit in-memory fallback for local tests."""
from __future__ import annotations

import json
import os
import time
from typing import Any


class CacheBackend:
    def __init__(self) -> None:
        self.url = os.getenv("REDIS_URL")
        self._local: dict[str, tuple[float, Any]] = {}
        self._client = None
        if self.url:
            try:
                import redis
                self._client = redis.from_url(self.url, decode_responses=True)
            except ImportError:
                self._client = None

    def get(self, key: str) -> Any | None:
        if self._client:
            value = self._client.get(key)
            return json.loads(value) if value else None
        item = self._local.get(key)
        return item[1] if item and item[0] > time.time() else None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        if self._client:
            self._client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
        else:
            self._local[key] = (time.time() + ttl, value)

    def allow(self, key: str, limit: int, window: int = 60) -> bool:
        if self._client:
            count = self._client.incr(key)
            if count == 1:
                self._client.expire(key, window)
            return count <= limit
        return True
