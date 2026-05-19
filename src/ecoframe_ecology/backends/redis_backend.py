"""
RedisBackend: distributed brain registry over Redis.

Enables multi-machine brain populations — all workers register to the
same Redis instance. Zero caller code changes vs LocalRegistryBackend.

Install: pip install ecoframe-ecology[redis]

Usage:
    registry = BrainRegistry(backend='redis', url='redis://broker:6379')
    # Same API as LocalRegistryBackend — fully swappable
"""
from __future__ import annotations
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecoframe_ecology.registry import BrainEntry

_KEY_PREFIX = "ecoframe:brain:"
_EXPIRY_S   = 300   # brain entry expires after 5 min of no update


class RedisBackend:
    def __init__(self, url: str = 'redis://localhost:6379', **kwargs):
        try:
            import redis
        except ImportError:
            raise ImportError("redis required: pip install ecoframe-ecology[redis]")
        self._r = redis.from_url(url, decode_responses=True, **kwargs)

    def upsert(self, entry: 'BrainEntry') -> None:
        key  = f"{_KEY_PREFIX}{entry.brain_id}"
        data = {
            'brain_id': entry.brain_id,
            'ce_ema':   entry.ce_ema,
            'surprise': entry.surprise,
            'steps':    entry.steps,
            'env_id':   entry.env_id,
            'scale':    entry.scale,
            'load':     entry.load,
        }
        self._r.setex(key, _EXPIRY_S, json.dumps(data))

    def get(self, brain_id: str) -> 'BrainEntry | None':
        raw = self._r.get(f"{_KEY_PREFIX}{brain_id}")
        return _from_json(raw) if raw else None

    def all(self) -> list['BrainEntry']:
        keys = self._r.keys(f"{_KEY_PREFIX}*")
        pipe = self._r.pipeline()
        for k in keys:
            pipe.get(k)
        return [_from_json(v) for v in pipe.execute() if v]

    def remove(self, brain_id: str) -> None:
        self._r.delete(f"{_KEY_PREFIX}{brain_id}")

    def count(self) -> int:
        return len(self._r.keys(f"{_KEY_PREFIX}*"))


def _from_json(raw: str) -> 'BrainEntry':
    from ecoframe_ecology.registry import BrainEntry
    d = json.loads(raw)
    return BrainEntry(**d)
