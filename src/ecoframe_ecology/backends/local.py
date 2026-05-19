"""
LocalRegistryBackend: in-process brain registry. Default backend.

All operations are in-memory, single-machine, zero network deps.
Swappable: replace with RedisBackend or NatsBackend for distributed
multi-machine populations — zero caller code changes.

Backend contract:
    upsert(entry: BrainEntry) → None        register / update a brain
    get(brain_id: str) → BrainEntry | None  read one brain's state
    all() → list[BrainEntry]                read all registered brains
    remove(brain_id: str) → None            deregister a brain
    count() → int                           number of registered brains
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecoframe_ecology.registry import BrainEntry


class LocalRegistryBackend:
    """Thread-safe in-memory registry. Default for single-machine training."""

    def __init__(self):
        self._store: dict[str, 'BrainEntry'] = {}

    def upsert(self, entry: 'BrainEntry') -> None:
        self._store[entry.brain_id] = entry

    def get(self, brain_id: str) -> 'BrainEntry | None':
        return self._store.get(brain_id)

    def all(self) -> list['BrainEntry']:
        return list(self._store.values())

    def remove(self, brain_id: str) -> None:
        self._store.pop(brain_id, None)

    def count(self) -> int:
        return len(self._store)
