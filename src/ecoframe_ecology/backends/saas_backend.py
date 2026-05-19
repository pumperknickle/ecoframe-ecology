"""
SaaSBackend: managed brain registry via REST API.

Enables a hosted registry service — multiple training clusters can
register their brains to the same service and share population state.

Install: pip install ecoframe-ecology[saas]

Usage:
    registry = BrainRegistry(
        backend='saas',
        url='https://registry.yourdomain.com',
        api_key='...',
    )

The service just needs to implement four endpoints:
    PUT  /brains/{brain_id}     upsert
    GET  /brains/{brain_id}     get one
    GET  /brains                list all
    DELETE /brains/{brain_id}   remove
"""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecoframe_ecology.registry import BrainEntry


class SaaSBackend:
    def __init__(self, url: str, api_key: str = "", **kwargs):
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install ecoframe-ecology[saas]")
        self._base    = url.rstrip('/')
        self._headers = {'Authorization': f'Bearer {api_key}'} if api_key else {}
        self._client  = httpx.Client(headers=self._headers, **kwargs)

    def upsert(self, entry: 'BrainEntry') -> None:
        self._client.put(
            f"{self._base}/brains/{entry.brain_id}",
            json={
                'brain_id': entry.brain_id,
                'ce_ema':   entry.ce_ema,
                'surprise': entry.surprise,
                'steps':    entry.steps,
                'env_id':   entry.env_id,
                'scale':    entry.scale,
                'load':     entry.load,
            },
        ).raise_for_status()

    def get(self, brain_id: str) -> 'BrainEntry | None':
        r = self._client.get(f"{self._base}/brains/{brain_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return _from_dict(r.json())

    def all(self) -> list['BrainEntry']:
        r = self._client.get(f"{self._base}/brains")
        r.raise_for_status()
        return [_from_dict(d) for d in r.json()]

    def remove(self, brain_id: str) -> None:
        self._client.delete(f"{self._base}/brains/{brain_id}")

    def count(self) -> int:
        return len(self.all())


def _from_dict(d: dict) -> 'BrainEntry':
    from ecoframe_ecology.registry import BrainEntry
    return BrainEntry(
        brain_id = d['brain_id'],
        ce_ema   = d.get('ce_ema', 5.5),
        surprise = d.get('surprise', 0.0),
        steps    = d.get('steps', 0),
        env_id   = d.get('env_id', ''),
        scale    = d.get('scale', ''),
        load     = d.get('load', 1.0),
    )
