"""
ecoframe-ecology: brain registry and multi-brain population coordination.

Backend is swappable — same routing logic across deployment targets:

    registry = BrainRegistry(backend='local')                   # single machine
    registry = BrainRegistry(backend='redis', url='redis://..') # multi-machine
    registry = BrainRegistry(backend='saas',  url='https://..')  # managed service

Usage:
    from ecoframe_ecology import BrainRegistry, BrainEntry
    from ecoframe import Field

    field    = Field(backend='local')
    registry = BrainRegistry(field=field, backend='local')

    registry.register('brain_0', scale='small', env_id='metadrive_roundabout')

    # Each training step:
    recommended_env = registry.update(
        'brain_0', ce_ema=2.3, surprise=0.4, steps=150_000,
        env_id='metadrive_roundabout')
    if recommended_env:
        # Brain has plateaued — MetaEnvironment should switch to recommended_env
        meta.navigate('brain_0', recommended_env)
"""
from ecoframe_ecology.registry import BrainRegistry, BrainEntry

__version__ = "0.1.0"
__all__ = ["BrainRegistry", "BrainEntry"]
