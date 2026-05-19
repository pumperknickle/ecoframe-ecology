# ecoframe-ecology

Brain registry and multi-brain population coordination for [ecoframe](https://github.com/pumperknickle/ecoframe).

Tracks active brain instances, monitors their learning progress, and recommends environment routing when a brain plateaus. Backend is swappable — same API works in-process, across machines via Redis, or against a managed registry service.

## Installation

```bash
pip install ecoframe-ecology
# optional backends
pip install ecoframe-ecology[redis]  # multi-machine populations
pip install ecoframe-ecology[saas]   # managed registry service
```

## BrainRegistry

```python
from ecoframe.field import Field
from ecoframe_ecology.registry import BrainRegistry

field    = Field()
registry = BrainRegistry(field, backend='local', verbose=True)

# Register a brain
entry = registry.register(brain_id='brain_0', scale='small')

# Update after each training step — returns recommended env_id or None
recommended = registry.update(
    brain_id='brain_0',
    ce_ema=4.2,
    surprise=0.3,
    steps=1000,
    env_id='roundabout',
)
if recommended:
    brain.exit(current_env)
    brain.enter(all_envs[recommended])
```

### Routing

When a brain's CE EMA hasn't improved by `min_improvement` for `plateau_window` steps, `BrainRegistry` queries the Field for `EnvironmentSignal`s and recommends the environment with the highest expected learning:

```
expected_learning(env) = env.curiosity × (1 - env.load_fraction)
```

Environments that require certifications the brain hasn't earned are filtered out. Cert environments the brain attempted too recently are rate-limited (step-based, not time-based — the brain must earn gradient steps before retrying).

### Certification

When a `CertSignal` arrives in the Field for a brain, call `registry.record_cert()` to update the brain's certification list and record the attempt for rate limiting. (MetaEnvironment does this automatically when wired to a registry.)

```python
registry.record_cert(brain_id='brain_0', cert_signal=sig)
# entry.certifications now includes sig.cert_name if passed
```

### Backends

| Backend | When to use |
|---------|-------------|
| `local` (default) | Single-machine development |
| `redis` | Multi-machine populations sharing a Redis instance |
| `saas` | Managed registry via REST API |

Swap backend — zero caller code changes:

```python
registry = BrainRegistry(field, backend='redis', url='redis://broker:6379')
```

Backend contract: `upsert(entry)`, `get(brain_id)`, `all()`, `remove(brain_id)`, `count()`.

### Population views

```python
registry.summary()
# {'count': 4, 'avg_ce_ema': 3.7, 'avg_steps': 52000, 'envs': ['roundabout', 'highway']}

registry.all_brains()   # list[BrainEntry]
registry.count()        # int
```

## BrainEntry

```python
@dataclass
class BrainEntry:
    brain_id:        str
    ce_ema:          float          # current smoothed CE loss
    steps:           int            # total training steps
    env_id:          str            # current environment
    certifications:  list[str]      # earned cert names
    cert_attempts:   dict           # cert_name → step of last attempt
    cert_retry_after: dict          # cert_name → steps required before retry
    plateau_steps:   int            # steps since last improvement
```

## Relation to ecoframe

`ecoframe-ecology` depends on `ecoframe`. The Field-based signal flow is symmetric:

- Environments publish `EnvironmentSignal` → brains discover available envs
- Brains publish `BrainSignal` → registry tracks population state
- Cert environments publish `CertSignal` → registry updates certifications

No central coordinator. Discovery and routing emerge from field gradients.
