"""
BrainRegistry: tracks active brains and routes them to environments.

Symmetric to the Field-based environment discovery:
  Environments publish EnvironmentSignal → brains discover them
  Brains publish BrainSignal → registry tracks population state

Routing logic (no designer curriculum):
  expected_learning(brain, env) = env.curiosity × (1 - env.load_fraction)
  When brain's ce_ema plateaus → route to highest expected_learning env

Backend is swappable — same routing logic works locally, across machines,
or against a SaaS registry service:

    registry = BrainRegistry(backend='local')          # development
    registry = BrainRegistry(backend='redis', url=...) # multi-machine
    registry = BrainRegistry(backend='saas',  url=...) # managed service
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ecoframe.field import Field
from ecoframe.signal import BrainSignal, CertSignal, EnvironmentSignal


@dataclass
class BrainEntry:
    """State snapshot for one registered brain."""
    brain_id:      str
    ce_ema:        float = 5.5
    ce_prev:       float = 5.5
    surprise:      float = 0.0
    steps:         int   = 0
    env_id:        str   = ""
    scale:         str   = ""
    load:          float = 1.0
    plateau_steps: int   = 0

    # Certification state
    certifications:   list = field(default_factory=list)  # earned cert names
    cert_attempts:    dict = field(default_factory=dict)  # cert_name → step when last attempted
    cert_retry_after: dict = field(default_factory=dict)  # cert_name → steps required before retry


class BrainRegistry:
    """
    Tracks brain instances and recommends environment routing.

    Backend contract (all backends implement the same interface):
        upsert(entry), get(brain_id), all(), remove(brain_id), count()

    Swapping backend = changing one constructor argument.
    """

    def __init__(
        self,
        field:           Field,
        backend:         str   = 'local',
        plateau_window:  int   = 500,     # steps before routing recommendation
        min_improvement: float = 0.05,    # ce_ema delta to count as learning
        verbose:         bool  = False,
        **backend_kwargs,
    ):
        self._field          = field
        self._plateau_window = plateau_window
        self._min_improvement = min_improvement
        self._verbose        = verbose
        self._step           = 0
        self._backend        = self._make_backend(backend, **backend_kwargs)

    # ── Registration ──────────────────────────────────────────────────────────

    def register(
        self,
        brain_id: str,
        scale:    str   = "",
        env_id:   str   = "",
    ) -> BrainEntry:
        """Register a new brain instance."""
        entry = BrainEntry(brain_id=brain_id, scale=scale, env_id=env_id)
        self._backend.upsert(entry)
        self._field.register_agent(brain_id, pos=(0.0, 0.0))
        if self._verbose:
            print(f"BrainRegistry: registered brain={brain_id} scale={scale}",
                  flush=True)
        return entry

    def deregister(self, brain_id: str) -> None:
        self._backend.remove(brain_id)

    # ── Update ────────────────────────────────────────────────────────────────

    def update(
        self,
        brain_id: str,
        ce_ema:   float,
        surprise: float = 0.0,
        steps:    int   = 0,
        env_id:   str   = "",
        load:     float = 1.0,
    ) -> 'str | None':
        """
        Update brain metrics. Publishes BrainSignal to Field every 100 steps.
        Returns recommended env_id if brain should switch, else None.
        """
        self._step += 1
        entry = self._backend.get(brain_id)
        if entry is None:
            return None

        # Plateau detection
        improvement = entry.ce_ema - ce_ema
        if improvement >= self._min_improvement:
            entry.plateau_steps = 0
        else:
            entry.plateau_steps += 1

        entry.ce_prev  = entry.ce_ema
        entry.ce_ema   = ce_ema
        entry.surprise = surprise
        entry.steps    = steps
        entry.env_id   = env_id
        entry.load     = load
        self._backend.upsert(entry)

        if self._step % 100 == 0:
            self._publish(entry)

        if entry.plateau_steps >= self._plateau_window:
            return self._recommend_env(brain_id, env_id)

        return None

    # ── Routing ───────────────────────────────────────────────────────────────

    def record_cert(self, brain_id: str, cert_signal: 'CertSignal') -> None:
        """
        Called when a CertSignal arrives in the Field for this brain.
        Updates certifications and records the attempt (for rate limiting).
        """
        entry = self._backend.get(brain_id)
        if entry is None:
            return
        # Record attempt + retry_after from the signal (for rate limiting)
        entry.cert_attempts[cert_signal.cert_name]    = entry.steps
        entry.cert_retry_after[cert_signal.cert_name] = cert_signal.retry_after_steps
        if cert_signal.passed >= 0.5:
            if cert_signal.cert_name not in entry.certifications:
                entry.certifications.append(cert_signal.cert_name)
                if self._verbose:
                    print(f"BrainRegistry: {brain_id} earned cert "
                          f"'{cert_signal.cert_name}' "
                          f"(score={cert_signal.score:.2f})", flush=True)
        else:
            if self._verbose:
                print(f"BrainRegistry: {brain_id} failed cert "
                      f"'{cert_signal.cert_name}' — retry after "
                      f"{cert_signal.retry_after_steps} steps", flush=True)
        self._backend.upsert(entry)

    def _recommend_env(self, brain_id: str, current_env_id: str) -> 'str | None':
        """
        Recommend the highest expected-learning environment for this brain.

        Filters out environments that:
          1. Are the brain's current env
          2. Require certs the brain hasn't earned
          3. Are cert envs the brain attempted too recently (rate limited)

        expected_learning = curiosity × (1 - load_fraction)
        """
        entry = self._backend.get(brain_id)
        if entry is None:
            return None

        env_sigs = []
        for s in self._field.query(pos=(0.0, 0.0), radius=100.0):
            if not isinstance(s, EnvironmentSignal):
                continue
            if s.publisher == current_env_id:
                continue

            # Check prerequisite certs
            if s.required_certs:
                required = [c for c in s.required_certs.split(',') if c]
                if not all(c in entry.certifications for c in required):
                    continue   # brain hasn't earned required certs

            # Check cert rate limit (applies to cert envs = those that issue certs)
            # A cert env's publisher ID appears in cert_attempts after an attempt
            last_attempt = entry.cert_attempts.get(s.publisher, -1)
            if last_attempt >= 0:
                retry_after = entry.cert_retry_after.get(s.publisher, 5000)
                if (entry.steps - last_attempt) < retry_after:
                    continue   # rate limited: hasn't earned enough steps since last attempt

            env_sigs.append(s)

        if not env_sigs:
            return None

        best = max(env_sigs,
                   key=lambda s: s.curiosity * (1.0 - min(s.load_fraction, 0.95)))

        if best.curiosity <= entry.ce_ema:
            return None  # current env still has more to teach

        if self._verbose:
            print(f"BrainRegistry: {brain_id} plateau={entry.plateau_steps} "
                  f"→ recommending '{best.publisher}' "
                  f"(curiosity={best.curiosity:.2f} "
                  f"certs={entry.certifications})", flush=True)
        return best.publisher

    # ── Population views ──────────────────────────────────────────────────────

    def all_brains(self) -> list[BrainEntry]:
        return self._backend.all()

    def count(self) -> int:
        return self._backend.count()

    def summary(self) -> dict:
        brains = self.all_brains()
        return {
            'count':       len(brains),
            'avg_ce_ema':  sum(b.ce_ema for b in brains) / max(1, len(brains)),
            'avg_steps':   sum(b.steps  for b in brains) / max(1, len(brains)),
            'envs':        list({b.env_id for b in brains if b.env_id}),
        }

    # ── Backend factory ───────────────────────────────────────────────────────

    def _publish(self, entry: BrainEntry) -> None:
        sig = BrainSignal(
            position       = (0.0, 0.0),
            timestamp      = self._step,
            publisher      = entry.brain_id,
            ce_ema         = entry.ce_ema,
            surprise       = entry.surprise,
            steps          = float(entry.steps),
            load           = entry.load,
            env_id         = entry.env_id,
            scale          = entry.scale,
            certifications = ','.join(entry.certifications),
        )
        self._field.publish(entry.brain_id, sig)

    @staticmethod
    def _make_backend(backend: str, **kwargs) -> Any:
        if backend == 'local':
            from ecoframe_ecology.backends.local import LocalRegistryBackend
            return LocalRegistryBackend()
        elif backend == 'redis':
            from ecoframe_ecology.backends.redis_backend import RedisBackend
            return RedisBackend(**kwargs)
        elif backend == 'saas':
            from ecoframe_ecology.backends.saas_backend import SaaSBackend
            return SaaSBackend(**kwargs)
        else:
            raise ValueError(
                f"Unknown registry backend: {backend!r}. "
                "Available: 'local'. Optional: 'redis', 'saas'."
            )
