"""
Certification end-to-end tests.

Tests:
  1. CertSignal: tensor rejection, channels correct
  2. required_certs on EnvironmentSignal filters routing
  3. BrainRegistry.record_cert: pass earns cert, fail doesn't
  4. Rate limiting: brain blocked from cert env until retry_after_steps elapsed
  5. Prerequisite gate: brain without cert cannot be routed to gated env
  6. Prerequisite gate: brain WITH cert CAN be routed to gated env
  7. Multiple certs tracked independently
  8. BrainSignal includes certifications string
  9. Failed cert still records attempt (rate limits even on failure)
 10. Rate limit respects step count, not time
"""
import pytest
from ecoframe.field import Field
from ecoframe.signal import BrainSignal, CertSignal, EnvironmentSignal
from ecoframe_ecology import BrainRegistry, BrainEntry


def _env_signal(field, env_id, curiosity=3.0, load=0.1,
                required_certs="", pos=(2., 0.)):
    field.register_agent(env_id, pos=pos)
    field.publish(env_id, EnvironmentSignal(
        position=pos, timestamp=1, publisher=env_id,
        curiosity=curiosity, load_fraction=load,
        required_certs=required_certs,
    ))


def _cert_signal(field, brain_id, cert_name, passed=True, score=0.9,
                 retry_after=500):
    field.register_agent(f"cert_{cert_name}", pos=(5., 0.))
    sig = CertSignal(
        position=(5., 0.), timestamp=1,
        publisher=f"cert_{cert_name}",
        brain_id=brain_id, cert_name=cert_name,
        passed=1.0 if passed else 0.0,
        score=score, retry_after_steps=retry_after,
    )
    field.publish(f"cert_{cert_name}", sig)
    return sig


@pytest.fixture
def setup():
    field    = Field(backend='local')
    registry = BrainRegistry(field, backend='local',
                              plateau_window=5, verbose=False)
    registry.register("brain0", scale="small", env_id="metadrive")
    return field, registry


# ── CertSignal type ───────────────────────────────────────────────────────────

def test_cert_signal_rejects_tensor():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    sig = CertSignal(position=(0.,0.), timestamp=1, publisher="c0",
                     cert_name="test", passed=1.0)
    with pytest.raises(TypeError):
        sig.passed = torch.tensor(1.0)


def test_cert_signal_channels():
    sig = CertSignal(position=(0.,0.), timestamp=1, publisher="c0",
                     cert_name="test", passed=1.0, score=0.9)
    assert sig.R == pytest.approx(1.0 + 0.9)


def test_cert_signal_has_retry_after():
    sig = CertSignal(position=(0.,0.), timestamp=1, publisher="c0",
                     cert_name="test", passed=0.0, retry_after_steps=1000)
    assert sig.retry_after_steps == 1000


# ── record_cert ───────────────────────────────────────────────────────────────

def test_pass_earns_cert(setup):
    field, registry = setup
    sig = _cert_signal(field, "brain0", "driving_basics", passed=True)
    registry.record_cert("brain0", sig)
    entry = registry._backend.get("brain0")
    assert "driving_basics" in entry.certifications


def test_fail_does_not_earn_cert(setup):
    field, registry = setup
    sig = _cert_signal(field, "brain0", "driving_basics", passed=False)
    registry.record_cert("brain0", sig)
    entry = registry._backend.get("brain0")
    assert "driving_basics" not in entry.certifications


def test_fail_still_records_attempt(setup):
    """Rate limit applies even on failure — brain must train before retrying."""
    field, registry = setup
    sig = _cert_signal(field, "brain0", "driving_basics", passed=False)
    registry.record_cert("brain0", sig)
    entry = registry._backend.get("brain0")
    assert "driving_basics" in entry.cert_attempts


def test_pass_records_attempt(setup):
    field, registry = setup
    sig = _cert_signal(field, "brain0", "driving_basics", passed=True)
    registry.record_cert("brain0", sig)
    entry = registry._backend.get("brain0")
    assert "driving_basics" in entry.cert_attempts


# ── Rate limiting ─────────────────────────────────────────────────────────────

def test_rate_limit_blocks_cert_env_after_attempt(setup):
    """After a cert attempt, brain cannot be routed back until retry_after elapsed."""
    field, registry = setup
    # Publish cert env in field
    _env_signal(field, "cert_driving", curiosity=5.0)

    # Record a failed attempt — sets cert_attempts["cert_driving"] = 0
    sig = _cert_signal(field, "brain0", "cert_driving",
                       passed=False, retry_after=500)
    sig_entry = registry._backend.get("brain0")
    sig_entry.cert_attempts["cert_driving"] = 0   # attempted at step 0
    sig_entry.steps = 100  # only 100 steps since attempt
    registry._backend.upsert(sig_entry)

    # Plateau → should NOT recommend cert env (rate limited)
    rec = None
    for i in range(10):
        rec = registry.update("brain0", ce_ema=2.0, steps=100, env_id="metadrive")
    assert rec != "cert_driving", f"Should be rate limited, got {rec!r}"


def test_rate_limit_lifts_after_enough_steps(setup):
    """After retry_after_steps of training, brain can attempt cert again."""
    field, registry = setup
    # cert env has curiosity=5.0; brain's ce_ema must be higher for routing
    _env_signal(field, "cert_driving", curiosity=5.0)

    # Set up: attempted at step 0, now at step 600 (> retry_after=500)
    entry = registry._backend.get("brain0")
    entry.cert_attempts["cert_driving"]    = 0
    entry.cert_retry_after["cert_driving"] = 500   # retry_after from CertSignal
    entry.steps = 600
    entry.ce_ema = 2.0
    entry.ce_prev = 2.0
    entry.plateau_steps = 100
    registry._backend.upsert(entry)

    # One update with flat CE — plateau fires, rate limit lifted, cert recommended
    rec = registry.update("brain0", ce_ema=2.0, steps=600, env_id="metadrive")
    assert rec == "cert_driving", f"Should recommend cert env, got {rec!r}"


def test_rate_limit_is_step_based_not_time_based(setup):
    """Rate limit counts training steps, not wall-clock seconds."""
    field, registry = setup
    _env_signal(field, "cert_driving", curiosity=5.0)

    # Attempted 1 step ago — blocked regardless of how much real time passed
    entry = registry._backend.get("brain0")
    entry.cert_attempts["cert_driving"] = 99
    entry.steps = 100  # only 1 step since attempt
    registry._backend.upsert(entry)

    rec = None
    for _ in range(10):
        rec = registry.update("brain0", ce_ema=2.0, steps=100, env_id="metadrive")
    assert rec != "cert_driving"


# ── Prerequisite gating ───────────────────────────────────────────────────────

def test_gated_env_blocked_without_cert(setup):
    """Brain without required cert cannot be routed to gated environment."""
    field, registry = setup
    _env_signal(field, "advanced_env", curiosity=5.0,
                required_certs="driving_basics")

    rec = None
    for i in range(10):
        rec = registry.update("brain0", ce_ema=2.0, steps=i, env_id="metadrive")
    assert rec != "advanced_env", "Brain lacks required cert, should not be routed"


def test_gated_env_accessible_with_cert(setup):
    """Brain WITH required cert CAN be routed to gated environment."""
    field, registry = setup
    _env_signal(field, "advanced_env", curiosity=5.0,
                required_certs="driving_basics")

    # Grant the cert
    sig = _cert_signal(field, "brain0", "driving_basics", passed=True)
    registry.record_cert("brain0", sig)

    rec = None
    for i in range(10):
        rec = registry.update("brain0", ce_ema=2.0, steps=i, env_id="metadrive")
    assert rec == "advanced_env", f"Brain has cert, expected advanced_env, got {rec!r}"


def test_multiple_required_certs_all_needed(setup):
    """All required certs must be earned — one is not enough."""
    field, registry = setup
    _env_signal(field, "expert_env", curiosity=5.0,
                required_certs="driving_basics,lane_keeping")

    # Only earn one cert
    sig = _cert_signal(field, "brain0", "driving_basics", passed=True)
    registry.record_cert("brain0", sig)

    rec = None
    for i in range(10):
        rec = registry.update("brain0", ce_ema=2.0, steps=i, env_id="metadrive")
    assert rec != "expert_env", "Missing lane_keeping cert"


def test_multiple_required_certs_both_earned(setup):
    """Both certs earned → gated env accessible."""
    field, registry = setup
    _env_signal(field, "expert_env", curiosity=5.0,
                required_certs="driving_basics,lane_keeping")

    for cert in ["driving_basics", "lane_keeping"]:
        sig = _cert_signal(field, "brain0", cert, passed=True)
        registry.record_cert("brain0", sig)

    rec = None
    for i in range(10):
        rec = registry.update("brain0", ce_ema=2.0, steps=i, env_id="metadrive")
    assert rec == "expert_env"


# ── BrainSignal includes certs ────────────────────────────────────────────────

def test_brain_signal_includes_certifications(setup):
    """BrainSignal published to Field contains earned certifications."""
    field, registry = setup
    sig = _cert_signal(field, "brain0", "driving_basics", passed=True)
    registry.record_cert("brain0", sig)

    # Trigger publish (every 100 steps)
    for i in range(101):
        registry.update("brain0", ce_ema=2.0, steps=i, env_id="metadrive")

    brain_sigs = [s for s in field.query(pos=(0.,0.), radius=1.0)
                  if isinstance(s, BrainSignal)]
    assert len(brain_sigs) >= 1
    assert "driving_basics" in brain_sigs[0].certifications


# ── Multiple certs independent ────────────────────────────────────────────────

def test_multiple_certs_tracked_independently(setup):
    field, registry = setup
    for cert, passed in [("cert_a", True), ("cert_b", False), ("cert_c", True)]:
        sig = _cert_signal(field, "brain0", cert, passed=passed)
        registry.record_cert("brain0", sig)

    entry = registry._backend.get("brain0")
    assert "cert_a"   in entry.certifications
    assert "cert_b" not in entry.certifications
    assert "cert_c"   in entry.certifications
    assert "cert_a"   in entry.cert_attempts
    assert "cert_b"   in entry.cert_attempts   # failed but attempt recorded
    assert "cert_c"   in entry.cert_attempts
