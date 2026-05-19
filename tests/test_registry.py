"""
BrainRegistry tests.

Validates:
  1. register() creates BrainEntry with correct fields
  2. update() tracks metrics and detects plateau
  3. update() returns recommended env when brain plateaus
  4. update() returns None while brain is still learning
  5. BrainSignal published to Field after update
  6. all_brains() and summary() reflect registered state
  7. deregister() removes brain
  8. Backend swappable: LocalRegistryBackend API contract
  9. EnvironmentSignal required in Field for routing recommendation
 10. Brain does NOT get routed to its current environment
"""
import pytest
from ecoframe.field import Field
from ecoframe.signal import BrainSignal, EnvironmentSignal
from ecoframe_ecology import BrainRegistry, BrainEntry


@pytest.fixture
def field():
    return Field(backend='local')


@pytest.fixture
def registry(field):
    return BrainRegistry(field=field, plateau_window=5, min_improvement=0.1,
                         verbose=False)


def _publish_env_signal(field, env_id, curiosity=3.0, load=0.2):
    field.register_agent(env_id, pos=(2.0, 0.0))
    field.publish(env_id, EnvironmentSignal(
        position=(2.0, 0.0), timestamp=1, publisher=env_id,
        curiosity=curiosity, load_fraction=load, env_type="test",
    ))


# ── Registration ──────────────────────────────────────────────────────────────

def test_register_creates_entry(registry):
    entry = registry.register("brain_0", scale="small", env_id="metadrive")
    assert isinstance(entry, BrainEntry)
    assert entry.brain_id == "brain_0"
    assert entry.scale    == "small"
    assert entry.env_id   == "metadrive"


def test_register_appears_in_all_brains(registry):
    registry.register("brain_0")
    registry.register("brain_1")
    ids = [b.brain_id for b in registry.all_brains()]
    assert "brain_0" in ids
    assert "brain_1" in ids


def test_deregister_removes(registry):
    registry.register("brain_0")
    registry.deregister("brain_0")
    assert registry.count() == 0


# ── Metric update ─────────────────────────────────────────────────────────────

def test_update_tracks_ce_ema(registry):
    registry.register("brain_0")
    registry.update("brain_0", ce_ema=3.0, steps=1000)
    entry = registry._backend.get("brain_0")
    assert entry.ce_ema == pytest.approx(3.0)


def test_update_returns_none_while_learning(registry, field):
    registry.register("brain_0", env_id="env_a")
    _publish_env_signal(field, "env_b", curiosity=4.0)
    # Improving ce_ema → should not recommend switch
    for i in range(10):
        result = registry.update("brain_0", ce_ema=5.0 - i * 0.2, steps=i)
    assert result is None   # still improving


def test_update_returns_recommendation_on_plateau(registry, field):
    registry.register("brain_0", env_id="env_a")
    _publish_env_signal(field, "env_b", curiosity=4.0)
    # Flat ce_ema for plateau_window steps → should recommend env_b
    result = None
    for i in range(10):
        result = registry.update("brain_0", ce_ema=2.0, steps=i,
                                 env_id="env_a")
    assert result == "env_b"


def test_plateau_resets_on_improvement(registry, field):
    registry.register("brain_0", env_id="env_a")
    _publish_env_signal(field, "env_b", curiosity=4.0)
    # Plateau for a while
    for _ in range(4):
        registry.update("brain_0", ce_ema=2.0, steps=1, env_id="env_a")
    # Suddenly improve — plateau counter resets
    registry.update("brain_0", ce_ema=1.5, steps=5, env_id="env_a")
    # One more flat step — not enough for full plateau_window=5
    result = registry.update("brain_0", ce_ema=1.5, steps=6, env_id="env_a")
    assert result is None


def test_no_routing_to_current_env(registry, field):
    registry.register("brain_0", env_id="env_a")
    _publish_env_signal(field, "env_a", curiosity=5.0)  # high curiosity but current
    result = None
    for i in range(10):
        result = registry.update("brain_0", ce_ema=2.0, steps=i, env_id="env_a")
    assert result != "env_a"   # never route back to current env


def test_no_routing_when_no_better_env(registry, field):
    registry.register("brain_0", env_id="env_a")
    # Low curiosity env published — brain's ce_ema (2.0) > env curiosity (1.0)
    _publish_env_signal(field, "env_b", curiosity=1.0)
    result = None
    for i in range(10):
        result = registry.update("brain_0", ce_ema=2.0, steps=i, env_id="env_a")
    assert result is None   # no better env available


# ── Field integration ─────────────────────────────────────────────────────────

def test_brain_signal_published_to_field(registry, field):
    registry.register("brain_0", scale="small")
    # Trigger publish by updating 100 times
    for i in range(100):
        registry.update("brain_0", ce_ema=3.0, steps=i)
    sigs = field.query(pos=(0.0, 0.0), radius=1.0)
    brain_sigs = [s for s in sigs if isinstance(s, BrainSignal)]
    assert len(brain_sigs) >= 1
    assert brain_sigs[0].publisher == "brain_0"
    assert brain_sigs[0].scale == "small"


def test_brain_signal_has_correct_fields(registry, field):
    registry.register("brain_0", scale="small", env_id="metadrive")
    for i in range(100):
        registry.update("brain_0", ce_ema=2.5, surprise=0.3,
                        steps=1000, env_id="metadrive", load=1.0)
    sigs = [s for s in field.query(pos=(0.,0.), radius=1.)
            if isinstance(s, BrainSignal)]
    s = sigs[0]
    assert s.ce_ema   == pytest.approx(2.5)
    assert s.surprise == pytest.approx(0.3)
    assert s.env_id   == "metadrive"


# ── Population ────────────────────────────────────────────────────────────────

def test_summary_reflects_all_brains(registry):
    registry.register("b0", scale="small")
    registry.register("b1", scale="nano")
    registry.update("b0", ce_ema=2.0, steps=100, env_id="env_a")
    registry.update("b1", ce_ema=4.0, steps=50,  env_id="env_b")
    s = registry.summary()
    assert s['count'] == 2
    assert s['avg_ce_ema'] == pytest.approx(3.0)
    assert "env_a" in s['envs']


def test_count(registry):
    assert registry.count() == 0
    registry.register("b0")
    registry.register("b1")
    assert registry.count() == 2
    registry.deregister("b0")
    assert registry.count() == 1


# ── Backend swappability ──────────────────────────────────────────────────────

def test_backend_swappable_api():
    """LocalRegistryBackend fulfils the full backend contract."""
    from ecoframe_ecology.backends.local import LocalRegistryBackend
    b = LocalRegistryBackend()
    e = BrainEntry(brain_id="b0", ce_ema=3.0)
    b.upsert(e)
    assert b.get("b0").brain_id == "b0"
    assert b.count() == 1
    assert len(b.all()) == 1
    b.remove("b0")
    assert b.get("b0") is None
    assert b.count() == 0


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        BrainRegistry(Field(backend='local'), backend='blockchain')
