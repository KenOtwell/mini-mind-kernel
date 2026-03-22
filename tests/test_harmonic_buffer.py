"""Tests for progeny.src.harmonic_buffer."""
from __future__ import annotations

import numpy as np
import pytest

from shared.constants import EMOTIONAL_DIM, ZERO_SEMAGRAM
from progeny.src.harmonic_buffer import (
    EmotionalDelta,
    HarmonicBuffer,
    HarmonicConfig,
    HarmonicState,
    configure,
)


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset to default config after each test."""
    configure(HarmonicConfig())
    yield
    configure(HarmonicConfig())


def _angry_semagram() -> list[float]:
    """A semagram with high anger (axis 1)."""
    sem = [0.0] * EMOTIONAL_DIM
    sem[1] = 0.8  # anger
    sem[0] = 0.1  # slight fear
    sem[8] = 0.3  # some residual
    return sem


def _calm_semagram() -> list[float]:
    """A semagram with high safety (axis 7) and joy (axis 6)."""
    sem = [0.0] * EMOTIONAL_DIM
    sem[6] = 0.6  # joy
    sem[7] = 0.7  # safety
    sem[8] = 0.2  # low residual
    return sem


# ---------------------------------------------------------------------------
# HarmonicBuffer
# ---------------------------------------------------------------------------

class TestHarmonicBuffer:
    def test_zero_init(self):
        buf = HarmonicBuffer()
        assert buf.fast.tolist() == ZERO_SEMAGRAM
        assert buf.medium.tolist() == ZERO_SEMAGRAM
        assert buf.slow.tolist() == ZERO_SEMAGRAM
        assert not buf._initialized

    def test_first_update_sets_all_traces(self):
        """First delta IS initial values — all traces set directly."""
        buf = HarmonicBuffer()
        angry = _angry_semagram()
        delta = buf.update(angry)

        np.testing.assert_allclose(buf.fast, angry, atol=1e-6)
        np.testing.assert_allclose(buf.medium, angry, atol=1e-6)
        np.testing.assert_allclose(buf.slow, angry, atol=1e-6)
        assert buf._initialized

    def test_first_update_delta_is_full_semagram(self):
        buf = HarmonicBuffer()
        angry = _angry_semagram()
        delta = buf.update(angry)
        np.testing.assert_allclose(delta.delta, angry, atol=1e-6)

    def test_ema_decay_fast_reacts_more(self):
        """Fast trace should move more toward new input than slow."""
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        calm = _calm_semagram()
        buf.update(calm)

        # Fast (α=0.7) should be closer to calm than slow (α=0.1)
        fast_dist = np.linalg.norm(buf.fast - np.array(calm))
        slow_dist = np.linalg.norm(buf.slow - np.array(calm))
        assert fast_dist < slow_dist

    def test_curvature_positive_on_change(self):
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        delta = buf.update(_calm_semagram())
        assert delta.curvature > 0

    def test_curvature_zero_on_same_input(self):
        """Repeated same semagram should drive curvature toward zero."""
        buf = HarmonicBuffer()
        sem = _angry_semagram()
        buf.update(sem)
        # Several updates with same input — curvature should shrink
        for _ in range(20):
            delta = buf.update(sem)
        assert delta.curvature < 0.01

    def test_snap_detects_phase_change(self):
        """Snap should be large when curvature changes abruptly."""
        buf = HarmonicBuffer()
        sem = _angry_semagram()
        buf.update(sem)
        # Repeated calm — curvature settling
        for _ in range(5):
            buf.update(sem)
        # Sudden shift to calm — curvature spikes, snap should be large
        delta = buf.update(_calm_semagram())
        assert abs(delta.snap) > 0.01

    def test_lambda_coherent_after_init(self):
        """After first update, fast == slow, so λ should be 1.0."""
        buf = HarmonicBuffer()
        delta = buf.update(_angry_semagram())
        assert delta.lambda_t == pytest.approx(1.0, abs=0.01)

    def test_lambda_diverges_on_shift(self):
        """After a big emotional shift, fast diverges from slow → λ drops."""
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        # Shift to calm — fast moves toward calm, slow stays near angry
        delta = buf.update(_calm_semagram())
        assert delta.lambda_t < 0.99  # Some divergence

    def test_get_semagram(self):
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        sem = buf.get_semagram()
        assert len(sem) == EMOTIONAL_DIM
        assert sem == buf.fast.tolist()


# ---------------------------------------------------------------------------
# HarmonicState (multi-agent container)
# ---------------------------------------------------------------------------

class TestHarmonicState:
    def test_unknown_agent_returns_zero(self):
        state = HarmonicState()
        sem = state.get_semagram("Nobody")
        assert sem == ZERO_SEMAGRAM

    def test_update_creates_buffer(self):
        state = HarmonicState()
        delta = state.update("Lydia", _angry_semagram())
        assert isinstance(delta, EmotionalDelta)
        assert "Lydia" in state.agent_ids

    def test_multiple_agents_independent(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        state.update("Mikael", _calm_semagram())

        lydia_sem = state.get_semagram("Lydia")
        mikael_sem = state.get_semagram("Mikael")

        # Lydia should be angry-ish, Mikael calm-ish
        assert lydia_sem[1] > mikael_sem[1]  # anger
        assert mikael_sem[6] > lydia_sem[6]  # joy

    def test_reset_clears_all(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        state.update("Mikael", _calm_semagram())
        state.reset()
        assert state.agent_ids == []
        assert state.get_semagram("Lydia") == ZERO_SEMAGRAM

    def test_remove_agent(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        state.remove_agent("Lydia")
        assert state.get_semagram("Lydia") == ZERO_SEMAGRAM

    def test_get_delta_returns_none_for_unknown(self):
        state = HarmonicState()
        assert state.get_delta("Nobody") is None

    def test_get_delta_after_update(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        delta = state.get_delta("Lydia")
        assert delta is not None
        assert len(delta.semagram) == EMOTIONAL_DIM


# ---------------------------------------------------------------------------
# Emotional momentum scenario
# ---------------------------------------------------------------------------

class TestEmotionalMomentum:
    def test_multi_turn_convergence(self):
        """After many updates with the same semagram, traces converge."""
        buf = HarmonicBuffer()
        sem = _angry_semagram()
        for _ in range(50):
            buf.update(sem)

        np.testing.assert_allclose(buf.fast, sem, atol=0.01)
        np.testing.assert_allclose(buf.medium, sem, atol=0.01)
        np.testing.assert_allclose(buf.slow, sem, atol=0.01)

    def test_slow_trace_resists_change(self):
        """Slow trace should barely move after a single contrary input."""
        buf = HarmonicBuffer()
        angry = _angry_semagram()
        # Build up angry state over many turns
        for _ in range(20):
            buf.update(angry)
        slow_before = buf.slow.copy()
        # One calm input
        buf.update(_calm_semagram())
        slow_after = buf.slow
        # Slow should have moved very little
        drift = np.linalg.norm(slow_after - slow_before)
        assert drift < 0.15  # α=0.1 * one step
