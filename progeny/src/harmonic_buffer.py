"""
Harmonic buffer for Progeny — per-agent emotional state with temporal depth.

Each agent maintains three EMA (exponential moving average) traces over the
9d emotional semagram, at different time-scales:

  Fast  (α=0.7): reactive — captures immediate emotional shifts
  Medium (α=0.3): smoothed — captures mood trends across turns
  Slow  (α=0.1): glacial — captures personality-level drift

Derived signals:
  curvature: magnitude of delta between consecutive fast values.
             The priority gradient — how fast is emotion changing?
  snap:      change in curvature (2nd derivative).
             Event boundary detector — marks emotional phase transitions.
  λ(t):      cosine similarity between fast and slow traces.
             Emotional tension indicator — when low, the agent's immediate
             feelings diverge from their deep temperament. Tension wants
             to resolve, and that resolution drives emergent behavior.

Zero-init pattern: new agents start at ZERO_SEMAGRAM. The first emotional
update IS the initial value — no separate initialization needed.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from shared.constants import EMOTIONAL_DIM, ZERO_SEMAGRAM

logger = logging.getLogger(__name__)


@dataclass
class HarmonicConfig:
    """EMA decay rates for the three harmonic traces."""
    alpha_fast: float = 0.7
    alpha_medium: float = 0.3
    alpha_slow: float = 0.1


# Module-level default config
_config = HarmonicConfig()


def configure(config: HarmonicConfig) -> None:
    """Override the module-level harmonic config."""
    global _config
    _config = config


@dataclass
class EmotionalDelta:
    """Result of a harmonic buffer update — the emotional change signal.

    Passed to the prompt formatter so the LLM can calibrate response
    intensity. Maps to the EmotionalState schema fields.
    """
    semagram: list[float]    # Current 9d state (fast trace)
    delta: list[float]       # 9d change vector (new - previous fast)
    curvature: float         # Magnitude of delta (priority gradient)
    snap: float              # Change in curvature (event boundary)
    lambda_t: float          # Fast-slow coherence (tension indicator)


@dataclass
class HarmonicBuffer:
    """Per-agent emotional state with three EMA time-scales.

    All traces are 9d numpy arrays matching EMOTIONAL_AXES order.
    Zero-init: all traces start at zero. First update sets initial values.
    """
    fast: np.ndarray = field(default_factory=lambda: np.zeros(EMOTIONAL_DIM, dtype=np.float32))
    medium: np.ndarray = field(default_factory=lambda: np.zeros(EMOTIONAL_DIM, dtype=np.float32))
    slow: np.ndarray = field(default_factory=lambda: np.zeros(EMOTIONAL_DIM, dtype=np.float32))
    prev_curvature: float = 0.0
    _initialized: bool = False

    def update(self, new_semagram: list[float] | np.ndarray) -> EmotionalDelta:
        """Update all three EMA traces with a new 9d semagram.

        On first call, sets all traces to the new semagram directly
        (zero-init pattern: first delta IS initial values).

        Returns an EmotionalDelta with the change signals.
        """
        new = np.asarray(new_semagram, dtype=np.float32)
        prev_fast = self.fast.copy()

        if not self._initialized:
            # First observation: set all traces directly
            self.fast = new.copy()
            self.medium = new.copy()
            self.slow = new.copy()
            self._initialized = True
            delta = new  # First delta is the full semagram
        else:
            # EMA update: trace = α * new + (1 - α) * trace
            self.fast = _config.alpha_fast * new + (1 - _config.alpha_fast) * self.fast
            self.medium = _config.alpha_medium * new + (1 - _config.alpha_medium) * self.medium
            self.slow = _config.alpha_slow * new + (1 - _config.alpha_slow) * self.slow
            delta = self.fast - prev_fast

        # Curvature: magnitude of the delta (how fast is emotion changing?)
        curvature = float(np.linalg.norm(delta))

        # Snap: change in curvature (event boundary detector)
        snap = curvature - self.prev_curvature
        self.prev_curvature = curvature

        # λ(t): cosine similarity between fast and slow
        # High (~1.0) = coherent, low/negative = emotional tension
        lambda_t = _cosine_similarity(self.fast, self.slow)

        return EmotionalDelta(
            semagram=self.fast.tolist(),
            delta=delta.tolist(),
            curvature=curvature,
            snap=snap,
            lambda_t=lambda_t,
        )

    def get_semagram(self) -> list[float]:
        """Return the current emotional state (fast trace) as a list."""
        return self.fast.tolist()


class HarmonicState:
    """Container managing harmonic buffers for all agents.

    One HarmonicBuffer per agent_id. Agents are created on first
    encounter (zero-init). Survives across turns — emotional momentum
    persists until session reset.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, HarmonicBuffer] = {}

    def update(self, agent_id: str, new_semagram: list[float] | np.ndarray) -> EmotionalDelta:
        """Update an agent's harmonic buffer with a new semagram.

        Creates the buffer on first encounter (zero-init pattern).

        Args:
            agent_id: The agent to update.
            new_semagram: 9d emotional projection from text embedding.

        Returns:
            EmotionalDelta with change signals for prompt building.
        """
        buf = self._get_or_create(agent_id)
        return buf.update(new_semagram)

    def get_semagram(self, agent_id: str) -> list[float]:
        """Return an agent's current emotional state.

        Returns ZERO_SEMAGRAM if the agent has no buffer yet.
        """
        buf = self._buffers.get(agent_id)
        if buf is None:
            return list(ZERO_SEMAGRAM)
        return buf.get_semagram()

    def get_delta(self, agent_id: str) -> EmotionalDelta | None:
        """Return the last EmotionalDelta for an agent, or None."""
        buf = self._buffers.get(agent_id)
        if buf is None or not buf._initialized:
            return None
        # Reconstruct from current state (no new update)
        return EmotionalDelta(
            semagram=buf.fast.tolist(),
            delta=(buf.fast - buf.medium).tolist(),  # Approximate: fast-medium divergence
            curvature=buf.prev_curvature,
            snap=0.0,  # No new snap without a fresh update
            lambda_t=_cosine_similarity(buf.fast, buf.slow),
        )

    def reset(self) -> None:
        """Clear all buffers (session reset)."""
        self._buffers.clear()
        logger.info("Harmonic state reset — all agent buffers cleared")

    def remove_agent(self, agent_id: str) -> None:
        """Remove a specific agent's buffer."""
        self._buffers.pop(agent_id, None)

    @property
    def agent_ids(self) -> list[str]:
        """List of all tracked agent IDs."""
        return list(self._buffers.keys())

    def _get_or_create(self, agent_id: str) -> HarmonicBuffer:
        if agent_id not in self._buffers:
            self._buffers[agent_id] = HarmonicBuffer()
        return self._buffers[agent_id]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 for zero vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
