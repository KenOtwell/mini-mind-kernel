"""NPC modulator construction — Creation Engine actor values → AxisModulators.

This module owns the vocabulary translation layer between Skyrim's
Creation Engine actor properties (Aggression, Confidence, Morality, Mood,
Assistance) and mindcore's domain-agnostic AxisModulators physics.

The physics (how emotional signals propagate through the harmonic buffer)
live in mindcore. The vocabulary (what 'Frenzied' or 'Foolhardy' means)
lives here.

Previously build_modulators() lived in mindcore.harmonic_buffer, which
violated the layer contract (mindcore should not know about Creation Engine).
It was moved here in the layer-hygiene refactor (August 2026).

See mindcore/ARCHITECTURE.md for the layering contract.
"""
from __future__ import annotations

from shared.constants import MOOD_TO_AXIS
from mindcore.harmonic_buffer import AxisModulators, DEFAULT_MOOD_PULL


def build_modulators(
    aggression: int = 0,
    confidence: int = 2,
    morality: int = 3,
    mood: int = 0,
    assistance: int = 0,
) -> AxisModulators:
    """Construct an AxisModulators from raw Creation Engine actor values.

    Translates the five Skyrim actor property integers into the generic
    physics parameters that mindcore's harmonic buffer understands.
    Integer ranges and NPC vocabulary are entirely contained here.

    Args:
        aggression: 0=Unaggressive .. 3=Frenzied
        confidence: 0=Cowardly .. 4=Foolhardy
        morality:   0=Any crime .. 3=No crime (used by response_expander;
                    not a buffer-physics parameter — carry separately if needed)
        mood:       0=Neutral .. 7=Disgusted (Creation Engine enum)
        assistance: 0=Nobody .. 2=Friends and allies (future cross-agent use;
                    not a buffer-physics parameter — carry separately if needed)

    Returns:
        AxisModulators with normalized physics parameters.
    """
    mood_axis = MOOD_TO_AXIS.get(mood)
    return AxisModulators(
        reactivity_gain=max(0.0, min(1.0, aggression / 3.0)),
        fear_dampening=max(0.0, min(1.0, confidence / 4.0)),
        mood_axis=mood_axis,
        mood_pull=DEFAULT_MOOD_PULL if mood_axis is not None else 0.0,
    )
