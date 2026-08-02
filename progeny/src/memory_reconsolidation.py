"""Progeny shim -- re-exports mindcore.reconsolidation and wraps it with Neo defaults.

All logic lives in mindcore.reconsolidation (portable, client-injectable).
This module provides backward-compatible names so routes.py and tests need
no changes: run_reconsolidation() here calls the mindcore engine using
Progeny's get_client() singleton and COLLECTION_NPC_MEMORIES as defaults.
"""
from __future__ import annotations

from typing import Any, Callable, Optional
import logging

# Re-export all public symbols from the portable engine.
from mindcore.reconsolidation import (  # noqa: F401
    MODE_CONTENT, MODE_DCROSS,
    DEFAULT_TOP_K, DEFAULT_SCAN_LIMIT, DEFAULT_DISSONANCE_THRESHOLD,
    AGE_GAIN, DEFAULT_BLEND, DEFAULT_MAX_DRIFT, DEFAULT_HYSTERESIS,
    ProbeResult, ReconReport, ReconWriter, Memory,
    predicted_reaction, memory_dissonance, dcross_dissonance,
    select_dissonant, clamp_drift, eligible_to_reconsolidate,
    is_stalled_after, next_version, next_attempts,
    reconsolidated_point_id,
    run_reconsolidation as _run_reconsolidation_portable,
)

from shared.constants import COLLECTION_NPC_MEMORIES
from .memory_writer import MemoryWriter
from .qdrant_client import get_client

logger = logging.getLogger(__name__)


async def run_reconsolidation(
    agent_ids: list[str],
    slow_buffer_fn: Callable[[str], list[float]],
    *,
    writer: MemoryWriter,
    reinterpret_fn: Optional[Callable[[str], Any]] = None,
    collection: str = COLLECTION_NPC_MEMORIES,
    **kwargs,
) -> ReconReport:
    """Neo wrapper: delegates to mindcore engine with Progeny singleton client.

    Drop-in replacement -- all existing callers (routes.py) work unchanged.
    Pass collection= to target a different collection (e.g. for testing).
    """
    return await _run_reconsolidation_portable(
        agent_ids,
        slow_buffer_fn,
        writer=writer,
        client=get_client(),
        collection=collection,
        reinterpret_fn=reinterpret_fn,
        **kwargs,
    )
