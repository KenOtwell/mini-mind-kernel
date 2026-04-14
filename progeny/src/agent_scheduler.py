"""
Many-Mind agent scheduler for Progeny.

Assigns tiers and filters agents for each turn's prompt based on:
  - Distance from player (concentric rings → tier assignment)
  - Collaboration status (active task, follower → tier floor)
  - Curvature-driven promotion (high emotional volatility → tier boost)
  - Harmonic cadence (tier determines how often an agent appears)

Tier 0 (Fundamental, ~5m):  every prompt, full agent block.
Tier 1 (1st Harmonic, ~20m): every 2nd prompt, abbreviated block.
Tier 2 (2nd Harmonic, ~50m): every 4th prompt, minimal block.
Tier 3+ (Higher Harmonics):  every 16th prompt, stub block.

Dispatch groups partition the roster into parallel LLM calls:
  - Solo groups: one Tier 0 agent per call (dedicated attention)
  - Batch groups: multiple agents sharing one call (efficient for
    lower tiers or when slots are scarce)
Each group becomes an independent prompt → LLM → expand pipeline.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from shared.config import settings

logger = logging.getLogger(__name__)

# Curvature threshold for tier promotion — agents with curvature above
# this get promoted to at least Tier 1 regardless of distance.
CURVATURE_PROMOTION_THRESHOLD = 0.2


@dataclass
class NpcScheduleInfo:
    """Input data for scheduling one NPC.

    Built by the caller (routes.py) from event accumulator data and
    harmonic state. The scheduler doesn't reach into pipeline state
    directly — it receives this clean interface.
    """
    agent_id: str
    position: Optional[list[float]] = None  # [x, y, z] or None if unknown
    is_collaborating: bool = False           # Follower, active task, etc.
    curvature: float = 0.0                   # From harmonic buffer


@dataclass
class ScheduledAgent:
    """An agent scheduled for this turn's prompt."""
    agent_id: str
    tier: int
    ticks_since_last_action: int


@dataclass
class DispatchGroup:
    """
    A group of agents dispatched to a single LLM call.

    Solo groups contain one Tier 0 agent (dedicated slot).
    Batch groups contain multiple agents sharing one call.
    """
    agents: list[ScheduledAgent]
    label: str  # e.g. "solo:Lydia", "batch:tier1", "batch:overflow"

    @property
    def agent_ids(self) -> list[str]:
        return [a.agent_id for a in self.agents]

    @property
    def is_solo(self) -> bool:
        return len(self.agents) == 1 and self.agents[0].tier == 0


class AgentScheduler:
    """Determines which agents appear in each turn's prompt and at what tier.

    Distance-based tiering with collaboration floor and curvature promotion.
    Harmonic cadence filtering ensures each tier appears at its natural
    frequency. The prompt context window is virtual memory for NPC minds;
    agents page in and out based on priority.
    """

    def __init__(self) -> None:
        self._turn_counter: int = 0
        # Per-agent tick count since last LLM output
        self._ticks_since_action: dict[str, int] = {}

    @property
    def turn_counter(self) -> int:
        return self._turn_counter

    def schedule(
        self,
        active_npc_ids: list[str],
        npc_info: list[NpcScheduleInfo] | None = None,
        player_position: list[float] | None = None,
    ) -> list[ScheduledAgent]:
        """Produce an ordered roster for this turn.

        Increments turn counter and ticks_since_last_action for all known
        agents. Assigns tiers by distance, applies collaboration floor and
        curvature promotion, then filters by harmonic cadence.

        Args:
            active_npc_ids: All NPCs in loaded cells.
            npc_info: Per-NPC metadata (position, collaboration, curvature).
                If None, falls back to all-Tier-0 (Phase 1 compat).
            player_position: Player's [x, y, z]. If None, all distances
                are treated as unknown (Tier 0 fallback).
        """
        self._turn_counter += 1
        cfg = settings.scheduler

        # Increment ticks for all known agents
        for agent_id in self._ticks_since_action:
            self._ticks_since_action[agent_id] += 1

        # Ensure all active NPCs are tracked
        for agent_id in active_npc_ids:
            if agent_id not in self._ticks_since_action:
                self._ticks_since_action[agent_id] = 0

        # Build info lookup
        info_map: dict[str, NpcScheduleInfo] = {}
        if npc_info is not None:
            info_map = {n.agent_id: n for n in npc_info}

        # Assign tiers for all active NPCs
        all_tiered: list[ScheduledAgent] = []
        for agent_id in active_npc_ids:
            info = info_map.get(agent_id)
            tier = _compute_tier(info, player_position, cfg)
            all_tiered.append(ScheduledAgent(
                agent_id=agent_id,
                tier=tier,
                ticks_since_last_action=self._ticks_since_action.get(agent_id, 0),
            ))

        # Harmonic cadence filter: only include agents whose tier cadence
        # aligns with the current turn counter.
        # Exception: new agents (ticks_since == 0) always appear on their
        # first turn — ensures the recognition bootstrap and first-encounter
        # pipeline always fire regardless of tier cadence.
        cadences = {
            0: cfg.tier0_cadence,
            1: cfg.tier1_cadence,
            2: cfg.tier2_cadence,
        }
        roster = []
        for agent in all_tiered:
            cadence = cadences.get(agent.tier, cfg.tier3_cadence)
            is_new = agent.ticks_since_last_action == 0
            if is_new or self._turn_counter % cadence == 0:
                roster.append(agent)

        # Cap total agents per prompt
        roster = roster[:cfg.max_agents_per_prompt]

        # Sort: Tier 0 first (highest priority), then by tier
        roster.sort(key=lambda a: a.tier)

        return roster

    def record_action(self, agent_id: str) -> None:
        """Reset ticks_since_last_action for an agent (LLM produced output)."""
        self._ticks_since_action[agent_id] = 0

    def plan_dispatch(self, roster: list[ScheduledAgent]) -> list[DispatchGroup]:
        """
        Partition a roster into dispatch groups for parallel LLM calls.

        Strategy:
          1. Tier 0 agents get solo groups (one agent per LLM call)
             up to max_parallel_slots.
          2. If more Tier 0 agents than slots, overflow goes to a batch.
          3. All non-Tier-0 agents are batched into one shared call.
          4. If only 1 agent total, still returns one group (no overhead).

        Returns groups in priority order: solo Tier 0 first, then batches.
        """
        if not roster:
            return []

        max_slots = settings.scheduler.max_parallel_slots

        # Separate by tier
        tier0 = [a for a in roster if a.tier == 0]
        lower_tiers = [a for a in roster if a.tier > 0]

        groups: list[DispatchGroup] = []

        # Solo dispatch for Tier 0 agents (up to slot limit)
        solo_count = min(len(tier0), max_slots)
        for agent in tier0[:solo_count]:
            groups.append(DispatchGroup(
                agents=[agent],
                label=f"solo:{agent.agent_id}",
            ))

        # Overflow Tier 0 agents that didn't get solo slots
        overflow = tier0[solo_count:]
        if overflow:
            groups.append(DispatchGroup(
                agents=overflow,
                label="batch:tier0-overflow",
            ))

        # Batch all lower-tier agents together
        if lower_tiers:
            groups.append(DispatchGroup(
                agents=lower_tiers,
                label="batch:lower-tiers",
            ))

        logger.debug(
            "Dispatch plan: %d groups (%d solo, %d batched) for %d agents",
            len(groups), solo_count,
            len(groups) - solo_count, len(roster),
        )

        return groups

    def remove_agent(self, agent_id: str) -> None:
        """Stop tracking an agent (left loaded cells, died, etc.)."""
        self._ticks_since_action.pop(agent_id, None)


# ---------------------------------------------------------------------------
# Tier computation
# ---------------------------------------------------------------------------

def _compute_tier(
    info: NpcScheduleInfo | None,
    player_position: list[float] | None,
    cfg: object,
) -> int:
    """Compute the scheduling tier for one NPC.

    Priority chain:
      1. Distance-based tier (concentric rings from player)
      2. Collaboration floor (min tier for active collaborators)
      3. Curvature-driven promotion (volatile agents promote up)

    If position data is unavailable, defaults to Tier 0 (Phase 1 compat).
    """
    if info is None or player_position is None or info.position is None:
        return 0  # No position data — Tier 0 fallback

    # Distance from player
    dist = _euclidean_distance(info.position, player_position)

    # Assign tier by distance thresholds
    if dist <= cfg.tier0_distance:
        tier = 0
    elif dist <= cfg.tier1_distance:
        tier = 1
    elif dist <= cfg.tier2_distance:
        tier = 2
    else:
        tier = 3

    # Collaboration floor: collaborators get at least this tier
    if info.is_collaborating:
        tier = min(tier, cfg.collaboration_floor_tier)

    # Curvature-driven promotion: volatile agents promote to at least Tier 1
    if info.curvature >= CURVATURE_PROMOTION_THRESHOLD and tier > 1:
        tier = 1
        logger.debug(
            "Curvature promotion: %s (curvature=%.3f) → Tier %d",
            info.agent_id, info.curvature, tier,
        )

    return tier


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """3D Euclidean distance between two position vectors."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))
