"""
Many-Mind agent scheduler for Progeny.

Assigns tiers and filters agents for each turn's prompt. Phase 1: all
active NPCs get Tier 0 (no distance data yet). Full distance-based
tiering, collaboration floor, and curvature-driven promotion come in
Phase 2 when Qdrant and harmonic buffers are wired up.

The scheduler tracks turn_counter (global) and ticks_since_last_action
(per agent) to support harmonic cadence filtering and LLM temporal
awareness.

Dispatch groups partition the roster into parallel LLM calls:
  - Solo groups: one Tier 0 agent per call (dedicated attention)
  - Batch groups: multiple agents sharing one call (efficient for
    lower tiers or when slots are scarce)
Each group becomes an independent prompt → LLM → expand pipeline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from shared.config import settings

logger = logging.getLogger(__name__)


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
    """
    Determines which agents appear in each turn's prompt and at what tier.

    Phase 1: all active NPCs → Tier 0, every turn.
    Phase 2: distance-based tiering, collaboration floor, curvature
    promotion, harmonic cadence filtering.
    """

    def __init__(self) -> None:
        self._turn_counter: int = 0
        # Per-agent tick count since last LLM output
        self._ticks_since_action: dict[str, int] = {}

    @property
    def turn_counter(self) -> int:
        return self._turn_counter

    def schedule(self, active_npc_ids: list[str]) -> list[ScheduledAgent]:
        """
        Produce an ordered roster for this turn.

        Increments turn counter and ticks_since_last_action for all known
        agents. Returns the roster of agents to include in the prompt.
        """
        self._turn_counter += 1

        # Increment ticks for all known agents
        for agent_id in self._ticks_since_action:
            self._ticks_since_action[agent_id] += 1

        # Ensure all active NPCs are tracked
        for agent_id in active_npc_ids:
            if agent_id not in self._ticks_since_action:
                # Zero-init: first appearance, no prior ticks
                self._ticks_since_action[agent_id] = 0

        # Phase 1: all active NPCs at Tier 0
        # Phase 2 will add: distance-based tier, collaboration floor,
        # curvature promotion, cadence filter
        max_agents = settings.scheduler.max_agents_per_prompt
        roster = []
        for agent_id in active_npc_ids[:max_agents]:
            roster.append(ScheduledAgent(
                agent_id=agent_id,
                tier=0,
                ticks_since_last_action=self._ticks_since_action.get(agent_id, 0),
            ))

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
