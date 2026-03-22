"""Tests for parallel dispatch group partitioning."""
from __future__ import annotations

from unittest.mock import patch

from progeny.src.agent_scheduler import AgentScheduler, ScheduledAgent, DispatchGroup


def _agent(name: str, tier: int = 0, ticks: int = 0) -> ScheduledAgent:
    return ScheduledAgent(agent_id=name, tier=tier, ticks_since_last_action=ticks)


class TestPlanDispatch:
    def test_single_agent_one_group(self):
        sched = AgentScheduler()
        roster = [_agent("Lydia")]
        groups = sched.plan_dispatch(roster)
        assert len(groups) == 1
        assert groups[0].agent_ids == ["Lydia"]
        assert groups[0].label == "solo:Lydia"

    def test_two_tier0_two_solo_groups(self):
        sched = AgentScheduler()
        roster = [_agent("Lydia"), _agent("Belethor")]
        groups = sched.plan_dispatch(roster)
        assert len(groups) == 2
        assert groups[0].agent_ids == ["Lydia"]
        assert groups[1].agent_ids == ["Belethor"]
        assert all(g.is_solo for g in groups)

    def test_four_tier0_respects_max_slots(self):
        """With max_parallel_slots=4, 4 Tier 0 agents each get solo."""
        sched = AgentScheduler()
        roster = [_agent(n) for n in ["Lydia", "Belethor", "Ysolda", "Heimskr"]]
        groups = sched.plan_dispatch(roster)
        assert len(groups) == 4
        assert all(g.is_solo for g in groups)

    def test_overflow_beyond_max_slots(self):
        """5 Tier 0 agents with 4 slots: 4 solo + 1 overflow batch."""
        sched = AgentScheduler()
        roster = [_agent(n) for n in ["A", "B", "C", "D", "E"]]
        groups = sched.plan_dispatch(roster)
        assert len(groups) == 5  # 4 solo + 1 overflow
        solo_groups = [g for g in groups if g.label.startswith("solo:")]
        overflow_groups = [g for g in groups if "overflow" in g.label]
        assert len(solo_groups) == 4
        assert len(overflow_groups) == 1
        assert overflow_groups[0].agent_ids == ["E"]

    @patch("shared.config.settings.scheduler.max_parallel_slots", 2)
    def test_custom_max_slots(self):
        """Respects max_parallel_slots setting."""
        sched = AgentScheduler()
        roster = [_agent(n) for n in ["A", "B", "C", "D"]]
        groups = sched.plan_dispatch(roster)
        solo_groups = [g for g in groups if g.is_solo]
        assert len(solo_groups) == 2
        # C and D overflow to batch
        batch = [g for g in groups if not g.is_solo]
        assert len(batch) == 1
        assert sorted(batch[0].agent_ids) == ["C", "D"]

    def test_mixed_tiers(self):
        """Tier 0 get solo, lower tiers get batched together."""
        sched = AgentScheduler()
        roster = [
            _agent("Lydia", tier=0),
            _agent("Belethor", tier=1),
            _agent("Heimskr", tier=2),
            _agent("Nazeem", tier=3),
        ]
        groups = sched.plan_dispatch(roster)
        # 1 solo (Lydia) + 1 batch (lower tiers)
        assert len(groups) == 2
        assert groups[0].agent_ids == ["Lydia"]
        assert groups[0].is_solo
        assert sorted(groups[1].agent_ids) == ["Belethor", "Heimskr", "Nazeem"]
        assert "lower-tiers" in groups[1].label

    def test_empty_roster(self):
        sched = AgentScheduler()
        groups = sched.plan_dispatch([])
        assert groups == []

    def test_all_lower_tiers_one_batch(self):
        """No Tier 0 agents: everything goes in one batch."""
        sched = AgentScheduler()
        roster = [
            _agent("Belethor", tier=1),
            _agent("Heimskr", tier=2),
        ]
        groups = sched.plan_dispatch(roster)
        assert len(groups) == 1
        assert sorted(groups[0].agent_ids) == ["Belethor", "Heimskr"]


class TestDispatchGroupProperties:
    def test_agent_ids(self):
        g = DispatchGroup(
            agents=[_agent("Lydia"), _agent("Belethor")],
            label="batch:test",
        )
        assert g.agent_ids == ["Lydia", "Belethor"]

    def test_is_solo_true(self):
        g = DispatchGroup(agents=[_agent("Lydia", tier=0)], label="solo:Lydia")
        assert g.is_solo is True

    def test_is_solo_false_for_batch(self):
        g = DispatchGroup(
            agents=[_agent("Lydia"), _agent("Belethor")],
            label="batch:test",
        )
        assert g.is_solo is False

    def test_is_solo_false_for_non_tier0(self):
        g = DispatchGroup(agents=[_agent("Belethor", tier=1)], label="solo:Belethor")
        assert g.is_solo is False
