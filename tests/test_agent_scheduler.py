"""Tests for progeny.src.agent_scheduler."""
from __future__ import annotations

from progeny.src.agent_scheduler import AgentScheduler, ScheduledAgent


class TestSchedule:
    def test_all_active_npcs_scheduled(self):
        sched = AgentScheduler()
        roster = sched.schedule(["Lydia", "Belethor", "Ysolda"])
        assert len(roster) == 3
        assert [a.agent_id for a in roster] == ["Lydia", "Belethor", "Ysolda"]

    def test_all_tier_zero_in_phase1(self):
        sched = AgentScheduler()
        roster = sched.schedule(["Lydia", "Belethor"])
        assert all(a.tier == 0 for a in roster)

    def test_empty_npc_list(self):
        sched = AgentScheduler()
        roster = sched.schedule([])
        assert roster == []

    def test_max_agents_cap(self):
        """Respects max_agents_per_prompt from config."""
        sched = AgentScheduler()
        many_npcs = [f"NPC_{i}" for i in range(50)]
        roster = sched.schedule(many_npcs)
        assert len(roster) <= 16  # Default max from config


class TestTurnCounter:
    def test_increments_each_schedule(self):
        sched = AgentScheduler()
        assert sched.turn_counter == 0
        sched.schedule(["Lydia"])
        assert sched.turn_counter == 1
        sched.schedule(["Lydia"])
        assert sched.turn_counter == 2

    def test_empty_schedule_still_increments(self):
        sched = AgentScheduler()
        sched.schedule([])
        assert sched.turn_counter == 1


class TestTicksSinceLastAction:
    def test_new_agent_starts_at_zero(self):
        sched = AgentScheduler()
        roster = sched.schedule(["Lydia"])
        assert roster[0].ticks_since_last_action == 0

    def test_ticks_increment_each_turn(self):
        sched = AgentScheduler()
        sched.schedule(["Lydia"])
        roster = sched.schedule(["Lydia"])
        assert roster[0].ticks_since_last_action == 1
        roster = sched.schedule(["Lydia"])
        assert roster[0].ticks_since_last_action == 2

    def test_record_action_resets_ticks(self):
        sched = AgentScheduler()
        sched.schedule(["Lydia"])
        sched.schedule(["Lydia"])
        sched.record_action("Lydia")
        roster = sched.schedule(["Lydia"])
        # After record_action(0), then one schedule(+1) = 1
        assert roster[0].ticks_since_last_action == 1

    def test_ticks_tracked_independently_per_agent(self):
        sched = AgentScheduler()
        sched.schedule(["Lydia", "Belethor"])  # Turn 1: both new → 0
        sched.record_action("Lydia")            # Lydia reset → 0
        roster = sched.schedule(["Lydia", "Belethor"])  # Turn 2: both +1
        lydia = next(a for a in roster if a.agent_id == "Lydia")
        belethor = next(a for a in roster if a.agent_id == "Belethor")
        assert lydia.ticks_since_last_action == 1
        assert belethor.ticks_since_last_action == 1  # 0 (new) + 1 (turn 2)

    def test_remove_agent_stops_tracking(self):
        sched = AgentScheduler()
        sched.schedule(["Lydia"])
        sched.remove_agent("Lydia")
        # Re-adding starts fresh
        roster = sched.schedule(["Lydia"])
        assert roster[0].ticks_since_last_action == 0
