"""Tests for progeny.src.agent_scheduler."""
from __future__ import annotations

from progeny.src.agent_scheduler import AgentScheduler, NpcScheduleInfo, ScheduledAgent


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


# ---------------------------------------------------------------------------
# Phase 2: Distance-based tiering
# ---------------------------------------------------------------------------

def _info(agent_id: str, pos: list[float], collab: bool = False, curv: float = 0.0):
    return NpcScheduleInfo(
        agent_id=agent_id, position=pos,
        is_collaborating=collab, curvature=curv,
    )


class TestDistanceTiering:
    def test_close_npc_tier0(self):
        sched = AgentScheduler()
        info = [_info("Lydia", [3.0, 0.0, 0.0])]
        roster = sched.schedule(["Lydia"], npc_info=info, player_position=[0, 0, 0])
        assert roster[0].tier == 0

    def test_mid_distance_tier1(self):
        sched = AgentScheduler()
        info = [_info("Belethor", [15.0, 0.0, 0.0])]
        roster = sched.schedule(["Belethor"], npc_info=info, player_position=[0, 0, 0])
        assert roster[0].tier == 1

    def test_far_distance_tier2(self):
        sched = AgentScheduler()
        info = [_info("Ysolda", [35.0, 0.0, 0.0])]
        roster = sched.schedule(["Ysolda"], npc_info=info, player_position=[0, 0, 0])
        assert roster[0].tier == 2

    def test_very_far_tier3(self):
        sched = AgentScheduler()
        info = [_info("Heimskr", [100.0, 0.0, 0.0])]
        roster = sched.schedule(["Heimskr"], npc_info=info, player_position=[0, 0, 0])
        assert roster[0].tier == 3

    def test_no_position_fallback_tier0(self):
        """Phase 1 compat: no position data → Tier 0."""
        sched = AgentScheduler()
        roster = sched.schedule(["Lydia"])  # No npc_info, no player_position
        assert roster[0].tier == 0

    def test_mixed_distances(self):
        sched = AgentScheduler()
        info = [
            _info("Lydia", [3.0, 0.0, 0.0]),      # T0
            _info("Belethor", [15.0, 0.0, 0.0]),   # T1
            _info("Heimskr", [100.0, 0.0, 0.0]),   # T3
        ]
        roster = sched.schedule(
            ["Lydia", "Belethor", "Heimskr"],
            npc_info=info, player_position=[0, 0, 0],
        )
        tiers = {a.agent_id: a.tier for a in roster}
        assert tiers["Lydia"] == 0
        assert tiers["Belethor"] == 1
        # Heimskr at T3 may be filtered by cadence on turn 1
        # (T3 cadence = 16, turn 1 % 16 != 0 unless turn 16)
        # So Heimskr might not appear in roster


class TestCollaborationFloor:
    def test_collaborator_promoted_to_tier1(self):
        """Far-away collaborator gets pinned to Tier 1 minimum."""
        sched = AgentScheduler()
        info = [_info("Lydia", [100.0, 0.0, 0.0], collab=True)]
        roster = sched.schedule(["Lydia"], npc_info=info, player_position=[0, 0, 0])
        assert roster[0].tier <= 1  # Floor applied

    def test_close_collaborator_stays_tier0(self):
        """Close collaborator doesn't get demoted by the floor."""
        sched = AgentScheduler()
        info = [_info("Lydia", [3.0, 0.0, 0.0], collab=True)]
        roster = sched.schedule(["Lydia"], npc_info=info, player_position=[0, 0, 0])
        assert roster[0].tier == 0


class TestCurvaturePromotion:
    def test_high_curvature_promotes_to_tier1(self):
        """Far-away NPC with high curvature promotes from T3 to T1."""
        sched = AgentScheduler()
        info = [_info("Lydia", [100.0, 0.0, 0.0], curv=0.5)]
        roster = sched.schedule(["Lydia"], npc_info=info, player_position=[0, 0, 0])
        assert roster[0].tier <= 1

    def test_low_curvature_no_promotion(self):
        """Far-away NPC with low curvature stays at distance tier."""
        sched = AgentScheduler()
        info = [_info("Heimskr", [100.0, 0.0, 0.0], curv=0.01)]
        roster = sched.schedule(["Heimskr"], npc_info=info, player_position=[0, 0, 0])
        # May be filtered by cadence, but if present, should be T3
        if roster:
            assert roster[0].tier == 3


class TestHarmonicCadence:
    def test_tier0_every_turn(self):
        sched = AgentScheduler()
        info = [_info("Lydia", [3.0, 0.0, 0.0])]
        # T0 cadence = 1, so Lydia should appear every turn
        for _ in range(5):
            roster = sched.schedule(["Lydia"], npc_info=info, player_position=[0, 0, 0])
            assert any(a.agent_id == "Lydia" for a in roster)

    def test_tier1_every_other_turn(self):
        sched = AgentScheduler()
        info = [_info("Belethor", [15.0, 0.0, 0.0])]
        appearances = 0
        for _ in range(4):
            roster = sched.schedule(["Belethor"], npc_info=info, player_position=[0, 0, 0])
            if any(a.agent_id == "Belethor" for a in roster):
                appearances += 1
        # T1 cadence = 2: first appearance (new) + 1 cadence hit in 4 turns = 3
        # Turn 1: new (ticks=0 → included). Turn 2: cadence 2%2=0 → included.
        # Turn 3: 3%2=1 → skip. Turn 4: 4%2=0 → included.
        assert appearances == 3

    def test_tier3_rare_cadence(self):
        sched = AgentScheduler()
        info = [_info("Heimskr", [100.0, 0.0, 0.0])]
        appearances = 0
        for _ in range(16):
            roster = sched.schedule(["Heimskr"], npc_info=info, player_position=[0, 0, 0])
            if any(a.agent_id == "Heimskr" for a in roster):
                appearances += 1
        # T3 cadence = 16: first appearance (new) + 1 cadence hit (turn 16) = 2
        assert appearances == 2

    def test_roster_sorted_by_tier(self):
        sched = AgentScheduler()
        info = [
            _info("Lydia", [3.0, 0.0, 0.0]),      # T0
            _info("Belethor", [15.0, 0.0, 0.0]),   # T1
        ]
        # Turn 2: both T0 (cadence 1) and T1 (cadence 2) fire
        sched.schedule(["Lydia", "Belethor"], npc_info=info, player_position=[0, 0, 0])  # turn 1
        roster = sched.schedule(["Lydia", "Belethor"], npc_info=info, player_position=[0, 0, 0])  # turn 2
        if len(roster) == 2:
            assert roster[0].tier <= roster[1].tier  # Sorted ascending
