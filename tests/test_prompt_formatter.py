"""Tests for progeny.src.prompt_formatter."""
from __future__ import annotations

import json

from shared.schemas import TypedEvent
from progeny.src.event_accumulator import AgentBuffer, TurnContext
from progeny.src.agent_scheduler import ScheduledAgent
from progeny.src.prompt_formatter import build_prompt, SYSTEM_PROMPT, INSTRUCTION_PROMPT


def _make_context(
    player_input: str = "What do you think about the civil war?",
    active_npc_ids: list[str] | None = None,
) -> TurnContext:
    if active_npc_ids is None:
        active_npc_ids = ["Lydia"]
    buffers = {}
    for npc in active_npc_ids:
        buffers[npc] = AgentBuffer(agent_id=npc)
    return TurnContext(
        player_input=player_input,
        agent_buffers=buffers,
        active_npc_ids=active_npc_ids,
        world_events=[],
        session_events=[],
    )


def _make_roster(agent_ids: list[str]) -> list[ScheduledAgent]:
    return [
        ScheduledAgent(agent_id=aid, tier=0, ticks_since_last_action=0)
        for aid in agent_ids
    ]


class TestMessageStructure:
    def test_two_messages(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        assert len(messages) == 2

    def test_message_roles(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_content(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        system = messages[0]["content"]
        assert "Many-Mind Kernel" in system
        assert "ACTOR VALUES" in system
        assert "Aggression" in system
        assert "RESPONSE FORMAT" in system

    def test_instruction_in_user_message(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        assert "valid JSON" in messages[1]["content"]


class TestDataPayload:
    def test_data_is_valid_json(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "agents" in data
        assert "player_input" in data
        assert "group_context" in data

    def test_group_context_has_present_npcs(self):
        ctx = _make_context(active_npc_ids=["Lydia", "Belethor"])
        roster = _make_roster(["Lydia", "Belethor"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "present_npcs" in data["group_context"]
        assert set(data["group_context"]["present_npcs"]) == {"Lydia", "Belethor"}

    def test_group_context_has_location(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "location" in data["group_context"]

    def test_player_input_included(self):
        ctx = _make_context("Tell me about dragons")
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert data["player_input"]["text"] == "Tell me about dragons"

    def test_agent_count_matches_roster(self):
        ctx = _make_context(active_npc_ids=["Lydia", "Belethor", "Ysolda"])
        roster = _make_roster(["Lydia", "Belethor", "Ysolda"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert len(data["agents"]) == 3


class TestAgentBlocks:
    def test_agent_block_has_required_fields(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert agent["agent_id"] == "Lydia"
        assert "tier" in agent
        assert "ticks_since_last_action" in agent
        assert "harmonic_state" in agent
        assert len(agent["harmonic_state"]["base_vector"]) == 9

    def test_agent_block_includes_dialogue_history(self):
        ctx = _make_context()
        ctx.agent_buffers["Lydia"].dialogue_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "I am sworn to carry your burdens."},
        ]
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert len(agent["dialogue_history"]) == 2

    def test_agent_block_includes_recent_events(self):
        ctx = _make_context()
        ctx.agent_buffers["Lydia"].events = [
            TypedEvent(
                event_type="_speech",
                local_ts="2024-01-01",
                game_ts=100.0,
                raw_data="I heard something",
            ),
        ]
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert len(agent["recent_events"]) == 1

    def test_ticks_since_last_action_propagated(self):
        ctx = _make_context()
        roster = [ScheduledAgent(agent_id="Lydia", tier=0, ticks_since_last_action=5)]
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert data["agents"][0]["ticks_since_last_action"] == 5


class TestFactPoolIntegration:
    def test_shared_facts_in_group_context(self):
        """Facts known by ALL present NPCs appear in group_context."""
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.add_fact("Dragon attacked", "event", 100.0, ["Player", "Lydia"])

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_knowledge" in gc
        assert "Dragon attacked" in gc["shared_knowledge"]

    def test_private_facts_in_agent_block(self):
        """Facts known by one NPC but not all go to the agent's private block."""
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        # Lydia knows a secret; Player doesn't
        pool.add_fact("Overheard Thalmor plot", "event", 100.0, ["Lydia"])
        # Both know this one (goes to shared, not private)
        pool.add_fact("Dragon attacked", "event", 101.0, ["Player", "Lydia"])

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert "private_knowledge" in agent
        assert "Overheard Thalmor plot" in agent["private_knowledge"]
        # Dragon attacked is shared, not private
        assert "Dragon attacked" not in agent.get("private_knowledge", [])

    def test_no_fact_pool_still_works(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)  # no fact_pool
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        # No private_knowledge key when no facts
        assert "private_knowledge" not in data["agents"][0]

    def test_lore_in_group_context(self):
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.bit_index.get_or_assign("Lydia")
        pool.add_lore("Skyrim is the homeland of the Nords")

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "lore" in gc
        assert "Skyrim" in gc["lore"][0]


class TestGroupTimeline:
    def test_shared_recent_in_group_context(self):
        """Group memory verbatim entries appear as shared_recent."""
        from progeny.src.event_accumulator import TieredMemory
        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            verbatim=[{"role": "Player", "content": "Watch out!"}],
        )
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_recent" in gc
        assert gc["shared_recent"][0]["content"] == "Watch out!"

    def test_shared_history_in_group_context(self):
        """Group memory compressed entries appear as shared_history."""
        from progeny.src.event_accumulator import TieredMemory
        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            compressed=["Player: Watch out! [combat]"],
        )
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_history" in gc
        assert "Watch out" in gc["shared_history"][0]

    def test_shared_anchors_in_group_context(self):
        """Group memory keywords appear as shared_anchors."""
        from progeny.src.event_accumulator import TieredMemory
        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            keywords=["Lydia:afraid | dragon | combat"],
        )
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_anchors" in gc
        assert "dragon" in gc["shared_anchors"][0]

    def test_empty_group_memory_omitted(self):
        """Empty group memory fields don't appear in the prompt."""
        ctx = _make_context()  # default TieredMemory is empty
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_recent" not in gc
        assert "shared_history" not in gc
        assert "shared_anchors" not in gc


class TestGroupDisplay:
    def test_group_display_shows_fast_buffer(self):
        """NPCs with emotional state show their fast buffer in group display."""
        from progeny.src.harmonic_buffer import HarmonicState
        state = HarmonicState()
        angry = [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
        state.update("Lydia", angry)

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, harmonic_state=state)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "group_display" in gc
        assert len(gc["group_display"]) == 1
        assert gc["group_display"][0]["name"] == "Lydia"
        assert len(gc["group_display"][0]["demeanor"]) == 9

    def test_zero_state_npcs_excluded_from_display(self):
        """NPCs with no emotional state yet don't appear in group display."""
        from progeny.src.harmonic_buffer import HarmonicState
        state = HarmonicState()  # No updates — all zero

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, harmonic_state=state)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        # group_display should be empty or absent
        display = data["group_context"].get("group_display", [])
        assert display == []
