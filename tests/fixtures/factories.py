"""
Test data factories for the Many-Mind Kernel.

Each factory returns a realistic SKSE wire-format string or EventPayload
for a specific scenario. Used by both wire_protocol and round-trip tests.
"""
from __future__ import annotations

from uuid import uuid4
from datetime import datetime, timezone

from shared.schemas import (
    EventPayload, GameEvent, EmotionalState, AgentMemoryContext,
    MemorySummary, LoreHit, NpcMetadata, ActorValues,
    PlayerState, WorldState, CompressionTier,
    TurnResponse, AgentResponse, ActorValueDeltas, ActionCommand,
    ExtractionLevel,
)
from shared.constants import ZERO_SEMAGRAM


# ---------------------------------------------------------------------------
# Raw SKSE wire strings (what AIAgent.dll actually sends)
# ---------------------------------------------------------------------------

# Calm conversation — player talks to Lydia in Whiterun
WIRE_INPUTTEXT = "inputtext|1710624000|54321.0|What do you think about the civil war?"

# Speech input variant
WIRE_INPUTTEXT_S = "inputtext_s|1710624001|54321.5|I need your help with something"

# Game state events (NPC info, environment, location)
WIRE_INFO = "info|1710624002|54322.0|Lydia drew her weapon"
WIRE_INFONPC = "infonpc|1710624003|54323.0|Lydia|health:100|stamina:80"
WIRE_INFOLOC = "infoloc|1710624004|54324.0|WhiterunExterior|clear|afternoon"
WIRE_LOCATION = "location|1710624005|54325.0|WhiterunExterior"

# Narrative events
WIRE_CHAT = "chat|1710624006|54326.0|Belethor|Lydia|Do come back"
WIRE_DEATH = "death|1710624007|54327.0|Bandit was killed by Lydia"
WIRE_QUEST = "quest|1710624008|54328.0|BQ01|Started"
WIRE_BOOK = "book|1710624009|54329.0|The Book of the Dragonborn|Ancient tales..."

# Control events
WIRE_REQUEST = "request|1710624010|54330.0|"
WIRE_REQUEST_EMPTY = "request|1710624011|54331.0|"

# Session events
WIRE_GOODNIGHT = "goodnight|1710624012|54332.0|"

# Function return
WIRE_FUNCRET = "funcret|1710624013|54333.0|Follow|success"

# Error case
WIRE_CHATNF = "chatnf|1710624014|54334.0|NonExistentNPC"

# Edge cases
WIRE_MALFORMED_SHORT = "info|123"
WIRE_MALFORMED_EMPTY = ""
WIRE_MALFORMED_BAD_TS = "info|123|not_a_number|some data"
WIRE_DATA_WITH_PIPES = "info|1710624015|54335.0|Lydia says|she wants|to help"


# ---------------------------------------------------------------------------
# EventPayload factories
# ---------------------------------------------------------------------------

def make_turn_payload(
    input_text: str = "What do you think about the civil war?",
    npcs: dict[str, NpcMetadata] | None = None,
    game_ts: float = 54321.0,
) -> EventPayload:
    """Create a turn-trigger EventPayload (player typed something)."""
    if npcs is None:
        npcs = {"Lydia": _lydia_metadata()}
    return EventPayload(
        event_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        game_ts=game_ts,
        event=GameEvent(type="inputtext", raw_data=input_text, source_agent="Player"),
        is_turn_trigger=True,
        emotional_state={
            "Lydia": EmotionalState(
                base_vector=[0.1, 0.6, 0.2, 0.0, 0.3, 0.0, 0.1, 0.7, 0.4],
                delta=[0.0, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0, -0.1, 0.02],
                curvature=0.15,
                snap=0.03,
            ),
        },
        memory_context={
            "Lydia": AgentMemoryContext(
                retrieved_keys=["point-001", "point-002"],
                summaries=[
                    MemorySummary(text="Defended at Western Watchtower.", tier=CompressionTier.MOD, arc_id="arc-001"),
                ],
                lore_hits=[
                    LoreHit(topic="Civil War", content="The conflict between Imperial Legion and Stormcloaks..."),
                ],
            ),
        },
        npc_metadata=npcs,
        player=PlayerState(position=[101.0, 200.5, 50.0], cell="WhiterunExterior", input_text=input_text),
        world_state=WorldState(weather="clear", time_of_day=14.5),
        urgency=0.15,
    )


def make_data_payload(
    event_type: str = "info",
    data: str = "Lydia drew her weapon",
    game_ts: float = 54322.0,
) -> EventPayload:
    """Create a non-turn data event payload."""
    return EventPayload(
        event_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        game_ts=game_ts,
        event=GameEvent(type=event_type, raw_data=data),
        is_turn_trigger=False,
        player=PlayerState(position=[101.0, 200.5, 50.0], cell="WhiterunExterior"),
        world_state=WorldState(),
    )


def make_multi_npc_payload() -> EventPayload:
    """Create a turn payload with multiple NPCs (Whiterun market walk)."""
    return make_turn_payload(
        input_text="Hello everyone!",
        npcs={
            "Lydia": _lydia_metadata(),
            "Belethor": NpcMetadata(
                position=[105.0, 210.0, 50.0], cell="WhiterunExterior",
                actor_values=ActorValues(Aggression=0, Confidence=1, Mood=3),
            ),
            "Ysolda": NpcMetadata(
                position=[108.0, 205.0, 50.0], cell="WhiterunExterior",
                actor_values=ActorValues(Confidence=2, Mood=3),
            ),
            "Heimskr": NpcMetadata(
                position=[120.0, 215.0, 50.0], cell="WhiterunExterior",
                actor_values=ActorValues(Mood=1),
            ),
        },
    )


def make_combat_payload() -> EventPayload:
    """Create a high-urgency combat scenario."""
    return EventPayload(
        event_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        game_ts=60000.0,
        event=GameEvent(type="inputtext", raw_data="Watch out!", source_agent="Player"),
        is_turn_trigger=True,
        emotional_state={
            "Lydia": EmotionalState(
                base_vector=[0.8, 0.9, 0.1, 0.0, 0.7, 0.0, 0.0, 0.2, 0.6],
                delta=[0.3, 0.4, 0.0, 0.0, 0.3, 0.0, 0.0, -0.3, 0.1],
                curvature=0.85,
                snap=0.72,
            ),
        },
        npc_metadata={"Lydia": _lydia_metadata()},
        player=PlayerState(position=[50.0, 50.0, 10.0], cell="BleakFallsBarrow01", input_text="Watch out!"),
        world_state=WorldState(weather="clear", time_of_day=22.0),
        urgency=0.72,
    )


def make_quest_scene_payload() -> EventPayload:
    """Create a payload where an NPC is in a scripted quest scene."""
    return make_turn_payload(
        input_text="Lydia, what's going on?",
        npcs={
            "Lydia": NpcMetadata(
                position=[100.0, 200.0, 50.0], cell="WhiterunDragonsreach",
                in_scene=True,  # Quest-collision guard should activate
                is_follower=True,
                actor_values=ActorValues(Confidence=3, Mood=0),
            ),
        },
    )


def make_cell_transition_payload() -> EventPayload:
    """Create a cell transition event (entering a new area)."""
    return EventPayload(
        event_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        game_ts=70000.0,
        event=GameEvent(type="location", raw_data="WhiterunDragonsreach"),
        is_turn_trigger=False,
        player=PlayerState(position=[0.0, 0.0, 0.0], cell="WhiterunDragonsreach"),
        world_state=WorldState(cell_transition=True, reset=True),
    )


# ---------------------------------------------------------------------------
# Expected TurnResponse factories
# ---------------------------------------------------------------------------

def make_lydia_response() -> TurnResponse:
    """Expected stub response for Lydia."""
    return TurnResponse(
        event_id=uuid4(),  # Will be overridden in tests
        responses=[
            AgentResponse(
                agent_id="Lydia",
                utterance="I am sworn to carry your burdens.",
                actor_value_deltas=ActorValueDeltas(Confidence=3, Mood=0, Assistance=2),
                actions=[ActionCommand(command="Follow", target="Player")],
                extraction_level=ExtractionLevel.STRICT,
            ),
        ],
        processing_time_ms=0,
        model_used="stub-canned",
    )


# ---------------------------------------------------------------------------
# Expected CHIM wire output
# ---------------------------------------------------------------------------

# What Falcon should return to SKSE for Lydia's stub response
EXPECTED_WIRE_LYDIA = (
    "Lydia|dialogue|I am sworn to carry your burdens.\r\n"
    "Lydia|command|Follow@Player\r\n"
)

EXPECTED_WIRE_MULTI_NPC = (
    "Lydia|dialogue|I am sworn to carry your burdens.\r\n"
    "Lydia|command|Follow@Player\r\n"
    "Belethor|dialogue|Do come back.\r\n"
    "Ysolda|dialogue|I've been thinking about trading with the Khajiit caravans.\r\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lydia_metadata() -> NpcMetadata:
    return NpcMetadata(
        position=[100.5, 200.3, 50.0],
        cell="WhiterunExterior",
        level=25,
        hp=100.0, mp=50.0, sp=80.0,
        equipment=["Steel Sword", "Iron Shield", "Steel Armor"],
        actor_values=ActorValues(Aggression=1, Confidence=2, Morality=3, Mood=0, Assistance=2),
        is_follower=True,
    )
