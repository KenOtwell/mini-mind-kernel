"""
Canonical data schemas for the Many-Mind Kernel.

THE CONTRACT — shared types between Falcon and Progeny.
Both services import from here. If it's not in this file,
it's not part of the interface.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """SKSE event types from the wire protocol."""
    INPUTTEXT = "inputtext"
    INPUTTEXT_S = "inputtext_s"
    INFO = "info"
    INFONPC = "infonpc"
    INFOLOC = "infoloc"
    LOCATION = "location"
    CHAT = "chat"
    DEATH = "death"
    DIARY = "diary"
    QUEST = "quest"
    QUEST_JSON = "_quest"
    BOOK = "book"
    REQUEST = "request"
    FUNCRET = "funcret"
    GOODNIGHT = "goodnight"
    SPEECH = "_speech"
    FORCE_CURRENT_TASK = "force_current_task"
    CHATNF = "chatnf"


class CompressionTier(str, Enum):
    """Memory compression tiers in Qdrant."""
    RAW = "RAW"
    MOD = "MOD"
    MAX = "MAX"


class ExtractionLevel(str, Enum):
    """How cleanly the LLM response parsed."""
    STRICT = "strict"
    REPAIRED = "repaired"
    REGEX = "regex"
    PLAINTEXT = "plaintext"


class PrivacyLevel(str, Enum):
    """Memory access control levels."""
    PRIVATE = "PRIVATE"
    SEMI_PRIVATE = "SEMI_PRIVATE"
    COLLECTIVE = "COLLECTIVE"
    ANONYMOUS = "ANONYMOUS"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class GameEvent(BaseModel):
    """A single game event parsed from SKSE wire format."""
    type: str  # Kept as str for flexibility with unknown event types
    raw_data: str
    source_agent: Optional[str] = None


class EmotionalState(BaseModel):
    """Per-agent emotional state, computed by Falcon."""
    base_vector: list[float] = Field(..., min_length=9, max_length=9,
                                     description="Current 9d semagram")
    delta: list[float] = Field(..., min_length=9, max_length=9,
                               description="Change this tick")
    curvature: float = Field(0.0, description="1st derivative — priority gradient")
    snap: float = Field(0.0, description="2nd derivative — event boundary detector")


class MemorySummary(BaseModel):
    """A memory summary from retrieval."""
    text: str
    tier: CompressionTier
    arc_id: Optional[str] = None


class LoreHit(BaseModel):
    """A lore entry from Oghma retrieval."""
    topic: str
    content: str


class AgentMemoryContext(BaseModel):
    """Retrieved memory context for one agent."""
    retrieved_keys: list[str] = Field(default_factory=list)
    summaries: list[MemorySummary] = Field(default_factory=list)
    lore_hits: list[LoreHit] = Field(default_factory=list)


class ActorValues(BaseModel):
    """Current Creation Engine actor values for an NPC."""
    Aggression: int = Field(0, ge=0, le=3)
    Confidence: int = Field(2, ge=0, le=4)
    Morality: int = Field(3, ge=0, le=3)
    Mood: int = Field(0, ge=0, le=7)
    Assistance: int = Field(0, ge=0, le=2)


class NpcMetadata(BaseModel):
    """Per-NPC metadata from SKSE."""
    position: list[float] = Field(..., min_length=3, max_length=3)
    cell: str
    in_scene: bool = False
    level: int = 1
    hp: float = 100.0
    mp: float = 100.0
    sp: float = 100.0
    equipment: list[str] = Field(default_factory=list)
    actor_values: ActorValues = Field(default_factory=ActorValues)
    is_follower: bool = False
    active_task: Optional[str] = None


class PlayerState(BaseModel):
    """Current player state."""
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
                                  min_length=3, max_length=3)
    cell: str = "Unknown"
    input_text: Optional[str] = None


class WorldState(BaseModel):
    """World state delta."""
    weather: str = "clear"
    time_of_day: float = 12.0
    cell_transition: bool = False
    reset: bool = False


# ---------------------------------------------------------------------------
# Top-level payloads — THE CONTRACT
# ---------------------------------------------------------------------------

class EventPayload(BaseModel):
    """
    Falcon → Progeny: per-event payload.

    Sent to POST /ingest on every SKSE event (except Falcon-local ones).
    Turn triggers include memory_context; data events do not.
    """
    event_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    game_ts: float

    event: GameEvent
    is_turn_trigger: bool

    emotional_state: dict[str, EmotionalState] = Field(
        default_factory=dict,
        description="Keyed by agent_id. Only affected agents included.")
    memory_context: Optional[dict[str, AgentMemoryContext]] = Field(
        None,
        description="Only present when is_turn_trigger == true")
    npc_metadata: dict[str, NpcMetadata] = Field(
        default_factory=dict,
        description="All NPCs in loaded cells")
    player: PlayerState = Field(default_factory=PlayerState)
    world_state: WorldState = Field(default_factory=WorldState)
    urgency: float = Field(0.0, description="Max snap across active agents")


class ActionCommand(BaseModel):
    """A single action from the 43-command vocabulary."""
    command: str
    target: Optional[str] = None
    item: Optional[str] = None


class ActorValueDeltas(BaseModel):
    """LLM-proposed changes to Creation Engine actor values."""
    Aggression: Optional[int] = Field(None, ge=0, le=3)
    Confidence: Optional[int] = Field(None, ge=0, le=4)
    Morality: Optional[int] = Field(None, ge=0, le=3)
    Mood: Optional[int] = Field(None, ge=0, le=7)
    Assistance: Optional[int] = Field(None, ge=0, le=2)


class UpdatedHarmonics(BaseModel):
    """LLM-proposed emotional state update."""
    base_vector: list[float] = Field(..., min_length=9, max_length=9)


class NewMemory(BaseModel):
    """A memory the LLM wants to store."""
    text: str


class AgentResponse(BaseModel):
    """Response for a single agent from the LLM turn."""
    agent_id: str
    utterance: Optional[str] = None
    actor_value_deltas: Optional[ActorValueDeltas] = None
    actions: list[ActionCommand] = Field(default_factory=list)
    updated_harmonics: Optional[UpdatedHarmonics] = None
    new_memories: list[NewMemory] = Field(default_factory=list)
    extraction_level: ExtractionLevel = ExtractionLevel.STRICT


class TurnResponse(BaseModel):
    """
    Progeny → Falcon: per-turn response.

    Returned synchronously as HTTP response body to a turn-trigger EventPayload.
    """
    event_id: UUID
    turn_id: UUID = Field(default_factory=uuid4)
    responses: list[AgentResponse] = Field(default_factory=list)
    processing_time_ms: int = 0
    model_used: str = "stub"


class AckResponse(BaseModel):
    """
    Progeny → Falcon: ack for non-turn events.

    Returned immediately — Progeny accumulated the event.
    """
    event_id: UUID
    status: str = "accumulated"
