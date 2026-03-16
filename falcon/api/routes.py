"""
Falcon API routes.

POST /comm.php — SKSE compatibility endpoint (the main interface)
GET  /health   — service health check
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Request, Response

from falcon.src.wire_protocol import parse_event, format_turn_response
from falcon.src.progeny_protocol import send_to_progeny
from shared.schemas import (
    EventPayload, GameEvent, PlayerState, WorldState, TurnResponse,
)
from shared.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Response queue — Falcon's ONE piece of ephemeral state
# ---------------------------------------------------------------------------
# SKSE polls with 'request' events. We queue Progeny results here.
# Lost on restart (acceptable — next turn regenerates).

_response_queue: deque[str] = deque()


# ---------------------------------------------------------------------------
# SKSE compatibility endpoint
# ---------------------------------------------------------------------------

@router.post("/comm.php")
@router.post("/{path:path}")  # Catch configurable AIAgent.ini paths
async def comm_endpoint(request: Request) -> Response:
    """
    SKSE compatibility endpoint.

    Accepts pipe-delimited wire format, returns CHIM wire format.
    This is the entry point that replaces HerikaServer's comm.php.
    """
    raw_body = (await request.body()).decode("utf-8", errors="replace")
    parsed = parse_event(raw_body)

    if parsed is None:
        return _empty()

    logger.debug("SKSE event: type=%s game_ts=%.1f trigger=%s local=%s",
                 parsed.event_type, parsed.game_ts,
                 parsed.is_turn_trigger, parsed.is_local)

    # --- SKSE polling for response ---
    if parsed.event_type == "request":
        return _dequeue_response()

    # --- Chat target not found (Falcon-local) ---
    if parsed.event_type == "chatnf":
        logger.warning("SKSE: NPC not found — %s", parsed.data[:80])
        return _empty()

    # --- Session end ---
    if parsed.is_session:
        logger.info("Session end signal (goodnight)")
        # TODO: notify Progeny to flush agent state
        return _empty()

    # --- Build EventPayload from parsed event ---
    payload = _build_payload(parsed)

    # --- Forward to Progeny ---
    if parsed.is_turn_trigger:
        # Async: fire in background, queue result for SKSE polling
        asyncio.create_task(_process_turn(payload))
    elif parsed.needs_forwarding:
        # Fire-and-forget: forward data event for accumulation
        asyncio.create_task(_forward_data_event(payload))

    return _empty()


# ---------------------------------------------------------------------------
# Payload construction
# ---------------------------------------------------------------------------

def _build_payload(parsed) -> EventPayload:
    """
    Build an EventPayload from a parsed SKSE event.

    TODO: This is currently minimal — needs integration with:
    - emotional_delta.py (compute 9d projection and delta)
    - embedding.py (embed event text)
    - memory_retrieval.py (retrieve memories on turn triggers)
    - SKSE metadata parsing (extract NPC positions, actor values, etc.)
    """
    return EventPayload(
        event_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        game_ts=parsed.game_ts,
        event=GameEvent(
            type=parsed.event_type,
            raw_data=parsed.data,
            source_agent="Player" if parsed.is_turn_trigger else None,
        ),
        is_turn_trigger=parsed.is_turn_trigger,
        player=PlayerState(
            position=[0.0, 0.0, 0.0],  # TODO: extract from SKSE
            cell="Unknown",              # TODO: extract from SKSE
            input_text=parsed.data if parsed.is_turn_trigger else None,
        ),
        world_state=WorldState(),
    )


# ---------------------------------------------------------------------------
# Async processing
# ---------------------------------------------------------------------------

async def _process_turn(payload: EventPayload) -> None:
    """Background: send turn to Progeny, process response, queue for SKSE."""
    try:
        turn_response = await send_to_progeny(payload)

        if turn_response and isinstance(turn_response, TurnResponse) and turn_response.responses:
            # TODO: Run bidirectional delta on each agent's utterance
            wire_output = format_turn_response(
                [r.model_dump() for r in turn_response.responses]
            )
            if wire_output:
                _response_queue.append(wire_output)
                logger.info("Turn response queued (%d agents, %d chars)",
                           len(turn_response.responses), len(wire_output))
        else:
            logger.debug("No turn response from Progeny (or empty)")

    except Exception:
        logger.exception("Failed to process turn")


async def _forward_data_event(payload: EventPayload) -> None:
    """Background: forward non-turn event to Progeny for accumulation."""
    try:
        await send_to_progeny(payload)
    except Exception:
        logger.debug("Failed to forward data event (non-critical)")


# ---------------------------------------------------------------------------
# Response queue management
# ---------------------------------------------------------------------------

def _dequeue_response() -> Response:
    """Dequeue the next pending response for SKSE, or return empty."""
    if _response_queue:
        wire_text = _response_queue.popleft()
        logger.debug("Dequeuing response (%d chars, %d remaining)",
                     len(wire_text), len(_response_queue))
        return Response(content=wire_text, media_type="text/plain")
    return _empty()


def _empty() -> Response:
    """Return empty response (SKSE interprets as 'nothing to do')."""
    return Response(content="", media_type="text/plain")


# ---------------------------------------------------------------------------
# Health / debug
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Health check — Falcon service status."""
    return {
        "status": "ok",
        "service": "falcon",
        "progeny_url": settings.progeny.base_url,
        "queue_depth": len(_response_queue),
    }
