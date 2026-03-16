"""
Falcon API routes.

POST /comm.php — SKSE compatibility endpoint (the main interface)
GET  /health   — service health check
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Request, Response

from falcon.src.wire_protocol import parse_event, format_turn_response
from falcon.src.progeny_protocol import send_package
from falcon.src.event_parsers import parse_typed_data
from falcon.src.tick_accumulator import TickAccumulator
from shared.schemas import TypedEvent, TickPackage, TurnResponse
from shared.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# SKSE polls with 'request' events. We queue Progeny results here.
# Lost on restart (acceptable — next turn regenerates).
_response_queue: deque[str] = deque()

# Tick accumulator: initialised in startup(), torn down in shutdown().
_tick_accumulator: Optional[TickAccumulator] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def startup() -> None:
    """Initialise the tick accumulator (called from lifespan in server.py)."""
    global _tick_accumulator
    _tick_accumulator = TickAccumulator(
        interval_seconds=settings.falcon.tick_interval_seconds,
        on_package=_process_tick,
    )
    await _tick_accumulator.start()


async def shutdown() -> None:
    """Shut down the tick accumulator (called from lifespan in server.py)."""
    if _tick_accumulator is not None:
        await _tick_accumulator.stop()


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

    # --- Just-say: queue verbatim wire output for SKSE polling ---
    if parsed.event_type == "just_say":
        _response_queue.append(parsed.data)
        return _empty()

    # --- Session lifecycle ---
    if parsed.is_session:
        if parsed.event_type in ("init", "wipe", "playerdied"):
            if _tick_accumulator is not None:
                await _tick_accumulator.clear_npcs()
            logger.info("Session reset (%s) — NPC registry cleared", parsed.event_type)
        else:
            logger.info("Session signal: %s", parsed.event_type)
        return _empty()

    # --- All other events: decode structure, push to tick accumulator ---
    if _tick_accumulator is not None:
        event = TypedEvent(
            event_type=parsed.event_type,
            local_ts=datetime.now(timezone.utc).isoformat(),
            game_ts=parsed.game_ts,
            raw_data=parsed.data,
            parsed_data=parse_typed_data(parsed.event_type, parsed.data),
            is_turn_trigger=parsed.is_turn_trigger,
        )
        await _tick_accumulator.push(event)
    else:
        logger.warning("TickAccumulator not initialised — dropping %s event",
                       parsed.event_type)

    return _empty()


# ---------------------------------------------------------------------------
# Tick processing
# ---------------------------------------------------------------------------

async def _process_tick(package: TickPackage) -> None:
    """Send TickPackage to Progeny and queue any TurnResponse for SKSE polling."""
    try:
        result = await send_package(package)

        if result and isinstance(result, TurnResponse) and result.responses:
            # TODO: Run bidirectional delta on each agent's utterance
            wire_output = format_turn_response(
                [r.model_dump() for r in result.responses]
            )
            if wire_output:
                _response_queue.append(wire_output)
                logger.info("Turn response queued (%d agents, %d chars)",
                            len(result.responses), len(wire_output))
        else:
            logger.debug("No turn response from Progeny (or empty)")

    except Exception:
        logger.exception("Failed to process tick package")


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
    active_npcs = _tick_accumulator.get_active_npc_count() if _tick_accumulator else 0
    return {
        "status": "ok",
        "service": "falcon",
        "progeny_url": settings.progeny.base_url,
        "queue_depth": len(_response_queue),
        "active_npcs": active_npcs,
        "tick_interval_s": settings.falcon.tick_interval_seconds,
    }
