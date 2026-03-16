"""
Falcon → Progeny HTTP client.

Sends EventPayloads to POST /ingest, receives TurnResponse or AckResponse.
This is Falcon's only interface to Progeny — it does not know Progeny's internals.
"""
from __future__ import annotations

import logging
from typing import Optional, Union

import httpx

from shared.schemas import EventPayload, TurnResponse, AckResponse
from shared.config import settings

logger = logging.getLogger(__name__)

# Module-level reusable client
_client: Optional[httpx.AsyncClient] = None


async def get_client() -> httpx.AsyncClient:
    """Get or create the async HTTP client for Progeny."""
    global _client
    if _client is None or _client.is_closed:
        _client = httpx.AsyncClient(
            base_url=settings.progeny.base_url,
            timeout=settings.progeny.timeout_seconds,
        )
    return _client


async def send_to_progeny(
    payload: EventPayload,
) -> Optional[Union[TurnResponse, AckResponse]]:
    """
    Send an EventPayload to Progeny's /ingest endpoint.

    Returns:
        TurnResponse for turn triggers
        AckResponse for data events
        None on failure (graceful degradation — Falcon returns empty to SKSE)
    """
    client = await get_client()

    try:
        response = await client.post(
            "/ingest",
            json=payload.model_dump(mode="json"),
        )
        response.raise_for_status()
        data = response.json()

        if payload.is_turn_trigger:
            return TurnResponse.model_validate(data)
        return AckResponse.model_validate(data)

    except httpx.ConnectError:
        logger.warning("Progeny unreachable at %s — NPCs continue on engine AI",
                       settings.progeny.base_url)
        return None

    except httpx.TimeoutException:
        logger.warning("Progeny timeout (%.1fs) — NPCs continue on engine AI",
                       settings.progeny.timeout_seconds)
        return None

    except httpx.HTTPStatusError as exc:
        logger.warning("Progeny returned %d: %s", exc.response.status_code,
                       exc.response.text[:200])
        return None

    except Exception:
        logger.exception("Progeny communication error")
        return None


async def close() -> None:
    """Shut down the HTTP client cleanly."""
    global _client
    if _client and not _client.is_closed:
        await _client.aclose()
        _client = None
