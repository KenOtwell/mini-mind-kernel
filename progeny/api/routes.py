"""
Progeny API routes.

POST /ingest  — the spine. Accepts TickPackage from Falcon, orchestrates
               the full cognitive pipeline, returns TurnResponse or AckResponse.
GET  /health  — liveness + LLM connectivity check.

Parallel dispatch: the scheduler partitions the roster into dispatch groups
(solo calls for Tier 0 agents, batch calls for lower tiers). Each group
runs its own prompt → LLM → expand pipeline concurrently via asyncio.gather.
All groups share the same system prompt + world state prefix for KV cache reuse.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Union
from uuid import uuid4

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from shared.schemas import (
    AckResponse,
    AgentResponse,
    LLMTimings,
    TickPackage,
    TurnResponse,
)
from shared.constants import COLLECTION_NPC_MEMORIES
from shared import qdrant_wrapper
from progeny.src.event_accumulator import EventAccumulator, TurnContext
from progeny.src.agent_scheduler import AgentScheduler, DispatchGroup
from progeny.src.fact_pool import FactPool
from progeny.src import prompt_formatter
from progeny.src import llm_client
from progeny.src.llm_client import GenerateResult, LLMError
from progeny.src import response_expander
from progeny.src.memory_compressor import slide_window
from progeny.src import emotional_delta
from progeny.src.harmonic_buffer import HarmonicState
from progeny.src.memory_writer import MemoryWriter
from progeny.src.memory_retrieval import MemoryRetriever, MemoryBundle
from progeny.src.compression import ArcCompressor, DEFAULT_SNAP_THRESHOLD
from progeny.src import qdrant_client as progeny_qdrant
from shared import embedding as shared_embedding
from shared import emotional as shared_emotional

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Pipeline state — lives for the process lifetime.
# Phase 2: these move to a proper application state container.
# ---------------------------------------------------------------------------
_fact_pool = FactPool()
_accumulator = EventAccumulator(fact_pool=_fact_pool)
_scheduler = AgentScheduler()
_harmonic_state = HarmonicState()
_memory_writer = MemoryWriter()
_memory_retriever = MemoryRetriever()
_arc_compressor = ArcCompressor(writer=_memory_writer)

# Reminding queue — one-tick-delayed retrieval results.
# Retrieval from tick N enters the prompt on tick N+1 (not N).
# This is the anti-recursion guard from the Living Doc: a memory that
# enters the prompt is immediately in the current context and excluded
# from future retrieval. Remindings can never trigger more retrieval
# in the same cycle because the output of retrieval is separated from
# the input of retrieval by exactly one tick.
_reminding_queue: dict[str, MemoryBundle] = {}

# Pipeline serialization lock
# through mutable state (_accumulator, _harmonic_state, _scheduler).
# Within a single tick, parallel dispatch groups still run concurrently
# via asyncio.gather (they only read shared state during LLM calls).
# Future: the pipelined context_manager/llm_executor split will allow
# overlapping Stage A (next tick's context) with Stage B (current LLM gen)
# while keeping state mutations serial.
_pipeline_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Recognition bootstrap — presence-change retrieval
# ---------------------------------------------------------------------------

async def _fire_recognition_retrieval(
    entered_npc_ids: list[str],
    all_active_npc_ids: list[str],
) -> None:
    """Fire referent-filtered retrieval when NPCs enter the scene.

    For each existing agent who was already present, retrieve memories
    involving each newcomer. Results go into _reminding_queue (one-tick
    delay) — the recognition surfaces on the next prompt, private to
    each agent. The newcomer's face triggers the recall; the agent's
    face may change (curvature spike) but the group context sees nothing.

    Lightweight: uses only the emotional axis (the agent's current state)
    with a referent filter. No semantic query needed — the face IS the
    query. Retrieval limit is small (3 memories per newcomer per agent).
    """
    global _reminding_queue

    if not shared_embedding.is_loaded() or not shared_emotional.is_loaded():
        return

    # Existing agents = everyone active EXCEPT the newcomers
    entered_set = set(entered_npc_ids)
    existing_agents = [npc for npc in all_active_npc_ids if npc not in entered_set]

    if not existing_agents:
        return

    recognition_count = 0
    for agent_id in existing_agents:
        emo_query = _harmonic_state.get_semagram(agent_id)
        # Dummy semantic query — recognition is emotion/referent driven.
        # Use the agent's own emotional state projected to 384d as a
        # semantic stand-in. In practice, the referent filter does the
        # heavy lifting; the semantic axis is secondary here.
        semantic_query = [0.0] * 384  # Neutral — referent filter dominates

        for newcomer_id in entered_npc_ids:
            try:
                bundle = await _memory_retriever.retrieve_for_agent(
                    agent_id=agent_id,
                    semantic_query=semantic_query,
                    emotional_query=emo_query,
                    lambda_t=0.8,  # Emotion-first — "this feels like that time..."
                    current_game_ts=time.time(),
                    referents=[newcomer_id],
                    broad_limit=10,
                    final_limit=3,  # Lightweight — just top recognition hits
                )
                if bundle.recent or bundle.summaries:
                    # Merge into existing reminding queue entry for this agent
                    existing = _reminding_queue.get(agent_id)
                    if existing is None:
                        _reminding_queue[agent_id] = bundle
                    else:
                        existing.recent.extend(bundle.recent)
                        existing.summaries.extend(bundle.summaries)
                        existing.expandable_refs.extend(bundle.expandable_refs)
                    recognition_count += 1
            except Exception as exc:
                logger.debug(
                    "Recognition retrieval failed for %s re: %s: %s",
                    agent_id, newcomer_id, exc,
                )

    if recognition_count:
        logger.info(
            "Recognition bootstrap: %d agents recalled memories of %d newcomers",
            recognition_count, len(entered_npc_ids),
        )


# ---------------------------------------------------------------------------
# Per-group pipeline: prompt → LLM → expand
# ---------------------------------------------------------------------------

async def _run_group(
    group: DispatchGroup,
    turn_context: TurnContext,
    all_active_npc_ids: list[str],
    emotional_deltas: dict | None = None,
    memory_bundles: dict[str, MemoryBundle] | None = None,
) -> tuple[list[AgentResponse], GenerateResult | None]:
    """
    Execute the full pipeline for one dispatch group.

    Returns (agent_responses, generate_result). On LLM error, returns
    empty responses and None result (graceful degradation per group).
    """
    messages = prompt_formatter.build_prompt(
        turn_context, group.agents, all_active_npc_ids,
        harmonic_state=_harmonic_state,
        emotional_deltas=emotional_deltas,
        fact_pool=_fact_pool,
        memory_bundles=memory_bundles,
    )

    try:
        result = await llm_client.generate(messages)
    except LLMError as exc:
        logger.error("Group %s: LLM error — %s", group.label, exc)
        # Graceful degradation: empty responses for this group
        empty = [AgentResponse(agent_id=aid) for aid in group.agent_ids]
        return empty, None

    agent_responses = response_expander.expand_response(
        result.content, group.agent_ids,
    )
    return agent_responses, result


def _aggregate_timings(results: list[GenerateResult | None]) -> LLMTimings:
    """
    Aggregate timing data across parallel dispatch groups.

    Tokens are summed (total work). Wall times use max (parallel overlap).
    """
    total_prompt_tok = 0
    total_prompt_ms = 0.0
    total_gen_tok = 0
    total_gen_ms = 0.0
    total_cache_tok = 0

    for r in results:
        if r is None:
            continue
        total_prompt_tok += r.prompt_tokens
        total_prompt_ms = max(total_prompt_ms, r.prompt_ms)
        total_gen_tok += r.generated_tokens
        total_gen_ms = max(total_gen_ms, r.generation_ms)
        total_cache_tok += r.cache_tokens

    prompt_tps = (total_prompt_tok / (total_prompt_ms / 1000.0)) if total_prompt_ms > 0 else 0.0
    gen_tps = (total_gen_tok / (total_gen_ms / 1000.0)) if total_gen_ms > 0 else 0.0

    return LLMTimings(
        prompt_tokens=total_prompt_tok,
        prompt_ms=total_prompt_ms,
        prompt_tokens_per_sec=round(prompt_tps, 1),
        generated_tokens=total_gen_tok,
        generation_ms=total_gen_ms,
        generation_tokens_per_sec=round(gen_tps, 1),
        cache_tokens=total_cache_tok,
    )


@router.post("/ingest", response_model=Union[TurnResponse, AckResponse])
async def ingest(package: TickPackage) -> TurnResponse | AckResponse:
    """
    Ingest a TickPackage from Falcon.

    Pipeline: accumulate → detect turn → schedule → dispatch groups →
    parallel (prompt → LLM → expand) → merge → return.

    Serialized via _pipeline_lock: concurrent WebSocket ticks wait rather
    than racing through mutable harmonic/accumulator/scheduler state.
    Parallel dispatch groups within a single tick are unaffected.
    """
    async with _pipeline_lock:
        return await _ingest_inner(package)


async def _ingest_inner(package: TickPackage) -> TurnResponse | AckResponse:
    """Inner pipeline — runs under _pipeline_lock."""
    tick_id = package.tick_id

    # Step 1: Accumulate events, detect turn boundary
    turn_context = _accumulator.ingest(package)

    if turn_context is None:
        logger.debug("Tick %s: data-only, accumulated %d events", tick_id, len(package.events))
        return AckResponse(tick_id=tick_id)

    # --- Turn trigger detected — full pipeline ---
    global _reminding_queue
    start_ms = time.monotonic()
    logger.info("Tick %s: turn trigger — player: %s", tick_id, turn_context.player_input[:80])

    # Consume last tick's remindings — these become this tick's memory context.
    # The one-tick delay is the anti-recursion guard: retrieval results from
    # tick N appear in the prompt on tick N+1, never on tick N itself.
    prior_remindings = _reminding_queue
    _reminding_queue = {}  # Clear for this tick's fresh retrieval
    if prior_remindings:
        logger.info(
            "Tick %s: injecting %d prior remindings into prompt",
            tick_id, len(prior_remindings),
        )

    # Record player input in dialogue history for all active agents
    _accumulator.record_player_input(turn_context.player_input)

    # Emotional pipeline: inbound text → 9d projection → update harmonic buffers
    emotional_delta.process_inbound(turn_context, _harmonic_state)

    # Recognition bootstrap — presence-change retrieval trigger.
    # When an NPC enters the scene, existing agents fire referent-filtered
    # retrieval against the newcomer's ID. "Walk into a room, see an old
    # friend, the last few times you were together pop into your head."
    # Results go to _reminding_queue (one-tick delay) — private, Layer 2.
    if turn_context.presence_changes.entered:
        await _fire_recognition_retrieval(
            turn_context.presence_changes.entered,
            turn_context.active_npc_ids,
        )

    # Persist inbound events to Qdrant via enrichment wrapper (RAW writes)
    # Each event text gets embedded, projected to 9d, and stored with dual vectors.
    # Non-blocking: failures are logged but don't stop the turn.
    try:
        qdrant_cli = progeny_qdrant.get_client()
        for agent_id, buf in turn_context.agent_buffers.items():
            for event in buf.events:
                if event.raw_data:
                    await qdrant_wrapper.ingest(
                        client=qdrant_cli,
                        text=event.raw_data,
                        collection=COLLECTION_NPC_MEMORIES,
                        agent_id=agent_id,
                        game_ts=event.game_ts,
                        event_type=event.event_type,
                    )
    except Exception as exc:
        logger.warning("RAW write pass failed (non-fatal): %s", exc)

    # Step 2: Schedule agents
    roster = _scheduler.schedule(turn_context.active_npc_ids)
    if not roster:
        logger.warning("Tick %s: no agents to schedule", tick_id)
        return TurnResponse(tick_id=tick_id, processing_time_ms=0, model_used="none")

    all_agent_ids = [a.agent_id for a in roster]
    logger.info("Tick %s: scheduled %d agents: %s", tick_id, len(roster), all_agent_ids)

    # Capture current emotional deltas for prompt building
    emotional_deltas = {
        a.agent_id: _harmonic_state.get_delta(a.agent_id)
        for a in roster
    }

    # Qdrant-backed memory retrieval — one-tick-delayed reminding protocol.
    #
    # Retrieval results go into _reminding_queue for the NEXT tick's prompt,
    # not this tick's. This tick's prompt uses prior_remindings (last tick's
    # retrieval). The one-tick delay prevents retrieval → prompt → retrieval
    # recursion: a memory in the prompt is in the current context and can
    # never be "re-remembered" because retrieval output is separated from
    # retrieval input by exactly one tick boundary.
    #
    # Semantic query: player's input embedding (what was said).
    # Emotional query: agent's current delta (shift direction) for
    # shift-congruent retrieval — "find events whose emotional content
    # matches the direction I'm currently moving."
    # NOTE: subject/object perspective inversion is an open research issue.
    try:
        if shared_embedding.is_loaded() and shared_emotional.is_loaded():
            player_emb = shared_embedding.embed_one(turn_context.player_input)
            semantic_query = player_emb.tolist()
            for agent in roster:
                agent_id = agent.agent_id
                delta = emotional_deltas.get(agent_id)
                lambda_t = delta.lambda_t if delta else 0.5
                # Shift-congruent: use delta when curvature is meaningful,
                # fall back to absolute state in calm periods.
                if delta and delta.curvature > 0.01:
                    emo_query = delta.delta
                else:
                    emo_query = _harmonic_state.get_semagram(agent_id)
                bundle = await _memory_retriever.retrieve_for_agent(
                    agent_id=agent_id,
                    semantic_query=semantic_query,
                    emotional_query=emo_query,
                    lambda_t=lambda_t,
                    current_game_ts=time.time(),
                    referents=turn_context.active_npc_ids,
                )
                # Store in reminding queue for NEXT tick (not this prompt)
                _reminding_queue[agent_id] = bundle
            logger.info(
                "Tick %s: retrieved %d bundles → reminding queue (next tick)",
                tick_id, len(_reminding_queue),
            )
    except Exception as exc:
        logger.warning("Memory retrieval failed (non-fatal): %s", exc)

    # This tick's prompt uses PRIOR remindings (last tick's retrieval),
    # not the fresh retrieval we just stored in _reminding_queue.
    memory_bundles: dict[str, MemoryBundle] = prior_remindings

    # Merge recognition remindings into memory_bundles if any exist.
    # Recognition retrieval fired earlier in this tick goes into
    # _reminding_queue. These will surface on the NEXT tick like all
    # other remindings. No special handling needed — the queue is unified.

    # Step 3: Partition into dispatch groups
    groups = _scheduler.plan_dispatch(roster)
    logger.info(
        "Tick %s: dispatching %d groups: %s",
        tick_id, len(groups), [g.label for g in groups],
    )

    # Step 4: Parallel execution — all groups run concurrently
    group_tasks = [
        _run_group(group, turn_context, turn_context.active_npc_ids,
                   emotional_deltas, memory_bundles)
        for group in groups
    ]
    group_results = await asyncio.gather(*group_tasks)

    # Step 5: Merge results in roster order
    all_responses: list[AgentResponse] = []
    all_gen_results: list[GenerateResult | None] = []
    for responses, gen_result in group_results:
        all_responses.extend(responses)
        all_gen_results.append(gen_result)

    # Record outputs into dialogue history (behavior adoption)
    # Also write utterances to Qdrant via enrichment wrapper and set utterance_key
    # for keys-over-wire: Falcon reads text by key instead of inline.
    try:
        qdrant_cli = progeny_qdrant.get_client()
    except RuntimeError:
        qdrant_cli = None

    for resp in all_responses:
        if resp.utterance:
            _accumulator.record_agent_output(resp.agent_id, resp.utterance)
            # Write utterance to Qdrant, set utterance_key for Falcon
            if qdrant_cli is not None:
                try:
                    key = await qdrant_wrapper.ingest(
                        client=qdrant_cli,
                        text=resp.utterance,
                        collection=COLLECTION_NPC_MEMORIES,
                        agent_id=resp.agent_id,
                        game_ts=time.time(),  # wall time — no game_ts for LLM output
                        event_type="response",
                    )
                    if key:
                        resp.utterance_key = key
                except Exception as exc:
                    logger.warning("Utterance write failed for %s (non-fatal): %s", resp.agent_id, exc)

        _scheduler.record_action(resp.agent_id)
        # Extract SetCurrentTask actions → persist as active_task
        for action in resp.actions:
            if action.command == "SetCurrentTask":
                buf = _accumulator._agent_buffers.get(resp.agent_id)
                if buf:
                    buf.active_task = action.target or ""
                    logger.info(
                        "Agent %s task: %s",
                        resp.agent_id,
                        buf.active_task or "(cleared)",
                    )

    # Emotional adoption: agent's own words shift its state (bidirectional pipeline)
    outbound_deltas = emotional_delta.process_outbound(all_responses, _harmonic_state)

    # Arc compression: check snap threshold for each agent after emotional adoption.
    # If snap exceeds threshold, generate a MOD-tier arc summary from recent RAW points.
    for agent_id, delta in outbound_deltas.items():
        if _arc_compressor.should_generate_arc(delta.snap):
            try:
                await _arc_compressor.generate_arc_summary(
                    agent_id=agent_id,
                    arc_start_ts=0.0,  # TODO: track arc start per agent
                    arc_end_ts=delta.semagram[0] if delta.semagram else 0.0,
                    semantic_vector=delta.semagram,  # 9d used as placeholder
                    emotional_vector=delta.semagram,
                    game_ts=time.time(),
                )
            except Exception as exc:
                logger.warning("Arc compression failed for %s (non-fatal): %s", agent_id, exc)

    # Slide memory windows — compress overflow after new entries recorded
    for agent_id in turn_context.active_npc_ids:
        buf = _accumulator._agent_buffers.get(agent_id)
        if buf:
            slide_window(buf.memory)
    # Slide the group timeline through the same compression pipeline
    slide_window(_accumulator._group_memory)

    elapsed_ms = int((time.monotonic() - start_ms) * 1000)
    timings = _aggregate_timings(all_gen_results)

    logger.info(
        "Tick %s: turn complete in %dms, %d responses across %d groups "
        "(prompt: %d tok/%.0fms, gen: %d tok/%.0fms, cached: %d tok)",
        tick_id, elapsed_ms, len(all_responses), len(groups),
        timings.prompt_tokens, timings.prompt_ms,
        timings.generated_tokens, timings.generation_ms,
        timings.cache_tokens,
    )

    model_used = "error" if all(r is None for r in all_gen_results) else "llama.cpp"

    return TurnResponse(
        tick_id=tick_id,
        responses=all_responses,
        processing_time_ms=elapsed_ms,
        model_used=model_used,
        llm_timings=timings,
    )



# ---------------------------------------------------------------------------
# WebSocket channel — persistent Falcon↔Progeny link
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def ws_channel(websocket: WebSocket) -> None:
    """Persistent bidirectional channel for Falcon tick delivery and response return.

    Falcon sends tick frames, Progeny processes through the same ingest()
    pipeline and sends response frames back when ready. No HTTP round-trip
    per tick — events and responses flow independently.

    Frame format (JSON):
        Falcon→Progeny:  {"type": "tick", "data": <TickPackage>}
        Progeny→Falcon:  {"type": "ack", "data": <AckResponse>}
                          {"type": "turn_response", "data": <TurnResponse>}
    """
    await websocket.accept()
    logger.info("Falcon WebSocket connected")

    async def _process_and_respond(pkg: TickPackage) -> None:
        """Process a tick in the background; send result when ready.

        Runs as a fire-and-forget task so the WebSocket receive loop
        stays responsive to pings during long LLM generation.
        """
        try:
            result = await ingest(pkg)
            if isinstance(result, TurnResponse):
                await websocket.send_text(json.dumps({
                    "type": "turn_response",
                    "data": result.model_dump(mode="json"),
                }))
                logger.info("WS: sent turn_response (%d agents)", len(result.responses))
            else:
                await websocket.send_text(json.dumps({
                    "type": "ack",
                    "data": result.model_dump(mode="json"),
                }))
        except Exception:
            logger.exception("WS: error processing tick %s", pkg.tick_id)

    try:
        while True:
            raw = await websocket.receive_text()
            frame = json.loads(raw)

            if frame.get("type") == "heartbeat":
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                continue

            if frame.get("type") != "tick":
                logger.warning("Unknown WS frame type: %s", frame.get("type"))
                continue

            package = TickPackage.model_validate(frame["data"])
            # Fire-and-forget: don't block the receive loop on LLM generation
            asyncio.create_task(_process_and_respond(package))
    except WebSocketDisconnect:
        logger.info("Falcon WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")


@router.get("/health")
async def health() -> dict:
    """Liveness check + LLM connectivity."""
    llm_ok = await llm_client.health_check()
    return {
        "status": "ok",
        "llm_connected": llm_ok,
        "turn_counter": _scheduler.turn_counter,
    }
