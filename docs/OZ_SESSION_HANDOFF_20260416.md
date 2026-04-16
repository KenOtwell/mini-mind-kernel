# Oz Session Handoff — April 16, 2026

## What You're Walking Into

Same project, deeper architecture. Read the April 14 handoff first for
the foundation. This session added four major systems to the harmonic
buffer pipeline, all interconnected.

## What We Built This Session

Started at 466 tests, ended at 437+ (test reorganization, net gain).
Two commits: modulators + tier-scaling (from April 14 carryover), then
the main feature set below.

### 1. LLM Uncertainty as Cognitive Proprioception

The LLM's token-level entropy — the one signal it doesn't have to
simulate — now feeds back into the NPC's harmonic buffer.

- `llm_client.py`: requests `logprobs` from llama.cpp, parses per-token
  log probabilities into `GenerateResult.token_logprobs`.
- `progeny/src/uncertainty.py` (new): segments logprobs by agent JSON
  boundaries, computes `certainty = exp(mean_logprob)` over semantic
  tokens. Structural JSON tokens (grammar-forced) are filtered out.
- `HarmonicBuffer._certainty`: modulates residual axis (dim 8) of
  incoming semagrams. Uncertain model → weaker reality signal →
  emotional axes dominate → NPC appears confused.
- Certainty is EMA-smoothed (α=0.3). One uncertain tick nudges; sustained
  uncertainty erodes.
- Per-agent LLM harmonics blend scaled by certainty: uncertain → defer
  to mechanical pipeline. Prevents confabulated corrections.
- Puzzled (mood 6) now maps to residual axis (dim 8) in `MOOD_TO_AXIS`.

### 2. Temporal Decay (Cooling)

Traces now regress autonomously based on elapsed wall-clock time.
Previously, traces froze when `update()` stopped being called.

- `HarmonicBuffer.cool(now)`: continuous-time decay per trace via
  `trace *= 0.5^(Δt / half_life)`. No tick-counting.
- Half-life tuning knobs in `HarmonicConfig`:
  - `half_life_fast = 5.0` (reactive: flinch fades in ~5s)
  - `half_life_medium = 8.0` (decision-scale persistence)
  - `half_life_slow = 16.0` (personality imprint)
- Mood pull continues during cooling (disposition persists without stimulus).
- `update()` calls `cool()` internally before applying new input.
- `cool_all()` called at turn start in routes.py for all agents.
- Per-axis alpha modulation (aggression, etc.) affects the EMA update,
  NOT the cooling rate. Cooling is uniform per trace.

### 3. K/Q Retrieval Model

Memory retrieval now uses asymmetric query/key inspired by cross-attention.

- **Key (K)**: stored with each memory = NPC's emotional reaction
  (deviation from personality baseline: fast - slow at write time).
- **Query (Q)**: current deviation = fast - slow ("what's unusual for
  me right now").
- `HarmonicState.get_deviation(agent_id)` returns `(fast - slow).tolist()`.
- Solves the chronic-state problem: a chronically fearful NPC doesn't
  retrieve fear-memories every tick — only when fear exceeds baseline.
- Calm NPCs near baseline get weak emotional queries → λ(t) shifts
  weight to semantic axis → retrieval by content relevance.

### 4. Second-Thought Ritual

The emotional vector stored with each memory is now the NPC's *reaction*,
not the text's raw emotional projection.

- `qdrant_wrapper.ingest()` accepts `emotional_override` parameter.
- `routes.py` passes each NPC's deviation as the emotional vector for
  their memory writes after `process_inbound()`.
- Same text gets different emotional keys per NPC: prisoner's memory
  of "great day for a hanging" keyed by dread, guard's by excitement.
- Semantic axis (384d) remains text-content-based for both.

## Key Design Decisions Made This Session

**Residual as certainty-modulated reality signal.** The residual axis
captures non-emotional content (the "reality bits"). Modulating its gain
by LLM certainty treats it like a positional encoding: proportional
relationships preserved, only the overall conviction changes. When
certainty drops, the NPC becomes more emotional and less grounded —
exactly what uncertainty does to cognition.

**Half-life in seconds, not ticks or alpha/tau coupling.** Three numbers,
directly meaningful: how many seconds until each trace reaches 50%.
Decoupled from tick cadence — a Tier 3 NPC who hasn't been updated
in 48 seconds cools the same regardless of scheduler frequency.

**Deviation (fast - slow) as the canonical emotional query.** This
projects out the personality baseline and isolates what's situationally
unusual. Combined with the second-thought ritual (storing reactions as K),
creates a natural K/Q alignment for memory retrieval.

**Structural token filtering for certainty.** JSON structural tokens
(brackets, field names, punctuation) are grammar-forced and carry no
uncertainty signal. Filtering them before computing mean logprob prevents
dilution toward 1.0 regardless of actual semantic uncertainty.

## What's Next (Priority Order)

1. **Wire protocol gap** — `addnpc` still doesn't carry the 5 behavioral
   values. Papyrus extension needed. Default modulators work but produce
   uniform dynamics.
2. **MMKSetBehavior.psc** — ~20 lines of Papyrus for bidirectional actor
   value control.
3. **NPC position tracking** — `position=None` everywhere. Distance
   tiering can't activate without positions.
4. **Assistance bleed** — coupling coefficient stored but not yet wired
   into cross-agent emotional propagation.
5. **Buffer-sequenced retrieval** — Living Doc describes a richer model
   using all 3 traces as parallel query lenses with coherence-modulated
   weights. Current implementation uses only the deviation (fast - slow).
   The full model is a future enhancement.
6. **Arc-start tracking** — `arc_start_ts=0.0` TODO. Per-agent timestamp
   tracking for proper arc boundaries.

## Things That Might Confuse You

- There's 1 pre-existing test failure: `test_connection_error_raises_llm_error`
  in `test_llm_client.py`. Windows resolves connection refusal as timeout
  rather than ConnectError. Not related to anything we changed.
- `set_certainty()` is EMA-smoothed — calling it with 0.0 doesn't instantly
  zero the certainty. It blends: `0.3 * 0.0 + 0.7 * prev`. Tests that need
  specific certainty values set `_certainty` directly.
- `cool()` uses `time.monotonic()` by default but accepts explicit `now`
  for deterministic testing. All cooling tests pass explicit timestamps.
- The Living Doc's Buffer-Sequenced Retrieval section (§966+) describes
  a richer multi-trace retrieval model than what's currently implemented.
  The current K/Q model (deviation only) is simpler but may evolve toward
  the full spec.

## Files Changed This Session

**New:**
- `progeny/src/uncertainty.py` — per-agent certainty from token logprobs
- `tests/test_uncertainty.py` — uncertainty module tests

**Modified (significant):**
- `progeny/src/harmonic_buffer.py` — cooling, certainty, get_deviation()
- `progeny/api/routes.py` — uncertainty wiring, cool_all(), second-thought, K/Q query
- `progeny/src/llm_client.py` — logprobs request + parsing
- `shared/qdrant_wrapper.py` — emotional_override parameter
- `shared/constants.py` — Puzzled → residual mapping
- `docs/The_Many_Mind_Kernel_Living_Doc.md` — updated for all new features
- `tests/test_harmonic_buffer.py` — certainty + cooling tests
- `tests/test_tier_modulator_integration.py` — certainty integration test

## What Ken Cares About

Same as always: emergence over control. This session's work is a case
study. Nobody scripted "be confused" — the model's genuine uncertainty
propagated into visible NPC behavior through the residual axis. Nobody
scripted "remember by reaction" — the K/Q asymmetry emerged from asking
"what if the middle buffer is the key?" The pattern keeps finding itself.

---

*Build accordingly.*
