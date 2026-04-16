"""
Per-agent uncertainty extraction from LLM token log probabilities.

The LLM's token-level entropy is the one cognitive signal it doesn't
have to simulate — it's genuinely experiencing uncertainty when the
probability distribution flattens. This module captures that signal
and attributes it to individual NPCs in multi-agent dispatch groups.

Certainty factor: exp(mean_logprob) over an agent's token span.
This is the geometric mean token probability — a standard confidence
metric. Range [0, 1]:
  1.0 = every token was the obvious choice (high confidence)
  0.0 = every token was a coin flip (total uncertainty)
  ~0.5 = moderate confidence (typical for nuanced responses)

The certainty factor modulates the residual axis (dim 8) of the
harmonic buffer: uncertain model → weaker reality signal → emotional
axes relatively dominate → the NPC becomes more reactive, less
grounded. Recovery is natural: when certainty returns, the residual
strengthens and the NPC re-grounds.

See Living Doc §LLM Uncertainty as Cognitive Proprioception.
"""
from __future__ import annotations

import logging
import math
import re
from typing import Any

logger = logging.getLogger(__name__)

# Regex to find "agent_id" markers in the reconstructed token text.
# Matches: "agent_id": "SomeName" or "agent_id":"SomeName"
# Captures the agent name in group 1.
_AGENT_ID_RE = re.compile(r'"agent_id"\s*:\s*"([^"]+)"')

# ---------------------------------------------------------------------------
# Structural token filtering
# ---------------------------------------------------------------------------
# JSON structural tokens are grammar-forced — the model has no choice about
# them, so their near-zero logprobs carry no uncertainty signal. Including
# them dilutes the certainty metric toward 1.0 regardless of actual semantic
# uncertainty. We filter them out before computing the mean.

# Single-character JSON punctuation and whitespace
_STRUCTURAL_CHARS = frozenset('{}\'[],:" \n\r\t')

# Field name tokens the model must produce (schema-forced, not decisions).
# Lowercased for matching.
_FIELD_NAME_TOKENS = frozenset({
    '"agent_id"', '"utterance"', '"brief_utterance"',
    '"actor_value_deltas"', '"actions"', '"updated_harmonics"',
    '"base_vector"', '"command"', '"target"', '"item"',
    '"responses"', '"new_memories"', '"text"',
    '"aggression"', '"confidence"', '"morality"', '"mood"', '"assistance"',
})


def _is_structural(token: str) -> bool:
    """Check if a token is grammar-forced JSON structure.

    Structural tokens carry no uncertainty signal — the model must
    produce them regardless of its confidence about the NPC's state.
    """
    stripped = token.strip()
    if not stripped:
        return True  # whitespace-only
    # Single-char JSON punctuation
    if len(stripped) <= 2 and all(c in _STRUCTURAL_CHARS for c in stripped):
        return True
    # Known field names (case-insensitive)
    if stripped.lower() in _FIELD_NAME_TOKENS:
        return True
    # Common multi-char structural patterns
    if stripped in (': "', '": ', ': [', '":', '},', '],', '}, ', '], '):
        return True
    return False


def compute_certainty(
    token_logprobs: list[dict[str, Any]] | None,
    agent_ids: list[str],
) -> dict[str, float]:
    """Compute per-agent certainty factors from token log probabilities.

    Segments the token sequence by agent response boundaries, then
    computes certainty = exp(mean_logprob) for each agent's span.

    Args:
        token_logprobs: List of {token: str, logprob: float} dicts from
            the LLM response. None if logprobs unavailable.
        agent_ids: Expected agent IDs in this dispatch group, in order.

    Returns:
        Dict mapping agent_id → certainty factor in [0, 1].
        Returns 1.0 for all agents if logprobs are unavailable.
    """
    # Fallback: no logprobs → no modulation (certainty = 1.0)
    if not token_logprobs or not agent_ids:
        return {aid: 1.0 for aid in agent_ids}

    # Solo dispatch: single agent gets all tokens, skip segmentation
    if len(agent_ids) == 1:
        certainty = _segment_certainty(token_logprobs)
        return {agent_ids[0]: certainty}

    # Multi-agent: segment by agent_id boundaries in the token stream
    segments = _segment_by_agent(token_logprobs, agent_ids)

    result: dict[str, float] = {}
    for agent_id in agent_ids:
        agent_tokens = segments.get(agent_id)
        if agent_tokens:
            result[agent_id] = _segment_certainty(agent_tokens)
        else:
            # Agent not found in output (LLM skipped it) → default
            result[agent_id] = 1.0

    return result


def _segment_certainty(tokens: list[dict[str, Any]]) -> float:
    """Compute certainty from a segment of token logprobs.

    Filters out structural JSON tokens (grammar-forced, carry no
    uncertainty signal) before computing the mean. Only semantic
    content tokens — utterances, values, action choices — contribute.

    certainty = exp(mean_logprob) = geometric mean of token probabilities.
    Clamped to [0, 1].

    Empty segments (or all-structural) return 1.0 (no evidence).
    """
    if not tokens:
        return 1.0

    # Filter: only semantic tokens with valid logprobs
    valid: list[float] = []
    for t in tokens:
        tok_str = t.get("token", "")
        lp = t.get("logprob", 0.0)
        if not isinstance(lp, (int, float)):
            continue
        if _is_structural(tok_str):
            continue
        valid.append(lp)

    if not valid:
        return 1.0  # All structural — no semantic signal

    mean_lp = sum(valid) / len(valid)
    # exp(mean_logprob): logprob is always ≤ 0, so result is in (0, 1]
    # Clamp for safety (floating point edge cases)
    return max(0.0, min(1.0, math.exp(mean_lp)))


def _segment_by_agent(
    token_logprobs: list[dict[str, Any]],
    agent_ids: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Segment token logprobs by agent response boundaries.

    Reconstructs the response text from the token sequence, finds
    "agent_id": "Name" markers, and maps character offsets back to
    token indices. Each agent's segment spans from their marker to
    the next agent's marker (or end of sequence).

    Args:
        token_logprobs: Full token sequence with logprobs.
        agent_ids: Expected agent IDs to look for.

    Returns:
        Dict mapping agent_id → list of token logprob entries for
        that agent's response segment.
    """
    # Reconstruct text and build character offset → token index mapping
    text_parts: list[str] = []
    # char_offset_to_token[i] = index into token_logprobs for the token
    # that starts at character offset i in the reconstructed text.
    token_starts: list[int] = []  # character offset where each token starts
    offset = 0
    for entry in token_logprobs:
        token_str = entry.get("token", "")
        token_starts.append(offset)
        text_parts.append(token_str)
        offset += len(token_str)

    full_text = "".join(text_parts)

    # Find agent_id boundaries in the reconstructed text
    agent_char_offsets: list[tuple[str, int]] = []
    for match in _AGENT_ID_RE.finditer(full_text):
        name = match.group(1)
        if name in agent_ids:
            agent_char_offsets.append((name, match.start()))

    if not agent_char_offsets:
        # No agent boundaries found — can't segment
        logger.debug("No agent_id boundaries found in token stream")
        return {}

    # Map character offsets to token indices via binary search
    def _char_to_token_idx(char_offset: int) -> int:
        """Find the token index that contains this character offset."""
        # Binary search: find the last token_start ≤ char_offset
        lo, hi = 0, len(token_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if token_starts[mid] <= char_offset:
                lo = mid
            else:
                hi = mid - 1
        return lo

    # Build segments: each agent's tokens span from their marker to
    # the next agent's marker (or end of sequence)
    segments: dict[str, list[dict[str, Any]]] = {}
    for i, (name, char_off) in enumerate(agent_char_offsets):
        start_tok = _char_to_token_idx(char_off)
        if i + 1 < len(agent_char_offsets):
            end_tok = _char_to_token_idx(agent_char_offsets[i + 1][1])
        else:
            end_tok = len(token_logprobs)
        segments[name] = token_logprobs[start_tok:end_tok]

    return segments
