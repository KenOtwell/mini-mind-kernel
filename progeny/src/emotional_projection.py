"""
Emotional projection for Progeny — 384d → 9d semagram.

Projects sentence embeddings onto the 9-dimensional emotional space defined
by the Gram-Schmidt orthonormalized bases in emotional_bases_9d.npz.

Axes 0-7: fear, anger, love, disgust, excitement, sadness, joy, safety
Axis 8:   residual magnitude — how much semantic content the 8 axes
          did NOT capture. A built-in novelty/complexity signal.

The bases are perfectly orthonormal (max off-diagonal dot ~1e-16).
Projection is a simple dot product — no inverse needed.

NOTE: This module is a candidate to move to shared/emotional.py so that
both the Qdrant enrichment wrapper (used by Falcon and Progeny for
auto-embed on ingestion) and Progeny's harmonic buffer logic share
the same projection math. The API will remain identical.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from shared.constants import EMOTIONAL_AXES, EMOTIONAL_DIM

logger = logging.getLogger(__name__)

# Module-level state — loaded once by load_bases()
_bases: np.ndarray | None = None       # (8, 384) orthonormal GS bases
_bases_path: Path = Path(__file__).resolve().parents[2] / "shared" / "data" / "emotional_bases_9d.npz"


def load_bases(path: Path | None = None) -> None:
    """Load the 9d emotional projection bases from the npz file.

    Called once at startup. The bases matrix is (8, 384) — the 8
    emotional axes. Residual (axis 8) is computed, not stored.

    Args:
        path: Override path to the npz file. Defaults to shared/data/.
    """
    global _bases
    if _bases is not None:
        return

    npz_path = path or _bases_path
    logger.info("Loading emotional bases from %s", npz_path)
    data = np.load(npz_path)

    # Stack the 8 emotional axis vectors into a (8, 384) matrix
    emotion_names = [ax for ax in EMOTIONAL_AXES if ax != "residual"]
    basis_vectors = [data[name].astype(np.float32) for name in emotion_names]
    _bases = np.stack(basis_vectors)  # (8, 384)

    logger.info(
        "Emotional bases loaded: %d axes × %dd, residual computed",
        _bases.shape[0], _bases.shape[1],
    )


def _ensure_bases() -> np.ndarray:
    """Lazy init guard. Returns the bases matrix."""
    if _bases is None:
        load_bases()
    return _bases


def project(embedding: np.ndarray) -> list[float]:
    """Project a single 384d embedding onto the 9d emotional space.

    Math:
        emb_norm = embedding / norm(embedding)
        coeffs = bases_8x384 @ emb_norm    → 8 projection coefficients
        reconstruction = coeffs @ bases_8x384
        residual_mag = norm(emb_norm - reconstruction)
        semagram = [coeffs..., residual_mag]

    Args:
        embedding: (384,) float array — raw sentence embedding.

    Returns:
        9-element list of floats — the emotional semagram.
    """
    bases = _ensure_bases()

    # Normalize the embedding
    emb_norm = _safe_normalize(embedding)

    # Project onto 8 emotional axes
    coeffs = bases @ emb_norm  # (8,)

    # Residual: magnitude of the unprojected component
    reconstruction = coeffs @ bases  # (384,)
    residual_vec = emb_norm - reconstruction
    residual_mag = float(np.linalg.norm(residual_vec))

    # 9d semagram: 8 coefficients + residual magnitude
    semagram = coeffs.tolist() + [residual_mag]
    return semagram


def project_batch(embeddings: np.ndarray) -> np.ndarray:
    """Project a batch of 384d embeddings to 9d semagrams.

    Vectorized version of project() for efficiency.

    Args:
        embeddings: (N, 384) float array.

    Returns:
        (N, 9) float array — batch of semagrams.
    """
    bases = _ensure_bases()
    n = embeddings.shape[0]
    if n == 0:
        return np.empty((0, EMOTIONAL_DIM), dtype=np.float32)

    # Normalize each embedding
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # Guard against zero-norm embeddings
    norms = np.maximum(norms, 1e-10)
    emb_normed = embeddings / norms  # (N, 384)

    # Project onto 8 emotional axes: (N, 384) @ (384, 8) → (N, 8)
    coeffs = emb_normed @ bases.T  # (N, 8)

    # Reconstruction and residual
    reconstructions = coeffs @ bases  # (N, 384)
    residual_vecs = emb_normed - reconstructions  # (N, 384)
    residual_mags = np.linalg.norm(residual_vecs, axis=1, keepdims=True)  # (N, 1)

    # Stack: (N, 8) + (N, 1) → (N, 9)
    semagrams = np.hstack([coeffs, residual_mags]).astype(np.float32)
    return semagrams


def _safe_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a vector, guarding against zero magnitude."""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return np.zeros_like(vec)
    return vec / norm


def is_loaded() -> bool:
    """Check if bases are loaded."""
    return _bases is not None
