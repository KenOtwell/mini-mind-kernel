"""
Embedding module for Progeny — sentence-level semantic embeddings.

Loads all-MiniLM-L6-v2 on CPU (384d output). Singleton pattern: the model
loads once at startup or lazily on first call. All embedding happens here —
Progeny is the single authority for text → vector transformation.

Thread-safe: SentenceTransformer.encode() is internally safe for concurrent
calls from async gather.
"""
from __future__ import annotations

import logging

import numpy as np

from shared.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — initialized by load_model() or lazily
_model = None


def load_model() -> None:
    """Load the embedding model into memory.

    Called once at startup from server.py lifespan. If not called
    explicitly, the model loads lazily on first embed() call.
    """
    global _model
    if _model is not None:
        return

    from sentence_transformers import SentenceTransformer

    model_name = settings.embedding.model_name
    device = settings.embedding.device
    logger.info("Loading embedding model %s on %s...", model_name, device)
    _model = SentenceTransformer(model_name, device=device)
    dim = _model.get_sentence_embedding_dimension()
    logger.info("Embedding model loaded: %s (%dd)", model_name, dim)


def _ensure_model():
    """Lazy init guard."""
    if _model is None:
        load_model()


def embed(texts: list[str]) -> np.ndarray:
    """Batch encode texts to semantic embeddings.

    Args:
        texts: List of strings to embed.

    Returns:
        numpy array of shape (N, 384), float32.
    """
    _ensure_model()
    if not texts:
        return np.empty((0, settings.embedding.semantic_dim), dtype=np.float32)
    return _model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)


def embed_one(text: str) -> np.ndarray:
    """Embed a single text string.

    Args:
        text: String to embed.

    Returns:
        numpy array of shape (384,), float32.
    """
    return embed([text])[0]


def is_loaded() -> bool:
    """Check if the embedding model is loaded."""
    return _model is not None
