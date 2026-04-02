"""
Speech-to-text via faster-whisper (CPU).

Lazy-loads the model on first transcription request to avoid slowing
Falcon startup. CHIM sends WAV audio to /stt.php; this module
transcribes it and returns plain text.
"""
from __future__ import annotations

import io
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Model size — configurable via env var, default "base" (~150MB, <1s on CPU)
_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "base")
_LANGUAGE = os.environ.get("WHISPER_LANG", "en")

_model: Optional[object] = None


def _get_model():
    """Lazy-load the faster-whisper model on first use."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        logger.info("Loading faster-whisper model '%s' on CPU...", _MODEL_SIZE)
        _model = WhisperModel(
            _MODEL_SIZE,
            device="cpu",
            compute_type="int8",
        )
        logger.info("faster-whisper model loaded.")
    return _model


def transcribe(audio_bytes: bytes) -> str:
    """
    Transcribe WAV audio bytes to text.

    Returns the concatenated text from all detected segments,
    or empty string if nothing was recognized.
    """
    model = _get_model()
    audio_stream = io.BytesIO(audio_bytes)

    segments, info = model.transcribe(
        audio_stream,
        language=_LANGUAGE,
        beam_size=3,
        vad_filter=True,
    )

    text_parts = [segment.text.strip() for segment in segments]
    result = " ".join(text_parts).strip()

    logger.info("STT: %.1fs audio, lang=%s, text='%s'",
                info.duration, info.language, result[:120])
    return result
