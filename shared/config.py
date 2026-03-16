"""
Configuration for the Many-Mind Kernel.

Settings for both Falcon and Progeny services.
Override via environment variables (FALCON_PORT=8080, etc.)
or by modifying the defaults below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


@dataclass
class QdrantConfig:
    """Qdrant connection — both services connect to the Gaming PC instance."""
    host: str = field(default_factory=lambda: _env("QDRANT_HOST", "127.0.0.1"))
    rest_port: int = field(default_factory=lambda: _env_int("QDRANT_REST_PORT", 6333))
    grpc_port: int = field(default_factory=lambda: _env_int("QDRANT_GRPC_PORT", 6334))


@dataclass
class FalconConfig:
    """Falcon service settings (Gaming PC)."""
    host: str = field(default_factory=lambda: _env("FALCON_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("FALCON_PORT", 8000))
    comm_path: str = field(default_factory=lambda: _env("FALCON_COMM_PATH", "/comm.php"))


@dataclass
class ProgenyConfig:
    """Progeny service settings (Beelink or stub)."""
    host: str = field(default_factory=lambda: _env("PROGENY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PROGENY_PORT", 8001))
    timeout_seconds: float = field(default_factory=lambda: _env_float("PROGENY_TIMEOUT", 30.0))
    retry_attempts: int = field(default_factory=lambda: _env_int("PROGENY_RETRIES", 2))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class EmbeddingConfig:
    """Embedding model settings (Falcon only, CPU)."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    semantic_dim: int = 384
    emotional_dim: int = 9


@dataclass
class SchedulerConfig:
    """Many-Mind Scheduling thresholds and cadences."""
    # Distance tiers (game units, approximate meters)
    tier0_distance: float = field(default_factory=lambda: _env_float("SCHED_TIER0_DIST", 5.0))
    tier1_distance: float = field(default_factory=lambda: _env_float("SCHED_TIER1_DIST", 20.0))
    tier2_distance: float = field(default_factory=lambda: _env_float("SCHED_TIER2_DIST", 50.0))

    # Harmonic cadence (include agent every N turns at their tier)
    tier0_cadence: int = 1    # every prompt
    tier1_cadence: int = 2    # every 2nd
    tier2_cadence: int = 4    # every 4th
    tier3_cadence: int = 16   # every 16th

    max_agents_per_prompt: int = 16
    collaboration_floor_tier: int = 1  # min tier for collaborating NPCs


@dataclass
class Settings:
    """Top-level settings container."""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    falcon: FalconConfig = field(default_factory=FalconConfig)
    progeny: ProgenyConfig = field(default_factory=ProgenyConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    debug: bool = field(default_factory=lambda: _env_bool("MMK_DEBUG", False))
    log_level: str = field(default_factory=lambda: _env("MMK_LOG_LEVEL", "INFO"))


# Singleton — import this
settings = Settings()
