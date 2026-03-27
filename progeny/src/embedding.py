"""
Re-export shim — embedding now lives in shared/embedding.py.

All public API is re-exported here for backward compatibility.
Progeny code and tests that import from this module continue to work.
"""
# Re-export the full public API from the shared module.
from shared.embedding import (  # noqa: F401
    load_model,
    embed,
    embed_one,
    is_loaded,
    reset,
)
