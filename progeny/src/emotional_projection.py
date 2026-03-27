"""
Re-export shim — emotional projection now lives in shared/emotional.py.

All public API is re-exported here for backward compatibility.
Progeny code and tests that import from this module continue to work.
"""
# Re-export the full public API from the shared module.
from shared.emotional import (  # noqa: F401
    load_bases,
    project,
    project_batch,
    is_loaded,
    reset,
)
