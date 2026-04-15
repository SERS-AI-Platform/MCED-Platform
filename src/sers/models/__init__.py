"""SERS-AI model registry and model implementations.

This package is the single import point for all models in the project.
See `docs/DIRECTORY_CONVENTIONS.md` and `docs/MODEL_STATUS.md` for the
model taxonomy (production / baseline-legacy / experimental / archived).
"""

from sers.models._registry import (
    MODEL_REGISTRY,
    ModelSpec,
    get_spec,
    list_by_category,
    list_all,
)

__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "get_spec",
    "list_by_category",
    "list_all",
]
