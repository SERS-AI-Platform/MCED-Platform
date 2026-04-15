"""sers.validation — input data validation for SERS spectra.

Backward-compatible re-exports from .core so existing
`from sers.validation import validate_spectrum, validate_processed_spectrum` keeps working.

Submodules:
    core      — main validation functions + dataclasses
    protocol  — protocol-level validation (variance_convergence, etc.)
"""

from sers.validation.core import *  # noqa: F401,F403
from sers.validation import core as _core

__all__ = getattr(_core, "__all__", [name for name in dir(_core) if not name.startswith("_")])
