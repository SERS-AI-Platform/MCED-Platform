"""sers.preprocessing — spectrum preprocessing (post-QC).

Backward-compatible re-exports from .core so existing
`from sers.preprocessing import preprocess_single_spectrum, ...` keeps working.

Submodules:
    core                  — main preprocessing pipeline (trim/smooth/baseline/normalize)
    signal                — low-level signal ops (baseline_correction, snv, smooth, resample)
    calibration_transfer  — PDS / cross-instrument calibration
"""

from sers.preprocessing.core import *  # noqa: F401,F403
from sers.preprocessing import core as _core

__all__ = getattr(_core, "__all__", [name for name in dir(_core) if not name.startswith("_")])
