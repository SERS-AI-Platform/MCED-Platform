"""sers.io — file I/O and ingestion.

Backward-compatible re-exports from .core so existing
`from sers.io import read_spectrum, ...` continues to work.
"""

from sers.io.core import *  # noqa: F401,F403
from sers.io import core as _core

__all__ = getattr(_core, "__all__", [name for name in dir(_core) if not name.startswith("_")])
