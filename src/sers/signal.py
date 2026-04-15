"""Deprecated shim — use `sers.preprocessing.signal` instead.

Kept temporarily so that existing imports `from sers.signal import ...`
continue to work. Will be removed in a future release.
"""

from sers.preprocessing.signal import *  # noqa: F401,F403
from sers.preprocessing import signal as _s

__all__ = getattr(_s, "__all__", [n for n in dir(_s) if not n.startswith("_")])
