"""Deprecated compat shim for `models.train_stacking`. Use
`scripts/training/train_usersnet.py` directly, or import helpers from
`sers.models.usersnet.stacking`.
"""

import importlib.util as _iu
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "training" / "train_usersnet.py"

if not _SCRIPT.exists():
    raise ImportError(f"Not found: {_SCRIPT}")

_spec = _iu.spec_from_file_location("_train_usersnet_compat", _SCRIPT)
_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

del _iu, _Path, _spec, _mod
