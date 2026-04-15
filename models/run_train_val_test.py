"""Deprecated compat shim. See scripts/experiments/_archive/2026-04_train_val_test_split.py"""

import importlib.util as _iu
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "experiments" / "_archive" / "2026-04_train_val_test_split.py"

if not _SCRIPT.exists():
    raise ImportError(f"Not found: {_SCRIPT}")

_spec = _iu.spec_from_file_location("_train_val_test_compat", _SCRIPT)
_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

del _iu, _Path, _spec, _mod
