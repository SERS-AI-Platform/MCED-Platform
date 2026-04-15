"""Deprecated compat shim. Loads the ResNet legacy training script as a module
so old imports `from models.train import ...` continue to work.

New code: use `sers.models._legacy.resnet_v1` (library) + directly run the
script `scripts/training/_legacy/train_resnet.py`.
"""

import importlib.util as _iu
from pathlib import Path as _Path
import sys as _sys

_ROOT = _Path(__file__).resolve().parent.parent
_SCRIPT = _ROOT / "scripts" / "training" / "_legacy" / "train_resnet.py"

if not _SCRIPT.exists():
    raise ImportError(
        f"Legacy train_resnet.py not found at {_SCRIPT}. "
        "Repo may be mid-reorg; see docs/REORG_PLAN.md."
    )

_spec = _iu.spec_from_file_location("_train_resnet_compat", _SCRIPT)
_mod = _iu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

# Re-export all public names
for _name in dir(_mod):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_mod, _name)

del _iu, _Path, _sys, _spec, _mod
