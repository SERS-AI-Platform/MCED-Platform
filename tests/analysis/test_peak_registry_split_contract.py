from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.training.train_peak_evidence_model import _validate_registry_provenance


def digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def write_registry_provenance(root: Path, sample_ids: list[str]) -> Path:
    registry = root / "peak_registry.csv"
    registry.write_text("peak_id\nP1\n", encoding="utf-8-sig")
    criteria = {
        "selection_scope": {
            "split_role": "train",
            "subject_count": len(sample_ids),
            "sample_id_sha256": digest(sample_ids),
        }
    }
    registry.with_name("peak_registry_criteria.json").write_text(
        json.dumps(criteria), encoding="utf-8"
    )
    return registry


def test_registry_provenance_accepts_the_locked_training_subjects(tmp_path: Path) -> None:
    sample_ids = ["PRO_1", "NOR_1", "PAN_YPAN_2"]
    registry = write_registry_provenance(tmp_path, sample_ids)

    scope = _validate_registry_provenance(registry, np.asarray(sample_ids, dtype=object))

    assert scope["split_role"] == "train"
    assert scope["subject_count"] == 3


def test_registry_provenance_rejects_test_subject_leakage(tmp_path: Path) -> None:
    registry = write_registry_provenance(tmp_path, ["PRO_1", "NOR_1", "PAN_YPAN_2"])

    with pytest.raises(ValueError, match="do not match"):
        _validate_registry_provenance(
            registry,
            np.asarray(["PRO_1", "NOR_1", "PAN_YPAN_99"], dtype=object),
        )
