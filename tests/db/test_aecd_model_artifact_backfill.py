from __future__ import annotations

import json
from pathlib import Path

from scripts.db.aecd_model_artifact_backfill_core import (
    discover_model_bundles,
    flatten_mlflow_params,
    parameter_changes,
)


def test_discover_model_bundles_keeps_checkpoint_and_summary_together(tmp_path: Path) -> None:
    # Given: one historical model version with a checkpoint, summary, and a prediction array.
    version_dir = tmp_path / "experiment_001" / "resnet18" / "v001"
    checkpoint_dir = version_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "fold_0.pt").write_bytes(b"checkpoint")
    (version_dir / "fold_predictions.npz").write_bytes(b"patient-linked output")
    (version_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "experiment": "experiment_001",
                "model_name": "resnet18",
                "model": "ResNet18-1D",
                "version": "v001",
                "config": {"learning_rate": 0.0005},
            }
        ),
        encoding="utf-8",
    )

    # When: historical model bundles are discovered.
    bundles = discover_model_bundles((tmp_path,))

    # Then: the model checkpoint is retained and prediction arrays are not treated as model files.
    assert len(bundles) == 1
    assert bundles[0].model_files == (checkpoint_dir / "fold_0.pt",)
    assert bundles[0].metadata_path == version_dir / "training_summary.json"
    assert bundles[0].source_version == "v001"


def test_flatten_and_compare_params_exposes_changed_hyperparameters() -> None:
    # Given: two historical configurations from consecutive model versions.
    previous = {
        "model_name": "resnet18",
        "config": {"learning_rate": 0.0005, "n_epochs": 150},
    }
    current = {
        "model_name": "resnet18",
        "config": {"learning_rate": 0.0003, "n_epochs": 150},
    }

    # When: parameters are flattened and compared.
    previous_params = dict(flatten_mlflow_params(previous))
    current_params = dict(flatten_mlflow_params(current))
    changes = parameter_changes(previous_params, current_params)

    # Then: only the changed hyperparameter is reported.
    assert changes == {
        "config_learning_rate": {"from": "0.0005", "to": "0.0003"}
    }
