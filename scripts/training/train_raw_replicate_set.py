from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import click
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from sers.config import load_config
from sers.raw_set.data import CohortSpec, load_patient_records
from sers.raw_set.model import RawSetConfig
from sers.raw_set.training import (
    FoldData,
    RawTrainingConfig,
    binary_metrics,
    fit_raw_set,
    fit_standardizer,
    gradient_wavenumber_importance,
    predict_raw_set,
    standardize_records,
)


@dataclass(frozen=True, slots=True)
class ExperimentOptions:
    data_root: Path
    config_path: Path
    out_dir: Path
    selected_groups: tuple[str, ...]
    folds: int
    epochs: int
    grid_points: int
    device: str


def _cohorts(config_path: Path, selected_groups: tuple[str, ...]) -> tuple[CohortSpec, ...]:
    config = load_config(config_path)
    selected = {group.upper() for group in selected_groups}
    cohorts = []
    for folder, group in config.folder_to_group.items():
        category = config.display.category_map.get(group)
        if group == "SPAN" or category not in {"cancer", "non_cancer", "control"}:
            continue
        if selected and group not in selected:
            continue
        cohorts.append(CohortSpec(folder=folder, group=group, cancer=category == "cancer"))
    return tuple(cohorts)


def _baseline_metrics(data: FoldData) -> dict[str, float]:
    standardizer = fit_standardizer(data.train)
    train = standardize_records(data.train, standardizer)
    validation = standardize_records(data.validation, standardizer)
    train_x = np.stack([record.replicates.mean(axis=0) for record in train])
    validation_x = np.stack([record.replicates.mean(axis=0) for record in validation])
    train_y = np.array([record.cancer for record in train], dtype=int)
    validation_y = np.array([record.cancer for record in validation], dtype=int)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=2_000,
        random_state=42,
    )
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(validation_x)[:, 1]
    return {
        "auc": float(roc_auc_score(validation_y, probabilities)),
        "brier": float(brier_score_loss(validation_y, probabilities)),
        "accuracy": float(accuracy_score(validation_y, probabilities >= 0.5)),
    }


def _write_rows(path: Path, rows: list[dict[str, str | int | float]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _experiment_config(
    options: ExperimentOptions,
    auxiliary: bool,
) -> RawTrainingConfig:
    return RawTrainingConfig(
        model=RawSetConfig(n_wavenumbers=options.grid_points),
        epochs=options.epochs,
        reconstruction_weight=0.2 if auxiliary else 0.0,
        consistency_weight=0.1 if auxiliary else 0.0,
        device=options.device,
    )


@click.command()
@click.option(
    "--data-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/raw_data"),
    show_default=True,
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("config/config.yaml"),
    show_default=True,
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("results/raw_set/experiment"),
    show_default=True,
)
@click.option("--group", "selected_groups", multiple=True)
@click.option("--folds", type=click.IntRange(2, 10), default=5, show_default=True)
@click.option("--epochs", type=click.IntRange(1, 1_000), default=30, show_default=True)
@click.option("--grid-points", type=click.IntRange(128, 4_096), default=933, show_default=True)
@click.option("--device", type=click.Choice(["cpu", "cuda"]), default="cuda", show_default=True)
def main(
    data_root: Path,
    config_path: Path,
    out_dir: Path,
    selected_groups: tuple[str, ...],
    folds: int,
    epochs: int,
    grid_points: int,
    device: str,
) -> None:
    _run(
        ExperimentOptions(
            data_root=data_root,
            config_path=config_path,
            out_dir=out_dir,
            selected_groups=selected_groups,
            folds=folds,
            epochs=epochs,
            grid_points=grid_points,
            device=device,
        )
    )


def _run(options: ExperimentOptions) -> None:
    if options.device == "cuda" and not torch.cuda.is_available():
        raise click.UsageError("CUDA was requested but is not available")
    cohorts = _cohorts(options.config_path, options.selected_groups)
    grid = np.linspace(400.0, 2200.0, options.grid_points, dtype=np.float32)
    records = load_patient_records(options.data_root, cohorts, grid)
    labels = np.array([record.cancer for record in records], dtype=int)
    if len(np.unique(labels)) != 2:
        raise click.UsageError("selected cohorts must contain cancer and non-cancer patients")

    options.out_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, str | int | float]] = []
    prediction_rows: list[dict[str, str | int | float]] = []
    importance = np.zeros(options.grid_points, dtype=np.float64)
    splitter = StratifiedKFold(n_splits=options.folds, shuffle=True, random_state=42)
    for fold, (train_index, validation_index) in enumerate(splitter.split(records, labels)):
        data = FoldData(
            train=tuple(records[index] for index in train_index),
            validation=tuple(records[index] for index in validation_index),
        )
        baseline = _baseline_metrics(data)
        metric_rows.append(
            {
                "model": "raw_mean_logistic",
                "fold": fold,
                "auc": baseline["auc"],
                "brier": baseline["brier"],
                "accuracy": baseline["accuracy"],
                "train_patients": len(data.train),
                "validation_patients": len(data.validation),
            }
        )
        for model_name, auxiliary in (
            ("raw_set_classification", False),
            ("raw_set_auxiliary", True),
        ):
            fitted = fit_raw_set(
                data,
                _experiment_config(options, auxiliary),
            )
            predictions = predict_raw_set(fitted, data.validation)
            metrics = binary_metrics(predictions)
            metric_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "auc": metrics.auc,
                    "brier": metrics.brier,
                    "accuracy": metrics.accuracy,
                    "train_patients": len(data.train),
                    "validation_patients": len(data.validation),
                }
            )
            for key, group, label, probability in zip(
                predictions.patient_keys,
                predictions.groups,
                predictions.labels,
                predictions.probabilities,
            ):
                prediction_rows.append(
                    {
                        "model": model_name,
                        "fold": fold,
                        "patient_key": key,
                        "group": group,
                        "label": int(label),
                        "probability": float(probability),
                    }
                )
            if auxiliary:
                importance += gradient_wavenumber_importance(fitted, data.validation)
                torch.save(
                    fitted.model.state_dict(),
                    options.out_dir / f"fold_{fold}_auxiliary.pt",
                )

    _write_rows(options.out_dir / "fold_metrics.csv", metric_rows)
    _write_rows(options.out_dir / "patient_predictions.csv", prediction_rows)
    importance_rows = [
        {
            "wavenumber_cm-1": float(wavenumber),
            "mean_abs_input_gradient": float(score / options.folds),
        }
        for wavenumber, score in zip(grid, importance)
    ]
    _write_rows(options.out_dir / "candidate_wavenumber_importance.csv", importance_rows)
    summary = {
        "records": len(records),
        "cohorts": [asdict(cohort) for cohort in cohorts],
        "folds": options.folds,
        "epochs": options.epochs,
        "grid_points": options.grid_points,
        "usable_average_targets": sum(record.average_usable for record in records),
        "hospital_confound_warning": (
            "Cancer Screening estimates may be hospital-confounded; "
            "do not interpret cross-hospital generalization from this experiment."
        ),
    }
    (options.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    click.echo(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
