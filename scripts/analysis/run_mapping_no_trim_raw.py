#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib", "openpyxl", "torch"]
# ///
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence, TypeAlias

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "patent"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import torch
from mapping_repeat_average_core import GROUP_ORDER, load_subjects
from run_mapping_multiscale_resnet import (
    BLOCK_CHOICES,
    PreparedSubject,
    _resnet_predict,
    build_rows,
    fit_logistic,
    fit_resnet,
    patient_oof_rows,
    patient_training_means,
    repeat_evaluation,
    save_repeat_figures,
    save_roc_figure,
    save_training_history,
    set_seed,
    summarize_repeat_rows,
    write_csv,
)
from sklearn.model_selection import StratifiedGroupKFold

DEFAULT_OUT = REPO / "results" / "mapping_no_trim_raw_20260826_v1"
FULL_GRID = np.linspace(50.66316, 3299.9, 933, dtype=float)
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class NoTrimDataError(ValueError):
    pass


def prepare_no_trim(grid: np.ndarray) -> tuple[list[PreparedSubject], dict[str, JsonValue]]:
    subjects, raw_axis = load_subjects(grid)
    prepared: list[PreparedSubject] = []
    finite_counts: list[int] = []
    for subject in subjects:
        native_axis = np.asarray(subject.native_axis, dtype=float)
        rows = np.vstack(
            [np.interp(grid, native_axis, np.asarray(row, dtype=float)) for row in subject.native_replicates]
        ).astype(np.float32)
        finite = np.isfinite(rows).all(axis=1)
        rows = rows[finite]
        if len(rows) == 0:
            raise NoTrimDataError(f"subject {subject.ordinal} has no finite raw spectra")
        finite_counts.append(len(rows))
        prepared.append(
            PreparedSubject(
                ordinal=subject.ordinal,
                group=subject.group,
                x=grid.copy(),
                all_spectra=rows,
                qc_keep=np.ones(len(rows), dtype=bool),
                qc_corr=np.full(len(rows), np.nan),
                qc_rsd_pct=float("nan"),
            )
        )
    metadata: dict[str, JsonValue] = {
        "raw_axis": raw_axis,
        "subject_count": len(prepared),
        "group_counts": {group: sum(item.group == group for item in prepared) for group in GROUP_ORDER},
        "input_repeats": int(sum(len(subject.native_replicates) for subject in subjects)),
        "model_repeats": int(sum(finite_counts)),
        "finite_repeats": int(sum(finite_counts)),
        "nonfinite_repeats_removed": int(sum(len(subject.native_replicates) for subject in subjects) - sum(finite_counts)),
        "qc_rule": "No spectral QC filtering; finite-value gate only. All finite raw repeats remain in the locked evaluation.",
        "grid": {"min_cm-1": float(grid[0]), "max_cm-1": float(grid[-1]), "points": int(len(grid)), "step_cm-1": float(np.median(np.diff(grid)))},
        "preprocessing": {"trim": None, "smooth": None, "baseline": None, "normalization": None, "alignment": "linear interpolation onto fixed full-range 933-point grid"},
    }
    return prepared, metadata


def write_report(out: Path, metadata: dict[str, JsonValue], metrics: Sequence[dict[str, JsonValue]], args: argparse.Namespace) -> None:
    rows = [row for row in metrics if row.get("aggregation") == "mean"]
    lines = [
        "# Mapping cohort no-trim raw ablation",
        "",
        f"- 대상: {metadata['subject_count']}명, {metadata['model_repeats']}개 finite raw repeat.",
        "- 전처리: trim, smoothing, baseline correction, SNV, robust QC filtering 모두 미적용.",
        f"- 전체 raw range {metadata['grid']['min_cm-1']:.3f}–{metadata['grid']['max_cm-1']:.3f} cm⁻¹를 933-point grid로 선형 보간했다.",
        f"- Multi-scale 1D ResNet: {args.n_blocks} blocks, position-aware={args.position_aware}, epochs={args.epochs}.",
        "- Patient-level StratifiedGroupKFold 5-fold OOF; LR는 patient mean, ResNet은 121개 repeat를 학습에 사용.",
        "",
        "| Task | Model | ROC-AUC | Macro ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        values = {key: float(row.get(key, np.nan)) for key in ("roc_auc", "macro_roc_auc", "balanced_accuracy", "sensitivity", "specificity", "macro_f1")}
        shown = ["-" if not np.isfinite(values[key]) else f"{values[key]:.4f}" for key in values]
        lines.append(f"| {row['task']} | {row['model']} | {' | '.join(shown)} |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mc-iterations", type=int, default=100)
    parser.add_argument("--n-blocks", type=int, choices=BLOCK_CHOICES, default=3)
    parser.add_argument("--position-aware", action="store_true")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    subjects, metadata = prepare_no_trim(FULL_GRID)
    X, patient_ids, y_binary, y_three, repeat_ids = build_rows(subjects)
    np.savez_compressed(out / "preprocessed_no_trim_raw_arrays.npz", X=X, patient_ids=patient_ids, y_binary=y_binary, y_three=y_three, repeat_ids=repeat_ids)
    metadata.update({"device": device, "n_blocks": args.n_blocks, "blocks_per_stage": args.n_blocks // 3, "position_aware": args.position_aware, "channels": [32, 64, 128], "kernel_sizes": [3, 5, 7], "epochs": args.epochs, "mc_iterations": args.mc_iterations, "seed": args.seed})
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions = {"lr_binary": np.full((len(X), 2), np.nan), "resnet_binary": np.full((len(X), 2), np.nan), "lr_three": np.full((len(X), 3), np.nan), "resnet_three": np.full((len(X), 3), np.nan)}
    history: list[dict[str, JsonValue]] = []
    fold_rows: list[dict[str, JsonValue]] = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y_binary, groups=patient_ids), start=1):
        inner_train, inner_val = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold).split(X[train_index], y_binary[train_index], groups=patient_ids[train_index]))
        for task_name, y_values, key, n_classes in (("binary", y_binary, "binary", 2), ("three", y_three, "three", 3)):
            train_mean, train_labels, train_patients = patient_training_means(X, y_values, patient_ids, train_index)
            lr_model, best_c = fit_logistic(train_mean, train_labels, train_patients, 200 + fold)
            predictions[f"lr_{key}"][test_index] = lr_model.predict_proba(X[test_index])
            resnet_fit = fit_resnet(X[train_index][inner_train], y_values[train_index][inner_train], X[train_index][inner_val], y_values[train_index][inner_val], n_classes, args.n_blocks, args.position_aware, args.seed + fold * 10 + n_classes, args.epochs, device)
            predictions[f"resnet_{key}"][test_index] = _resnet_predict(resnet_fit.model, X[test_index], device)
            history.extend({"fold": fold, "task": task_name, "model": "Multi-scale ResNet", **row} for row in resnet_fit.history)
            fold_rows.append({"fold": fold, "task": task_name, "best_c": best_c, "best_epoch": resnet_fit.best_epoch, "n_train_patients": len(np.unique(patient_ids[train_index])), "n_test_patients": len(np.unique(patient_ids[test_index]))})
            del resnet_fit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(out / "fold_summary.csv", fold_rows)
    save_training_history(out, history)
    np.savez_compressed(out / "oof_prediction_arrays.npz", **predictions)
    metrics, aggregates = patient_oof_rows(patient_ids, y_binary, y_three, predictions)
    write_csv(out / "oof_metrics.csv", metrics)
    patient_rows: list[dict[str, JsonValue]] = []
    for (task, model, aggregation), (ids, y_true, probability) in aggregates.items():
        for index, patient in enumerate(ids):
            row: dict[str, JsonValue] = {"task": task, "model": model, "aggregation": aggregation, "subject_ordinal": int(patient), "true_label": int(y_true[index])}
            row.update({f"probability_class_{class_index}": float(probability[index, class_index]) for class_index in range(probability.shape[1])})
            patient_rows.append(row)
    write_csv(out / "patient_oof_predictions.csv", patient_rows)
    repeat_rows, stability_rows, minimum_rows = repeat_evaluation(patient_ids, y_binary, y_three, predictions, args.mc_iterations, args.seed)
    repeat_summary = summarize_repeat_rows(repeat_rows)
    write_csv(out / "repeat_metrics_mc.csv", repeat_rows)
    write_csv(out / "repeat_metrics_summary.csv", repeat_summary)
    write_csv(out / "prediction_stability.csv", stability_rows)
    write_csv(out / "minimum_repeat_count.csv", [row for row in minimum_rows if "n" in row])
    save_repeat_figures(out, repeat_summary, stability_rows)
    save_roc_figure(out, aggregates)
    write_report(out, metadata, metrics, args)
    print(json.dumps({"output_dir": str(out), "subjects": len(subjects), "rows": len(X), "device": device}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
