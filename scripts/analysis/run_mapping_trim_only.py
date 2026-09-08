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

DEFAULT_OUT = REPO / "results" / "mapping_trim_only_20260826_v1"
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]


class TrimOnlyDataError(ValueError):
    pass


def prepare_trim_only(grid: np.ndarray) -> tuple[list[PreparedSubject], dict[str, JsonValue]]:
    subjects, raw_axis = load_subjects(grid)
    prepared: list[PreparedSubject] = []
    native_counts: list[int] = []
    finite_counts: list[int] = []
    for subject in subjects:
        native_axis = np.asarray(subject.native_axis, dtype=float)
        mask = (native_axis >= 400.0) & (native_axis <= 2200.0)
        x_trim = native_axis[mask]
        if len(x_trim) < 2:
            raise TrimOnlyDataError(f"subject {subject.ordinal} has too few points after trim")
        rows = np.vstack(
            [np.interp(grid, x_trim, np.asarray(row, dtype=float)[mask]) for row in subject.native_replicates]
        ).astype(np.float32)
        finite = np.isfinite(rows).all(axis=1)
        rows = rows[finite]
        if len(rows) == 0:
            raise TrimOnlyDataError(f"subject {subject.ordinal} has no finite trim-only spectra")
        native_counts.append(len(subject.native_replicates))
        finite_counts.append(len(rows))
        prepared.append(
            PreparedSubject(
                ordinal=subject.ordinal,
                group=subject.group,
                x=grid.copy(),
                all_spectra=rows,
                qc_keep=np.ones(len(rows), dtype=bool),
                qc_corr=np.full(len(rows), np.nan, dtype=float),
                qc_rsd_pct=float("nan"),
            )
        )
    metadata: dict[str, JsonValue] = {
        "raw_axis": raw_axis,
        "subject_count": len(prepared),
        "group_counts": {group: sum(item.group == group for item in prepared) for group in GROUP_ORDER},
        "input_repeats": int(sum(native_counts)),
        "model_repeats": int(sum(finite_counts)),
        "finite_repeats": int(sum(finite_counts)),
        "nonfinite_repeats_removed": int(sum(native_counts) - sum(finite_counts)),
        "qc_rule": "No spectral QC filtering; finite-value gate only. All finite raw repeats remain in the locked evaluation.",
        "grid": {
            "min_cm-1": float(grid[0]),
            "max_cm-1": float(grid[-1]),
            "points": int(len(grid)),
            "step_cm-1": float(np.median(np.diff(grid))),
        },
        "preprocessing": {
            "trim": [400.0, 2200.0],
            "smooth": None,
            "baseline": None,
            "normalization": None,
            "alignment": "linear interpolation onto fixed 402-2198 cm-1 grid for model input shape only",
        },
    }
    return prepared, metadata


def write_report(out: Path, metadata: dict[str, JsonValue], metrics: Sequence[dict[str, JsonValue]], args: argparse.Namespace) -> None:
    rows = [row for row in metrics if row.get("aggregation") == "mean"]

    def value(row: dict[str, JsonValue], key: str) -> str:
        number = float(row.get(key, np.nan))
        return "-" if not np.isfinite(number) else f"{number:.4f}"

    lines = [
        "# Mapping cohort trim-only ablation",
        "",
        "## 실험 정의",
        "",
        f"- 대상: {metadata['subject_count']}명, {metadata['model_repeats']}개 finite raw repeat.",
        "- 전처리: 400–2200 cm⁻¹ trim만 적용. smoothing, baseline correction, SNV, robust QC filtering은 적용하지 않았다.",
        "- 933-point 입력 shape를 위해 trim된 raw intensity를 402–2198 cm⁻¹ 공통 grid에 선형 보간했다. 이는 shape alignment일 뿐 신호 보정이 아니다.",
        f"- ResNet: Multi-scale 1D ResNet, {args.n_blocks} blocks, position-aware={args.position_aware}, epochs={args.epochs}.",
        "- 검증: patient-level StratifiedGroupKFold 5-fold OOF, group=patient ordinal.",
        "- LR-reference는 patient mean spectrum으로 학습하고, ResNet은 각 finite repeat를 학습에 사용했다.",
        "",
        "## Patient mean aggregation OOF",
        "",
        "| Task | Model | ROC-AUC | Macro ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {value(row, 'roc_auc')} | {value(row, 'macro_roc_auc')} | "
            f"{value(row, 'balanced_accuracy')} | {value(row, 'sensitivity')} | {value(row, 'specificity')} | {value(row, 'macro_f1')} |"
        )
    lines.extend(
        [
            "",
            "## 해석 주의사항",
            "",
            "- Cancer Screening 성능은 병원·측정 조건 confounding 가능성이 있으므로 외부 일반화 성능으로 해석하지 않는다.",
            "- 최소 반복 횟수 판정과 Monte-Carlo 결과는 `minimum_repeat_count.csv`, `repeat_metrics_summary.csv`, `prediction_stability.csv`에 저장했다.",
        ]
    )
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
    grid = np.linspace(402.0, 2198.0, 933, dtype=float)
    subjects, metadata = prepare_trim_only(grid)
    X, patient_ids, y_binary, y_three, repeat_ids = build_rows(subjects)
    np.savez_compressed(
        out / "preprocessed_trim_only_arrays.npz",
        X=X,
        patient_ids=patient_ids,
        y_binary=y_binary,
        y_three=y_three,
        repeat_ids=repeat_ids,
    )
    metadata.update(
        {
            "device": device,
            "n_blocks": args.n_blocks,
            "blocks_per_stage": args.n_blocks // 3,
            "position_aware": args.position_aware,
            "channels": [32, 64, 128],
            "kernel_sizes": [3, 5, 7],
            "epochs": args.epochs,
            "mc_iterations": args.mc_iterations,
            "seed": args.seed,
        }
    )
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions = {
        "lr_binary": np.full((len(X), 2), np.nan),
        "resnet_binary": np.full((len(X), 2), np.nan),
        "lr_three": np.full((len(X), 3), np.nan),
        "resnet_three": np.full((len(X), 3), np.nan),
    }
    history: list[dict[str, JsonValue]] = []
    fold_rows: list[dict[str, JsonValue]] = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y_binary, groups=patient_ids), start=1):
        inner_train, inner_val = next(
            StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold).split(
                X[train_index], y_binary[train_index], groups=patient_ids[train_index]
            )
        )
        for task_name, y_values, key, n_classes in (
            ("binary", y_binary, "binary", 2),
            ("three", y_three, "three", 3),
        ):
            train_mean, train_labels, train_patients = patient_training_means(X, y_values, patient_ids, train_index)
            lr_model, best_c = fit_logistic(train_mean, train_labels, train_patients, 200 + fold)
            predictions[f"lr_{key}"][test_index] = lr_model.predict_proba(X[test_index])
            resnet_fit = fit_resnet(
                X[train_index][inner_train],
                y_values[train_index][inner_train],
                X[train_index][inner_val],
                y_values[train_index][inner_val],
                n_classes,
                args.n_blocks,
                args.position_aware,
                args.seed + fold * 10 + n_classes,
                args.epochs,
                device,
            )
            predictions[f"resnet_{key}"][test_index] = _resnet_predict(resnet_fit.model, X[test_index], device)
            history.extend({"fold": fold, "task": task_name, "model": "Multi-scale ResNet", **row} for row in resnet_fit.history)
            fold_rows.append(
                {
                    "fold": fold,
                    "task": task_name,
                    "best_c": best_c,
                    "best_epoch": resnet_fit.best_epoch,
                    "n_train_patients": len(np.unique(patient_ids[train_index])),
                    "n_test_patients": len(np.unique(patient_ids[test_index])),
                }
            )
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
            row: dict[str, JsonValue] = {
                "task": task,
                "model": model,
                "aggregation": aggregation,
                "subject_ordinal": int(patient),
                "true_label": int(y_true[index]),
            }
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
