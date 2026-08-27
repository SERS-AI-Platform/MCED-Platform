#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "scikit-learn", "matplotlib", "openpyxl", "joblib", "torch"]
# ///
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "patent"))
sys.path.insert(0, str(REPO / "scripts" / "deployment"))

from mapping_resnet_data import (  # noqa: E402
    CLASS_NAMES,
    GROUP_ORDER,
    GROUP_TO_CLASS,
    PreparedSubject,
    build_rows,
    patient_training_means,
    preprocess_subjects,
    write_csv,
)
from mapping_resnet_evaluation import (  # noqa: E402
    aggregate_probabilities,
    metric_row,
    patient_oof_rows,
)
from mapping_resnet_model import (  # noqa: E402
    BLOCK_CHOICES,
    MultiScaleResidualBlock,
    MultiScaleResNet,
    ResNetFit,
    _auc,
    _resnet_predict,
    fit_logistic,
    fit_resnet,
    set_seed,
)
from mapping_resnet_outputs import (  # noqa: E402
    save_repeat_figures,
    save_roc_figure,
    save_training_history,
)
from mapping_resnet_reference import run_locked_stk_reference, write_report  # noqa: E402
from mapping_resnet_repeat import (  # noqa: E402
    CANDIDATE_N,
    minimum_repeat_rows,
    repeat_evaluation,
    summarize_repeat_rows,
)

DEFAULT_OUT = REPO / "results" / "mapping_multiscale_resnet_20260825_v1"

__all__ = [
    "BLOCK_CHOICES",
    "CLASS_NAMES",
    "CANDIDATE_N",
    "GROUP_ORDER",
    "GROUP_TO_CLASS",
    "MultiScaleResidualBlock",
    "MultiScaleResNet",
    "PreparedSubject",
    "ResNetFit",
    "_auc",
    "_resnet_predict",
    "aggregate_probabilities",
    "build_rows",
    "fit_logistic",
    "fit_resnet",
    "minimum_repeat_rows",
    "metric_row",
    "patient_oof_rows",
    "patient_training_means",
    "preprocess_subjects",
    "repeat_evaluation",
    "run_locked_stk_reference",
    "save_repeat_figures",
    "save_roc_figure",
    "save_training_history",
    "set_seed",
    "summarize_repeat_rows",
    "write_csv",
    "write_report",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mc-iterations", type=int, default=100)
    parser.add_argument("--n-blocks", type=int, choices=BLOCK_CHOICES, default=3)
    parser.add_argument("--position-aware", action="store_true")
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--skip-stk", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    grid = np.linspace(402.0, 2198.0, 933, dtype=float)
    subjects, metadata = preprocess_subjects(grid)
    X, patient_ids, y_binary, y_three, repeat_ids = build_rows(subjects)
    np.savez_compressed(out / "preprocessed_mapping_arrays.npz", X=X, patient_ids=patient_ids, y_binary=y_binary, y_three=y_three, repeat_ids=repeat_ids)
    metadata.update({"device": device, "n_blocks": args.n_blocks, "blocks_per_stage": args.n_blocks // 3, "position_aware": args.position_aware, "channels": [32, 64, 128], "kernel_sizes": [3, 5, 7], "epochs": args.epochs, "mc_iterations": args.mc_iterations, "seed": args.seed})
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    predictions = {
        "lr_binary": np.full((len(X), 2), np.nan, dtype=float),
        "resnet_binary": np.full((len(X), 2), np.nan, dtype=float),
        "lr_three": np.full((len(X), 3), np.nan, dtype=float),
        "resnet_three": np.full((len(X), 3), np.nan, dtype=float),
    }
    history: list[dict[str, object]] = []
    fold_rows: list[dict[str, object]] = []
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_index, test_index) in enumerate(splitter.split(X, y_binary, groups=patient_ids), start=1):
        inner = next(StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=100 + fold).split(X[train_index], y_binary[train_index], groups=patient_ids[train_index]))
        inner_train, inner_val = inner
        for task_name, y_values, key, n_classes in (("binary", y_binary, "binary", 2), ("three", y_three, "three", 3)):
            train_y = y_values[train_index]
            val_y = y_values[train_index][inner_val]
            lr_X, lr_y, lr_groups = patient_training_means(X, y_values, patient_ids, train_index)
            lr_model, best_c = fit_logistic(lr_X, lr_y, lr_groups, 200 + fold)
            lr_probability = lr_model.predict_proba(X[test_index])
            predictions[f"lr_{key}"][test_index] = lr_probability
            resnet_fit = fit_resnet(
                X[train_index][inner_train],
                train_y[inner_train],
                X[train_index][inner_val],
                val_y,
                n_classes,
                args.n_blocks,
                args.position_aware,
                args.seed + fold * 10 + n_classes,
                args.epochs,
                device,
            )
            predictions[f"resnet_{key}"][test_index] = _resnet_predict(resnet_fit.model, X[test_index], device)
            for row in resnet_fit.history:
                history.append({"fold": fold, "task": task_name, "model": "Multi-scale ResNet", **row})
            fold_rows.append({"fold": fold, "task": task_name, "best_c": best_c, "best_epoch": resnet_fit.best_epoch, "n_train_patients": len(np.unique(patient_ids[train_index])), "n_test_patients": len(np.unique(patient_ids[test_index]))})
            del resnet_fit
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    write_csv(out / "fold_summary.csv", fold_rows)
    save_training_history(out, history)
    np.savez_compressed(out / "oof_prediction_arrays.npz", **predictions)
    metrics, aggregates = patient_oof_rows(patient_ids, y_binary, y_three, predictions)
    write_csv(out / "oof_metrics.csv", metrics)
    oof_rows: list[dict[str, object]] = []
    for (task, model, aggregation), (ids, y_true, probability) in aggregates.items():
        for index, patient in enumerate(ids):
            row: dict[str, object] = {"task": task, "model": model, "aggregation": aggregation, "subject_ordinal": int(patient), "true_label": int(y_true[index])}
            if probability.shape[1] == 2:
                row["probability_class_1"] = float(probability[index, 1])
            else:
                row.update({f"probability_class_{class_index}": float(probability[index, class_index]) for class_index in range(probability.shape[1])})
            oof_rows.append(row)
    write_csv(out / "patient_oof_predictions.csv", oof_rows)
    repeat_rows, stability_rows, minimum_rows = repeat_evaluation(patient_ids, y_binary, y_three, predictions, args.mc_iterations, args.seed)
    write_csv(out / "repeat_metrics_mc.csv", repeat_rows)
    repeat_summary = summarize_repeat_rows(repeat_rows)
    write_csv(out / "repeat_metrics_summary.csv", repeat_summary)
    write_csv(out / "prediction_stability.csv", stability_rows)
    write_csv(out / "minimum_repeat_count.csv", [row for row in minimum_rows if "n" in row])
    save_repeat_figures(out, repeat_summary, stability_rows)
    save_roc_figure(out, aggregates)
    stk_summary = {"status": "skipped", "reason": "--skip-stk"} if args.skip_stk else run_locked_stk_reference(subjects, out)
    write_report(out, metadata, metrics, minimum_rows, stk_summary, args)
    print(json.dumps({"output_dir": str(out), "subjects": len(subjects), "rows": len(X), "device": device, "stk_v2": stk_summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
