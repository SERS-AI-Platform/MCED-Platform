#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from aecd_api_model_3class_engine import (
    CLASS_INDEX,
    CLASS_NAMES,
    evaluate_condition,
    save_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
OUTPUT_DIR = NOTEBOOK_DIR / "aecd_api_model_3class_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PipelineLoadError(RuntimeError):
    pass


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PipelineLoadError(f"Could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    mean_module = load_module(
        "mean_spectrum_pipeline",
        REPO_ROOT / "scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py",
    )
    dwt_module = load_module(
        "patent_dwt_pipeline",
        NOTEBOOK_DIR / "aecd_api_model_patent_dwt_clinical_performance.py",
    )
    items, _cohorts, _api_metadata = mean_module.load_api_spectra()
    calibrations, _calibration_metadata = (
        mean_module.load_standard_material_calibrations()
    )
    grid, _step, _points = mean_module.construct_common_grid(items)
    _subject_keys, labels, replicates, _alignment = mean_module.group_and_align_subjects(
        items, grid, calibrations
    )
    labels_array = np.asarray([CLASS_INDEX[label] for label in labels], dtype=np.int64)
    groups = np.arange(len(labels_array), dtype=np.int64)
    mean_values, _qc_rows, _noise = mean_module.build_mean_spectra(replicates, grid)
    direct_specs = [
        (name, "raw", factory)
        for name, factory in mean_module.make_direct_base_models()
    ]
    results = []
    folds = []
    matrices = []
    summary, fold_rows, matrix = evaluate_condition(
        "current direct QC-passed subject mean",
        {"raw": mean_values},
        labels_array,
        groups,
        direct_specs,
    )
    results.append(summary)
    folds.extend(fold_rows)
    matrices.append((summary["condition"], matrix))
    save_rows(results, OUTPUT_DIR / "clinical_3class_checkpoint_summary.csv")
    print(f"[3-class] completed: {summary['condition']}", flush=True)

    dwt_values = []
    max_level = dwt_module.pywt.dwt_max_level(
        len(grid), dwt_module.pywt.Wavelet(dwt_module.DWT_WAVELET).dec_len
    )
    for subject_replicates in replicates:
        dwt_mean, _quality, _thresholds, _example = dwt_module.patent_preprocess_subject(
            subject_replicates,
            dwt_module.DWT_WAVELET,
            dwt_module.DWT_MODE,
            max_level,
        )
        dwt_values.append(dwt_mean)
    raw_values = np.vstack([subject_replicates.mean(axis=0) for subject_replicates in replicates])
    raw_stk_views, _channels = dwt_module.build_stkv2_views(raw_values, grid)
    dwt_stk_views, _channels = dwt_module.build_stkv2_views(
        np.vstack(dwt_values), grid
    )
    import train_stkv2_stacking

    stk_specs = train_stkv2_stacking.make_base_models()
    for condition, views in (
        ("DWT comparison raw subject mean", raw_stk_views),
        ("Patent DWT preprocessing", dwt_stk_views),
    ):
        summary, fold_rows, matrix = evaluate_condition(
            condition, views, labels_array, groups, stk_specs
        )
        results.append(summary)
        folds.extend(fold_rows)
        matrices.append((condition, matrix))
        save_rows(results, OUTPUT_DIR / "clinical_3class_checkpoint_summary.csv")
        print(f"[3-class] completed: {condition}", flush=True)

    qc_spectra = []
    qc_groups = []
    qc_labels = []
    for subject_index, subject_replicates in enumerate(replicates):
        keep, _distances, _limit = mean_module.qc_repeats(subject_replicates)
        qc_spectra.extend(subject_replicates[keep])
        qc_groups.extend([subject_index] * int(keep.sum()))
        qc_labels.extend([labels_array[subject_index]] * int(keep.sum()))
    all_qc_values = np.asarray(qc_spectra, dtype=np.float64)
    all_qc_groups = np.asarray(qc_groups, dtype=np.int64)
    all_qc_labels = np.asarray(qc_labels, dtype=np.int64)
    all_qc_weights = 1.0 / np.bincount(all_qc_groups)[all_qc_groups]
    summary, fold_rows, matrix = evaluate_condition(
        "current all QC-passed spectra",
        {"raw": all_qc_values},
        all_qc_labels,
        all_qc_groups,
        direct_specs,
        all_qc_weights,
    )
    summary["model_input_spectra"] = int(len(all_qc_values))
    results.append(summary)
    folds.extend(fold_rows)
    matrices.append((summary["condition"], matrix))
    save_rows(results, OUTPUT_DIR / "clinical_3class_checkpoint_summary.csv")
    print(f"[3-class] completed: {summary['condition']}", flush=True)

    save_rows(results, OUTPUT_DIR / "clinical_3class_performance_summary.csv")
    save_rows(folds, OUTPUT_DIR / "clinical_3class_fold_metrics.csv")
    matrix_rows = []
    for condition, matrix in matrices:
        for true_index, row in enumerate(matrix):
            for predicted_index, count in enumerate(row):
                matrix_rows.append(
                    {
                        "condition": condition,
                        "true_class": CLASS_NAMES[true_index],
                        "predicted_class": CLASS_NAMES[predicted_index],
                        "count": int(count),
                    }
                )
    save_rows(matrix_rows, OUTPUT_DIR / "clinical_3class_confusion_matrices.csv")
    metadata = {
        "classes": CLASS_NAMES,
        "cohort_subjects": int(len(labels_array)),
        "class_counts": {
            name: int(np.sum(labels_array == index))
            for name, index in CLASS_INDEX.items()
        },
        "input_conditions": [row["condition"] for row in results],
        "evaluation": "5x5 nested GroupKFold; all-QC repeat probabilities averaged per subject",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[data] subjects={len(labels_array)} qc_spectra={len(all_qc_values)}")
    for row in results:
        print(
            f"[3-class] {row['condition']}: macro_auc={row['macro_ovr_auc']:.4f} "
            f"bacc={row['balanced_accuracy']:.4f} macro_f1={row['macro_f1']:.4f}"
        )
    print(f"[output] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
