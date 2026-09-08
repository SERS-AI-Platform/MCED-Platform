from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = (
    REPO_ROOT / "scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py"
)
OUTPUT_DIR = REPO_ROOT / "notebooks/aecd_api_model_all_qc_spectra_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
PUBLICATION_DIR = (
    REPO_ROOT / "publications/전향검체/보라매병원/figures/all_qc_spectra"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
PUBLICATION_DIR.mkdir(parents=True, exist_ok=True)
DECISION_THRESHOLD = 0.4
OUTER_FOLDS = 5
INNER_FOLDS = 5


def load_pipeline():
    spec = importlib.util.spec_from_file_location("mean_spectrum_pipeline", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the mean-spectrum pipeline module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def flatten_qc_passed_repeats(
    pipeline,
    subject_keys: list[str],
    labels: list[str],
    replicate_matrices: list[np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, np.ndarray],
    np.ndarray,
]:
    spectra = []
    row_subject_indices = []
    row_repeat_indices = []
    row_labels = []
    qc_rows = []
    noise_vectors = []
    for subject_index, (subject_key, label, replicates) in enumerate(
        zip(subject_keys, labels, replicate_matrices, strict=True)
    ):
        keep, distances, limit = pipeline.qc_repeats(replicates)
        passed_indices = np.flatnonzero(keep)
        passed = replicates[keep]
        representative = passed.mean(axis=0)
        residuals = passed - representative
        noise_sigma = np.std(residuals, axis=0, ddof=1)
        noise_vectors.append(noise_sigma)
        for repeat_index in passed_indices:
            spectra.append(replicates[int(repeat_index)])
            row_subject_indices.append(subject_index)
            row_repeat_indices.append(int(repeat_index))
            row_labels.append(label)
        qc_rows.append(
            {
                "subject_index_internal": int(subject_index),
                "subject_key": str(subject_key),
                "label": str(label),
                "repeats_total": int(len(replicates)),
                "repeats_passed": int(len(passed_indices)),
                "repeats_excluded": int(len(replicates) - len(passed_indices)),
                "qc_pass_rate": float(np.mean(keep)),
                "qc_distance_median": float(np.median(distances)),
                "qc_distance_max": float(np.max(distances)),
                "qc_distance_limit": float(limit),
                "noise_floor_global_median": float(np.median(noise_sigma)),
                "noise_sigma_mean": float(np.mean(noise_sigma)),
                "noise_sigma_p95": float(np.percentile(noise_sigma, 95)),
            }
        )
    X = np.asarray(spectra, dtype=np.float32)
    subject_indices = np.asarray(row_subject_indices, dtype=np.int64)
    repeat_indices = np.asarray(row_repeat_indices, dtype=np.int64)
    labels_array = np.asarray(row_labels)
    counts = np.bincount(subject_indices, minlength=len(subject_keys)).astype(np.float64)
    sample_weights = 1.0 / counts[subject_indices]
    return (
        X,
        subject_indices,
        repeat_indices,
        labels_array,
        pd.DataFrame(qc_rows),
        np.vstack(noise_vectors),
    ), sample_weights


def make_base_models(pipeline):
    base_specs = pipeline.make_direct_base_models()
    return [
        (name.replace("raw_mean", "all_qc"), factory)
        for name, factory in base_specs
    ]


def fit_weighted_model(name: str, factory, X, y, sample_weight):
    model = factory()
    if name.startswith("lr_") or name.startswith("ridge_"):
        model.fit(
            X,
            y,
            logisticregression__sample_weight=sample_weight,
        )
    else:
        model.fit(X, y, sample_weight=sample_weight)
    return model


def aggregate_subject_predictions(
    subject_indices: np.ndarray,
    y: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "subject_index_internal": subject_indices,
            "y_true": y,
            "probability": probability,
        }
    )
    return (
        frame.groupby("subject_index_internal", as_index=False)
        .agg(
            y_true=("y_true", "first"),
            probability=("probability", "mean"),
            qc_spectrum_count=("probability", "size"),
        )
        .sort_values("subject_index_internal")
        .reset_index(drop=True)
    )


def run_nested_all_qc_stacking(
    pipeline,
    X: np.ndarray,
    y: np.ndarray,
    subject_indices: np.ndarray,
    sample_weights: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    groups = subject_indices
    base_specs = make_base_models(pipeline)
    outer = GroupKFold(n_splits=OUTER_FOLDS)
    oof_probability = np.full(len(y), np.nan, dtype=np.float64)
    fold_rows = []
    for fold, (train_index, test_index) in enumerate(
        outer.split(X, y, groups),
        start=1,
    ):
        X_train = X[train_index]
        y_train = y[train_index]
        groups_train = groups[train_index]
        weights_train = sample_weights[train_index]
        inner = GroupKFold(n_splits=INNER_FOLDS)
        meta_train = np.zeros((len(train_index), len(base_specs)), dtype=np.float64)
        for model_index, (name, factory) in enumerate(base_specs):
            oof_column = np.zeros(len(train_index), dtype=np.float64)
            for inner_train, inner_test in inner.split(
                X_train,
                y_train,
                groups_train,
            ):
                model = fit_weighted_model(
                    name,
                    factory,
                    X_train[inner_train],
                    y_train[inner_train],
                    weights_train[inner_train],
                )
                oof_column[inner_test] = model.predict_proba(
                    X_train[inner_test]
                )[:, 1]
            meta_train[:, model_index] = oof_column
        meta = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            random_state=pipeline.RANDOM_STATE,
        )
        meta.fit(meta_train, y_train, sample_weight=weights_train)
        meta_test = np.zeros((len(test_index), len(base_specs)), dtype=np.float64)
        for model_index, (name, factory) in enumerate(base_specs):
            model = fit_weighted_model(
                name,
                factory,
                X_train,
                y_train,
                weights_train,
            )
            meta_test[:, model_index] = model.predict_proba(X[test_index])[:, 1]
        oof_probability[test_index] = meta.predict_proba(meta_test)[:, 1]
        subject_test = aggregate_subject_predictions(
            groups[test_index],
            y[test_index],
            oof_probability[test_index],
        )
        fold_prediction = (
            subject_test["probability"].to_numpy() >= DECISION_THRESHOLD
        ).astype(int)
        fold_rows.append(
            {
                "fold": int(fold),
                "n_train_subjects": int(np.unique(groups[train_index]).size),
                "n_test_subjects": int(np.unique(groups[test_index]).size),
                "n_train_spectra": int(len(train_index)),
                "n_test_spectra": int(len(test_index)),
                "test_positive_subjects": int(subject_test["y_true"].sum()),
                "test_negative_subjects": int((1 - subject_test["y_true"]).sum()),
                "spectrum_auc": float(
                    roc_auc_score(y[test_index], oof_probability[test_index])
                ),
                "subject_auc": float(
                    roc_auc_score(subject_test["y_true"], subject_test["probability"])
                ),
                "subject_balanced_accuracy": float(
                    balanced_accuracy_score(subject_test["y_true"], fold_prediction)
                ),
            }
        )
        print(
            f"  [outer {fold}] subject_auc={fold_rows[-1]['subject_auc']:.4f} "
            f"spectrum_auc={fold_rows[-1]['spectrum_auc']:.4f}"
        )
    if np.isnan(oof_probability).any():
        raise RuntimeError("Nested CV did not produce a prediction for every spectrum.")
    subject_predictions = aggregate_subject_predictions(
        subject_indices,
        y,
        oof_probability,
    )
    return oof_probability, subject_predictions, pd.DataFrame(fold_rows)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def plot_subject_roc(subject_predictions: pd.DataFrame, path: Path) -> None:
    y = subject_predictions["y_true"].to_numpy()
    probability = subject_predictions["probability"].to_numpy()
    fpr, tpr, _ = roc_curve(y, probability)
    auc = roc_auc_score(y, probability)
    fig, axis = plt.subplots(figsize=(6.5, 6.0))
    axis.plot(fpr, tpr, color="#7b2cbf", linewidth=2, label=f"Subject OOF ROC (AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], color="#777777", linestyle="--", linewidth=1)
    axis.set_xlabel("False-positive rate")
    axis.set_ylabel("True-positive rate")
    axis.set_title("All QC-passed spectra, subject-level clinical performance")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_qc_counts(qc_frame: pd.DataFrame, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.hist(
        qc_frame["repeats_passed"],
        bins=np.arange(qc_frame["repeats_passed"].min() - 0.5, qc_frame["repeats_passed"].max() + 1.5, 1),
        color="#457b9d",
        alpha=0.8,
    )
    axis.set_xlabel("QC-passed spectra per subject")
    axis.set_ylabel("Subjects")
    axis.set_title("QC-passed repeat count used as model inputs")
    axis.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_subject_repeat_scores(
    subject_predictions: pd.DataFrame,
    labels: np.ndarray,
    path: Path,
) -> None:
    colors = {0: "#457b9d", 1: "#9d0208"}
    frame = subject_predictions.copy()
    frame["label"] = labels[frame["subject_index_internal"].to_numpy()]
    fig, axis = plt.subplots(figsize=(10, 5))
    for label, subset in frame.groupby("label"):
        axis.scatter(
            subset["qc_spectrum_count"],
            subset["probability"],
            color=colors[int(label)],
            alpha=0.8,
            s=30,
            label="prostate" if int(label) else "non-prostate",
        )
    axis.axhline(DECISION_THRESHOLD, color="#777777", linestyle="--", linewidth=1)
    axis.set_xlabel("QC-passed spectra per subject")
    axis.set_ylabel("Mean OOF probability")
    axis.set_title("Subject-level score after aggregating all repeat predictions")
    axis.legend(frameon=False)
    axis.grid(alpha=0.18)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    pipeline = load_pipeline()
    items, cohorts, api_metadata = pipeline.load_api_spectra()
    common_grid, observed_step, common_points = pipeline.construct_common_grid(items)
    subject_keys, labels, replicate_matrices, alignment_metadata = (
        pipeline.group_and_align_subjects(items, common_grid)
    )
    flattened, sample_weights = flatten_qc_passed_repeats(
        pipeline,
        subject_keys,
        labels,
        replicate_matrices,
    )
    X, subject_indices, repeat_indices, row_labels, qc_frame, noise_vectors = flattened
    labels_array = np.asarray(labels)
    y = (row_labels == pipeline.POSITIVE_LABEL).astype(int)
    subject_y = (labels_array == pipeline.POSITIVE_LABEL).astype(int)
    oof_probability, subject_predictions, fold_metrics = run_nested_all_qc_stacking(
        pipeline,
        X,
        y,
        subject_indices,
        sample_weights,
    )
    spectrum_prediction = (oof_probability >= DECISION_THRESHOLD).astype(int)
    subject_prediction = (
        subject_predictions["probability"].to_numpy() >= DECISION_THRESHOLD
    ).astype(int)
    subject_true = subject_predictions["y_true"].to_numpy()
    confusion = confusion_matrix(subject_true, subject_prediction, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()
    metrics = {
        "method": "all_qc_passed_aligned_spectra_with_subject_level_aggregation",
        "model_input_spectra": int(len(X)),
        "independent_subjects": int(len(labels_array)),
        "threshold": DECISION_THRESHOLD,
        "spectrum_oof_auc_secondary": float(roc_auc_score(y, oof_probability)),
        "subject_oof_auc_primary": float(
            roc_auc_score(subject_true, subject_predictions["probability"])
        ),
        "subject_balanced_accuracy": float(
            balanced_accuracy_score(subject_true, subject_prediction)
        ),
        "subject_f1": float(f1_score(subject_true, subject_prediction, zero_division=0)),
        "subject_precision": float(
            precision_score(subject_true, subject_prediction, zero_division=0)
        ),
        "subject_recall": float(
            recall_score(subject_true, subject_prediction, zero_division=0)
        ),
        "fold_subject_auc_mean": float(fold_metrics["subject_auc"].mean()),
        "fold_subject_auc_std": float(fold_metrics["subject_auc"].std()),
        "fold_spectrum_auc_mean": float(fold_metrics["spectrum_auc"].mean()),
        "fold_spectrum_auc_std": float(fold_metrics["spectrum_auc"].std()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }
    total_repeats = int(sum(len(matrix) for matrix in replicate_matrices))
    passed_repeats = int(len(X))
    qc_summary = {
        "subjects_total": int(len(subject_keys)),
        "total_repeats": total_repeats,
        "passed_repeats": passed_repeats,
        "excluded_repeats": int(total_repeats - passed_repeats),
        "repeat_pass_rate": float(passed_repeats / total_repeats),
        "qc_rule": "robust standardized RMS spectral distance from within-subject median; median/MAD cutoff",
        "qc_mad_threshold": pipeline.QC_MAD_THRESHOLD,
        "minimum_passed_repeats": 2,
    }
    subject_predictions.insert(
        1,
        "label",
        labels_array[subject_predictions["subject_index_internal"].to_numpy()],
    )
    spectrum_predictions = pd.DataFrame(
        {
            "subject_index_internal": subject_indices,
            "repeat_index": repeat_indices,
            "label": row_labels,
            "y_true": y,
            "oof_probability": oof_probability,
            "prediction": spectrum_prediction,
        }
    )
    manifest = spectrum_predictions.drop(columns=["y_true", "oof_probability", "prediction"])
    noise_summary = pd.DataFrame(
        {
            "wavenumber_cm1": common_grid,
            "noise_sigma_median": np.median(noise_vectors, axis=0),
            "noise_sigma_p25": np.percentile(noise_vectors, 25, axis=0),
            "noise_sigma_p75": np.percentile(noise_vectors, 75, axis=0),
            "subject_count": len(noise_vectors),
        }
    )
    save_csv(pd.DataFrame([metrics]), OUTPUT_DIR / "clinical_performance_summary.csv")
    save_csv(fold_metrics, OUTPUT_DIR / "clinical_fold_metrics.csv")
    save_csv(pd.DataFrame([qc_summary]), OUTPUT_DIR / "qc_summary.csv")
    save_csv(qc_frame, OUTPUT_DIR / "subject_qc_noise_summary.csv")
    save_csv(noise_summary, OUTPUT_DIR / "noise_summary.csv")
    save_csv(spectrum_predictions, OUTPUT_DIR / "oof_spectrum_predictions.csv")
    save_csv(subject_predictions, OUTPUT_DIR / "oof_subject_predictions.csv")
    save_csv(manifest, OUTPUT_DIR / "qc_passed_spectrum_manifest.csv")
    save_csv(cohorts, OUTPUT_DIR / "api_cohort_summary.csv")
    save_csv(
        pd.DataFrame([alignment_metadata["wavenumber_calibration"]]),
        OUTPUT_DIR / "wavenumber_calibration_summary.csv",
    )
    np.savez_compressed(
        OUTPUT_DIR / "qc_passed_aligned_spectra.npz",
        spectra=X,
        common_grid=common_grid.astype(np.float32),
    )
    plot_subject_roc(subject_predictions, FIGURE_DIR / "clinical_roc_subject_level.png")
    plot_qc_counts(qc_frame, FIGURE_DIR / "qc_passed_repeat_count.png")
    plot_subject_repeat_scores(
        subject_predictions,
        subject_y,
        FIGURE_DIR / "subject_score_vs_repeat_count.png",
    )
    publication_figures = {}
    for source_name, target_name in {
        "clinical_roc_subject_level.png": "clinical_roc_subject_level.png",
        "qc_passed_repeat_count.png": "qc_passed_repeat_count.png",
        "subject_score_vs_repeat_count.png": "subject_score_vs_repeat_count.png",
    }.items():
        target = PUBLICATION_DIR / target_name
        shutil.copy2(FIGURE_DIR / source_name, target)
        publication_figures[target_name] = str(target)
    metadata = {
        "data_source": "AECD REST API",
        "source_mean_pipeline": str(SOURCE_SCRIPT),
        "cohort": {
            "subjects_total": int(len(subject_keys)),
            "positive_subjects": int(subject_y.sum()),
            "negative_subjects": int((1 - subject_y).sum()),
            "labels": sorted(set(labels)),
        },
        "input": {
            "model_input": "all QC-passed reference-aligned spectra",
            "spectra_count": int(len(X)),
            "subject_count": int(len(subject_keys)),
            "features_per_spectrum": int(X.shape[1]),
            "subject_weighting": "each QC-passed spectrum has weight 1 / QC-passed spectrum count for its subject",
            "mean_spectrum_used_as_model_input": False,
            "raw_aligned_intensity_used": True,
        },
        "qc": qc_summary,
        "noise_estimation": {
            "used_for_model_input": False,
            "used_for_qc_and_reporting": True,
            "definition": "per-wavenumber SD of QC-passed repeat residuals around the subject mean spectrum",
        },
        "evaluation": {
            "split_unit": "subject",
            "outer_group_kfold": OUTER_FOLDS,
            "inner_group_kfold": INNER_FOLDS,
            "primary_metric": "subject-level OOF AUC after averaging repeat probabilities",
            "secondary_metric": "spectrum-level OOF AUC",
        },
        "model": {
            "base_models": [
                "StandardScaler + LogisticRegression(C=1.0)",
                "XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05)",
                "RandomForestClassifier(n_estimators=400)",
                "StandardScaler + LogisticRegression(C=0.1)",
            ],
            "meta_model": "elastic-net LogisticRegression",
        },
        "clinical_metrics": metrics,
        "publication_figures": publication_figures,
        "interpretation_limit": "Internal same-cohort evaluation; not cross-hospital generalization.",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# All QC-passed spectrum clinical evaluation",
                "",
                f"The model used all {len(X)} QC-passed aligned spectra as input.",
                "Repeated spectra from one subject were kept in the same GroupKFold split.",
                "Primary clinical metrics aggregate repeat probabilities to one score per subject.",
                "The subject-level OOF AUC is the primary metric; spectrum-level AUC is secondary.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print("=== All QC-passed spectrum clinical evaluation ===")
    print(f"[data] subjects={len(subject_keys)} spectra={len(X)} X={X.shape}")
    print(f"[QC] passed={passed_repeats}/{total_repeats} ({qc_summary['repeat_pass_rate']:.2%})")
    print(
        f"[clinical] subject-level OOF AUC={metrics['subject_oof_auc_primary']:.4f} "
        f"spectrum-level OOF AUC={metrics['spectrum_oof_auc_secondary']:.4f}"
    )
    print(f"[output] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
