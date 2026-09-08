from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from severance_data import CohortSpectra, PreprocessingTrace
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def save_figure(fig: plt.Figure, figure_dir: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(figure_dir / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_preprocessing(traces: tuple[PreprocessingTrace, ...], figure_dir: Path) -> None:
    groups = ("YNOR", "YPAN")
    labels = ("Control", "Pancreatic cancer")
    colors = ("#2C7FB8", "#D95F02")
    stage_names = tuple(stage.name for stage in traces[0].stages)
    fig, axes = plt.subplots(len(stage_names), len(groups), figsize=(12, 12), squeeze=False)
    for row, stage_name in enumerate(stage_names):
        for column, (group, label, color) in enumerate(zip(groups, labels, colors, strict=True)):
            axis = axes[row, column]
            selected = [trace for trace in traces if trace.group == group]
            stages = [trace.stages[row] for trace in selected]
            for stage in stages[:25]:
                axis.plot(stage.x, stage.y, color=color, alpha=0.12, lw=0.8)
            x_reference = stages[0].x
            matrix = np.vstack([np.interp(x_reference, stage.x, stage.y) for stage in stages])
            axis.plot(x_reference, matrix.mean(axis=0), color=color, lw=1.6)
            axis.set_title(f"{label} (raw n={len(selected)})" if row == 0 else "")
            axis.set_ylabel(stage_name)
            axis.grid(alpha=0.2)
    fig.suptitle("Fig01. Severance prospective cohort preprocessing audit", y=0.995, fontsize=15)
    save_figure(fig, figure_dir, "fig01_severance_preprocessing_pipeline")


def plot_clean_pan_comparison(cohort: CohortSpectra, figure_dir: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    items = (
        ("Clean retrospective CPAN", cohort.clean_cpan, "#4D4D4D"),
        ("Severance prospective YPAN", cohort.ypan, "#D95F02"),
    )
    for label, matrix, color in items:
        mean = matrix.mean(axis=0)
        sem = matrix.std(axis=0, ddof=1) / np.sqrt(len(matrix))
        axes[0].plot(cohort.grid, mean, label=f"{label} (n={len(matrix)})", color=color, lw=1.7)
        axes[0].fill_between(
            cohort.grid, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.18
        )
    difference = cohort.ypan.mean(axis=0) - cohort.clean_cpan.mean(axis=0)
    axes[1].plot(cohort.grid, difference, color="#B2182B", lw=1.5)
    axes[0].set_title("Same preprocessing: clean CPAN vs Severance YPAN")
    axes[0].legend(frameon=False)
    axes[1].set_title("Mean difference: Severance YPAN - clean CPAN")
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    for axis in axes:
        axis.set_ylabel("SNV intensity")
        axis.grid(alpha=0.2)
    save_figure(fig, figure_dir, "fig02_severance_vs_legacy_pan_processed")


def plot_screening_performance(predictions: Path, figure_dir: Path) -> None:
    with predictions.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    labels = np.asarray([int(row["true_label"]) for row in rows], dtype=int)
    probability = np.asarray([float(row["mean_probability"]) for row in rows], dtype=float)
    predicted = (probability >= 0.5).astype(int)
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    consensus_auc = roc_auc_score(labels, probability)
    with (predictions.parent / "severance_lr_repeated_cv_auc.csv").open(
        encoding="utf-8-sig"
    ) as handle:
        repeated_auc = np.asarray(
            [float(row["pooled_oof_auroc"]) for row in csv.DictReader(handle)], dtype=float
        )
    metrics = (
        repeated_auc.mean(),
        consensus_auc,
        accuracy_score(labels, predicted),
        recall_score(labels, predicted),
        tn / (tn + fp),
    )
    fpr, tpr, _ = roc_curve(labels, probability)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    axes[0].imshow(matrix, cmap="Blues")
    axes[0].set_xticks((0, 1), ("Control", "Cancer"))
    axes[0].set_yticks((0, 1), ("Control", "Cancer"))
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Screening confusion matrix")
    for row in range(2):
        for column in range(2):
            axes[0].text(column, row, str(matrix[row, column]), ha="center", va="center")
    axes[1].plot(fpr, tpr, color="#D95F02", label=f"Consensus OOF AUC={consensus_auc:.3f}")
    axes[1].plot((0, 1), (0, 1), "--", color="#777777", lw=0.8)
    axes[1].set_xlabel("False positive rate")
    axes[1].set_ylabel("True positive rate")
    axes[1].set_title("Screening ROC")
    axes[1].legend(loc="lower right", frameon=False)
    metric_labels = (
        "Repeated CV\nmean AUC",
        "Consensus\nOOF AUC",
        "Accuracy",
        "Sensitivity",
        "Specificity",
    )
    axes[2].bar(np.arange(5), metrics, color="#D95F02", alpha=0.85)
    axes[2].set_xticks(np.arange(5), metric_labels, rotation=25, ha="right")
    axes[2].set_ylim(0, 1.05)
    axes[2].set_title("Optimized LR metrics")
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle("Fig03. Severance Cancer Screening: optimized repeated-CV LR", y=1.02)
    save_figure(fig, figure_dir, "fig03_screening_binary_auc_confusion_matrix")


def plot_stage_difference(
    cohort: CohortSpectra, clinical_mapping: Path, figure_dir: Path, table_dir: Path
) -> None:
    with clinical_mapping.open(encoding="utf-8-sig") as handle:
        stages = {
            row["subject_id"]: row["stage_1_2_or_3_4"]
            for row in csv.DictReader(handle)
            if row["source_group"] == "YPAN"
        }
    early = np.vstack(
        [
            spectrum
            for subject_id, spectrum in zip(cohort.ypan_subject_ids, cohort.ypan, strict=True)
            if stages.get(subject_id) == "1-2"
        ]
    )
    advanced = np.vstack(
        [
            spectrum
            for subject_id, spectrum in zip(cohort.ypan_subject_ids, cohort.ypan, strict=True)
            if stages.get(subject_id) == "3-4"
        ]
    )
    with (table_dir / "fig05_pancreatic_stage_group_counts.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["stage_group", "n_subjects"])
        writer.writerows((("Stage 1-2", len(early)), ("Stage 3-4", len(advanced))))
    combined = np.vstack((early, advanced))
    pca = PCA(n_components=2, random_state=42).fit_transform(combined)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].scatter(
        pca[: len(early), 0],
        pca[: len(early), 1],
        color="#1B9E77",
        label=f"Stage 1-2 (n={len(early)})",
    )
    axes[0].scatter(
        pca[len(early) :, 0],
        pca[len(early) :, 1],
        color="#D95F02",
        label=f"Stage 3-4 (n={len(advanced)})",
    )
    axes[0].set_title("PCA of YPAN spectra")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False)
    axes[1].plot(
        cohort.grid, early.mean(axis=0), color="#1B9E77", label=f"Stage 1-2 (n={len(early)})"
    )
    axes[1].plot(
        cohort.grid, advanced.mean(axis=0), color="#D95F02", label=f"Stage 3-4 (n={len(advanced)})"
    )
    axes[1].set_title("Mean spectra by clinical stage")
    axes[1].legend(frameon=False)
    difference = advanced.mean(axis=0) - early.mean(axis=0)
    axes[2].plot(cohort.grid, difference, color="#B2182B")
    peak_indices, _ = find_peaks(
        np.abs(difference), distance=10, prominence=max(0.05, float(np.ptp(difference)) * 0.05)
    )
    top_indices = peak_indices[np.argsort(np.abs(difference[peak_indices]))[-8:]]
    axes[2].scatter(cohort.grid[top_indices], difference[top_indices], color="#B2182B", s=18)
    axes[2].set_title("Difference: Stage 3-4 - Stage 1-2")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[2].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("Fig05. Severance pancreatic cancer internal subgroup check by stage", y=1.02)
    save_figure(fig, figure_dir, "fig05_pancreatic_stage_group_difference")
