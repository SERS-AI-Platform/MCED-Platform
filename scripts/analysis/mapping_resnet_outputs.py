from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mapping_resnet_data import write_csv
from mapping_resnet_model import _auc


def save_training_history(out: Path, history: Sequence[dict[str, object]]) -> None:
    write_csv(out / "training_history.csv", list(history))
    for task in ("binary", "three"):
        task_rows = [row for row in history if row.get("task") == task]
        if not task_rows:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for model in ("Multi-scale ResNet",):
            for fold in sorted({int(row["fold"]) for row in task_rows}):
                rows = [row for row in task_rows if row.get("model") == model and int(row["fold"]) == fold]
                if not rows:
                    continue
                epochs = [int(row["epoch"]) for row in rows]
                axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], alpha=0.55, label=f"fold {fold} train")
                axes[0].plot(epochs, [float(row["validation_loss"]) for row in rows], alpha=0.85, linestyle="--", label=f"fold {fold} val")
                axes[1].plot(epochs, [float(row["train_auc"]) for row in rows], alpha=0.55)
                axes[1].plot(epochs, [float(row["validation_auc"]) for row in rows], alpha=0.85, linestyle="--")
        axes[0].set(title=f"{task}: train/validation loss", xlabel="epoch", ylabel="loss")
        axes[1].set(title=f"{task}: train/validation AUC", xlabel="epoch", ylabel="AUC")
        axes[0].legend(fontsize=7, ncol=2)
        fig.savefig(out / f"figure_training_{task}.png", dpi=180)
        plt.close(fig)


def save_roc_figure(out: Path, aggregates: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    task = "cancer_vs_non_cancer"
    fig, axis = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    for model, color in (("LR-reference", "#4C78A8"), ("Multi-scale ResNet", "#F58518"), ("LR+ResNet fixed blend", "#54A24B")):
        _, y_true, prob = aggregates[(task, model, "mean")]
        fpr = []
        tpr = []
        thresholds = np.linspace(0, 1, 201)
        for threshold in thresholds:
            pred = (prob[:, 1] >= threshold).astype(int)
            fpr.append(float(np.mean(pred[y_true == 0])))
            tpr.append(float(np.mean(pred[y_true == 1])))
        axis.plot(fpr, tpr, label=f"{model} AUC={_auc(y_true, prob):.3f}", color=color)
    axis.plot([0, 1], [0, 1], color="#999999", linestyle="--", linewidth=1)
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Patient-level OOF ROC")
    axis.legend(fontsize=8)
    fig.savefig(out / "figure_3_oof_roc.png", dpi=180)
    plt.close(fig)


def save_repeat_figures(out: Path, summary_rows: Sequence[dict[str, object]], stability_rows: Sequence[dict[str, object]]) -> None:
    main = [row for row in summary_rows if row.get("aggregation") == "mean"]
    for metric, filename, title in (
        ("roc_auc", "figure_4_repeat_roc_auc.png", "Repeat count vs ROC-AUC"),
        ("balanced_accuracy", "figure_5_repeat_balanced_accuracy.png", "Repeat count vs balanced accuracy"),
    ):
        fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
        for model, color in (("LR-reference", "#4C78A8"), ("Multi-scale ResNet", "#F58518"), ("LR+ResNet fixed blend", "#54A24B")):
            rows = [row for row in main if row.get("task") == "cancer_vs_non_cancer" and row.get("model") == model]
            by_n = defaultdict(list)
            for row in rows:
                by_n[int(row["n"])].append(float(row[f"{metric}_mean"]))
            ns = sorted(by_n)
            axis.plot(ns, [np.nanmean(by_n[n]) for n in ns], marker="o", label=model, color=color)
        axis.set(xlabel="number of spectra", ylabel=metric, title=title)
        axis.legend(fontsize=8)
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for model, color in (("LR-reference", "#4C78A8"), ("Multi-scale ResNet", "#F58518"), ("LR+ResNet fixed blend", "#54A24B")):
        rows = [row for row in stability_rows if row.get("task") == "cancer_vs_non_cancer" and row.get("model") == model and row.get("aggregation") == "mean"]
        axis.plot([int(row["n"]) for row in rows], [float(row["probability_sd_mean"]) for row in rows], marker="o", label=model, color=color)
    axis.set(xlabel="number of spectra", ylabel="mean probability SD", title="Prediction stability")
    axis.legend(fontsize=8)
    fig.savefig(out / "figure_6_prediction_stability.png", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for model, color in (("LR-reference", "#4C78A8"), ("Multi-scale ResNet", "#F58518"), ("LR+ResNet fixed blend", "#54A24B")):
        rows = [row for row in main if row.get("task") == "cancer_vs_non_cancer" and row.get("model") == model]
        by_n = defaultdict(list)
        for row in rows:
            by_n[int(row["n"])].append(float(row["roc_auc_mean"]))
        ns = sorted(by_n)
        values = [np.nanmean(by_n[n]) for n in ns]
        gains = [0.0] + [values[i] - values[i - 1] for i in range(1, len(values))]
        axis.plot(ns, gains, marker="o", label=model, color=color)
    axis.axhline(0, color="#999999", linewidth=1)
    axis.set(xlabel="number of spectra", ylabel="incremental AUC gain", title="Marginal improvement by repeat count")
    axis.legend(fontsize=8)
    fig.savefig(out / "figure_7_repeat_improvement.png", dpi=180)
    plt.close(fig)
