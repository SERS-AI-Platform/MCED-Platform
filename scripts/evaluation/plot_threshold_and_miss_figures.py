#!/usr/bin/env python3
"""Create threshold-sweep and miss-analysis figures for current STK-V2 outputs."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment.sers_predict import StackingPredictor
from scripts.evaluation.eval_current_model_acquisition_sets import collect_records
from scripts.evaluation.eval_reference_normalized_balanced_lot import (
    THERMO_BORAMAE_ROOT,
    THERMO_STANDARD_ROOT,
    build_day_calibrations,
    preprocess_records,
)

DEFAULT_INPUT = PROJECT_ROOT / "results" / "evaluation" / "reference_normalized_balanced_lot"
DEFAULT_OUT = PROJECT_ROOT / "results" / "evaluation" / "threshold_miss_analysis"
VARIANTS = ("raw", "axis", "axis_intensity", "axis_intensity_blank")
COHORTS = {
    "thermo_standard": lambda df: df["source"].eq("balanced_lot_thermo_standard"),
    "thermo_boramae": lambda df: df["source"].eq("balanced_lot_thermo_boramae_20260602"),
    "thermo_all": lambda df: df["source"].isin(
        ["balanced_lot_thermo_standard", "balanced_lot_thermo_boramae_20260602"]
    ),
}
CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
DEFAULT_THRESHOLD = 0.60
CATEGORY_COLORS = {
    "TP cancer detected": "#2F855A",
    "FN cancer missed": "#2B6CB0",
    "TN non-cancer negative": "#4A5568",
    "FP non-cancer overcalled": "#C9473D",
}


def threshold_grid() -> np.ndarray:
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 1.0, 201),
                np.linspace(0.90, 0.999, 200),
                np.array([0.48, 0.60, 0.70, 0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 0.999]),
            ]
        )
    )
    return np.round(grid, 6)


def metrics_at_threshold(df: pd.DataFrame, threshold: float) -> dict[str, object]:
    y = df["true_binary"].astype(int).to_numpy()
    p = df["cancer_probability"].astype(float).to_numpy()
    pred = (p > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = recall_score(y, pred, zero_division=0)
    spec = recall_score(1 - y, 1 - pred, zero_division=0)
    return {
        "threshold": float(threshold),
        "n": int(len(df)),
        "n_cancer": int(y.sum()),
        "n_non_cancer": int((1 - y).sum()),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "youden_j": float(sens + spec - 1.0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def threshold_sweep(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([metrics_at_threshold(df, t) for t in threshold_grid()])


def pick_row(df: pd.DataFrame, mask: pd.Series, sort_cols: list[str]) -> pd.Series | None:
    cand = df[mask].copy()
    if cand.empty:
        return None
    return cand.sort_values(sort_cols, ascending=[False] * len(sort_cols)).iloc[0]


def candidate_thresholds(sweep: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []

    current_idx = (sweep["threshold"] - DEFAULT_THRESHOLD).abs().idxmin()
    current = sweep.loc[current_idx].copy()
    current["rule"] = "current_0.60"
    rows.append(current)

    best_bal = sweep.sort_values(["balanced_accuracy", "f1", "specificity"], ascending=False).iloc[0].copy()
    best_bal["rule"] = "max_balanced_accuracy"
    rows.append(best_bal)

    for target in [0.80, 0.90, 0.95]:
        row = pick_row(sweep, sweep["sensitivity"].ge(target), ["specificity", "balanced_accuracy", "f1"])
        if row is not None:
            row = row.copy()
            row["rule"] = f"max_spec_given_sens_ge_{target:.2f}"
            rows.append(row)

    for target in [0.50, 0.70]:
        row = pick_row(sweep, sweep["specificity"].ge(target), ["sensitivity", "balanced_accuracy", "f1"])
        if row is not None:
            row = row.copy()
            row["rule"] = f"max_sens_given_spec_ge_{target:.2f}"
            rows.append(row)

    out = pd.DataFrame(rows)
    cols = ["rule"] + [c for c in out.columns if c != "rule"]
    return out[cols].drop_duplicates(subset=["rule"])


def load_variant(input_dir: Path, variant: str) -> pd.DataFrame:
    path = input_dir / f"{variant}_per_sample_predictions.csv"
    df = pd.read_csv(path)
    df["variant"] = variant
    return df


def savefig(fig: plt.Figure, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def focus_records(cohort: str):
    records = []
    if cohort in {"thermo_standard", "thermo_all"}:
        records.extend(
            collect_records(
                THERMO_STANDARD_ROOT,
                "balanced_lot_thermo_standard",
                "thermo",
                include_background_only=False,
            )
        )
    if cohort in {"thermo_boramae", "thermo_all"}:
        records.extend(
            collect_records(
                THERMO_BORAMAE_ROOT,
                "balanced_lot_thermo_boramae_20260602",
                "thermo",
                include_background_only=False,
            )
        )
    return records


def load_focus_features(model_dir: Path, variant: str, cohort: str) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    predictor = StackingPredictor(model_dir)
    records = focus_records(cohort)
    if not records:
        return np.empty((0, 3, 0)), pd.DataFrame(), predictor.grid, pd.DataFrame()
    cals = build_day_calibrations(predictor.grid)
    X, meta, errors = preprocess_records(predictor, records, cals, variant)
    return X, meta, predictor.grid, errors


def profile_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.nanmedian(values, axis=0),
        np.nanpercentile(values, 25, axis=0),
        np.nanpercentile(values, 75, axis=0),
    )


def plot_feature_profiles(
    pred_df: pd.DataFrame,
    X: np.ndarray,
    meta: pd.DataFrame,
    grid: np.ndarray,
    threshold: float,
    out: Path,
    title: str,
) -> None:
    if X.size == 0 or meta.empty:
        return

    indexed = meta[["uid"]].copy()
    indexed["_feature_idx"] = np.arange(len(meta))
    work = pred_df.merge(indexed, on="uid", how="inner")
    if work.empty:
        return

    work["pred_at_threshold"] = work["cancer_probability"].gt(threshold).astype(int)
    work["profile_category"] = np.select(
        [
            work["true_binary"].eq(1) & work["pred_at_threshold"].eq(1),
            work["true_binary"].eq(1) & work["pred_at_threshold"].eq(0),
            work["true_binary"].eq(0) & work["pred_at_threshold"].eq(0),
            work["true_binary"].eq(0) & work["pred_at_threshold"].eq(1),
        ],
        [
            "TP cancer detected",
            "FN cancer missed",
            "TN non-cancer negative",
            "FP non-cancer overcalled",
        ],
        default="other",
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    ax = axes[0]
    for category in CATEGORY_COLORS:
        rows = work[work["profile_category"].eq(category)]
        if rows.empty:
            continue
        arr = X[rows["_feature_idx"].to_numpy(dtype=int), 0, :]
        med, q25, q75 = profile_stats(arr)
        ax.plot(grid, med, lw=1.8, color=CATEGORY_COLORS[category], label=f"{category} (n={len(rows)})")
        ax.fill_between(grid, q25, q75, color=CATEGORY_COLORS[category], alpha=0.12, linewidth=0)
    ax.set_title("Model channel 0 profile by binary outcome")
    ax.set_xlabel("Wavenumber (cm-1)")
    ax.set_ylabel("Preprocessed intensity")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1]
    fn = work[work["profile_category"].eq("FN cancer missed")].copy()
    if fn.empty:
        ax.axis("off")
        ax.set_title("Missed cancers by source group (none)")
    else:
        top_groups = fn["source_group"].value_counts().head(6).index.tolist()
        cmap = plt.get_cmap("tab10")
        for i, group in enumerate(top_groups):
            rows = fn[fn["source_group"].eq(group)]
            arr = X[rows["_feature_idx"].to_numpy(dtype=int), 0, :]
            med, q25, q75 = profile_stats(arr)
            color = cmap(i % 10)
            ax.plot(grid, med, lw=1.8, color=color, label=f"{group} FN (n={len(rows)})")
            ax.fill_between(grid, q25, q75, color=color, alpha=0.10, linewidth=0)
        ax.set_title("Missed cancers by source group, model channel 0")
        ax.set_xlabel("Wavenumber (cm-1)")
        ax.set_ylabel("Preprocessed intensity")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=3)

    fig.suptitle(f"{title} | threshold={threshold:.3f}", fontsize=14, fontweight="bold")
    savefig(fig, out)


def add_candidate_lines(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    styles = {
        "current_0.60": ("#444444", "--"),
        "max_balanced_accuracy": ("#C9473D", "-"),
        "max_spec_given_sens_ge_0.90": ("#2B6CB0", ":"),
        "max_spec_given_sens_ge_0.95": ("#6B46C1", ":"),
        "max_sens_given_spec_ge_0.50": ("#2F855A", "-."),
    }
    for _, row in candidates.iterrows():
        rule = str(row["rule"])
        if rule not in styles:
            continue
        color, ls = styles[rule]
        ax.axvline(float(row["threshold"]), color=color, linestyle=ls, lw=1.2, alpha=0.85)


def plot_threshold_sweep(sweep: pd.DataFrame, candidates: pd.DataFrame, out: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    ax = axes[0, 0]
    for col, color in [
        ("sensitivity", "#2B6CB0"),
        ("specificity", "#C9473D"),
        ("balanced_accuracy", "#2F855A"),
        ("f1", "#805AD5"),
    ]:
        ax.plot(sweep["threshold"], sweep[col], label=col, lw=2, color=color)
    add_candidate_lines(ax, candidates)
    ax.set_title("Metrics across threshold")
    ax.set_xlabel("Cancer probability threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    zoom = sweep[sweep["threshold"].ge(0.90)]
    for col, color in [
        ("sensitivity", "#2B6CB0"),
        ("specificity", "#C9473D"),
        ("balanced_accuracy", "#2F855A"),
        ("f1", "#805AD5"),
    ]:
        ax.plot(zoom["threshold"], zoom[col], label=col, lw=2, color=color)
    add_candidate_lines(ax, candidates)
    ax.set_title("High-threshold zoom")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(sweep["threshold"], sweep["fp"], label="False positives", color="#C9473D", lw=2)
    ax.plot(sweep["threshold"], sweep["fn"], label="False negatives", color="#2B6CB0", lw=2)
    add_candidate_lines(ax, candidates)
    ax.set_title("Error counts")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(zoom["threshold"], zoom["fp"], label="False positives", color="#C9473D", lw=2)
    ax.plot(zoom["threshold"], zoom["fn"], label="False negatives", color="#2B6CB0", lw=2)
    add_candidate_lines(ax, candidates)
    ax.set_title("Error counts, high-threshold zoom")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    savefig(fig, out)


def plot_variant_summary(all_sweeps: pd.DataFrame, all_candidates: pd.DataFrame, out: Path) -> None:
    cur = all_sweeps[
        all_sweeps["cohort"].eq("thermo_standard")
        & all_sweeps["threshold"].sub(DEFAULT_THRESHOLD).abs().lt(1e-9)
    ].copy()
    best = all_candidates[
        all_candidates["cohort"].eq("thermo_standard")
        & all_candidates["rule"].eq("max_balanced_accuracy")
    ].copy()

    variants = list(VARIANTS)
    x = np.arange(len(variants))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, frame, title in [
        (axes[0], cur, "Current threshold 0.60"),
        (axes[1], best, "Best balanced-accuracy threshold"),
    ]:
        frame = frame.set_index("variant").loc[variants].reset_index()
        ax.bar(x - width, frame["sensitivity"], width, label="Sensitivity", color="#2B6CB0")
        ax.bar(x, frame["specificity"], width, label="Specificity", color="#C9473D")
        ax.bar(x + width, frame["balanced_accuracy"], width, label="Balanced Acc", color="#2F855A")
        for i, t in enumerate(frame["threshold"]):
            ax.text(i, 1.03, f"t={t:.3f}", ha="center", va="bottom", fontsize=8, rotation=35)
        ax.set_xticks(x)
        ax.set_xticklabels(variants, rotation=25, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    fig.suptitle("Thermo standard balanced lot: variant comparison", fontsize=14, fontweight="bold")
    savefig(fig, out)


def plot_probability_by_group(df: pd.DataFrame, candidates: pd.DataFrame, out: Path, title: str) -> None:
    order = sorted(df["source_group"].astype(str).unique())
    data = [df.loc[df["source_group"].astype(str).eq(g), "cancer_probability"].to_numpy() for g in order]
    colors = ["#D9EAF7" if df.loc[df["source_group"].astype(str).eq(g), "true_binary"].iloc[0] == 0 else "#F6D6D2" for g in order]

    fig, ax = plt.subplots(figsize=(13, 5.5), constrained_layout=True)
    bp = ax.boxplot(data, labels=order, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor("#444444")
    rng = np.random.default_rng(42)
    for i, values in enumerate(data, start=1):
        if len(values) > 120:
            values = rng.choice(values, size=120, replace=False)
        x = rng.normal(i, 0.05, size=len(values))
        ax.scatter(x, values, s=8, alpha=0.35, color="#333333", linewidths=0)
    add_candidate_lines(ax, candidates)
    for _, row in candidates.iterrows():
        if str(row["rule"]) in {"current_0.60", "max_balanced_accuracy"}:
            ax.axhline(float(row["threshold"]), lw=1.1, alpha=0.8)
            ax.text(len(order) + 0.3, float(row["threshold"]), str(row["rule"]), fontsize=8, va="center")
    ax.set_ylim(-0.03, 1.03)
    ax.set_ylabel("Cancer probability")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.25)
    savefig(fig, out)


def matrix_for_error_rate(df: pd.DataFrame, threshold: float, cancer: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = df[df["true_binary"].eq(1 if cancer else 0)].copy()
    if cancer:
        work["error"] = work["cancer_probability"].le(threshold).astype(int)
    else:
        work["error"] = work["cancer_probability"].gt(threshold).astype(int)
    total = work.pivot_table(index="source_group", columns="date_key", values="uid", aggfunc="count", fill_value=0)
    err = work.pivot_table(index="source_group", columns="date_key", values="error", aggfunc="sum", fill_value=0)
    err = err.reindex(index=total.index, columns=total.columns, fill_value=0)
    rate = err.divide(total.where(total > 0), fill_value=0).fillna(0)
    ann = err.astype(int).astype(str) + "/" + total.astype(int).astype(str)
    return rate, ann


def draw_heatmap(ax: plt.Axes, rate: pd.DataFrame, ann: pd.DataFrame, title: str, cmap: str) -> None:
    im = ax.imshow(rate.to_numpy(), vmin=0, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(np.arange(rate.shape[1]))
    ax.set_yticks(np.arange(rate.shape[0]))
    ax.set_xticklabels(rate.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(rate.index, fontsize=8)
    ax.set_title(title)
    for i in range(rate.shape[0]):
        for j in range(rate.shape[1]):
            value = float(rate.iloc[i, j])
            if str(ann.iloc[i, j]) == "0/0":
                text = ""
            else:
                text = f"{ann.iloc[i, j]}\n{value:.0%}"
            ax.text(j, i, text, ha="center", va="center", fontsize=6, color="white" if value > 0.55 else "#222222")
    return im


def plot_error_heatmaps(df: pd.DataFrame, threshold: float, out: Path, title: str) -> None:
    fn_rate, fn_ann = matrix_for_error_rate(df, threshold, cancer=True)
    fp_rate, fp_ann = matrix_for_error_rate(df, threshold, cancer=False)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.5), constrained_layout=True)
    im1 = draw_heatmap(axes[0], fn_rate, fn_ann, "Cancer missed rate (FN / total)", "Reds")
    im2 = draw_heatmap(axes[1], fp_rate, fp_ann, "Non-cancer overcalled rate (FP / total)", "Blues")
    fig.colorbar(im1, ax=axes[0], fraction=0.025, pad=0.01)
    fig.colorbar(im2, ax=axes[1], fraction=0.025, pad=0.01)
    fig.suptitle(f"{title} | threshold={threshold:.3f}", fontsize=14, fontweight="bold")
    savefig(fig, out)


def plot_error_counts_by_threshold(df: pd.DataFrame, sweep: pd.DataFrame, out: Path, title: str) -> None:
    thresholds = sweep["threshold"].to_numpy()
    cancer = df[df["true_binary"].eq(1)].copy()
    non = df[df["true_binary"].eq(0)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, frame, condition, label, groups in [
        (axes[0, 0], cancer, "fn", "Missed cancers", sorted(cancer["source_group"].unique())),
        (axes[1, 0], non, "fp", "Overcalled non-cancers", sorted(non["source_group"].unique())),
    ]:
        for group in groups:
            probs = frame.loc[frame["source_group"].eq(group), "cancer_probability"].to_numpy()
            if condition == "fn":
                counts = np.array([(probs <= t).sum() for t in thresholds])
            else:
                counts = np.array([(probs > t).sum() for t in thresholds])
            ax.plot(thresholds, counts, lw=1.8, label=group)
        ax.axvline(DEFAULT_THRESHOLD, color="#444444", linestyle="--", lw=1)
        ax.set_title(label)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, ncol=2)

    zoom_thresholds = thresholds[thresholds >= 0.90]
    for ax, frame, condition, label, groups in [
        (axes[0, 1], cancer, "fn", "Missed cancers, high-threshold zoom", sorted(cancer["source_group"].unique())),
        (axes[1, 1], non, "fp", "Overcalled non-cancers, high-threshold zoom", sorted(non["source_group"].unique())),
    ]:
        for group in groups:
            probs = frame.loc[frame["source_group"].eq(group), "cancer_probability"].to_numpy()
            if condition == "fn":
                counts = np.array([(probs <= t).sum() for t in zoom_thresholds])
            else:
                counts = np.array([(probs > t).sum() for t in zoom_thresholds])
            ax.plot(zoom_thresholds, counts, lw=1.8, label=group)
        ax.set_title(label)
        ax.set_xlabel("Threshold")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.25)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    savefig(fig, out)


def plot_type_confusion(df: pd.DataFrame, out: Path, title: str) -> None:
    cancer = df[df["true_type"].notna()].copy()
    cm = confusion_matrix(cancer["true_type"].astype(str), cancer["pred_type"].astype(str), labels=CANCER_TYPES)
    denom = cm.sum(axis=1, keepdims=True)
    rate = np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    im = ax.imshow(rate, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(CANCER_TYPES)))
    ax.set_yticks(np.arange(len(CANCER_TYPES)))
    ax.set_xticklabels(CANCER_TYPES, rotation=45, ha="right")
    ax.set_yticklabels(CANCER_TYPES)
    ax.set_xlabel("Predicted type")
    ax.set_ylabel("True type")
    ax.set_title(title)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text = f"{cm[i, j]}\n{rate[i, j]:.0%}" if cm[i].sum() else "0"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="white" if rate[i, j] > 0.55 else "#222222")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, out)


def write_error_case_tables(df: pd.DataFrame, out_dir: Path, threshold: float, prefix: str) -> None:
    pred = df["cancer_probability"].gt(threshold).astype(int)
    work = df.copy()
    work["threshold"] = threshold
    work["pred_at_threshold"] = pred
    cols = [
        "uid", "source", "date_key", "source_group", "group", "sample_id", "sample_key",
        "n_replicates", "true_binary", "cancer_probability", "pred_at_threshold",
        "pred_type", "true_type",
    ]
    fn = work[work["true_binary"].eq(1) & work["pred_at_threshold"].eq(0)].sort_values("cancer_probability")
    fp = work[work["true_binary"].eq(0) & work["pred_at_threshold"].eq(1)].sort_values("cancer_probability", ascending=False)
    type_mis = work[work["true_type"].notna() & ~work["true_type"].astype(str).eq(work["pred_type"].astype(str))].sort_values(
        ["true_type", "pred_type", "cancer_probability"]
    )
    fn[cols].to_csv(out_dir / f"{prefix}_false_negatives_t{threshold:.3f}.csv", index=False)
    fp[cols].to_csv(out_dir / f"{prefix}_false_positives_t{threshold:.3f}.csv", index=False)
    type_mis[cols].to_csv(out_dir / f"{prefix}_type_misclassifications.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for ax, frame, title, color in [
        (axes[0], fn.head(25).iloc[::-1], "Lowest-probability missed cancers", "#2B6CB0"),
        (axes[1], fp.head(25).iloc[::-1], "Highest-probability overcalled non-cancers", "#C9473D"),
    ]:
        if frame.empty:
            ax.axis("off")
            ax.set_title(title + " (none)")
            continue
        labels = frame["source_group"].astype(str) + " " + frame["sample_key"].astype(str) + " | " + frame["date_key"].astype(str)
        ax.barh(np.arange(len(frame)), frame["cancer_probability"], color=color, alpha=0.85)
        ax.set_yticks(np.arange(len(frame)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.axvline(threshold, color="#222222", linestyle="--", lw=1)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("Cancer probability")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle(f"Error case examples at threshold={threshold:.3f}", fontsize=14, fontweight="bold")
    savefig(fig, out_dir / f"{prefix}_top_error_cases_t{threshold:.3f}.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "usersnet" / "current")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--focus-variant", default="raw", choices=list(VARIANTS))
    parser.add_argument("--focus-cohort", default="thermo_standard", choices=list(COHORTS))
    parser.add_argument("--skip-feature-profiles", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    all_sweep_rows = []
    all_candidate_rows = []
    variant_frames = {variant: load_variant(args.input_dir, variant) for variant in VARIANTS}

    for variant, frame in variant_frames.items():
        for cohort, mask_fn in COHORTS.items():
            sub = frame[mask_fn(frame)].copy()
            sweep = threshold_sweep(sub)
            sweep["variant"] = variant
            sweep["cohort"] = cohort
            candidates = candidate_thresholds(sweep)
            candidates["variant"] = variant
            candidates["cohort"] = cohort
            all_sweep_rows.append(sweep)
            all_candidate_rows.append(candidates)

    all_sweeps = pd.concat(all_sweep_rows, ignore_index=True)
    all_candidates = pd.concat(all_candidate_rows, ignore_index=True)
    all_sweeps.to_csv(args.output_dir / "threshold_sweep_summary.csv", index=False)
    all_candidates.to_csv(args.output_dir / "threshold_candidate_summary.csv", index=False)

    focus_df = variant_frames[args.focus_variant]
    focus_df = focus_df[COHORTS[args.focus_cohort](focus_df)].copy()
    focus_sweep = all_sweeps[
        all_sweeps["variant"].eq(args.focus_variant) & all_sweeps["cohort"].eq(args.focus_cohort)
    ].copy()
    focus_candidates = all_candidates[
        all_candidates["variant"].eq(args.focus_variant) & all_candidates["cohort"].eq(args.focus_cohort)
    ].copy()

    prefix = f"{args.focus_variant}_{args.focus_cohort}"
    plot_threshold_sweep(
        focus_sweep,
        focus_candidates,
        fig_dir / f"{prefix}_threshold_sweep.png",
        f"{args.focus_variant} | {args.focus_cohort}: threshold sweep",
    )
    plot_variant_summary(all_sweeps, all_candidates, fig_dir / "thermo_standard_variant_threshold_summary.png")
    plot_probability_by_group(
        focus_df,
        focus_candidates,
        fig_dir / f"{prefix}_probability_by_source_group.png",
        f"{args.focus_variant} | {args.focus_cohort}: cancer probability by source group",
    )

    best_t = float(
        focus_candidates.loc[focus_candidates["rule"].eq("max_balanced_accuracy"), "threshold"].iloc[0]
    )
    feature_errors = pd.DataFrame()
    feature_X = np.empty((0, 3, 0))
    feature_meta = pd.DataFrame()
    feature_grid = np.array([])
    if not args.skip_feature_profiles:
        feature_X, feature_meta, feature_grid, feature_errors = load_focus_features(
            args.model_dir,
            args.focus_variant,
            args.focus_cohort,
        )

    for threshold, label in [(DEFAULT_THRESHOLD, "current"), (best_t, "best_balanced")]:
        plot_error_heatmaps(
            focus_df,
            threshold,
            fig_dir / f"{prefix}_{label}_error_heatmaps.png",
            f"{args.focus_variant} | {args.focus_cohort}",
        )
        write_error_case_tables(focus_df, args.output_dir, threshold, f"{prefix}_{label}")
        if not args.skip_feature_profiles:
            plot_feature_profiles(
                focus_df,
                feature_X,
                feature_meta,
                feature_grid,
                threshold,
                fig_dir / f"{prefix}_{label}_feature_profiles_t{threshold:.3f}.png",
                f"{args.focus_variant} | {args.focus_cohort}: error feature profiles",
            )

    plot_error_counts_by_threshold(
        focus_df,
        focus_sweep,
        fig_dir / f"{prefix}_error_counts_by_threshold.png",
        f"{args.focus_variant} | {args.focus_cohort}: error counts by group",
    )
    plot_type_confusion(
        focus_df,
        fig_dir / f"{prefix}_stage2_type_confusion.png",
        f"{args.focus_variant} | {args.focus_cohort}: Stage 2 type confusion",
    )

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "focus_variant": args.focus_variant,
        "focus_cohort": args.focus_cohort,
        "default_threshold": DEFAULT_THRESHOLD,
        "best_balanced_threshold": best_t,
        "n_feature_profile_preprocess_errors": int(len(feature_errors)) if not feature_errors.empty else 0,
        "notes": [
            "Threshold sweeps reuse saved current-model sample-level probabilities; no model retraining was performed.",
            "False negatives are cancer samples with probability <= threshold.",
            "False positives are non-cancer samples with probability > threshold.",
            "Feature profile figures rebuild the same sample-level 3-channel tensors and plot channel 0 median/IQR profiles.",
        ],
        "focus_candidates": focus_candidates.to_dict(orient="records"),
    }
    with open(args.output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(focus_candidates.to_string(index=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
