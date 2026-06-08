#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate per-device figures for group-level and sample-level matching."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("SERS_ORGANIZED_DATA_ROOT", REPO_ROOT / "data" / "sers_clinical_organized")).expanduser()
INPUT_DIR = ROOT / "06_analysis_results" / "primary_to_lot_balanced_same_id_test_axis_aligned"
INPUT_CSV = INPUT_DIR / "sample_majority_predictions.csv"
OUT_DIR = ROOT / "06_analysis_results" / "group_sample_match_by_device"
FIG_DIR = OUT_DIR / "figures"

FEATURE = "vector_derivative_snv"
DATASET_DISPLAY = {
    "primary_leave_one_replicate_cv": "Primary Thermo CV",
    "lot_balanced_thermo": "Thermo",
    "lot_balanced_handheld": "Handheld",
    "lot_balanced_medical": "Medical",
}
DATASET_ORDER = ["Primary Thermo CV", "Thermo", "Handheld", "Medical"]
DEVICE_ORDER = ["Thermo", "Handheld", "Medical"]
GROUP_ORDER = ["NOR", "YNOR", "H.D.", "YPAN", "CPAN", "SPAN", "PRO", "BLC", "BRE", "CRC", "DIA", "HBP", "LUN", "OVA"]
STATUS_ORDER = ["Exact sample match", "Group only", "Wrong group"]
STATUS_COLORS = {
    "Exact sample match": "#2ca02c",
    "Group only": "#ffbf00",
    "Wrong group": "#d62728",
}


def sample_sort_key(sample_id: str) -> tuple[str, int]:
    group, number = sample_id.rsplit(" ", 1)
    try:
        num = int(number)
    except ValueError:
        num = 0
    return group, num


def prepare_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_CSV)
    df = df[(df["feature"] == FEATURE)].copy()
    df = df[
        (df["candidate_mode"] == "dataset_overlap_only")
        | (df["test_dataset"] == "primary_leave_one_replicate_cv")
    ].copy()
    df["dataset_label"] = df["test_dataset"].map(DATASET_DISPLAY).fillna(df["test_dataset"])
    df["pred_group"] = df["majority_pred_sample_id"].astype(str).str.rsplit(" ", n=1).str[0]
    df["true_sample_number"] = df["true_sample_id"].astype(str).str.rsplit(" ", n=1).str[1].astype(int)
    df["pred_sample_number"] = df["majority_pred_sample_id"].astype(str).str.rsplit(" ", n=1).str[1].astype(int)
    df["match_status"] = np.select(
        [
            df["majority_correct_identity"].astype(bool),
            df["majority_correct_group"].astype(bool),
        ],
        ["Exact sample match", "Group only"],
        default="Wrong group",
    )
    df["match_status"] = pd.Categorical(df["match_status"], categories=STATUS_ORDER, ordered=True)
    df["dataset_label"] = pd.Categorical(df["dataset_label"], categories=DATASET_ORDER, ordered=True)
    df["true_group"] = pd.Categorical(df["true_group"], categories=[g for g in GROUP_ORDER if g in set(df["true_group"])], ordered=True)
    return df.sort_values(["dataset_label", "true_group", "true_sample_number"])


def write_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        df.groupby(["dataset_label"], observed=True)
        .agg(
            n_samples=("true_sample_id", "count"),
            sample_accuracy=("majority_correct_identity", "mean"),
            group_accuracy=("majority_correct_group", "mean"),
            median_true_rank=("median_true_rank", "median"),
            median_vote_fraction=("majority_vote_fraction", "median"),
        )
        .reset_index()
    )
    status = (
        df.groupby(["dataset_label", "match_status"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    totals = status.groupby("dataset_label", observed=True)["n"].transform("sum")
    status["fraction"] = status["n"] / totals
    summary.to_csv(OUT_DIR / "device_group_sample_summary.csv", index=False)
    status.to_csv(OUT_DIR / "device_match_status_counts.csv", index=False)

    by_group = (
        df.groupby(["dataset_label", "true_group"], observed=True)
        .agg(
            n_samples=("true_sample_id", "count"),
            sample_accuracy=("majority_correct_identity", "mean"),
            group_accuracy=("majority_correct_group", "mean"),
            median_true_rank=("median_true_rank", "median"),
        )
        .reset_index()
    )
    by_group.to_csv(OUT_DIR / "device_group_level_accuracy.csv", index=False)
    df.to_csv(OUT_DIR / "device_sample_level_predictions.csv", index=False)
    return summary, by_group


def plot_status_summary(df: pd.DataFrame, summary: pd.DataFrame) -> None:
    status = (
        df.groupby(["dataset_label", "match_status"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    status["fraction"] = status["n"] / status.groupby("dataset_label", observed=True)["n"].transform("sum")
    pivot = status.pivot(index="dataset_label", columns="match_status", values="fraction").fillna(0)
    pivot = pivot.reindex([label for label in DATASET_ORDER if label in pivot.index])
    pivot = pivot.reindex(columns=STATUS_ORDER).fillna(0)

    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    bottom = np.zeros(len(pivot))
    x = np.arange(len(pivot))
    for status_name in STATUS_ORDER:
        vals = pivot[status_name].to_numpy()
        ax.bar(x, vals, bottom=bottom, label=status_name, color=STATUS_COLORS[status_name], width=0.62)
        for idx, val in enumerate(vals):
            if val >= 0.045:
                ax.text(idx, bottom[idx] + val / 2, f"{val:.1%}", ha="center", va="center", fontsize=9)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of true samples")
    ax.set_title("Per-device outcome: exact sample match, group-only match, or wrong group")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, frameon=False)

    for idx, label in enumerate(pivot.index):
        row = summary[summary["dataset_label"].astype(str) == str(label)]
        if row.empty:
            continue
        ax.text(
            idx,
            1.025,
            f"n={int(row.iloc[0]['n_samples'])}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.subplots_adjust(bottom=0.24, left=0.08, right=0.98, top=0.88)
    fig.savefig(FIG_DIR / "01_device_group_then_sample_status_stacked.png", dpi=220)
    plt.close(fig)

    metric_rows = []
    for _, row in summary.iterrows():
        metric_rows.extend(
            [
                {"dataset_label": row["dataset_label"], "metric": "Group accuracy", "accuracy": row["group_accuracy"]},
                {"dataset_label": row["dataset_label"], "metric": "Exact sample accuracy", "accuracy": row["sample_accuracy"]},
            ]
        )
    metric_df = pd.DataFrame(metric_rows)
    fig, ax = plt.subplots(figsize=(12.4, 6.4))
    sns.barplot(
        data=metric_df,
        x="dataset_label",
        y="accuracy",
        hue="metric",
        order=[label for label in DATASET_ORDER if label in set(metric_df["dataset_label"].astype(str))],
        ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Sample-majority accuracy")
    ax.set_title("Group matching is also low, and exact sample matching is near zero")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, frameon=False)
    fig.subplots_adjust(bottom=0.25, left=0.08, right=0.98, top=0.88)
    fig.savefig(FIG_DIR / "02_device_group_vs_exact_sample_accuracy.png", dpi=220)
    plt.close(fig)


def plot_group_confusion(df: pd.DataFrame) -> None:
    devices = [label for label in DEVICE_ORDER if label in set(df["dataset_label"].astype(str))]
    fig, axes = plt.subplots(1, len(devices), figsize=(6.8 * len(devices), 6.8), squeeze=False)
    for ax, device in zip(axes[0], devices):
        subset = df[df["dataset_label"].astype(str) == device].copy()
        table = pd.crosstab(subset["true_group"].astype(str), subset["pred_group"].astype(str))
        table = table.reindex(index=[g for g in GROUP_ORDER if g in table.index], fill_value=0)
        table = table.reindex(columns=[g for g in GROUP_ORDER if g in table.columns], fill_value=0)
        row_sum = table.sum(axis=1).replace(0, np.nan)
        pct = table.div(row_sum, axis=0)
        labels = table.astype(str)
        for r in table.index:
            for c in table.columns:
                labels.loc[r, c] = f"{pct.loc[r, c]:.2f}\n{table.loc[r, c]}" if table.loc[r, c] else ""
        sns.heatmap(pct, annot=labels, fmt="", vmin=0, vmax=1, cmap="mako", ax=ax, cbar=device == devices[-1])
        ax.set_title(device)
        ax.set_xlabel("Predicted group")
        ax.set_ylabel("True group" if device == devices[0] else "")
    fig.suptitle("Group confusion by device: cell = row fraction and count")
    fig.subplots_adjust(bottom=0.15, left=0.06, right=0.99, top=0.86, wspace=0.35)
    fig.savefig(FIG_DIR / "03_group_confusion_by_device.png", dpi=220)
    plt.close(fig)


def plot_status_by_group(df: pd.DataFrame) -> None:
    devices = [label for label in DEVICE_ORDER if label in set(df["dataset_label"].astype(str))]
    fig, axes = plt.subplots(len(devices), 1, figsize=(13.5, 4.4 * len(devices)), sharex=True)
    if len(devices) == 1:
        axes = [axes]
    for ax, device in zip(axes, devices):
        subset = df[df["dataset_label"].astype(str) == device].copy()
        status = (
            subset.groupby(["true_group", "match_status"], observed=True)
            .size()
            .rename("n")
            .reset_index()
        )
        status["fraction"] = status["n"] / status.groupby("true_group", observed=True)["n"].transform("sum")
        pivot = status.pivot(index="true_group", columns="match_status", values="fraction").fillna(0)
        pivot = pivot.reindex([g for g in GROUP_ORDER if g in pivot.index])
        pivot = pivot.reindex(columns=STATUS_ORDER).fillna(0)
        bottom = np.zeros(len(pivot))
        x = np.arange(len(pivot))
        for status_name in STATUS_ORDER:
            vals = pivot[status_name].to_numpy()
            ax.bar(x, vals, bottom=bottom, label=status_name, color=STATUS_COLORS[status_name], width=0.78)
            bottom += vals
        ax.set_ylim(0, 1)
        ax.set_ylabel(device)
        ax.set_title(f"{device}: group first, then exact sample match status")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=0)
        ax.grid(axis="y", alpha=0.25)
    axes[-1].set_xlabel("True group")
    axes[0].legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.08, left=0.07, right=0.99, top=0.91, hspace=0.42)
    fig.savefig(FIG_DIR / "04_match_status_by_true_group_and_device.png", dpi=220)
    plt.close(fig)


def plot_sample_grid(df: pd.DataFrame) -> None:
    devices = [label for label in DEVICE_ORDER if label in set(df["dataset_label"].astype(str))]
    groups = [g for g in GROUP_ORDER if g in set(df["true_group"].astype(str))]
    fig, axes = plt.subplots(len(devices), 1, figsize=(15, 4.9 * len(devices)), sharex=False)
    if len(devices) == 1:
        axes = [axes]
    for ax, device in zip(axes, devices):
        subset = df[df["dataset_label"].astype(str) == device].copy()
        rows = []
        for y, group in enumerate(groups):
            gdf = subset[subset["true_group"].astype(str) == group].sort_values("true_sample_number")
            for x, (_, row) in enumerate(gdf.iterrows()):
                rows.append(
                    {
                        "x": x,
                        "y": y,
                        "group": group,
                        "status": row["match_status"],
                        "true_sample_id": row["true_sample_id"],
                        "pred_sample_id": row["majority_pred_sample_id"],
                        "rank": row["median_true_rank"],
                    }
                )
        plot_df = pd.DataFrame(rows)
        for status_name in STATUS_ORDER:
            s = plot_df[plot_df["status"].astype(str) == status_name]
            ax.scatter(
                s["x"],
                s["y"],
                s=16,
                c=STATUS_COLORS[status_name],
                label=status_name,
                alpha=0.82,
                linewidth=0,
            )
        ax.set_yticks(np.arange(len(groups)))
        ax.set_yticklabels(groups)
        ax.set_xlabel("Samples sorted within each true group")
        ax.set_ylabel("")
        ax.set_title(f"{device}: every dot is one true sample")
        ax.set_xlim(-2, max(10, int(plot_df["x"].max()) + 2))
        ax.grid(axis="x", alpha=0.15)
    axes[0].legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=3, frameon=False)
    fig.suptitle("Sample-level status grid by device")
    fig.subplots_adjust(bottom=0.06, left=0.06, right=0.99, top=0.92, hspace=0.36)
    fig.savefig(FIG_DIR / "05_sample_level_match_status_grid_by_device.png", dpi=220)
    plt.close(fig)


def plot_rank_by_group(df: pd.DataFrame) -> None:
    devices = [label for label in DEVICE_ORDER if label in set(df["dataset_label"].astype(str))]
    subset = df[df["dataset_label"].astype(str).isin(devices)].copy()
    subset["median_true_rank"] = pd.to_numeric(subset["median_true_rank"], errors="coerce")
    fig, axes = plt.subplots(len(devices), 1, figsize=(14, 4.5 * len(devices)), sharex=True)
    if len(devices) == 1:
        axes = [axes]
    for ax, device in zip(axes, devices):
        s = subset[subset["dataset_label"].astype(str) == device].copy()
        sns.boxplot(
            data=s,
            x="true_group",
            y="median_true_rank",
            order=[g for g in GROUP_ORDER if g in set(s["true_group"].astype(str))],
            showfliers=False,
            color="#a6bddb",
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_ylabel("True sample rank\nlog scale")
        ax.set_title(f"{device}: lower rank means the true sample is closer")
        ax.axhline(1, color="0.2", linewidth=1, linestyle="--")
        ax.grid(axis="y", which="both", alpha=0.25)
    axes[-1].set_xlabel("True group")
    fig.subplots_adjust(bottom=0.08, left=0.08, right=0.99, top=0.94, hspace=0.42)
    fig.savefig(FIG_DIR / "06_true_sample_rank_by_group_and_device.png", dpi=220)
    plt.close(fig)


def plot_group_sample_accuracy_scatter(by_group: pd.DataFrame) -> None:
    plot_df = by_group[by_group["dataset_label"].astype(str).isin(DEVICE_ORDER)].copy()
    fig, ax = plt.subplots(figsize=(9.4, 7.2))
    sns.scatterplot(
        data=plot_df,
        x="group_accuracy",
        y="sample_accuracy",
        hue="dataset_label",
        style="true_group",
        size="n_samples",
        sizes=(45, 210),
        alpha=0.85,
        ax=ax,
    )
    ax.plot([0, 1], [0, 1], color="0.4", linestyle="--", linewidth=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, max(0.18, float(plot_df["sample_accuracy"].max()) + 0.05))
    ax.set_xlabel("Group accuracy")
    ax.set_ylabel("Exact sample accuracy")
    ax.set_title("Per-group result: group may match, but exact sample rarely matches")
    ax.legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    fig.subplots_adjust(bottom=0.11, left=0.10, right=0.76, top=0.90)
    fig.savefig(FIG_DIR / "07_group_accuracy_vs_sample_accuracy_by_group.png", dpi=220)
    plt.close(fig)


def write_guide(summary: pd.DataFrame) -> None:
    lines = [
        "# Group and Sample Matching by Device",
        "",
        "이 figure set은 각 장비에서 먼저 clinical group이 맞는지, 그리고 그 안에서 exact sample ID까지 맞는지를 보기 위한 것입니다.",
        "",
        "상태 정의:",
        "",
        "- Exact sample match: predicted sample ID가 true sample ID와 완전히 같음.",
        "- Group only: clinical group은 맞지만 sample ID는 다름.",
        "- Wrong group: clinical group부터 틀림.",
        "",
        "## Device Summary",
        "",
        "| Dataset | n | group acc. | exact sample acc. | median true rank |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['dataset_label']} | {int(row['n_samples'])} | "
            f"{float(row['group_accuracy']):.4f} | {float(row['sample_accuracy']):.4f} | "
            f"{float(row['median_true_rank']):.1f} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `01_device_group_then_sample_status_stacked.png`: group/sample matching 상태를 장비별 stacked bar로 표시.",
            "- `02_device_group_vs_exact_sample_accuracy.png`: group accuracy와 exact sample accuracy를 직접 비교.",
            "- `03_group_confusion_by_device.png`: true group vs predicted group confusion matrix.",
            "- `04_match_status_by_true_group_and_device.png`: 각 group에서 exact/group-only/wrong-group 비율.",
            "- `05_sample_level_match_status_grid_by_device.png`: 모든 sample을 점으로 표시. 초록은 exact sample, 노랑은 group-only, 빨강은 wrong group.",
            "- `06_true_sample_rank_by_group_and_device.png`: true sample이 후보 중 몇 등인지. 1이면 exact match.",
            "- `07_group_accuracy_vs_sample_accuracy_by_group.png`: group별 group accuracy와 sample accuracy 관계.",
            "",
            "## Interpretation",
            "",
            "현재 결과는 대부분의 장비에서 exact sample match가 거의 없고, group-only match도 제한적입니다. 즉 문제가 sample ID 수준에서만 생긴 것이 아니라, group-level transfer도 충분히 안정적이지 않습니다.",
        ]
    )
    (OUT_DIR / "FIGURE_GUIDE_KO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists():
        import shutil

        shutil.rmtree(OUT_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    df = prepare_data()
    summary, by_group = write_tables(df)
    plot_status_summary(df, summary)
    plot_group_confusion(df)
    plot_status_by_group(df)
    plot_sample_grid(df)
    plot_rank_by_group(df)
    plot_group_sample_accuracy_scatter(by_group)
    write_guide(summary)

    print(f"Wrote group/sample matching figures: {OUT_DIR}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
