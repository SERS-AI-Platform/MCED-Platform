#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate diagnostics for primary vs lot-balanced SERS mismatch.

The figures focus on residual differences after PS/Si x-axis alignment:

- where primary and lot-balanced sample centroids sit in PCA space,
- whether the same sample is closer than wrong primary samples,
- which wavenumber regions differ most from primary,
- whether specific dates explain the failure.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_primary_to_lot_balanced_same_id_test as same_id  # noqa: E402


OUT_DIR = same_id.ROOT / "06_analysis_results" / "primary_lot_balanced_difference_diagnostics"
FIG_DIR = OUT_DIR / "figures"
PRIMARY_DATASET = "primary_thermo"
FEATURE = "vector_derivative_snv"
SPECTRUM_FEATURE = "vector_snv"
DATASET_DISPLAY = {
    PRIMARY_DATASET: "Primary",
    "lot_balanced_thermo": "Thermo",
    "lot_balanced_handheld": "Handheld",
    "lot_balanced_medical": "Medical",
}
DATASET_ORDER = ["Primary", "Thermo", "Handheld", "Medical"]
DATASET_COLORS = {
    "Primary": "#2b2b2b",
    "Thermo": "#1f77b4",
    "Handheld": "#2ca02c",
    "Medical": "#d62728",
}
GROUP_ORDER = ["NOR", "YNOR", "H.D.", "YPAN", "CPAN", "SPAN", "PRO", "BLC", "BRE", "CRC", "DIA", "HBP", "LUN", "OVA"]


def display_dataset(dataset: str) -> str:
    return DATASET_DISPLAY.get(dataset, dataset)


def date_root_for_record(record: same_id.SpectrumRecord) -> Path | None:
    return same_id.find_lot_balanced_date_root(record.path)


def date_label_for_record(record: same_id.SpectrumRecord) -> str:
    root = date_root_for_record(record)
    if root is None:
        return "primary"
    return root.name.replace("_Urine test", "")


def centroid_vectors(
    records: list[same_id.SpectrumRecord],
    dataset: str,
    sample_ids: set[str],
) -> tuple[list[dict[str, object]], dict[str, dict[str, np.ndarray]]]:
    grouped: dict[str, list[same_id.SpectrumRecord]] = defaultdict(list)
    for record in records:
        if record.sample_id in sample_ids:
            grouped[record.sample_id].append(record)

    rows: list[dict[str, object]] = []
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for sample_id, items in sorted(grouped.items(), key=lambda x: (x[0].split()[0], int(x[0].split()[1]))):
        dates = sorted({date_label_for_record(item) for item in items})
        roots = sorted({str(date_root_for_record(item)) for item in items if date_root_for_record(item) is not None})
        vectors[sample_id] = {
            FEATURE: np.mean([getattr(item, FEATURE) for item in items], axis=0),
            SPECTRUM_FEATURE: np.mean([getattr(item, SPECTRUM_FEATURE) for item in items], axis=0),
        }
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": display_dataset(dataset),
                "sample_id": sample_id,
                "group": items[0].group,
                "sample_number": items[0].sample_number,
                "n_replicates": len(items),
                "date_label": dates[0] if len(dates) == 1 else "mixed",
                "date_root": roots[0] if len(roots) == 1 else "",
            }
        )
    return rows, vectors


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def run_pca(matrix: np.ndarray, n_components: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = normalize_rows(matrix)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    n_components = min(n_components, eigvecs.shape[1])
    coords = centered @ eigvecs[:, :n_components]
    explained = eigvals / eigvals.sum()
    return coords, explained[:n_components], eigvecs[:, :n_components]


def prepare_data() -> tuple[
    list[same_id.SpectrumRecord],
    dict[str, list[same_id.SpectrumRecord]],
    pd.DataFrame,
    dict[str, dict[str, dict[str, np.ndarray]]],
    dict[str, set[str]],
]:
    same_id.AXIS_ALIGNMENT_MODE = "reference_shift"
    same_id.SKIPPED_SPECTRA.clear()
    same_id.REFERENCE_SHIFT_INFO.clear()
    same_id.REFERENCE_ANCHOR_ROWS.clear()
    same_id.REFERENCE_READ_ERRORS.clear()

    primary_records = same_id.load_records(same_id.PRIMARY_ROOT, "primary_pooled", PRIMARY_DATASET)
    primary_ids = {record.sample_id for record in primary_records}
    balanced_by_dataset = {
        name: same_id.load_records(path, "date_lot_balanced", name)
        for name, path in same_id.BALANCED_ROOTS.items()
        if path.exists()
    }
    overlap_by_dataset = {
        name: primary_ids & {record.sample_id for record in records}
        for name, records in balanced_by_dataset.items()
    }
    union_overlap = set().union(*overlap_by_dataset.values())

    centroid_rows: list[dict[str, object]] = []
    vectors_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    rows, vectors = centroid_vectors(primary_records, PRIMARY_DATASET, union_overlap)
    centroid_rows.extend(rows)
    vectors_by_dataset[PRIMARY_DATASET] = vectors
    for dataset, records in balanced_by_dataset.items():
        rows, vectors = centroid_vectors(records, dataset, overlap_by_dataset[dataset])
        centroid_rows.extend(rows)
        vectors_by_dataset[dataset] = vectors

    return primary_records, balanced_by_dataset, pd.DataFrame(centroid_rows), vectors_by_dataset, overlap_by_dataset


def build_pair_tables(
    centroid_df: pd.DataFrame,
    vectors_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]],
    overlap_by_dataset: dict[str, set[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(7)
    pair_rows: list[dict[str, object]] = []
    similarity_rows: list[dict[str, object]] = []
    primary_vectors = vectors_by_dataset[PRIMARY_DATASET]
    primary_meta = centroid_df[centroid_df["dataset"] == PRIMARY_DATASET].set_index("sample_id")

    for dataset, overlap in overlap_by_dataset.items():
        sample_ids = sorted(overlap, key=lambda sid: (sid.split()[0], int(sid.split()[1])))
        candidate_labels = [sid for sid in sample_ids if sid in primary_vectors and sid in vectors_by_dataset[dataset]]
        primary_matrix = normalize_rows(np.vstack([primary_vectors[sid][FEATURE] for sid in candidate_labels]))
        balanced_matrix = normalize_rows(np.vstack([vectors_by_dataset[dataset][sid][FEATURE] for sid in candidate_labels]))
        sims = balanced_matrix @ primary_matrix.T
        label_index = {sid: idx for idx, sid in enumerate(candidate_labels)}
        dataset_meta = centroid_df[centroid_df["dataset"] == dataset].set_index("sample_id")

        for row_idx, sample_id in enumerate(candidate_labels):
            group = str(primary_meta.loc[sample_id, "group"])
            true_idx = label_index[sample_id]
            sim_row = sims[row_idx]
            order = np.argsort(-sim_row)
            pred_idx = int(order[0])
            if pred_idx == true_idx and len(order) > 1:
                best_wrong_idx = int(order[1])
            else:
                best_wrong_idx = pred_idx
            pred_sample_id = candidate_labels[pred_idx]
            best_wrong_sample_id = candidate_labels[best_wrong_idx]
            true_rank = int(np.where(order == true_idx)[0][0]) + 1
            pair_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": display_dataset(dataset),
                    "sample_id": sample_id,
                    "group": group,
                    "date_label": str(dataset_meta.loc[sample_id, "date_label"]),
                    "date_root": str(dataset_meta.loc[sample_id, "date_root"]),
                    "true_similarity": float(sim_row[true_idx]),
                    "best_wrong_similarity": float(sim_row[best_wrong_idx]),
                    "best_wrong_sample_id": best_wrong_sample_id,
                    "best_wrong_group": str(primary_meta.loc[best_wrong_sample_id, "group"]),
                    "pred_sample_id": pred_sample_id,
                    "pred_group": str(primary_meta.loc[pred_sample_id, "group"]),
                    "correct_identity": pred_sample_id == sample_id,
                    "correct_group": str(primary_meta.loc[pred_sample_id, "group"]) == group,
                    "true_rank": true_rank,
                    "similarity_margin_true_minus_best_wrong": float(sim_row[true_idx] - sim_row[best_wrong_idx]),
                }
            )
            similarity_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": display_dataset(dataset),
                    "sample_id": sample_id,
                    "comparison": "True same sample",
                    "similarity": float(sim_row[true_idx]),
                }
            )
            similarity_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": display_dataset(dataset),
                    "sample_id": sample_id,
                    "comparison": "Best wrong primary",
                    "similarity": float(sim_row[best_wrong_idx]),
                }
            )

            same_group = [
                idx
                for idx, sid in enumerate(candidate_labels)
                if sid != sample_id and str(primary_meta.loc[sid, "group"]) == group
            ]
            other_group = [
                idx for idx, sid in enumerate(candidate_labels) if str(primary_meta.loc[sid, "group"]) != group
            ]
            for comparison, candidates in [
                ("Random same-group primary", same_group),
                ("Random other-group primary", other_group),
            ]:
                if not candidates:
                    continue
                chosen = rng.choice(candidates, size=min(5, len(candidates)), replace=False)
                for idx in chosen:
                    similarity_rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": display_dataset(dataset),
                            "sample_id": sample_id,
                            "comparison": comparison,
                            "similarity": float(sim_row[int(idx)]),
                        }
                    )
    return pd.DataFrame(pair_rows), pd.DataFrame(similarity_rows)


def add_pca_coordinates(
    centroid_df: pd.DataFrame,
    vectors_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    vectors = []
    rows = []
    for _, row in centroid_df.iterrows():
        dataset = str(row["dataset"])
        sample_id = str(row["sample_id"])
        vectors.append(vectors_by_dataset[dataset][sample_id][FEATURE])
        rows.append(row.to_dict())
    matrix = np.vstack(vectors)
    coords, explained, _ = run_pca(matrix, n_components=20)
    pca_df = pd.DataFrame(rows)
    for idx in range(coords.shape[1]):
        pca_df[f"PC{idx + 1}"] = coords[:, idx]
    explained_df = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(explained))], "explained_variance": explained})
    return pca_df, explained_df


def write_pca_figures(pca_df: pd.DataFrame, explained_df: pd.DataFrame) -> None:
    pc1 = float(explained_df.loc[0, "explained_variance"]) * 100
    pc2 = float(explained_df.loc[1, "explained_variance"]) * 100

    fig, axes = plt.subplots(1, 2, figsize=(17.5, 7.2))
    dataset_order = [label for label in DATASET_ORDER if label in set(pca_df["dataset_label"])]
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="dataset_label",
        hue_order=dataset_order,
        palette=DATASET_COLORS,
        s=18,
        alpha=0.58,
        linewidth=0,
        ax=axes[0],
    )
    axes[0].set_title("Sample centroids colored by acquisition")
    axes[0].set_xlabel(f"PC1 ({pc1:.1f}% var.)")
    axes[0].set_ylabel(f"PC2 ({pc2:.1f}% var.)")
    axes[0].legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False)

    group_order = [group for group in GROUP_ORDER if group in set(pca_df["group"])]
    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="group",
        hue_order=group_order,
        palette="tab20",
        s=18,
        alpha=0.55,
        linewidth=0,
        ax=axes[1],
    )
    axes[1].set_title("Same PCA colored by clinical group")
    axes[1].set_xlabel(f"PC1 ({pc1:.1f}% var.)")
    axes[1].set_ylabel("")
    axes[1].legend(title="", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=1.6)
    fig.suptitle("Primary vs lot-balanced centroid space after PS/Si x-axis alignment")
    fig.subplots_adjust(bottom=0.20, left=0.06, right=0.86, top=0.88, wspace=0.12)
    fig.savefig(FIG_DIR / "01_pca_centroids_acquisition_vs_group.png", dpi=200)
    plt.close(fig)

    balanced = pca_df[pca_df["dataset"] != PRIMARY_DATASET].copy()
    primary_xy = pca_df[pca_df["dataset"] == PRIMARY_DATASET].set_index("sample_id")[["PC1", "PC2"]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ["lot_balanced_thermo", "lot_balanced_handheld", "lot_balanced_medical"]):
        subset = balanced[balanced["dataset"] == dataset].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        plot_subset = subset.sample(min(260, len(subset)), random_state=7)
        ax.scatter(primary_xy["PC1"], primary_xy["PC2"], s=4, c="0.85", alpha=0.35, linewidth=0)
        for _, row in plot_subset.iterrows():
            sid = str(row["sample_id"])
            if sid not in primary_xy.index:
                continue
            start = primary_xy.loc[sid]
            ax.plot([start["PC1"], row["PC1"]], [start["PC2"], row["PC2"]], color=DATASET_COLORS[display_dataset(dataset)], alpha=0.16, linewidth=0.7)
        ax.scatter(
            plot_subset["PC1"],
            plot_subset["PC2"],
            s=11,
            c=DATASET_COLORS[display_dataset(dataset)],
            alpha=0.65,
            linewidth=0,
            label=display_dataset(dataset),
        )
        ax.set_title(display_dataset(dataset))
        ax.set_xlabel(f"PC1 ({pc1:.1f}% var.)")
    axes[0].set_ylabel(f"PC2 ({pc2:.1f}% var.)")
    fig.suptitle("Same sample centroids move away from primary in a systematic direction")
    fig.subplots_adjust(bottom=0.14, left=0.06, right=0.99, top=0.84, wspace=0.08)
    fig.savefig(FIG_DIR / "02_pca_same_sample_shift_vectors.png", dpi=200)
    plt.close(fig)


def write_similarity_figures(pair_df: pd.DataFrame, similarity_df: pd.DataFrame) -> None:
    order = [label for label in DATASET_ORDER if label != "Primary" and label in set(pair_df["dataset_label"])]
    comparison_order = [
        "True same sample",
        "Best wrong primary",
        "Random same-group primary",
        "Random other-group primary",
    ]

    fig, ax = plt.subplots(figsize=(14, 7.4))
    sns.boxplot(
        data=similarity_df,
        x="dataset_label",
        y="similarity",
        hue="comparison",
        order=order,
        hue_order=[c for c in comparison_order if c in set(similarity_df["comparison"])],
        showfliers=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("Cosine similarity to primary centroid")
    ax.set_title("The true primary sample is not more similar than wrong primary samples")
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False)
    fig.subplots_adjust(bottom=0.24, left=0.08, right=0.99, top=0.90)
    fig.savefig(FIG_DIR / "03_same_id_vs_wrong_primary_similarity.png", dpi=200)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.8), sharex=True, sharey=True)
    for ax, dataset in zip(axes, ["lot_balanced_thermo", "lot_balanced_handheld", "lot_balanced_medical"]):
        subset = pair_df[pair_df["dataset"] == dataset]
        if subset.empty:
            ax.set_visible(False)
            continue
        ax.scatter(
            subset["true_similarity"],
            subset["best_wrong_similarity"],
            c=np.where(subset["correct_identity"], "#2ca02c", "#d62728"),
            s=12,
            alpha=0.55,
            linewidth=0,
        )
        lim_min = float(np.nanmin([subset["true_similarity"].min(), subset["best_wrong_similarity"].min()]))
        lim_max = float(np.nanmax([subset["true_similarity"].max(), subset["best_wrong_similarity"].max()]))
        pad = (lim_max - lim_min) * 0.05
        ax.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad], color="0.2", linewidth=1, linestyle="--")
        ax.set_title(display_dataset(dataset))
        ax.set_xlabel("Similarity to true primary sample")
        ax.set_xlim(lim_min - pad, lim_max + pad)
        ax.set_ylim(lim_min - pad, lim_max + pad)
        n = len(subset)
        correct = float(subset["correct_identity"].mean()) * 100
        ax.text(0.04, 0.95, f"same-ID top1: {correct:.1f}%\nn={n}", transform=ax.transAxes, va="top", ha="left")
    axes[0].set_ylabel("Similarity to best wrong primary sample")
    fig.suptitle("Most lot-balanced samples sit above the diagonal: a wrong primary centroid is closer")
    fig.subplots_adjust(bottom=0.14, left=0.06, right=0.99, top=0.84, wspace=0.10)
    fig.savefig(FIG_DIR / "04_true_vs_best_wrong_similarity_scatter.png", dpi=200)
    plt.close(fig)

    margin_df = pair_df.copy()
    margin_df["margin_wrong_minus_true"] = margin_df["best_wrong_similarity"] - margin_df["true_similarity"]
    fig, ax = plt.subplots(figsize=(11.5, 6.8))
    sns.violinplot(
        data=margin_df,
        x="dataset_label",
        y="margin_wrong_minus_true",
        order=order,
        inner=None,
        cut=0,
        palette={label: DATASET_COLORS[label] for label in order},
        ax=ax,
    )
    sns.boxplot(
        data=margin_df,
        x="dataset_label",
        y="margin_wrong_minus_true",
        order=order,
        width=0.22,
        showfliers=False,
        color="white",
        ax=ax,
    )
    ax.axhline(0, color="0.2", linewidth=1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("Best wrong similarity minus true similarity")
    ax.set_title("Positive margin means the same sample loses to another primary sample")
    fig.subplots_adjust(bottom=0.13, left=0.10, right=0.98, top=0.90)
    fig.savefig(FIG_DIR / "05_wrong_minus_true_similarity_margin.png", dpi=200)
    plt.close(fig)


def paired_delta_tables(
    vectors_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]],
    overlap_by_dataset: dict[str, set[str]],
    centroid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary_meta = centroid_df[centroid_df["dataset"] == PRIMARY_DATASET].set_index("sample_id")
    rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    for dataset, overlap in overlap_by_dataset.items():
        for sample_id in sorted(overlap, key=lambda sid: (sid.split()[0], int(sid.split()[1]))):
            if sample_id not in vectors_by_dataset[dataset] or sample_id not in vectors_by_dataset[PRIMARY_DATASET]:
                continue
            primary = vectors_by_dataset[PRIMARY_DATASET][sample_id][SPECTRUM_FEATURE]
            balanced = vectors_by_dataset[dataset][sample_id][SPECTRUM_FEATURE]
            diff = balanced - primary
            group = str(primary_meta.loc[sample_id, "group"])
            for idx, wn in enumerate(same_id.GRID):
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": display_dataset(dataset),
                        "sample_id": sample_id,
                        "group": group,
                        "wavenumber": float(wn),
                        "delta": float(diff[idx]),
                        "abs_delta": float(abs(diff[idx])),
                    }
                )
            group_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": display_dataset(dataset),
                    "sample_id": sample_id,
                    "group": group,
                    "mean_abs_delta": float(np.mean(np.abs(diff))),
                    "rms_delta": float(np.sqrt(np.mean(diff**2))),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(group_rows)


def write_spectrum_difference_figures(
    vectors_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]],
    overlap_by_dataset: dict[str, set[str]],
    centroid_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    mean_rows: list[dict[str, object]] = []
    diff_rows: list[dict[str, object]] = []
    primary_vectors = vectors_by_dataset[PRIMARY_DATASET]

    union_overlap = set().union(*overlap_by_dataset.values())
    primary_mean = np.mean([primary_vectors[sid][SPECTRUM_FEATURE] for sid in union_overlap if sid in primary_vectors], axis=0)
    for idx, wn in enumerate(same_id.GRID):
        mean_rows.append({"dataset_label": "Primary", "wavenumber": float(wn), "mean_snv": float(primary_mean[idx])})

    for dataset, overlap in overlap_by_dataset.items():
        sample_ids = [sid for sid in overlap if sid in vectors_by_dataset[dataset] and sid in primary_vectors]
        balanced_mean = np.mean([vectors_by_dataset[dataset][sid][SPECTRUM_FEATURE] for sid in sample_ids], axis=0)
        paired_diff = np.mean(
            [vectors_by_dataset[dataset][sid][SPECTRUM_FEATURE] - primary_vectors[sid][SPECTRUM_FEATURE] for sid in sample_ids],
            axis=0,
        )
        for idx, wn in enumerate(same_id.GRID):
            mean_rows.append({"dataset_label": display_dataset(dataset), "wavenumber": float(wn), "mean_snv": float(balanced_mean[idx])})
            diff_rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": display_dataset(dataset),
                    "wavenumber": float(wn),
                    "mean_delta_vs_primary": float(paired_diff[idx]),
                    "abs_mean_delta_vs_primary": float(abs(paired_diff[idx])),
                }
            )

    mean_df = pd.DataFrame(mean_rows)
    diff_df = pd.DataFrame(diff_rows)

    fig, axes = plt.subplots(2, 1, figsize=(15, 8.4), sharex=True, gridspec_kw={"height_ratios": [1.1, 1.0]})
    for label in DATASET_ORDER:
        subset = mean_df[mean_df["dataset_label"] == label]
        if subset.empty:
            continue
        axes[0].plot(subset["wavenumber"], subset["mean_snv"], label=label, color=DATASET_COLORS[label], linewidth=1.4)
    axes[0].set_ylabel("Mean SNV intensity")
    axes[0].set_title("Average spectrum shape differs after PS/Si x-axis alignment")
    axes[0].legend(title="", loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4, frameon=False)

    for label in [x for x in DATASET_ORDER if x != "Primary"]:
        subset = diff_df[diff_df["dataset_label"] == label]
        if subset.empty:
            continue
        axes[1].plot(
            subset["wavenumber"],
            subset["mean_delta_vs_primary"],
            label=f"{label} - Primary",
            color=DATASET_COLORS[label],
            linewidth=1.2,
        )
    axes[1].axhline(0, color="0.2", linewidth=0.9)
    axes[1].set_xlabel("Raman shift (cm-1)")
    axes[1].set_ylabel("Mean paired delta")
    axes[1].set_xlim(400, 1800)
    fig.subplots_adjust(bottom=0.10, left=0.08, right=0.99, top=0.89, hspace=0.12)
    fig.savefig(FIG_DIR / "06_mean_snv_spectra_and_delta_vs_primary.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 5.8))
    for label in [x for x in DATASET_ORDER if x != "Primary"]:
        subset = diff_df[diff_df["dataset_label"] == label]
        if subset.empty:
            continue
        ax.plot(
            subset["wavenumber"],
            subset["abs_mean_delta_vs_primary"],
            label=label,
            color=DATASET_COLORS[label],
            linewidth=1.3,
        )
    ax.set_xlabel("Raman shift (cm-1)")
    ax.set_ylabel("Absolute mean paired delta")
    ax.set_title("Wavenumber regions with the largest residual difference from primary")
    ax.set_xlim(400, 1800)
    ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)
    fig.subplots_adjust(bottom=0.23, left=0.08, right=0.99, top=0.89)
    fig.savefig(FIG_DIR / "07_abs_delta_by_wavenumber_vs_primary.png", dpi=200)
    plt.close(fig)

    delta_df, group_delta = paired_delta_tables(vectors_by_dataset, overlap_by_dataset, centroid_df)
    binned = delta_df.copy()
    binned["wn_bin"] = (np.floor(binned["wavenumber"] / 10) * 10).astype(int)
    heat = (
        binned.groupby(["dataset_label", "group", "wn_bin"])["delta"]
        .mean()
        .reset_index()
    )
    heat["row_label"] = heat["dataset_label"] + " / " + heat["group"]
    row_order = [
        f"{dataset} / {group}"
        for dataset in [x for x in DATASET_ORDER if x != "Primary"]
        for group in GROUP_ORDER
        if f"{dataset} / {group}" in set(heat["row_label"])
    ]
    matrix = heat.pivot(index="row_label", columns="wn_bin", values="delta").reindex(row_order)
    vmax = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98))
    fig, ax = plt.subplots(figsize=(16.5, max(8.5, 0.32 * len(matrix.index))))
    sns.heatmap(matrix, cmap="vlag", center=0, vmin=-vmax, vmax=vmax, ax=ax, cbar_kws={"label": "Mean SNV delta vs primary"})
    ax.set_xlabel("Raman shift bin (cm-1)")
    ax.set_ylabel("")
    ax.set_title("Group-level residual spectral difference from matched primary samples")
    visible_ticks = []
    visible_labels = []
    columns = list(matrix.columns)
    for idx, col in enumerate(columns):
        if col % 200 == 0:
            visible_ticks.append(idx + 0.5)
            visible_labels.append(str(col))
    ax.set_xticks(visible_ticks)
    ax.set_xticklabels(visible_labels, rotation=0)
    fig.subplots_adjust(bottom=0.08, left=0.13, right=0.98, top=0.92)
    fig.savefig(FIG_DIR / "08_group_delta_heatmap_vs_primary.png", dpi=200)
    plt.close(fig)

    return diff_df, group_delta


def write_date_figures(pair_df: pd.DataFrame) -> pd.DataFrame:
    date_metrics = (
        pair_df.groupby(["dataset", "dataset_label", "date_label", "date_root"])
        .agg(
            n=("sample_id", "count"),
            same_id_accuracy=("correct_identity", "mean"),
            group_accuracy=("correct_group", "mean"),
            median_true_rank=("true_rank", "median"),
            mean_true_similarity=("true_similarity", "mean"),
            mean_margin=("similarity_margin_true_minus_best_wrong", "mean"),
        )
        .reset_index()
    )
    date_metrics.to_csv(OUT_DIR / "date_level_similarity_metrics.csv", index=False)

    heat = date_metrics.pivot(index="date_label", columns="dataset_label", values="same_id_accuracy")
    heat = heat.reindex(columns=[label for label in DATASET_ORDER if label != "Primary" and label in heat.columns])
    n_table = date_metrics.pivot(index="date_label", columns="dataset_label", values="n").fillna(0).astype(int)
    n_table = n_table.reindex(columns=heat.columns)
    labels = heat.copy().astype(object)
    for idx in heat.index:
        for col in heat.columns:
            val = heat.loc[idx, col]
            n = n_table.loc[idx, col] if idx in n_table.index and col in n_table.columns else 0
            labels.loc[idx, col] = "" if pd.isna(val) else f"{val:.2f}\nn={n}"
    fig, ax = plt.subplots(figsize=(9.2, max(7.8, 0.36 * len(heat.index))))
    sns.heatmap(heat, annot=labels, fmt="", vmin=0, vmax=0.12, cmap="mako", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Date folder")
    ax.set_title("Same-ID accuracy remains low across date folders")
    fig.subplots_adjust(bottom=0.08, left=0.18, right=1.02, top=0.92)
    fig.savefig(FIG_DIR / "09_same_id_accuracy_by_date_heatmap.png", dpi=200)
    plt.close(fig)

    ref_df = pd.DataFrame(list(same_id.REFERENCE_SHIFT_INFO.values()))
    if not ref_df.empty:
        ref_df.to_csv(OUT_DIR / "reference_axis_shift_by_date_used.csv", index=False)
        merged = date_metrics.merge(ref_df[["date_root", "median_shift_cm", "status"]], on="date_root", how="left")
        merged["abs_reference_shift_cm"] = merged["median_shift_cm"].abs()
        plot_df = merged[merged["status"] == "ok"].copy()
        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10.5, 6.4))
            sns.scatterplot(
                data=plot_df,
                x="abs_reference_shift_cm",
                y="same_id_accuracy",
                hue="dataset_label",
                hue_order=[label for label in DATASET_ORDER if label != "Primary"],
                palette=DATASET_COLORS,
                size="n",
                sizes=(35, 180),
                alpha=0.78,
                ax=ax,
            )
            ax.set_xlabel("Absolute PS/Si reference x-axis shift (cm-1)")
            ax.set_ylabel("Date-level same-ID accuracy")
            ax.set_title("Reference x-axis shift size does not explain the mismatch")
            ax.legend(title="", loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False)
            fig.subplots_adjust(bottom=0.25, left=0.10, right=0.98, top=0.88)
            fig.savefig(FIG_DIR / "10_reference_shift_vs_same_id_accuracy.png", dpi=200)
            plt.close(fig)
    pd.DataFrame(same_id.REFERENCE_ANCHOR_ROWS).to_csv(OUT_DIR / "reference_axis_anchor_detections_used.csv", index=False)
    pd.DataFrame(same_id.REFERENCE_READ_ERRORS).to_csv(OUT_DIR / "reference_axis_read_errors_used.csv", index=False)
    return date_metrics


def write_failure_examples(
    pair_df: pd.DataFrame,
    vectors_by_dataset: dict[str, dict[str, dict[str, np.ndarray]]],
) -> None:
    selected_rows = []
    for dataset in ["lot_balanced_thermo", "lot_balanced_handheld", "lot_balanced_medical"]:
        subset = pair_df[(pair_df["dataset"] == dataset) & (~pair_df["correct_identity"])].copy()
        subset["wrong_minus_true"] = subset["best_wrong_similarity"] - subset["true_similarity"]
        selected_rows.append(subset.nlargest(2, "wrong_minus_true"))
    examples = pd.concat(selected_rows, ignore_index=True)
    if examples.empty:
        return
    fig, axes = plt.subplots(len(examples), 1, figsize=(14, max(9, 2.25 * len(examples))), sharex=True)
    if len(examples) == 1:
        axes = [axes]
    for ax, (_, row) in zip(axes, examples.iterrows()):
        dataset = str(row["dataset"])
        sample_id = str(row["sample_id"])
        wrong_id = str(row["best_wrong_sample_id"])
        lot_vec = vectors_by_dataset[dataset][sample_id][SPECTRUM_FEATURE]
        true_primary = vectors_by_dataset[PRIMARY_DATASET][sample_id][SPECTRUM_FEATURE]
        wrong_primary = vectors_by_dataset[PRIMARY_DATASET][wrong_id][SPECTRUM_FEATURE]
        ax.plot(same_id.GRID, lot_vec, color=DATASET_COLORS[display_dataset(dataset)], linewidth=1.0, label=f"{display_dataset(dataset)} {sample_id}")
        ax.plot(same_id.GRID, true_primary, color="0.15", linewidth=1.0, label=f"Primary true {sample_id}")
        ax.plot(same_id.GRID, wrong_primary, color="#ff7f0e", linewidth=1.0, alpha=0.95, label=f"Primary wrong {wrong_id}")
        ax.set_ylabel("SNV")
        ax.set_title(
            f"{display_dataset(dataset)} {sample_id}: true sim={row['true_similarity']:.3f}, "
            f"wrong sim={row['best_wrong_similarity']:.3f}, rank={int(row['true_rank'])}"
        )
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8)
    axes[-1].set_xlabel("Raman shift (cm-1)")
    axes[-1].set_xlim(400, 1800)
    fig.suptitle("Failure examples: the wrong primary spectrum is closer than the true primary spectrum")
    fig.subplots_adjust(bottom=0.06, left=0.07, right=0.99, top=0.94, hspace=0.52)
    fig.savefig(FIG_DIR / "11_failure_examples_true_vs_wrong_primary_overlay.png", dpi=200)
    plt.close(fig)


def write_variance_explained_figure(pca_df: pd.DataFrame, explained_df: pd.DataFrame) -> pd.DataFrame:
    pc_cols = [col for col in pca_df.columns if col.startswith("PC")][:20]
    weights = explained_df["explained_variance"].iloc[: len(pc_cols)].to_numpy()
    weights = weights / weights.sum()
    rows = []
    for label_col, label_name in [
        ("dataset_label", "Acquisition/instrument"),
        ("group", "Clinical group"),
        ("date_label", "Date folder"),
    ]:
        y = pca_df[label_col].astype(str)
        if label_col == "date_label":
            usable = pca_df[pca_df["dataset"] != PRIMARY_DATASET].copy()
            y = usable[label_col].astype(str)
            xdf = usable[pc_cols]
        else:
            xdf = pca_df[pc_cols]
        scores = []
        for idx, col in enumerate(pc_cols):
            values = xdf[col].to_numpy()
            overall = values.mean()
            total_ss = float(np.sum((values - overall) ** 2))
            between_ss = 0.0
            for _, group_values in pd.Series(values).groupby(y.to_numpy()):
                arr = group_values.to_numpy()
                between_ss += len(arr) * float((arr.mean() - overall) ** 2)
            scores.append(0.0 if total_ss == 0 else between_ss / total_ss)
        rows.append({"factor": label_name, "weighted_pc_r2": float(np.sum(np.array(scores) * weights[: len(scores)]))})
    result = pd.DataFrame(rows)
    result.to_csv(OUT_DIR / "pc_variance_explained_by_metadata.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    sns.barplot(data=result, x="factor", y="weighted_pc_r2", color="#4c78a8", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("Weighted one-way R2 across first 20 PCs")
    ax.set_title("Acquisition/date explains a large share of spectral feature variation")
    ax.set_ylim(0, max(0.05, float(result["weighted_pc_r2"].max()) * 1.25))
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", fontsize=9)
    fig.subplots_adjust(bottom=0.16, left=0.13, right=0.98, top=0.88)
    fig.savefig(FIG_DIR / "12_pc_variance_explained_by_metadata.png", dpi=200)
    plt.close(fig)
    return result


def write_report(
    centroid_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    date_metrics: pd.DataFrame,
    variance_df: pd.DataFrame,
) -> None:
    summary = (
        pair_df.groupby("dataset_label")
        .agg(
            n=("sample_id", "count"),
            same_id_accuracy=("correct_identity", "mean"),
            group_accuracy=("correct_group", "mean"),
            median_true_rank=("true_rank", "median"),
            median_true_similarity=("true_similarity", "median"),
            median_best_wrong_similarity=("best_wrong_similarity", "median"),
            median_margin=("similarity_margin_true_minus_best_wrong", "median"),
        )
        .reindex([label for label in DATASET_ORDER if label != "Primary"])
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "diagnostic_similarity_summary.csv", index=False)
    centroid_df.to_csv(OUT_DIR / "sample_centroid_manifest.csv", index=False)

    lines = [
        "# Primary vs Lot-Balanced Difference Diagnostics",
        "",
        "## Scope",
        "",
        "These figures diagnose the residual mismatch after PS/Si reference x-axis alignment.",
        "Primary spectra are used as the reference acquisition; lot-balanced Thermo, Handheld, and Medical sample centroids are compared to matched primary sample centroids.",
        "",
        "## Main Finding",
        "",
        "The same lot-balanced sample is usually not closest to its true primary centroid. A wrong primary sample often has higher cosine similarity than the true primary sample.",
        "This points to residual acquisition/date/lot/instrument effects and spectral-shape differences, not just a simple wavenumber-axis offset.",
        "",
        "## Similarity Summary",
        "",
        "| Dataset | n | same-ID acc. | group acc. | median true rank | median true sim. | median best wrong sim. | median true-best margin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset_label"]),
                    str(int(row["n"])),
                    f"{float(row['same_id_accuracy']):.4f}",
                    f"{float(row['group_accuracy']):.4f}",
                    f"{float(row['median_true_rank']):.1f}",
                    f"{float(row['median_true_similarity']):.3f}",
                    f"{float(row['median_best_wrong_similarity']):.3f}",
                    f"{float(row['median_margin']):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/01_pca_centroids_acquisition_vs_group.png`: whether acquisition separates the feature space.",
            "- `figures/02_pca_same_sample_shift_vectors.png`: same sample movement from primary to each lot-balanced acquisition.",
            "- `figures/03_same_id_vs_wrong_primary_similarity.png`: true primary similarity versus wrong-primary similarity distributions.",
            "- `figures/04_true_vs_best_wrong_similarity_scatter.png`: pointwise true-vs-best-wrong comparison.",
            "- `figures/05_wrong_minus_true_similarity_margin.png`: how often the wrong primary centroid wins.",
            "- `figures/06_mean_snv_spectra_and_delta_vs_primary.png`: average residual spectrum shape difference.",
            "- `figures/07_abs_delta_by_wavenumber_vs_primary.png`: wavenumber regions with largest residual difference.",
            "- `figures/08_group_delta_heatmap_vs_primary.png`: group-level residual spectral differences.",
            "- `figures/09_same_id_accuracy_by_date_heatmap.png`: date-level same-ID performance.",
            "- `figures/10_reference_shift_vs_same_id_accuracy.png`: shows reference x-axis shift size is not enough to explain mismatch.",
            "- `figures/11_failure_examples_true_vs_wrong_primary_overlay.png`: concrete failed examples.",
            "- `figures/12_pc_variance_explained_by_metadata.png`: metadata factor effect size over first 20 PCs.",
            "",
            "## Metadata Effect Size",
            "",
        ]
    )
    for _, row in variance_df.iterrows():
        lines.append(f"- {row['factor']}: weighted PC R2 = {float(row['weighted_pc_r2']):.3f}")
    lines.extend(
        [
            "",
            "## Date Summary",
            "",
            f"- Date-level rows: {len(date_metrics)}",
            f"- Sample centroid rows: {len(centroid_df)}",
        ]
    )
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if OUT_DIR.exists():
        import shutil

        shutil.rmtree(OUT_DIR)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    _, _, centroid_df, vectors_by_dataset, overlap_by_dataset = prepare_data()
    pair_df, similarity_df = build_pair_tables(centroid_df, vectors_by_dataset, overlap_by_dataset)
    pca_df, explained_df = add_pca_coordinates(centroid_df, vectors_by_dataset)

    centroid_df.to_csv(OUT_DIR / "sample_centroid_manifest.csv", index=False)
    pca_df.to_csv(OUT_DIR / "sample_centroid_pca_coordinates.csv", index=False)
    explained_df.to_csv(OUT_DIR / "pca_explained_variance.csv", index=False)
    pair_df.to_csv(OUT_DIR / "primary_to_lot_similarity_pairs.csv", index=False)
    similarity_df.to_csv(OUT_DIR / "primary_similarity_distribution_rows.csv", index=False)
    pd.DataFrame(same_id.SKIPPED_SPECTRA).to_csv(OUT_DIR / "skipped_spectra.csv", index=False)

    write_pca_figures(pca_df, explained_df)
    write_similarity_figures(pair_df, similarity_df)
    write_spectrum_difference_figures(vectors_by_dataset, overlap_by_dataset, centroid_df)
    date_metrics = write_date_figures(pair_df)
    write_failure_examples(pair_df, vectors_by_dataset)
    variance_df = write_variance_explained_figure(pca_df, explained_df)
    write_report(centroid_df, pair_df, date_metrics, variance_df)

    print(f"Wrote diagnostics: {OUT_DIR}")
    print(pair_df.groupby("dataset_label")["correct_identity"].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
