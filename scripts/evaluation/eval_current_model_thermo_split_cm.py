#!/usr/bin/env python3
"""Thermo-only 60/20/20 current-model confusion matrix diagnostics.

This script scores the current STK-V2 artifact on the primary Thermo
acquisition, assigns sample-level train/val/test split labels, and renders
binary and cancer-type confusion matrices.

The current artifact is not retrained here. Split-specific metrics are
therefore diagnostic views of the current model predictions, not unbiased
holdout estimates.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.eval_current_model_acquisition_sets import (
    DATA_ROOT,
    CANCER_GROUPS,
    TYPE_GROUP_MAP,
    collect_records,
    predict_samples,
    preprocess_to_samples,
    score_frame,
)
from scripts.deployment.sers_predict import StackingPredictor


CANCER_TYPES = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
BINARY_LABELS = ["Non-cancer", "Cancer"]
DEFAULT_OUT = PROJECT_ROOT / "results" / "evaluation" / "current_model_thermo_6_2_2_cm"
THERMO_PRIMARY_ROOT = (
    DATA_ROOT / "02_sers_primary_pooled_acquisition" / "thermo" / "raw_data"
)


def assign_6_2_2_split(
    df: pd.DataFrame,
    *,
    stratify_col: str,
    seed: int,
) -> pd.Series:
    """Assign sample-level train/val/test labels with group stratification."""
    idx = np.arange(len(df))
    strat = df[stratify_col].astype(str).to_numpy()
    trainval_idx, test_idx = train_test_split(
        idx,
        test_size=0.20,
        random_state=seed,
        stratify=strat,
    )
    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.25,
        random_state=seed,
        stratify=df.iloc[trainval_idx][stratify_col].astype(str).to_numpy(),
    )

    split = pd.Series(index=df.index, dtype=object)
    split.iloc[train_idx] = "train"
    split.iloc[val_idx] = "val"
    split.iloc[test_idx] = "test"
    return split


def row_normalize(cm: np.ndarray) -> np.ndarray:
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)


def save_matrix_csvs(
    out_dir: Path,
    prefix: str,
    split: str,
    cm: np.ndarray,
    labels: list[str],
) -> None:
    counts = pd.DataFrame(cm, index=labels, columns=labels)
    norm = pd.DataFrame(row_normalize(cm), index=labels, columns=labels)
    counts.to_csv(out_dir / f"{prefix}_{split}_counts.csv")
    norm.to_csv(out_dir / f"{prefix}_{split}_row_normalized.csv")


def draw_cm_grid(
    matrices: dict[str, np.ndarray],
    labels: list[str],
    title: str,
    out_path: Path,
    *,
    cmap: str = "Blues",
) -> None:
    split_order = ["all", "train", "val", "test"]
    n = len(split_order)
    fig_w = 4.0 * n if len(labels) <= 2 else 4.8 * n
    fig_h = 3.9 if len(labels) <= 2 else 5.1
    fig, axes = plt.subplots(1, n, figsize=(fig_w, fig_h), constrained_layout=True)
    if n == 1:
        axes = [axes]

    vmax = 1.0
    for ax, split in zip(axes, split_order):
        cm = matrices[split]
        norm = row_normalize(cm)
        ax.imshow(norm, cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(split.capitalize(), fontsize=12, fontweight="bold")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        max_count = max(int(cm.max()), 1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                pct = norm[i, j] * 100.0
                color = "white" if norm[i, j] > 0.55 else "#1f2933"
                count = int(cm[i, j])
                text = f"{count}\n{pct:.1f}%" if count or cm[i].sum() else "0"
                weight = "bold" if count >= max_count * 0.5 else "normal"
                ax.text(
                    j,
                    i,
                    text,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    fontweight=weight,
                )
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_binary_cm(df: pd.DataFrame) -> np.ndarray:
    return confusion_matrix(
        df["true_binary"].to_numpy(),
        df["pred_binary_balanced"].to_numpy(),
        labels=[0, 1],
    )


def build_type_cm(df: pd.DataFrame) -> np.ndarray:
    cancer = df[df["true_type"].notna()].copy()
    return confusion_matrix(
        cancer["true_type"].astype(str).to_numpy(),
        cancer["pred_type"].astype(str).to_numpy(),
        labels=CANCER_TYPES,
    )


def score_splits(df: pd.DataFrame, subset: str) -> pd.DataFrame:
    rows = []
    for split in ["all", "train", "val", "test"]:
        part = df if split == "all" else df[df["split"].eq(split)]
        row = score_frame(part, f"{subset}_{split}")
        row["split"] = split
        rows.append(row)
    return pd.DataFrame(rows)


def run_subset(
    pred_df: pd.DataFrame,
    *,
    subset_name: str,
    subset_mask: pd.Series,
    stratify_col: str,
    seed: int,
    output_dir: Path,
) -> pd.DataFrame:
    out_dir = output_dir / subset_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pred_df[subset_mask].copy().reset_index(drop=True)
    df["split"] = assign_6_2_2_split(df, stratify_col=stratify_col, seed=seed)

    metrics = score_splits(df, subset_name)
    group_counts = (
        df.groupby(["split", "source_group", "group"], dropna=False)
        .agg(n=("uid", "size"), n_replicates_median=("n_replicates", "median"))
        .reset_index()
        .sort_values(["split", "source_group", "group"])
    )

    df.to_csv(out_dir / "per_sample_predictions.csv", index=False)
    metrics.to_csv(out_dir / "split_metrics.csv", index=False)
    group_counts.to_csv(out_dir / "split_group_counts.csv", index=False)

    binary_mats: dict[str, np.ndarray] = {}
    type_mats: dict[str, np.ndarray] = {}
    for split in ["all", "train", "val", "test"]:
        part = df if split == "all" else df[df["split"].eq(split)]
        binary_mats[split] = build_binary_cm(part)
        type_mats[split] = build_type_cm(part)
        save_matrix_csvs(out_dir, "binary_cm", split, binary_mats[split], BINARY_LABELS)
        save_matrix_csvs(out_dir, "type_cm", split, type_mats[split], CANCER_TYPES)

    draw_cm_grid(
        binary_mats,
        BINARY_LABELS,
        f"{subset_name}: Stage 1 binary CM",
        out_dir / "binary_confusion_matrices.png",
        cmap="Blues",
    )
    draw_cm_grid(
        type_mats,
        CANCER_TYPES,
        f"{subset_name}: Stage 2 cancer-type CM",
        out_dir / "type_confusion_matrices.png",
        cmap="Greens",
    )

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "usersnet" / "current")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = StackingPredictor(args.model_dir)

    records = collect_records(
        THERMO_PRIMARY_ROOT,
        "primary_pooled_thermo",
        "thermo",
        include_background_only=False,
    )
    if not records:
        raise RuntimeError(f"No records collected from {THERMO_PRIMARY_ROOT}")

    X, meta, errors = preprocess_to_samples(predictor, records)
    pred_df = predict_samples(predictor, X, meta)

    pred_df["true_type"] = pred_df["group"].map(TYPE_GROUP_MAP)
    pred_df["is_cancer"] = pred_df["group"].isin(CANCER_GROUPS)
    pred_df.to_csv(args.output_dir / "thermo_primary_all_predictions.csv", index=False)
    pd.DataFrame(errors).to_csv(args.output_dir / "preprocess_errors.csv", index=False)

    all_metrics = []
    all_metrics.append(
        run_subset(
            pred_df,
            subset_name="standard_1569",
            subset_mask=~pred_df["source_group"].isin(["YPAN", "YNOR"]),
            stratify_col="group",
            seed=args.seed,
            output_dir=args.output_dir,
        )
    )
    all_metrics.append(
        run_subset(
            pred_df,
            subset_name="all_source_preserved_1628",
            subset_mask=pd.Series(True, index=pred_df.index),
            stratify_col="source_group",
            seed=args.seed,
            output_dir=args.output_dir,
        )
    )
    summary = pd.concat(all_metrics, ignore_index=True)
    summary.to_csv(args.output_dir / "summary_split_metrics.csv", index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir),
        "mode": "balanced",
        "threshold": predictor.operating_modes.get("balanced", {}).get("threshold", 0.60),
        "root": str(THERMO_PRIMARY_ROOT),
        "n_records": len(records),
        "n_samples_all_source_preserved": int(len(pred_df)),
        "n_preprocess_errors": len(errors),
        "seed": args.seed,
        "notes": [
            "Current model predictions are split-labeled for diagnostics; the model is not retrained.",
            "Splits are sample-level after replicate averaging.",
            "standard_1569 excludes YPAN and YNOR to match the current model's primary standard cohort size.",
            "all_source_preserved_1628 keeps YPAN and YNOR as distinct sample keys before aliasing labels to PAN/NOR.",
        ],
        "metrics": summary.to_dict(orient="records"),
    }
    with open(args.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(summary.to_string(index=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
