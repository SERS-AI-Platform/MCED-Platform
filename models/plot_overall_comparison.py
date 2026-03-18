from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 12
LEGEND_SIZE = 11
ANNOTATION_SIZE = 10

MODEL_DISPLAY_NAMES = {
    "cnn1d": "CNN1D-Shallow",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "resnet18": "ResNet18-1D",
    "xgboost": "XGBoost-Hierarchical",
}

MODEL_NAME_ALIASES = {
    "cnn1d-shallow": "cnn1d",
    "cnn1d": "cnn1d",
    "logistic regression": "logistic_regression",
    "random forest": "random_forest",
    "resnet18-1d": "resnet18",
    "xgboost-hierarchical": "xgboost",
}


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate and plot model comparison results")
    p.add_argument(
        "--roots",
        nargs="+",
        default=[
            "models/results/01_benchmarks/main_5models",
            "models/results/01_benchmarks/allspectra_3models",
            "models/results/02_tuning",
            "models/results/05_subset_analysis",
            "models/results/_archive",
        ],
    )
    p.add_argument("--output", "-o", default="models/results/04_comparisons/overall_comparison")
    return p.parse_args()


def _normalize_model_name(summary: dict, summary_path: Path) -> str:
    model_name = summary.get("model_name")
    if model_name:
        return str(model_name).strip().lower()

    model_label = str(summary.get("model", "")).strip().lower()
    if model_label in MODEL_NAME_ALIASES:
        return MODEL_NAME_ALIASES[model_label]

    parent_name = summary_path.parent.parent.name.lower() if summary_path.parent.parent else ""
    if parent_name in MODEL_DISPLAY_NAMES:
        return parent_name

    return model_label.replace(" ", "_").replace("-", "_") or summary_path.parent.name.lower()


def _display_model_name(model_name: str, summary: dict) -> str:
    return str(summary.get("model") or MODEL_DISPLAY_NAMES.get(model_name, model_name))


def _resolve_run_label(summary: dict, summary_path: Path, root: Path) -> tuple[str, str]:
    version = str(summary.get("version") or "").strip()
    if version:
        return version, version

    parent = summary_path.parent
    if parent.name.lower().startswith("v"):
        return parent.name, parent.name
    if parent == root:
        return parent.name, "legacy"
    return parent.name, parent.name


def _load_latest_map(root: Path) -> dict[str, str]:
    latest_map: dict[str, str] = {}
    for latest_path in root.rglob("latest.txt"):
        try:
            latest_version = latest_path.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        latest_map[str(latest_path.parent.resolve())] = latest_version
    return latest_map


def discover_summaries(roots):
    rows = []
    seen_paths = set()

    for root_str in roots:
        root = Path(root_str)
        if not root.exists():
            continue

        latest_map = _load_latest_map(root)
        for summary_path in root.rglob("training_summary.json"):
            summary_path = summary_path.resolve()
            if summary_path in seen_paths:
                continue
            seen_paths.add(summary_path)

            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)

            run_dir = summary_path.parent
            model_name = _normalize_model_name(summary, summary_path)
            model_display = _display_model_name(model_name, summary)
            version_label, version_tag = _resolve_run_label(summary, summary_path, root.resolve())
            parent_dir = str(run_dir.parent.resolve())
            latest_version = latest_map.get(parent_dir)
            is_latest = False
            if latest_version:
                is_latest = run_dir.name == latest_version
            elif run_dir == root.resolve():
                is_latest = True

            metrics = summary.get("metrics", {})
            row = {
                "source_root": root.name,
                "source_path": str(run_dir),
                "model_name": model_name,
                "model": model_display,
                "version": version_tag,
                "version_label": version_label,
                "label": f"{model_display} [{version_label}]",
                "run_group": run_dir.parent.name if run_dir.parent != run_dir else root.name,
                "n_samples": summary.get("n_samples"),
                "n_features": summary.get("n_features"),
                "val_s1_auc": metrics.get("val_s1_auc"),
                "val_s1_accuracy": metrics.get("val_s1_accuracy"),
                "val_s2_accuracy": metrics.get("val_s2_accuracy"),
                "val_s2_auc": metrics.get("val_s2_auc"),
                "val_s2_f1_macro": metrics.get("val_s2_f1_macro"),
                "train_s1_auc": metrics.get("train_s1_auc"),
                "train_s2_auc": metrics.get("train_s2_auc"),
                "train_s2_f1_macro": metrics.get("train_s2_f1_macro"),
                "is_latest": bool(is_latest),
            }
            rows.append(row)

    return pd.DataFrame(rows)


def _apply_axis_style(ax, title=None, xlabel=None):
    if title is not None:
        ax.set_title(title, fontsize=TITLE_SIZE - 1, fontweight="bold", pad=10)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, width=1.0, length=4)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_alpha(0.85)


def _plot_metric_panels(df, output_path, title, sort_metric="val_s2_f1_macro"):
    if df.empty:
        return

    plot_df = df.sort_values(
        [sort_metric, "val_s2_auc", "val_s1_auc"],
        ascending=False,
    ).reset_index(drop=True)
    labels = [f"{row['label']}\n{row['source_root']}" for _, row in plot_df.iterrows()]
    y = np.arange(len(plot_df))
    panels = [
        ("val_s1_auc", "Stage 1 AUROC", "#2a9d8f"),
        ("val_s2_f1_macro", "Stage 2 Macro F1", "#e76f51"),
        ("val_s2_auc", "Stage 2 AUROC", "#457b9d"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(19, max(7, len(plot_df) * 0.5)), sharey=True)
    for ax, (metric, panel_title, color) in zip(axes, panels):
        values = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        left = max(0.0, float(finite_values.min()) - 0.08) if len(finite_values) else 0.0

        ax.barh(y, values, color=color, alpha=0.9)
        ax.set_xlim(left, 1.0)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=TICK_SIZE)
        ax.grid(True, axis="x", alpha=0.22)
        _apply_axis_style(ax, title=panel_title, xlabel="Score")

        for yi, val in zip(y, values):
            if np.isfinite(val):
                ax.text(min(val + 0.008, 0.995), yi, f"{val:.3f}", va="center", fontsize=ANNOTATION_SIZE)

    axes[0].invert_yaxis()
    fig.suptitle(title, fontsize=TITLE_SIZE, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_best_per_model(best_df, output_path):
    if best_df.empty:
        return

    plot_df = best_df.sort_values("val_s2_f1_macro", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.7 * len(plot_df) + 1.5)))

    ax.barh(plot_df["model"], plot_df["val_s2_f1_macro"], color="#c8553d", alpha=0.9, label="Stage 2 Macro F1")
    ax.scatter(plot_df["val_s2_auc"], plot_df["model"], color="#457b9d", s=60, zorder=3, label="Stage 2 AUROC")

    for row in plot_df.itertuples(index=False):
        ax.text(
            min(float(row.val_s2_f1_macro) + 0.01, 0.995),
            row.model,
            f"{row.version_label} | F1={row.val_s2_f1_macro:.3f}",
            va="center",
            ha="left",
            fontsize=ANNOTATION_SIZE,
        )

    ax.set_xlim(0, 1.0)
    ax.grid(True, axis="x", alpha=0.22)
    _apply_axis_style(ax, title="Best Run per Model", xlabel="Validation Score")
    ax.legend(fontsize=LEGEND_SIZE, loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = discover_summaries(args.roots)
    if df.empty:
        raise SystemExit("No training_summary.json files found under the provided roots.")

    df = (
        df.sort_values(["model_name", "source_root", "version_label", "source_path"])
        .drop_duplicates(subset=["source_path"], keep="last")
        .reset_index(drop=True)
    )
    df.to_csv(out_dir / "overall_model_comparison.csv", index=False)

    _plot_metric_panels(
        df,
        out_dir / "overall_model_comparison.png",
        "Overall Model Comparison",
    )

    latest_df = (
        df[df["is_latest"]]
        .sort_values(["model_name", "val_s2_f1_macro", "val_s2_auc"], ascending=[True, False, False])
        .drop_duplicates(subset=["model_name", "source_root"], keep="first")
        .reset_index(drop=True)
    )
    latest_df.to_csv(out_dir / "latest_model_comparison.csv", index=False)
    _plot_metric_panels(
        latest_df,
        out_dir / "latest_model_comparison.png",
        "Latest Run Comparison",
    )

    torch_df = df[df["model_name"].isin(["resnet18", "cnn1d"])].copy()
    if not torch_df.empty:
        _plot_metric_panels(
            torch_df,
            out_dir / "torch_model_comparison.png",
            "Torch Model Comparison",
        )

    best_df = (
        df.sort_values(["val_s2_f1_macro", "val_s2_auc", "val_s1_auc"], ascending=False)
        .groupby("model_name", as_index=False)
        .first()
        .sort_values("val_s2_f1_macro", ascending=False)
        .reset_index(drop=True)
    )
    best_df.to_csv(out_dir / "best_per_model.csv", index=False)
    _plot_best_per_model(best_df, out_dir / "best_per_model.png")

    print(out_dir / "overall_model_comparison.png")


if __name__ == "__main__":
    raise SystemExit(main())
