from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import ModelConfig
from models.train import (
    MODEL_DISPLAY_NAMES,
    aggregate_replicates,
    apply_class_selection,
    apply_training_overrides,
    configure_torch_runtime,
    load_processed_spectra,
    resolve_runtime_config,
    resolve_logreg_params,
    run_classical_cv,
    run_torch_cv,
)

logger = logging.getLogger(__name__)

TITLE_SIZE = 18
LABEL_SIZE = 15
TICK_SIZE = 12
LEGEND_SIZE = 11
ANNOTATION_SIZE = 10


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare Logistic Regression vs NN learning curves as subject count increases."
    )
    parser.add_argument("--input", "-i", default="results/processed_spectra.csv")
    parser.add_argument("--config", "-c", default="config/config.yaml")
    parser.add_argument(
        "--output",
        "-o",
        default="models/results/03_learning_curves",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "cnn1d", "resnet18"],
        default=["logistic_regression", "cnn1d", "resnet18"],
    )
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--aggregate", choices=["medoid", "mean", "none"], default="medoid")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--dropout-rate", type=float, default=None)
    parser.add_argument("--stage2-loss-weight", type=float, default=None)
    parser.add_argument("--head-hidden-dim", type=int, default=None)
    parser.add_argument("--resnet-channels", nargs="+", type=int, default=None)
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=None)
    parser.add_argument("--class-balance-beta", type=float, default=None)
    parser.add_argument("--logreg-c", type=float, default=None)
    parser.add_argument("--logreg-max-iter", type=int, default=None)
    parser.add_argument("--cancer-types", nargs="+", default=None)
    parser.add_argument("--non-cancer-groups", nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        else "cpu"
    )


def _load_model_config(args, n_features: int) -> ModelConfig:
    try:
        with open(args.config, encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        logger.warning(f"Config not found: {args.config}. Using defaults.")
        raw_cfg = {}

    config = (
        ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=n_features)
        if raw_cfg else ModelConfig(n_spectral_features=n_features)
    )
    config = apply_training_overrides(config, args)
    config = apply_class_selection(
        config,
        cancer_types=args.cancer_types,
        non_cancer_groups=args.non_cancer_groups,
    )
    return config


def _build_modeling_dataframe(df: pd.DataFrame, config: ModelConfig) -> tuple[pd.DataFrame, list[str]]:
    feature_cols = [c for c in df.columns if c.startswith("x_")]
    if not feature_cols:
        raise ValueError("No spectral feature columns found.")

    valid_groups = set(config.cancer_types) | set(config.non_cancer_groups)
    df_valid = df[df["group"].isin(valid_groups)].copy()
    if df_valid.empty:
        raise ValueError("No rows remain after applying class selection.")

    if "sample_id" not in df_valid.columns:
        df_valid["sample_id"] = np.arange(len(df_valid)).astype(str)
    else:
        df_valid["sample_id"] = df_valid["sample_id"].astype(str)

    df_valid["subject_key"] = (
        df_valid["group"].astype(str) + "__" + df_valid["sample_id"].astype(str)
    )
    df_valid["binary_label"] = df_valid["group"].isin(config.cancer_types).astype(int)
    df_valid["cancer_type_label"] = df_valid["group"].apply(config.cancer_type_index).astype(int)
    return df_valid.reset_index(drop=True), feature_cols


def _sample_subject_subset(
    df: pd.DataFrame,
    fraction: float,
    rng: np.random.Generator,
    min_subjects_per_group: int,
) -> pd.DataFrame:
    selected_keys: list[str] = []
    subject_df = df[["group", "subject_key"]].drop_duplicates()

    for group, group_subjects in subject_df.groupby("group", sort=True):
        keys = group_subjects["subject_key"].to_numpy()
        n_group = len(keys)
        if n_group == 0:
            continue
        target = int(math.ceil(n_group * fraction))
        target = max(min_subjects_per_group, target)
        target = min(target, n_group)
        choice = rng.choice(keys, size=target, replace=False)
        selected_keys.extend(choice.tolist())

    subset = df[df["subject_key"].isin(selected_keys)].copy()
    return subset.reset_index(drop=True)


def _build_arrays(df_subset: pd.DataFrame, feature_cols: list[str]):
    X = df_subset[feature_cols].to_numpy(dtype=np.float32)
    binary_labels = df_subset["binary_label"].to_numpy(dtype=int)
    cancer_type_labels = df_subset["cancer_type_label"].to_numpy(dtype=int)
    sample_ids = df_subset["subject_key"].to_numpy(dtype=str)
    groups_arr = df_subset["group"].to_numpy(dtype=str)
    return X, binary_labels, cancer_type_labels, sample_ids, groups_arr


def _metric_columns():
    return [
        "val_s1_auc",
        "val_s1_accuracy",
        "val_s2_accuracy",
        "val_s2_auc",
        "val_s2_f1_macro",
    ]


def _plot_learning_curve(summary_df: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    if summary_df.empty:
        return

    colors = {
        "logistic_regression": "#c8553d",
        "cnn1d": "#4d9de0",
        "resnet18": "#287271",
    }

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for model_name, sub in summary_df.groupby("model_name", sort=False):
        sub = sub.sort_values("fraction")
        color = colors.get(model_name, "#444444")
        label = MODEL_DISPLAY_NAMES.get(model_name, model_name)
        y = sub[f"{metric}_mean"].to_numpy(dtype=float)
        y_std = sub[f"{metric}_std"].fillna(0.0).to_numpy(dtype=float)
        x = sub["fraction"].to_numpy(dtype=float)

        ax.plot(x, y, marker="o", linewidth=2.4, markersize=7, color=color, label=label)
        ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.16)

        last_row = sub.iloc[-1]
        if np.isfinite(last_row[f"{metric}_mean"]):
            ax.text(
                float(last_row["fraction"]) + 0.01,
                float(last_row[f"{metric}_mean"]),
                f"{last_row[f'{metric}_mean']:.3f}",
                fontsize=ANNOTATION_SIZE,
                va="center",
            )

    ax.set_title(title, fontsize=TITLE_SIZE, fontweight="bold", pad=12)
    ax.set_xlabel("Subject Fraction", fontsize=LABEL_SIZE)
    ax.set_ylabel(metric.replace("_", " ").upper(), fontsize=LABEL_SIZE)
    ax.set_xlim(max(0.0, summary_df["fraction"].min() - 0.02), min(1.05, summary_df["fraction"].max() + 0.08))
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.22)
    ax.tick_params(axis="both", labelsize=TICK_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, frameon=True, loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_interpretation(summary_df: pd.DataFrame, out_path: Path) -> None:
    metric = "val_s2_f1_macro"
    lines = []
    lines.append("Learning Curve Interpretation")
    lines.append("=" * 40)
    lines.append("")

    best_rows = (
        summary_df.sort_values([f"{metric}_mean", "fraction"], ascending=[False, False])
        .groupby("fraction", as_index=False)
        .first()
        .sort_values("fraction")
    )
    lines.append("Best model by fraction (mean validation Stage 2 macro F1):")
    for row in best_rows.itertuples(index=False):
        lines.append(
            f"- fraction={row.fraction:.2f}: {MODEL_DISPLAY_NAMES.get(row.model_name, row.model_name)} "
            f"({getattr(row, metric + '_mean'):.3f})"
        )

    logistic_rows = summary_df[summary_df["model_name"] == "logistic_regression"].copy()
    nn_rows = summary_df[summary_df["model_name"].isin(["cnn1d", "resnet18"])].copy()
    if not logistic_rows.empty and not nn_rows.empty:
        best_nn = (
            nn_rows.sort_values([metric + "_mean", "fraction"], ascending=[False, False])
            .groupby("fraction", as_index=False)
            .first()
        )
        merged = logistic_rows.merge(best_nn, on="fraction", suffixes=("_logistic", "_best_nn"))
        overtake = merged[metric + "_mean_best_nn"] > merged[metric + "_mean_logistic"]
        lines.append("")
        if overtake.any():
            first = merged[overtake].sort_values("fraction").iloc[0]
            lines.append(
                "A neural network overtakes logistic regression at "
                f"fraction={first['fraction']:.2f} "
                f"({MODEL_DISPLAY_NAMES.get(first['model_name_best_nn'], first['model_name_best_nn'])})."
            )
        else:
            lines.append(
                "Within the tested fractions, logistic regression remains ahead of the best neural network "
                "on mean validation Stage 2 macro F1."
            )

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "learning_curve.log", mode="w", encoding="utf-8"),
        ],
    )

    device = resolve_device(args.device)
    configure_torch_runtime(device)
    logger.info(f"Device: {device}")

    df = load_processed_spectra(args.input)
    feature_cols = [c for c in df.columns if c.startswith("x_")]
    config = _load_model_config(args, len(feature_cols))
    logger.info(f"Cancer types: {config.cancer_types}")
    logger.info(f"Non-cancer groups: {config.non_cancer_groups}")

    df_agg = aggregate_replicates(df, feature_cols, args.aggregate)
    modeling_df, feature_cols = _build_modeling_dataframe(df_agg, config)

    fractions = sorted(set(float(f) for f in args.fractions))
    if not fractions:
        raise ValueError("At least one fraction is required.")

    runtime_args = SimpleNamespace(
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        no_amp=args.no_amp,
    )
    runtime = resolve_runtime_config(runtime_args, device)

    results_rows = []
    sampling_rows = []
    min_subjects_per_group = max(2, args.n_splits)

    for repeat_idx in range(args.repeats):
        repeat_seed = args.seed + repeat_idx
        logger.info("")
        logger.info(f"[Repeat {repeat_idx + 1}/{args.repeats}] seed={repeat_seed}")
        repeat_rng = np.random.default_rng(repeat_seed)

        for fraction in fractions:
            logger.info(f"  Fraction={fraction:.2f}")
            subset_df = _sample_subject_subset(
                modeling_df,
                fraction,
                repeat_rng,
                min_subjects_per_group=min_subjects_per_group,
            )
            subject_counts = subset_df.groupby("group")["subject_key"].nunique().to_dict()
            actual_fraction = (
                subset_df["subject_key"].nunique() / modeling_df["subject_key"].nunique()
            )
            sampling_rows.append(
                {
                    "repeat": repeat_idx,
                    "seed": repeat_seed,
                    "requested_fraction": fraction,
                    "actual_fraction": actual_fraction,
                    "n_subjects": subset_df["subject_key"].nunique(),
                    "n_rows": len(subset_df),
                    "subject_counts_json": json.dumps(subject_counts, ensure_ascii=True),
                }
            )

            subset_config = ModelConfig(**config.__dict__)
            subset_config.random_state = repeat_seed

            X, bl, ctl, sample_ids, groups_arr = _build_arrays(subset_df, feature_cols)

            for model_name in args.models:
                logger.info(f"    Model={MODEL_DISPLAY_NAMES.get(model_name, model_name)}")
                if model_name == "logistic_regression":
                    model_params = resolve_logreg_params(args)
                    results = run_classical_cv(
                        model_name,
                        X,
                        bl,
                        ctl,
                        sample_ids,
                        groups_arr,
                        subset_config,
                        model_params,
                        n_splits=args.n_splits,
                        ckpt_dir=None,
                    )
                else:
                    results = run_torch_cv(
                        X,
                        bl,
                        ctl,
                        sample_ids,
                        groups_arr,
                        subset_config,
                        device,
                        model_name=model_name,
                        runtime=runtime,
                        n_splits=args.n_splits,
                        ckpt_dir=None,
                    )

                row = {
                    "repeat": repeat_idx,
                    "seed": repeat_seed,
                    "fraction": fraction,
                    "actual_fraction": actual_fraction,
                    "n_subjects": subset_df["subject_key"].nunique(),
                    "n_rows": len(subset_df),
                    "model_name": model_name,
                    "model": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                    "aggregate": args.aggregate,
                    "n_splits": args.n_splits,
                }
                for metric_name in _metric_columns():
                    row[metric_name] = results["overall"].get(metric_name, np.nan)
                results_rows.append(row)

    results_df = pd.DataFrame(results_rows)
    sampling_df = pd.DataFrame(sampling_rows)
    if results_df.empty:
        raise RuntimeError("No learning-curve results were produced.")

    results_df.to_csv(out_dir / "learning_curve_metrics.csv", index=False)
    sampling_df.to_csv(out_dir / "learning_curve_sampling.csv", index=False)

    summary_df = (
        results_df.groupby(["model_name", "model", "fraction"], as_index=False)
        .agg(
            n_runs=("repeat", "count"),
            actual_fraction_mean=("actual_fraction", "mean"),
            n_subjects_mean=("n_subjects", "mean"),
            **{
                f"{metric}_mean": (metric, "mean")
                for metric in _metric_columns()
            },
            **{
                f"{metric}_std": (metric, "std")
                for metric in _metric_columns()
            },
        )
        .sort_values(["fraction", "model_name"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(out_dir / "learning_curve_summary.csv", index=False)

    _plot_learning_curve(
        summary_df,
        "val_s2_f1_macro",
        out_dir / "learning_curve_stage2_f1.png",
        "Learning Curve - Stage 2 Macro F1",
    )
    _plot_learning_curve(
        summary_df,
        "val_s2_auc",
        out_dir / "learning_curve_stage2_auc.png",
        "Learning Curve - Stage 2 AUROC",
    )
    _plot_learning_curve(
        summary_df,
        "val_s1_auc",
        out_dir / "learning_curve_stage1_auc.png",
        "Learning Curve - Stage 1 AUROC",
    )
    _write_interpretation(summary_df, out_dir / "learning_curve_interpretation.txt")

    logger.info("")
    logger.info(f"Saved: {out_dir / 'learning_curve_metrics.csv'}")
    logger.info(f"Saved: {out_dir / 'learning_curve_summary.csv'}")
    logger.info(f"Saved: {out_dir / 'learning_curve_stage2_f1.png'}")
    logger.info(f"Saved: {out_dir / 'learning_curve_stage2_auc.png'}")
    logger.info(f"Saved: {out_dir / 'learning_curve_stage1_auc.png'}")
    logger.info(f"Saved: {out_dir / 'learning_curve_interpretation.txt'}")


if __name__ == "__main__":
    raise SystemExit(main())
