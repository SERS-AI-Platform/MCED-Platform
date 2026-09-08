"""Validate baseline-correction choices under the current preprocessing/QC flow.

This script compares the current rolling-minimum baseline correction against
reference-friendly penalized least-squares alternatives while keeping the rest
of the pipeline fixed:

    load raw -> fixed grid -> calibration -> raw-signal QC
    -> smooth -> baseline method -> SNV -> fixed-grid resampling
    -> processed-signal QC -> subject aggregation -> LR evaluation

Outputs are written to results/baseline_method_validation/.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers import parse_filename, read_spectrum
from src.sers.config import RAW_DATA_DIR, RESULTS_DIR, load_config
from src.sers.io import find_spectra, make_common_grid, make_fixed_grid
from src.sers.preprocessing import (
    FINGERPRINT_REGION,
    baseline_correction,
    calibrate_spectra_batch,
    normalize_spectrum,
    resample,
    smooth,
    trim_spectrum,
    trim_to_grid,
)
from src.sers.qc.qc import (
    apply_per_spectrum_corr_qc,
    apply_stage1_qc,
    enforce_min_replicates,
)

LOGGER = logging.getLogger("baseline_method_validation")

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")
GROUP_ALIASES = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}
VALID_SOURCE_GROUPS = set(CANCER_TYPES) | set(NON_CANCER_GROUPS) | set(GROUP_ALIASES)


@dataclass(frozen=True)
class BaselineMethod:
    name: str
    label: str
    fn: Callable[[np.ndarray], np.ndarray]
    reference_family: str


def baseline_asls(y: np.ndarray, lam: float = 1e6, p: float = 0.01, niter: int = 10) -> np.ndarray:
    """Asymmetric least-squares baseline correction."""
    length = len(y)
    diff = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2), dtype=float).tocsc()
    weights = np.ones(length)
    for _ in range(niter):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        z = spsolve((weight_matrix + lam * diff.dot(diff.T)).tocsc(), weights * y)
        weights = p * (y > z) + (1 - p) * (y <= z)
    return y - z


def baseline_arpls(
    y: np.ndarray, lam: float = 1e6, ratio: float = 1e-6, niter: int = 20
) -> np.ndarray:
    """Asymmetrically reweighted penalized least-squares baseline correction."""
    length = len(y)
    diff = sparse.diags([1, -2, 1], [0, -1, -2], shape=(length, length - 2), dtype=float).tocsc()
    hessian = lam * diff.dot(diff.T)
    weights = np.ones(length)
    last_weights = weights.copy()
    for _ in range(niter):
        weight_matrix = sparse.spdiags(weights, 0, length, length)
        z = spsolve((weight_matrix + hessian).tocsc(), weights * y)
        residual = y - z
        negative = residual[residual < 0]
        if len(negative) < 2:
            break
        mean_neg = negative.mean()
        std_neg = negative.std()
        if std_neg <= 1e-12:
            break
        exponent = 2.0 * (residual - (2.0 * std_neg - mean_neg)) / std_neg
        weights = 1.0 / (1.0 + np.exp(np.clip(exponent, -60.0, 60.0)))
        denom = np.linalg.norm(last_weights)
        if denom > 0 and np.linalg.norm(weights - last_weights) / denom < ratio:
            break
        last_weights = weights.copy()
    return y - z


def build_methods() -> list[BaselineMethod]:
    return [
        BaselineMethod(
            name="rolling_minimum",
            label="Rolling minimum\nw=101",
            fn=lambda y: baseline_correction(y, window=101),
            reference_family="in-house lower-envelope",
        ),
        BaselineMethod(
            name="asls_1e6_p001",
            label="AsLS\nlambda=1e6, p=0.01",
            fn=lambda y: baseline_asls(y, lam=1e6, p=0.01, niter=10),
            reference_family="asymmetric least squares",
        ),
        BaselineMethod(
            name="asls_1e7_p0001",
            label="AsLS\nlambda=1e7, p=0.001",
            fn=lambda y: baseline_asls(y, lam=1e7, p=0.001, niter=15),
            reference_family="asymmetric least squares",
        ),
        BaselineMethod(
            name="arpls_1e6",
            label="arPLS\nlambda=1e6",
            fn=lambda y: baseline_arpls(y, lam=1e6, niter=20),
            reference_family="adaptive penalized least squares",
        ),
    ]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_raw_spectra(config, max_subjects_per_group: int | None = None) -> dict:
    files = list(find_spectra(Path(RAW_DATA_DIR), pattern="*.csv"))
    raw: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for fp in files:
        try:
            sid = parse_filename(
                fp, fallback_group=config.folder_to_group.get(fp.parent.name, "UNK")
            )
            if sid.group not in VALID_SOURCE_GROUPS:
                continue
            x, y = read_spectrum(fp)
            raw[(sid.group, sid.sample_id, sid.replicate)] = (x, y)
        except Exception:
            continue

    if max_subjects_per_group is None:
        return raw

    keep_subjects: set[tuple[str, str]] = set()
    for group in sorted({key[0] for key in raw}):
        subjects = sorted({key[1] for key in raw if key[0] == group})[:max_subjects_per_group]
        keep_subjects.update((group, subject) for subject in subjects)
    return {key: value for key, value in raw.items() if (key[0], key[1]) in keep_subjects}


def preprocess_with_baseline(
    raw_spectra: dict,
    proc_grid: np.ndarray,
    method: BaselineMethod,
    smooth_window: int,
    smooth_poly: int,
    normalization: str,
) -> tuple[dict, float]:
    processed = {}
    start = time.perf_counter()
    for key, (x, y) in raw_spectra.items():
        x_trim, y_proc = trim_spectrum(x, y.copy(), region=FINGERPRINT_REGION)
        y_proc = smooth(y_proc, window_length=smooth_window, polyorder=smooth_poly)
        y_proc = method.fn(y_proc)
        y_proc = normalize_spectrum(y_proc, method=normalization)
        processed[key] = resample(x_trim, y_proc, proc_grid)
    elapsed = time.perf_counter() - start
    return processed, elapsed


def corr_to_mean_distribution(processed: dict) -> np.ndarray:
    grouped: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}
    for (group, sample_id, replicate), y in processed.items():
        grouped.setdefault((group, sample_id), []).append((replicate, y))

    corr_values = []
    for reps in grouped.values():
        if len(reps) < 2:
            continue
        matrix = np.vstack([y for _, y in reps])
        for row_i in range(len(reps)):
            mask = np.arange(len(reps)) != row_i
            mean_other = matrix[mask].mean(axis=0)
            corr = np.corrcoef(matrix[row_i], mean_other)[0, 1]
            if np.isfinite(corr):
                corr_values.append(float(corr))
    return np.asarray(corr_values, dtype=float)


def aggregate_subjects(
    processed: dict, passed_keys: set
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    buckets: dict[str, dict] = {}
    for key in passed_keys:
        if key not in processed:
            continue
        source_group, sample_id, replicate = key
        group = GROUP_ALIASES.get(source_group, source_group)
        if group not in set(CANCER_TYPES) | set(NON_CANCER_GROUPS):
            continue
        subject_uid = f"{source_group}_{sample_id}"
        bucket = buckets.setdefault(
            subject_uid,
            {"group": group, "source_group": source_group, "spectra": []},
        )
        bucket["spectra"].append(processed[key])

    rows = []
    for subject_uid, item in sorted(buckets.items()):
        if not item["spectra"]:
            continue
        rows.append((subject_uid, item["group"], np.vstack(item["spectra"]).mean(axis=0)))

    if not rows:
        raise RuntimeError("No subjects left after QC filtering")

    subject_ids = np.asarray([row[0] for row in rows], dtype=object)
    groups = np.asarray([row[1] for row in rows], dtype=object)
    x = np.vstack([row[2] for row in rows]).astype(float)
    y_bin = np.asarray([1 if group in CANCER_TYPES else 0 for group in groups], dtype=int)
    return x, groups, y_bin, subject_ids


def evaluate_lr(
    x: np.ndarray,
    groups: np.ndarray,
    y_bin: np.ndarray,
    n_repeats: int,
    n_folds: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    class_counts = pd.Series(groups).value_counts()
    folds = min(n_folds, int(class_counts.min()))
    if folds < 2:
        raise RuntimeError(f"Not enough subjects per class for CV: {class_counts.to_dict()}")

    cancer_type_to_int = {name: idx for idx, name in enumerate(CANCER_TYPES)}
    y_type = np.asarray([cancer_type_to_int.get(group, -1) for group in groups], dtype=int)

    rows = []
    for repeat_idx in range(n_repeats):
        cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + repeat_idx * 101)
        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(x, groups)):
            binary_model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            )
            binary_model.fit(x[train_idx], y_bin[train_idx])
            s1_prob = binary_model.predict_proba(x[test_idx])[:, 1]
            s1_pred = (s1_prob >= 0.5).astype(int)

            cancer_train = y_bin[train_idx] == 1
            cancer_test = y_bin[test_idx] == 1
            type_macro_f1 = np.nan
            if cancer_train.sum() > 0 and cancer_test.sum() > 0:
                type_model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        max_iter=3000,
                        solver="lbfgs",
                        class_weight="balanced",
                    ),
                )
                type_model.fit(x[train_idx][cancer_train], y_type[train_idx][cancer_train])
                type_pred = type_model.predict(x[test_idx][cancer_test])
                type_macro_f1 = f1_score(
                    y_type[test_idx][cancer_test],
                    type_pred,
                    labels=list(range(len(CANCER_TYPES))),
                    average="macro",
                    zero_division=0,
                )

            rows.append(
                {
                    "repeat": repeat_idx,
                    "fold": fold_idx,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "s1_auc": float(roc_auc_score(y_bin[test_idx], s1_prob)),
                    "s1_f1": float(f1_score(y_bin[test_idx], s1_pred, zero_division=0)),
                    "s1_sensitivity": float(
                        recall_score(y_bin[test_idx], s1_pred, zero_division=0)
                    ),
                    "s1_specificity": float(
                        recall_score(1 - y_bin[test_idx], 1 - s1_pred, zero_division=0)
                    ),
                    "s2_type_macro_f1": float(type_macro_f1)
                    if np.isfinite(type_macro_f1)
                    else np.nan,
                }
            )

    fold_df = pd.DataFrame(rows)
    summary = {
        "n_subjects_eval": int(len(x)),
        "n_cancer_subjects": int(y_bin.sum()),
        "n_non_cancer_subjects": int((y_bin == 0).sum()),
        "n_cv_folds": int(folds),
        "s1_auc_mean": float(fold_df["s1_auc"].mean()),
        "s1_auc_std": float(fold_df["s1_auc"].std(ddof=1)),
        "s1_f1_mean": float(fold_df["s1_f1"].mean()),
        "s1_f1_std": float(fold_df["s1_f1"].std(ddof=1)),
        "s1_sensitivity_mean": float(fold_df["s1_sensitivity"].mean()),
        "s1_specificity_mean": float(fold_df["s1_specificity"].mean()),
        "s2_type_macro_f1_mean": float(fold_df["s2_type_macro_f1"].mean()),
        "s2_type_macro_f1_std": float(fold_df["s2_type_macro_f1"].std(ddof=1)),
    }
    return fold_df, summary


def make_summary_figure(summary_df: pd.DataFrame, out_path: Path) -> None:
    methods = summary_df["label"].tolist()
    colors = ["#4e79a7", "#f28e2b", "#b07aa1", "#59a14f"]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=180)
    fig.patch.set_facecolor("white")
    axes = axes.ravel()

    def bar(ax, values, errors, title, ylabel, ylim=None):
        x_pos = np.arange(len(methods))
        ax.bar(
            x_pos,
            values,
            yerr=errors,
            color=colors[: len(methods)],
            capsize=4,
            edgecolor="#263244",
            linewidth=0.7,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=8)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_ylabel(ylabel)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)

    bar(
        axes[0],
        summary_df["s1_auc_mean"],
        summary_df["s1_auc_std"],
        "A. Binary cancer detection",
        "AUC",
        (0.90, 1.0),
    )
    bar(
        axes[1],
        summary_df["s2_type_macro_f1_mean"],
        summary_df["s2_type_macro_f1_std"],
        "B. Cancer type classification",
        "Macro-F1",
        (0.50, 1.0),
    )
    bar(
        axes[2],
        summary_df["stage2_final_subject_retention_pct"],
        np.zeros(len(summary_df)),
        "C. Subject retention after v2 QC",
        "% retained",
        (0, 105),
    )
    bar(
        axes[3],
        summary_df["corr_to_mean_p05"],
        np.zeros(len(summary_df)),
        "D. Replicate consistency",
        "Corr-to-mean p5",
        (0.80, 1.0),
    )
    bar(
        axes[4],
        summary_df["preprocess_time_s"],
        np.zeros(len(summary_df)),
        "E. Preprocessing runtime",
        "Seconds",
        None,
    )

    axes[5].axis("off")
    best_auc = summary_df.loc[summary_df["s1_auc_mean"].idxmax()]
    best_f1 = summary_df.loc[summary_df["s2_type_macro_f1_mean"].idxmax()]
    fastest = summary_df.loc[summary_df["preprocess_time_s"].idxmin()]
    note = (
        "Validation design\n"
        "- Fixed grid: 402-2198 cm^-1, 935 points\n"
        "- Calibration: urea 1001.4 cm^-1\n"
        "- QC: v2 raw + processed replicate filters\n"
        "- Evaluation: subject-aggregated logistic regression\n\n"
        f"Best binary AUC: {best_auc['method']} ({best_auc['s1_auc_mean']:.4f})\n"
        f"Best type F1: {best_f1['method']} ({best_f1['s2_type_macro_f1_mean']:.4f})\n"
        f"Fastest: {fastest['method']} ({fastest['preprocess_time_s']:.1f}s)\n\n"
        "Use this as baseline-method validation, not final stacking performance."
    )
    axes[5].text(0.02, 0.98, note, va="top", ha="left", fontsize=10, family="monospace")

    fig.suptitle(
        "Baseline Correction Method Validation Under Current SERS-AI Pipeline",
        fontsize=16,
        weight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-subjects-per-group",
        type=int,
        default=None,
        help="Optional cap for a quick smoke test. Default uses all valid subjects.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()

    out_dir = Path(RESULTS_DIR) / "baseline_method_validation"
    fig_dir = Path(RESULTS_DIR) / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading config and spectra")
    config = load_config(args.config)
    raw_spectra = load_raw_spectra(config, max_subjects_per_group=args.max_subjects_per_group)
    LOGGER.info("Loaded %d valid raw spectra", len(raw_spectra))

    fixed_grid = make_fixed_grid(config)
    if fixed_grid is None:
        fixed_grid = make_common_grid([x for x, _ in raw_spectra.values()])
    proc_grid = trim_to_grid(fixed_grid, region=FINGERPRINT_REGION)
    LOGGER.info(
        "Target grid: %d points (%.1f-%.1f cm^-1)", len(proc_grid), proc_grid[0], proc_grid[-1]
    )

    prep = config.preprocessing
    raw_for_qc = raw_spectra
    if getattr(prep, "do_calibration", False):
        LOGGER.info("Calibrating raw spectra")
        raw_for_qc, shift_df = calibrate_spectra_batch(
            raw_spectra,
            target_wn=getattr(prep, "calibration_reference_wn", 1001.4),
            window=getattr(prep, "calibration_window", 10.0),
        )
        shift_df.to_csv(out_dir / "calibration_shifts.csv", index=False, encoding="utf-8-sig")

    qc = config.qc
    LOGGER.info("Running raw-signal QC")
    s1_pass, stage1_drops = apply_stage1_qc(
        raw_for_qc,
        fingerprint_region=tuple(qc.fingerprint_region),
        intensity_gate_ratio=qc.intensity_gate_ratio,
        cosmic_isolation=getattr(qc, "cosmic_isolation", 2.0),
        cosmic_height=getattr(qc, "cosmic_height", 0.3),
        sat_plateau=getattr(qc, "saturation_plateau", 5),
    )
    stage1_drops.to_csv(out_dir / "stage1_raw_qc_drops.csv", index=False, encoding="utf-8-sig")
    raw_stage1 = {key: raw_for_qc[key] for key in s1_pass}
    LOGGER.info("Raw-signal QC retained %d/%d spectra", len(raw_stage1), len(raw_for_qc))

    summary_rows = []
    all_fold_rows = []
    methods = build_methods()
    normalization = "snv" if getattr(prep, "use_snv", True) else "none"

    for method in methods:
        LOGGER.info("Running method: %s", method.name)
        processed, preprocess_time_s = preprocess_with_baseline(
            raw_stage1,
            proc_grid,
            method,
            smooth_window=getattr(prep, "smooth_window", 11),
            smooth_poly=getattr(prep, "smooth_poly", 3),
            normalization=normalization,
        )

        corr_values = corr_to_mean_distribution(processed)
        corr_thr = getattr(qc, "per_spectrum_corr_threshold", 0.925)
        min_reps = getattr(qc, "min_reps_after_qc", 4)
        s2a_pass, s2a_drops = apply_per_spectrum_corr_qc(processed, corr_threshold=corr_thr)
        final_pass, s2b_drops = enforce_min_replicates(s2a_pass, min_n=min_reps)

        x_eval, groups_eval, y_bin_eval, subject_ids = aggregate_subjects(processed, final_pass)
        fold_df, eval_summary = evaluate_lr(
            x_eval,
            groups_eval,
            y_bin_eval,
            n_repeats=args.n_repeats,
            n_folds=args.n_folds,
            seed=args.seed,
        )
        fold_df.insert(0, "method", method.name)
        all_fold_rows.append(fold_df)

        raw_subject_count = len({(GROUP_ALIASES.get(g, g), f"{g}_{s}") for g, s, _ in raw_stage1})
        final_subject_count = len(subject_ids)
        summary = {
            "method": method.name,
            "label": method.label,
            "reference_family": method.reference_family,
            "preprocess_time_s": float(preprocess_time_s),
            "stage1_input_spectra": int(len(raw_for_qc)),
            "stage1_retained_spectra": int(len(raw_stage1)),
            "stage2_input_spectra": int(len(processed)),
            "stage2a_retained_spectra": int(len(s2a_pass)),
            "stage2a_dropped_spectra": int(len(s2a_drops)),
            "stage2b_final_spectra": int(len(final_pass)),
            "stage2b_dropped_subjects": int(len(s2b_drops)),
            "stage2_final_subjects": int(final_subject_count),
            "stage2_final_subject_retention_pct": float(
                100 * final_subject_count / raw_subject_count
            ),
            "corr_to_mean_mean": float(np.mean(corr_values)) if len(corr_values) else np.nan,
            "corr_to_mean_median": float(np.median(corr_values)) if len(corr_values) else np.nan,
            "corr_to_mean_p05": float(np.quantile(corr_values, 0.05))
            if len(corr_values)
            else np.nan,
        }
        summary.update(eval_summary)
        summary_rows.append(summary)

        pd.DataFrame(s2a_drops).to_csv(
            out_dir / f"{method.name}_stage2a_drops.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(s2b_drops).to_csv(
            out_dir / f"{method.name}_stage2b_dropped_subjects.csv",
            index=False,
            encoding="utf-8-sig",
        )
        LOGGER.info(
            "%s: AUC %.4f, type F1 %.4f, retention %.1f%%, time %.1fs",
            method.name,
            summary["s1_auc_mean"],
            summary["s2_type_macro_f1_mean"],
            summary["stage2_final_subject_retention_pct"],
            preprocess_time_s,
        )

    summary_df = pd.DataFrame(summary_rows)
    fold_df_all = pd.concat(all_fold_rows, ignore_index=True)
    summary_df.to_csv(
        out_dir / "baseline_method_validation_summary.csv", index=False, encoding="utf-8-sig"
    )
    fold_df_all.to_csv(
        out_dir / "baseline_method_validation_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )

    figure_path = fig_dir / "baseline_method_validation_summary.png"
    make_summary_figure(summary_df, figure_path)

    LOGGER.info("Saved summary: %s", out_dir / "baseline_method_validation_summary.csv")
    LOGGER.info("Saved fold metrics: %s", out_dir / "baseline_method_validation_fold_metrics.csv")
    LOGGER.info("Saved figure: %s", figure_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
