#!/usr/bin/env python3
"""Render figures from STK-V2 fixed-split artifacts.

Expected input files under ``--run-dir``:
  - fixed_test_predictions.npz
  - fixed_val_predictions.npz
  - fixed_train_oof.npz
  - roc_curve_data.npz
  - single_vs_ensemble.csv
  - cancer_type_metrics.csv
  - confusion_matrix_counts.csv
  - confusion_matrix_normalized.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from .utils import (
        NATURE_BLUE,
        NATURE_GOLD,
        NATURE_GRAY,
        NATURE_GRID,
        NATURE_MUTED,
        NATURE_RED,
        NATURE_TEAL,
        NATURE_TEXT,
        add_panel_label,
        apply_nature_style,
        get_palette_and_labels,
        light_colormap,
        save_nature_figure,
        softened_color,
        style_axis,
        style_matrix_axis,
    )
except ImportError:
    from utils import (  # type: ignore
        NATURE_BLUE,
        NATURE_GOLD,
        NATURE_GRAY,
        NATURE_GRID,
        NATURE_MUTED,
        NATURE_RED,
        NATURE_TEAL,
        NATURE_TEXT,
        add_panel_label,
        apply_nature_style,
        get_palette_and_labels,
        light_colormap,
        save_nature_figure,
        softened_color,
        style_axis,
        style_matrix_axis,
    )

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")
KNOWN_PEAKS = (
    (448.1, "ring_deform", 15), (538.7, "SS_stretch", 15),
    (617.7, "CS_stretch", 15), (683.3, "creatinine", 15),
    (723.8, "adenine", 15), (795.1, "hippuric", 15),
    (849.1, "tyrosine", 15), (895.4, "uric_acid", 15),
    (933.9, "creatinine2", 15), (999.5, "phe_urea", 15),
    (1147.9, "uric_CN", 15), (1230.8, "amide_III", 20),
    (1292.5, "CH2_twist", 15), (1352.3, "trp_fermi", 15),
    (1448.7, "CH2_deform", 20), (1597.1, "purine_CC", 20),
    (1651.1, "amide_I", 20),
)
PEAK_RATIOS = (
    ("phe_urea", "adenine"), ("phe_urea", "creatinine"),
    ("hippuric", "creatinine"), ("CS_stretch", "creatinine"),
    ("amide_I", "CH2_deform"), ("adenine", "purine_CC"),
    ("tyrosine", "phe_urea"),
)
FALLBACK_GROUP_COLORS = {
    "PRO": "#CC79A7",
    "BRE": "#56A6D6",
    "OVA": "#7B3294",
    "LUN": "#2A9D8F",
    "CRC": "#2B6CB0",
    "PAN": "#C94842",
    "BLC": "#D99A21",
}
TASK_LABELS = {
    "binary": "Cancer vs Non-cancer",
    "type": "Cancer Type",
}
METRIC_LABELS = {
    "auroc": "AUROC",
    "pr_auc": "PR-AUC",
    "accuracy": "Accuracy",
    "sensitivity": "Sensitivity",
    "specificity": "Specificity",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "ovr_auroc_macro": "OVR AUROC",
    "log_loss": "Log Loss",
}


def _default_fig_dir(run_dir: Path) -> Path:
    return Path("results") / "figures" / "training" / run_dir.name


def _save_with_aliases(fig: plt.Figure, out: Path, dpi: int, aliases: list[Path] | None = None) -> None:
    save_nature_figure(fig, out, dpi=dpi, aliases=aliases)


def _as_path_list(items: list[Path] | Path | None) -> list[Path]:
    if items is None:
        return []
    if isinstance(items, list):
        return items
    return [items]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_cancer_types(run_dir: Path, n_classes: int | None = None) -> list[str]:
    meta = _read_json(run_dir / "fixed_split_results.json")
    names = list(meta.get("cancer_types") or CANCER_TYPES)
    if n_classes is not None:
        names = names[:n_classes]
        if len(names) < n_classes:
            names.extend([f"class_{i}" for i in range(len(names), n_classes)])
    return names


def _load_model_names(run_dir: Path, oof: np.lib.npyio.NpzFile | None = None) -> list[str]:
    meta = _read_json(run_dir / "fixed_split_results.json")
    names = list(meta.get("model_names") or [])
    if names:
        return names
    if oof is None:
        oof_path = run_dir / "fixed_train_oof.npz"
        if not oof_path.exists():
            return []
        oof = np.load(oof_path, allow_pickle=True)
    return sorted(
        k[:-3] for k in oof.files
        if k.endswith("_s1") and not k.startswith("meta_")
    )


def _load_group_colors(config_path: Path | None = None) -> dict[str, str]:
    colors = dict(FALLBACK_GROUP_COLORS)
    try:
        _, _, palette, _ = get_palette_and_labels()
        colors.update(palette)
    except Exception:
        pass
    config_path = config_path or (_repo_root() / "config" / "config.yaml")
    if yaml is None or not config_path.exists():
        return colors
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    display_colors = ((cfg.get("display") or {}).get("group_colors") or {})
    for key, value in display_colors.items():
        colors.setdefault(str(key), str(value))
    return colors


def _load_fixed_predictions(run_dir: Path, split: str = "test") -> np.lib.npyio.NpzFile | None:
    path = run_dir / f"fixed_{split}_predictions.npz"
    if not path.exists():
        return None
    return np.load(path, allow_pickle=True)


def _prob_clip(prob: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(prob, dtype=float), 1e-15, 1 - 1e-15)


def _present_label_indices(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    keep = set(int(v) for v in y_true if 0 <= int(v) < n_classes)
    keep.update(int(v) for v in y_pred if 0 <= int(v) < n_classes)
    return sorted(keep)


def _row_normalize(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    denom = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, denom, out=np.zeros_like(cm, dtype=float), where=denom > 0)


def _parse_group_list(values: list[str] | None) -> set[str] | None:
    if values is None:
        return None
    out: set[str] = set()
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                out.add(part.upper())
    return out


def _default_exclude_source_groups(run_dir: Path) -> set[str]:
    summary = _read_json(run_dir / "fixed_split_run_summary.json")
    return {str(v).upper() for v in summary.get("exclude_source_groups", [])}


def peak_feature_names() -> list[str]:
    names: list[str] = []
    for kind in ("area", "height", "fwhm", "shift"):
        names.extend([f"{peak_name}_{kind}" for _, peak_name, _ in KNOWN_PEAKS])
    names.extend([f"ratio_{num}_over_{den}" for num, den in PEAK_RATIOS])
    return names


def _peak_entity(feature_name: str) -> str:
    for suffix in ("_area", "_height", "_fwhm", "_shift"):
        if feature_name.endswith(suffix):
            return feature_name[: -len(suffix)]
    if feature_name.startswith("ratio_"):
        return feature_name[len("ratio_"):].replace("_over_", "/")
    return feature_name


def _roc_curve_with_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    fpr_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    valid = np.isfinite(scores)
    y_true = y_true[valid]
    scores = scores[valid]
    if len(np.unique(y_true)) < 2:
        raise ValueError("ROC requires both positive and negative labels")

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = float(roc_auc_score(y_true, scores))
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 101)

    result: dict[str, np.ndarray | float] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc,
        "auc_ci_low": np.nan,
        "auc_ci_high": np.nan,
        "fpr_grid": fpr_grid,
        "tpr_ci_low": np.full_like(fpr_grid, np.nan, dtype=float),
        "tpr_ci_high": np.full_like(fpr_grid, np.nan, dtype=float),
    }
    if n_bootstrap <= 0:
        return result

    rng = np.random.default_rng(seed)
    aucs: list[float] = []
    tprs: list[np.ndarray] = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], scores[idx])
        aucs.append(float(roc_auc_score(y_true[idx], scores[idx])))
        interp = np.interp(fpr_grid, fpr_b, tpr_b)
        interp[0] = 0.0
        interp[-1] = 1.0
        tprs.append(interp)

    if aucs:
        result["auc_ci_low"], result["auc_ci_high"] = np.percentile(aucs, [2.5, 97.5])
    if tprs:
        tpr_arr = np.vstack(tprs)
        result["tpr_ci_low"] = np.percentile(tpr_arr, 2.5, axis=0)
        result["tpr_ci_high"] = np.percentile(tpr_arr, 97.5, axis=0)
    return result


def _auc_label(prefix: str, roc_stats: dict[str, np.ndarray | float], digits: int = 3) -> str:
    auc = float(roc_stats["auc"])
    lo = float(roc_stats["auc_ci_low"])
    hi = float(roc_stats["auc_ci_high"])
    if np.isfinite(lo) and np.isfinite(hi):
        return f"{prefix} AUC={auc:.{digits}f} (95% CI {lo:.{digits}f}-{hi:.{digits}f})"
    return f"{prefix} AUC={auc:.{digits}f}"


def _binary_eval(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    prob = _prob_clip(prob)
    pred = (prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "n": float(len(y_true)),
        "log_loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan,
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan,
        "f1": float(f1_score(y_true, pred, zero_division=0)),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = float(roc_auc_score(y_true, prob))
        out["pr_auc"] = float(average_precision_score(y_true, prob))
    else:
        out["auroc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def _cancer_type_arrays(d: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_bin = np.asarray(d["y_bin"]).astype(int)
    y_type = np.asarray(d["y_type"]).astype(int)
    s2_prob = np.asarray(d["s2_prob"], dtype=float)
    cancer_mask = y_bin == 1
    return y_type[cancer_mask], s2_prob[cancer_mask], cancer_mask


def _type_log_loss(y_true: np.ndarray, prob: np.ndarray, n_classes: int) -> float:
    if len(y_true) == 0:
        return np.nan
    return float(log_loss(y_true, _prob_clip(prob), labels=list(range(n_classes))))


def _type_eval(y_true: np.ndarray, prob: np.ndarray, n_classes: int) -> dict[str, float]:
    if len(y_true) == 0:
        return {
            "n": 0.0,
            "log_loss": np.nan,
            "accuracy": np.nan,
            "macro_precision": np.nan,
            "macro_recall": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "ovr_auroc_macro": np.nan,
        }
    y_true = np.asarray(y_true, dtype=int)
    prob = np.asarray(prob, dtype=float)
    pred = prob.argmax(axis=1)
    present = _present_label_indices(y_true, pred, n_classes)
    aucs = []
    for idx in present:
        y_one = (y_true == idx).astype(int)
        if y_one.sum() == 0 or y_one.sum() == len(y_one):
            continue
        aucs.append(float(roc_auc_score(y_one, prob[:, idx])))
    return {
        "n": float(len(y_true)),
        "log_loss": _type_log_loss(y_true, prob, n_classes),
        "accuracy": float(np.mean(pred == y_true)),
        "macro_precision": float(precision_score(y_true, pred, labels=present, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, labels=present, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, labels=present, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=present, average="weighted", zero_division=0)),
        "ovr_auroc_macro": float(np.mean(aucs)) if aucs else np.nan,
    }


def _compute_cancer_type_metrics(
    y_true: np.ndarray,
    prob: np.ndarray,
    labels: list[str],
) -> pd.DataFrame:
    n_classes = len(labels)
    if len(y_true) == 0:
        return pd.DataFrame()
    pred = prob.argmax(axis=1)
    present = _present_label_indices(y_true, pred, n_classes)
    cm = confusion_matrix(y_true, pred, labels=present)
    rows = []
    for pos, idx in enumerate(present):
        tp = cm[pos, pos]
        fn = cm[pos, :].sum() - tp
        fp = cm[:, pos].sum() - tp
        tn = cm.sum() - tp - fp - fn
        rows.append({
            "cancer": labels[idx],
            "n": int((y_true == idx).sum()),
            "predicted_n": int((pred == idx).sum()),
            "sensitivity": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "specificity": float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
            "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
            "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def plot_cancer_vs_non_cancer_roc(
    run_dir: Path,
    fig_dir: Path,
    dpi: int = 300,
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> Path | None:
    pred_data = _load_fixed_predictions(run_dir, "test")
    if pred_data is not None:
        roc_stats = _roc_curve_with_ci(
            np.asarray(pred_data["y_bin"]).astype(int),
            np.asarray(pred_data["s1_prob"], dtype=float),
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed,
        )
        fpr = np.asarray(roc_stats["fpr"], dtype=float)
        tpr = np.asarray(roc_stats["tpr"], dtype=float)
        label = _auc_label("", roc_stats, digits=4).strip()
        pd.DataFrame([{
            "task": TASK_LABELS["binary"],
            "auc": float(roc_stats["auc"]),
            "auc_ci_low": float(roc_stats["auc_ci_low"]),
            "auc_ci_high": float(roc_stats["auc_ci_high"]),
            "n_bootstrap": int(n_bootstrap),
        }]).to_csv(fig_dir / "cancer_vs_non_cancer_roc_ci.csv", index=False)
    else:
        roc_path = run_dir / "roc_curve_data.npz"
        if not roc_path.exists():
            return None
        d = np.load(roc_path)
        fpr = d["fpr"]
        tpr = d["tpr"]
        auc = float(np.ravel(d["auc"])[0]) if "auc" in d.files else np.nan
        label = f"AUC={auc:.4f}"
        roc_stats = None

    fig, ax = plt.subplots(figsize=(4.15, 3.75))
    ax.plot(fpr, tpr, color=NATURE_BLUE, lw=1.65, label="ROC")
    if pred_data is not None:
        fpr_grid = np.asarray(roc_stats["fpr_grid"], dtype=float)
        low = np.asarray(roc_stats["tpr_ci_low"], dtype=float)
        high = np.asarray(roc_stats["tpr_ci_high"], dtype=float)
        if np.isfinite(low).any() and np.isfinite(high).any():
            ax.fill_between(
                fpr_grid, low, high,
                color=NATURE_BLUE, alpha=0.16, linewidth=0,
            )
    ax.plot([0, 1], [0, 1], color=NATURE_MUTED, ls=(0, (3, 3)), lw=0.85)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Cancer detection ROC")
    ax.text(
        0.98, 0.10, label.replace("AUC=", "AUROC = "),
        transform=ax.transAxes,
        ha="right", va="bottom",
        fontsize=7.2,
        color=NATURE_TEXT,
        bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": NATURE_GRID, "linewidth": 0.6},
    )
    style_axis(ax, grid="both")
    fig.tight_layout()
    out = fig_dir / "cancer_vs_non_cancer_roc.png"
    _save_with_aliases(fig, out, dpi, [fig_dir / "fixed_stage1_roc.png"])
    plt.close(fig)
    return out


def plot_cancer_type_confusion(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    pred_data = _load_fixed_predictions(run_dir, "test")
    if pred_data is not None:
        y_true, prob, _ = _cancer_type_arrays(pred_data)
        labels_all = _load_cancer_types(run_dir, prob.shape[1])
        if len(y_true) == 0:
            return None
        pred = prob.argmax(axis=1)
        keep = _present_label_indices(y_true, pred, len(labels_all))
        cm_values = confusion_matrix(y_true, pred, labels=keep)
        cm_norm_values = _row_normalize(cm_values)
        labels = [labels_all[i] for i in keep]
    else:
        counts_path = run_dir / "confusion_matrix_counts.csv"
        norm_path = run_dir / "confusion_matrix_normalized.csv"
        if not counts_path.exists() or not norm_path.exists():
            return None
        cm = pd.read_csv(counts_path, index_col=0)
        cm_norm = pd.read_csv(norm_path, index_col=0)
        keep_mask = (cm.sum(axis=0) + cm.sum(axis=1)) > 0
        cm = cm.loc[keep_mask, keep_mask]
        cm_norm = cm_norm.loc[cm.index, cm.columns]
        cm_values = cm.values
        cm_norm_values = cm_norm.values
        labels = cm.index.astype(str).tolist()
    if len(labels) == 0:
        return None

    fig, ax = plt.subplots(figsize=(4.9, 4.45))
    cmap = light_colormap("fixed_confusion_blue", NATURE_BLUE)
    im = ax.imshow(cm_norm_values, cmap=cmap, vmin=0, vmax=1, aspect="equal")
    for (i, j), value in np.ndenumerate(cm_values):
        norm_value = float(cm_norm_values[i, j])
        if int(value) == 0 and norm_value < 0.005:
            continue
        ax.text(
            j, i, f"{int(value)}\n{norm_value:.2f}",
            ha="center", va="center",
            color="white" if norm_value > 0.62 else NATURE_TEXT,
            fontsize=7.4,
            fontweight="bold" if i == j else "normal",
        )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Cancer-type confusion matrix")
    style_matrix_axis(ax, len(labels), len(labels))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Row-normalized", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
    fig.tight_layout()
    out = fig_dir / "cancer_type_confusion_matrix.png"
    _save_with_aliases(fig, out, dpi, [fig_dir / "fixed_type_confusion_matrix.png"])
    plt.close(fig)
    return out


def plot_cancer_type_metrics(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    pred_data = _load_fixed_predictions(run_dir, "test")
    if pred_data is not None:
        y_true, prob, _ = _cancer_type_arrays(pred_data)
        labels = _load_cancer_types(run_dir, prob.shape[1])
        df = _compute_cancer_type_metrics(y_true, prob, labels)
    else:
        metrics_path = run_dir / "cancer_type_metrics.csv"
        if not metrics_path.exists():
            return None
        df = pd.read_csv(metrics_path)
        value_cols = [c for c in ["n", "predicted_n", "sensitivity", "precision", "recall", "f1"] if c in df.columns]
        if value_cols:
            nonempty = df[value_cols].fillna(0).sum(axis=1) > 0
            df = df.loc[nonempty].copy()
    if df.empty:
        return None
    metrics = ["sensitivity", "specificity", "precision", "f1"]
    present = [m for m in metrics if m in df.columns]
    if not present:
        return None
    df.to_csv(fig_dir / "cancer_type_metrics_present.csv", index=False)

    cancers = df["cancer"].astype(str).tolist()
    matrix = df[present].astype(float).T.to_numpy()
    fig_width = max(5.2, 2.2 + 0.45 * len(cancers))
    fig_height = max(2.75, 1.4 + 0.34 * len(present))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(matrix, cmap=light_colormap("fixed_metric_teal", NATURE_TEAL), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cancers)))
    ax.set_xticklabels(cancers)
    ax.set_yticks(np.arange(len(present)))
    ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in present])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(
            j, i, f"{value:.2f}",
            ha="center", va="center",
            fontsize=7.2,
            color="white" if value > 0.68 else NATURE_TEXT,
            fontweight="bold" if value > 0.90 else "normal",
        )
    ax.set_xlabel("Cancer type")
    ax.set_title("Cancer-type classification metrics")
    style_matrix_axis(ax, len(present), len(cancers))
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Score", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7, length=2, width=0.5)
    fig.tight_layout()
    out = fig_dir / "cancer_type_metrics.png"
    _save_with_aliases(fig, out, dpi, [fig_dir / "fixed_cancer_type_metrics.png"])
    plt.close(fig)
    return out


def plot_single_vs_ensemble(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    csv_path = run_dir / "single_vs_ensemble.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    if not {"model", "type", "auc", "f1_type"}.issubset(df.columns):
        return None

    color_map = {"single": NATURE_GRAY, "ensemble": NATURE_GOLD, "meta": NATURE_RED}
    marker_map = {"single": "o", "ensemble": "s", "meta": "^"}
    label_map = {"single": "Single model", "ensemble": "Simple ensemble", "meta": "Stacked meta"}
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    for typ in ["single", "ensemble", "meta"]:
        sub = df[df["type"] == typ]
        if sub.empty:
            continue
        ax.scatter(
            sub["auc"], sub["f1_type"],
            s=48 if typ == "single" else 68,
            marker=marker_map.get(typ, "o"),
            color=color_map.get(typ, "#333333"),
            edgecolor="white", linewidth=0.7,
            label=label_map.get(typ, str(typ)),
            alpha=0.92,
            zorder=2 if typ == "single" else 4,
        )
        for _, row in sub.iterrows():
            is_key = str(row["type"]) != "single"
            if not is_key:
                continue
            ax.annotate(
                str(row["model"]), (row["auc"], row["f1_type"]),
                xytext=(5, -9 if typ == "ensemble" else 5), textcoords="offset points",
                fontsize=7.1,
                fontweight="bold",
                color=NATURE_TEXT,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.70},
            )
    x_min = max(0.0, float(df["auc"].min()) - 0.025)
    y_min = max(0.0, float(df["f1_type"].min()) - 0.035)
    ax.set_xlim(x_min, min(1.01, float(df["auc"].max()) + 0.05))
    ax.set_ylim(y_min, min(1.01, float(df["f1_type"].max()) + 0.06))
    ax.set_xlabel("Cancer detection AUROC")
    ax.set_ylabel("Cancer-type macro F1")
    ax.set_title("Single models vs stacked ensemble")
    style_axis(ax, grid="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = fig_dir / "single_vs_ensemble.png"
    _save_with_aliases(fig, out, dpi, [fig_dir / "fixed_single_vs_ensemble.png"])
    plt.close(fig)
    return out


def plot_cancer_type_roc(
    run_dir: Path,
    fig_dir: Path,
    dpi: int = 300,
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
) -> Path | None:
    pred_data = _load_fixed_predictions(run_dir, "test")
    if pred_data is None:
        return None
    y_true, prob, _ = _cancer_type_arrays(pred_data)
    if len(y_true) == 0:
        return None
    labels = _load_cancer_types(run_dir, prob.shape[1])
    colors = _load_group_colors()

    fig, ax = plt.subplots(figsize=(4.65, 4.05))
    plotted = 0
    ci_rows = []
    for idx, label in enumerate(labels):
        y_one = (y_true == idx).astype(int)
        if y_one.sum() == 0 or y_one.sum() == len(y_one):
            continue
        roc_stats = _roc_curve_with_ci(
            y_one,
            prob[:, idx],
            n_bootstrap=n_bootstrap,
            seed=bootstrap_seed + idx + 1,
        )
        fpr = np.asarray(roc_stats["fpr"], dtype=float)
        tpr = np.asarray(roc_stats["tpr"], dtype=float)
        color = colors.get(label, "#333333")
        ax.plot(
            fpr, tpr, lw=1.35,
            color=color,
            label=f"{label} AUROC {float(roc_stats['auc']):.2f}",
        )
        low = np.asarray(roc_stats["tpr_ci_low"], dtype=float)
        high = np.asarray(roc_stats["tpr_ci_high"], dtype=float)
        if np.isfinite(low).any() and np.isfinite(high).any():
            ax.fill_between(
                np.asarray(roc_stats["fpr_grid"], dtype=float),
                low, high, color=color, alpha=0.06, linewidth=0,
            )
        ci_rows.append({
            "cancer_type": label,
            "auc": float(roc_stats["auc"]),
            "auc_ci_low": float(roc_stats["auc_ci_low"]),
            "auc_ci_high": float(roc_stats["auc_ci_high"]),
            "n_positive": int(y_one.sum()),
            "n_negative": int(len(y_one) - y_one.sum()),
            "n_bootstrap": int(n_bootstrap),
        })
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return None
    pd.DataFrame(ci_rows).to_csv(fig_dir / "cancer_type_roc_ci.csv", index=False)
    ax.plot([0, 1], [0, 1], color=NATURE_MUTED, ls=(0, (3, 3)), lw=0.8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("One-vs-rest cancer-type ROC")
    style_axis(ax, grid="both")
    ax.legend(loc="lower right", fontsize=6.5, ncol=1)
    fig.tight_layout()
    out = fig_dir / "cancer_type_roc_ovr.png"
    _save_with_aliases(fig, out, dpi)
    plt.close(fig)
    return out


def _best_meta_name(run_dir: Path) -> str:
    meta = _read_json(run_dir / "fixed_split_results.json")
    return str(meta.get("best_meta") or "elasticnet")


def _make_linear_meta(task: str, meta_name: str) -> LogisticRegression | None:
    if meta_name == "elasticnet":
        return LogisticRegression(
            C=0.5, penalty="elasticnet", l1_ratio=0.5, max_iter=2000,
            solver="saga", n_jobs=1, random_state=42,
        )
    if meta_name == "lr":
        return LogisticRegression(
            C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1,
        )
    return None


def _expand_sparse_proba(model: LogisticRegression, prob: np.ndarray, n_classes: int) -> np.ndarray:
    full = np.zeros((prob.shape[0], n_classes), dtype=float)
    for col, cls in enumerate(model.classes_.astype(int)):
        if 0 <= cls < n_classes:
            full[:, cls] = prob[:, col]
    return full


def _fit_linear_meta_from_oof(
    run_dir: Path,
) -> tuple[LogisticRegression, LogisticRegression, np.lib.npyio.NpzFile, list[str]] | None:
    oof_path = run_dir / "fixed_train_oof.npz"
    if not oof_path.exists():
        return None
    oof = np.load(oof_path, allow_pickle=True)
    meta_name = _best_meta_name(run_dir)
    s1 = _make_linear_meta("binary", meta_name)
    s2 = _make_linear_meta("multiclass", meta_name)
    if s1 is None or s2 is None:
        return None
    x_s1 = np.asarray(oof["meta_s1"], dtype=float)
    y_bin = np.asarray(oof["y_bin"]).astype(int)
    x_s2 = np.asarray(oof["meta_s2"], dtype=float)
    y_type = np.asarray(oof["y_type"]).astype(int)
    cancer = y_bin == 1
    if cancer.sum() == 0:
        return None
    s1.fit(x_s1, y_bin)
    s2.fit(x_s2[cancer], y_type[cancer])
    return s1, s2, oof, _load_model_names(run_dir, oof)


def _training_meta_predictions(
    run_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    fitted = _fit_linear_meta_from_oof(run_dir)
    if fitted is None:
        return None
    s1, s2, oof, _ = fitted
    x_s1 = np.asarray(oof["meta_s1"], dtype=float)
    x_s2 = np.asarray(oof["meta_s2"], dtype=float)
    y_bin = np.asarray(oof["y_bin"]).astype(int)
    y_type = np.asarray(oof["y_type"]).astype(int)
    n_classes = x_s2.shape[1] // max(len(_load_model_names(run_dir, oof)), 1)
    s1_prob = s1.predict_proba(x_s1)[:, 1]
    s2_prob = np.zeros((len(y_bin), n_classes), dtype=float)
    cancer = y_bin == 1
    s2_prob[cancer] = _expand_sparse_proba(
        s2, s2.predict_proba(x_s2[cancer]), n_classes,
    )
    return y_bin, y_type, s1_prob, s2_prob


def compute_loss_summary(run_dir: Path) -> pd.DataFrame:
    rows = []
    train_pred = _training_meta_predictions(run_dir)
    if train_pred is not None:
        y_bin, y_type, s1_prob, s2_prob = train_pred
        rows.extend(_loss_rows("Train OOF", y_bin, y_type, s1_prob, s2_prob))
    for split_name, label in [("val", "Validation Set"), ("test", "Test Set")]:
        d = _load_fixed_predictions(run_dir, split_name)
        if d is None:
            continue
        rows.extend(_loss_rows(
            label,
            np.asarray(d["y_bin"]).astype(int),
            np.asarray(d["y_type"]).astype(int),
            np.asarray(d["s1_prob"], dtype=float),
            np.asarray(d["s2_prob"], dtype=float),
        ))
    return pd.DataFrame(rows)


def _loss_rows(
    split_label: str,
    y_bin: np.ndarray,
    y_type: np.ndarray,
    s1_prob: np.ndarray,
    s2_prob: np.ndarray,
) -> list[dict[str, object]]:
    n_classes = s2_prob.shape[1]
    rows = [{
        "split": split_label,
        "task": TASK_LABELS["binary"],
        "loss": "log_loss",
        "value": float(log_loss(y_bin, _prob_clip(s1_prob), labels=[0, 1])),
    }]
    cancer = y_bin == 1
    if cancer.sum() > 0:
        rows.append({
            "split": split_label,
            "task": TASK_LABELS["type"],
            "loss": "log_loss",
            "value": _type_log_loss(y_type[cancer], s2_prob[cancer], n_classes),
        })
    return rows


def plot_loss_summary(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    df = compute_loss_summary(run_dir)
    if df.empty:
        return None
    df.to_csv(fig_dir / "loss_summary.csv", index=False)
    split_order = [s for s in ["Train OOF", "Validation Set", "Test Set"] if s in set(df["split"])]
    task_order = [t for t in [TASK_LABELS["binary"], TASK_LABELS["type"]] if t in set(df["task"])]
    width = 0.8 / max(len(split_order), 1)
    x = np.arange(len(task_order))
    colors = {"Train OOF": NATURE_GRAY, "Validation Set": NATURE_BLUE, "Test Set": NATURE_RED}

    fig, ax = plt.subplots(figsize=(4.8, 3.25))
    for i, split in enumerate(split_order):
        sub = df[df["split"] == split].set_index("task")
        values = [float(sub.loc[t, "value"]) if t in sub.index else np.nan for t in task_order]
        offset = (i - (len(split_order) - 1) / 2) * width
        bars = ax.bar(
            x + offset, values,
            width=width * 0.82,
            label=split,
            color=colors.get(split, "#333333"),
            edgecolor="white",
            linewidth=0.5,
        )
        for bar, value in zip(bars, values):
            if np.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.01,
                    f"{value:.2f}",
                    ha="center", va="bottom",
                    fontsize=6.6,
                    rotation=90,
                    color=NATURE_MUTED,
                )
    ax.set_xticks(x)
    ax.set_xticklabels(task_order)
    ax.set_ylabel("Log loss")
    ax.set_title("Log loss by split")
    ax.margins(y=0.18)
    style_axis(ax, grid="y")
    ax.legend(loc="upper left", ncol=1)
    fig.tight_layout()
    out = fig_dir / "loss_summary.png"
    _save_with_aliases(fig, out, dpi)
    plt.close(fig)
    return out


def compute_test_evaluation(run_dir: Path) -> pd.DataFrame:
    d = _load_fixed_predictions(run_dir, "test")
    if d is None:
        return pd.DataFrame()
    y_bin = np.asarray(d["y_bin"]).astype(int)
    y_type = np.asarray(d["y_type"]).astype(int)
    s1_prob = np.asarray(d["s1_prob"], dtype=float)
    s2_prob = np.asarray(d["s2_prob"], dtype=float)

    rows = []
    for metric, value in _binary_eval(y_bin, s1_prob).items():
        rows.append({"task": TASK_LABELS["binary"], "metric": metric, "value": value})

    cancer = y_bin == 1
    type_metrics = _type_eval(y_type[cancer], s2_prob[cancer], s2_prob.shape[1])
    for metric, value in type_metrics.items():
        rows.append({"task": TASK_LABELS["type"], "metric": metric, "value": value})
    return pd.DataFrame(rows)


def plot_test_evaluation_indices(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    df = compute_test_evaluation(run_dir)
    if df.empty:
        return None
    df.to_csv(fig_dir / "test_classification_evaluation.csv", index=False)

    score_metrics = {
        TASK_LABELS["binary"]: ["auroc", "pr_auc", "accuracy", "sensitivity", "specificity", "precision", "f1"],
        TASK_LABELS["type"]: ["ovr_auroc_macro", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1"],
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.3, 3.65), sharex=True)
    colors = {TASK_LABELS["binary"]: NATURE_BLUE, TASK_LABELS["type"]: NATURE_RED}
    for ax, task in zip(axes, [TASK_LABELS["binary"], TASK_LABELS["type"]]):
        sub = df[df["task"] == task].set_index("metric")
        metrics = [m for m in score_metrics[task] if m in sub.index and np.isfinite(float(sub.loc[m, "value"]))]
        values = [float(sub.loc[m, "value"]) for m in metrics]
        y = np.arange(len(metrics))
        ax.barh(y, values, color=colors[task], alpha=0.90, height=0.62)
        ax.set_yticks(y)
        ax.set_yticklabels([METRIC_LABELS.get(m, m) for m in metrics])
        ax.set_xlim(0, 1.04)
        ax.set_title(task)
        style_axis(ax, grid="x")
        ax.invert_yaxis()
        for yi, value in zip(y, values):
            inside = value > 0.86
            ax.text(
                value - 0.025 if inside else value + 0.018,
                yi,
                f"{value:.3f}",
                va="center",
                ha="right" if inside else "left",
                fontsize=6.9,
                color="white" if inside else NATURE_MUTED,
            )
    axes[0].set_xlabel("Score")
    axes[1].set_xlabel("Score")
    add_panel_label(axes[0], "a", x=-0.20, y=1.08)
    add_panel_label(axes[1], "b", x=-0.18, y=1.08)
    fig.suptitle("Test-set classification performance", y=1.02)
    fig.tight_layout()
    out = fig_dir / "test_classification_evaluation_indices.png"
    _save_with_aliases(fig, out, dpi)
    plt.close(fig)
    return out


def plot_meta_shap_input_importance(run_dir: Path, fig_dir: Path, dpi: int = 300) -> Path | None:
    fitted = _fit_linear_meta_from_oof(run_dir)
    if fitted is None:
        return None
    s1, s2, oof, model_names = fitted
    if not model_names:
        return None
    x_s1 = np.asarray(oof["meta_s1"], dtype=float)
    x_s2 = np.asarray(oof["meta_s2"], dtype=float)
    y_bin = np.asarray(oof["y_bin"]).astype(int)
    cancer = y_bin == 1
    n_models = len(model_names)
    n_classes = x_s2.shape[1] // n_models
    cancer_types = _load_cancer_types(run_dir, n_classes)

    coef_s1 = np.ravel(s1.coef_)
    shap_s1 = np.abs((x_s1 - x_s1.mean(axis=0)) * coef_s1).mean(axis=0)

    coef_s2 = np.asarray(s2.coef_, dtype=float)
    centered_s2 = x_s2[cancer] - x_s2[cancer].mean(axis=0)
    shap_s2_feature = np.abs(centered_s2[:, None, :] * coef_s2[None, :, :]).mean(axis=(0, 1))
    shap_s2 = shap_s2_feature.reshape(n_models, n_classes).mean(axis=1)
    shap_s2_by_type = shap_s2_feature.reshape(n_models, n_classes)

    rows = []
    for model, value in zip(model_names, shap_s1):
        rows.append({"task": TASK_LABELS["binary"], "base_model": model, "mean_abs_linear_shap": float(value)})
    for model, value in zip(model_names, shap_s2):
        rows.append({"task": TASK_LABELS["type"], "base_model": model, "mean_abs_linear_shap": float(value)})
    pd.DataFrame(rows).to_csv(fig_dir / "meta_shap_input_importance.csv", index=False)

    by_type_rows = []
    for mi, model in enumerate(model_names):
        for ci, ct in enumerate(cancer_types):
            by_type_rows.append({
                "base_model": model,
                "cancer_type": ct,
                "mean_abs_linear_shap": float(shap_s2_by_type[mi, ci]),
            })
    pd.DataFrame(by_type_rows).to_csv(fig_dir / "meta_shap_input_importance_by_type.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.75))
    for ax, values, title, color in [
        (axes[0], shap_s1, "Cancer detection inputs", NATURE_BLUE),
        (axes[1], shap_s2, "Cancer-type inputs", NATURE_RED),
    ]:
        order = np.argsort(values)[::-1]
        ordered_values = values[order]
        y = np.arange(len(order))
        ax.barh(y, ordered_values, color=color, alpha=0.90, height=0.62)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels([model_names[i] for i in order])
        ax.invert_yaxis()
        ax.set_xlabel("Mean |linear SHAP contribution|")
        ax.set_title(title)
        style_axis(ax, grid="x")
        x_max = float(np.nanmax(ordered_values)) if len(ordered_values) else 0.0
        ax.set_xlim(0, x_max * 1.18 if x_max > 0 else 1)
        for yi, value in zip(y, ordered_values):
            ax.text(value + max(x_max * 0.02, 1e-6), yi, f"{value:.3g}", va="center", fontsize=6.7, color=NATURE_MUTED)
    add_panel_label(axes[0], "a", x=-0.26, y=1.08)
    add_panel_label(axes[1], "b", x=-0.24, y=1.08)
    fig.tight_layout()
    out = fig_dir / "meta_shap_input_importance.png"
    _save_with_aliases(fig, out, dpi)
    plt.close(fig)
    return out


def _load_or_create_peak_features(
    run_dir: Path,
    *,
    data_dir: Path | None = None,
    data_type: str = "processed_csv",
    exclude_source_groups: set[str] | None = None,
    force_recompute: bool = False,
) -> np.lib.npyio.NpzFile | None:
    artifact_path = run_dir / "fixed_peak_features.npz"
    if artifact_path.exists() and not force_recompute:
        return np.load(artifact_path, allow_pickle=True)
    if data_dir is None:
        return None

    indices_path = run_dir / "fixed_split_indices.npz"
    if not indices_path.exists():
        return None

    # Reuse the active training loader/extractor so peak features match training.
    from scripts.training import train_usersnet as trainer

    trainer.DATA_TYPE = data_type
    effective_exclude = (
        set(exclude_source_groups)
        if exclude_source_groups is not None
        else _default_exclude_source_groups(run_dir)
    )
    X_3ch, _df_meta, grid, groups, sample_ids, y_bin, y_type = trainer.load_data(
        data_dir,
        exclude_source_groups=effective_exclude,
    )
    idx = np.load(indices_path, allow_pickle=True)
    train_idx = idx["train_idx"]
    val_idx = idx["val_idx"]
    test_idx = idx["test_idx"]
    if max(train_idx.max(initial=-1), val_idx.max(initial=-1), test_idx.max(initial=-1)) >= len(X_3ch):
        raise ValueError(
            "fixed_split_indices.npz is incompatible with loaded data. "
            "Check --data-dir, --data-type, and --exclude-source-groups."
        )

    X_peak_all = trainer.extract_peak_features(X_3ch[:, 0, :], grid)
    np.savez_compressed(
        artifact_path,
        X_peak_train=X_peak_all[train_idx],
        X_peak_val=X_peak_all[val_idx],
        X_peak_test=X_peak_all[test_idx],
        y_bin_train=np.asarray(y_bin[train_idx], dtype=np.int64),
        y_bin_val=np.asarray(y_bin[val_idx], dtype=np.int64),
        y_bin_test=np.asarray(y_bin[test_idx], dtype=np.int64),
        y_type_train=np.asarray(y_type[train_idx], dtype=np.int64),
        y_type_val=np.asarray(y_type[val_idx], dtype=np.int64),
        y_type_test=np.asarray(y_type[test_idx], dtype=np.int64),
        sample_ids_train=np.asarray(sample_ids[train_idx], dtype=object),
        sample_ids_val=np.asarray(sample_ids[val_idx], dtype=object),
        sample_ids_test=np.asarray(sample_ids[test_idx], dtype=object),
        groups_train=np.asarray(groups[train_idx], dtype=object),
        groups_val=np.asarray(groups[val_idx], dtype=object),
        groups_test=np.asarray(groups[test_idx], dtype=object),
        feature_names=np.asarray(peak_feature_names(), dtype=object),
        cancer_types=np.asarray(_load_cancer_types(run_dir, len(CANCER_TYPES)), dtype=object),
        exclude_source_groups=np.asarray(sorted(effective_exclude), dtype=object),
    )
    return np.load(artifact_path, allow_pickle=True)


def _make_peak_lr() -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver="saga",
            class_weight="balanced",
            n_jobs=1,
            random_state=42,
        ),
    )


def _linear_pipeline_contributions(model, X: np.ndarray, background: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scaler = model.named_steps["standardscaler"]
    clf = model.named_steps["logisticregression"]
    X_std = scaler.transform(X)
    bg_mean = scaler.transform(background).mean(axis=0)
    coef = np.asarray(clf.coef_, dtype=float)
    contrib = (X_std[:, None, :] - bg_mean[None, None, :]) * coef[None, :, :]
    return contrib, np.asarray(clf.classes_, dtype=int)


def _feature_importance_rows(
    values: np.ndarray,
    feature_names: list[str],
    *,
    task: str,
    cancer_type: str = "overall",
) -> list[dict[str, object]]:
    return [
        {
            "base_model": "lr_peak",
            "task": task,
            "cancer_type": cancer_type,
            "feature": feature,
            "peak_entity": _peak_entity(feature),
            "mean_abs_linear_shap": float(value),
        }
        for feature, value in zip(feature_names, values)
    ]


def _aggregate_peak_entities(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["base_model", "task", "cancer_type", "peak_entity"], as_index=False)["mean_abs_linear_shap"]
        .sum()
        .sort_values(["task", "cancer_type", "mean_abs_linear_shap"], ascending=[True, True, False])
    )


def _plot_top_importance_panels(
    df: pd.DataFrame,
    value_col: str,
    label_col: str,
    out: Path,
    dpi: int,
    title: str,
) -> Path:
    tasks = [TASK_LABELS["binary"], TASK_LABELS["type"]]
    colors = {TASK_LABELS["binary"]: NATURE_BLUE, TASK_LABELS["type"]: NATURE_RED}
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 4.25))
    for ax, task in zip(axes, tasks):
        sub = (
            df[(df["task"] == task) & (df["cancer_type"] == "overall")]
            .sort_values(value_col, ascending=False)
            .head(15)
            .iloc[::-1]
        )
        values = sub[value_col].astype(float).to_numpy()
        y = np.arange(len(sub))
        ax.barh(y, values, color=colors[task], alpha=0.90, height=0.68)
        ax.set_yticks(y)
        ax.set_yticklabels(sub[label_col])
        ax.set_xlabel("Mean |linear SHAP contribution|")
        ax.set_title(task)
        style_axis(ax, grid="x")
        x_max = float(np.nanmax(values)) if len(values) else 0.0
        ax.set_xlim(0, x_max * 1.18 if x_max > 0 else 1)
        for yi, value in zip(y, values):
            ax.text(value + max(x_max * 0.02, 1e-6), yi, f"{value:.3g}", va="center", fontsize=6.6, color=NATURE_MUTED)
    add_panel_label(axes[0], "a", x=-0.32, y=1.07)
    add_panel_label(axes[1], "b", x=-0.32, y=1.07)
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    _save_with_aliases(fig, out, dpi)
    plt.close(fig)
    return out


def _plot_peak_type_heatmap(
    shap_df: pd.DataFrame,
    cancer_types: list[str],
    colors: dict[str, str],
    fig_dir: Path,
    dpi: int,
    *,
    label_col: str = "feature",
    out_name: str = "peak_level_shap_by_cancer_type.png",
    title: str = "lr_peak Test Set Peak-Feature SHAP by Cancer Type",
) -> Path | None:
    class_df = shap_df[
        (shap_df["task"] == TASK_LABELS["type"])
        & (shap_df["cancer_type"] != "overall")
    ].copy()
    if class_df.empty:
        return None
    top_labels = (
        class_df.groupby(label_col)["mean_abs_linear_shap"]
        .sum()
        .sort_values(ascending=False)
        .head(20)
        .index
        .tolist()
    )
    present_types = [ct for ct in cancer_types if ct in set(class_df["cancer_type"])]
    if not top_labels or not present_types:
        return None
    mat = np.zeros((len(present_types), len(top_labels)))
    for i, ct in enumerate(present_types):
        sub = class_df[class_df["cancer_type"] == ct].groupby(label_col)["mean_abs_linear_shap"].sum()
        for j, label in enumerate(top_labels):
            if label in sub.index:
                mat[i, j] = float(sub.loc[label])

    fig_width = max(7.3, 2.8 + 0.24 * len(top_labels))
    fig_height = max(3.25, 1.5 + 0.30 * len(present_types))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(mat, cmap=light_colormap("peak_type_shap_gold", NATURE_GOLD), aspect="auto")
    ax.set_xticks(np.arange(len(top_labels)))
    ax.set_xticklabels(top_labels, rotation=50, ha="right", fontsize=6.8)
    ax.set_yticks(np.arange(len(present_types)))
    ax.set_yticklabels(present_types)
    for tick, ct in zip(ax.get_yticklabels(), present_types):
        tick.set_color(colors.get(ct, "black"))
        tick.set_fontweight("bold")
    ax.set_title(title)
    style_matrix_axis(ax, len(present_types), len(top_labels))
    cbar = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.02)
    cbar.set_label("Mean |linear SHAP contribution|", fontsize=7.2)
    cbar.ax.tick_params(labelsize=6.8, length=2, width=0.5)
    fig.tight_layout()
    out = fig_dir / out_name
    _save_with_aliases(fig, out, dpi)
    plt.close(fig)
    return out


def plot_peak_level_shap(
    run_dir: Path,
    fig_dir: Path,
    dpi: int = 300,
    *,
    data_dir: Path | None = None,
    data_type: str = "processed_csv",
    exclude_source_groups: set[str] | None = None,
    force_peak_recompute: bool = False,
) -> list[Path]:
    peak = _load_or_create_peak_features(
        run_dir,
        data_dir=data_dir,
        data_type=data_type,
        exclude_source_groups=exclude_source_groups,
        force_recompute=force_peak_recompute,
    )
    if peak is None:
        return []

    X_train = np.asarray(peak["X_peak_train"], dtype=float)
    X_test = np.asarray(peak["X_peak_test"], dtype=float)
    y_bin_train = np.asarray(peak["y_bin_train"], dtype=int)
    y_bin_test = np.asarray(peak["y_bin_test"], dtype=int)
    y_type_train = np.asarray(peak["y_type_train"], dtype=int)
    y_type_test = np.asarray(peak["y_type_test"], dtype=int)
    feature_names = [str(v) for v in peak["feature_names"]]
    cancer_types = [str(v) for v in peak["cancer_types"]]
    colors = _load_group_colors()

    if X_train.shape[1] != len(feature_names):
        feature_names = peak_feature_names()[: X_train.shape[1]]

    rows: list[dict[str, object]] = []
    s1 = _make_peak_lr()
    s1.fit(X_train, y_bin_train)
    s1_contrib, _ = _linear_pipeline_contributions(s1, X_test, X_train)
    s1_importance = np.abs(s1_contrib[:, 0, :]).mean(axis=0)
    rows.extend(_feature_importance_rows(
        s1_importance, feature_names,
        task=TASK_LABELS["binary"],
    ))

    cancer_train = y_bin_train == 1
    cancer_test = y_bin_test == 1
    if cancer_train.sum() > 10 and cancer_test.sum() > 0:
        s2 = _make_peak_lr()
        s2.fit(X_train[cancer_train], y_type_train[cancer_train])
        s2_contrib, s2_classes = _linear_pipeline_contributions(
            s2, X_test[cancer_test], X_train[cancer_train],
        )
        s2_overall = np.abs(s2_contrib).mean(axis=(0, 1))
        rows.extend(_feature_importance_rows(
            s2_overall, feature_names,
            task=TASK_LABELS["type"],
        ))
        y_type_test_cancer = y_type_test[cancer_test]
        for class_pos, class_id in enumerate(s2_classes):
            if class_id < 0 or class_id >= len(cancer_types):
                continue
            sample_mask = y_type_test_cancer == class_id
            if not sample_mask.any():
                continue
            class_values = np.abs(s2_contrib[sample_mask, class_pos, :]).mean(axis=0)
            rows.extend(_feature_importance_rows(
                class_values, feature_names,
                task=TASK_LABELS["type"],
                cancer_type=cancer_types[class_id],
            ))

    feature_df = pd.DataFrame(rows)
    if feature_df.empty:
        return []
    feature_df = feature_df.sort_values(
        ["task", "cancer_type", "mean_abs_linear_shap"],
        ascending=[True, True, False],
    )
    feature_df.to_csv(fig_dir / "peak_level_shap_features.csv", index=False)

    peak_df = _aggregate_peak_entities(feature_df)
    peak_df.to_csv(fig_dir / "peak_level_shap_peaks.csv", index=False)

    written: list[Path] = []
    written.append(_plot_top_importance_panels(
        feature_df, "mean_abs_linear_shap", "feature",
        fig_dir / "peak_level_shap_features.png", dpi,
        "lr_peak Test Set Peak-Feature SHAP",
    ))
    _plot_top_importance_panels(
        feature_df, "mean_abs_linear_shap", "feature",
        fig_dir / "peak_level_shap.png", dpi,
        "lr_peak Test Set Peak-Feature SHAP",
    )
    written.append(_plot_top_importance_panels(
        peak_df, "mean_abs_linear_shap", "peak_entity",
        fig_dir / "peak_level_shap_peaks.png", dpi,
        "lr_peak Test Set Peak Summary",
    ))
    heatmap = _plot_peak_type_heatmap(
        feature_df,
        cancer_types,
        colors,
        fig_dir,
        dpi,
        label_col="feature",
        out_name="peak_level_shap_by_cancer_type.png",
        title="lr_peak Test Set Peak-Feature SHAP by Cancer Type",
    )
    if heatmap is not None:
        written.append(heatmap)
    return written


def render_fixed_split_figures(
    run_dir: Path,
    fig_dir: Path | None = None,
    dpi: int = 300,
    *,
    n_bootstrap: int = 1000,
    bootstrap_seed: int = 42,
    data_dir: Path | None = None,
    data_type: str = "processed_csv",
    exclude_source_groups: set[str] | None = None,
    force_peak_recompute: bool = False,
) -> list[Path]:
    apply_nature_style()
    run_dir = Path(run_dir)
    fig_dir = Path(fig_dir) if fig_dir is not None else _default_fig_dir(run_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    written = [
        plot_cancer_vs_non_cancer_roc(
            run_dir, fig_dir, dpi=dpi,
            n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed,
        ),
        plot_cancer_type_confusion(run_dir, fig_dir, dpi=dpi),
        plot_cancer_type_metrics(run_dir, fig_dir, dpi=dpi),
        plot_single_vs_ensemble(run_dir, fig_dir, dpi=dpi),
        plot_cancer_type_roc(
            run_dir, fig_dir, dpi=dpi,
            n_bootstrap=n_bootstrap, bootstrap_seed=bootstrap_seed,
        ),
        plot_loss_summary(run_dir, fig_dir, dpi=dpi),
        plot_test_evaluation_indices(run_dir, fig_dir, dpi=dpi),
        plot_meta_shap_input_importance(run_dir, fig_dir, dpi=dpi),
        plot_peak_level_shap(
            run_dir, fig_dir, dpi=dpi,
            data_dir=data_dir,
            data_type=data_type,
            exclude_source_groups=exclude_source_groups,
            force_peak_recompute=force_peak_recompute,
        ),
    ]
    return [p for item in written for p in _as_path_list(item)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render STK-V2 fixed-split figures.")
    parser.add_argument("--run-dir", type=Path, default=Path("results/training/stacking_v2_fixed"))
    parser.add_argument("--fig-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap replicates for ROC 95%% CI; use 0 to disable CI.")
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Optional data source used to reconstruct peak features for peak-level SHAP.")
    parser.add_argument("--data-type", choices=["raw_spectrum", "processed_csv"], default="processed_csv")
    parser.add_argument("--exclude-source-groups", action="append", default=None,
                        help="Source groups excluded in the run, e.g. YPAN. Defaults to run summary when present.")
    parser.add_argument("--force-peak-recompute", action="store_true")
    args = parser.parse_args()

    written = render_fixed_split_figures(
        args.run_dir,
        args.fig_dir,
        args.dpi,
        n_bootstrap=args.n_bootstrap,
        bootstrap_seed=args.bootstrap_seed,
        data_dir=args.data_dir,
        data_type=args.data_type,
        exclude_source_groups=_parse_group_list(args.exclude_source_groups),
        force_peak_recompute=args.force_peak_recompute,
    )
    if not written:
        print(f"No fixed-split figures rendered from {args.run_dir}")
        return 1
    for path in written:
        print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
