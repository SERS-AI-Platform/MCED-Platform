#!/usr/bin/env python3
"""Calibrate fixed-split STK-V2 probabilities and report operating thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")


def _load_predictions(run_dir: Path, split: str) -> dict[str, np.ndarray]:
    path = run_dir / f"fixed_{split}_predictions.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _fit_platt(y_val: np.ndarray, p_val: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(_logit(p_val).reshape(-1, 1), y_val.astype(int))
    return model


def _apply_platt(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        acc = float(y[mask].mean())
        conf = float(p[mask].mean())
        value += mask.mean() * abs(acc - conf)
    return float(value)


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece_10bin": _ece(y, p, n_bins=10),
        "threshold": float(threshold),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "accuracy": float((tp + tn) / max(tp + tn + fp + fn, 1)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def _choose_thresholds(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y.astype(int), p)
    specificity = 1.0 - fpr
    finite = np.isfinite(thresholds)
    fpr = fpr[finite]
    tpr = tpr[finite]
    specificity = specificity[finite]
    thresholds = thresholds[finite]

    balanced = float(thresholds[np.argmax(tpr - fpr)])

    sens_mask = tpr >= 0.95
    if sens_mask.any():
        idx = np.where(sens_mask)[0][np.argmax(specificity[sens_mask])]
    else:
        idx = int(np.argmin(np.abs(tpr - 0.95)))
    screening = float(thresholds[idx])

    spec_mask = specificity >= 0.95
    if spec_mask.any():
        idx = np.where(spec_mask)[0][np.argmax(tpr[spec_mask])]
    else:
        idx = int(np.argmin(np.abs(specificity - 0.95)))
    confirmatory = float(thresholds[idx])

    return {
        "balanced": balanced,
        "screening_high_sensitivity": screening,
        "confirmatory_high_specificity": confirmatory,
    }


def _temperature_scale(probs: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    probs = _normalize_probabilities(probs)
    logits = np.log(probs)
    y = np.asarray(y).astype(int)

    def objective(log_temp: float) -> float:
        temp = float(np.exp(log_temp))
        pred = softmax(logits / temp, axis=1)
        return float(log_loss(y, pred, labels=list(range(probs.shape[1]))))

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(10.0)), method="bounded")
    temp = float(np.exp(result.x))
    return softmax(logits / temp, axis=1), temp


def _normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
    denom = probs.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return probs / denom


def _confidence_ece(y: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == y.astype(int)).astype(float)
    return _ece(correct, conf, n_bins=n_bins)


def _plot_reliability(y: np.ndarray, raw: np.ndarray, cal: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for values, label, color in [
        (raw, "Raw", "#777777"),
        (cal, "Calibrated", "#1f77b4"),
    ]:
        bins = np.linspace(0.0, 1.0, 11)
        xs = []
        ys = []
        ns = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (values >= lo) & (values <= hi if hi == 1.0 else values < hi)
            if not mask.any():
                continue
            xs.append(float(values[mask].mean()))
            ys.append(float(y[mask].mean()))
            ns.append(int(mask.sum()))
        ax.plot(xs, ys, marker="o", lw=1.2, color=color, label=label)
        for x, yy, n in zip(xs, ys, ns):
            ax.text(x, yy, str(n), fontsize=6, color=color, ha="center", va="bottom")
    ax.plot([0, 1], [0, 1], "--", color="#bbbbbb", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Binary cancer probability calibration")
    ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"binary_reliability_curve.{ext}", dpi=300)
    plt.close(fig)


def build(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    val = _load_predictions(run_dir, "val")
    test = _load_predictions(run_dir, "test")

    yv = val["y_bin"].astype(int)
    yt = test["y_bin"].astype(int)
    pv = val["s1_prob"].astype(float)
    pt = test["s1_prob"].astype(float)

    platt = _fit_platt(yv, pv)
    pv_cal = _apply_platt(platt, pv)
    pt_cal = _apply_platt(platt, pt)
    thresholds = _choose_thresholds(yv, pv_cal)

    binary_report = {
        "probability_interpretation": (
            "Calibrated P(training label=cancer | spectrum, preprocessing, cohort). "
            "This is not an absolute population disease probability."
        ),
        "calibration_method": "Platt logistic regression fitted on validation split logits",
        "validation_raw": _binary_metrics(yv, pv),
        "validation_calibrated": _binary_metrics(yv, pv_cal),
        "test_raw": _binary_metrics(yt, pt),
        "test_calibrated": _binary_metrics(yt, pt_cal),
        "operating_thresholds_from_validation": {
            name: {
                "threshold": float(thresh),
                "validation": _binary_metrics(yv, pv_cal, thresh),
                "test": _binary_metrics(yt, pt_cal, thresh),
            }
            for name, thresh in thresholds.items()
        },
        "platt_intercept": float(platt.intercept_[0]),
        "platt_logit_coefficient": float(platt.coef_[0, 0]),
    }

    cancer_val = yv == 1
    cancer_test = yt == 1
    s2_val_raw = _normalize_probabilities(val["s2_prob"][cancer_val].astype(float))
    s2_test_raw = _normalize_probabilities(test["s2_prob"][cancer_test].astype(float))
    y2_val = val["y_type"][cancer_val].astype(int)
    y2_test = test["y_type"][cancer_test].astype(int)
    s2_val_cal, temp = _temperature_scale(s2_val_raw, y2_val)
    logits_test = np.log(s2_test_raw)
    s2_test_cal = softmax(logits_test / temp, axis=1)
    multi_report = {
        "calibration_method": "Single-temperature scaling fitted on validation cancer cases",
        "temperature": float(temp),
        "validation_raw": {
            "log_loss": float(log_loss(y2_val, s2_val_raw, labels=list(range(len(CANCER_TYPES))))),
            "accuracy": float(accuracy_score(y2_val, s2_val_raw.argmax(axis=1))),
            "macro_f1": float(
                f1_score(y2_val, s2_val_raw.argmax(axis=1), average="macro", zero_division=0)
            ),
            "confidence_ece_10bin": _confidence_ece(y2_val, s2_val_raw),
        },
        "validation_calibrated": {
            "log_loss": float(log_loss(y2_val, s2_val_cal, labels=list(range(len(CANCER_TYPES))))),
            "accuracy": float(accuracy_score(y2_val, s2_val_cal.argmax(axis=1))),
            "macro_f1": float(
                f1_score(y2_val, s2_val_cal.argmax(axis=1), average="macro", zero_division=0)
            ),
            "confidence_ece_10bin": _confidence_ece(y2_val, s2_val_cal),
        },
        "test_raw": {
            "log_loss": float(
                log_loss(y2_test, s2_test_raw, labels=list(range(len(CANCER_TYPES))))
            ),
            "accuracy": float(accuracy_score(y2_test, s2_test_raw.argmax(axis=1))),
            "macro_f1": float(
                f1_score(y2_test, s2_test_raw.argmax(axis=1), average="macro", zero_division=0)
            ),
            "confidence_ece_10bin": _confidence_ece(y2_test, s2_test_raw),
        },
        "test_calibrated": {
            "log_loss": float(
                log_loss(y2_test, s2_test_cal, labels=list(range(len(CANCER_TYPES))))
            ),
            "accuracy": float(accuracy_score(y2_test, s2_test_cal.argmax(axis=1))),
            "macro_f1": float(
                f1_score(y2_test, s2_test_cal.argmax(axis=1), average="macro", zero_division=0)
            ),
            "confidence_ece_10bin": _confidence_ece(y2_test, s2_test_cal),
        },
    }

    rows = []
    s2_cal_iter = iter(s2_test_cal)
    for i in range(len(yt)):
        row = {
            "sample_id": str(test["sample_ids"][i]),
            "group": str(test["groups"][i]) if "groups" in test else "",
            "true_binary": int(yt[i]),
            "raw_cancer_probability": float(pt[i]),
            "calibrated_cancer_probability": float(pt_cal[i]),
        }
        for name, thresh in thresholds.items():
            row[f"class_at_{name}"] = "Cancer" if pt_cal[i] >= thresh else "Non-cancer"
        if yt[i] == 1:
            p2 = next(s2_cal_iter)
            for j, cancer in enumerate(CANCER_TYPES):
                row[f"calibrated_prob_{cancer}"] = float(p2[j])
            row["predicted_cancer_type"] = CANCER_TYPES[int(np.argmax(p2))]
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        out_dir / "calibrated_test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "run_dir": str(run_dir),
        "binary": binary_report,
        "multiclass_cancer_type": multi_report,
        "cancer_type_order": list(CANCER_TYPES),
    }
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {"section": "binary_test_raw", **binary_report["test_raw"]},
            {"section": "binary_test_calibrated", **binary_report["test_calibrated"]},
        ]
    ).to_csv(out_dir / "binary_calibration_metrics.csv", index=False, encoding="utf-8-sig")
    _plot_reliability(yt, pt, pt_cal, out_dir)

    print(f"Output: {out_dir}")
    print(json.dumps(binary_report["test_calibrated"], indent=2))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.run_dir.resolve(), args.out_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
