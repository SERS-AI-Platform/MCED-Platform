#!/usr/bin/env python3
"""Train an interpretable sparse model from registered SERS peak windows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training import train_usersnet as stk  # noqa: E402

CANCER_TYPES = stk.CANCER_TYPES
C_GRID = (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _fit_platt(y_val: np.ndarray, p_val: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(_logit(p_val).reshape(-1, 1), y_val.astype(int))
    return model


def _apply_platt(model: LogisticRegression, p: np.ndarray) -> np.ndarray:
    return model.predict_proba(_logit(p).reshape(-1, 1))[:, 1]


def _normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(probs, dtype=float), 1e-8, 1.0)
    denom = probs.sum(axis=1, keepdims=True)
    denom[denom <= 0] = 1.0
    return probs / denom


def _temperature_scale(probs: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    probs = _normalize_probabilities(probs)
    logits = np.log(probs)
    y = np.asarray(y).astype(int)

    def objective(log_temp: float) -> float:
        temp = float(np.exp(log_temp))
        pred = softmax(logits / temp, axis=1)
        return float(log_loss(y, pred, labels=list(range(probs.shape[1]))))

    result = minimize_scalar(objective, bounds=(np.log(0.05), np.log(10.0)), method="bounded")
    temp = float(np.exp(result.x))
    return temp, softmax(logits / temp, axis=1)


def _apply_temperature(probs: np.ndarray, temp: float) -> np.ndarray:
    probs = _normalize_probabilities(probs)
    return softmax(np.log(probs) / temp, axis=1)


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


def _binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    y = y.astype(int)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
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


def _load_training_data(cohort_csv: Path):
    stk.PROJECT_ROOT = PROJECT_ROOT
    stk.DATA_TYPE = "processed_csv"
    stk.DATA_DIR = cohort_csv
    stk.EXCLUDE_SOURCE_GROUPS = set()
    return stk.load_data(cohort_csv)


def _select_registry_peaks(registry_csv: Path, peak_filter: str) -> pd.DataFrame:
    registry = pd.read_csv(registry_csv)
    if peak_filter == "disease":
        registry = registry[registry["disease_discriminating_peak"].astype(bool)]
    elif peak_filter == "accepted":
        registry = registry[registry["accepted_peak"].astype(bool)]
    else:
        raise ValueError(f"Unknown peak filter: {peak_filter}")
    if registry.empty:
        raise ValueError(f"No registry peaks after filter={peak_filter}")
    return registry.reset_index(drop=True)


def _extract_peak_features(
    X_raw: np.ndarray,
    grid: np.ndarray,
    registry: pd.DataFrame,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    features = []
    names: list[str] = []
    feature_meta = []
    for _, row in registry.iterrows():
        peak_id = str(row["peak_id"])
        left = float(row["window_left_cm-1"])
        right = float(row["window_right_cm-1"])
        center = float(row["center_cm-1"])
        mask = (grid >= left) & (grid <= right)
        if not mask.any():
            idx = int(np.argmin(np.abs(grid - center)))
            mask[idx] = True
        Xw = X_raw[:, mask]
        gw = grid[mask]
        height = Xw.max(axis=1)
        area = trapezoid(Xw, gw, axis=1) if Xw.shape[1] > 1 else Xw[:, 0]
        contrast = Xw.max(axis=1) - Xw.min(axis=1)
        for kind, values in [
            ("height", height),
            ("area", area),
            ("contrast", contrast),
        ]:
            features.append(values)
            name = f"{peak_id}:{kind}"
            names.append(name)
            feature_meta.append(
                {
                    "feature": name,
                    "peak_id": peak_id,
                    "feature_kind": kind,
                    "display_group": row["display_group"],
                    "center_cm-1": center,
                    "window_left_cm-1": left,
                    "window_right_cm-1": right,
                    "bh_q_vs_control": row.get("bh_q_vs_control", np.nan),
                    "cohen_d_vs_control": row.get("cohen_d_vs_control", np.nan),
                }
            )
    return np.column_stack(features), names, pd.DataFrame(feature_meta)


def _make_binary_model(C: float, n_jobs: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=8000,
            n_jobs=n_jobs,
            random_state=42,
        ),
    )


def _make_multiclass_model(C: float, n_jobs: int) -> object:
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            penalty="l1",
            solver="saga",
            class_weight="balanced",
            max_iter=8000,
            multi_class="multinomial",
            n_jobs=n_jobs,
            random_state=42,
        ),
    )


def _coef_table(
    model, feature_names: list[str], feature_meta: pd.DataFrame, classes=None
) -> pd.DataFrame:
    lr = model.named_steps["logisticregression"]
    coef = lr.coef_
    rows = []
    if coef.shape[0] == 1:
        for feature, value in zip(feature_names, coef[0]):
            rows.append({"target": "Cancer", "feature": feature, "coefficient": float(value)})
    else:
        class_labels = list(classes if classes is not None else lr.classes_)
        for class_idx, row_coef in zip(class_labels, coef):
            target = (
                CANCER_TYPES[int(class_idx)]
                if int(class_idx) < len(CANCER_TYPES)
                else str(class_idx)
            )
            for feature, value in zip(feature_names, row_coef):
                rows.append({"target": target, "feature": feature, "coefficient": float(value)})
    out = pd.DataFrame(rows)
    out = out.merge(feature_meta, on="feature", how="left")
    out["abs_coefficient"] = out["coefficient"].abs()
    return out.sort_values(["target", "abs_coefficient"], ascending=[True, False])


def build(
    cohort_csv: Path,
    registry_csv: Path,
    split_npz: Path,
    out_dir: Path,
    *,
    peak_filter: str,
    n_jobs: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_3ch, df_meta, grid, groups, sample_ids, y_bin, y_type = _load_training_data(cohort_csv)
    split = np.load(split_npz)
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]
    registry = _select_registry_peaks(registry_csv, peak_filter)
    X_peak, feature_names, feature_meta = _extract_peak_features(X_3ch[:, 0, :], grid, registry)

    selection_rows = []
    best_binary = None
    best_binary_key = None
    for C in C_GRID:
        model = _make_binary_model(C, n_jobs)
        model.fit(X_peak[train_idx], y_bin[train_idx].astype(int))
        val_prob = model.predict_proba(X_peak[val_idx])[:, 1]
        auc = float(roc_auc_score(y_bin[val_idx].astype(int), val_prob))
        nonzero = int(np.count_nonzero(model.named_steps["logisticregression"].coef_))
        selection_rows.append(
            {"task": "binary", "C": C, "val_auc": auc, "nonzero_coefficients": nonzero}
        )
        key = (auc, -nonzero)
        if best_binary_key is None or key > best_binary_key:
            best_binary_key = key
            best_binary = model

    assert best_binary is not None
    val_raw = best_binary.predict_proba(X_peak[val_idx])[:, 1]
    test_raw = best_binary.predict_proba(X_peak[test_idx])[:, 1]
    platt = _fit_platt(y_bin[val_idx].astype(int), val_raw)
    val_cal = _apply_platt(platt, val_raw)
    test_cal = _apply_platt(platt, test_raw)
    thresholds = _choose_thresholds(y_bin[val_idx].astype(int), val_cal)

    cancer_train = train_idx[y_bin[train_idx] == 1]
    cancer_val = val_idx[y_bin[val_idx] == 1]
    cancer_test = test_idx[y_bin[test_idx] == 1]
    best_multi = None
    best_multi_key = None
    for C in C_GRID:
        model = _make_multiclass_model(C, n_jobs)
        model.fit(X_peak[cancer_train], y_type[cancer_train].astype(int))
        val_prob = _normalize_probabilities(model.predict_proba(X_peak[cancer_val]))
        pred = val_prob.argmax(axis=1)
        macro_f1 = float(
            f1_score(y_type[cancer_val].astype(int), pred, average="macro", zero_division=0)
        )
        nonzero = int(np.count_nonzero(model.named_steps["logisticregression"].coef_))
        selection_rows.append(
            {
                "task": "cancer_type",
                "C": C,
                "val_macro_f1": macro_f1,
                "nonzero_coefficients": nonzero,
            }
        )
        key = (macro_f1, -nonzero)
        if best_multi_key is None or key > best_multi_key:
            best_multi_key = key
            best_multi = model

    assert best_multi is not None
    multi_val_raw = _normalize_probabilities(best_multi.predict_proba(X_peak[cancer_val]))
    multi_test_raw = _normalize_probabilities(best_multi.predict_proba(X_peak[cancer_test]))
    temp, multi_val_cal = _temperature_scale(multi_val_raw, y_type[cancer_val].astype(int))
    multi_test_cal = _apply_temperature(multi_test_raw, temp)

    binary_report = {
        "selected_C": float(best_binary.named_steps["logisticregression"].C),
        "calibration": "Platt logistic regression on validation split",
        "platt_intercept": float(platt.intercept_[0]),
        "platt_logit_coefficient": float(platt.coef_[0, 0]),
        "thresholds_from_validation": {
            name: {
                "threshold": float(thresh),
                "validation": _binary_metrics(y_bin[val_idx].astype(int), val_cal, thresh),
                "test": _binary_metrics(y_bin[test_idx].astype(int), test_cal, thresh),
            }
            for name, thresh in thresholds.items()
        },
        "test_at_0_5": _binary_metrics(y_bin[test_idx].astype(int), test_cal, 0.5),
    }
    multi_report = {
        "selected_C": float(best_multi.named_steps["logisticregression"].C),
        "calibration": "single-temperature scaling on validation cancer cases",
        "temperature": float(temp),
        "validation_raw": {
            "log_loss": float(
                log_loss(
                    y_type[cancer_val].astype(int),
                    multi_val_raw,
                    labels=list(range(len(CANCER_TYPES))),
                )
            ),
            "accuracy": float(
                accuracy_score(y_type[cancer_val].astype(int), multi_val_raw.argmax(axis=1))
            ),
            "macro_f1": float(
                f1_score(
                    y_type[cancer_val].astype(int),
                    multi_val_raw.argmax(axis=1),
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "test_calibrated": {
            "log_loss": float(
                log_loss(
                    y_type[cancer_test].astype(int),
                    multi_test_cal,
                    labels=list(range(len(CANCER_TYPES))),
                )
            ),
            "accuracy": float(
                accuracy_score(y_type[cancer_test].astype(int), multi_test_cal.argmax(axis=1))
            ),
            "macro_f1": float(
                f1_score(
                    y_type[cancer_test].astype(int),
                    multi_test_cal.argmax(axis=1),
                    average="macro",
                    zero_division=0,
                )
            ),
        },
    }

    pd.DataFrame(selection_rows).to_csv(
        out_dir / "model_selection.csv", index=False, encoding="utf-8-sig"
    )
    _coef_table(best_binary, feature_names, feature_meta).to_csv(
        out_dir / "binary_peak_coefficients.csv", index=False, encoding="utf-8-sig"
    )
    _coef_table(
        best_multi,
        feature_names,
        feature_meta,
        classes=best_multi.named_steps["logisticregression"].classes_,
    ).to_csv(out_dir / "cancer_type_peak_coefficients.csv", index=False, encoding="utf-8-sig")

    pred_rows = []
    cancer_probs_iter = iter(multi_test_cal)
    for idx, raw_p, cal_p in zip(test_idx, test_raw, test_cal):
        row = {
            "sample_id": str(sample_ids[idx]),
            "group": str(groups[idx]),
            "true_binary": int(y_bin[idx]),
            "raw_cancer_probability": float(raw_p),
            "calibrated_cancer_probability": float(cal_p),
            "class_balanced_threshold": "Cancer"
            if cal_p >= thresholds["balanced"]
            else "Non-cancer",
        }
        if y_bin[idx] == 1:
            p2 = next(cancer_probs_iter)
            for j, cancer in enumerate(CANCER_TYPES):
                row[f"calibrated_prob_{cancer}"] = float(p2[j])
            row["predicted_cancer_type"] = CANCER_TYPES[int(np.argmax(p2))]
        pred_rows.append(row)
    pd.DataFrame(pred_rows).to_csv(
        out_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "model": "sparse_peak_evidence_model_v1",
        "cohort_csv": str(cohort_csv),
        "registry_csv": str(registry_csv),
        "split_npz": str(split_npz),
        "peak_filter": peak_filter,
        "n_registry_peaks": int(len(registry)),
        "n_features": int(X_peak.shape[1]),
        "feature_definition": "height, area, and contrast for each accepted data-driven peak window",
        "probability_interpretation": (
            "Calibrated P(training label | peak-window features, cohort distribution); "
            "not an absolute population disease probability."
        ),
        "binary": binary_report,
        "cancer_type": multi_report,
        "cancer_type_order": list(CANCER_TYPES),
    }
    (out_dir / "peak_evidence_model_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Output: {out_dir}")
    print(
        json.dumps(
            {
                "binary_test_at_0_5": binary_report["test_at_0_5"],
                "cancer_type_test": multi_report["test_calibrated"],
            },
            indent=2,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-csv", type=Path, required=True)
    parser.add_argument("--registry-csv", type=Path, required=True)
    parser.add_argument("--split-npz", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--peak-filter", choices=["accepted", "disease"], default="accepted")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    build(
        args.cohort_csv.resolve(),
        args.registry_csv.resolve(),
        args.split_npz.resolve(),
        args.out_dir.resolve(),
        peak_filter=args.peak_filter,
        n_jobs=args.n_jobs,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
