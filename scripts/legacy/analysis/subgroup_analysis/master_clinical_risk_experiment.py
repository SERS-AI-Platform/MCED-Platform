#!/usr/bin/env python3
"""Risk-score experiments anchored on data/clinical_data/master_clinical.csv.

The main cohort is the SERS-linked subset of master_clinical.csv
(`in_spectra == True`). Exclusion and review categories are kept as cohort
flags and are not used as model features.

Outputs are written under results/clinical_master_risk by default:

* cohort_summary.csv
* marker_qc_by_group.csv
* feature_set_availability.csv
* global_model_metrics.csv
* cancer_specific_metrics.csv
* lo_source_metrics.csv
* calibration_curves.csv
* *_oof_predictions.csv
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MASTER = PROJECT_ROOT / "data/clinical_data/master_clinical.csv"
DEFAULT_UNIFIED = PROJECT_ROOT / "data/clinical_data/clinical_unified_202604172056.csv"
DEFAULT_SPECTRA = PROJECT_ROOT / "results/processed_spectra.csv"
DEFAULT_SQL_SCHEMA = PROJECT_ROOT / "scripts/db/clinical_unified/01_schema.sql"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results/clinical_master_risk"

GROUP_NORMALIZATION = {
    "BLC": "BLA",
    "COL": "CRC",
    "CPAN": "PAN",
}

CANCER_GROUPS = ("PRO", "PAN", "LUN", "OVA", "BRE", "CRC", "BLA")
NON_CANCER_GROUPS = ("NOR", "YNOR", "DIA", "HBP", "H.D.")
MODEL_COHORTS = ("full_1628", "clean", "strict_screening")

DEMOGRAPHIC_NUMERIC = ("age", "bmi")
DEMOGRAPHIC_CATEGORICAL = ("sex",)
SOURCE_CATEGORICAL = ("source_file", "hospital_code")
GLOBAL_MARKERS = ("afp", "cea", "ca19_9", "psa", "ca125", "free_psa")
COMMON_LAB_PANEL = (
    "wbc",
    "rbc",
    "hb",
    "hct",
    "platelet",
    "neutrophil_pct",
    "lymphocyte_pct",
    "ast",
    "alt",
    "alp",
    "ggt",
    "bun",
    "creatinine",
    "uric_acid",
    "glucose",
    "total_protein",
    "albumin",
    "total_bilirubin",
    "ldh",
    "calcium",
    "hs_crp",
    "sodium",
    "potassium",
    "chloride",
    "total_cholesterol",
    "triglyceride",
    "hdl_c",
    "ldl_c",
    "hba1c",
    "ua_sg",
    "ua_ph",
)


@dataclass(frozen=True)
class FeatureSet:
    name: str
    spectral: bool = False
    demographics: bool = False
    source: bool = False
    markers: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskSpec:
    task: str
    cancer_groups: tuple[str, ...]
    control_groups: tuple[str, ...]
    marker_candidates: tuple[str, ...]
    notes: str = ""


CANCER_TASKS = (
    TaskSpec(
        "PRO_vs_NOR",
        ("PRO",),
        ("NOR",),
        ("psa", "free_psa"),
        "Primary prostate screen: NOR vs PRO.",
    ),
    TaskSpec(
        "PRO_vs_controls",
        ("PRO",),
        ("NOR", "DIA", "HBP", "H.D."),
        ("psa", "free_psa"),
        "Secondary prostate screen: comorbidity controls included.",
    ),
    TaskSpec(
        "PAN_vs_NOR",
        ("PAN",),
        ("NOR",),
        ("ca19_9", "cea"),
        "PAN/CPAN unified; NOR controls.",
    ),
    TaskSpec(
        "PAN_vs_YNOR",
        ("PAN",),
        ("YNOR",),
        ("ca19_9", "cea"),
        "PAN/CPAN unified; YNOR controls.",
    ),
    TaskSpec("LUN_vs_NOR", ("LUN",), ("NOR",), ("cea",), "Lung cancer screen."),
    TaskSpec(
        "OVA_vs_NOR",
        ("OVA",),
        ("NOR",),
        ("cea", "ca19_9", "ca125"),
        "CA125 is available only if verified in source data.",
    ),
    TaskSpec("BRE_vs_NOR", ("BRE",), ("NOR",), ("cea",), "Exploratory small-n breast screen."),
    TaskSpec("CRC_vs_NOR", ("CRC",), ("NOR",), (), "CRC stage/TNM subgroup interpretation."),
    TaskSpec(
        "BLA_vs_NOR",
        ("BLA",),
        ("NOR",),
        COMMON_LAB_PANEL,
        "Bladder screen using CBC/chemistry/urine lab panel when available.",
    ),
)


def normalize_group(value: object) -> str:
    if pd.isna(value):
        return ""
    group = str(value).strip()
    return GROUP_NORMALIZATION.get(group, group)


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def read_csv_robust(path: Path, **kwargs) -> pd.DataFrame:
    encodings = ("utf-8-sig", "utf-8", "cp949")
    last_error: Exception | None = None
    for encoding in encodings:
        try:
            return pd.read_csv(path, encoding=encoding, low_memory=False, **kwargs)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(path, low_memory=False, **kwargs)


def parse_source_hospital_map(sql_path: Path) -> dict[str, str]:
    if not sql_path.exists():
        return {}
    text = sql_path.read_text(encoding="utf-8")
    pattern = re.compile(
        r"\(\s*'(?P<source>[^']+)'\s*,\s*'(?P<group>[^']+)'\s*,\s*"
        r"'(?P<protocol>[^']+)'\s*,\s*'(?P<hospital>[^']+)'\s*\)"
    )
    return {m.group("source"): m.group("hospital") for m in pattern.finditer(text)}


def load_master(master_path: Path, sql_schema_path: Path) -> pd.DataFrame:
    master = read_csv_robust(master_path)
    required = {
        "patient_id",
        "disease_group",
        "source_file",
        "spectral_group",
        "spectral_sample_id",
        "in_spectra",
        "exclusion_category",
    }
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"master_clinical is missing required columns: {missing}")

    master["in_spectra"] = parse_bool_series(master["in_spectra"])
    master = master[master["in_spectra"]].copy()
    master["disease_group_norm"] = master["disease_group"].map(normalize_group)
    master["spectral_group_key"] = master["spectral_group"].astype(str).str.strip()
    master["spectral_sample_id"] = pd.to_numeric(master["spectral_sample_id"], errors="coerce")
    master = master.dropna(subset=["spectral_group_key", "spectral_sample_id"]).copy()
    master["spectral_sample_id"] = master["spectral_sample_id"].astype(int)
    master["patient_spectral_key"] = (
        master["spectral_group_key"] + ":" + master["spectral_sample_id"].astype(str)
    )

    category = master["exclusion_category"].fillna("").astype(str).str.strip()
    master["cohort_flag_definite_exclude"] = category.eq("definite_exclude")
    master["cohort_flag_review_periop"] = category.eq("review_periop")
    master["cohort_flag_review_undefined"] = category.eq("review_undefined")
    master["cohort_flag_review_any"] = category.str.startswith("review_")
    master["exclusion_category_filled"] = category.mask(category.eq(""), "(usable)")

    source_map = parse_source_hospital_map(sql_schema_path)
    master["hospital_code"] = master["source_file"].map(source_map).fillna("UNMAPPED")
    master["is_cancer"] = master["disease_group_norm"].isin(CANCER_GROUPS).astype(int)
    master["is_analysis_group"] = master["disease_group_norm"].isin(
        set(CANCER_GROUPS) | set(NON_CANCER_GROUPS)
    )
    return master


def load_patient_spectra(spectra_path: Path) -> tuple[pd.DataFrame, list[str]]:
    spectra = read_csv_robust(spectra_path)
    feature_cols = sorted(
        [col for col in spectra.columns if col.startswith("x_")],
        key=lambda c: float(c.replace("x_", "")),
    )
    if not feature_cols:
        raise ValueError(f"No spectral feature columns found in {spectra_path}")
    spectra["group"] = spectra["group"].astype(str).str.strip()
    spectra["sample_id"] = pd.to_numeric(spectra["sample_id"], errors="coerce")
    spectra = spectra.dropna(subset=["group", "sample_id"]).copy()
    spectra["sample_id"] = spectra["sample_id"].astype(int)
    agg = spectra.groupby(["group", "sample_id"], as_index=False)[feature_cols].mean()
    agg["patient_spectral_key"] = agg["group"] + ":" + agg["sample_id"].astype(str)
    return agg, feature_cols


def merge_master_spectra(master: pd.DataFrame, spectra: pd.DataFrame) -> pd.DataFrame:
    duplicate_keys = master["patient_spectral_key"].duplicated(keep=False)
    if duplicate_keys.any():
        dup = master.loc[duplicate_keys, ["patient_id", "patient_spectral_key"]]
        raise ValueError(f"Duplicate master spectral keys found:\n{dup.head(20)}")

    merged = master.merge(
        spectra.drop(columns=["group", "sample_id"]),
        on="patient_spectral_key",
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(master):
        missing = sorted(set(master["patient_spectral_key"]) - set(merged["patient_spectral_key"]))
        raise ValueError(
            f"Only merged {len(merged)}/{len(master)} in_spectra rows. "
            f"First missing keys: {missing[:20]}"
        )
    return merged


def build_cohorts(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    cohorts = {
        "full_1628": df.copy(),
        "review_included": df.copy(),
        "clean": df[~df["cohort_flag_definite_exclude"]].copy(),
        "strict_screening": df[
            ~(
                df["cohort_flag_definite_exclude"]
                | df["cohort_flag_review_periop"]
                | df["cohort_flag_review_undefined"]
            )
        ].copy(),
    }
    return cohorts


def write_cohort_summary(cohorts: dict[str, pd.DataFrame], out_dir: Path) -> pd.DataFrame:
    rows = []
    for cohort_name, cohort in cohorts.items():
        rows.append(
            {
                "cohort": cohort_name,
                "disease_group": "__overall__",
                "n": len(cohort),
                "n_cancer": int(cohort["is_cancer"].sum()),
                "n_noncancer": int((cohort["is_cancer"] == 0).sum()),
                "definite_exclude": int(cohort["cohort_flag_definite_exclude"].sum()),
                "review_periop": int(cohort["cohort_flag_review_periop"].sum()),
                "review_undefined": int(cohort["cohort_flag_review_undefined"].sum()),
            }
        )
        for group, sub in cohort.groupby("disease_group_norm", dropna=False):
            rows.append(
                {
                    "cohort": cohort_name,
                    "disease_group": group,
                    "n": len(sub),
                    "n_cancer": int(sub["is_cancer"].sum()),
                    "n_noncancer": int((sub["is_cancer"] == 0).sum()),
                    "definite_exclude": int(sub["cohort_flag_definite_exclude"].sum()),
                    "review_periop": int(sub["cohort_flag_review_periop"].sum()),
                    "review_undefined": int(sub["cohort_flag_review_undefined"].sum()),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "cohort_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def load_unified_for_marker_qc(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    unified = read_csv_robust(path)
    if "disease_group" in unified.columns:
        unified["disease_group_norm"] = unified["disease_group"].map(normalize_group)
    return unified


def marker_qc(master: pd.DataFrame, unified: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    marker_sources = {
        "afp": ("afp", "blood_afp"),
        "cea": ("cea", "blood_cea"),
        "ca19_9": ("ca19_9", "blood_ca199"),
        "ca125": ("ca125", "blood_ca125"),
        "psa": ("psa", "blood_psa"),
        "free_psa": ("free_psa", "blood_free_psa"),
    }
    groups = sorted(set(master["disease_group_norm"].dropna()) | set(CANCER_GROUPS) | set(NON_CANCER_GROUPS))
    rows = []
    for marker, (master_col, unified_col) in marker_sources.items():
        for group in groups:
            m_sub = master[master["disease_group_norm"] == group]
            u_sub = (
                unified[unified["disease_group_norm"] == group]
                if "disease_group_norm" in unified.columns
                else pd.DataFrame()
            )
            rows.append(
                {
                    "marker": marker,
                    "disease_group": group,
                    "master_column": master_col if master_col in master.columns else "",
                    "unified_column": unified_col if unified_col in unified.columns else "",
                    "master_n": len(m_sub),
                    "master_nonmissing": (
                        int(pd.to_numeric(m_sub[master_col], errors="coerce").notna().sum())
                        if master_col in m_sub.columns
                        else 0
                    ),
                    "unified_n": len(u_sub),
                    "unified_nonmissing": (
                        int(pd.to_numeric(u_sub[unified_col], errors="coerce").notna().sum())
                        if unified_col in u_sub.columns
                        else 0
                    ),
                }
            )
    qc = pd.DataFrame(rows)
    qc["master_available_pct"] = np.where(qc["master_n"] > 0, qc["master_nonmissing"] / qc["master_n"], np.nan)
    qc["unified_available_pct"] = np.where(
        qc["unified_n"] > 0, qc["unified_nonmissing"] / qc["unified_n"], np.nan
    )
    qc.to_csv(out_dir / "marker_qc_by_group.csv", index=False, encoding="utf-8-sig")
    return qc


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def usable_marker_columns(
    df: pd.DataFrame,
    y: np.ndarray,
    candidates: Iterable[str],
    min_per_class: int,
) -> tuple[tuple[str, ...], str]:
    usable = []
    skipped = []
    for col in candidates:
        if col not in df.columns:
            skipped.append(f"{col}:missing_column")
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        pos = int(values[y == 1].notna().sum())
        neg = int(values[y == 0].notna().sum())
        if pos >= min_per_class and neg >= min_per_class:
            usable.append(col)
        else:
            skipped.append(f"{col}:insufficient_nonmissing(pos={pos},neg={neg})")
    reason = ";".join(skipped)
    return tuple(usable), reason


def make_feature_sets(marker_cols: tuple[str, ...]) -> list[FeatureSet]:
    sets = [
        FeatureSet("sers_only", spectral=True),
        FeatureSet("demographics_only", demographics=True),
        FeatureSet("source_region_only", source=True),
        FeatureSet("sers_demographics", spectral=True, demographics=True),
    ]
    if marker_cols:
        sets.insert(3, FeatureSet("tumor_marker_only", markers=marker_cols))
        sets.append(FeatureSet("sers_demographics_markers", spectral=True, demographics=True, markers=marker_cols))
    return sets


def feature_columns_for_set(
    feature_set: FeatureSet,
    spectral_cols: list[str],
) -> tuple[list[str], list[str]]:
    numeric: list[str] = []
    categorical: list[str] = []
    if feature_set.spectral:
        numeric.extend(spectral_cols)
    if feature_set.demographics:
        numeric.extend(DEMOGRAPHIC_NUMERIC)
        categorical.extend(DEMOGRAPHIC_CATEGORICAL)
    if feature_set.source:
        categorical.extend(SOURCE_CATEGORICAL)
    if feature_set.markers:
        numeric.extend(feature_set.markers)
    numeric = list(dict.fromkeys(numeric))
    categorical = list(dict.fromkeys(categorical))
    return numeric, categorical


def make_pipeline_for_features(
    numeric_cols: list[str],
    categorical_cols: list[str],
    random_state: int,
    calibration_cv: int = 0,
    y_train: np.ndarray | None = None,
) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("No features selected")
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=2000,
        random_state=random_state,
        solver="liblinear",
    )
    base = Pipeline([("preprocessor", preprocessor), ("model", model)])
    if calibration_cv <= 1 or y_train is None:
        return base
    class_counts = np.bincount(y_train.astype(int), minlength=2)
    inner_cv = int(min(calibration_cv, class_counts.min()))
    if inner_cv < 2:
        return base
    return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=inner_cv)


def choose_cv_splits(y: np.ndarray, requested: int) -> int:
    counts = np.bincount(y.astype(int), minlength=2)
    return int(min(requested, counts.min()))


def run_oof_predictions(
    df: pd.DataFrame,
    y: np.ndarray,
    feature_set: FeatureSet,
    spectral_cols: list[str],
    n_splits: int,
    random_state: int,
    calibration_cv: int,
) -> tuple[np.ndarray, np.ndarray]:
    splits = choose_cv_splits(y, n_splits)
    if splits < 2:
        raise ValueError(f"Need at least two samples in each class; class counts={np.bincount(y)}")
    numeric_cols, categorical_cols = feature_columns_for_set(feature_set, spectral_cols)
    missing_cols = sorted((set(numeric_cols) | set(categorical_cols)) - set(df.columns))
    if missing_cols:
        raise ValueError(f"Feature columns are missing from dataframe: {missing_cols}")

    scores = np.full(len(df), np.nan)
    folds = np.full(len(df), -1, dtype=int)
    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=random_state)
    groups = df["patient_spectral_key"].astype(str).to_numpy()
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(df, y, groups=groups)):
        model = make_pipeline_for_features(
            numeric_cols,
            categorical_cols,
            random_state + fold_idx,
            calibration_cv=calibration_cv,
            y_train=y[train_idx],
        )
        model.fit(df.iloc[train_idx], y[train_idx])
        scores[test_idx] = model.predict_proba(df.iloc[test_idx])[:, 1]
        folds[test_idx] = fold_idx
    return scores, folds


def fixed_sensitivity_specificity(y: np.ndarray, scores: np.ndarray, target_sensitivity: float) -> tuple[float, float, float]:
    fpr, tpr, thresholds = roc_curve(y, scores)
    eligible = np.where(tpr >= target_sensitivity)[0]
    if len(eligible) == 0:
        return np.nan, np.nan, np.nan
    specificities = 1.0 - fpr[eligible]
    best_local = int(np.argmax(specificities))
    idx = int(eligible[best_local])
    return float(specificities[best_local]), float(thresholds[idx]), float(tpr[idx])


def compute_binary_metrics(y: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = y.astype(int)
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    spec95, th95, sens95 = fixed_sensitivity_specificity(y, scores, 0.95)
    spec98, th98, sens98 = fixed_sensitivity_specificity(y, scores, 0.98)
    return {
        "auroc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "auprc": float(average_precision_score(y, scores)) if len(np.unique(y)) == 2 else np.nan,
        "brier": float(brier_score_loss(y, scores)),
        "threshold": threshold,
        "sensitivity": float(sens),
        "specificity": float(spec),
        "ppv": float(ppv),
        "npv": float(npv),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "specificity_at_sensitivity_95": spec95,
        "threshold_at_sensitivity_95": th95,
        "achieved_sensitivity_95": sens95,
        "specificity_at_sensitivity_98": spec98,
        "threshold_at_sensitivity_98": th98,
        "achieved_sensitivity_98": sens98,
    }


def calibration_rows(
    y: np.ndarray,
    scores: np.ndarray,
    cohort: str,
    task: str,
    feature_set: str,
    n_bins: int = 10,
) -> list[dict[str, object]]:
    rows = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for idx in range(n_bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == 0:
            mask = (scores >= left) & (scores <= right)
        else:
            mask = (scores > left) & (scores <= right)
        if not np.any(mask):
            continue
        y_bin = y[mask]
        score_bin = scores[mask]
        rows.append(
            {
                "cohort": cohort,
                "task": task,
                "feature_set": feature_set,
                "bin_index": idx,
                "bin_left": float(left),
                "bin_right": float(right),
                "bin_n": int(mask.sum()),
                "mean_predicted_risk": float(np.mean(score_bin)),
                "observed_cancer_rate": float(np.mean(y_bin)),
            }
        )
    return rows


def subgroup_rows(
    pred_df: pd.DataFrame,
    cohort: str,
    task: str,
    feature_set: str,
    subgroup_cols: Iterable[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for col in subgroup_cols:
        if col not in pred_df.columns:
            continue
        for value, sub in pred_df.groupby(col, dropna=False):
            y = sub["label"].to_numpy(dtype=int)
            scores = sub["risk_score"].to_numpy(dtype=float)
            row = {
                "cohort": cohort,
                "task": task,
                "feature_set": feature_set,
                "subgroup_column": col,
                "subgroup_value": value,
                "n": len(sub),
                "n_cancer": int(y.sum()),
                "n_noncancer": int((y == 0).sum()),
                "mean_risk": float(np.nanmean(scores)),
            }
            if len(np.unique(y)) == 2 and len(sub) >= 4:
                row.update(compute_binary_metrics(y, scores))
            elif y.sum() == len(y):
                row["sensitivity"] = float((scores >= 0.5).mean())
            elif y.sum() == 0:
                row["specificity"] = float((scores < 0.5).mean())
            rows.append(row)
    return rows


def task_base_prediction_frame(
    df: pd.DataFrame,
    y: np.ndarray,
    scores: np.ndarray,
    folds: np.ndarray,
    cohort: str,
    task: str,
    feature_set: str,
) -> pd.DataFrame:
    keep_cols = [
        "patient_id",
        "patient_spectral_key",
        "disease_group",
        "disease_group_norm",
        "source_file",
        "hospital_code",
        "exclusion_category_filled",
        "cohort_flag_definite_exclude",
        "cohort_flag_review_periop",
        "cohort_flag_review_undefined",
        "cancer_stage_group",
        "stage",
        "t_stage",
        "n_stage",
        "m_stage",
    ]
    available = [col for col in keep_cols if col in df.columns]
    pred = df[available].copy()
    pred.insert(0, "cohort", cohort)
    pred.insert(1, "task", task)
    pred.insert(2, "feature_set", feature_set)
    pred["label"] = y.astype(int)
    pred["risk_score"] = scores
    pred["fold"] = folds
    return pred


def run_task_feature_sets(
    task_df: pd.DataFrame,
    y: np.ndarray,
    spectral_cols: list[str],
    cohort_name: str,
    task_name: str,
    cohort_denominator: int,
    marker_candidates: tuple[str, ...],
    n_splits: int,
    random_state: int,
    min_marker_per_class: int,
    calibration_cv: int,
) -> tuple[list[dict[str, object]], list[pd.DataFrame], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    marker_cols, marker_skip_reason = usable_marker_columns(
        task_df, y, marker_candidates, min_marker_per_class
    )
    feature_sets = make_feature_sets(marker_cols)
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    calibration: list[dict[str, object]] = []
    subgroup: list[dict[str, object]] = []
    feature_availability: list[dict[str, object]] = []

    for fs in feature_sets:
        numeric_cols, categorical_cols = feature_columns_for_set(fs, spectral_cols)
        feature_availability.append(
            {
                "cohort": cohort_name,
                "task": task_name,
                "feature_set": fs.name,
                "status": "ok",
                "marker_candidates": ",".join(marker_candidates),
                "marker_columns_used": ",".join(fs.markers),
                "marker_skip_reason": marker_skip_reason if marker_candidates else "",
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(categorical_cols),
                "n": len(task_df),
                "n_cancer": int(y.sum()),
                "n_noncancer": int((y == 0).sum()),
            }
        )
        scores, folds = run_oof_predictions(
            task_df,
            y,
            fs,
            spectral_cols,
            n_splits=n_splits,
            random_state=random_state,
            calibration_cv=calibration_cv,
        )
        metrics = compute_binary_metrics(y, scores)
        metrics.update(
            {
                "cohort": cohort_name,
                "task": task_name,
                "feature_set": fs.name,
                "status": "ok",
                "cohort_denominator": cohort_denominator,
                "n": len(task_df),
                "n_cancer": int(y.sum()),
                "n_noncancer": int((y == 0).sum()),
                "marker_columns_used": ",".join(fs.markers),
                "calibration_method": "sigmoid" if calibration_cv > 1 else "none",
                "calibration_cv": calibration_cv if calibration_cv > 1 else 0,
            }
        )
        metric_rows.append(metrics)
        pred = task_base_prediction_frame(task_df, y, scores, folds, cohort_name, task_name, fs.name)
        prediction_frames.append(pred)
        calibration.extend(calibration_rows(y, scores, cohort_name, task_name, fs.name))
        subgroup.extend(
            subgroup_rows(
                pred,
                cohort_name,
                task_name,
                fs.name,
                (
                    "disease_group_norm",
                    "source_file",
                    "hospital_code",
                    "exclusion_category_filled",
                    "cancer_stage_group",
                ),
            )
        )
    return metric_rows, prediction_frames, calibration, subgroup, feature_availability


def run_global_experiments(
    cohorts: dict[str, pd.DataFrame],
    spectral_cols: list[str],
    n_splits: int,
    random_state: int,
    min_marker_per_class: int,
    calibration_cv: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    calibration: list[dict[str, object]] = []
    subgroup: list[dict[str, object]] = []
    availability: list[dict[str, object]] = []

    for cohort_name in MODEL_COHORTS:
        cohort = cohorts[cohort_name]
        task_df = cohort[cohort["is_analysis_group"]].copy()
        y = task_df["is_cancer"].to_numpy(dtype=int)
        rows, preds, cal, sub, avail = run_task_feature_sets(
            task_df,
            y,
            spectral_cols,
            cohort_name,
            "global_cancer_vs_noncancer",
            len(cohort),
            GLOBAL_MARKERS,
            n_splits,
            random_state,
            min_marker_per_class,
            calibration_cv,
        )
        metrics.extend(rows)
        predictions.extend(preds)
        calibration.extend(cal)
        subgroup.extend(sub)
        availability.extend(avail)

    return (
        pd.DataFrame(metrics),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(calibration),
        pd.DataFrame(subgroup),
        pd.DataFrame(availability),
    )


def insufficient_task_row(
    cohort_name: str,
    task: str,
    cohort_denominator: int,
    n: int,
    n_cancer: int,
    n_noncancer: int,
    reason: str,
) -> dict[str, object]:
    return {
        "cohort": cohort_name,
        "task": task,
        "feature_set": "",
        "status": "skipped",
        "skip_reason": reason,
        "cohort_denominator": cohort_denominator,
        "n": n,
        "n_cancer": n_cancer,
        "n_noncancer": n_noncancer,
    }


def run_cancer_specific_experiments(
    cohorts: dict[str, pd.DataFrame],
    spectral_cols: list[str],
    n_splits: int,
    random_state: int,
    min_marker_per_class: int,
    calibration_cv: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    calibration: list[dict[str, object]] = []
    subgroup: list[dict[str, object]] = []
    availability: list[dict[str, object]] = []

    for cohort_name in MODEL_COHORTS:
        cohort = cohorts[cohort_name]
        for task in CANCER_TASKS:
            wanted = set(task.cancer_groups) | set(task.control_groups)
            task_df = cohort[cohort["disease_group_norm"].isin(wanted)].copy()
            task_df["task_label"] = task_df["disease_group_norm"].isin(task.cancer_groups).astype(int)
            y = task_df["task_label"].to_numpy(dtype=int)
            n_cancer = int(y.sum())
            n_noncancer = int((y == 0).sum())
            splits = choose_cv_splits(y, n_splits) if len(y) else 0
            if len(y) == 0 or n_cancer < 2 or n_noncancer < 2 or splits < 2:
                metrics.append(
                    insufficient_task_row(
                        cohort_name,
                        task.task,
                        len(cohort),
                        len(task_df),
                        n_cancer,
                        n_noncancer,
                        "insufficient_class_counts",
                    )
                )
                continue
            rows, preds, cal, sub, avail = run_task_feature_sets(
                task_df,
                y,
                spectral_cols,
                cohort_name,
                task.task,
                len(cohort),
                task.marker_candidates,
                n_splits,
                random_state,
                min_marker_per_class,
                calibration_cv,
            )
            for row in rows:
                row["task_notes"] = task.notes
            metrics.extend(rows)
            predictions.extend(preds)
            calibration.extend(cal)
            subgroup.extend(sub)
            availability.extend(avail)

    return (
        pd.DataFrame(metrics),
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame(),
        pd.DataFrame(calibration),
        pd.DataFrame(subgroup),
        pd.DataFrame(availability),
    )


def run_leave_one_source(
    cohorts: dict[str, pd.DataFrame],
    spectral_cols: list[str],
    random_state: int,
    min_marker_per_class: int,
    calibration_cv: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cohort_name in MODEL_COHORTS:
        cohort = cohorts[cohort_name]
        task_df = cohort[cohort["is_analysis_group"]].copy()
        y_all = task_df["is_cancer"].to_numpy(dtype=int)
        marker_cols, _ = usable_marker_columns(task_df, y_all, GLOBAL_MARKERS, min_marker_per_class)
        feature_sets = (
            FeatureSet("sers_only", spectral=True),
            FeatureSet("demographics_only", demographics=True),
            FeatureSet("source_region_only", source=True),
            FeatureSet("sers_demographics", spectral=True, demographics=True),
            FeatureSet("sers_demographics_markers", spectral=True, demographics=True, markers=marker_cols),
        )
        if not marker_cols:
            feature_sets = tuple(fs for fs in feature_sets if fs.name != "sers_demographics_markers")
        for source_file, holdout in task_df.groupby("source_file", dropna=False):
            test_idx = holdout.index
            train = task_df.drop(index=test_idx)
            test = task_df.loc[test_idx]
            y_train = train["is_cancer"].to_numpy(dtype=int)
            y_test = test["is_cancer"].to_numpy(dtype=int)
            if len(np.unique(y_train)) < 2:
                for fs in feature_sets:
                    rows.append(
                        {
                            "cohort": cohort_name,
                            "task": "global_cancer_vs_noncancer",
                            "feature_set": fs.name,
                            "status": "skipped",
                            "skip_reason": "training_set_missing_class",
                            "holdout_source_file": source_file,
                            "cohort_denominator": len(cohort),
                            "n_train": len(train),
                            "n_test": len(test),
                            "n_test_cancer": int(y_test.sum()),
                            "n_test_noncancer": int((y_test == 0).sum()),
                        }
                    )
                continue
            for fs in feature_sets:
                numeric_cols, categorical_cols = feature_columns_for_set(fs, spectral_cols)
                model = make_pipeline_for_features(
                    numeric_cols,
                    categorical_cols,
                    random_state,
                    calibration_cv=calibration_cv,
                    y_train=y_train,
                )
                model.fit(train, y_train)
                scores = model.predict_proba(test)[:, 1]
                row: dict[str, object] = {
                    "cohort": cohort_name,
                    "task": "global_cancer_vs_noncancer",
                    "feature_set": fs.name,
                    "status": "ok",
                    "holdout_source_file": source_file,
                    "cohort_denominator": len(cohort),
                    "n_train": len(train),
                    "n_test": len(test),
                    "n_test_cancer": int(y_test.sum()),
                    "n_test_noncancer": int((y_test == 0).sum()),
                    "mean_risk": float(np.mean(scores)),
                    "calibration_method": "sigmoid" if calibration_cv > 1 else "none",
                    "calibration_cv": calibration_cv if calibration_cv > 1 else 0,
                }
                if len(np.unique(y_test)) == 2:
                    row.update(compute_binary_metrics(y_test, scores))
                elif y_test.sum() == len(y_test):
                    row["sensitivity"] = float((scores >= 0.5).mean())
                elif y_test.sum() == 0:
                    row["specificity"] = float((scores < 0.5).mean())
                rows.append(row)
    return pd.DataFrame(rows)


def write_frame(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--clinical-unified", type=Path, default=DEFAULT_UNIFIED)
    parser.add_argument("--spectra", type=Path, default=DEFAULT_SPECTRA)
    parser.add_argument("--sql-schema", type=Path, default=DEFAULT_SQL_SCHEMA)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mode", choices=("audit", "global", "all"), default="all")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--min-marker-per-class", type=int, default=5)
    parser.add_argument(
        "--calibration-cv",
        type=int,
        default=3,
        help="Inner CV folds for sigmoid probability calibration. Use 0 or 1 to disable.",
    )
    parser.add_argument("--skip-lo-source", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    master = load_master(args.master, args.sql_schema)
    spectra, spectral_cols = load_patient_spectra(args.spectra)
    merged = merge_master_spectra(master, spectra)
    cohorts = build_cohorts(merged)
    unified = load_unified_for_marker_qc(args.clinical_unified)

    cohort_summary = write_cohort_summary(cohorts, args.out_dir)
    marker_qc(merged, unified, args.out_dir)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "master": str(args.master),
        "spectra": str(args.spectra),
        "clinical_unified": str(args.clinical_unified),
        "sql_schema": str(args.sql_schema),
        "n_spectral_features": len(spectral_cols),
        "n_in_spectra_merged": len(merged),
        "calibration_method": "sigmoid" if args.calibration_cv > 1 else "none",
        "calibration_cv": args.calibration_cv if args.calibration_cv > 1 else 0,
        "cohorts": {
            name: {
                "n": int(len(df)),
                "n_cancer": int(df["is_cancer"].sum()),
                "n_noncancer": int((df["is_cancer"] == 0).sum()),
            }
            for name, df in cohorts.items()
        },
        "cancer_groups": CANCER_GROUPS,
        "non_cancer_groups": NON_CANCER_GROUPS,
    }
    (args.out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[cohort] merged SERS-linked rows: {len(merged)}")
    print(
        cohort_summary[cohort_summary["disease_group"].eq("__overall__")]
        .sort_values("cohort")
        .to_string(index=False)
    )

    if args.mode == "audit":
        print(f"[written] {args.out_dir}")
        return 0

    global_metrics, global_preds, global_cal, global_subgroup, global_avail = run_global_experiments(
        cohorts,
        spectral_cols,
        n_splits=args.n_splits,
        random_state=args.random_state,
        min_marker_per_class=args.min_marker_per_class,
        calibration_cv=args.calibration_cv,
    )
    write_frame(global_metrics, args.out_dir / "global_model_metrics.csv")
    write_frame(global_preds, args.out_dir / "global_oof_predictions.csv")

    calibration_frames = [global_cal]
    subgroup_frames = [global_subgroup]
    availability_frames = [global_avail]

    if args.mode == "all":
        cancer_metrics, cancer_preds, cancer_cal, cancer_subgroup, cancer_avail = run_cancer_specific_experiments(
            cohorts,
            spectral_cols,
            n_splits=args.n_splits,
            random_state=args.random_state,
            min_marker_per_class=args.min_marker_per_class,
            calibration_cv=args.calibration_cv,
        )
        write_frame(cancer_metrics, args.out_dir / "cancer_specific_metrics.csv")
        write_frame(cancer_preds, args.out_dir / "cancer_specific_oof_predictions.csv")
        calibration_frames.append(cancer_cal)
        subgroup_frames.append(cancer_subgroup)
        availability_frames.append(cancer_avail)

    if not args.skip_lo_source:
        lo_source = run_leave_one_source(
            cohorts,
            spectral_cols,
            random_state=args.random_state,
            min_marker_per_class=args.min_marker_per_class,
            calibration_cv=args.calibration_cv,
        )
        write_frame(lo_source, args.out_dir / "lo_source_metrics.csv")

    calibration_all = pd.concat(calibration_frames, ignore_index=True)
    subgroup_all = pd.concat(subgroup_frames, ignore_index=True)
    availability_all = pd.concat(availability_frames, ignore_index=True)
    write_frame(calibration_all, args.out_dir / "calibration_curves.csv")
    write_frame(subgroup_all, args.out_dir / "subgroup_metrics.csv")
    write_frame(availability_all, args.out_dir / "feature_set_availability.csv")

    print(f"[global] metric rows: {len(global_metrics)}")
    if args.mode == "all":
        print(f"[cancer-specific] metric rows: {len(cancer_metrics)}")
    print(f"[written] {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
