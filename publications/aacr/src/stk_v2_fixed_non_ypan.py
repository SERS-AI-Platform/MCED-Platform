"""Shared helpers for AACR figures from the STK-V2 non-YPAN fixed split."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CANCER_TYPES = ("PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC")


class AacrEnvironmentError(RuntimeError):
    def __init__(self, name: str, unsupported: tuple[str, ...]) -> None:
        self.name = name
        self.unsupported = unsupported
        values = ", ".join(unsupported)
        super().__init__(f"{name} contains unsupported values: {values}")


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    if not value:
        return default
    return Path(value).expanduser().resolve()


def _env_tuple(
    name: str,
    default: tuple[str, ...],
    allowed: frozenset[str],
) -> tuple[str, ...]:
    value = os.environ.get(name)
    if value is None:
        return default
    items = [part.strip().upper() for part in value.split(",") if part.strip()]
    if not items:
        return default
    unsupported = tuple(sorted(set(items) - allowed))
    if unsupported:
        raise AacrEnvironmentError(name, unsupported)
    return tuple(items)


def _env_exclude(default: tuple[str, ...] = ("YPAN",)) -> set[str]:
    value = os.environ.get("SERS_AACR_EXCLUDE_SOURCE_GROUPS")
    if value is None:
        return set(default)
    return {part.strip().upper() for part in value.split(",") if part.strip()}


RUN_DIR = _env_path("SERS_AACR_RUN_DIR", ROOT / "results" / "training" / "stacking_v2_non_ypan")
DATA_DIR = _env_path("SERS_AACR_DATA_DIR", ROOT / "results" / "preprocessing_dacr_all")
TRAINING_FIG_DIR = _env_path(
    "SERS_AACR_TRAINING_FIG_DIR",
    ROOT / "results" / "figures" / "training" / RUN_DIR.name,
)
ATTRIBUTION_PATH = RUN_DIR / "overall_stacking_peak_attribution.csv"
SPECTRAL_IMPORTANCE_PATH = RUN_DIR / "overall_stacking_spectral_importance.npz"
TYPE_FEATURE_IMPORTANCE_PATH = RUN_DIR / "cancer_type_peak_feature_importance.csv"
TYPE_SPECTRAL_IMPORTANCE_PATH = RUN_DIR / "cancer_type_spectral_feature_importance.npz"
BINARY_FEATURE_IMPORTANCE_PATH = RUN_DIR / "cancer_detection_peak_feature_importance.csv"
BINARY_SPECTRAL_IMPORTANCE_PATH = RUN_DIR / "cancer_detection_spectral_feature_importance.npz"
BINARY_TYPE_FEATURE_IMPORTANCE_PATH = (
    RUN_DIR / "cancer_detection_by_type_peak_feature_importance.csv"
)
BINARY_TYPE_SPECTRAL_IMPORTANCE_PATH = (
    RUN_DIR / "cancer_detection_by_type_spectral_feature_importance.npz"
)

DISPLAY_CANCERS = _env_tuple(
    "SERS_AACR_DISPLAY_CANCERS",
    ("LUN", "CRC", "BLC", "PRO", "OVA", "PAN"),
    frozenset(CANCER_TYPES),
)
EXCLUDE_SOURCE_GROUPS = _env_exclude()
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")
BASE_MODELS = (
    "lr_raw",
    "lr_d1",
    "lr_d2",
    "lr_concat",
    "lr_peak",
    "xgb_raw",
    "xgb_d1",
    "rf_raw",
    "rf_d1",
    "ridge_concat",
)

KNOWN_PEAKS = (
    (448.1, "ring_deform", 15),
    (538.7, "SS_stretch", 15),
    (617.7, "CS_stretch", 15),
    (683.3, "creatinine", 15),
    (723.8, "adenine", 15),
    (795.1, "hippuric", 15),
    (849.1, "tyrosine", 15),
    (895.4, "uric_acid", 15),
    (933.9, "creatinine2", 15),
    (999.5, "phe_urea", 15),
    (1147.9, "uric_CN", 15),
    (1230.8, "amide_III", 20),
    (1292.5, "CH2_twist", 15),
    (1352.3, "trp_fermi", 15),
    (1448.7, "CH2_deform", 20),
    (1597.1, "purine_CC", 20),
    (1651.1, "amide_I", 20),
)
PEAK_RATIOS = (
    ("phe_urea", "adenine"),
    ("phe_urea", "creatinine"),
    ("hippuric", "creatinine"),
    ("CS_stretch", "creatinine"),
    ("amide_I", "CH2_deform"),
    ("adenine", "purine_CC"),
    ("tyrosine", "phe_urea"),
)
PEAK_ASSIGNMENT_CANDIDATES = {
    "ring_deform": (
        "Ring/skeletal deformation",
        "protein or nucleic-acid ring modes",
    ),
    "SS_stretch": (
        "S-S stretching",
        "disulfide-rich protein",
        "cysteine/cystine-related mode",
    ),
    "CS_stretch": (
        "C-S stretching",
        "protein sulfur-containing residues",
        "creatinine-related mode",
    ),
    "creatinine": (
        "Creatinine",
        "nucleic-acid ring mode",
        "amino-acid side-chain mode",
    ),
    "adenine": (
        "Adenine",
        "nucleic-acid purine mode",
        "uric-acid-related mode",
    ),
    "hippuric": (
        "Hippuric acid",
        "protein C-C/C-N mode",
        "ring breathing mode",
    ),
    "tyrosine": (
        "Tyrosine",
        "protein ring mode",
        "collagen/proline-related mode",
    ),
    "uric_acid": (
        "Uric acid",
        "nucleic-acid backbone/ring mode",
        "C-C skeletal mode",
    ),
    "creatinine2": (
        "Creatinine",
        "protein C-C skeletal mode",
        "carbohydrate-related mode",
    ),
    "phe_urea": (
        "Phenylalanine",
        "urea",
        "aromatic ring breathing",
    ),
    "uric_CN": (
        "Uric acid C-N stretching",
        "protein C-N mode",
        "nucleic-acid ring mode",
    ),
    "amide_III": (
        "Amide III / protein",
        "nucleic-acid base mode",
        "CH bending",
    ),
    "CH2_twist": (
        "CH2 twisting / lipid",
        "protein CH deformation",
        "collagen-related mode",
    ),
    "trp_fermi": (
        "Tryptophan / Fermi resonance",
        "nucleic-acid base mode",
        "CH deformation",
    ),
    "CH2_deform": (
        "CH2 deformation / lipid",
        "protein CH2/CH3 bending",
        "collagen-related mode",
    ),
    "purine_CC": (
        "Purine C=C/C=N",
        "adenine/guanine",
        "aromatic amino-acid mode",
    ),
    "amide_I": (
        "Amide I / protein",
        "C=C lipid stretching",
        "water/protein carbonyl mode",
    ),
}
PEAK_ASSIGNMENTS = {name: candidates[0] for name, candidates in PEAK_ASSIGNMENT_CANDIDATES.items()}


def peak_representative_assignment(peak_name: str) -> str:
    return PEAK_ASSIGNMENT_CANDIDATES.get(peak_name, (peak_name,))[0]


def peak_candidate_assignments(peak_name: str) -> str:
    return "; ".join(PEAK_ASSIGNMENT_CANDIDATES.get(peak_name, (peak_name,)))


def add_peak_assignment_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "peak_name" not in df.columns:
        return df
    df["representative_assignment"] = df["peak_name"].map(peak_representative_assignment)
    df["candidate_assignments"] = df["peak_name"].map(peak_candidate_assignments)
    if "major_assignment" not in df.columns:
        df["major_assignment"] = df["representative_assignment"]
    return df


def load_test_predictions(run_dir: Path = RUN_DIR) -> dict[str, np.ndarray]:
    d = np.load(run_dir / "fixed_test_predictions.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_cancer_types(run_dir: Path = RUN_DIR) -> list[str]:
    path = run_dir / "fixed_split_results.json"
    if not path.exists():
        return list(CANCER_TYPES)
    with path.open("r", encoding="utf-8") as f:
        return list((json.load(f)).get("cancer_types", CANCER_TYPES))


def bootstrap_roc_ci(
    y_true: np.ndarray,
    score: np.ndarray,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
    fpr_grid: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    y_true = np.asarray(y_true, dtype=int)
    score = np.asarray(score, dtype=float)
    fpr, tpr, _ = roc_curve(y_true, score)
    auc_value = float(roc_auc_score(y_true, score))
    if fpr_grid is None:
        fpr_grid = np.linspace(0, 1, 101)
    rng = np.random.default_rng(seed)
    aucs = []
    tprs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        if len(np.unique(y_true[idx])) < 2:
            continue
        fpr_b, tpr_b, _ = roc_curve(y_true[idx], score[idx])
        aucs.append(float(roc_auc_score(y_true[idx], score[idx])))
        interp = np.interp(fpr_grid, fpr_b, tpr_b)
        interp[0] = 0.0
        interp[-1] = 1.0
        tprs.append(interp)
    if aucs:
        auc_ci = np.percentile(aucs, [2.5, 97.5])
    else:
        auc_ci = [np.nan, np.nan]
    if tprs:
        tpr_arr = np.vstack(tprs)
        tpr_ci = np.percentile(tpr_arr, [2.5, 97.5], axis=0)
    else:
        tpr_ci = np.vstack([np.full_like(fpr_grid, np.nan), np.full_like(fpr_grid, np.nan)])
    return {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_value,
        "auc_ci_low": float(auc_ci[0]),
        "auc_ci_high": float(auc_ci[1]),
        "fpr_grid": fpr_grid,
        "tpr_ci_low": tpr_ci[0],
        "tpr_ci_high": tpr_ci[1],
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def bootstrap_metric_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn,
    *,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple[float, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(y_true), len(y_true))
        try:
            values.append(float(metric_fn(y_true[idx], y_score[idx])))
        except ValueError:
            continue
    if not values:
        return np.nan, np.nan
    return tuple(float(v) for v in np.percentile(values, [2.5, 97.5]))


def load_non_ypan_data():
    from scripts.training import train_usersnet as trainer

    trainer.DATA_TYPE = "processed_csv"
    return trainer.load_data(DATA_DIR, exclude_source_groups=EXCLUDE_SOURCE_GROUPS)


def load_plot_data():
    """Load spectra for display using the configured source-group filter."""
    from scripts.training import train_usersnet as trainer

    trainer.DATA_TYPE = "processed_csv"
    return trainer.load_data(DATA_DIR, exclude_source_groups=EXCLUDE_SOURCE_GROUPS)


def get_fixed_split():
    d = np.load(RUN_DIR / "fixed_split_indices.npz", allow_pickle=True)
    return d["train_idx"], d["val_idx"], d["test_idx"]


def get_meta_input_weights() -> pd.DataFrame:
    path = TRAINING_FIG_DIR / "meta_shap_input_importance.csv"
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = _compute_meta_input_weights()
    rows = []
    for model in BASE_MODELS:
        s1 = df[(df["task"] == "Cancer vs Non-cancer") & (df["base_model"] == model)][
            "mean_abs_linear_shap"
        ]
        s2 = df[(df["task"] == "Cancer Type") & (df["base_model"] == model)]["mean_abs_linear_shap"]
        rows.append(
            {
                "base_model": model,
                "s1_weight": float(s1.iloc[0]) if len(s1) else 0.0,
                "s2_weight": float(s2.iloc[0]) if len(s2) else 0.0,
            }
        )
    out = pd.DataFrame(rows)
    for col in ("s1_weight", "s2_weight"):
        total = out[col].sum()
        out[col] = out[col] / total if total > 0 else 0.0
    out["combined_weight"] = 0.5 * out["s1_weight"] + 0.5 * out["s2_weight"]
    return out


def _new_meta_classifier(meta_name: str) -> LogisticRegression:
    if meta_name == "elasticnet":
        return LogisticRegression(
            C=0.5,
            penalty="elasticnet",
            l1_ratio=0.5,
            max_iter=2000,
            solver="saga",
            n_jobs=1,
            random_state=42,
        )
    return LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", n_jobs=1)


def _compute_meta_input_weights() -> pd.DataFrame:
    oof = np.load(RUN_DIR / "fixed_train_oof.npz", allow_pickle=True)
    meta = json.loads((RUN_DIR / "fixed_split_results.json").read_text())
    meta_name = meta.get("best_meta", "elasticnet")
    rows = []
    x1 = np.asarray(oof["meta_s1"], dtype=float)
    yb = np.asarray(oof["y_bin"], dtype=int)
    m1 = _new_meta_classifier(meta_name)
    m1.fit(x1, yb)
    s1_imp = np.abs((x1 - x1.mean(axis=0)) * np.ravel(m1.coef_)).mean(axis=0)
    x2 = np.asarray(oof["meta_s2"], dtype=float)
    yt = np.asarray(oof["y_type"], dtype=int)
    cancer = yb == 1
    m2 = _new_meta_classifier(meta_name)
    m2.fit(x2[cancer], yt[cancer])
    coefs = np.asarray(m2.coef_, dtype=float)
    s2_imp_feature = np.abs(
        (x2[cancer] - x2[cancer].mean(axis=0))[:, None, :] * coefs[None, :, :]
    ).mean(axis=(0, 1))
    s2_imp = s2_imp_feature.reshape(len(BASE_MODELS), len(CANCER_TYPES)).mean(axis=1)
    for model, value in zip(BASE_MODELS, s1_imp):
        rows.append(
            {
                "task": "Cancer vs Non-cancer",
                "base_model": model,
                "mean_abs_linear_shap": float(value),
            }
        )
    for model, value in zip(BASE_MODELS, s2_imp):
        rows.append(
            {"task": "Cancer Type", "base_model": model, "mean_abs_linear_shap": float(value)}
        )
    return pd.DataFrame(rows)


def peak_feature_names() -> list[str]:
    names = []
    for kind in ("area", "height", "fwhm", "shift"):
        names.extend([f"{name}_{kind}" for _, name, _ in KNOWN_PEAKS])
    names.extend([f"ratio_{a}_over_{b}" for a, b in PEAK_RATIOS])
    return names


def _extract_estimator(model):
    if isinstance(model, Pipeline):
        return list(model.named_steps.values())[-1]
    return model


def _feature_importance(model, X_eval: np.ndarray, X_bg: np.ndarray) -> np.ndarray:
    if isinstance(model, Pipeline):
        scaler = list(model.named_steps.values())[0]
        estimator = _extract_estimator(model)
        X_eval_t = scaler.transform(X_eval)
        X_bg_t = scaler.transform(X_bg)
    else:
        estimator = _extract_estimator(model)
        X_eval_t = X_eval
        X_bg_t = X_bg

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        bg = X_bg_t.mean(axis=0)
        contrib = np.abs((X_eval_t[:, None, :] - bg[None, None, :]) * coef[None, :, :])
        return contrib.mean(axis=(0, 1))
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float).ravel()
    return np.zeros(X_eval.shape[1], dtype=float)


def _class_feature_importance(
    model,
    X_eval: np.ndarray,
    X_bg: np.ndarray,
    *,
    target_class: int,
    fitted_classes: np.ndarray,
) -> np.ndarray:
    if len(X_eval) == 0:
        return np.zeros(X_bg.shape[1], dtype=float)
    if isinstance(model, Pipeline):
        scaler = list(model.named_steps.values())[0]
        estimator = _extract_estimator(model)
        X_eval_t = scaler.transform(X_eval)
        X_bg_t = scaler.transform(X_bg)
    else:
        estimator = _extract_estimator(model)
        X_eval_t = X_eval
        X_bg_t = X_bg

    if hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        if coef.ndim == 1 or coef.shape[0] == 1:
            class_coef = coef.reshape(-1)
        else:
            classes = np.asarray(fitted_classes, dtype=int)
            matches = np.where(classes == int(target_class))[0]
            if len(matches) == 0:
                return np.zeros(X_eval.shape[1], dtype=float)
            class_coef = coef[int(matches[0])]
        bg = X_bg_t.mean(axis=0)
        return np.abs((X_eval_t - bg) * class_coef).mean(axis=0)
    if hasattr(estimator, "feature_importances_"):
        return np.asarray(estimator.feature_importances_, dtype=float).ravel()
    return np.zeros(X_eval.shape[1], dtype=float)


def get_meta_type_input_weights() -> pd.DataFrame:
    oof = np.load(RUN_DIR / "fixed_train_oof.npz", allow_pickle=True)
    meta = json.loads((RUN_DIR / "fixed_split_results.json").read_text())
    meta_name = meta.get("best_meta", "elasticnet")
    x2 = np.asarray(oof["meta_s2"], dtype=float)
    yb = np.asarray(oof["y_bin"], dtype=int)
    yt = np.asarray(oof["y_type"], dtype=int)
    cancer = yb == 1
    model = _new_meta_classifier(meta_name)
    model.fit(x2[cancer], yt[cancer])
    coefs = np.asarray(model.coef_, dtype=float)
    classes = np.asarray(getattr(model, "classes_", np.unique(yt[cancer])), dtype=int)
    rows = []
    n_types = len(CANCER_TYPES)
    x2_c = x2[cancer]
    for class_pos, class_id in enumerate(classes):
        if class_id < 0 or class_id >= n_types:
            continue
        raw_weights = []
        for model_idx, base_model in enumerate(BASE_MODELS):
            col = model_idx * n_types + int(class_id)
            if col >= x2_c.shape[1]:
                value = 0.0
            else:
                centered = x2_c[:, col] - x2_c[:, col].mean()
                value = float(np.abs(centered * coefs[class_pos, col]).mean())
            raw_weights.append(value)
        total = float(np.sum(raw_weights))
        for base_model, value in zip(BASE_MODELS, raw_weights):
            rows.append(
                {
                    "cancer_type": CANCER_TYPES[int(class_id)],
                    "base_model": base_model,
                    "weight": value / total if total > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _map_feature_importance_to_grid(
    model_name: str,
    spec: dict,
    imp: np.ndarray,
    grid: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    n_grid = len(grid)
    score = np.zeros(n_grid, dtype=float)
    peak_scores = {name: 0.0 for _, name, _ in KNOWN_PEAKS}
    channels = spec["channels"]
    if channels == "peak":
        names = peak_feature_names()
        peak_lookup = {name: (center, half_w) for center, name, half_w in KNOWN_PEAKS}
        for value, feature in zip(imp, names):
            if feature.startswith("ratio_"):
                body = feature[len("ratio_") :]
                if "_over_" in body:
                    a, b = body.split("_over_", 1)
                    targets = [a, b]
                else:
                    targets = []
            else:
                targets = [feature.rsplit("_", 1)[0]]
            for target in targets:
                if target not in peak_lookup:
                    continue
                center, half_w = peak_lookup[target]
                band = np.abs(grid - center) <= half_w
                if band.any():
                    score[band] += float(value) / max(len(targets), 1)
                peak_scores[target] += float(value) / max(len(targets), 1)
        return score, peak_scores

    if len(channels) == 1:
        mapped = imp[:n_grid]
    else:
        usable = imp[: n_grid * len(channels)]
        mapped = usable.reshape(len(channels), n_grid).mean(axis=0)
    score += mapped
    for center, name, half_w in KNOWN_PEAKS:
        band = np.abs(grid - center) <= half_w
        if band.any():
            peak_scores[name] += float(mapped[band].sum())
    return score, peak_scores


def compute_overall_stacking_attribution(
    force: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if ATTRIBUTION_PATH.exists() and SPECTRAL_IMPORTANCE_PATH.exists() and not force:
        df = add_peak_assignment_columns(pd.read_csv(ATTRIBUTION_PATH))
        d = np.load(SPECTRAL_IMPORTANCE_PATH, allow_pickle=True)
        return df, d["grid"], d["score"]

    from scripts.training import train_usersnet as trainer

    X_3ch, _meta, grid, groups, sample_ids, y_bin, y_type = load_non_ypan_data()
    train_idx, _val_idx, test_idx = get_fixed_split()
    peak_artifact = np.load(RUN_DIR / "fixed_peak_features.npz", allow_pickle=True)
    X_peak_train = np.asarray(peak_artifact["X_peak_train"], dtype=float)
    X_peak_test = np.asarray(peak_artifact["X_peak_test"], dtype=float)

    X_train = X_3ch[train_idx]
    X_test = X_3ch[test_idx]
    yb_train = y_bin[train_idx].astype(int)
    yt_train = y_type[train_idx].astype(int)
    yb_test = y_bin[test_idx].astype(int)
    cancer_train = yb_train == 1
    cancer_test = yb_test == 1
    weights = get_meta_input_weights().set_index("base_model")

    total_score = np.zeros(len(grid), dtype=float)
    peak_totals = {name: 0.0 for _, name, _ in KNOWN_PEAKS}

    for model_name, spec in trainer.EXTENDED_BASE_MODELS.items():
        if model_name not in weights.index:
            continue
        channels = spec["channels"]
        if channels == "peak":
            Xtr = X_peak_train
            Xte = X_peak_test
            Xtr_c = X_peak_train[cancer_train]
            Xte_c = X_peak_test[cancer_test]
        else:
            Xtr = X_train[:, channels, :].reshape(len(X_train), -1)
            Xte = X_test[:, channels, :].reshape(len(X_test), -1)
            Xtr_c = Xtr[cancer_train]
            Xte_c = Xte[cancer_test]

        s1_model = trainer.build_classifier(spec["model"], "binary")
        s1_model.fit(Xtr, yb_train)
        s1_imp = _feature_importance(s1_model, Xte, Xtr)
        s1_grid, s1_peak = _map_feature_importance_to_grid(model_name, spec, s1_imp, grid)
        s1_grid = s1_grid / s1_grid.sum() if s1_grid.sum() > 0 else s1_grid
        s1_weight = float(weights.loc[model_name, "s1_weight"])
        total_score += s1_weight * s1_grid
        for peak_name, value in s1_peak.items():
            peak_totals[peak_name] += s1_weight * value

        if cancer_train.sum() > 10 and cancer_test.sum() > 0:
            s2_model = trainer.build_classifier(spec["model"], "multiclass", len(CANCER_TYPES))
            s2_fit = trainer.fit_sparse_multiclass(s2_model, Xtr_c, yt_train[cancer_train])
            s2_model_obj = s2_fit[0]
            s2_imp = _feature_importance(s2_model_obj, Xte_c, Xtr_c)
            s2_grid, s2_peak = _map_feature_importance_to_grid(model_name, spec, s2_imp, grid)
            s2_grid = s2_grid / s2_grid.sum() if s2_grid.sum() > 0 else s2_grid
            s2_weight = float(weights.loc[model_name, "s2_weight"])
            total_score += s2_weight * s2_grid
            for peak_name, value in s2_peak.items():
                peak_totals[peak_name] += s2_weight * value

    total_score = gaussian_filter1d(total_score, sigma=2)
    if total_score.max() > 0:
        total_score = total_score / total_score.max()

    peak_rows = []
    for center, name, half_w in KNOWN_PEAKS:
        band = np.abs(grid - center) <= half_w
        band_score = float(total_score[band].sum()) if band.any() else 0.0
        peak_rows.append(
            {
                "peak_position_cm-1": int(round(center)),
                "peak_name": name,
                "major_assignment": peak_representative_assignment(name),
                "representative_assignment": peak_representative_assignment(name),
                "candidate_assignments": peak_candidate_assignments(name),
                "shade_min_cm-1": int(round(center - half_w)),
                "shade_max_cm-1": int(round(center + half_w)),
                "overall_attribution_score": band_score,
                "rank": 0,
            }
        )
    df = pd.DataFrame(peak_rows).sort_values("overall_attribution_score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df.to_csv(ATTRIBUTION_PATH, index=False, encoding="utf-8-sig")
    np.savez_compressed(SPECTRAL_IMPORTANCE_PATH, grid=grid, score=total_score)
    return df, grid, total_score


def compute_cancer_type_peak_feature_importance(
    force: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    if (
        TYPE_FEATURE_IMPORTANCE_PATH.exists()
        and TYPE_SPECTRAL_IMPORTANCE_PATH.exists()
        and not force
    ):
        df = add_peak_assignment_columns(pd.read_csv(TYPE_FEATURE_IMPORTANCE_PATH))
        d = np.load(TYPE_SPECTRAL_IMPORTANCE_PATH, allow_pickle=True)
        return df, d["grid"], {str(k): d[k] for k in d.files if k != "grid"}

    from scripts.training import train_usersnet as trainer

    X_3ch, _meta, grid, groups, _sample_ids, y_bin, y_type = load_non_ypan_data()
    train_idx, _val_idx, test_idx = get_fixed_split()
    peak_artifact = np.load(RUN_DIR / "fixed_peak_features.npz", allow_pickle=True)
    X_peak_train = np.asarray(peak_artifact["X_peak_train"], dtype=float)
    X_peak_test = np.asarray(peak_artifact["X_peak_test"], dtype=float)

    X_train = X_3ch[train_idx]
    X_test = X_3ch[test_idx]
    yb_train = y_bin[train_idx].astype(int)
    yt_train = y_type[train_idx].astype(int)
    yb_test = y_bin[test_idx].astype(int)
    yt_test = y_type[test_idx].astype(int)
    cancer_train = yb_train == 1
    cancer_test = yb_test == 1
    weights_df = get_meta_type_input_weights()
    weights = {
        (row["cancer_type"], row["base_model"]): float(row["weight"])
        for _, row in weights_df.iterrows()
    }

    score_by_type = {ct: np.zeros(len(grid), dtype=float) for ct in DISPLAY_CANCERS}
    type_index = {ct: i for i, ct in enumerate(CANCER_TYPES)}

    for model_name, spec in trainer.EXTENDED_BASE_MODELS.items():
        if model_name not in BASE_MODELS:
            continue
        channels = spec["channels"]
        if channels == "peak":
            Xtr_c = X_peak_train[cancer_train]
            Xte_c = X_peak_test[cancer_test]
        else:
            Xtr = X_train[:, channels, :].reshape(len(X_train), -1)
            Xte = X_test[:, channels, :].reshape(len(X_test), -1)
            Xtr_c = Xtr[cancer_train]
            Xte_c = Xte[cancer_test]

        s2_model = trainer.build_classifier(spec["model"], "multiclass", len(CANCER_TYPES))
        s2_model_obj, fitted_classes, _remapped = trainer.fit_sparse_multiclass(
            s2_model, Xtr_c, yt_train[cancer_train]
        )
        yt_c_test = yt_test[cancer_test]
        for cancer_type in DISPLAY_CANCERS:
            target_idx = type_index[cancer_type]
            target_mask = yt_c_test == target_idx
            if not target_mask.any():
                continue
            imp = _class_feature_importance(
                s2_model_obj,
                Xte_c[target_mask],
                Xtr_c,
                target_class=target_idx,
                fitted_classes=fitted_classes,
            )
            grid_score, _peak_score = _map_feature_importance_to_grid(model_name, spec, imp, grid)
            if grid_score.sum() > 0:
                grid_score = grid_score / grid_score.sum()
            weight = weights.get((cancer_type, model_name), 0.0)
            score_by_type[cancer_type] += weight * grid_score

    peak_rows = []
    npz_payload = {"grid": grid}
    for cancer_type, score in score_by_type.items():
        score = gaussian_filter1d(score, sigma=2)
        if score.max() > 0:
            score = score / score.max()
        npz_payload[cancer_type] = score
        rows = []
        for center, name, half_w in KNOWN_PEAKS:
            band = np.abs(grid - center) <= half_w
            feature_score = float(score[band].sum()) if band.any() else 0.0
            rows.append(
                {
                    "cancer_type": cancer_type,
                    "peak_position_cm-1": int(round(center)),
                    "peak_name": name,
                    "major_assignment": peak_representative_assignment(name),
                    "representative_assignment": peak_representative_assignment(name),
                    "candidate_assignments": peak_candidate_assignments(name),
                    "shade_min_cm-1": int(round(center - half_w)),
                    "shade_max_cm-1": int(round(center + half_w)),
                    "feature_importance_score": feature_score,
                    "rank": 0,
                }
            )
        sub = pd.DataFrame(rows).sort_values("feature_importance_score", ascending=False)
        sub["rank"] = np.arange(1, len(sub) + 1)
        peak_rows.extend(sub.to_dict("records"))
    df = pd.DataFrame(peak_rows)
    df.to_csv(TYPE_FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
    np.savez_compressed(TYPE_SPECTRAL_IMPORTANCE_PATH, **npz_payload)
    return df, grid, {k: v for k, v in npz_payload.items() if k != "grid"}


def compute_cancer_detection_peak_feature_importance(
    force: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if (
        BINARY_FEATURE_IMPORTANCE_PATH.exists()
        and BINARY_SPECTRAL_IMPORTANCE_PATH.exists()
        and not force
    ):
        df = add_peak_assignment_columns(pd.read_csv(BINARY_FEATURE_IMPORTANCE_PATH))
        d = np.load(BINARY_SPECTRAL_IMPORTANCE_PATH, allow_pickle=True)
        return df, d["grid"], d["score"]

    from scripts.training import train_usersnet as trainer

    X_3ch, _meta, grid, _groups, _sample_ids, y_bin, _y_type = load_non_ypan_data()
    train_idx, _val_idx, test_idx = get_fixed_split()
    peak_artifact = np.load(RUN_DIR / "fixed_peak_features.npz", allow_pickle=True)
    X_peak_train = np.asarray(peak_artifact["X_peak_train"], dtype=float)
    X_peak_test = np.asarray(peak_artifact["X_peak_test"], dtype=float)

    X_train = X_3ch[train_idx]
    X_test = X_3ch[test_idx]
    yb_train = y_bin[train_idx].astype(int)
    weights = get_meta_input_weights().set_index("base_model")

    total_score = np.zeros(len(grid), dtype=float)
    for model_name, spec in trainer.EXTENDED_BASE_MODELS.items():
        if model_name not in weights.index:
            continue
        channels = spec["channels"]
        if channels == "peak":
            Xtr = X_peak_train
            Xte = X_peak_test
        else:
            Xtr = X_train[:, channels, :].reshape(len(X_train), -1)
            Xte = X_test[:, channels, :].reshape(len(X_test), -1)

        s1_model = trainer.build_classifier(spec["model"], "binary")
        s1_model.fit(Xtr, yb_train)
        imp = _feature_importance(s1_model, Xte, Xtr)
        grid_score, _peak_score = _map_feature_importance_to_grid(model_name, spec, imp, grid)
        if grid_score.sum() > 0:
            grid_score = grid_score / grid_score.sum()
        total_score += float(weights.loc[model_name, "s1_weight"]) * grid_score

    total_score = gaussian_filter1d(total_score, sigma=2)
    if total_score.max() > 0:
        total_score = total_score / total_score.max()

    peak_rows = []
    for center, name, half_w in KNOWN_PEAKS:
        band = np.abs(grid - center) <= half_w
        feature_score = float(total_score[band].sum()) if band.any() else 0.0
        peak_rows.append(
            {
                "rank": 0,
                "peak_position_cm-1": int(round(center)),
                "peak_name": name,
                "major_assignment": peak_representative_assignment(name),
                "representative_assignment": peak_representative_assignment(name),
                "candidate_assignments": peak_candidate_assignments(name),
                "shade_min_cm-1": int(round(center - half_w)),
                "shade_max_cm-1": int(round(center + half_w)),
                "feature_importance_score": feature_score,
            }
        )
    df = pd.DataFrame(peak_rows).sort_values("feature_importance_score", ascending=False)
    df["rank"] = np.arange(1, len(df) + 1)
    df.to_csv(BINARY_FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
    np.savez_compressed(BINARY_SPECTRAL_IMPORTANCE_PATH, grid=grid, score=total_score)
    return df, grid, total_score


def compute_cancer_detection_by_type_peak_feature_importance(
    force: bool = False,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, np.ndarray]]:
    if (
        BINARY_TYPE_FEATURE_IMPORTANCE_PATH.exists()
        and BINARY_TYPE_SPECTRAL_IMPORTANCE_PATH.exists()
        and not force
    ):
        df = add_peak_assignment_columns(pd.read_csv(BINARY_TYPE_FEATURE_IMPORTANCE_PATH))
        d = np.load(BINARY_TYPE_SPECTRAL_IMPORTANCE_PATH, allow_pickle=True)
        return df, d["grid"], {str(k): d[k] for k in d.files if k != "grid"}

    from scripts.training import train_usersnet as trainer

    X_3ch, _meta, grid, groups, _sample_ids, y_bin, _y_type = load_non_ypan_data()
    train_idx, _val_idx, test_idx = get_fixed_split()
    peak_artifact = np.load(RUN_DIR / "fixed_peak_features.npz", allow_pickle=True)
    X_peak_train = np.asarray(peak_artifact["X_peak_train"], dtype=float)
    X_peak_test = np.asarray(peak_artifact["X_peak_test"], dtype=float)

    X_train = X_3ch[train_idx]
    X_test = X_3ch[test_idx]
    yb_train = y_bin[train_idx].astype(int)
    groups_test = groups[test_idx]
    weights = get_meta_input_weights().set_index("base_model")
    score_by_type = {ct: np.zeros(len(grid), dtype=float) for ct in DISPLAY_CANCERS}

    for model_name, spec in trainer.EXTENDED_BASE_MODELS.items():
        if model_name not in weights.index:
            continue
        channels = spec["channels"]
        if channels == "peak":
            Xtr = X_peak_train
            Xte = X_peak_test
            Xbg = X_peak_train[yb_train == 0]
        else:
            Xtr = X_train[:, channels, :].reshape(len(X_train), -1)
            Xte = X_test[:, channels, :].reshape(len(X_test), -1)
            Xbg = Xtr[yb_train == 0]

        s1_model = trainer.build_classifier(spec["model"], "binary")
        s1_model.fit(Xtr, yb_train)
        weight = float(weights.loc[model_name, "s1_weight"])
        for cancer_type in DISPLAY_CANCERS:
            target_mask = groups_test == cancer_type
            if not target_mask.any():
                continue
            imp = _feature_importance(s1_model, Xte[target_mask], Xbg)
            grid_score, _peak_score = _map_feature_importance_to_grid(model_name, spec, imp, grid)
            if grid_score.sum() > 0:
                grid_score = grid_score / grid_score.sum()
            score_by_type[cancer_type] += weight * grid_score

    peak_rows = []
    npz_payload = {"grid": grid}
    for cancer_type, score in score_by_type.items():
        score = gaussian_filter1d(score, sigma=2)
        if score.max() > 0:
            score = score / score.max()
        npz_payload[cancer_type] = score
        rows = []
        for center, name, half_w in KNOWN_PEAKS:
            band = np.abs(grid - center) <= half_w
            feature_score = float(score[band].sum()) if band.any() else 0.0
            rows.append(
                {
                    "cancer_type": cancer_type,
                    "rank": 0,
                    "peak_position_cm-1": int(round(center)),
                    "peak_name": name,
                    "major_assignment": peak_representative_assignment(name),
                    "representative_assignment": peak_representative_assignment(name),
                    "candidate_assignments": peak_candidate_assignments(name),
                    "shade_min_cm-1": int(round(center - half_w)),
                    "shade_max_cm-1": int(round(center + half_w)),
                    "feature_importance_score": feature_score,
                }
            )
        sub = pd.DataFrame(rows).sort_values("feature_importance_score", ascending=False)
        sub["rank"] = np.arange(1, len(sub) + 1)
        peak_rows.extend(sub.to_dict("records"))
    df = pd.DataFrame(peak_rows)
    df.to_csv(BINARY_TYPE_FEATURE_IMPORTANCE_PATH, index=False, encoding="utf-8-sig")
    np.savez_compressed(BINARY_TYPE_SPECTRAL_IMPORTANCE_PATH, **npz_payload)
    return df, grid, {k: v for k, v in npz_payload.items() if k != "grid"}


def group_spectrum_stats(
    groups_to_keep: tuple[str, ...] = DISPLAY_CANCERS + ("CONTROL",),
) -> dict[str, dict[str, np.ndarray | int]]:
    X_3ch, _meta, grid, groups, _sample_ids, _yb, _yt = load_plot_data()
    stats = {}
    for group in groups_to_keep:
        if group == "CONTROL":
            mask = np.isin(groups, NON_CANCER_GROUPS)
        elif group == "CANCER":
            mask = np.isin(groups, CANCER_TYPES)
        else:
            mask = groups == group
        if not mask.any():
            continue
        spectra = X_3ch[mask, 0, :]
        stats[group] = {
            "mean": spectra.mean(axis=0),
            "sem": spectra.std(axis=0, ddof=1) / np.sqrt(len(spectra)),
            "n": int(len(spectra)),
        }
    return stats


def binary_metric_table(n_bootstrap: int = 1000) -> pd.DataFrame:
    d = load_test_predictions()
    y = d["y_bin"].astype(int)
    p = d["s1_prob"].astype(float)
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rows = []
    counts = {
        "Accuracy": (int((pred == y).sum()), len(y), accuracy_score(y, pred)),
        "Sensitivity": (int(tp), int(tp + fn), recall_score(y, pred, zero_division=0)),
        "Specificity": (int(tn), int(tn + fp), tn / (tn + fp) if (tn + fp) else np.nan),
        "Precision": (int(tp), int(tp + fp), precision_score(y, pred, zero_division=0)),
    }
    for metric, (k, n, value) in counts.items():
        lo, hi = wilson_ci(k, n)
        rows.append({"metric": metric, "value": value, "ci_low": lo, "ci_high": hi, "n": n})
    f1_lo, f1_hi = bootstrap_metric_ci(
        y,
        p,
        lambda yy, pp: f1_score(yy, (pp >= 0.5).astype(int), zero_division=0),
        n_bootstrap=n_bootstrap,
        seed=44,
    )
    rows.append(
        {"metric": "F1", "value": f1_score(y, pred), "ci_low": f1_lo, "ci_high": f1_hi, "n": len(y)}
    )
    roc_stats = bootstrap_roc_ci(y, p, n_bootstrap=n_bootstrap, seed=42)
    rows.append(
        {
            "metric": "AUROC",
            "value": float(roc_stats["auc"]),
            "ci_low": float(roc_stats["auc_ci_low"]),
            "ci_high": float(roc_stats["auc_ci_high"]),
            "n": len(y),
        }
    )
    return pd.DataFrame(rows)


def cancer_type_sensitivity_table() -> pd.DataFrame:
    d = load_test_predictions()
    labels = load_cancer_types()
    yb = d["y_bin"].astype(int)
    yt = d["y_type"].astype(int)
    prob = d["s2_prob"].astype(float)
    cancer = yb == 1
    yt_c = yt[cancer]
    pred = prob[cancer].argmax(axis=1)
    present = sorted(set(yt_c.tolist()) | set(pred.tolist()))
    rows = []
    for idx in present:
        if idx < 0 or idx >= len(labels):
            continue
        n = int((yt_c == idx).sum())
        correct = int(((yt_c == idx) & (pred == idx)).sum())
        if n == 0:
            continue
        lo, hi = wilson_ci(correct, n)
        rows.append(
            {
                "cancer_type": labels[idx],
                "sensitivity": correct / n,
                "ci_low": lo,
                "ci_high": hi,
                "correct": correct,
                "n": n,
            }
        )
    return pd.DataFrame(rows)
