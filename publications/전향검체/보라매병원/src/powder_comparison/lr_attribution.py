from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True, slots=True)
class PeakContributionBatch:
    names: tuple[str, ...]
    global_importance: np.ndarray
    local_contributions: np.ndarray


@dataclass(frozen=True, slots=True)
class LrPeakAttributions:
    legacy: PeakContributionBatch
    transfer: PeakContributionBatch
    native: PeakContributionBatch


def _fit_binary_lr(features: np.ndarray, labels: np.ndarray, seed: int) -> GridSearchCV:
    estimator = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight="balanced", solver="lbfgs"),
    )
    search = GridSearchCV(
        estimator,
        {"logisticregression__C": (0.001, 0.01, 0.1, 1.0, 10.0)},
        scoring="balanced_accuracy",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        n_jobs=-1,
    )
    return search.fit(features, labels)


def _model_parts(search: GridSearchCV) -> tuple[StandardScaler, LogisticRegression]:
    pipeline = search.best_estimator_
    return pipeline.named_steps["standardscaler"], pipeline.named_steps["logisticregression"]


def _peak_indices(
    grid: np.ndarray,
    legacy_importance: np.ndarray,
    powder_importance: np.ndarray,
    *,
    count: int,
    minimum_distance_cm: float,
) -> np.ndarray:
    grid_step = float(np.median(np.diff(grid)))
    distance = max(1, int(round(minimum_distance_cm / grid_step)))
    selected: set[int] = set()
    for importance in (legacy_importance, powder_importance):
        candidates, _ = find_peaks(importance, distance=distance)
        ranked = candidates[np.argsort(-importance[candidates])[:count]]
        selected.update(int(index) for index in ranked)
    return np.array(sorted(selected), dtype=int)


def fit_lr_peak_attributions(
    grid: np.ndarray,
    legacy: np.ndarray,
    powder: np.ndarray,
    clinical_groups: np.ndarray,
    *,
    peak_count: int = 20,
    minimum_distance_cm: float = 18.0,
) -> LrPeakAttributions:
    labels = (clinical_groups == "Cancer").astype(int)
    legacy_search = _fit_binary_lr(legacy, labels, seed=100)
    powder_search = _fit_binary_lr(powder, labels, seed=200)
    legacy_scaler, legacy_model = _model_parts(legacy_search)
    powder_scaler, powder_model = _model_parts(powder_search)
    legacy_coefficients = legacy_model.coef_[0]
    powder_coefficients = powder_model.coef_[0]
    legacy_importance = np.abs(legacy_coefficients)
    powder_importance = np.abs(powder_coefficients)
    indices = _peak_indices(
        grid,
        legacy_importance,
        powder_importance,
        count=peak_count,
        minimum_distance_cm=minimum_distance_cm,
    )
    names = tuple(f"{grid[index]:.1f} cm^-1" for index in indices)
    return LrPeakAttributions(
        legacy=PeakContributionBatch(
            names,
            legacy_importance[indices] / np.sum(legacy_importance[indices]),
            (legacy_scaler.transform(legacy) * legacy_coefficients)[:, indices],
        ),
        transfer=PeakContributionBatch(
            names,
            legacy_importance[indices] / np.sum(legacy_importance[indices]),
            (legacy_scaler.transform(powder) * legacy_coefficients)[:, indices],
        ),
        native=PeakContributionBatch(
            names,
            powder_importance[indices] / np.sum(powder_importance[indices]),
            (powder_scaler.transform(powder) * powder_coefficients)[:, indices],
        ),
    )
