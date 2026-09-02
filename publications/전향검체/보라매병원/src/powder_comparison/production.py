from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powder_comparison.cohort import PairedSubject, SubjectFiles
from scripts.deployment.sers_predict import (
    MIN_DETECTED_REPLICATE_COUNT,
    MIN_VALID_REPLICATE_COUNT,
    StackingPredictor,
    _has_critical_qc_failure,
)
from src.sers.io import read_spectrum


@dataclass(frozen=True, slots=True)
class ProductionBatch:
    probabilities: np.ndarray
    predictions: np.ndarray
    qc_passed: np.ndarray
    valid: np.ndarray
    peak_features: np.ndarray


@dataclass(frozen=True, slots=True)
class PeakContributionBatch:
    names: tuple[str, ...]
    local_contributions: np.ndarray
    global_importance: np.ndarray


@dataclass(frozen=True, slots=True)
class _SubjectScore:
    probability: float
    prediction: int
    qc_passed: int
    valid: bool
    peak_features: np.ndarray


def _invalid_score(feature_count: int) -> _SubjectScore:
    return _SubjectScore(
        probability=float("nan"),
        prediction=-1,
        qc_passed=0,
        valid=False,
        peak_features=np.full(feature_count, np.nan, dtype=float),
    )


def _score_subject(predictor: StackingPredictor, subject: SubjectFiles) -> _SubjectScore:
    spectra = []
    for path in subject.replicates:
        try:
            x_axis, intensity = read_spectrum(path)
            spectra.append(
                {
                    "filepath": path,
                    "x": x_axis,
                    "y": intensity,
                    "features": predictor.preprocess(x_axis, intensity, instrument="thermo"),
                    "multichannel": predictor._preprocess_multichannel(
                        x_axis, intensity, instrument="thermo"
                    ),
                }
            )
        except (OSError, ValueError):
            continue
    for spectrum in spectra:
        spectrum["peak_features"] = predictor._extract_peak_features_single(
            spectrum["multichannel"][0]
        )
    checked = predictor._qc_replicate_set(spectra)
    passed = [spectrum for spectrum in checked if spectrum["qc_pass"]]
    failed = [spectrum for spectrum in checked if not spectrum["qc_pass"]]
    if _has_critical_qc_failure(failed) or len(passed) < MIN_VALID_REPLICATE_COUNT:
        return _invalid_score(len(predictor.known_peaks) * 4 + 7)

    probabilities: list[float] = []
    peak_features: list[np.ndarray] = []
    for spectrum in passed:
        s1_probabilities, _ = predictor._get_base_predictions(
            spectrum["multichannel"], spectrum["peak_features"]
        )
        cancer_probability = float(
            predictor.meta_s1.predict_proba(s1_probabilities.reshape(1, -1))[0, 1]
        )
        probabilities.append(cancer_probability)
        peak_features.append(spectrum["peak_features"])
    detected = sum(probability > predictor.threshold for probability in probabilities)
    return _SubjectScore(
        probability=float(np.mean(probabilities)),
        prediction=int(detected >= MIN_DETECTED_REPLICATE_COUNT),
        qc_passed=len(passed),
        valid=True,
        peak_features=np.vstack(peak_features).mean(axis=0),
    )


def score_production_model(
    predictor: StackingPredictor,
    pairs: tuple[PairedSubject, ...],
    *,
    powder: bool,
) -> ProductionBatch:
    scores = [_score_subject(predictor, pair.powder if powder else pair.legacy) for pair in pairs]
    return ProductionBatch(
        probabilities=np.array([score.probability for score in scores], dtype=float),
        predictions=np.array([score.prediction for score in scores], dtype=int),
        qc_passed=np.array([score.qc_passed for score in scores], dtype=int),
        valid=np.array([score.valid for score in scores], dtype=bool),
        peak_features=np.vstack([score.peak_features for score in scores]),
    )


def lr_peak_contributions(
    predictor: StackingPredictor, peak_features: np.ndarray
) -> PeakContributionBatch:
    peak_model = predictor.base_models["lr_peak"]["s1"]
    scaler = peak_model.named_steps["standardscaler"]
    classifier = peak_model.named_steps["logisticregression"]
    scaled = scaler.transform(peak_features)
    feature_contributions = scaled * classifier.coef_[0]
    peak_count = len(predictor.known_peaks)
    local = (
        feature_contributions[:, : peak_count * 4]
        .reshape(len(peak_features), 4, peak_count)
        .sum(axis=1)
    )
    return PeakContributionBatch(
        names=tuple(str(peak[1]) for peak in predictor.known_peaks),
        local_contributions=local,
        global_importance=np.mean(np.abs(local), axis=0),
    )
