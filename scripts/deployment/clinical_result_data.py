"""Compatibility adapters for current and historical prediction JSON."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TypedDict

from .sers_predict import SSI_DECISION_CUTOFF

JsonValue = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]


class PersistedPrediction(TypedDict):
    cancer_detected: JsonValue
    cancer_signal_score: JsonValue
    screening_index: JsonValue
    ssi_score: JsonValue
    ssi_threshold: JsonValue
    model_probability_mean: JsonValue
    model_probability_threshold: JsonValue
    decision_level: JsonValue
    decision_policy: JsonValue
    cancer_signal_spectra_count: int
    qc_valid_spectra_count: int
    cancer_type_prediction: JsonValue
    cancer_type_confidence: JsonValue
    cancer_type_probabilities: JsonValue
    per_replicate: JsonValue
    patient_decision: JsonValue


def build_persisted_prediction(result: Mapping) -> PersistedPrediction:
    raw_decision = dict(result.get("patient_decision", {}))
    raw_decision.pop("majority_vote", None)
    per_replicate = list(result.get("per_replicate", []))
    cancer_signal_count = sum(1 for replicate in per_replicate if replicate.get("cancer_detected"))
    qc_valid_count = len(per_replicate)
    ssi_score = raw_decision.get("ssi_score", raw_decision.get("screening_index", 0.0))
    raw_decision["cancer_signal_spectra_count"] = cancer_signal_count
    raw_decision["qc_valid_spectra_count"] = qc_valid_count
    decision_level = raw_decision.get("decision_level", result.get("decision_level"))
    decision_policy = raw_decision.get("decision_policy", result.get("decision_policy"))
    return {
        "cancer_detected": raw_decision.get("cancer_detected", False),
        "cancer_signal_score": ssi_score,
        "screening_index": ssi_score,
        "ssi_score": ssi_score,
        "ssi_threshold": raw_decision.get("ssi_threshold", SSI_DECISION_CUTOFF),
        "model_probability_mean": raw_decision.get("model_probability_mean"),
        "model_probability_threshold": raw_decision.get("model_probability_threshold"),
        "decision_level": decision_level,
        "decision_policy": decision_policy,
        "cancer_signal_spectra_count": cancer_signal_count,
        "qc_valid_spectra_count": qc_valid_count,
        "cancer_type_prediction": result.get("cancer_type_prediction"),
        "cancer_type_confidence": result.get("cancer_type_confidence"),
        "cancer_type_probabilities": result.get("cancer_type_probabilities", {}),
        "per_replicate": per_replicate,
        "patient_decision": raw_decision,
    }


def get_signal_counts(result: Mapping) -> tuple[int | None, int | None]:
    cancer_count = result.get("cancer_signal_spectra_count")
    qc_count = result.get("qc_valid_spectra_count")
    decision = result.get("patient_decision", {})
    if cancer_count is None:
        cancer_count = decision.get("cancer_signal_spectra_count")
    if qc_count is None:
        qc_count = decision.get("qc_valid_spectra_count")
    if cancer_count is not None and qc_count is not None:
        return int(cancer_count), int(qc_count)
    majority_vote = decision.get("majority_vote") or result.get("majority_vote")
    if isinstance(majority_vote, str) and "/" in majority_vote:
        numerator, denominator = majority_vote.split("/", 1)
        if numerator.isdigit() and denominator.isdigit():
            return int(numerator), int(denominator)
    return None, None
