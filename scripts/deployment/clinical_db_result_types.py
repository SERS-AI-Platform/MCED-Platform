from __future__ import annotations

from typing import TypedDict

from sers.master_data.lineage_privacy import JsonValue


class PatientDecisionResult(TypedDict, total=False):
    decision_level: str
    decision_policy: str


class ClinicalPredictionResult(TypedDict, total=False):
    cancer_detected: bool
    decision_level: str
    decision_policy: str
    screening_index: float
    cancer_signal_score: float
    majority_vote: str | None
    cancer_type_prediction: str | None
    cancer_type_confidence: float | None
    cancer_type_probabilities: dict[str, float]
    per_replicate: list[JsonValue]
    patient_decision: PatientDecisionResult
