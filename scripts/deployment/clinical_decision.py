"""
Clinical decision display helpers for the usability-test build.
"""

from __future__ import annotations


SSI_SCALE = 10.0


def screening_index_to_ssi(screening_index: float | None) -> float:
    """Convert patient-level cancer probability to the 0-10 SSI display scale."""
    value = float(screening_index or 0.0)
    value = min(max(value, 0.0), 1.0)
    return round(value * SSI_SCALE, 1)


def final_decision(prediction: dict, qc_valid: bool) -> str:
    """Return final display decision: invalid, strong_positive, positive, or negative."""
    if not qc_valid:
        return "invalid"
    if not prediction.get("cancer_detected"):
        return "negative"
    ssi = screening_index_to_ssi(prediction.get("screening_index"))
    majority_vote = str(prediction.get("majority_vote") or "")
    if ssi >= 7.0 and majority_vote.startswith(("4/", "5/")):
        return "strong_positive"
    return "positive"


def type_confidence(cancer_types_sorted: list[dict]) -> dict:
    """Classify top cancer-type confidence using top probability and top-2 gap."""
    if not cancer_types_sorted:
        return {
            "level": None,
            "top": None,
            "second": None,
            "gap": None,
        }

    top = cancer_types_sorted[0]
    second = cancer_types_sorted[1] if len(cancer_types_sorted) > 1 else {"code": None, "prob": 0.0}
    top_prob = float(top.get("prob") or 0.0)
    second_prob = float(second.get("prob") or 0.0)
    gap = top_prob - second_prob

    if top_prob >= 0.50 or gap >= 0.20:
        level = "high"
    elif top_prob >= 0.30 or gap >= 0.10:
        level = "medium"
    else:
        level = "low"

    return {
        "level": level,
        "top": top,
        "second": second,
        "gap": gap,
    }
