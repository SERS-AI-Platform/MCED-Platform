"""
Clinical decision display helpers for the usability-test build.
"""

from __future__ import annotations

from src.sers.scoring import ssi_risk_level

SSI_SCALE = 10.0

# SSI risk band N per src/sers/scoring.py validation (STK-V2 nested OOF CV,
# scripts/analysis/validate_ssi_risk_bands.py). MODERATE is a small cohort —
# flagged so the UI can show a caveat.
RISK_BAND_N = {"LOW": 388, "MODERATE": 50, "HIGH": 1190}
RISK_BAND_RANGES = {
    "LOW": "SSI 0.0-1.0",
    "MODERATE": "SSI 1.0-4.0",
    "HIGH": "SSI 4.0-10.0",
}
RISK_BAND_OBSERVED_CANCER_RATE = {
    "LOW": 1.8,
    "MODERATE": 46.0,
    "HIGH": 98.2,
}
RISK_BAND_SMALL_N_THRESHOLD = 100


def ssi_risk_info(ssi: float) -> dict:
    """Return risk-band info for an SSI value: level, colors, and small-N flag."""
    level, color, bg = ssi_risk_level(ssi)
    n = RISK_BAND_N.get(level)
    return {
        "level": level,
        "color": color,
        "bg": bg,
        "n": n,
        "range_label": RISK_BAND_RANGES.get(level),
        "observed_cancer_rate": RISK_BAND_OBSERVED_CANCER_RATE.get(level),
        "small_n": n is not None and n < RISK_BAND_SMALL_N_THRESHOLD,
    }


def screening_index_to_ssi(screening_index: float | None) -> float:
    """Return SSI on the 0-10 display scale.

    Older saved rows may store a 0-1 probability-like screening index, while
    current predictions store SSI directly.
    """
    value = float(screening_index or 0.0)
    if value <= 1.0:
        return round(min(max(value, 0.0), 1.0) * SSI_SCALE, 1)
    return round(min(max(value, 0.0), SSI_SCALE), 1)


def final_decision(prediction: dict, qc_valid: bool) -> str:
    """Return final display decision: invalid, positive, or negative."""
    if not qc_valid:
        return "invalid"
    if not prediction.get("cancer_detected"):
        return "negative"
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
