"""
QC summary helpers for the clinical usability-test workflow.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


REASON_KEYS = (
    "low_intensity",
    "spike_noise",
    "saturation",
    "low_correlation",
    "other",
)

DEFAULT_MIN_VALID_COUNT = 3


def classify_qc_flag(flag: str) -> str:
    """Map raw QC flag text to user-facing failure reason categories."""
    text = (flag or "").lower()
    if "intensity" in text or "fp_mean" in text or "low_signal" in text:
        return "low_intensity"
    if "spike" in text or "cosmic" in text:
        return "spike_noise"
    if "saturat" in text or "overflow" in text:
        return "saturation"
    if "correlation" in text or "corr" in text:
        return "low_correlation"
    return "other"


def _load_flags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(v) for v in raw]
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v) for v in parsed]
        except json.JSONDecodeError:
            pass
        return [raw] if raw else []
    return [str(raw)]


def build_qc_summary(spectra: list[dict], min_valid_count: int) -> dict:
    """Build display/report QC summary from persisted spectrum rows."""
    total = len(spectra)
    passed = sum(1 for sp in spectra if sp.get("qc_pass"))
    failed = total - passed
    pass_rate = (passed / total * 100.0) if total else 0.0
    valid = passed >= min_valid_count

    reason_counts = Counter({key: 0 for key in REASON_KEYS})
    for sp in spectra:
        if sp.get("qc_pass"):
            continue
        flags = _load_flags(sp.get("qc_flags"))
        if not flags:
            reason_counts["other"] += 1
            continue
        for flag in flags:
            reason_counts[classify_qc_flag(flag)] += 1

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "min_valid_count": min_valid_count,
        "valid": valid,
        "reason_counts": dict(reason_counts),
    }
