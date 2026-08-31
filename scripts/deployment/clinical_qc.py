"""
QC summary helpers for the clinical usability-test workflow.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict


class QcSummary(TypedDict):
    total: int
    passed: int
    failed: int
    pass_rate: float
    min_valid_count: int
    valid: bool
    has_critical_failure: bool
    critical_failure_count: int
    reason_counts: dict[str, int]

REASON_KEYS = (
    "low_intensity",
    "spike_noise",
    "saturation",
    "low_correlation",
    "other",
)
CRITICAL_REASON_KEYS = frozenset({"low_intensity", "spike_noise", "saturation"})

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
            return [raw] if raw else []
        return [raw] if raw else []
    return [str(raw)]


def build_qc_summary(
    spectra: Sequence[Mapping[str, Any]], min_valid_count: int
) -> QcSummary:
    """Build display/report QC summary from persisted spectrum rows."""
    total = len(spectra)
    passed = sum(1 for sp in spectra if sp.get("qc_pass"))
    failed = total - passed
    pass_rate = (passed / total * 100.0) if total else 0.0
    reason_counts = Counter({key: 0 for key in REASON_KEYS})
    for sp in spectra:
        flags = _load_flags(sp.get("qc_flags"))
        if sp.get("qc_pass"):
            for flag in flags:
                reason = classify_qc_flag(flag)
                if reason in CRITICAL_REASON_KEYS:
                    reason_counts[reason] += 1
            continue
        if not flags:
            reason_counts["other"] += 1
            continue
        for flag in flags:
            reason_counts[classify_qc_flag(flag)] += 1

    critical_failure_count = sum(reason_counts[key] for key in CRITICAL_REASON_KEYS)
    has_critical_failure = critical_failure_count > 0
    valid = passed >= min_valid_count and not has_critical_failure

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "min_valid_count": min_valid_count,
        "valid": valid,
        "has_critical_failure": has_critical_failure,
        "critical_failure_count": critical_failure_count,
        "reason_counts": dict(reason_counts),
    }
