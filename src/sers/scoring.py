"""
SERS Screening Index (SSI) and Cancer Type Index (CTI) scoring module.

Transforms raw model probabilities into clinically interpretable 0-10 indices,
analogous to OVA1 (ovarian cancer) multivariate index assay.

SSI uses a piecewise linear transformation anchored at the operating mode
threshold, which always maps to SSI = 4.0 (the positive/negative cutoff).
"""

SSI_CUTOFF = 4.0
SSI_MAX = 10.0

RISK_LEVELS = {
    "LOW":          (0.0, 2.0, "#059669", "#f0fdf4"),
    "LOW_MODERATE": (2.0, 4.0, "#84cc16", "#f7fee7"),
    "MODERATE":     (4.0, 6.0, "#d97706", "#fffbeb"),
    "HIGH":         (6.0, 8.0, "#ea580c", "#fff7ed"),
    "VERY_HIGH":    (8.0, 10.0, "#dc2626", "#fef2f2"),
}


def probability_to_ssi(prob: float, threshold: float, cutoff: float = SSI_CUTOFF) -> float:
    """Convert cancer probability [0,1] to SSI [0,10] via piecewise linear transform.

    The operating mode threshold always maps to the cutoff value (default 4.0).
    Below threshold: linear from 0→cutoff. Above threshold: linear from cutoff→10.

    Properties:
        - Monotonically increasing
        - Invertible (see ssi_to_probability)
        - Rank-preserving: AUC, Sens, Spec unchanged
    """
    prob = max(0.0, min(1.0, prob))
    if threshold <= 0 or threshold >= 1:
        return prob * SSI_MAX

    if prob <= threshold:
        return cutoff * (prob / threshold)
    else:
        return cutoff + (SSI_MAX - cutoff) * (prob - threshold) / (1.0 - threshold)


def ssi_to_probability(ssi: float, threshold: float, cutoff: float = SSI_CUTOFF) -> float:
    """Inverse transform: SSI [0,10] back to probability [0,1]."""
    ssi = max(0.0, min(SSI_MAX, ssi))
    if threshold <= 0 or threshold >= 1:
        return ssi / SSI_MAX

    if ssi <= cutoff:
        return threshold * (ssi / cutoff)
    else:
        return threshold + (1.0 - threshold) * (ssi - cutoff) / (SSI_MAX - cutoff)


def probability_to_cti(prob: float) -> float:
    """Convert softmax probability to Cancer Type Index (CTI) [0,10]."""
    return round(max(0.0, min(SSI_MAX, prob * SSI_MAX)), 1)


def compute_cti_scores(type_probs: dict[str, float]) -> dict[str, float]:
    """Convert all cancer type probabilities to CTI scores."""
    return {name: probability_to_cti(p) for name, p in type_probs.items()}


def ssi_risk_level(ssi: float) -> tuple[str, str, str]:
    """Return (level_name, text_color_hex, bg_color_hex) for an SSI value."""
    for name, (lo, hi, color, bg) in RISK_LEVELS.items():
        if ssi < hi or name == "VERY_HIGH":
            return name, color, bg
    return "VERY_HIGH", "#dc2626", "#fef2f2"


def format_ssi(ssi: float) -> str:
    """Format SSI to one decimal place."""
    return f"{ssi:.1f}"
