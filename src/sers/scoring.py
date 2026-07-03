"""
SERS Screening Index (SSI) and Cancer Type Index (CTI) scoring module.

Transforms raw model probabilities into clinically interpretable 0-10 indices,
analogous to OVA1 (ovarian cancer) multivariate index assay.

SSI uses a piecewise linear transformation anchored at the operating mode
threshold, which always maps to SSI = 4.0 (the positive/negative cutoff).
"""

SSI_CUTOFF = 4.0
SSI_MAX = 10.0

# 3-tier risk stratification, validated against STK-V2 nested 5-fold OOF CV
# (n=1628; scripts/analysis/validate_ssi_risk_bands.py). Boundaries:
#   - 1.0: data-driven split (CART, min_leaf=30) separating the near-zero-risk
#     tail from an elevated-risk population, both still below the decision cutoff.
#   - 4.0: SSI_CUTOFF, i.e. the fixed "balanced" decision threshold — HIGH is
#     exactly the population flagged cancer_detected=True.
# Observed cancer rate: LOW 1.8% (n=388), MODERATE 46.0% (n=50, 95% CI
# 32.7-59.7%), HIGH 98.2% (n=1190). MODERATE sits below the binary decision
# cutoff but at materially elevated risk — flag this band's smaller N when
# displaying it.
RISK_LEVELS = {
    "LOW":      (0.0, 1.0, "#059669", "#f0fdf4"),
    "MODERATE": (1.0, 4.0, "#d97706", "#fffbeb"),
    "HIGH":     (4.0, 10.0, "#dc2626", "#fef2f2"),
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
        if ssi < hi or name == "HIGH":
            return name, color, bg
    return "HIGH", "#dc2626", "#fef2f2"


def format_ssi(ssi: float) -> str:
    """Format SSI to one decimal place."""
    return f"{ssi:.1f}"
