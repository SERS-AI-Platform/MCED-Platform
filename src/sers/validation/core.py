"""Input data validation for SERS spectra.

Provides structured validation of spectral data at various stages:
- Raw CSV structure validation
- Spectral value checks (NaN, Inf, range)
- Duplicate detection
- Post-preprocessing sanity checks

All validators return ``ValidationResult`` dataclasses so that callers
can decide how to handle issues (log, raise, skip) without being
forced into a single error-handling strategy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_WAVENUMBER_MIN = 400.0   # cm⁻¹
EXPECTED_WAVENUMBER_MAX = 2200.0  # cm⁻¹
WAVENUMBER_TOLERANCE = 50.0       # allow slight overshoot


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class ValidationIssue:
    """A single validation problem."""

    level: str          # "error" or "warning"
    code: str           # machine-readable identifier, e.g. "nan_values"
    message: str        # human-readable description
    context: dict = field(default_factory=dict)  # extra info (row idx, key, …)


@dataclass
class ValidationResult:
    """Aggregated result of one or more validation checks."""

    issues: List[ValidationIssue] = field(default_factory=list)

    # ------------------------------------------------------------------
    @property
    def is_valid(self) -> bool:
        """True when there are no *error*-level issues."""
        return not any(i.level == "error" for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.level == "warning" for i in self.issues)

    @property
    def errors(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def warnings(self) -> List[ValidationIssue]:
        return [i for i in self.issues if i.level == "warning"]

    def add(self, level: str, code: str, message: str, **ctx) -> None:
        self.issues.append(ValidationIssue(level=level, code=code, message=message, context=ctx))

    def merge(self, other: "ValidationResult") -> None:
        """Merge issues from *other* into this result."""
        self.issues.extend(other.issues)

    def summary(self) -> str:
        n_err = len(self.errors)
        n_warn = len(self.warnings)
        status = "PASS" if self.is_valid else "FAIL"
        lines = [f"Validation {status}: {n_err} error(s), {n_warn} warning(s)"]
        for issue in self.issues:
            lines.append(f"  [{issue.level.upper()}] {issue.code}: {issue.message}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# 1. CSV / DataFrame structure
# ---------------------------------------------------------------------------
def validate_csv_structure(
    df: pd.DataFrame,
    *,
    wavenumber_col: str = "raman_shift",
    intensity_col: str = "intensity",
) -> ValidationResult:
    """Validate that a two-column spectral DataFrame has the expected structure.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate (typically freshly read from CSV).
    wavenumber_col, intensity_col : str
        Expected column names.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    if df.empty:
        result.add("error", "empty_dataframe", "DataFrame is empty")
        return result

    if df.shape[1] < 2:
        result.add("error", "insufficient_columns",
                    f"Expected >=2 columns, got {df.shape[1]}")
        return result

    for col in (wavenumber_col, intensity_col):
        if col not in df.columns:
            result.add("warning", "missing_column",
                        f"Column '{col}' not found; columns are {list(df.columns)}",
                        expected=col, actual=list(df.columns))

    if len(df) < 10:
        result.add("warning", "too_few_rows",
                    f"Only {len(df)} rows — expected hundreds of spectral points")

    return result


# ---------------------------------------------------------------------------
# 2. Spectral value checks (single spectrum)
# ---------------------------------------------------------------------------
def validate_spectrum(
    x: np.ndarray,
    y: np.ndarray,
    *,
    check_intensity_nonneg: bool = False,
    label: str = "",
) -> ValidationResult:
    """Validate a single (wavenumber, intensity) spectrum pair.

    Parameters
    ----------
    x : np.ndarray
        Wavenumber values (cm⁻¹).
    y : np.ndarray
        Intensity values.
    check_intensity_nonneg : bool
        If True, flag negative intensities as errors (useful post-baseline).
    label : str
        Optional identifier for log messages.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    ctx = {"label": label} if label else {}

    # --- shape ---
    if x.shape != y.shape:
        result.add("error", "shape_mismatch",
                    f"x shape {x.shape} != y shape {y.shape}", **ctx)
        return result  # further checks meaningless

    if len(x) == 0:
        result.add("error", "empty_spectrum", "Spectrum has 0 points", **ctx)
        return result

    # --- NaN / Inf ---
    n_nan_x = int(np.isnan(x).sum())
    n_nan_y = int(np.isnan(y).sum())
    n_inf_x = int(np.isinf(x).sum())
    n_inf_y = int(np.isinf(y).sum())

    if n_nan_x > 0:
        result.add("error", "nan_wavenumber",
                    f"{n_nan_x} NaN value(s) in wavenumber array", **ctx)
    if n_nan_y > 0:
        result.add("error", "nan_intensity",
                    f"{n_nan_y} NaN value(s) in intensity array", **ctx)
    if n_inf_x > 0:
        result.add("error", "inf_wavenumber",
                    f"{n_inf_x} Inf value(s) in wavenumber array", **ctx)
    if n_inf_y > 0:
        result.add("error", "inf_intensity",
                    f"{n_inf_y} Inf value(s) in intensity array", **ctx)

    # Stop early if there are NaN/Inf (min/max would fail)
    if not result.is_valid:
        return result

    # --- wavenumber range ---
    x_min, x_max = float(x.min()), float(x.max())
    lo = EXPECTED_WAVENUMBER_MIN - WAVENUMBER_TOLERANCE
    hi = EXPECTED_WAVENUMBER_MAX + WAVENUMBER_TOLERANCE

    if x_min < lo:
        result.add("warning", "wavenumber_below_range",
                    f"Wavenumber min {x_min:.1f} < expected {EXPECTED_WAVENUMBER_MIN} cm⁻¹",
                    x_min=x_min, **ctx)
    if x_max > hi:
        result.add("warning", "wavenumber_above_range",
                    f"Wavenumber max {x_max:.1f} > expected {EXPECTED_WAVENUMBER_MAX} cm⁻¹",
                    x_max=x_max, **ctx)

    # --- intensity non-negative (post-preprocessing) ---
    if check_intensity_nonneg:
        n_neg = int((y < -1e-10).sum())
        if n_neg > 0:
            result.add("warning", "negative_intensity",
                        f"{n_neg} negative intensity value(s) (min={float(y.min()):.4f})",
                        n_negative=n_neg, **ctx)

    # --- constant spectrum (dead channel / flat signal) ---
    if np.ptp(y) < 1e-10:
        result.add("warning", "flat_spectrum",
                    "Intensity has zero range — possibly dead channel", **ctx)

    return result


# ---------------------------------------------------------------------------
# 3. Batch-level checks
# ---------------------------------------------------------------------------
def validate_spectra_batch(
    spectra: Dict[tuple, Tuple[np.ndarray, np.ndarray]],
    *,
    check_intensity_nonneg: bool = False,
) -> ValidationResult:
    """Validate a dict of raw spectra (as returned by ``load_dataset``).

    Runs per-spectrum checks and also detects duplicates.

    Parameters
    ----------
    spectra : dict
        Keys are tuple identifiers, values are (x, y) pairs.
    check_intensity_nonneg : bool
        Passed through to ``validate_spectrum``.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    if not spectra:
        result.add("error", "empty_batch", "No spectra provided")
        return result

    for key, (x, y) in spectra.items():
        label = str(key)
        per_spec = validate_spectrum(
            x, y,
            check_intensity_nonneg=check_intensity_nonneg,
            label=label,
        )
        result.merge(per_spec)

    # --- duplicate detection ---
    dup_result = detect_duplicate_samples(spectra)
    result.merge(dup_result)

    return result


def detect_duplicate_samples(
    spectra: Dict[tuple, Tuple[np.ndarray, np.ndarray]],
    *,
    correlation_threshold: float = 0.9999,
) -> ValidationResult:
    """Detect likely duplicate spectra by pairwise correlation.

    To keep this lightweight, only checks spectra within the same group
    (first element of the key tuple).  Exact-duplicate keys are always
    flagged regardless of correlation.

    Parameters
    ----------
    spectra : dict
        Keys are tuple identifiers, values are (x, y) pairs.
    correlation_threshold : float
        Pearson r above which two spectra are flagged as duplicates.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()

    # Group keys by group label (first element)
    from collections import defaultdict
    groups: Dict[str, list] = defaultdict(list)
    for key in spectra:
        groups[key[0]].append(key)

    for group, keys in groups.items():
        if len(keys) < 2:
            continue

        # Compare intensities only (they should already share same x grid
        # within a loaded batch, but we handle varying lengths gracefully)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                _, yi = spectra[keys[i]]
                _, yj = spectra[keys[j]]

                if len(yi) != len(yj):
                    continue  # different grids — not duplicates

                if len(yi) == 0:
                    continue

                corr = np.corrcoef(yi, yj)[0, 1]
                if np.isnan(corr):
                    continue

                if corr >= correlation_threshold:
                    result.add(
                        "warning",
                        "duplicate_spectrum",
                        f"Spectra {keys[i]} and {keys[j]} are near-identical "
                        f"(r={corr:.6f})",
                        key_a=keys[i],
                        key_b=keys[j],
                        correlation=float(corr),
                    )

    return result


# ---------------------------------------------------------------------------
# 4. Post-preprocessing validation
# ---------------------------------------------------------------------------
def validate_processed_spectrum(
    y: np.ndarray,
    grid: np.ndarray,
    *,
    label: str = "",
) -> ValidationResult:
    """Quick sanity check on a preprocessed spectrum.

    Parameters
    ----------
    y : np.ndarray
        Preprocessed intensity values (on common grid).
    grid : np.ndarray
        The common wavenumber grid.
    label : str
        Optional identifier.

    Returns
    -------
    ValidationResult
    """
    result = ValidationResult()
    ctx = {"label": label} if label else {}

    if len(y) != len(grid):
        result.add("error", "grid_length_mismatch",
                    f"Spectrum length {len(y)} != grid length {len(grid)}", **ctx)
        return result

    n_nan = int(np.isnan(y).sum())
    n_inf = int(np.isinf(y).sum())
    if n_nan > 0:
        result.add("error", "nan_after_preprocessing",
                    f"{n_nan} NaN(s) in processed spectrum", **ctx)
    if n_inf > 0:
        result.add("error", "inf_after_preprocessing",
                    f"{n_inf} Inf(s) in processed spectrum", **ctx)

    if np.ptp(y) < 1e-12:
        result.add("warning", "flat_after_preprocessing",
                    "Processed spectrum is effectively constant", **ctx)

    return result
