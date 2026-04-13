"""
Quality control functions for SERS spectroscopy.

QC Pipeline (3 levels):
    Level 0: Intensity Gate       → "Did SERS enhancement work?"
    Level 1: Replicate RSD        → "Is intensity reproducible?"
    Level 1: Replicate Correlation → "Is spectral shape consistent?"

QC Philosophy:
    SERS urine spectra are metabolite superpositions — not isolated peaks.
    Traditional SNR/SBR metrics FAIL for biofluid SERS because:
      - Higher-quality instruments resolve MORE substrate features
      - This paradoxically DECREASES SNR/SBR metrics
      - Replicate variance captures true measurement noise better

    Our approach:
      - Intensity Gate: adaptive threshold catches SERS enhancement failure
      - Replicate RSD: measures intensity reproducibility (normalized by max)
      - Replicate Correlation: measures spectral shape consistency

QC Thresholds (from QCConfig):
    - Intensity gate ratio: 0.1 (fp_mean < median × 0.1 → enhancement failure)
    - Replicate RSD < 5%: Good measurement reproducibility
    - Replicate Correlation > 0.95: Consistent spectral patterns

Reference:
    - KIMS (재료연) standard: Fingerprint region 400-2200 cm⁻¹

Changelog:
    v0.5.0 (2026-02) - Add intensity gate, remove SNR/SBR
    v0.4.0 (2026-02) - Replicate-only QC (deprecated SNR)
    v0.1.0 (2025-01) - Initial release with SNR
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

from sers.logging_config import setup_logging
setup_logging()

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.spatial.distance import pdist, squareform

if TYPE_CHECKING:
    from ..config import QCConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Default Thresholds (fallback when QCConfig not provided)
# =============================================================================
DEFAULT_RSD_THRESHOLD = 5.0
DEFAULT_CORR_THRESHOLD = 0.95
DEFAULT_INTENSITY_GATE_RATIO = 0.1
FINGERPRINT_REGION = (400.0, 2200.0)


# =============================================================================
# Helpers
# =============================================================================
def _interpolate_to_grid(
    x: np.ndarray,
    y: np.ndarray,
    common_grid: np.ndarray,
) -> np.ndarray:
    """Interpolate a single spectrum onto a common wavenumber grid.

    Parameters
    ----------
    x : np.ndarray
        Original wavenumber values.
    y : np.ndarray
        Original intensity values.
    common_grid : np.ndarray
        Target wavenumber grid.

    Returns
    -------
    np.ndarray
        Interpolated intensity values on common_grid.
    """
    f = interp1d(x, y, kind="linear", bounds_error=False, fill_value="extrapolate")
    return f(common_grid)


def _group_spectra_by_sample(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    common_grid: np.ndarray,
) -> Dict[Tuple[str, str], List[Tuple[str, np.ndarray]]]:
    """Group spectra by (group, sample_id), interpolating to common grid.

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    common_grid : np.ndarray
        Common wavenumber grid.

    Returns
    -------
    dict
        Keys (group, sample_id) → list of (replicate, y_interpolated)
    """
    samples: Dict[Tuple[str, str], List[Tuple[str, np.ndarray]]] = {}
    for (group, sid, rep), (x, y) in spectra.items():
        key = (group, sid)
        y_interp = _interpolate_to_grid(x, y, common_grid)
        samples.setdefault(key, []).append((rep, y_interp))
    return samples


# =============================================================================
# Level 0: Intensity Gate
# =============================================================================
def calculate_intensity_gate(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    fingerprint_region: Tuple[float, float] = FINGERPRINT_REGION,
    intensity_gate_ratio: float = DEFAULT_INTENSITY_GATE_RATIO,
) -> pd.DataFrame:
    """Calculate intensity gate for each spectrum.

    The intensity gate is an adaptive threshold that catches catastrophic
    SERS enhancement failures.  For each spectrum, the mean intensity in
    the fingerprint region (``fp_mean``) is computed.  The global median
    of all ``fp_mean`` values defines the reference.  Spectra with
    ``fp_mean < median × intensity_gate_ratio`` are flagged.

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    fingerprint_region : tuple of float
        (min, max) wavenumber for analysis.
    intensity_gate_ratio : float
        Fraction of median below which spectra are flagged (default 0.1).

    Returns
    -------
    pd.DataFrame
        Per-spectrum gate results:
        - group, sample_id, replicate
        - fp_mean: mean intensity in fingerprint region
        - gate_threshold: adaptive threshold (median × ratio)
        - gate_pass: bool

    Notes
    -----
    Why 0.1?
        - Normal equipment variation: fp_mean / median ≈ 0.45–0.56
        - Enhancement failure: fp_mean / median < 0.1
        - ratio=0.1 catches catastrophic failures without flagging
          normal measurement variation.
    """
    results = []
    fp_means = []

    for (group, sid, rep), (x, y) in spectra.items():
        mask = (x >= fingerprint_region[0]) & (x <= fingerprint_region[1])
        if mask.any():
            fp_mean = float(np.mean(np.abs(y[mask])))
        else:
            fp_mean = float(np.mean(np.abs(y)))
            logger.warning(
                f"No data in fingerprint region for {group}-{sid}-{rep}, "
                f"using full spectrum"
            )
        fp_means.append(fp_mean)
        results.append({
            "group": group,
            "sample_id": sid,
            "replicate": rep,
            "fp_mean": fp_mean,
        })

    # Adaptive threshold: median × ratio
    median_fp = float(np.median(fp_means)) if fp_means else 0.0
    threshold = median_fp * intensity_gate_ratio

    for row in results:
        row["gate_threshold"] = threshold
        row["gate_pass"] = row["fp_mean"] >= threshold

    df = pd.DataFrame(results)
    n_fail = int((~df["gate_pass"]).sum())
    logger.info(
        f"Intensity gate: {n_fail}/{len(df)} spectra failed "
        f"(threshold={threshold:.2f}, median={median_fp:.2f}, "
        f"ratio={intensity_gate_ratio})"
    )
    return df


def filter_by_intensity_gate(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    gate_df: Optional[pd.DataFrame] = None,
    fingerprint_region: Tuple[float, float] = FINGERPRINT_REGION,
    intensity_gate_ratio: float = DEFAULT_INTENSITY_GATE_RATIO,
) -> Tuple[Dict, List[Tuple]]:
    """Remove spectra that fail the intensity gate.

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    gate_df : pd.DataFrame, optional
        Pre-computed gate results from calculate_intensity_gate().
        If None, computed internally.
    fingerprint_region : tuple of float
    intensity_gate_ratio : float

    Returns
    -------
    filtered : dict
        Spectra that passed the gate.
    rejected : list of tuple
        Keys of rejected spectra.
    """
    if gate_df is None:
        gate_df = calculate_intensity_gate(
            spectra, fingerprint_region, intensity_gate_ratio
        )

    # Build set of passing keys
    passing_keys = set()
    for _, row in gate_df.iterrows():
        if row["gate_pass"]:
            passing_keys.add((row["group"], row["sample_id"], row["replicate"]))

    filtered = {}
    rejected = []
    for key, value in spectra.items():
        if key in passing_keys:
            filtered[key] = value
        else:
            rejected.append(key)

    logger.info(
        f"Intensity gate filter: kept {len(filtered)}/{len(spectra)} "
        f"({len(rejected)} rejected)"
    )
    return filtered, rejected


# =============================================================================
# Level 1: Replicate Reproducibility
# =============================================================================
def calculate_replicate_qc(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    common_grid: np.ndarray,
    fingerprint_region: Tuple[float, float] = FINGERPRINT_REGION,
) -> pd.DataFrame:
    """Calculate replicate reproducibility metrics for each sample.

    Measures: "Are repeated measurements consistent?"

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    common_grid : np.ndarray
        Common wavenumber grid for interpolation.
    fingerprint_region : tuple of float
        Region for focused RSD analysis (default: 400–2200 cm⁻¹).

    Returns
    -------
    pd.DataFrame
        Per-sample QC statistics:
        - group, sample_id, n_reps
        - mean_rsd, median_rsd, max_rsd, rsd_fingerprint
        - mean_corr, min_corr

    Notes
    -----
    RSD (Relative Standard Deviation) = (std / max_intensity) × 100

    Why RSD instead of CV?
        Baseline-corrected spectra have regions near zero or negative.
        Traditional CV (std/mean) explodes when mean ≈ 0.
        RSD normalises by max intensity, staying stable across all regions.

    Correlation = Pearson correlation between all replicate pairs.
    """
    samples = _group_spectra_by_sample(spectra, common_grid)
    fp_mask = (common_grid >= fingerprint_region[0]) & (
        common_grid <= fingerprint_region[1]
    )

    stats = []
    for (group, sid), replicate_list in samples.items():
        n_reps = len(replicate_list)
        if n_reps < 2:
            logger.warning(f"Sample {group}-{sid}: {n_reps} replicate(s), skipping")
            continue

        spectra_arr = [y for _, y in replicate_list]
        matrix = np.vstack(spectra_arr)

        # RSD
        mean_spec = matrix.mean(axis=0)
        std_spec = matrix.std(axis=0, ddof=1)
        max_intensity = mean_spec.max()

        if max_intensity > 1e-10:
            rsd = (std_spec / max_intensity) * 100
        else:
            rsd = np.zeros_like(std_spec)
            logger.warning(f"Sample {group}-{sid}: near-zero max intensity")

        rsd_fp = float(rsd[fp_mask].mean()) if fp_mask.any() else float(rsd.mean())

        # Pairwise correlations
        correlations = []
        for i in range(n_reps):
            for j in range(i + 1, n_reps):
                corr = np.corrcoef(spectra_arr[i], spectra_arr[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

        mean_corr = float(np.mean(correlations)) if correlations else np.nan
        min_corr = float(np.min(correlations)) if correlations else np.nan

        stats.append({
            "group": group,
            "sample_id": sid,
            "n_reps": n_reps,
            "mean_rsd": float(rsd.mean()),
            "median_rsd": float(np.median(rsd)),
            "max_rsd": float(rsd.max()),
            "rsd_fingerprint": rsd_fp,
            "mean_corr": mean_corr,
            "min_corr": min_corr,
        })

    df = pd.DataFrame(stats)
    if len(df) > 0:
        logger.info(
            f"Replicate QC: {len(df)} samples | "
            f"RSD={df['mean_rsd'].mean():.2f}% | "
            f"Corr={df['mean_corr'].mean():.4f}"
        )
    return df


# =============================================================================
# Failure Identification
# =============================================================================
def identify_qc_failures(
    qc_stats: pd.DataFrame,
    rsd_threshold: float = DEFAULT_RSD_THRESHOLD,
    corr_threshold: float = DEFAULT_CORR_THRESHOLD,
) -> pd.DataFrame:
    """Identify samples that fail QC criteria.

    Parameters
    ----------
    qc_stats : pd.DataFrame
        Output from calculate_replicate_qc().
    rsd_threshold : float
        Maximum acceptable mean RSD (%), default 5.0.
    corr_threshold : float
        Minimum acceptable correlation, default 0.95.

    Returns
    -------
    pd.DataFrame
        Failed samples with ``failure_reason`` column.
    """
    if len(qc_stats) == 0:
        return pd.DataFrame()

    conditions = pd.Series(False, index=qc_stats.index)

    if "mean_rsd" in qc_stats.columns:
        conditions |= qc_stats["mean_rsd"] > rsd_threshold
    if "mean_corr" in qc_stats.columns:
        conditions |= qc_stats["mean_corr"] < corr_threshold

    failed = qc_stats[conditions].copy()

    if len(failed) > 0:
        def _reason(row):
            reasons = []
            if "mean_rsd" in row and row["mean_rsd"] > rsd_threshold:
                reasons.append(f"High RSD ({row['mean_rsd']:.1f}% > {rsd_threshold}%)")
            if "mean_corr" in row and row["mean_corr"] < corr_threshold:
                reasons.append(
                    f"Low corr ({row['mean_corr']:.3f} < {corr_threshold})"
                )
            return "; ".join(reasons) if reasons else "Unknown"

        failed["failure_reason"] = failed.apply(_reason, axis=1)

    logger.info(
        f"QC failures: {len(failed)}/{len(qc_stats)} samples "
        f"({100 * len(failed) / max(len(qc_stats), 1):.1f}%)"
    )
    return failed


def summarize_qc_by_group(qc_stats: pd.DataFrame) -> pd.DataFrame:
    """Summarize QC statistics by group.

    Parameters
    ----------
    qc_stats : pd.DataFrame
        Output from calculate_replicate_qc().

    Returns
    -------
    pd.DataFrame
        Group-level summary with count, mean, std, min/max of key metrics.
    """
    if len(qc_stats) == 0:
        return pd.DataFrame()

    agg_dict = {"sample_id": "count"}
    if "mean_rsd" in qc_stats.columns:
        agg_dict["mean_rsd"] = ["mean", "std", "max"]
    if "mean_corr" in qc_stats.columns:
        agg_dict["mean_corr"] = ["mean", "std", "min"]

    summary = qc_stats.groupby("group").agg(agg_dict).round(3)
    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.rename(columns={"sample_id_count": "n_samples"})
    return summary


# =============================================================================
# Analysis: Variance Convergence
# =============================================================================
def calculate_variance_convergence(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    common_grid: np.ndarray,
    max_reps: Optional[int] = None,
) -> pd.DataFrame:
    """Analyse how variance converges with increasing replicate count.

    Answers: "How many replicates do we need?"

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    common_grid : np.ndarray
    max_reps : int, optional
        Maximum replicates to analyse (default: all available).

    Returns
    -------
    pd.DataFrame
        Columns: n_reps, mean_rsd, std_rsd, mean_corr, n_samples
    """
    samples = _group_spectra_by_sample(spectra, common_grid)
    # Flatten to lists only
    sample_spectra = {k: [y for _, y in v] for k, v in samples.items()}

    max_available = max(len(v) for v in sample_spectra.values()) if sample_spectra else 0
    if max_reps is None:
        max_reps = max_available
    else:
        max_reps = min(max_reps, max_available)

    results = []
    for n in range(2, max_reps + 1):
        rsds, corrs = [], []
        for spec_list in sample_spectra.values():
            if len(spec_list) < n:
                continue
            subset = spec_list[:n]
            matrix = np.vstack(subset)
            mean_spec = matrix.mean(axis=0)
            std_spec = matrix.std(axis=0, ddof=1)
            max_i = mean_spec.max()
            if max_i > 1e-10:
                rsds.append(float((std_spec / max_i * 100).mean()))
            for i in range(n):
                for j in range(i + 1, n):
                    c = np.corrcoef(subset[i], subset[j])[0, 1]
                    if not np.isnan(c):
                        corrs.append(c)

        results.append({
            "n_reps": n,
            "mean_rsd": np.mean(rsds) if rsds else np.nan,
            "std_rsd": np.std(rsds) if rsds else np.nan,
            "mean_corr": np.mean(corrs) if corrs else np.nan,
            "n_samples": len(rsds),
        })
    return pd.DataFrame(results)


# =============================================================================
# Filtering: Correlation
# =============================================================================
def filter_by_correlation(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    common_grid: np.ndarray,
    min_correlation: float = 0.90,
    reference: str = "median",
) -> Tuple[Dict, List[Tuple]]:
    """Filter replicates by correlation with reference spectrum.

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    common_grid : np.ndarray
    min_correlation : float
    reference : str
        ``"median"`` or ``"mean"``

    Returns
    -------
    filtered : dict   — spectra passing threshold (original x, y)
    rejected : list    — keys of rejected spectra
    """
    # Group with originals preserved
    samples: Dict[Tuple, List[Tuple]] = {}
    for key, (x, y) in spectra.items():
        group, sid, rep = key
        sample_key = (group, sid)
        y_interp = _interpolate_to_grid(x, y, common_grid)
        samples.setdefault(sample_key, []).append((key, y_interp, (x, y)))

    filtered, rejected = {}, []
    for sample_key, replicate_list in samples.items():
        if len(replicate_list) < 2:
            key, _, orig = replicate_list[0]
            filtered[key] = orig
            continue

        y_matrix = np.vstack([yi for _, yi, _ in replicate_list])
        ref = (
            np.median(y_matrix, axis=0)
            if reference == "median"
            else np.mean(y_matrix, axis=0)
        )

        for key, y_interp, orig in replicate_list:
            corr = np.corrcoef(y_interp, ref)[0, 1]
            if corr >= min_correlation:
                filtered[key] = orig
            else:
                rejected.append(key)
                logger.debug(f"Rejected {key}: corr={corr:.3f} < {min_correlation}")

    logger.info(
        f"Correlation filter: kept {len(filtered)}/{len(spectra)} "
        f"({len(rejected)} rejected)"
    )
    return filtered, rejected


# =============================================================================
# Aggregation: Medoid Selection
# =============================================================================
def find_medoid(
    spectra_list: List[np.ndarray],
    metric: str = "correlation",
) -> int:
    """Find medoid (most representative) spectrum.

    The medoid minimises total distance to all other spectra.

    Parameters
    ----------
    spectra_list : list of np.ndarray
        Spectra of equal length.
    metric : str
        ``"correlation"``, ``"euclidean"``, or ``"cosine"``

    Returns
    -------
    int
        Index of medoid spectrum.
    """
    if len(spectra_list) <= 1:
        return 0

    n = len(spectra_list)
    if metric == "correlation":
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                c = np.corrcoef(spectra_list[i], spectra_list[j])[0, 1]
                d = 1 - c if not np.isnan(c) else 1.0
                dist[i, j] = d
                dist[j, i] = d
    else:
        dist = squareform(pdist(np.vstack(spectra_list), metric=metric))

    return int(np.argmin(dist.sum(axis=1)))


def select_medoid_spectra(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    common_grid: np.ndarray,
    metric: str = "correlation",
) -> Dict[Tuple[str, str], np.ndarray]:
    """Select medoid spectrum for each sample.

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    common_grid : np.ndarray
    metric : str

    Returns
    -------
    dict
        Keys (group, sample_id) → y_medoid on common_grid
    """
    samples = _group_spectra_by_sample(spectra, common_grid)
    medoids = {}

    for sample_key, replicate_list in samples.items():
        if len(replicate_list) == 1:
            medoids[sample_key] = replicate_list[0][1]
        else:
            spec_list = [y for _, y in replicate_list]
            idx = find_medoid(spec_list, metric=metric)
            medoids[sample_key] = spec_list[idx]
            logger.debug(
                f"Medoid for {sample_key}: replicate {replicate_list[idx][0]}"
            )

    logger.info(f"Selected {len(medoids)} medoid spectra")
    return medoids


# =============================================================================
# Detection: Outliers
# =============================================================================
def detect_outliers(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    method: str = "zscore",
    threshold: float = 3.0,
    feature: str = "mean_intensity",
) -> List[Tuple]:
    """Detect outlier spectra based on intensity statistics.

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    method : str
        ``"zscore"`` or ``"iqr"``
    threshold : float
        Z-score threshold (default 3.0) or IQR multiplier.
    feature : str
        ``"mean_intensity"`` or ``"max_intensity"``

    Returns
    -------
    list
        Keys of outlier spectra.
    """
    feature_values = {}
    for key, (x, y) in spectra.items():
        if feature == "mean_intensity":
            feature_values[key] = float(y.mean())
        elif feature == "max_intensity":
            feature_values[key] = float(y.max())
        else:
            raise ValueError(f"Unknown feature: {feature}. Use 'mean_intensity' or 'max_intensity'.")

    values = np.array(list(feature_values.values()))
    outliers = []

    if method == "zscore":
        mean, std = values.mean(), values.std()
        for key, val in feature_values.items():
            z = abs(val - mean) / std if std > 0 else 0
            if z > threshold:
                outliers.append(key)
                logger.debug(f"Outlier {key}: z={z:.2f}")

    elif method == "iqr":
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - threshold * iqr, q3 + threshold * iqr
        for key, val in feature_values.items():
            if val < lo or val > hi:
                outliers.append(key)
                logger.debug(f"Outlier {key}: value={val:.2e}")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'zscore' or 'iqr'.")

    logger.info(f"Detected {len(outliers)} outliers ({method} on {feature})")
    return outliers


# =============================================================================
# Full QC Pipeline
# =============================================================================
def run_qc_pipeline(
    spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    common_grid: np.ndarray,
    qc_config: Optional["QCConfig"] = None,
    *,
    rsd_threshold: Optional[float] = None,
    corr_threshold: Optional[float] = None,
    intensity_gate_ratio: Optional[float] = None,
    fingerprint_region: Optional[Tuple[float, float]] = None,
    save_report: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run complete QC pipeline: Intensity Gate → Replicate QC → Failures.

    Parameters
    ----------
    spectra : dict
        Keys (group, sample_id, replicate) → (x, y)
    common_grid : np.ndarray
    qc_config : QCConfig, optional
        Canonical source of thresholds.  Individual kwargs override.
    rsd_threshold : float, optional
    corr_threshold : float, optional
    intensity_gate_ratio : float, optional
    fingerprint_region : tuple of float, optional
    save_report : str, optional
        Path to save text report.

    Returns
    -------
    gate_df : pd.DataFrame
        Per-spectrum intensity gate results.
    qc_stats : pd.DataFrame
        Per-sample replicate QC statistics.
    failures : pd.DataFrame
        Samples that failed QC.
    group_summary : pd.DataFrame
        Group-level summary.
    """
    # Resolve thresholds: explicit kwargs → qc_config → defaults
    _rsd = rsd_threshold
    _corr = corr_threshold
    _gate = intensity_gate_ratio
    _fp = fingerprint_region

    if qc_config is not None:
        if _rsd is None:
            _rsd = qc_config.rsd_threshold
        if _corr is None:
            _corr = qc_config.corr_threshold
        if _gate is None:
            _gate = qc_config.intensity_gate_ratio
        if _fp is None:
            _fp = qc_config.fingerprint_region

    _rsd = _rsd if _rsd is not None else DEFAULT_RSD_THRESHOLD
    _corr = _corr if _corr is not None else DEFAULT_CORR_THRESHOLD
    _gate = _gate if _gate is not None else DEFAULT_INTENSITY_GATE_RATIO
    _fp = _fp if _fp is not None else FINGERPRINT_REGION

    logger.info("=" * 60)
    logger.info("Running QC Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Fingerprint region: {_fp} cm⁻¹")
    logger.info(f"  Intensity gate ratio: {_gate}")
    logger.info(f"  RSD threshold: < {_rsd}%")
    logger.info(f"  Correlation threshold: > {_corr}")
    logger.info(f"  Total spectra: {len(spectra)}")

    # --- Level 0: Intensity Gate ---
    gate_df = calculate_intensity_gate(spectra, _fp, _gate)
    n_gate_fail = int((~gate_df["gate_pass"]).sum())

    # --- Level 1: Replicate QC (on ALL spectra, gate info is metadata) ---
    qc_stats = calculate_replicate_qc(spectra, common_grid, _fp)

    # --- Failure identification ---
    failures = identify_qc_failures(qc_stats, rsd_threshold=_rsd, corr_threshold=_corr)

    # --- Group summary ---
    group_summary = summarize_qc_by_group(qc_stats)

    # --- Logging ---
    n_total = len(qc_stats)
    n_fail = len(failures)
    logger.info("\n" + "=" * 40)
    logger.info("QC Summary")
    logger.info("=" * 40)
    logger.info(f"  Intensity gate failures: {n_gate_fail}/{len(gate_df)} spectra")
    logger.info(f"  Replicate QC: {n_total} samples")
    logger.info(f"  Passed: {n_total - n_fail}")
    logger.info(f"  Failed: {n_fail} ({100 * n_fail / max(n_total, 1):.1f}%)")
    if "mean_rsd" in qc_stats.columns and len(qc_stats) > 0:
        logger.info(
            f"  Mean RSD: {qc_stats['mean_rsd'].mean():.2f} ± "
            f"{qc_stats['mean_rsd'].std():.2f}%"
        )
    if "mean_corr" in qc_stats.columns and len(qc_stats) > 0:
        logger.info(
            f"  Mean Corr: {qc_stats['mean_corr'].mean():.4f} ± "
            f"{qc_stats['mean_corr'].std():.4f}"
        )

    # --- Report ---
    if save_report:
        with open(save_report, "w", encoding="utf-8") as f:
            f.write("SERS QC Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fingerprint region: {_fp} cm⁻¹\n\n")
            f.write("Thresholds:\n")
            f.write(f"  - Intensity gate ratio: {_gate}\n")
            f.write(f"  - RSD < {_rsd}%\n")
            f.write(f"  - Correlation > {_corr}\n\n")
            f.write("Results:\n")
            f.write(f"  - Total spectra: {len(spectra)}\n")
            f.write(f"  - Intensity gate failures: {n_gate_fail}\n")
            f.write(f"  - Total samples: {n_total}\n")
            f.write(f"  - Passed: {n_total - n_fail}\n")
            f.write(f"  - Failed: {n_fail}\n\n")
            f.write("Group Summary:\n")
            f.write(group_summary.to_string() + "\n\n")
            if len(failures) > 0:
                f.write("Failed Samples:\n")
                f.write(failures.to_string() + "\n")
        logger.info(f"Report saved to {save_report}")

    return gate_df, qc_stats, failures, group_summary

# =============================================================================
# v2 — Two-Stage QC (2026-04-08)
# =============================================================================
# Stage 1 (raw):
#     - Intensity gate (existing)
#     - Cosmic ray spike detection
#     - Detector saturation detection
# Stage 2 (post-preprocessing):
#     - Per-spectrum correlation-to-mean filter (per-replicate drop)
#     - Min replicates per subject enforcement
#
# Compared to v1 (calculate_replicate_qc → identify_qc_failures), v2 drops
# individual bad replicates instead of dropping entire subjects, which avoids
# the catastrophic ~67% loss observed for BLC under v1.

DEFAULT_COSMIC_PROMINENCE = 15.0  # (legacy) unused — now uses isolation criteria
DEFAULT_SAT_PLATEAU = 5
# v2 thresholds derived from NOR+YNOR replicate corr-to-mean distribution
# (Phase 7, 2026-04-08): p5 = 0.925 → per_spec_corr; min_reps=4 keeps NOR ≥95%
DEFAULT_PER_SPEC_CORR = 0.925
DEFAULT_MIN_REPS_AFTER_QC = 4


def detect_cosmic_ray(
    y: np.ndarray,
    isolation_ratio: float = 2.0,
    height_ratio: float = 0.3,
) -> bool:
    """Return True if the global max appears as an isolated single-pixel spike.

    A cosmic ray on a CCD is a *single* pixel that is dramatically higher
    than its immediate neighbors AND constitutes a large fraction of the
    spectrum's dynamic range.  Real Raman peaks have shoulders — even 2-px
    peaks have at least one supporting neighbor.

    Criteria (BOTH must hold):
        (peak - neighbor_avg) / max(neighbor_avg, ε)  > isolation_ratio
        (peak - neighbor_avg) / (max(y) - min(y))    > height_ratio

    Defaults are intentionally conservative — modern Raman CCDs with
    multi-replicate averaging rarely produce cosmic rays, and false positives
    here would discard genuine sharp metabolite peaks.
    """
    if len(y) < 5:
        return False
    yf = y.astype(float)
    imax = int(np.argmax(yf))
    if imax == 0 or imax == len(yf) - 1:
        return False
    peak = yf[imax]
    neighbor_avg = 0.5 * (yf[imax - 1] + yf[imax + 1])
    spectrum_range = float(np.max(yf) - np.min(yf))
    if spectrum_range <= 0 or neighbor_avg <= 0:
        return False
    isolation = (peak - neighbor_avg) / max(neighbor_avg, 1e-9)
    height_frac = (peak - neighbor_avg) / spectrum_range
    return isolation > isolation_ratio and height_frac > height_ratio


def detect_saturation(y: np.ndarray, plateau_min: int = DEFAULT_SAT_PLATEAU) -> bool:
    """Return True if a flat plateau at maximum intensity suggests saturation.

    Counts the longest consecutive run of points within 0.1% of max(y);
    if >= ``plateau_min``, the detector is likely clipping.
    """
    if len(y) < plateau_min:
        return False
    ymax = float(np.max(y))
    if ymax <= 0:
        return False
    tol = max(abs(ymax) * 1e-3, 1e-9)
    near_max = np.abs(y - ymax) < tol
    longest = current = 0
    for v in near_max:
        if v:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return longest >= plateau_min


def apply_stage1_qc(
    raw_spectra: Dict[Tuple, Tuple[np.ndarray, np.ndarray]],
    fingerprint_region: Tuple[float, float] = FINGERPRINT_REGION,
    intensity_gate_ratio: float = DEFAULT_INTENSITY_GATE_RATIO,
    cosmic_isolation: float = 2.0,
    cosmic_height: float = 0.3,
    sat_plateau: int = DEFAULT_SAT_PLATEAU,
) -> Tuple[set, pd.DataFrame]:
    """Stage 1 QC on raw spectra: intensity gate + cosmic + saturation.

    Returns (passed_keys, drop_log_df).  drop_log_df has one row per dropped
    spectrum with a ``reason`` column.
    """
    gate_df = calculate_intensity_gate(
        raw_spectra, fingerprint_region, intensity_gate_ratio
    )
    gate_pass = {
        (r["group"], r["sample_id"], r["replicate"])
        for _, r in gate_df.iterrows() if r["gate_pass"]
    }

    drops = []
    passed = set()
    for key, (x, y) in raw_spectra.items():
        reasons = []
        if key not in gate_pass:
            reasons.append("intensity_gate")
        if detect_cosmic_ray(y, cosmic_isolation, cosmic_height):
            reasons.append("cosmic_ray")
        if detect_saturation(y, sat_plateau):
            reasons.append("saturation")
        if reasons:
            g, s, r = key
            drops.append({
                "group": g, "sample_id": s, "replicate": r,
                "reason": ";".join(reasons),
            })
        else:
            passed.add(key)

    drop_df = pd.DataFrame(drops)
    logger.info(
        f"Stage 1 QC: {len(passed)}/{len(raw_spectra)} spectra passed "
        f"({len(drop_df)} dropped)"
    )
    return passed, drop_df


def apply_per_spectrum_corr_qc(
    processed_spectra: Dict[Tuple, np.ndarray],
    corr_threshold: float = DEFAULT_PER_SPEC_CORR,
) -> Tuple[set, pd.DataFrame]:
    """Stage 2a — drop individual replicates whose corr-to-mean-of-others < threshold.

    Unlike v1 (which fails the whole subject on RSD/corr breach), this only
    drops the offending replicate, leaving the rest of the subject intact.

    ``processed_spectra`` should already be on a common grid (so vstack works
    directly).  If a subject has only 1 replicate it is passed through.
    """
    # Group by (group, sample_id)
    groups: Dict[Tuple[str, str], List[Tuple[str, np.ndarray]]] = {}
    for (g, s, r), y in processed_spectra.items():
        groups.setdefault((g, s), []).append((r, y))

    drops = []
    passed: set = set()
    for (g, s), reps in groups.items():
        if len(reps) < 2:
            for r, _ in reps:
                passed.add((g, s, r))
            continue
        names = [r for r, _ in reps]
        mat = np.vstack([y for _, y in reps])
        for i, r in enumerate(names):
            mask = np.arange(len(reps)) != i
            mean_other = mat[mask].mean(axis=0)
            with np.errstate(invalid="ignore"):
                corr = float(np.corrcoef(mat[i], mean_other)[0, 1])
            if not np.isfinite(corr) or corr < corr_threshold:
                drops.append({
                    "group": g, "sample_id": s, "replicate": r,
                    "corr_to_others": corr if np.isfinite(corr) else None,
                    "reason": "low_corr_to_mean",
                })
            else:
                passed.add((g, s, r))

    drop_df = pd.DataFrame(drops)
    logger.info(
        f"Stage 2a (per-spec corr): {len(passed)}/{len(processed_spectra)} "
        f"replicates kept ({len(drop_df)} dropped)"
    )
    return passed, drop_df


def enforce_min_replicates(
    passed_keys: set,
    min_n: int = DEFAULT_MIN_REPS_AFTER_QC,
) -> Tuple[set, pd.DataFrame]:
    """Stage 2b — drop entire subjects whose surviving replicate count < min_n.

    Returns (final_keys, dropped_subjects_df).
    """
    counts: Dict[Tuple[str, str], int] = {}
    for g, s, _ in passed_keys:
        counts[(g, s)] = counts.get((g, s), 0) + 1

    dropped_subjects = [
        {"group": g, "sample_id": s, "n_reps_remaining": n, "reason": "min_reps"}
        for (g, s), n in counts.items() if n < min_n
    ]
    bad_subj = {(d["group"], d["sample_id"]) for d in dropped_subjects}
    final = {k for k in passed_keys if (k[0], k[1]) not in bad_subj}

    df = pd.DataFrame(dropped_subjects)
    logger.info(
        f"Stage 2b (min_reps={min_n}): "
        f"{len(final)}/{len(passed_keys)} replicates kept "
        f"({len(df)} subjects dropped)"
    )
    return final, df


__all__ = [
    "calculate_intensity_gate",
    "filter_by_intensity_gate",
    "calculate_replicate_qc",
    "identify_qc_failures",
    "summarize_qc_by_group",
    "calculate_variance_convergence",
    "filter_by_correlation",
    "find_medoid",
    "select_medoid_spectra",
    "detect_outliers",
    "run_qc_pipeline",
    # v2 — two-stage QC
    "detect_cosmic_ray",
    "detect_saturation",
    "apply_stage1_qc",
    "apply_per_spectrum_corr_qc",
    "enforce_min_replicates",
]
