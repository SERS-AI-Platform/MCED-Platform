from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from mapping_repeat_average_core import GROUP_ORDER, MappingSubject, robust_qc

Array = npt.NDArray[np.float64]


class CovarianceExperimentError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SubjectSummary:
    ordinal: int
    group: str
    retained_repeats: int
    noise_floor_median: float
    noise_p95: float
    residual_rms: float


@dataclass(frozen=True, slots=True)
class CovarianceSummary:
    name: str
    n_subjects: int
    degrees_of_freedom: int
    eigenvalues: Array
    eigenvectors: Array
    explained: Array
    cumulative: Array
    trace: float
    effective_rank: float
    components_95: int
    diagonal_sd: Array


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    grid: Array
    within: CovarianceSummary
    between_raw: CovarianceSummary
    between_corrected: CovarianceSummary
    group_within: tuple[CovarianceSummary, ...]
    subject_rows: tuple[SubjectSummary, ...]
    repeat_counts: Array
    mean_inverse_repeats: float
    raw_between_min_eigenvalue: float
    corrected_between_min_eigenvalue: float
    corrected_between_negative_variance_fraction: float
    raw_mean_covariance: Array
    corrected_between_covariance: Array


def corrected_between_covariance(
    raw_mean_covariance: Array,
    within_covariance: Array,
    average_inverse_repeats: float,
) -> Array:
    """Apply the method-of-moments correction to subject-mean covariance."""
    corrected = raw_mean_covariance - average_inverse_repeats * within_covariance
    return (corrected + corrected.T) / 2.0


def effective_rank(eigenvalues: Array) -> float:
    """Return entropy effective rank for non-negative eigenvalues."""
    values = np.asarray(eigenvalues, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if not len(values) or float(values.sum()) <= 0:
        return 0.0
    probabilities = values / values.sum()
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def ordered_eigenspectrum(covariance: Array) -> tuple[Array, Array, Array, Array]:
    """Return descending, non-negative eigenvalues and oriented eigenvectors."""
    symmetric = (covariance + covariance.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    order = np.argsort(values)[::-1]
    ordered_values = np.maximum(values[order], 0.0)
    ordered_vectors = vectors[:, order].copy()
    for index in range(ordered_vectors.shape[1]):
        pivot = int(np.argmax(np.abs(ordered_vectors[:, index])))
        if ordered_vectors[pivot, index] < 0:
            ordered_vectors[:, index] *= -1.0
    total = float(ordered_values.sum())
    explained = ordered_values / total if total > 0 else np.zeros_like(ordered_values)
    return ordered_values, ordered_vectors, explained, np.cumsum(explained)


def _summary(
    name: str,
    covariance: Array,
    n_subjects: int,
    degrees_of_freedom: int,
) -> CovarianceSummary:
    values, vectors, explained, cumulative = ordered_eigenspectrum(covariance)
    components_95 = int(np.searchsorted(cumulative, 0.95) + 1) if len(cumulative) else 0
    return CovarianceSummary(
        name=name,
        n_subjects=n_subjects,
        degrees_of_freedom=degrees_of_freedom,
        eigenvalues=values,
        eigenvectors=vectors,
        explained=explained,
        cumulative=cumulative,
        trace=float(values.sum()),
        effective_rank=effective_rank(values),
        components_95=components_95,
        diagonal_sd=np.sqrt(np.maximum(np.diag(covariance), 0.0)),
    )


def compute_experiment(grid: Array, subjects: list[MappingSubject]) -> ExperimentResult:
    """Estimate pooled within-repeat and corrected between-subject covariance."""
    point_count = len(grid)
    within_cross = np.zeros((point_count, point_count), dtype=float)
    group_cross = {group: np.zeros_like(within_cross) for group in GROUP_ORDER}
    group_df = {group: 0 for group in GROUP_ORDER}
    subject_means: list[Array] = []
    subject_groups: list[str] = []
    subject_rows: list[SubjectSummary] = []
    repeat_counts: list[int] = []

    for subject in subjects:
        keep, _ = robust_qc(subject.aligned_replicates)
        selected = np.asarray(subject.aligned_replicates[keep], dtype=float)
        if len(selected) < 3:
            raise CovarianceExperimentError(
                f"subject {subject.ordinal} has fewer than 3 retained repeats"
            )
        mean = selected.mean(axis=0)
        residual = selected - mean
        cross = residual.T @ residual
        degrees_of_freedom = len(selected) - 1
        within_cross += cross
        group_cross[subject.group] += cross
        group_df[subject.group] += degrees_of_freedom
        repeat_sd = selected.std(axis=0, ddof=1)
        subject_means.append(mean)
        subject_groups.append(subject.group)
        repeat_counts.append(len(selected))
        subject_rows.append(
            SubjectSummary(
                ordinal=subject.ordinal,
                group=subject.group,
                retained_repeats=len(selected),
                noise_floor_median=float(np.median(repeat_sd)),
                noise_p95=float(np.percentile(repeat_sd, 95)),
                residual_rms=float(np.sqrt(np.mean(residual**2))),
            )
        )

    total_degrees_of_freedom = sum(repeat_counts) - len(subjects)
    within_covariance = within_cross / max(total_degrees_of_freedom, 1)
    means = np.vstack(subject_means)
    centered_means = means - means.mean(axis=0)
    raw_mean_covariance = centered_means.T @ centered_means / max(len(means) - 1, 1)
    inverse_repeats = float(np.mean(1.0 / np.asarray(repeat_counts, dtype=float)))
    corrected = corrected_between_covariance(raw_mean_covariance, within_covariance, inverse_repeats)
    raw_between_eigenvalues = np.linalg.eigvalsh(raw_mean_covariance)
    corrected_eigenvalues = np.linalg.eigvalsh(corrected)
    negative = corrected_eigenvalues[corrected_eigenvalues < 0]
    negative_fraction = float(np.abs(negative).sum() / max(np.abs(corrected_eigenvalues).sum(), 1e-12))
    group_summaries = tuple(
        _summary(
            group,
            group_cross[group] / max(group_df[group], 1),
            sum(label == group for label in subject_groups),
            group_df[group],
        )
        for group in GROUP_ORDER
    )
    return ExperimentResult(
        grid=grid,
        within=_summary("within_repeat_noise", within_covariance, len(subjects), total_degrees_of_freedom),
        between_raw=_summary("between_subject_mean_raw", raw_mean_covariance, len(subjects), len(subjects) - 1),
        between_corrected=_summary("between_subject_mean_corrected", corrected, len(subjects), len(subjects) - 1),
        group_within=group_summaries,
        subject_rows=tuple(subject_rows),
        repeat_counts=np.asarray(repeat_counts, dtype=float),
        mean_inverse_repeats=inverse_repeats,
        raw_between_min_eigenvalue=float(raw_between_eigenvalues.min()),
        corrected_between_min_eigenvalue=float(corrected_eigenvalues.min()),
        corrected_between_negative_variance_fraction=negative_fraction,
        raw_mean_covariance=raw_mean_covariance,
        corrected_between_covariance=corrected,
    )
