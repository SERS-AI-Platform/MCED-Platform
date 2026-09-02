from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from .data import PairedSpectra

GROUPS: Final = ("Control", "Biopsy-negative", "Cancer")
GROUP_PAIRS: Final = (
    ("Cancer", "Control"),
    ("Biopsy-negative", "Control"),
    ("Cancer", "Biopsy-negative"),
)


@dataclass(frozen=True, slots=True)
class PairwiseSeparation:
    group_a: str
    group_b: str
    n_a: int
    n_b: int
    hedges_g: np.ndarray
    centroid_rms: float
    pooled_within_rms: float
    separation_to_spread: float

    @property
    def label(self) -> str:
        return f"{self.group_a} - {self.group_b}"


@dataclass(frozen=True, slots=True)
class CentroidMargins:
    own_distance: np.ndarray
    nearest_other_distance: np.ndarray
    margin: np.ndarray
    nearest_other_group: np.ndarray


@dataclass(frozen=True, slots=True)
class GroupSeparationDiagnostics:
    legacy_pairs: tuple[PairwiseSeparation, ...]
    powder_pairs: tuple[PairwiseSeparation, ...]
    legacy_margins: CentroidMargins
    powder_margins: CentroidMargins


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def _hedges_g(group_a: np.ndarray, group_b: np.ndarray) -> np.ndarray:
    n_a = len(group_a)
    n_b = len(group_b)
    pooled_variance = (
        (n_a - 1) * np.var(group_a, axis=0, ddof=1)
        + (n_b - 1) * np.var(group_b, axis=0, ddof=1)
    ) / (n_a + n_b - 2)
    pooled_sd = np.sqrt(pooled_variance)
    standardized = np.divide(
        np.mean(group_a, axis=0) - np.mean(group_b, axis=0),
        pooled_sd,
        out=np.zeros_like(pooled_sd),
        where=pooled_sd > 0.0,
    )
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return correction * standardized


def _pairwise_rows(values: np.ndarray, groups: np.ndarray) -> tuple[PairwiseSeparation, ...]:
    rows: list[PairwiseSeparation] = []
    for group_a, group_b in GROUP_PAIRS:
        values_a = values[groups == group_a]
        values_b = values[groups == group_b]
        centroid_a = np.mean(values_a, axis=0)
        centroid_b = np.mean(values_b, axis=0)
        residuals = np.vstack((values_a - centroid_a, values_b - centroid_b))
        centroid_rms = _rms(centroid_a - centroid_b)
        pooled_within_rms = _rms(residuals)
        rows.append(
            PairwiseSeparation(
                group_a=group_a,
                group_b=group_b,
                n_a=len(values_a),
                n_b=len(values_b),
                hedges_g=_hedges_g(values_a, values_b),
                centroid_rms=centroid_rms,
                pooled_within_rms=pooled_within_rms,
                separation_to_spread=centroid_rms / pooled_within_rms,
            )
        )
    return tuple(rows)


def _centroid_margins(values: np.ndarray, groups: np.ndarray) -> CentroidMargins:
    centroids = {group: np.mean(values[groups == group], axis=0) for group in GROUPS}
    own_distance = np.empty(len(values), dtype=float)
    nearest_other_distance = np.empty(len(values), dtype=float)
    nearest_other_group = np.empty(len(values), dtype="<U32")
    for index, spectrum in enumerate(values):
        own_group = str(groups[index])
        own_values = values[groups == own_group]
        own_centroid = (np.sum(own_values, axis=0) - spectrum) / (len(own_values) - 1)
        own_distance[index] = _rms(spectrum - own_centroid)
        other_groups = tuple(group for group in GROUPS if group != own_group)
        other_distances = np.array(
            [_rms(spectrum - centroids[group]) for group in other_groups],
            dtype=float,
        )
        nearest_index = int(np.argmin(other_distances))
        nearest_other_distance[index] = other_distances[nearest_index]
        nearest_other_group[index] = other_groups[nearest_index]
    return CentroidMargins(
        own_distance=own_distance,
        nearest_other_distance=nearest_other_distance,
        margin=nearest_other_distance - own_distance,
        nearest_other_group=nearest_other_group,
    )


def evaluate_group_separation(spectra: PairedSpectra) -> GroupSeparationDiagnostics:
    groups = spectra.clinical_groups
    return GroupSeparationDiagnostics(
        legacy_pairs=_pairwise_rows(spectra.legacy.spectra, groups),
        powder_pairs=_pairwise_rows(spectra.powder.spectra, groups),
        legacy_margins=_centroid_margins(spectra.legacy.spectra, groups),
        powder_margins=_centroid_margins(spectra.powder.spectra, groups),
    )
