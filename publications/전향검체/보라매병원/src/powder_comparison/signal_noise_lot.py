from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from boramae_data import preprocess_file

LOT_FILE_PATTERN: Final = re.compile(
    r"^Sigma\s+(?P<lot>[1-5])_BPRO\s+(?P<patient>\d+)_(?P<replicate>[1-5]|ave)\.CSV$",
    re.IGNORECASE,
)


class VarianceDesignError(ValueError):
    """Raised when a crossed patient × lot × replicate design is malformed."""


@dataclass(frozen=True, slots=True)
class VarianceComponents:
    patient: np.ndarray
    lot: np.ndarray
    residual: np.ndarray
    patient_fraction: np.ndarray
    lot_fraction: np.ndarray
    residual_fraction: np.ndarray


@dataclass(frozen=True, slots=True)
class LotReplicates:
    patient_ids: np.ndarray
    lot_ids: np.ndarray
    spectra: np.ndarray


def load_lot_replicates(root: Path, grid: np.ndarray) -> LotReplicates:
    """Load the balanced five-patient × five-lot × five-replicate powder experiment."""
    indexed: dict[tuple[int, int, int], Path] = {}
    for path in sorted(root.glob("*.CSV")):
        match = LOT_FILE_PATTERN.match(path.name)
        if match is None or match["replicate"].lower() == "ave":
            continue
        key = (int(match["patient"]), int(match["lot"]), int(match["replicate"]))
        if key in indexed:
            raise VarianceDesignError(f"Duplicate lot replicate: {path.name}")
        indexed[key] = path
    patient_ids = np.array(sorted({key[0] for key in indexed}), dtype=int)
    lot_ids = np.array(sorted({key[1] for key in indexed}), dtype=int)
    expected = {
        (int(patient), int(lot), replicate)
        for patient in patient_ids
        for lot in lot_ids
        for replicate in range(1, 6)
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        raise VarianceDesignError(f"Unbalanced powder lot inventory; missing={missing[:5]}")
    spectra = np.empty((len(patient_ids), len(lot_ids), 5, len(grid)), dtype=float)
    for patient_index, patient in enumerate(patient_ids):
        for lot_index, lot in enumerate(lot_ids):
            for replicate in range(1, 6):
                _, processed = preprocess_file(
                    indexed[(int(patient), int(lot), replicate)],
                    grid,
                )
                spectra[patient_index, lot_index, replicate - 1] = processed
    return LotReplicates(patient_ids=patient_ids, lot_ids=lot_ids, spectra=spectra)


def variance_components(values: np.ndarray) -> VarianceComponents:
    """Estimate balanced crossed variance components by ANOVA method of moments."""
    if values.ndim != 4:
        raise VarianceDesignError(
            "Variance-component input must be four-dimensional: patient × lot × replicate × feature"
        )
    patient_count, lot_count, replicate_count, _ = values.shape
    if min(patient_count, lot_count, replicate_count) < 2:
        raise VarianceDesignError("Every crossed-design axis requires at least two observations")
    grand = np.mean(values, axis=(0, 1, 2))
    patient_means = np.mean(values, axis=(1, 2))
    lot_means = np.mean(values, axis=(0, 2))
    fitted = patient_means[:, None, None, :] + lot_means[None, :, None, :] - grand
    residuals = values - fitted
    patient_ms = lot_count * replicate_count * np.var(patient_means, axis=0, ddof=1)
    lot_ms = patient_count * replicate_count * np.var(lot_means, axis=0, ddof=1)
    residual_ms = np.sum(np.square(residuals), axis=(0, 1, 2)) / (
        patient_count * lot_count * (replicate_count - 1)
    )
    patient = np.maximum(
        (patient_ms - residual_ms) / (lot_count * replicate_count),
        0.0,
    )
    lot = np.maximum((lot_ms - residual_ms) / (patient_count * replicate_count), 0.0)
    residual = np.maximum(residual_ms, 0.0)
    total = patient + lot + residual
    nonzero = total > 0.0
    return VarianceComponents(
        patient=patient,
        lot=lot,
        residual=residual,
        patient_fraction=np.divide(patient, total, out=np.zeros_like(total), where=nonzero),
        lot_fraction=np.divide(lot, total, out=np.zeros_like(total), where=nonzero),
        residual_fraction=np.divide(residual, total, out=np.zeros_like(total), where=nonzero),
    )
