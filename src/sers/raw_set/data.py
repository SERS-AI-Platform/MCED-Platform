from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from sers.io import parse_filename, read_spectrum
from sers.raw_set.audit import audit_average_file

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class CohortSpec:
    folder: str
    group: str
    cancer: bool


@dataclass(frozen=True, slots=True)
class PatientRecord:
    patient_key: str
    group: str
    cancer: bool
    replicates: FloatArray
    average: FloatArray
    average_usable: bool


@dataclass(frozen=True, slots=True)
class DuplicatePatientError(ValueError):
    patient_key: str

    def __str__(self) -> str:
        return f"patient appears in more than one configured cohort: {self.patient_key}"


class RawPatientItem(NamedTuple):
    spectra: torch.Tensor
    replicate_mask: torch.Tensor
    average: torch.Tensor
    average_mask: torch.Tensor
    label: torch.Tensor
    patient_key: str
    group: str


def _resample(path: Path, grid: np.ndarray) -> FloatArray:
    x, y = read_spectrum(path)
    resampled: FloatArray = np.asarray(np.interp(grid, x, y), dtype=np.float32)
    return resampled


def _average_by_sample(cohort_dir: Path) -> dict[str, Path]:
    candidates = list(cohort_dir.glob("*_ave.CSV")) + list(cohort_dir.glob("*_ave.csv"))
    for directory_name in ("Average data", "Averaged data"):
        average_dir = cohort_dir / directory_name
        candidates.extend(average_dir.glob("*_ave.CSV"))
        candidates.extend(average_dir.glob("*_ave.csv"))
    averages: dict[str, Path] = {}
    for path in candidates:
        sample_part = path.stem[:-4].rsplit(maxsplit=1)[-1]
        averages[sample_part] = path
    return averages


def _load_cohort(
    data_root: Path,
    spec: CohortSpec,
    grid: np.ndarray,
) -> list[PatientRecord]:
    cohort_dir = data_root / spec.folder
    replicate_paths: dict[str, list[tuple[int, Path]]] = {}
    for path in cohort_dir.iterdir():
        if not path.is_file() or path.suffix.casefold() != ".csv":
            continue
        try:
            spectrum_id = parse_filename(path, fallback_group=spec.group)
        except ValueError:
            continue
        replicate_paths.setdefault(spectrum_id.sample_id, []).append(
            (spectrum_id.replicate, path)
        )

    averages = _average_by_sample(cohort_dir)
    records: list[PatientRecord] = []
    for sample_id, indexed_paths in sorted(replicate_paths.items()):
        ordered_paths = [path for _, path in sorted(indexed_paths)]
        replicates = np.stack([_resample(path, grid) for path in ordered_paths])
        average_path = averages.get(sample_id)
        average_usable = False
        average = np.zeros(len(grid), dtype=np.float32)
        if average_path is not None:
            average_usable = audit_average_file(average_path).target_usable
            average = _resample(average_path, grid)
        records.append(
            PatientRecord(
                patient_key=f"{spec.group}:{sample_id}",
                group=spec.group,
                cancer=spec.cancer,
                replicates=replicates,
                average=average,
                average_usable=average_usable,
            )
        )
    return records


def load_patient_records(
    data_root: Path,
    cohorts: tuple[CohortSpec, ...],
    grid: np.ndarray,
) -> tuple[PatientRecord, ...]:
    by_patient: dict[str, PatientRecord] = {}
    for spec in cohorts:
        for record in _load_cohort(data_root, spec, grid):
            if record.patient_key in by_patient:
                raise DuplicatePatientError(record.patient_key)
            by_patient[record.patient_key] = record
    return tuple(by_patient[key] for key in sorted(by_patient))


class RawPatientDataset(Dataset[RawPatientItem]):
    def __init__(
        self,
        records: list[PatientRecord] | tuple[PatientRecord, ...],
        max_replicates: int = 5,
    ) -> None:
        self.records = tuple(records)
        self.max_replicates = max_replicates

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> RawPatientItem:
        record = self.records[index]
        count, n_wavenumbers = record.replicates.shape
        if count > self.max_replicates:
            msg = (
                f"{record.patient_key} has {count} replicates, "
                f"maximum is {self.max_replicates}"
            )
            raise ValueError(msg)
        spectra = torch.zeros(self.max_replicates, n_wavenumbers)
        spectra[:count] = torch.from_numpy(record.replicates)
        replicate_mask = torch.zeros(self.max_replicates, dtype=torch.bool)
        replicate_mask[:count] = True
        return RawPatientItem(
            spectra=spectra,
            replicate_mask=replicate_mask,
            average=torch.from_numpy(record.average),
            average_mask=torch.tensor(record.average_usable),
            label=torch.tensor(float(record.cancer)),
            patient_key=record.patient_key,
            group=record.group,
        )
