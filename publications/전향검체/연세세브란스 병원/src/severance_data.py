from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
from scipy.signal import savgol_filter

REPO: Final = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO / "src"))

from sers.io import read_spectrum  # noqa: E402
from sers.signal import baseline_correction, snv  # noqa: E402

OUT: Final = REPO / "publications" / "전향검체" / "연세세브란스 병원"
FIG_DIR: Final = OUT / "figures"
TABLE_DIR: Final = OUT / "tables"
PROCESSED_CSV: Final = REPO / "results" / "kao_20260610_updated_cohort" / "processed_spectra.csv"
CLEAN_MANIFEST: Final = REPO / "results" / "clean_cohort_20260605" / "clean_cohort_manifest.csv"
CLINICAL_MAPPING: Final = TABLE_DIR / "yonsei_clinical_data_mapping.csv"
PREDICTIONS: Final = TABLE_DIR / "yonsei_oof_predictions.csv"
OPTIMIZED_PREDICTIONS: Final = TABLE_DIR / "severance_lr_oof_predictions.csv"
YPAN_RAW: Final = REPO / "data" / "raw_data" / "10-3. Y-Pancreatic cancer (YPAN)"
YNOR_RAW: Final = REPO / "data" / "raw_data" / "12. Y-Normal (YNOR)"
RAW_FILE_RE: Final = re.compile(r"^(YPAN|YNOR)\s+([0-9]+)_([0-9]+|ave)\.CSV$", re.IGNORECASE)
EXCLUDED_REPEAT_ACQUISITIONS: Final = frozenset({("YNOR", "21")})


@dataclass(frozen=True, slots=True)
class CohortSpectra:
    grid: np.ndarray
    ypan_subject_ids: tuple[str, ...]
    ypan: np.ndarray
    ynor_subject_ids: tuple[str, ...]
    ynor: np.ndarray
    clean_cpan_subject_ids: tuple[str, ...]
    clean_cpan: np.ndarray


@dataclass(frozen=True, slots=True)
class StageSpectrum:
    name: str
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True, slots=True)
class PreprocessingTrace:
    group: str
    subject_id: str
    stages: tuple[StageSpectrum, ...]


def normalize_sample_id(value: str) -> str:
    return str(int(float(value)))


def load_clean_cpan_ids(path: Path) -> set[str]:
    with path.open(encoding="utf-8-sig") as handle:
        return {
            normalize_sample_id(row["sample_id"])
            for row in csv.DictReader(handle)
            if row["source_group"] == "CPAN"
        }


def load_cohort_spectra(processed_csv: Path, clean_manifest: Path) -> CohortSpectra:
    clean_cpan_ids = load_clean_cpan_ids(clean_manifest)
    grouped: dict[str, dict[str, list[np.ndarray]]] = {
        "YPAN": {},
        "YNOR": {},
        "CPAN": {},
    }
    with processed_csv.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = f"Missing CSV header: {processed_csv}"
            raise RuntimeError(msg)
        feature_columns = [name for name in reader.fieldnames if name.startswith("x_")]
        grid = np.asarray([float(name[2:]) for name in feature_columns], dtype=float)
        for row in reader:
            group = row["group"].upper()
            if group not in grouped:
                continue
            sample_id = normalize_sample_id(row["sample_id"])
            if (group, sample_id) in EXCLUDED_REPEAT_ACQUISITIONS:
                continue
            if group == "CPAN" and sample_id not in clean_cpan_ids:
                continue
            spectrum = np.asarray([float(row[name]) for name in feature_columns], dtype=float)
            grouped[group].setdefault(sample_id, []).append(spectrum)

    def aggregate(group: str) -> tuple[tuple[str, ...], np.ndarray]:
        ordered = sorted(grouped[group], key=int)
        subject_ids = tuple(f"{group}_{sample_id}" for sample_id in ordered)
        spectra = np.vstack(
            [np.vstack(grouped[group][sample_id]).mean(axis=0) for sample_id in ordered]
        )
        return subject_ids, spectra

    ypan_ids, ypan = aggregate("YPAN")
    ynor_ids, ynor = aggregate("YNOR")
    cpan_ids, cpan = aggregate("CPAN")
    return CohortSpectra(grid, ypan_ids, ypan, ynor_ids, ynor, cpan_ids, cpan)


def preprocess_file(path: Path, grid: np.ndarray) -> tuple[StageSpectrum, ...]:
    x, y = read_spectrum(path)
    mask = (x >= float(grid.min())) & (x <= float(grid.max()))
    x_trim = x[mask]
    y_trim = y[mask]
    y_smooth = savgol_filter(y_trim, window_length=11, polyorder=3, mode="interp")
    y_baseline = baseline_correction(y_smooth, window=101)
    y_snv = snv(y_baseline)
    y_grid = np.interp(grid, x_trim, y_snv)
    return (
        StageSpectrum("Raw", x, y),
        StageSpectrum("Trimmed", x_trim, y_trim),
        StageSpectrum("Smoothed", x_trim, y_smooth),
        StageSpectrum("Baseline corrected", x_trim, y_baseline),
        StageSpectrum("SNV + model grid", grid, y_grid),
    )


def load_preprocessing_traces(grid: np.ndarray) -> tuple[PreprocessingTrace, ...]:
    traces: list[PreprocessingTrace] = []
    for root in (YPAN_RAW, YNOR_RAW):
        for path in sorted(root.glob("*.CSV")):
            match = RAW_FILE_RE.match(path.name)
            if match is None or match.group(3).lower() != "1":
                continue
            group, sample_id, _ = match.groups()
            traces.append(
                PreprocessingTrace(
                    group=group.upper(),
                    subject_id=f"{group.upper()}_{sample_id}",
                    stages=preprocess_file(path, grid),
                )
            )
    return tuple(traces)


def ensure_output_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
