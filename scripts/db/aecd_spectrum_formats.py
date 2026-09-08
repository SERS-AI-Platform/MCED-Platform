from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt


class SpectrumFormatError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpectrumData:
    wavenumber: npt.NDArray[np.float64]
    intensities: npt.NDArray[np.float64]
    file_format: str
    metadata: dict[str, str]


def _validated_arrays(
    wavenumber: npt.NDArray[np.float64],
    intensities: npt.NDArray[np.float64],
    path: Path,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if wavenumber.ndim != 1 or intensities.ndim != 1 or len(wavenumber) != len(intensities) or len(wavenumber) < 2:
        raise SpectrumFormatError(f"Expected two equal one-dimensional arrays with at least two points: {path}")
    if not np.isfinite(wavenumber).all() or not np.isfinite(intensities).all():
        raise SpectrumFormatError(f"Spectrum contains non-finite values: {path}")
    if not np.all(np.diff(wavenumber) > 0):
        raise SpectrumFormatError(f"Wavenumber must be strictly increasing: {path}")
    return wavenumber.astype(np.float64, copy=False), intensities.astype(np.float64, copy=False)


def _read_handheld_csv(rows: list[list[str]], path: Path) -> SpectrumData | None:
    intensity_row = next((row for row in rows if row and row[0].strip().casefold() == "intensities"), None)
    if intensity_row is None:
        return None
    metadata: dict[str, str] = {}
    for row in rows:
        if len(row) >= 2 and row[0].strip().casefold() != "intensities":
            metadata[row[0].strip()] = ",".join(part.strip() for part in row[1:]).strip()
    values = intensity_row[1:]
    if len(values) == 1:
        values = [item.strip() for item in values[0].split(",")]
    try:
        intensities = np.asarray([float(value) for value in values if value.strip()], dtype=np.float64)
        first = float(metadata["Firstwavenumber"])
        last = float(metadata["LastWavenumber"])
    except (KeyError, ValueError) as exc:
        raise SpectrumFormatError(f"Invalid handheld metadata: {path}") from exc
    wavenumber = np.linspace(first, last, len(intensities), dtype=np.float64)
    x, y = _validated_arrays(wavenumber, intensities, path)
    return SpectrumData(x, y, "handheld_csv", metadata)


def read_spectrum(path: Path) -> SpectrumData:
    try:
        if path.suffix.casefold() == ".txt":
            values = np.loadtxt(path, dtype=np.float64, ndmin=2)
            if values.shape[1] != 2:
                raise SpectrumFormatError(f"Expected two columns, got {values.shape[1]}: {path}")
            x, y = _validated_arrays(values[:, 0], values[:, 1], path)
            return SpectrumData(x, y, "txt_2col", {})
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            rows = list(csv.reader(source))
        handheld = _read_handheld_csv(rows, path)
        if handheld is not None:
            return handheld
        if not rows or any(len(row) != 2 for row in rows):
            raise SpectrumFormatError(f"Expected a two-column CSV: {path}")
        values = np.asarray([[float(value) for value in row] for row in rows], dtype=np.float64)
        x, y = _validated_arrays(values[:, 0], values[:, 1], path)
        return SpectrumData(x, y, "csv_2col", {})
    except SpectrumFormatError:
        raise
    except (OSError, ValueError) as exc:
        raise SpectrumFormatError(f"Cannot read spectrum: {path}: {exc}") from exc
