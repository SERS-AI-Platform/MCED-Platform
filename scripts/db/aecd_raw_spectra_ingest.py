from __future__ import annotations

import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .aecd_spectrum_formats import SpectrumData, SpectrumFormatError
from .aecd_spectrum_formats import read_spectrum as read_spectrum_file

SOURCE_MAPPING_PATTERN: Final = re.compile(r"(?:^|;\s*)source_mapping=([^;]+)")


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True, order=True)
class SpectrumKey:
    source_mapping: str
    solum_label: str
    point_no: int


@dataclass(frozen=True, slots=True)
class SpectrumFile:
    key: SpectrumKey
    path: Path


@dataclass(frozen=True, slots=True)
class Discovery:
    files: tuple[SpectrumFile, ...]
    sample_count: int
    averages_excluded: int


@dataclass(frozen=True, slots=True)
class ValidationSummary:
    point_lengths: tuple[tuple[int, int], ...]
    manifest_sha256: str


def source_mapping_from_notes(notes: str) -> str:
    match = SOURCE_MAPPING_PATTERN.search(notes)
    if match is None:
        raise IngestError(f"run notes have no source_mapping: {notes}")
    return match.group(1).strip()


def _classify_csv(
    path: Path,
    source_mapping: str,
    solum_label: str,
) -> SpectrumFile | None:
    escaped_label = re.escape(solum_label)
    if re.fullmatch(rf"{escaped_label}_ave", path.stem, re.IGNORECASE):
        return None
    match = re.fullmatch(rf"{escaped_label}_(?:ave)?(\d{{4}})", path.stem, re.IGNORECASE)
    if match is None:
        raise IngestError(f"unrecognized spectrum filename: {path}")
    return SpectrumFile(
        SpectrumKey(source_mapping, solum_label.replace(" ", "_"), int(match.group(1))),
        path,
    )


def discover_spectrum_files(
    mapping_root: Path,
    expected_points_per_sample: int,
) -> Discovery:
    if expected_points_per_sample < 1:
        raise IngestError("expected_points_per_sample must be positive")
    mapping_dirs = tuple(sorted(mapping_root.glob("*_mapping")))
    if not mapping_dirs:
        raise IngestError(f"no mapping directories found: {mapping_root}")
    files: list[SpectrumFile] = []
    averages_excluded = 0
    sample_count = 0
    expected_points = set(range(1, expected_points_per_sample + 1))
    for mapping_dir in mapping_dirs:
        for sample_dir in sorted(path for path in mapping_dir.iterdir() if path.is_dir()):
            sample_count += 1
            sample_files: list[SpectrumFile] = []
            for path in sorted(
                item
                for item in sample_dir.iterdir()
                if item.is_file() and item.suffix.casefold() == ".csv"
            ):
                classified = _classify_csv(path, mapping_dir.name, sample_dir.name)
                if classified is None:
                    averages_excluded += 1
                else:
                    sample_files.append(classified)
            points = [item.key.point_no for item in sample_files]
            if len(points) != len(set(points)):
                raise IngestError(f"duplicate point number: {sample_dir}")
            if set(points) != expected_points:
                found_count = len(points)
                message = "; ".join(
                    (
                        f"point set mismatch: {sample_dir}",
                        f"expected 1..{expected_points_per_sample}, found {found_count}",
                    )
                )
                raise IngestError(message)
            files.extend(sample_files)
    return Discovery(
        tuple(sorted(files, key=lambda item: item.key)),
        sample_count,
        averages_excluded,
    )


def read_spectrum(path: Path) -> SpectrumData:
    try:
        return read_spectrum_file(path)
    except SpectrumFormatError as exc:
        raise IngestError(str(exc)) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_files(discovery: Discovery, mapping_root: Path) -> ValidationSummary:
    point_lengths: Counter[int] = Counter()
    manifest = hashlib.sha256()
    for item in discovery.files:
        data = read_spectrum(item.path)
        digest = sha256_file(item.path)
        point_lengths[len(data.wavenumber)] += 1
        manifest.update(item.path.relative_to(mapping_root).as_posix().encode())
        manifest.update(digest.encode())
    return ValidationSummary(tuple(sorted(point_lengths.items())), manifest.hexdigest())
