from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from sers.io import read_spectrum

_AVERAGE_DIRECTORIES: Final = frozenset({"average data", "averaged data"})


@dataclass(frozen=True, slots=True)
class AverageAuditPolicy:
    expected_replicates: int = 5
    max_normalized_rmse: float = 1e-4


@dataclass(frozen=True, slots=True)
class AverageAudit:
    average_path: Path
    replicate_paths: tuple[Path, ...]
    replicate_count: int
    rmse: float
    normalized_rmse: float
    correlation: float
    shared_grid: bool
    target_usable: bool


DEFAULT_AUDIT_POLICY: Final = AverageAuditPolicy()


def _replicate_directory(average_path: Path) -> Path:
    if average_path.parent.name.casefold() in _AVERAGE_DIRECTORIES:
        return average_path.parent.parent
    return average_path.parent


def find_average_replicates(average_path: Path) -> tuple[Path, ...]:
    """Find numeric replicate files belonging to a derived ``_ave`` spectrum."""

    if not average_path.stem.casefold().endswith("_ave"):
        msg = f"average filename must end in _ave: {average_path.name}"
        raise ValueError(msg)
    sample_stem = average_path.stem[:-4]
    replicate_pattern = re.compile(rf"^{re.escape(sample_stem)}_(\d+)$", re.IGNORECASE)
    matched: list[tuple[int, Path]] = []
    for path in _replicate_directory(average_path).glob(f"{sample_stem}_*"):
        match = replicate_pattern.fullmatch(path.stem)
        if path.is_file() and path.suffix.casefold() == ".csv" and match is not None:
            matched.append((int(match.group(1)), path))
    return tuple(path for _, path in sorted(matched))


def _align_to_average_grid(
    average_x: np.ndarray,
    replicate_path: Path,
) -> tuple[np.ndarray, bool]:
    replicate_x, replicate_y = read_spectrum(replicate_path)
    shared_grid = len(replicate_x) == len(average_x) and np.allclose(
        replicate_x,
        average_x,
        rtol=0.0,
        atol=1e-9,
    )
    if shared_grid:
        return replicate_y, True
    return np.interp(average_x, replicate_x, replicate_y), False


def audit_average_file(
    average_path: Path,
    policy: AverageAuditPolicy = DEFAULT_AUDIT_POLICY,
) -> AverageAudit:
    """Measure whether ``average_path`` is the arithmetic mean of its replicates."""

    replicate_paths = find_average_replicates(average_path)
    average_x, average_y = read_spectrum(average_path)
    aligned = [_align_to_average_grid(average_x, path) for path in replicate_paths]
    shared_grid = all(item[1] for item in aligned)

    if not aligned:
        return AverageAudit(
            average_path=average_path,
            replicate_paths=(),
            replicate_count=0,
            rmse=float("inf"),
            normalized_rmse=float("inf"),
            correlation=float("nan"),
            shared_grid=True,
            target_usable=False,
        )

    replicate_mean = np.mean([item[0] for item in aligned], axis=0)
    residual = average_y - replicate_mean
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    signal_scale = max(float(np.std(replicate_mean)), np.finfo(float).eps)
    normalized_rmse = rmse / signal_scale
    correlation = float(np.corrcoef(average_y, replicate_mean)[0, 1])
    target_usable = (
        len(replicate_paths) == policy.expected_replicates
        and normalized_rmse <= policy.max_normalized_rmse
    )
    return AverageAudit(
        average_path=average_path,
        replicate_paths=replicate_paths,
        replicate_count=len(replicate_paths),
        rmse=rmse,
        normalized_rmse=normalized_rmse,
        correlation=correlation,
        shared_grid=shared_grid,
        target_usable=target_usable,
    )
