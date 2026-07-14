#!/usr/bin/env python3
"""Utilities for strict native CSV trim-only spectrum reads.

The functions in this module intentionally do not call the project spectrum
reader or preprocessing pipeline. They parse sample names, read the first two
CSV columns as numeric x/y values, preserve row order, and optionally trim by x.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PRIMARY_ROOT = PROJECT_ROOT / "data" / "raw_data"
DEFAULT_CLINICAL_ROOT = PROJECT_ROOT / "data" / "임상데이터"
DATE_RE = re.compile(r"^20\d{6}")

SAMPLE_RE = re.compile(
    r"(?i)^(?P<group>YNOR|YPAN|CPAN|SPAN|BLC|BRE|CRC|DIA|H\.?\s?D\.?|HBP|LUN|NOR|OVA|PRO|PAN|HD)"
    r"\s+(?P<num>\d+)_(?P<rep>\d+|ave)(?:\(\d+\))?(?:_Sample_.*)?\.(?:csv|txt)$"
)


@dataclass(frozen=True)
class SpectrumKey:
    group: str
    sample_id: str
    replicate: str

    @property
    def subject_key(self) -> tuple[str, str]:
        return self.group, self.sample_id


def normalize_group(group: str) -> str:
    compact = re.sub(r"\s+", "", group.upper())
    if compact in {"H.D", "H.D.", "HD"}:
        return "H.D."
    if compact == "PAN":
        return "CPAN"
    return compact


def parse_sample(path: Path) -> SpectrumKey | None:
    lower_parts = [p.lower() for p in path.parts]
    if any(token in part for part in lower_parts for token in ("background", "reference", "blank")):
        return None
    if any(part in {"0. mb", "mb"} for part in lower_parts):
        return None
    if "_ave" in path.stem.lower():
        return None
    if re.search(r"(?i)(^|[\s_/.-])NF($|[\s_.-])", path.name):
        return None
    if re.search(r"(?i)(^|[\s_/.-])PO\.?\s+", path.name):
        return None
    match = SAMPLE_RE.match(path.name)
    if not match:
        return None
    rep = match.group("rep").lower()
    if rep == "ave":
        return None
    return SpectrumKey(
        group=normalize_group(match.group("group")),
        sample_id=str(int(match.group("num"))),
        replicate=str(int(rep)),
    )


def date_key_from_path(path: Path) -> str:
    for part in path.parts:
        if DATE_RE.match(part):
            return part.split("_", 1)[0]
    return "primary"


def date_root_from_path(path: Path) -> Path | None:
    candidate = path if path.is_dir() else path.parent
    for parent in (candidate, *candidate.parents):
        if DATE_RE.match(parent.name):
            return parent
    return None


def readable_spectrum_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = list(root.rglob("*.[Cc][Ss][Vv]")) + list(root.rglob("*.txt"))
    return sorted(p for p in files if p.is_file())


def is_average_file(path: Path) -> bool:
    return "_ave" in path.stem.lower()


def collect_control_files(date_root: Path | None, material: str) -> list[Path]:
    """Collect date-level control spectra without including averaged files."""
    if date_root is None or not date_root.exists():
        return []

    material_l = material.lower().replace("-", "_").replace(" ", "_")
    if material_l == "blank":
        folders = [p for p in date_root.rglob("*") if p.is_dir() and "blank" in p.name.lower()]
        files = [
            f
            for folder in folders
            for f in readable_spectrum_files(folder)
            if "background" not in {part.lower() for part in f.parts}
        ]
    elif material_l in {"ps", "reference_ps", "ref_ps"}:
        folders = []
        for folder in date_root.rglob("*"):
            if not folder.is_dir():
                continue
            name = re.sub(r"[_\s]+", " ", folder.name.lower())
            if (
                "reference" in name
                or "ref ps" in name
                or re.search(r"(^|[^a-z0-9])ps([^a-z0-9]|$)", name)
            ):
                folders.append(folder)
        files = [
            f
            for folder in folders
            for f in readable_spectrum_files(folder)
            if f.stem.strip().upper().startswith("PS")
        ]
    else:
        raise ValueError(f"Unsupported control material: {material}")

    raw = [f for f in files if not is_average_file(f)]
    return sorted(raw or files)


def sample_files(root: Path) -> list[Path]:
    return sorted(p for p in readable_spectrum_files(root) if parse_sample(p) is not None)


def read_native_xy(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read first two CSV columns as x/y values without reordering or smoothing."""
    df = pd.read_csv(path, header=None, usecols=[0, 1], names=["x", "y"])
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df["x"].to_numpy(dtype=float), df["y"].to_numpy(dtype=float)


def read_trimmed_xy(path: Path, x_min: float, x_max: float) -> tuple[np.ndarray, np.ndarray]:
    x, y = read_native_xy(path)
    mask = (x >= x_min) & (x <= x_max)
    return x[mask], y[mask]
