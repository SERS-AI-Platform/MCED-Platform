#!/usr/bin/env python3
"""Sample-level comparison of primary raw_data vs reference/blank clinical raw.

The primary raw_data tree was acquired without day-level reference/blank
controls. The clinical tree was re-acquired by date with PS/Si reference and
blank spectra. This script makes that difference visible at sample level:

1. Match spectra by (group, sample_id, replicate).
2. Build date-level calibration from clinical PS/Si/blank folders.
3. Compare clinical variants against primary raw_data after baseline-corrected
   SNV preprocessing:
   clinical_raw, axis_corrected, axis_intensity_corrected.
   Blank spectra are retained as date-level QC only, not subtracted directly.
4. Write replicate/sample/date summaries and outlier-oriented plots.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.io import read_spectrum
from src.sers.preprocessing import baseline_correction

GRID = np.arange(400.0, 1800.1, 1.0)
PS_ANCHORS = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1602.3], dtype=float)
SI_ANCHORS = np.array([520.7], dtype=float)
ALL_ANCHORS = np.concatenate([SI_ANCHORS, PS_ANCHORS])
SAMPLE_BASELINE_METHOD = "airpls"
VARIANTS = (
    "clinical_raw",
    "axis_corrected",
    "axis_intensity_corrected",
)

DEFAULT_PRIMARY_ROOT = PROJECT_ROOT / "data" / "raw_data"
DEFAULT_CLINICAL_ROOT = PROJECT_ROOT / "data" / "임상데이터"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "raw_data_vs_clinical_reference"

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

    @property
    def replicate_key(self) -> tuple[str, str, str]:
        return self.group, self.sample_id, self.replicate


@dataclass
class PreparedPrimary:
    path: Path
    key: SpectrumKey
    date_key: str
    y_snv: np.ndarray
    y_deriv_snv: np.ndarray
    fp_mean_abs: float
    fp_p95_abs: float


@dataclass
class PreparedVector:
    y_grid: np.ndarray
    y_baseline_corrected: np.ndarray
    y_snv: np.ndarray
    y_deriv_snv: np.ndarray
    fp_mean_abs: float
    fp_p95_abs: float


@dataclass
class DayCalibration:
    date_key: str
    root: Path
    slope: float = 1.0
    intercept: float = 0.0
    median_shift_expected_minus_observed: float = 0.0
    max_abs_error_before: float = np.nan
    n_anchor_pairs: int = 0
    n_ps: int = 0
    n_si: int = 0
    n_blank: int = 0
    ps_x: np.ndarray | None = None
    ps_y: np.ndarray | None = None
    si_x: np.ndarray | None = None
    si_y: np.ndarray | None = None
    blank_x: np.ndarray | None = None
    blank_y: np.ndarray | None = None
    gain_y: np.ndarray | None = None
    gain_available: bool = False
    ps_heights: dict[str, float] = field(default_factory=dict)
    used_identity_axis: bool = False
    note: str = ""

    @property
    def has_axis(self) -> bool:
        return self.n_anchor_pairs > 0 and not self.used_identity_axis

    @property
    def has_blank(self) -> bool:
        return self.blank_x is not None and self.blank_y is not None


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
        if re.match(r"^20\d{6}", part):
            return part.split("_", 1)[0]
    return "primary"


def readable_spectrum_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".txt"}


def sample_files(root: Path) -> list[Path]:
    files = list(root.rglob("*.[Cc][Ss][Vv]")) + list(root.rglob("*.txt"))
    return sorted(p for p in files if parse_sample(p) is not None)


def snv(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if not np.isfinite(std) or std <= 0:
        return y - mean
    return (y - mean) / std


def prepare_vector(x: np.ndarray, y: np.ndarray) -> PreparedVector:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    y_grid = np.interp(GRID, x, y)
    smoothed = savgol_filter(y_grid, window_length=15, polyorder=3, mode="interp")
    y_bc = baseline_correction(smoothed, method=SAMPLE_BASELINE_METHOD)
    y_snv = snv(y_bc)
    derivative = savgol_filter(
        y_bc,
        window_length=15,
        polyorder=3,
        deriv=1,
        delta=float(GRID[1] - GRID[0]),
        mode="interp",
    )
    abs_y = np.abs(y_grid)
    return PreparedVector(
        y_grid=y_grid,
        y_baseline_corrected=y_bc,
        y_snv=y_snv,
        y_deriv_snv=snv(derivative),
        fp_mean_abs=float(np.nanmean(abs_y)),
        fp_p95_abs=float(np.nanpercentile(abs_y, 95)),
    )


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=float) - float(np.nanmean(a))
    bb = np.asarray(b, dtype=float) - float(np.nanmean(b))
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 0:
        return np.nan
    return float((aa @ bb) / denom)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((np.asarray(a) - np.asarray(b)) ** 2)))


def reference_preprocess(y: np.ndarray) -> np.ndarray:
    y = savgol_filter(np.asarray(y, dtype=float), 11, 2, mode="interp")
    return baseline_correction(y, window=201)


def collect_material_files(root: Path, material: str) -> list[Path]:
    material_l = material.lower()
    if material_l == "blank":
        folders = [p for p in root.rglob("*") if p.is_dir() and "blank" in p.name.lower()]
        files = [f for folder in folders for f in folder.iterdir() if readable_spectrum_file(f)]
    else:
        folders = [
            p
            for p in root.rglob("*")
            if p.is_dir()
            and (
                "reference" in p.name.lower()
                or material_l in p.name.lower()
                or f"ref {material_l}" in p.name.lower()
            )
        ]
        files = [
            f
            for folder in folders
            for f in folder.iterdir()
            if readable_spectrum_file(f) and material_l in f.stem.lower()
        ]
    raw = [f for f in files if "_ave" not in f.stem.lower()]
    return sorted(raw or files)


def mean_spectrum(files: list[Path]) -> tuple[np.ndarray, np.ndarray, int] | None:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for path in files:
        try:
            x, y = read_spectrum(path)
        except Exception:
            continue
        xs.append(np.asarray(x, dtype=float))
        ys.append(np.asarray(y, dtype=float))
    if not ys:
        return None
    x0 = xs[0]
    mat = np.vstack([np.interp(x0, x, y) for x, y in zip(xs, ys)])
    return x0, mat.mean(axis=0), len(ys)


def local_peak_center(
    x: np.ndarray,
    y: np.ndarray,
    anchor: float,
    half_width: float = 15.0,
) -> tuple[float, float] | None:
    mask = (x >= anchor - half_width) & (x <= anchor + half_width)
    if int(mask.sum()) < 5:
        return None
    xx = x[mask]
    yy = y[mask]
    k = int(np.nanargmax(yy))
    center = float(xx[k])
    if 0 < k < len(xx) - 1:
        try:
            coef = np.polyfit(xx[k - 1 : k + 2], yy[k - 1 : k + 2], 2)
            if coef[0] < 0:
                candidate = float(-coef[1] / (2 * coef[0]))
                if anchor - half_width <= candidate <= anchor + half_width:
                    center = candidate
        except Exception:
            pass
    return center, float(yy[k])


def detect_anchor_rows(
    cal: DayCalibration,
    material: str,
    x: np.ndarray | None,
    y: np.ndarray | None,
    anchors: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if x is None or y is None:
        for anchor in anchors:
            rows.append(
                {
                    "date_key": cal.date_key,
                    "material": material,
                    "expected_cm": float(anchor),
                    "status": "missing_reference",
                }
            )
        return rows
    yp = reference_preprocess(y)
    for anchor in anchors:
        peak = local_peak_center(x, yp, float(anchor))
        if peak is None:
            rows.append(
                {
                    "date_key": cal.date_key,
                    "material": material,
                    "expected_cm": float(anchor),
                    "status": "not_detected",
                }
            )
            continue
        observed, height = peak
        rows.append(
            {
                "date_key": cal.date_key,
                "material": material,
                "expected_cm": float(anchor),
                "observed_cm": float(observed),
                "shift_expected_minus_observed_cm": float(anchor - observed),
                "height": float(height),
                "status": "ok",
            }
        )
    return rows


def fit_axis(cal: DayCalibration, anchor_rows: list[dict[str, object]]) -> None:
    ok = [r for r in anchor_rows if r.get("date_key") == cal.date_key and r.get("status") == "ok"]
    if not ok:
        cal.used_identity_axis = True
        cal.note = "no readable PS/SI reference"
        return

    expected = np.array([float(r["expected_cm"]) for r in ok], dtype=float)
    observed = np.array([float(r["observed_cm"]) for r in ok], dtype=float)
    cal.n_anchor_pairs = int(len(ok))
    cal.median_shift_expected_minus_observed = float(np.median(expected - observed))
    cal.max_abs_error_before = float(np.max(np.abs(expected - observed)))

    if len(ok) >= 2:
        slope, intercept = np.polyfit(observed, expected, 1)
        if not (0.98 <= slope <= 1.02 and abs(intercept) <= 25):
            slope, intercept = 1.0, cal.median_shift_expected_minus_observed
            cal.note = "affine rejected; median shift used"
    else:
        slope, intercept = 1.0, cal.median_shift_expected_minus_observed
        cal.note = "single anchor; median shift used"
    cal.slope = float(slope)
    cal.intercept = float(intercept)


def apply_axis(x: np.ndarray, cal: DayCalibration | None) -> np.ndarray:
    if cal is None:
        return np.asarray(x, dtype=float)
    return cal.slope * np.asarray(x, dtype=float) + cal.intercept


def anchor_heights_after_axis(cal: DayCalibration) -> dict[str, float]:
    if cal.ps_x is None or cal.ps_y is None or cal.used_identity_axis:
        return {}
    x_corr = apply_axis(cal.ps_x, cal)
    yp = reference_preprocess(cal.ps_y)
    heights: dict[str, float] = {}
    for anchor in PS_ANCHORS:
        peak = local_peak_center(x_corr, yp, float(anchor))
        if peak is not None:
            heights[f"{anchor:.1f}"] = float(peak[1])
    return heights


def build_day_calibrations(clinical_root: Path) -> tuple[dict[str, DayCalibration], pd.DataFrame]:
    date_roots = sorted(
        p for p in clinical_root.iterdir() if p.is_dir() and re.match(r"^20\d{6}", p.name)
    )
    calibrations: dict[str, DayCalibration] = {}
    anchor_rows: list[dict[str, object]] = []

    for root in date_roots:
        date_key = date_key_from_path(root)
        cal = DayCalibration(date_key=date_key, root=root)

        ps = mean_spectrum(collect_material_files(root, "PS"))
        if ps is not None:
            cal.ps_x, cal.ps_y, cal.n_ps = ps
        si = mean_spectrum(collect_material_files(root, "Si"))
        if si is not None:
            cal.si_x, cal.si_y, cal.n_si = si
        blank = mean_spectrum(collect_material_files(root, "blank"))
        if blank is not None:
            cal.blank_x, cal.blank_y, cal.n_blank = blank

        anchor_rows.extend(detect_anchor_rows(cal, "PS", cal.ps_x, cal.ps_y, PS_ANCHORS))
        anchor_rows.extend(detect_anchor_rows(cal, "SI", cal.si_x, cal.si_y, SI_ANCHORS))
        fit_axis(cal, anchor_rows)
        calibrations[date_key] = cal

    height_rows: list[dict[str, float]] = []
    for cal in calibrations.values():
        cal.ps_heights = anchor_heights_after_axis(cal)
        if cal.ps_heights:
            height_rows.append(cal.ps_heights)
    target_heights = pd.DataFrame(height_rows).median(axis=0).to_dict() if height_rows else {}

    for cal in calibrations.values():
        cal.gain_y = np.ones_like(GRID, dtype=float)
        if not target_heights or not cal.ps_heights:
            if not cal.note:
                cal.note = "insufficient PS anchors for gain"
            continue
        gains: list[float] = []
        for anchor in PS_ANCHORS:
            key = f"{anchor:.1f}"
            observed = float(cal.ps_heights.get(key, np.nan))
            target = float(target_heights.get(key, np.nan))
            if np.isfinite(observed) and observed > 1e-9 and np.isfinite(target):
                gains.append(float(np.clip(target / observed, 0.25, 4.0)))
        if len(gains) >= 2:
            # Use a scalar daily gain. A wavelength-dependent PS gain curve can
            # impose artificial slope on urine spectra and makes intensity
            # ratios change differently for each sample shape.
            scalar_gain = float(np.median(gains))
            cal.gain_y = np.full_like(GRID, scalar_gain, dtype=float)
            cal.gain_available = True
        elif not cal.note:
            cal.note = "insufficient PS anchors for gain"

    return calibrations, pd.DataFrame(anchor_rows)


def apply_reference_correction(
    x: np.ndarray,
    y: np.ndarray,
    cal: DayCalibration | None,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    flags: dict[str, object] = {
        "axis_applied": False,
        "gain_applied": False,
    }
    if variant == "clinical_raw" or cal is None:
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float), flags

    x_corr = apply_axis(x, cal)
    y_corr = np.asarray(y, dtype=float).copy()
    flags["axis_applied"] = bool(cal.has_axis)

    if "intensity" in variant and cal.gain_available and cal.gain_y is not None:
        gain = np.interp(x_corr, GRID, cal.gain_y, left=cal.gain_y[0], right=cal.gain_y[-1])
        y_corr = y_corr * gain
        flags["gain_applied"] = True

    return x_corr, y_corr, flags


def build_primary_index(
    primary_root: Path,
) -> tuple[dict[tuple[str, str, str], list[PreparedPrimary]], pd.DataFrame, pd.DataFrame]:
    index: dict[tuple[str, str, str], list[PreparedPrimary]] = defaultdict(list)
    inventory_rows: list[dict[str, object]] = []
    for path in sample_files(primary_root):
        key = parse_sample(path)
        if key is None:
            continue
        row = {
            "dataset": "primary_raw_data",
            "path": str(path),
            "group": key.group,
            "sample_id": key.sample_id,
            "replicate": key.replicate,
            "date_key": date_key_from_path(path),
            "parsed": True,
            "read_error": "",
        }
        try:
            x, y = read_spectrum(path)
            vec = prepare_vector(x, y)
            index[key.replicate_key].append(
                PreparedPrimary(
                    path=path,
                    key=key,
                    date_key=date_key_from_path(path),
                    y_snv=vec.y_snv,
                    y_deriv_snv=vec.y_deriv_snv,
                    fp_mean_abs=vec.fp_mean_abs,
                    fp_p95_abs=vec.fp_p95_abs,
                )
            )
        except Exception as exc:  # noqa: BLE001
            row["read_error"] = str(exc)
        inventory_rows.append(row)

    duplicate_rows: list[dict[str, object]] = []
    for key, records in sorted(index.items()):
        if len(records) <= 1:
            continue
        for idx, rec in enumerate(records, start=1):
            duplicate_rows.append(
                {
                    "group": key[0],
                    "sample_id": key[1],
                    "replicate": key[2],
                    "primary_candidate_index": idx,
                    "n_primary_candidates": len(records),
                    "primary_path": str(rec.path),
                }
            )
    return index, pd.DataFrame(inventory_rows), pd.DataFrame(duplicate_rows)


def variant_metrics(primary: PreparedPrimary, clinical: PreparedVector) -> dict[str, float]:
    return {
        "corr_snv": pearson(primary.y_snv, clinical.y_snv),
        "rmse_snv": rmse(primary.y_snv, clinical.y_snv),
        "derivative_corr": pearson(primary.y_deriv_snv, clinical.y_deriv_snv),
        "intensity_ratio_mean_abs": clinical.fp_mean_abs / primary.fp_mean_abs
        if primary.fp_mean_abs
        else np.nan,
        "intensity_ratio_p95_abs": clinical.fp_p95_abs / primary.fp_p95_abs
        if primary.fp_p95_abs
        else np.nan,
        "clinical_fp_mean_abs": clinical.fp_mean_abs,
        "clinical_fp_p95_abs": clinical.fp_p95_abs,
    }


def compare_clinical_records(
    clinical_root: Path,
    primary_index: dict[tuple[str, str, str], list[PreparedPrimary]],
    calibrations: dict[str, DayCalibration],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    unmatched_rows: list[dict[str, object]] = []
    inventory_rows: list[dict[str, object]] = []

    clinical_paths = sample_files(clinical_root)
    for i, path in enumerate(clinical_paths, start=1):
        if i % 1000 == 0:
            print(f"  processed clinical spectra: {i}/{len(clinical_paths)}")
        key = parse_sample(path)
        if key is None:
            continue
        date_key = date_key_from_path(path)
        cal = calibrations.get(date_key)
        inv = {
            "dataset": "clinical_raw",
            "path": str(path),
            "group": key.group,
            "sample_id": key.sample_id,
            "replicate": key.replicate,
            "date_key": date_key,
            "parsed": True,
            "read_error": "",
        }
        try:
            x, y = read_spectrum(path)
        except Exception as exc:  # noqa: BLE001
            inv["read_error"] = str(exc)
            inventory_rows.append(inv)
            continue
        inventory_rows.append(inv)

        primaries = primary_index.get(key.replicate_key, [])
        if not primaries:
            unmatched_rows.append(
                {
                    "group": key.group,
                    "sample_id": key.sample_id,
                    "replicate": key.replicate,
                    "date_key": date_key,
                    "clinical_path": str(path),
                    "reason": "no_primary_raw_data_match",
                }
            )
            continue

        variant_vectors: dict[str, tuple[PreparedVector, dict[str, bool]]] = {}
        for variant in VARIANTS:
            x_corr, y_corr, flags = apply_reference_correction(x, y, cal, variant)
            variant_vectors[variant] = (prepare_vector(x_corr, y_corr), flags)

        raw_to_final_rmse = rmse(
            variant_vectors["clinical_raw"][0].y_snv,
            variant_vectors["axis_intensity_corrected"][0].y_snv,
        )

        for primary_idx, primary in enumerate(primaries, start=1):
            row: dict[str, object] = {
                "group": key.group,
                "sample_id": key.sample_id,
                "replicate": key.replicate,
                "date_key": date_key,
                "clinical_path": str(path),
                "primary_path": str(primary.path),
                "primary_candidate_index": primary_idx,
                "n_primary_candidates": len(primaries),
                "is_primary_duplicate_key": len(primaries) > 1,
                "primary_fp_mean_abs": primary.fp_mean_abs,
                "primary_fp_p95_abs": primary.fp_p95_abs,
                "calibration_axis_available": bool(cal.has_axis) if cal else False,
                "calibration_gain_available": bool(cal.gain_available) if cal else False,
                "calibration_blank_available": bool(cal.has_blank) if cal else False,
                "clinical_raw_to_final_rmse_snv": raw_to_final_rmse,
            }
            for variant, (vec, flags) in variant_vectors.items():
                metrics = variant_metrics(primary, vec)
                for metric, value in metrics.items():
                    row[f"{variant}_{metric}"] = value
                for flag, value in flags.items():
                    row[f"{variant}_{flag}"] = value
            row["final_corr_snv"] = row["axis_intensity_corrected_corr_snv"]
            row["final_rmse_snv"] = row["axis_intensity_corrected_rmse_snv"]
            row["final_intensity_ratio_mean_abs"] = row[
                "axis_intensity_corrected_intensity_ratio_mean_abs"
            ]
            row["final_corr_delta_vs_clinical_raw"] = (
                row["axis_intensity_corrected_corr_snv"] - row["clinical_raw_corr_snv"]
            )
            rows.append(row)

    return pd.DataFrame(rows), pd.DataFrame(unmatched_rows), pd.DataFrame(inventory_rows)


def select_best_primary_candidate(all_candidates: pd.DataFrame) -> pd.DataFrame:
    if all_candidates.empty:
        return all_candidates.copy()
    sort_cols = [
        "group",
        "sample_id",
        "replicate",
        "date_key",
        "clinical_path",
        "final_corr_snv",
        "clinical_raw_corr_snv",
    ]
    best = all_candidates.sort_values(
        sort_cols,
        ascending=[True, True, True, True, True, False, False],
        na_position="last",
    ).drop_duplicates(
        ["group", "sample_id", "replicate", "date_key", "clinical_path"], keep="first"
    )
    best = best.copy()
    best["primary_selection_rule"] = "highest_axis_intensity_corrected_corr_snv"
    return best.sort_values(["group", "sample_id", "date_key", "replicate"]).reset_index(drop=True)


def sample_summary(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    grouped = best.groupby(["group", "sample_id", "date_key"], dropna=False)
    for (group, sample_id, date_key), sub in grouped:
        row: dict[str, object] = {
            "group": group,
            "sample_id": sample_id,
            "date_key": date_key,
            "n_replicates": int(len(sub)),
            "n_duplicate_primary_replicates": int(sub["is_primary_duplicate_key"].sum()),
            "primary_paths": " | ".join(sorted(set(sub["primary_path"].astype(str)))),
            "clinical_paths": " | ".join(sorted(set(sub["clinical_path"].astype(str)))),
        }
        metric_cols = [
            "clinical_raw_corr_snv",
            "axis_corrected_corr_snv",
            "axis_intensity_corrected_corr_snv",
            "final_corr_snv",
            "final_corr_delta_vs_clinical_raw",
            "clinical_raw_rmse_snv",
            "final_rmse_snv",
            "clinical_raw_intensity_ratio_mean_abs",
            "axis_intensity_corrected_intensity_ratio_mean_abs",
            "final_intensity_ratio_mean_abs",
            "clinical_raw_to_final_rmse_snv",
        ]
        for col in metric_cols:
            if col in sub:
                row[f"{col}_median"] = float(sub[col].median())
                row[f"{col}_min"] = float(sub[col].min())
                row[f"{col}_max"] = float(sub[col].max())
        row["final_corr_snv_std"] = (
            float(sub["final_corr_snv"].std(ddof=0)) if len(sub) > 1 else 0.0
        )
        final_corr = float(sub["final_corr_snv"].median())
        intensity = float(sub["final_intensity_ratio_mean_abs"].median())
        intensity_term = (
            abs(np.log2(intensity)) if np.isfinite(intensity) and intensity > 0 else 2.0
        )
        row["outlier_score"] = float(
            max(0.0, 1.0 - final_corr) + 0.15 * intensity_term + 0.10 * row["final_corr_snv_std"]
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values("outlier_score", ascending=False).reset_index(drop=True)


def summarize_by_group(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    rows = []
    for group, sub in best.groupby("group", dropna=False):
        row = {
            "group": group,
            "n_replicates": int(len(sub)),
            "n_subjects": int(sub[["group", "sample_id"]].drop_duplicates().shape[0]),
        }
        for col in [
            "clinical_raw_corr_snv",
            "axis_corrected_corr_snv",
            "axis_intensity_corrected_corr_snv",
            "final_corr_delta_vs_clinical_raw",
            "clinical_raw_intensity_ratio_mean_abs",
            "final_intensity_ratio_mean_abs",
        ]:
            row[f"{col}_median"] = float(sub[col].median())
            row[f"{col}_p10"] = float(sub[col].quantile(0.10))
            row[f"{col}_p90"] = float(sub[col].quantile(0.90))
        rows.append(row)
    total = {
        "group": "ALL",
        "n_replicates": int(len(best)),
        "n_subjects": int(best[["group", "sample_id"]].drop_duplicates().shape[0]),
    }
    for col in [
        "clinical_raw_corr_snv",
        "axis_corrected_corr_snv",
        "axis_intensity_corrected_corr_snv",
        "final_corr_delta_vs_clinical_raw",
        "clinical_raw_intensity_ratio_mean_abs",
        "final_intensity_ratio_mean_abs",
    ]:
        total[f"{col}_median"] = float(best[col].median())
        total[f"{col}_p10"] = float(best[col].quantile(0.10))
        total[f"{col}_p90"] = float(best[col].quantile(0.90))
    rows.append(total)
    return pd.DataFrame(rows)


def preprocessing_variant_summary(best: pd.DataFrame) -> pd.DataFrame:
    if best.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    scopes: list[tuple[str, list[str]]] = [
        ("ALL", []),
        ("group", ["group"]),
        ("date", ["date_key"]),
        ("group_date", ["group", "date_key"]),
    ]
    for level, group_cols in scopes:
        grouped = [((), best)] if not group_cols else best.groupby(group_cols, dropna=False)
        for key, sub in grouped:
            if group_cols:
                if not isinstance(key, tuple):
                    key = (key,)
                base = dict(zip(group_cols, key, strict=True))
            else:
                base = {}
            for variant in VARIANTS:
                corr_col = f"{variant}_corr_snv"
                rmse_col = f"{variant}_rmse_snv"
                ratio_col = f"{variant}_intensity_ratio_mean_abs"
                row = {
                    "level": level,
                    **base,
                    "variant": variant,
                    "n_replicates": int(len(sub)),
                    "n_subjects": int(sub[["group", "sample_id"]].drop_duplicates().shape[0]),
                    "corr_snv_median": float(sub[corr_col].median()) if corr_col in sub else np.nan,
                    "corr_snv_p10": float(sub[corr_col].quantile(0.10))
                    if corr_col in sub
                    else np.nan,
                    "corr_snv_p90": float(sub[corr_col].quantile(0.90))
                    if corr_col in sub
                    else np.nan,
                    "rmse_snv_median": float(sub[rmse_col].median()) if rmse_col in sub else np.nan,
                    "intensity_ratio_mean_abs_median": float(sub[ratio_col].median())
                    if ratio_col in sub
                    else np.nan,
                }
                rows.append(row)
    return pd.DataFrame(rows)


def calibration_summary(
    calibrations: dict[str, DayCalibration],
    best: pd.DataFrame,
) -> pd.DataFrame:
    date_metrics = {}
    if not best.empty:
        for date_key, sub in best.groupby("date_key"):
            date_metrics[date_key] = {
                "n_matched_replicates": int(len(sub)),
                "n_matched_subjects": int(sub[["group", "sample_id"]].drop_duplicates().shape[0]),
                "clinical_raw_corr_snv_median": float(sub["clinical_raw_corr_snv"].median()),
                "final_corr_snv_median": float(sub["final_corr_snv"].median()),
                "final_corr_delta_vs_clinical_raw_median": float(
                    sub["final_corr_delta_vs_clinical_raw"].median()
                ),
                "clinical_raw_intensity_ratio_mean_abs_median": float(
                    sub["clinical_raw_intensity_ratio_mean_abs"].median()
                ),
                "final_intensity_ratio_mean_abs_median": float(
                    sub["final_intensity_ratio_mean_abs"].median()
                ),
            }

    rows = []
    for key, cal in sorted(calibrations.items()):
        gain = cal.gain_y if cal.gain_y is not None else np.ones_like(GRID)
        blank_mean = (
            float(np.nanmean(np.abs(np.interp(GRID, cal.blank_x, cal.blank_y))))
            if cal.has_blank
            else np.nan
        )
        row = {
            "date_key": key,
            "date_root": str(cal.root),
            "n_ps_files_read": cal.n_ps,
            "n_si_files_read": cal.n_si,
            "n_blank_files_read": cal.n_blank,
            "axis_available": cal.has_axis,
            "gain_available": cal.gain_available,
            "blank_available": cal.has_blank,
            "n_anchor_pairs": cal.n_anchor_pairs,
            "axis_slope": cal.slope,
            "axis_intercept": cal.intercept,
            "median_shift_expected_minus_observed_cm": cal.median_shift_expected_minus_observed,
            "max_abs_error_before_cm": cal.max_abs_error_before,
            "gain_min": float(np.nanmin(gain)),
            "gain_median": float(np.nanmedian(gain)),
            "gain_max": float(np.nanmax(gain)),
            "blank_mean_abs_on_grid": blank_mean,
            "note": cal.note,
        }
        row.update(date_metrics.get(key, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def safe_plot_name(group: str, sample_id: str, date_key: str, rank: int) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{group}_{sample_id}_{date_key}")
    return f"{rank:04d}_{label}.png"


def reset_png_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for png in path.glob("*.png"):
        png.unlink()


def clean_output_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("*.csv", "*.json"):
        for path in out_dir.glob(suffix):
            path.unlink()
    reset_png_dir(out_dir / "figures" / "sample_reports")
    reset_png_dir(out_dir / "figures" / "date_reports")


def centroid_vectors(
    paths: list[str], cal: DayCalibration | None = None, variant: str = "clinical_raw"
) -> PreparedVector | None:
    grids: list[np.ndarray] = []
    for path_s in paths:
        try:
            x, y = read_spectrum(Path(path_s))
            x, y, _ = apply_reference_correction(x, y, cal, variant)
            grids.append(prepare_vector(x, y).y_grid)
        except Exception:
            continue
    if not grids:
        return None
    y = np.vstack(grids).mean(axis=0)
    return prepare_vector(GRID, y)


def plot_sample_reports(
    sample_df: pd.DataFrame,
    best: pd.DataFrame,
    calibrations: dict[str, DayCalibration],
    out_dir: Path,
    max_plots: int,
) -> None:
    if sample_df.empty or max_plots < 0:
        return
    report_dir = out_dir / "figures" / "sample_reports"
    reset_png_dir(report_dir)
    plot_df = sample_df.copy()
    if max_plots > 0:
        plot_df = plot_df.head(max_plots)

    for rank, sample in enumerate(plot_df.itertuples(index=False), start=1):
        sub = best[
            (best["group"] == sample.group)
            & (best["sample_id"].astype(str) == str(sample.sample_id))
            & (best["date_key"] == sample.date_key)
        ]
        if sub.empty:
            continue
        primary_vec = centroid_vectors(sorted(set(sub["primary_path"].astype(str))))
        clinical_raw_vec = centroid_vectors(sorted(set(sub["clinical_path"].astype(str))))
        final_vec = centroid_vectors(
            sorted(set(sub["clinical_path"].astype(str))),
            cal=calibrations.get(sample.date_key),
            variant="axis_intensity_corrected",
        )
        if primary_vec is None or clinical_raw_vec is None or final_vec is None:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        ax = axes[0, 0]
        ax.plot(GRID, primary_vec.y_grid, color="#1f77b4", linewidth=1.0)
        ax.set_title("primary raw_data centroid")
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("intensity")

        ax = axes[0, 1]
        ax.plot(GRID, clinical_raw_vec.y_grid, color="#ff7f0e", linewidth=1.0)
        ax.set_title("clinical raw centroid")
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("intensity")

        ax = axes[1, 0]
        ax.plot(
            GRID, primary_vec.y_snv, color="#1f77b4", linewidth=1.0, label="primary baseline+SNV"
        )
        ax.plot(
            GRID,
            final_vec.y_snv,
            color="#2ca02c",
            linewidth=1.0,
            label="axis+intensity baseline+SNV",
        )
        ax.set_title("shape after baseline-corrected preprocessing")
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("SNV")
        ax.legend(fontsize=8)

        ax = axes[1, 1]
        ax.plot(GRID, final_vec.y_snv - primary_vec.y_snv, color="#d62728", linewidth=1.0)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_title("final preprocessed - primary (SNV)")
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("delta SNV")

        title = (
            f"{sample.group} {sample.sample_id} | {sample.date_key} | "
            f"raw corr {sample.clinical_raw_corr_snv_median:.3f} -> "
            f"final {sample.final_corr_snv_median:.3f} | "
            f"intensity ratio {sample.final_intensity_ratio_mean_abs_median:.2f}"
        )
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(
            report_dir / safe_plot_name(sample.group, str(sample.sample_id), sample.date_key, rank),
            dpi=150,
        )
        plt.close(fig)


def plot_date_reports(
    calibrations: dict[str, DayCalibration],
    best: pd.DataFrame,
    out_dir: Path,
) -> None:
    report_dir = out_dir / "figures" / "date_reports"
    reset_png_dir(report_dir)
    for date_key, cal in sorted(calibrations.items()):
        sub = best[best["date_key"] == date_key] if not best.empty else pd.DataFrame()
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        ax = axes[0, 0]
        if cal.ps_x is not None and cal.ps_y is not None:
            x_corr = apply_axis(cal.ps_x, cal)
            ax.plot(x_corr, reference_preprocess(cal.ps_y), linewidth=1.0, color="#1f77b4")
            for anchor in PS_ANCHORS:
                ax.axvline(anchor, color="#d62728", alpha=0.35, linewidth=0.8)
            ax.set_xlim(550, 1650)
        else:
            ax.text(
                0.5,
                0.5,
                "No readable PS reference",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        ax.set_title("PS reference after axis mapping")
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("baseline-corrected intensity")

        ax = axes[0, 1]
        if cal.gain_y is not None:
            ax.plot(GRID, cal.gain_y, linewidth=1.0, color="#2ca02c")
            ax.axhline(1, color="black", linewidth=0.7)
            ax.set_ylim(
                max(0, float(np.nanmin(cal.gain_y)) - 0.1), float(np.nanmax(cal.gain_y)) + 0.1
            )
        else:
            ax.text(0.5, 0.5, "No gain curve", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("PS-anchor intensity gain")
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("gain")

        ax = axes[1, 0]
        if cal.has_blank:
            blank_x = apply_axis(cal.blank_x, cal)
            ax.plot(blank_x, cal.blank_y, linewidth=1.0, color="#9467bd")
            ax.set_xlim(400, 1800)
        else:
            ax.text(0.5, 0.5, "No readable blank", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("blank mean")
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("intensity")

        ax = axes[1, 1]
        if not sub.empty:
            ax.hist(sub["clinical_raw_corr_snv"].dropna(), bins=30, alpha=0.6, label="clinical raw")
            ax.hist(
                sub["final_corr_snv"].dropna(),
                bins=30,
                alpha=0.5,
                label="axis+intensity baseline+SNV",
            )
            ax.legend(fontsize=8)
        else:
            ax.text(
                0.5, 0.5, "No matched samples", ha="center", va="center", transform=ax.transAxes
            )
        ax.set_title("matched replicate correlation")
        ax.set_xlabel("Pearson corr (SNV)")
        ax.set_ylabel("count")

        fig.suptitle(
            f"{date_key} | axis={cal.has_axis} gain={cal.gain_available} blank={cal.has_blank} | {cal.note}",
            fontsize=11,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(report_dir / f"{date_key}.png", dpi=150)
        plt.close(fig)


def write_inventory_summary(
    primary_inventory: pd.DataFrame,
    clinical_inventory: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    rows = []
    for name, inv in [
        ("primary_raw_data", primary_inventory),
        ("clinical_raw", clinical_inventory),
    ]:
        ok = inv[inv["read_error"].fillna("") == ""]
        for group, sub in ok.groupby("group", dropna=False):
            rows.append(
                {
                    "dataset": name,
                    "group": group,
                    "n_replicates": int(len(sub)),
                    "n_subjects": int(sub[["group", "sample_id"]].drop_duplicates().shape[0]),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "inventory_summary_by_group.csv", index=False, encoding="utf-8-sig")
    pd.concat([primary_inventory, clinical_inventory], ignore_index=True).to_csv(
        out_dir / "file_inventory.csv", index=False, encoding="utf-8-sig"
    )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-root", type=Path, default=DEFAULT_PRIMARY_ROOT)
    parser.add_argument("--clinical-root", type=Path, default=DEFAULT_CLINICAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--max-sample-plots",
        type=int,
        default=200,
        help="Maximum outlier sample report PNGs to write. Use 0 for all, negative to skip sample plots.",
    )
    args = parser.parse_args()
    clean_output_dir(args.output_dir)

    print("Building day-level PS/SI/blank calibrations...")
    calibrations, anchor_df = build_day_calibrations(args.clinical_root)
    anchor_df.to_csv(
        args.output_dir / "reference_anchor_detections.csv", index=False, encoding="utf-8-sig"
    )

    print("Indexing primary raw_data spectra...")
    primary_index, primary_inventory, primary_duplicates = build_primary_index(args.primary_root)
    primary_duplicates.to_csv(
        args.output_dir / "primary_duplicate_replicate_keys.csv", index=False, encoding="utf-8-sig"
    )

    print("Comparing clinical spectra against all matching primary candidates...")
    all_candidates, unmatched, clinical_inventory = compare_clinical_records(
        args.clinical_root,
        primary_index,
        calibrations,
    )
    all_candidates.to_csv(
        args.output_dir / "replicate_level_comparison_all_primary_candidates.csv",
        index=False,
        encoding="utf-8-sig",
    )
    unmatched.to_csv(
        args.output_dir / "clinical_unmatched_spectra.csv", index=False, encoding="utf-8-sig"
    )

    best = select_best_primary_candidate(all_candidates)
    best.to_csv(args.output_dir / "sample_level_comparison.csv", index=False, encoding="utf-8-sig")
    best.to_csv(
        args.output_dir / "replicate_level_best_match.csv", index=False, encoding="utf-8-sig"
    )

    samples = sample_summary(best)
    samples.insert(0, "outlier_rank", np.arange(1, len(samples) + 1))
    samples.to_csv(args.output_dir / "sample_level_summary.csv", index=False, encoding="utf-8-sig")
    samples.to_csv(args.output_dir / "outlier_samples.csv", index=False, encoding="utf-8-sig")

    group_summary = summarize_by_group(best)
    group_summary.to_csv(
        args.output_dir / "comparison_summary_by_group.csv", index=False, encoding="utf-8-sig"
    )
    variant_summary = preprocessing_variant_summary(best)
    variant_summary.to_csv(
        args.output_dir / "preprocessing_variant_summary.csv", index=False, encoding="utf-8-sig"
    )

    cal_summary = calibration_summary(calibrations, best)
    cal_summary.to_csv(
        args.output_dir / "date_calibration_summary.csv", index=False, encoding="utf-8-sig"
    )
    cal_summary.to_csv(
        args.output_dir / "reference_calibration_summary.csv", index=False, encoding="utf-8-sig"
    )

    inventory_summary = write_inventory_summary(
        primary_inventory, clinical_inventory, args.output_dir
    )

    print("Writing date and sample report figures...")
    plot_date_reports(calibrations, best, args.output_dir)
    plot_sample_reports(samples, best, calibrations, args.output_dir, args.max_sample_plots)

    overlap_subjects = (
        set(map(tuple, best[["group", "sample_id"]].drop_duplicates().to_numpy()))
        if not best.empty
        else set()
    )
    overlap_reps = (
        set(map(tuple, best[["group", "sample_id", "replicate"]].drop_duplicates().to_numpy()))
        if not best.empty
        else set()
    )
    report = {
        "primary_root": str(args.primary_root),
        "clinical_root": str(args.clinical_root),
        "output_dir": str(args.output_dir),
        "n_primary_replicates_read": int(
            len(primary_inventory[primary_inventory["read_error"].fillna("") == ""])
        ),
        "n_primary_unique_replicate_keys": int(len(primary_index)),
        "n_primary_duplicate_replicate_keys": int(
            primary_duplicates[["group", "sample_id", "replicate"]].drop_duplicates().shape[0]
        )
        if not primary_duplicates.empty
        else 0,
        "n_clinical_replicates_read": int(
            len(clinical_inventory[clinical_inventory["read_error"].fillna("") == ""])
        ),
        "n_matched_replicate_keys": int(len(overlap_reps)),
        "n_matched_subjects": int(len(overlap_subjects)),
        "n_all_primary_candidate_rows": int(len(all_candidates)),
        "n_best_match_rows": int(len(best)),
        "n_unmatched_clinical_replicates": int(len(unmatched)),
        "n_dates": int(len(calibrations)),
        "n_dates_axis_available": int(sum(cal.has_axis for cal in calibrations.values())),
        "n_dates_gain_available": int(sum(cal.gain_available for cal in calibrations.values())),
        "n_dates_blank_available": int(sum(cal.has_blank for cal in calibrations.values())),
        "variants": list(VARIANTS),
        "primary_selection_rule": "highest_axis_intensity_corrected_corr_snv when duplicate primary candidates exist",
        "notes": [
            "sample_level_comparison.csv is replicate-level after explicit best primary-candidate selection.",
            "replicate_level_comparison_all_primary_candidates.csv preserves every duplicate primary candidate comparison.",
            "sample_level_summary.csv aggregates selected replicates by group/sample/date.",
            "Comparison spectra are interpolated, baseline-corrected, and SNV-normalized before correlation/RMSE.",
            "Blank spectra are retained as date-level QC only and are not subtracted from samples.",
            "20260506 reference files are .SPA and are not readable by src.sers.io.read_spectrum in the current environment.",
        ],
        "inventory_summary_by_group": inventory_summary.to_dict(orient="records"),
    }
    with open(args.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(
        pd.DataFrame([report])
        .drop(columns=["notes", "inventory_summary_by_group"])
        .to_string(index=False)
    )
    if not group_summary.empty:
        print("\nGroup medians:")
        cols = [
            "group",
            "n_replicates",
            "clinical_raw_corr_snv_median",
            "axis_corrected_corr_snv_median",
            "axis_intensity_corrected_corr_snv_median",
            "clinical_raw_intensity_ratio_mean_abs_median",
            "final_intensity_ratio_mean_abs_median",
        ]
        print(group_summary[cols].to_string(index=False))
    print(f"\nOutputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
