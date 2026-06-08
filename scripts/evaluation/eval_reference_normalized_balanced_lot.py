#!/usr/bin/env python3
"""Evaluate reference-normalized Thermo balanced-lot spectra.

This is a feasibility check for day/lot normalization. It builds label-free
corrections from each acquisition day's PS/SI/blank folders and then scores the
current STK-V2 artifact. The artifact is not retrained, so these metrics show
the effect of applying reference normalization at inference time only.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment.sers_predict import StackingPredictor
from scripts.evaluation.eval_current_model_acquisition_sets import (
    CANCER_GROUPS,
    DATA_ROOT,
    SpectrumRecord,
    collect_records,
    predict_samples,
    score_frame,
)
from src.sers.io import read_spectrum
from src.sers.preprocessing import baseline_correction


PS_ANCHORS = np.array([620.9, 795.8, 1001.4, 1031.8, 1155.3, 1602.3], dtype=float)
SI_ANCHORS = np.array([520.7], dtype=float)
ALL_ANCHORS = np.concatenate([SI_ANCHORS, PS_ANCHORS])
VARIANTS = ("raw", "axis", "axis_intensity", "axis_intensity_blank")
BINARY_LABELS = ["Non-cancer", "Cancer"]

THERMO_STANDARD_ROOT = (
    DATA_ROOT / "03_sers_date_lot_balanced_acquisition" / "thermo" / "임상데이터"
)
THERMO_BORAMAE_ROOT = (
    DATA_ROOT
    / "03_sers_date_lot_balanced_acquisition"
    / "thermo"
    / "Thermo"
    / "20260602_Urine test"
)
DEFAULT_OUT = PROJECT_ROOT / "results" / "evaluation" / "reference_normalized_balanced_lot"


@dataclass
class DayCalibration:
    date_key: str
    root: Path
    slope: float = 1.0
    intercept: float = 0.0
    median_shift: float = 0.0
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
    gain_x: np.ndarray | None = None
    gain_y: np.ndarray | None = None
    ps_heights: dict[str, float] = field(default_factory=dict)
    used_identity: bool = False
    note: str = ""

    @property
    def has_axis(self) -> bool:
        return self.n_anchor_pairs > 0 and not self.used_identity

    @property
    def has_gain(self) -> bool:
        return self.gain_x is not None and self.gain_y is not None

    @property
    def has_blank(self) -> bool:
        return self.blank_x is not None and self.blank_y is not None


def date_key_from_path(path: Path) -> str:
    for part in reversed(path.parts):
        if re.match(r"^20\d{6}", part):
            return part
    return "unknown"


def date_roots() -> list[Path]:
    roots = []
    if THERMO_STANDARD_ROOT.exists():
        roots.extend(
            p for p in THERMO_STANDARD_ROOT.iterdir()
            if p.is_dir() and re.match(r"^20\d{6}", p.name)
        )
    if THERMO_BORAMAE_ROOT.exists():
        roots.append(THERMO_BORAMAE_ROOT)
    return sorted(roots, key=lambda p: p.name)


def readable_spectrum_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".csv", ".txt"}


def collect_material_files(root: Path, material: str) -> list[Path]:
    material_l = material.lower()
    if material_l == "blank":
        folders = [p for p in root.rglob("*") if p.is_dir() and "blank" in p.name.lower()]
        files = [f for folder in folders for f in folder.iterdir() if readable_spectrum_file(f)]
    else:
        folders = [
            p for p in root.rglob("*")
            if p.is_dir()
            and (
                "reference" in p.name.lower()
                or material_l in p.name.lower()
                or f"ref {material_l}" in p.name.lower()
            )
        ]
        files = [
            f for folder in folders
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


def reference_preprocess(y: np.ndarray) -> np.ndarray:
    y = savgol_filter(np.asarray(y, dtype=float), 11, 2, mode="interp")
    return baseline_correction(y, window=201)


def local_peak_center(x: np.ndarray, y: np.ndarray, anchor: float, half_width: float = 15.0) -> tuple[float, float] | None:
    mask = (x >= anchor - half_width) & (x <= anchor + half_width)
    if int(mask.sum()) < 5:
        return None
    xx = x[mask]
    yy = y[mask]
    k = int(np.argmax(yy))
    center = float(xx[k])
    if 0 < k < len(xx) - 1:
        try:
            coef = np.polyfit(xx[k - 1:k + 2], yy[k - 1:k + 2], 2)
            if coef[0] < 0:
                candidate = float(-coef[1] / (2 * coef[0]))
                if anchor - half_width <= candidate <= anchor + half_width:
                    center = candidate
        except Exception:
            pass
    return center, float(yy[k])


def anchor_centers(x: np.ndarray, y: np.ndarray, anchors: np.ndarray) -> list[tuple[float, float, float]]:
    yp = reference_preprocess(y)
    out = []
    for anchor in anchors:
        peak = local_peak_center(x, yp, float(anchor))
        if peak is None:
            continue
        center, height = peak
        out.append((float(anchor), center, height))
    return out


def anchor_heights(x: np.ndarray, y: np.ndarray, anchors: np.ndarray) -> dict[str, float]:
    yp = reference_preprocess(y)
    heights = {}
    for anchor in anchors:
        peak = local_peak_center(x, yp, float(anchor))
        if peak is not None:
            heights[f"{anchor:.1f}"] = peak[1]
    return heights


def fit_axis(cal: DayCalibration) -> None:
    pairs: list[tuple[float, float, float]] = []
    if cal.ps_x is not None and cal.ps_y is not None:
        pairs.extend(anchor_centers(cal.ps_x, cal.ps_y, PS_ANCHORS))
    if cal.si_x is not None and cal.si_y is not None:
        pairs.extend(anchor_centers(cal.si_x, cal.si_y, SI_ANCHORS))
    if not pairs:
        cal.used_identity = True
        cal.note = "no readable PS/SI reference"
        return

    expected = np.array([p[0] for p in pairs], dtype=float)
    observed = np.array([p[1] for p in pairs], dtype=float)
    cal.n_anchor_pairs = int(len(pairs))
    cal.median_shift = float(np.median(expected - observed))
    cal.max_abs_error_before = float(np.max(np.abs(expected - observed)))

    if len(pairs) >= 2:
        slope, intercept = np.polyfit(observed, expected, 1)
        if not (0.98 <= slope <= 1.02 and abs(intercept) <= 25):
            slope, intercept = 1.0, cal.median_shift
            cal.note = "affine rejected; median shift used"
    else:
        slope, intercept = 1.0, cal.median_shift
        cal.note = "single anchor; median shift used"
    cal.slope = float(slope)
    cal.intercept = float(intercept)


def apply_axis(x: np.ndarray, cal: DayCalibration) -> np.ndarray:
    return cal.slope * np.asarray(x, dtype=float) + cal.intercept


def build_day_calibrations(grid: np.ndarray) -> dict[str, DayCalibration]:
    cals: dict[str, DayCalibration] = {}
    for root in date_roots():
        cal = DayCalibration(date_key=root.name, root=root)

        ps = mean_spectrum(collect_material_files(root, "PS"))
        if ps is not None:
            cal.ps_x, cal.ps_y, cal.n_ps = ps
        si = mean_spectrum(collect_material_files(root, "Si"))
        if si is not None:
            cal.si_x, cal.si_y, cal.n_si = si
        blank = mean_spectrum(collect_material_files(root, "blank"))
        if blank is not None:
            cal.blank_x, cal.blank_y, cal.n_blank = blank

        fit_axis(cal)
        cals[cal.date_key] = cal

    height_rows = []
    for cal in cals.values():
        if cal.ps_x is None or cal.ps_y is None or cal.used_identity:
            continue
        x_corr = apply_axis(cal.ps_x, cal)
        heights = anchor_heights(x_corr, cal.ps_y, PS_ANCHORS)
        cal.ps_heights = heights
        if len(heights) == len(PS_ANCHORS):
            height_rows.append(heights)

    if height_rows:
        height_df = pd.DataFrame(height_rows)
        target = height_df.median(axis=0).to_dict()
    else:
        target = {}

    for cal in cals.values():
        if not target or not cal.ps_heights:
            cal.gain_x = grid.copy()
            cal.gain_y = np.ones_like(grid, dtype=float)
            continue
        xs = []
        gains = []
        for anchor in PS_ANCHORS:
            key = f"{anchor:.1f}"
            observed = float(cal.ps_heights.get(key, np.nan))
            tgt = float(target.get(key, np.nan))
            if np.isfinite(observed) and observed > 1e-9 and np.isfinite(tgt):
                xs.append(float(anchor))
                gains.append(float(np.clip(tgt / observed, 0.25, 4.0)))
        if len(xs) >= 2:
            cal.gain_x = grid.copy()
            cal.gain_y = np.interp(grid, np.array(xs), np.array(gains), left=gains[0], right=gains[-1])
            cal.gain_y = savgol_filter(cal.gain_y, 31, 2, mode="interp")
            cal.gain_y = np.clip(cal.gain_y, 0.25, 4.0)
        else:
            cal.gain_x = grid.copy()
            cal.gain_y = np.ones_like(grid, dtype=float)
            if not cal.note:
                cal.note = "insufficient PS anchors for gain"

    return cals


def calibration_table(cals: dict[str, DayCalibration], records: list[SpectrumRecord]) -> pd.DataFrame:
    n_by_date: dict[str, int] = {}
    for rec in records:
        key = date_key_from_path(rec.path)
        n_by_date[key] = n_by_date.get(key, 0) + 1

    rows = []
    for key, cal in sorted(cals.items()):
        gain = cal.gain_y if cal.gain_y is not None else np.ones(1)
        rows.append({
            "date_key": key,
            "root": str(cal.root),
            "n_spectra": n_by_date.get(key, 0),
            "n_ps": cal.n_ps,
            "n_si": cal.n_si,
            "n_blank": cal.n_blank,
            "n_anchor_pairs": cal.n_anchor_pairs,
            "slope": cal.slope,
            "intercept": cal.intercept,
            "median_shift": cal.median_shift,
            "max_abs_error_before": cal.max_abs_error_before,
            "gain_min": float(np.nanmin(gain)),
            "gain_max": float(np.nanmax(gain)),
            "gain_median": float(np.nanmedian(gain)),
            "used_identity": cal.used_identity,
            "note": cal.note,
        })
    return pd.DataFrame(rows)


def apply_reference_correction(
    x: np.ndarray,
    y: np.ndarray,
    cal: DayCalibration | None,
    variant: str,
) -> tuple[np.ndarray, np.ndarray]:
    if cal is None or variant == "raw":
        return x, y

    x_corr = apply_axis(x, cal)
    y_corr = np.asarray(y, dtype=float).copy()

    if "intensity" in variant and cal.has_gain:
        gain = np.interp(x_corr, cal.gain_x, cal.gain_y, left=cal.gain_y[0], right=cal.gain_y[-1])
        y_corr = y_corr * gain

    if "blank" in variant and cal.has_blank:
        blank_x = apply_axis(cal.blank_x, cal)
        blank_y = np.asarray(cal.blank_y, dtype=float)
        if "intensity" in variant and cal.has_gain:
            blank_gain = np.interp(blank_x, cal.gain_x, cal.gain_y, left=cal.gain_y[0], right=cal.gain_y[-1])
            blank_y = blank_y * blank_gain
        blank_on_sample = np.interp(x_corr, blank_x, blank_y, left=blank_y[0], right=blank_y[-1])
        y_corr = y_corr - blank_on_sample

    return x_corr, y_corr


def preprocess_records(
    predictor: StackingPredictor,
    records: list[SpectrumRecord],
    cals: dict[str, DayCalibration],
    variant: str,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    by_uid: dict[str, list[np.ndarray]] = {}
    meta: dict[str, dict[str, object]] = {}
    errors: list[dict[str, str]] = []

    for rec in tqdm(records, desc=f"preprocess:{variant}", unit="spectrum"):
        date_key = date_key_from_path(rec.path)
        cal = cals.get(date_key)
        try:
            x, y = read_spectrum(rec.path)
            x, y = apply_reference_correction(x, y, cal, variant)
            multichannel = predictor._preprocess_multichannel(x, y, instrument=rec.instrument)
        except Exception as exc:  # noqa: BLE001
            errors.append({"variant": variant, "path": str(rec.path), "error": str(exc)})
            continue

        by_uid.setdefault(rec.uid, []).append(multichannel.astype(np.float32))
        meta.setdefault(
            rec.uid,
            {
                "uid": rec.uid,
                "source": rec.source,
                "instrument": rec.instrument,
                "source_group": rec.source_group,
                "group": rec.group,
                "sample_id": rec.sample_id,
                "sample_key": rec.sample_key,
                "date_key": date_key,
            },
        )

    rows = []
    tensors = []
    for uid in sorted(by_uid):
        tensors.append(np.stack(by_uid[uid], axis=0).mean(axis=0))
        row = dict(meta[uid])
        row["n_replicates"] = len(by_uid[uid])
        rows.append(row)

    if not tensors:
        return np.empty((0, 3, len(predictor.grid))), pd.DataFrame(), pd.DataFrame(errors)
    return np.stack(tensors, axis=0), pd.DataFrame(rows), pd.DataFrame(errors)


def subset_scores(pred_df: pd.DataFrame, variant: str) -> pd.DataFrame:
    rows = []
    subsets = {
        "thermo_standard": pred_df["source"].eq("balanced_lot_thermo_standard"),
        "thermo_boramae": pred_df["source"].eq("balanced_lot_thermo_boramae_20260602"),
        "thermo_all": pred_df["source"].isin([
            "balanced_lot_thermo_standard",
            "balanced_lot_thermo_boramae_20260602",
        ]),
    }
    for name, mask in subsets.items():
        row = score_frame(pred_df[mask], f"{variant}_{name}")
        row["variant"] = variant
        row["cohort"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def write_confusion_matrices(pred_df: pd.DataFrame, out_dir: Path, variant: str) -> None:
    rows = []
    subsets = {
        "thermo_standard": pred_df["source"].eq("balanced_lot_thermo_standard"),
        "thermo_boramae": pred_df["source"].eq("balanced_lot_thermo_boramae_20260602"),
        "thermo_all": pred_df["source"].isin([
            "balanced_lot_thermo_standard",
            "balanced_lot_thermo_boramae_20260602",
        ]),
    }
    for cohort, mask in subsets.items():
        df = pred_df[mask]
        cm = confusion_matrix(df["true_binary"], df["pred_binary_balanced"], labels=[0, 1])
        pd.DataFrame(cm, index=BINARY_LABELS, columns=BINARY_LABELS).to_csv(
            out_dir / f"{variant}_{cohort}_binary_cm_counts.csv"
        )
        tn, fp, fn, tp = cm.ravel()
        rows.append({"variant": variant, "cohort": cohort, "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    pd.DataFrame(rows).to_csv(out_dir / f"{variant}_binary_cm_summary.csv", index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "usersnet" / "current")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictor = StackingPredictor(args.model_dir)

    records = []
    records.extend(
        collect_records(
            THERMO_STANDARD_ROOT,
            "balanced_lot_thermo_standard",
            "thermo",
            include_background_only=False,
        )
    )
    records.extend(
        collect_records(
            THERMO_BORAMAE_ROOT,
            "balanced_lot_thermo_boramae_20260602",
            "thermo",
            include_background_only=False,
        )
    )
    if not records:
        raise RuntimeError("No Thermo balanced-lot spectra collected")

    cals = build_day_calibrations(predictor.grid)
    cal_df = calibration_table(cals, records)
    cal_df.to_csv(args.output_dir / "reference_calibration_summary.csv", index=False)

    all_scores = []
    all_errors = []
    for variant in args.variants:
        X, meta, errors = preprocess_records(predictor, records, cals, variant)
        pred = predict_samples(predictor, X, meta)
        pred["variant"] = variant
        pred.to_csv(args.output_dir / f"{variant}_per_sample_predictions.csv", index=False)
        scores = subset_scores(pred, variant)
        scores.to_csv(args.output_dir / f"{variant}_performance_summary.csv", index=False)
        write_confusion_matrices(pred, args.output_dir, variant)
        all_scores.append(scores)
        if not errors.empty:
            all_errors.append(errors)

    score_df = pd.concat(all_scores, ignore_index=True)
    score_df.to_csv(args.output_dir / "performance_summary.csv", index=False)
    if all_errors:
        pd.concat(all_errors, ignore_index=True).to_csv(args.output_dir / "preprocess_errors.csv", index=False)
    else:
        pd.DataFrame(columns=["variant", "path", "error"]).to_csv(args.output_dir / "preprocess_errors.csv", index=False)

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_dir": str(args.model_dir),
        "mode": "balanced",
        "threshold": predictor.operating_modes.get("balanced", {}).get("threshold", 0.60),
        "n_spectra": len(records),
        "n_samples_source_preserved": len({r.uid for r in records}),
        "variants": list(args.variants),
        "notes": [
            "Corrections are estimated only from PS/SI/blank reference folders; no clinical labels are used.",
            "raw: production current-model preprocessing with no reference correction.",
            "axis: applies daily PS/SI affine wavenumber-axis correction before predictor preprocessing.",
            "axis_intensity: additionally applies PS-anchor empirical gain correction.",
            "axis_intensity_blank: additionally subtracts daily blank mean where available.",
            "The current model was trained without this reference-normalized preprocessing, so this is an inference-time feasibility check rather than a retrained validation.",
        ],
        "scores": score_df.to_dict(orient="records"),
        "reference_coverage": cal_df.to_dict(orient="records"),
    }
    with open(args.output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(score_df.to_string(index=False))
    print(f"Outputs: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
