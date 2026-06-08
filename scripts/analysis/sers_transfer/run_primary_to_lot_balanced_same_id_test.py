#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test whether primary-trained sample identities transfer to lot-balanced data.

Important: prefixes are preserved exactly. BPRO is not PRO, and BNOR is not NOR.

The test uses only exact overlapping sample IDs, e.g. PRO 7 in primary can be
tested against PRO 7 in lot-balanced data, but BPRO 7 is treated as a different
identity and is not tested against PRO 7.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.signal import savgol_filter


REPO_ROOT = Path(__file__).resolve().parents[3]
ROOT = Path(os.environ.get("SERS_ORGANIZED_DATA_ROOT", REPO_ROOT / "data" / "sers_clinical_organized")).expanduser()
PRIMARY_ROOT = ROOT / "02_sers_primary_pooled_acquisition" / "thermo" / "raw_data"
BALANCED_ROOTS = {
    "lot_balanced_thermo": ROOT / "03_sers_date_lot_balanced_acquisition" / "thermo",
    "lot_balanced_handheld": ROOT / "03_sers_date_lot_balanced_acquisition" / "handheld",
    "lot_balanced_medical": ROOT / "03_sers_date_lot_balanced_acquisition" / "medical",
}
OUT_DIR = ROOT / "06_analysis_results" / "primary_to_lot_balanced_same_id_test"
FIG_DIR = OUT_DIR / "figures"
INVALID_OLD_DIR = ROOT / "06_analysis_results" / "primary_to_lot_balanced_identity_test"
INVALID_ARCHIVE_DIR = ROOT / "06_analysis_results" / "INVALID_primary_to_lot_balanced_identity_test_bpro_assumption"

GRID = np.arange(400.0, 1800.1, 1.0)
PS_ANCHORS = (620.9, 795.8, 1001.4, 1031.8, 1155.3, 1602.3)
SI_ANCHORS = (520.7,)
REFERENCE_SEARCH_WINDOW_CM = 12.0
AXIS_ALIGNMENT_MODE = "none"
SAMPLE_RE = re.compile(
    r"(?i)^(?P<prefix>PO\.\s*)?"
    r"(?P<group>BPRO|BNOR|YNOR|YPAN|CPAN|SPAN|BLC|BRE|CRC|DIA|H\.?\s?D\.?|HBP|LUN|NOR|OVA|PRO)"
    r"\s+(?P<num>\d+)"
    r"(?:\s+NF)?"
    r"_(?P<rep>\d+|ave)(?:\(\d+\))?(?:_Sample_.*)?"
    r"\.(?:csv|txt)$"
)
SKIPPED_SPECTRA: list[dict[str, str]] = []
REFERENCE_SHIFT_INFO: dict[Path, dict[str, object]] = {}
REFERENCE_ANCHOR_ROWS: list[dict[str, object]] = []
REFERENCE_READ_ERRORS: list[dict[str, str]] = []


@dataclass(frozen=True)
class SpectrumRecord:
    path: Path
    acquisition: str
    dataset: str
    group: str
    sample_number: int
    sample_id: str
    replicate_id: str
    vector_snv: np.ndarray
    vector_derivative_snv: np.ndarray


def normalize_group(group: str) -> str:
    group = re.sub(r"\s+", "", group.upper())
    if group in {"H.D", "HD", "H.D."}:
        return "H.D."
    return group


def parse_sample(path: Path) -> tuple[str, int, str] | None:
    if "Background" in path.parts:
        return None
    if any(part in {"Averaged data", "Average data", "_prediction_results"} for part in path.parts):
        return None
    if re.search(r"(?i)(^|[\s_/.-])NF($|[\s_.-])", path.name):
        return None
    if re.search(r"(?i)(^|[\s_/.-])PO\.?\s+", path.name):
        return None
    match = SAMPLE_RE.match(path.name)
    if not match:
        return None
    replicate_id = match.group("rep").lower()
    if replicate_id == "ave":
        return None
    group = normalize_group(match.group("group"))
    if group in {"BPRO", "BNOR"}:
        return None
    return group, int(match.group("num")), replicate_id


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix.lower() == ".txt":
        try:
            arr = np.genfromtxt(path, usecols=(0, 1), invalid_raise=False)
        except Exception as exc:  # noqa: BLE001 - include path in data parsing errors
            raise ValueError(f"Could not read 2-column TXT spectrum from {path}") from exc
        return clean_numeric_spectrum(arr, path)

    if is_handheld_metadata_csv(path):
        return read_handheld_metadata_csv(path)

    try:
        arr = np.genfromtxt(path, delimiter=",", usecols=(0, 1), invalid_raise=False)
    except Exception as exc:  # noqa: BLE001 - include path in data parsing errors
        raise ValueError(f"Could not read 2-column CSV spectrum from {path}") from exc
    return clean_numeric_spectrum(arr, path)


def clean_numeric_spectrum(arr: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected 2-column spectrum: {path}")
    arr = arr[np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])]
    if len(arr) < 10:
        raise ValueError(f"Too few valid numeric spectrum rows: {path}")
    x = arr[:, 0].astype(float)
    y = arr[:, 1].astype(float)
    order = np.argsort(x)
    return x[order], y[order]


def is_handheld_metadata_csv(path: Path) -> bool:
    if path.suffix.lower() != ".csv":
        return False
    try:
        with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
            first = handle.readline()
    except OSError:
        return False
    return first.startswith('"Name"') or first.startswith("Name,")


def read_handheld_metadata_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    metadata: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) >= 2:
                metadata[row[0]] = row[1]
    try:
        first = float(metadata["Firstwavenumber"])
        last = float(metadata["LastWavenumber"])
        intensities = np.array([float(v) for v in metadata["Intensities"].split(",")], dtype=float)
    except Exception as exc:  # noqa: BLE001 - include path in data parsing errors
        raise ValueError(f"Could not parse handheld metadata spectrum from {path}") from exc
    if len(intensities) < 10:
        raise ValueError(f"Too few handheld intensity values: {path}")
    x = np.linspace(first, last, len(intensities))
    return x, intensities


def snv(y: np.ndarray) -> np.ndarray:
    mean = float(np.nanmean(y))
    std = float(np.nanstd(y))
    if not np.isfinite(std) or std == 0:
        return y - mean
    return (y - mean) / std


def find_lot_balanced_date_root(path: Path) -> Path | None:
    try:
        path.relative_to(ROOT / "03_sers_date_lot_balanced_acquisition")
    except ValueError:
        return None
    for parent in path.parents:
        name = parent.name
        if re.match(r"^20\d{6}(?:_.+)?$", name):
            return parent
    return None


def reference_candidate_files(date_root: Path) -> list[tuple[str, Path]]:
    folder_specs: list[tuple[str | None, Path]] = [
        (None, date_root / "0. Reference"),
        (None, date_root / "0.Reference"),
        ("PS", date_root / "0. Ref_PS"),
        ("SI", date_root / "0. Ref_Si"),
        ("PS", date_root / "0. PS"),
        ("SI", date_root / "0. Si"),
    ]
    candidates: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for material_hint, folder in folder_specs:
        if not folder.exists():
            continue
        paths = sorted(list(folder.rglob("*.[Cc][Ss][Vv]")) + list(folder.rglob("*.txt")))
        for path in paths:
            if path in seen:
                continue
            if "Background" in path.parts:
                continue
            if "ave" in path.stem.lower():
                continue
            material = material_hint
            if material is None:
                stem = path.stem.strip().upper()
                if stem.startswith("PS"):
                    material = "PS"
                elif stem.startswith("SI"):
                    material = "SI"
            if material in {"PS", "SI"}:
                candidates.append((material, path))
                seen.add(path)
    return candidates


def smooth_for_peak_detection(y: np.ndarray) -> np.ndarray:
    if len(y) < 7:
        return y
    window = min(11, len(y) if len(y) % 2 == 1 else len(y) - 1)
    if window < 7:
        return y
    return savgol_filter(y, window_length=window, polyorder=2, mode="interp")


def detect_reference_peak(x: np.ndarray, y: np.ndarray, expected_cm: float) -> tuple[float, float] | None:
    mask = (x >= expected_cm - REFERENCE_SEARCH_WINDOW_CM) & (x <= expected_cm + REFERENCE_SEARCH_WINDOW_CM)
    if int(mask.sum()) < 3:
        return None
    x_window = x[mask]
    y_window = smooth_for_peak_detection(y[mask])
    peak_idx = int(np.nanargmax(y_window))
    observed_cm = float(x_window[peak_idx])
    return observed_cm, observed_cm - expected_cm


def estimate_reference_shift(date_root: Path) -> dict[str, object]:
    if date_root in REFERENCE_SHIFT_INFO:
        return REFERENCE_SHIFT_INFO[date_root]

    files = reference_candidate_files(date_root)
    shifts: list[float] = []
    materials = sorted({material for material, _ in files})
    for material, path in files:
        anchors = PS_ANCHORS if material == "PS" else SI_ANCHORS
        try:
            x, y = read_spectrum(path)
        except ValueError as exc:
            REFERENCE_READ_ERRORS.append(
                {
                    "date_root": str(date_root),
                    "material": material,
                    "path": str(path),
                    "reason": str(exc),
                }
            )
            continue
        for expected_cm in anchors:
            detected = detect_reference_peak(x, y, expected_cm)
            if detected is None:
                REFERENCE_ANCHOR_ROWS.append(
                    {
                        "date_root": str(date_root),
                        "date_label": date_root.name,
                        "material": material,
                        "reference_file": str(path),
                        "expected_cm": expected_cm,
                        "observed_cm": "",
                        "shift_cm": "",
                        "status": "not_in_range_or_no_peak",
                    }
                )
                continue
            observed_cm, shift_cm = detected
            shifts.append(shift_cm)
            REFERENCE_ANCHOR_ROWS.append(
                {
                    "date_root": str(date_root),
                    "date_label": date_root.name,
                    "material": material,
                    "reference_file": str(path),
                    "expected_cm": expected_cm,
                    "observed_cm": observed_cm,
                    "shift_cm": shift_cm,
                    "status": "ok",
                }
            )

    if shifts:
        median_shift = float(np.median(shifts))
        status = "ok"
    elif files:
        median_shift = 0.0
        status = "no_valid_reference_anchor"
    else:
        median_shift = 0.0
        status = "no_reference_files"

    info: dict[str, object] = {
        "date_root": str(date_root),
        "date_label": date_root.name,
        "reference_file_count": len(files),
        "reference_materials": ",".join(materials),
        "valid_anchor_count": len(shifts),
        "median_shift_cm": median_shift,
        "status": status,
    }
    REFERENCE_SHIFT_INFO[date_root] = info
    return info


def axis_shift_for_path(path: Path) -> float:
    if AXIS_ALIGNMENT_MODE != "reference_shift":
        return 0.0
    date_root = find_lot_balanced_date_root(path)
    if date_root is None:
        return 0.0
    info = estimate_reference_shift(date_root)
    if info["status"] != "ok":
        return 0.0
    return float(info["median_shift_cm"])


def preprocess(path: Path) -> tuple[np.ndarray, np.ndarray]:
    x, y = read_spectrum(path)
    shift_cm = axis_shift_for_path(path)
    if shift_cm:
        x = x - shift_cm
    interp = np.interp(GRID, x, y)
    vector_snv = snv(interp)
    smoothed = savgol_filter(interp, window_length=15, polyorder=3, mode="interp")
    derivative = savgol_filter(smoothed, window_length=15, polyorder=3, deriv=1, delta=1.0, mode="interp")
    return vector_snv, snv(derivative)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_records(root: Path, acquisition: str, dataset: str) -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    paths = sorted(list(root.rglob("*.[Cc][Ss][Vv]")) + list(root.rglob("*.txt")))
    for path in paths:
        parsed = parse_sample(path)
        if parsed is None:
            continue
        group, sample_number, replicate_id = parsed
        try:
            vector_snv, vector_derivative_snv = preprocess(path)
        except ValueError as exc:
            SKIPPED_SPECTRA.append(
                {
                    "dataset": dataset,
                    "path": str(path),
                    "reason": str(exc),
                }
            )
            continue
        records.append(
            SpectrumRecord(
                path=path,
                acquisition=acquisition,
                dataset=dataset,
                group=group,
                sample_number=sample_number,
                sample_id=f"{group} {sample_number}",
                replicate_id=replicate_id,
                vector_snv=vector_snv,
                vector_derivative_snv=vector_derivative_snv,
            )
        )
    return records


def centroid_matrix(records: list[SpectrumRecord], feature_name: str, candidate_ids: set[str]) -> tuple[list[str], np.ndarray]:
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for record in records:
        if record.sample_id in candidate_ids:
            grouped[record.sample_id].append(getattr(record, feature_name))
    labels = sorted(grouped, key=lambda sid: (sid.split()[0], int(sid.split()[1])))
    matrix = np.vstack([np.mean(grouped[label], axis=0) for label in labels])
    return labels, l2_normalize(matrix)


def predict_against_centroids(
    train_records: list[SpectrumRecord],
    test_records: list[SpectrumRecord],
    feature_name: str,
    candidate_ids: set[str],
    candidate_mode: str,
) -> list[dict[str, object]]:
    labels, centroids = centroid_matrix(train_records, feature_name, candidate_ids)
    label_index = {label: idx for idx, label in enumerate(labels)}
    rows: list[dict[str, object]] = []
    for record in test_records:
        if record.sample_id not in label_index:
            continue
        x = l2_normalize(getattr(record, feature_name).reshape(1, -1))
        sims = (x @ centroids.T).ravel()
        order = np.argsort(-sims)
        pred = labels[int(order[0])]
        true_rank = int(np.where(order == label_index[record.sample_id])[0][0]) + 1
        rows.append(
            {
                "feature": feature_name,
                "candidate_mode": candidate_mode,
                "test_dataset": record.dataset,
                "true_sample_id": record.sample_id,
                "true_group": record.group,
                "replicate_id": record.replicate_id,
                "pred_sample_id": pred,
                "pred_group": pred.split()[0],
                "correct_identity": pred == record.sample_id,
                "correct_group": pred.split()[0] == record.group,
                "true_rank": true_rank,
                "top3_contains_true": true_rank <= 3,
                "top5_contains_true": true_rank <= 5,
                "true_similarity": float(sims[label_index[record.sample_id]]),
                "top1_similarity": float(sims[order[0]]),
                "top5_sample_ids": "|".join(labels[int(i)] for i in order[:5]),
                "path": str(record.path),
            }
        )
    return rows


def primary_leave_one_replicate_cv(
    primary_records: list[SpectrumRecord],
    feature_name: str,
    candidate_ids: set[str],
    candidate_mode: str,
) -> list[dict[str, object]]:
    records = [record for record in primary_records if record.sample_id in candidate_ids]
    labels = sorted(candidate_ids, key=lambda sid: (sid.split()[0], int(sid.split()[1])))
    grouped: dict[str, list[SpectrumRecord]] = defaultdict(list)
    for record in records:
        grouped[record.sample_id].append(record)

    label_index = {label: idx for idx, label in enumerate(labels)}
    sums = np.zeros((len(labels), len(GRID)), dtype=float)
    counts = np.zeros(len(labels), dtype=int)
    for record in records:
        idx = label_index[record.sample_id]
        sums[idx] += getattr(record, feature_name)
        counts[idx] += 1

    base_centroids = sums / counts[:, None]
    base_centroids = l2_normalize(base_centroids)
    x_matrix = l2_normalize(np.vstack([getattr(record, feature_name) for record in records]))
    sims_matrix = x_matrix @ base_centroids.T

    rows: list[dict[str, object]] = []
    for row_idx, record in enumerate(records):
        true_idx = label_index[record.sample_id]
        sims = sims_matrix[row_idx].copy()
        if counts[true_idx] > 1:
            adjusted = (sums[true_idx] - getattr(record, feature_name)) / (counts[true_idx] - 1)
            adjusted = l2_normalize(adjusted.reshape(1, -1))[0]
            sims[true_idx] = float(x_matrix[row_idx] @ adjusted)
        order = np.argsort(-sims)
        pred = labels[int(order[0])]
        true_rank = int(np.where(order == true_idx)[0][0]) + 1
        rows.append(
            {
                "feature": feature_name,
                "candidate_mode": candidate_mode,
                "test_dataset": "primary_leave_one_replicate_cv",
                "true_sample_id": record.sample_id,
                "true_group": record.group,
                "replicate_id": record.replicate_id,
                "pred_sample_id": pred,
                "pred_group": pred.split()[0],
                "correct_identity": pred == record.sample_id,
                "correct_group": pred.split()[0] == record.group,
                "true_rank": true_rank,
                "top3_contains_true": true_rank <= 3,
                "top5_contains_true": true_rank <= 5,
                "true_similarity": float(sims[true_idx]),
                "top1_similarity": float(sims[order[0]]),
                "top5_sample_ids": "|".join(labels[int(i)] for i in order[:5]),
                "path": str(record.path),
            }
        )
    return rows


def sample_majority_rows(predictions: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped = defaultdict(list)
    for row in predictions:
        key = (row["feature"], row["candidate_mode"], row["test_dataset"], row["true_sample_id"])
        grouped[key].append(row)

    rows: list[dict[str, object]] = []
    for (feature, candidate_mode, test_dataset, true_sample_id), items in sorted(grouped.items()):
        counts = Counter(str(item["pred_sample_id"]) for item in items)
        pred, votes = counts.most_common(1)[0]
        true_group = str(items[0]["true_group"])
        rows.append(
            {
                "feature": feature,
                "candidate_mode": candidate_mode,
                "test_dataset": test_dataset,
                "true_sample_id": true_sample_id,
                "true_group": true_group,
                "n_replicates": len(items),
                "majority_pred_sample_id": pred,
                "majority_vote_fraction": votes / len(items),
                "majority_correct_identity": pred == true_sample_id,
                "majority_correct_group": pred.split()[0] == true_group,
                "replicate_identity_accuracy": np.mean([bool(item["correct_identity"]) for item in items]),
                "median_true_rank": float(np.median([int(item["true_rank"]) for item in items])),
                "pred_counts": ";".join(f"{k}:{v}" for k, v in counts.most_common()),
            }
        )
    return rows


def summary_rows(predictions: list[dict[str, object]], sample_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    grouped = defaultdict(list)
    for row in predictions:
        grouped[(row["feature"], row["candidate_mode"], row["test_dataset"])].append(row)
    for (feature, candidate_mode, test_dataset), items in sorted(grouped.items()):
        rows.append(
            {
                "level": "replicate",
                "feature": feature,
                "candidate_mode": candidate_mode,
                "test_dataset": test_dataset,
                "n": len(items),
                "identity_top1_accuracy": np.mean([bool(item["correct_identity"]) for item in items]),
                "identity_top3_accuracy": np.mean([bool(item["top3_contains_true"]) for item in items]),
                "identity_top5_accuracy": np.mean([bool(item["top5_contains_true"]) for item in items]),
                "group_accuracy": np.mean([bool(item["correct_group"]) for item in items]),
                "median_true_rank": float(np.median([int(item["true_rank"]) for item in items])),
            }
        )

    grouped_samples = defaultdict(list)
    for row in sample_rows:
        grouped_samples[(row["feature"], row["candidate_mode"], row["test_dataset"])].append(row)
    for (feature, candidate_mode, test_dataset), items in sorted(grouped_samples.items()):
        rows.append(
            {
                "level": "sample_majority",
                "feature": feature,
                "candidate_mode": candidate_mode,
                "test_dataset": test_dataset,
                "n": len(items),
                "identity_top1_accuracy": np.mean([bool(item["majority_correct_identity"]) for item in items]),
                "identity_top3_accuracy": "",
                "identity_top5_accuracy": "",
                "group_accuracy": np.mean([bool(item["majority_correct_group"]) for item in items]),
                "median_true_rank": float(np.median([float(item["median_true_rank"]) for item in items])),
            }
        )
    return rows


def write_report(primary_records: list[SpectrumRecord], balanced_by_dataset: dict[str, list[SpectrumRecord]], overlap_by_dataset: dict[str, set[str]], summary: pd.DataFrame) -> None:
    main = summary[
        (summary["feature"] == "vector_derivative_snv")
        & (summary["candidate_mode"] == "dataset_overlap_only")
    ]
    if AXIS_ALIGNMENT_MODE == "reference_shift":
        axis_text = (
            "Lot-balanced spectra were x-axis corrected by subtracting the "
            "date-level median PS/Si reference peak shift. Primary spectra were "
            "left unchanged because no primary PS/Si reference folders were found "
            "in the organized copy."
        )
    else:
        axis_text = "No reference-based x-axis correction was applied."
    lines = [
        "# Corrected Primary-to-Lot-Balanced Same-ID Test",
        "",
        "## Correction",
        "",
        "`BPRO` and `BNOR` are treated as distinct groups, not as `PRO`/`NOR`.",
        "The previous `BPRO -> PRO` assumption was invalid and the old output folder was archived as invalid.",
        "",
        "## Setup",
        "",
        "- Train/source acquisition: primary pooled Thermo raw data.",
        "- Test acquisitions: lot-balanced Thermo, Handheld, and Medical folders.",
        "- Candidate labels: exact overlapping sample IDs only per test dataset.",
        "- Excluded: averaged files, NF files, Po./post-op files, BPRO, BNOR, blank/reference/QC files.",
        "- Preprocessing: interpolate 400-1800 cm-1, SNV, and Savitzky-Golay first derivative + SNV.",
        f"- Axis alignment: {axis_text}",
        "",
        "## Data Used",
        "",
        f"- Primary replicate spectra: {len(primary_records)}",
        f"- Primary sample IDs: {len({r.sample_id for r in primary_records})}",
    ]
    for dataset, records in balanced_by_dataset.items():
        overlap = overlap_by_dataset[dataset]
        lines.append(f"- {dataset}: {len(records)} replicate spectra, {len({r.sample_id for r in records})} sample IDs, {len(overlap)} exact overlaps")
    lines.extend(["", "## Main Metrics", ""])
    if main.empty:
        lines.append("No exact-overlap rows were available.")
    else:
        headers = list(main.columns)
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for _, row in main.iterrows():
            values = []
            for header in headers:
                value = row[header]
                if isinstance(value, float):
                    value = f"{value:.4f}"
                values.append(str(value))
            lines.append("| " + " | ".join(values) + " |")
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `figures/01_accuracy_summary.png`",
            "- `figures/02_true_rank_distribution.png`",
            "- `figures/03_group_same_id_heatmap.png`",
            "- `figures/04_group_confusion_matrices.png`",
            "- `figures/05_before_after_axis_alignment_accuracy.png` when a baseline comparison is available.",
            "- `figures/06_before_after_axis_alignment_rank.png` when a baseline comparison is available.",
            "",
            "## Interpretation Guide",
            "",
            "Compare the primary leave-one-replicate control with lot-balanced test rows.",
            "If primary CV is high but lot-balanced same-ID accuracy is low, the same sample",
            "identity is not stable across acquisition/date/lot/instrument under this preprocessing.",
            "If disease/group accuracy remains high, disease-level signal may still transfer even",
            "when subject identity does not.",
        ]
    )
    (OUT_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_axis_alignment_tables() -> None:
    if AXIS_ALIGNMENT_MODE != "reference_shift":
        return
    pd.DataFrame(list(REFERENCE_SHIFT_INFO.values())).sort_values("date_label").to_csv(
        OUT_DIR / "reference_axis_shift_by_date.csv",
        index=False,
    )
    pd.DataFrame(REFERENCE_ANCHOR_ROWS).to_csv(OUT_DIR / "reference_axis_anchor_detections.csv", index=False)
    pd.DataFrame(REFERENCE_READ_ERRORS).to_csv(OUT_DIR / "reference_axis_read_errors.csv", index=False)


def derivative_sample_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    return summary_df[
        (summary_df["feature"] == "vector_derivative_snv")
        & (summary_df["level"] == "sample_majority")
        & (
            (summary_df["candidate_mode"] == "dataset_overlap_only")
            | (summary_df["test_dataset"] == "primary_leave_one_replicate_cv")
        )
    ].copy()


def write_before_after_comparison(baseline_summary_path: Path | None, summary_df: pd.DataFrame) -> None:
    if baseline_summary_path is None or not baseline_summary_path.exists():
        return

    baseline = pd.read_csv(baseline_summary_path)
    before = derivative_sample_summary(baseline)
    after = derivative_sample_summary(summary_df)
    if before.empty or after.empty:
        return

    before["alignment"] = "Before alignment"
    after["alignment"] = "PS/Si axis-aligned"
    compare = pd.concat([before, after], ignore_index=True)
    compare.to_csv(OUT_DIR / "before_after_axis_alignment_summary.csv", index=False)

    display_names = {
        "primary_leave_one_replicate_cv": "Primary CV",
        "lot_balanced_thermo": "Thermo",
        "lot_balanced_handheld": "Handheld",
        "lot_balanced_medical": "Medical",
    }
    dataset_order = ["Primary CV", "Thermo", "Handheld", "Medical"]
    compare["dataset_label"] = compare["test_dataset"].map(display_names).fillna(compare["test_dataset"])
    compare["identity_top1_accuracy"] = pd.to_numeric(compare["identity_top1_accuracy"], errors="coerce")
    compare["group_accuracy"] = pd.to_numeric(compare["group_accuracy"], errors="coerce")
    compare["median_true_rank"] = pd.to_numeric(compare["median_true_rank"], errors="coerce")
    order = [label for label in dataset_order if label in set(compare["dataset_label"])]

    metric_rows = []
    for _, row in compare.iterrows():
        metric_rows.append(
            {
                "dataset_label": row["dataset_label"],
                "alignment": row["alignment"],
                "metric": "Same-ID accuracy",
                "value": row["identity_top1_accuracy"],
            }
        )
        metric_rows.append(
            {
                "dataset_label": row["dataset_label"],
                "alignment": row["alignment"],
                "metric": "Disease/group accuracy",
                "value": row["group_accuracy"],
            }
        )
    metric_df = pd.DataFrame(metric_rows)

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.5), sharey=True)
    for ax, metric in zip(axes, ["Same-ID accuracy", "Disease/group accuracy"]):
        subset = metric_df[metric_df["metric"] == metric]
        sns.barplot(
            data=subset,
            x="dataset_label",
            y="value",
            hue="alignment",
            order=order,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Sample-majority accuracy" if metric == "Same-ID accuracy" else "")
        ax.set_title(metric)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=8)
    axes[0].legend_.remove()
    axes[1].legend(
        title="Alignment",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        frameon=False,
    )
    fig.suptitle("Effect of PS/Si reference x-axis alignment")
    fig.subplots_adjust(bottom=0.23, left=0.07, right=0.99, top=0.86, wspace=0.10)
    plt.savefig(FIG_DIR / "05_before_after_axis_alignment_accuracy.png", dpi=200)
    plt.close()

    rank_compare = compare[compare["test_dataset"] != "primary_leave_one_replicate_cv"].copy()
    if rank_compare.empty:
        return
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    sns.barplot(
        data=rank_compare,
        x="dataset_label",
        y="median_true_rank",
        hue="alignment",
        order=[label for label in dataset_order if label in set(rank_compare["dataset_label"])],
        ax=ax,
    )
    ax.set_yscale("log")
    ax.set_xlabel("")
    ax.set_ylabel("Median true sample rank, log scale")
    ax.set_title("Correct sample rank before and after x-axis alignment")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f", fontsize=8)
    ax.legend(
        title="Alignment",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.25, left=0.10, right=0.98, top=0.88)
    plt.savefig(FIG_DIR / "06_before_after_axis_alignment_rank.png", dpi=200)
    plt.close()


def write_figures(pred_df: pd.DataFrame, sample_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    title_suffix = " with PS/Si x-axis alignment" if AXIS_ALIGNMENT_MODE == "reference_shift" else ""
    display_names = {
        "primary_leave_one_replicate_cv": "Primary CV",
        "lot_balanced_thermo": "Thermo",
        "lot_balanced_handheld": "Handheld",
        "lot_balanced_medical": "Medical",
    }
    dataset_order = ["Primary CV", "Thermo", "Handheld", "Medical"]

    def add_display_dataset(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["dataset_label"] = df["test_dataset"].map(display_names).fillna(df["test_dataset"])
        return df

    derivative_summary = summary_df[
        (summary_df["feature"] == "vector_derivative_snv")
        & (summary_df["level"] == "sample_majority")
        & (
            (summary_df["candidate_mode"] == "dataset_overlap_only")
            | (summary_df["test_dataset"] == "primary_leave_one_replicate_cv")
        )
    ].copy()
    if not derivative_summary.empty:
        plot_rows = []
        for _, row in derivative_summary.iterrows():
            plot_rows.append(
                {
                    "dataset": row["test_dataset"],
                    "metric": "Same-ID accuracy",
                    "accuracy": float(row["identity_top1_accuracy"]),
                    "n": int(row["n"]),
                }
            )
            plot_rows.append(
                {
                    "dataset": row["test_dataset"],
                    "metric": "Disease/group accuracy",
                    "accuracy": float(row["group_accuracy"]),
                    "n": int(row["n"]),
                }
            )
        plot_df = pd.DataFrame(plot_rows).rename(columns={"dataset": "test_dataset"})
        plot_df = add_display_dataset(plot_df)
        order = [label for label in dataset_order if label in set(plot_df["dataset_label"])]
        fig, ax = plt.subplots(figsize=(13.5, 7.2))
        sns.barplot(
            data=plot_df,
            x="dataset_label",
            y="accuracy",
            hue="metric",
            order=order,
            ax=ax,
        )
        ax.set_ylim(0, 1)
        ax.set_xlabel("")
        ax.set_ylabel("Sample-majority accuracy")
        ax.set_title(f"Same-sample identity transfer outside the primary acquisition{title_suffix}")
        ax.legend(
            title="Metric",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=2,
            frameon=False,
        )
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=9)
        fig.subplots_adjust(bottom=0.24, left=0.08, right=0.98, top=0.90)
        plt.savefig(FIG_DIR / "01_accuracy_summary.png", dpi=200)
        plt.close()

    derivative_pred = pred_df[
        (pred_df["feature"] == "vector_derivative_snv")
        & (
            (pred_df["candidate_mode"] == "dataset_overlap_only")
            | (pred_df["test_dataset"] == "primary_leave_one_replicate_cv")
        )
    ].copy()
    if not derivative_pred.empty:
        derivative_pred = add_display_dataset(derivative_pred)
        order = [label for label in dataset_order if label in set(derivative_pred["dataset_label"])]
        fig, ax = plt.subplots(figsize=(13.5, 7.2))
        sns.boxplot(
            data=derivative_pred,
            x="dataset_label",
            y="true_rank",
            order=order,
            showfliers=False,
            ax=ax,
        )
        sns.stripplot(
            data=derivative_pred.sample(min(len(derivative_pred), 2500), random_state=7),
            x="dataset_label",
            y="true_rank",
            order=order,
            color="0.25",
            alpha=0.18,
            size=2,
            ax=ax,
        )
        ax.set_yscale("log")
        ax.set_xlabel("")
        ax.set_ylabel("Rank of the true sample ID, log scale")
        ax.set_title(f"Correct sample rank after primary-to-lot-balanced transfer{title_suffix}")
        fig.subplots_adjust(bottom=0.14, left=0.09, right=0.98, top=0.90)
        plt.savefig(FIG_DIR / "02_true_rank_distribution.png", dpi=200)
        plt.close()

    derivative_sample = sample_df[
        (sample_df["feature"] == "vector_derivative_snv")
        & (sample_df["candidate_mode"] == "dataset_overlap_only")
    ].copy()
    if not derivative_sample.empty:
        derivative_sample = add_display_dataset(derivative_sample)
        group_acc = (
            derivative_sample.groupby(["true_group", "test_dataset"])
            .agg(
                same_id_accuracy=("majority_correct_identity", "mean"),
                group_accuracy=("majority_correct_group", "mean"),
                n=("true_sample_id", "count"),
            )
            .reset_index()
        )
        group_acc["dataset_label"] = group_acc["test_dataset"].map(display_names).fillna(group_acc["test_dataset"])
        heat = group_acc.pivot(index="true_group", columns="dataset_label", values="same_id_accuracy")
        heat = heat.reindex(columns=[label for label in dataset_order if label in heat.columns])
        annot = group_acc.pivot(index="true_group", columns="dataset_label", values="n").fillna(0).astype(int)
        annot = annot.reindex(columns=heat.columns)
        labels = heat.copy().astype(object)
        for idx in heat.index:
            for col in heat.columns:
                val = heat.loc[idx, col]
                n = annot.loc[idx, col] if idx in annot.index and col in annot.columns else 0
                labels.loc[idx, col] = "" if pd.isna(val) else f"{val:.2f}\nn={n}"
        fig, ax = plt.subplots(figsize=(9.8, max(7.5, 0.52 * len(heat.index))))
        ax = sns.heatmap(heat, annot=labels, fmt="", vmin=0, vmax=1, cmap="viridis")
        ax.set_xlabel("")
        ax.set_ylabel("True group")
        ax.set_title(f"Same-ID accuracy by group after transfer{title_suffix}")
        fig.subplots_adjust(bottom=0.11, left=0.12, right=1.02, top=0.93)
        plt.savefig(FIG_DIR / "03_group_same_id_heatmap.png", dpi=200)
        plt.close()

        datasets = [label for label in dataset_order if label in set(derivative_sample["dataset_label"])]
        fig, axes = plt.subplots(1, len(datasets), figsize=(6.8 * len(datasets), 6.1), squeeze=False)
        for ax, dataset in zip(axes[0], datasets):
            subset = derivative_sample[derivative_sample["dataset_label"] == dataset].copy()
            subset["pred_group"] = subset["majority_pred_sample_id"].str.split().str[0]
            table = pd.crosstab(subset["true_group"], subset["pred_group"])
            sns.heatmap(table, annot=True, fmt="d", cmap="mako", ax=ax)
            ax.set_title(dataset)
            ax.set_xlabel("Predicted group")
            ax.set_ylabel("True group")
        fig.subplots_adjust(bottom=0.14, left=0.06, right=0.99, top=0.90, wspace=0.35)
        plt.savefig(FIG_DIR / "04_group_confusion_matrices.png", dpi=200)
        plt.close()

    index = """# Figure Index

## 01_accuracy_summary.png

Shows sample-majority accuracy after voting across replicates. `Primary CV` is
the internal positive control: each primary replicate is predicted using the
other primary replicates. Thermo/Handheld/Medical are primary-to-lot-balanced
transfer tests. If same-ID transfer worked, the same-ID bars would stay high
outside `Primary CV`. They do not.

## 02_true_rank_distribution.png

Shows the rank of the correct sample ID among all candidate sample IDs. Rank 1
means the model chose the correct sample. The y-axis is log-scaled because the
lot-balanced ranks are often hundreds of positions away from the correct ID.

## 03_group_same_id_heatmap.png

Shows same-ID accuracy separately by true disease/control group. Each cell also
prints the number of overlapping samples. This identifies whether failure is
global or concentrated in specific groups.

## 04_group_confusion_matrices.png

Shows where each group is predicted after sample-majority voting. This is not
same-ID accuracy; it is a disease/control group-level confusion view, useful for
seeing whether the transfer failure is just identity-level or also group-level.

## 05_before_after_axis_alignment_accuracy.png

Generated only for the PS/Si axis-aligned run. It compares the original
un-aligned result against the reference x-axis aligned result for same-ID and
disease/group accuracy.

## 06_before_after_axis_alignment_rank.png

Generated only for the PS/Si axis-aligned run. It compares the median rank of
the correct sample ID before and after reference x-axis alignment. Lower is
better; rank 1 means the correct sample is the top prediction.
"""
    (FIG_DIR / "FIGURE_INDEX.md").write_text(index, encoding="utf-8")


def archive_invalid_old_output() -> None:
    if not INVALID_OLD_DIR.exists():
        return
    if INVALID_ARCHIVE_DIR.exists():
        shutil.rmtree(INVALID_ARCHIVE_DIR)
    INVALID_OLD_DIR.rename(INVALID_ARCHIVE_DIR)
    (INVALID_ARCHIVE_DIR / "INVALID_ASSUMPTION.txt").write_text(
        "These outputs are invalid because they treated BPRO/BNOR as PRO/NOR. "
        "BPRO and BNOR are distinct groups and must not be normalized to PRO/NOR.\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for tables, report, and figures.",
    )
    parser.add_argument(
        "--axis-alignment",
        choices=["none", "reference_shift"],
        default="none",
        help="Use PS/Si reference peak shifts to correct lot-balanced x-axis positions.",
    )
    parser.add_argument(
        "--compare-baseline",
        type=Path,
        default=None,
        help="Optional baseline summary_metrics.csv for before/after comparison figures.",
    )
    return parser.parse_args()


def main() -> int:
    global AXIS_ALIGNMENT_MODE, FIG_DIR, OUT_DIR
    args = parse_args()
    OUT_DIR = args.out_dir
    FIG_DIR = OUT_DIR / "figures"
    AXIS_ALIGNMENT_MODE = args.axis_alignment
    SKIPPED_SPECTRA.clear()
    REFERENCE_SHIFT_INFO.clear()
    REFERENCE_ANCHOR_ROWS.clear()
    REFERENCE_READ_ERRORS.clear()

    archive_invalid_old_output()
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    primary_records = load_records(PRIMARY_ROOT, "primary_pooled", "primary_thermo")
    primary_ids = {record.sample_id for record in primary_records}

    balanced_by_dataset = {
        name: load_records(path, "date_lot_balanced", name)
        for name, path in BALANCED_ROOTS.items()
        if path.exists()
    }
    overlap_by_dataset = {
        name: primary_ids & {record.sample_id for record in records}
        for name, records in balanced_by_dataset.items()
    }

    overlap_rows = []
    for dataset, overlap in overlap_by_dataset.items():
        records = balanced_by_dataset[dataset]
        for sample_id in sorted(overlap, key=lambda x: (x.split()[0], int(x.split()[1]))):
            overlap_rows.append(
                {
                    "test_dataset": dataset,
                    "sample_id": sample_id,
                    "group": sample_id.split()[0],
                    "primary_replicates": sum(1 for r in primary_records if r.sample_id == sample_id),
                    "balanced_replicates": sum(1 for r in records if r.sample_id == sample_id),
                }
            )
    pd.DataFrame(overlap_rows).to_csv(OUT_DIR / "overlap_samples.csv", index=False)
    pd.DataFrame(SKIPPED_SPECTRA).to_csv(OUT_DIR / "skipped_spectra.csv", index=False)

    predictions: list[dict[str, object]] = []
    feature_names = ["vector_snv", "vector_derivative_snv"]
    for feature_name in feature_names:
        predictions.extend(
            primary_leave_one_replicate_cv(
                primary_records=primary_records,
                feature_name=feature_name,
                candidate_ids=primary_ids,
                candidate_mode="all_primary_ids",
            )
        )
        for dataset, records in balanced_by_dataset.items():
            overlap = overlap_by_dataset[dataset]
            if not overlap:
                continue
            predictions.extend(
                predict_against_centroids(
                    train_records=primary_records,
                    test_records=records,
                    feature_name=feature_name,
                    candidate_ids=overlap,
                    candidate_mode="dataset_overlap_only",
                )
            )

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(OUT_DIR / "per_replicate_predictions.csv", index=False)
    sample_rows = sample_majority_rows(predictions)
    sample_df = pd.DataFrame(sample_rows)
    sample_df.to_csv(OUT_DIR / "sample_majority_predictions.csv", index=False)
    summary_df = pd.DataFrame(summary_rows(predictions, sample_rows))
    summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)
    write_axis_alignment_tables()
    write_figures(pred_df, sample_df, summary_df)
    write_before_after_comparison(args.compare_baseline, summary_df)
    write_report(primary_records, balanced_by_dataset, overlap_by_dataset, summary_df)

    print(f"Wrote corrected outputs: {OUT_DIR}")
    print(f"Axis alignment mode: {AXIS_ALIGNMENT_MODE}")
    if AXIS_ALIGNMENT_MODE == "reference_shift":
        ok_count = sum(1 for info in REFERENCE_SHIFT_INFO.values() if info["status"] == "ok")
        print(f"Reference-shift date roots with valid anchors: {ok_count}/{len(REFERENCE_SHIFT_INFO)}")
    if INVALID_ARCHIVE_DIR.exists():
        print(f"Archived invalid previous outputs: {INVALID_ARCHIVE_DIR}")
    print(summary_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
