from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Literal, TypeAlias, assert_never

import numpy as np
import pywt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "patent"))
sys.path.insert(0, str(REPO / "scripts" / "analysis"))

import run_mapping_trim_only as runner
from mapping_repeat_average_core import GROUP_ORDER

from sers.signal import baseline_correction

Condition: TypeAlias = Literal[
    "raw_aligned",
    "baseline",
    "baseline_area",
    "baseline_dwt",
    "baseline_area_dwt",
]
CONDITIONS: tuple[Condition, ...] = (
    "raw_aligned",
    "baseline",
    "baseline_area",
    "baseline_dwt",
    "baseline_area_dwt",
)
JsonValue: TypeAlias = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
DWT_WAVELET = "db4"
DWT_MODE = "symmetric"
DWT_THRESHOLD_SCALE = 1.0


class PipelineDataError(ValueError):
    pass


def area_normalize(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(np.abs(values)))
    return values / total if total > 1e-10 else values.copy()


def dwt_soft_denoise(values: np.ndarray) -> np.ndarray:
    wavelet = pywt.Wavelet(DWT_WAVELET)
    level = pywt.dwt_max_level(len(values), wavelet.dec_len)
    coefficients = pywt.wavedec(values, DWT_WAVELET, mode=DWT_MODE, level=level)
    finest = coefficients[-1]
    sigma = float(np.median(np.abs(finest - np.median(finest))) / 0.6745)
    threshold = DWT_THRESHOLD_SCALE * sigma * float(np.sqrt(2.0 * np.log(len(values))))
    denoised = [coefficients[0]] + [
        pywt.threshold(detail, threshold, mode="soft") for detail in coefficients[1:]
    ]
    return np.asarray(pywt.waverec(denoised, DWT_WAVELET, mode=DWT_MODE)[: len(values)])


def transform(values: np.ndarray, condition: Condition) -> np.ndarray:
    match condition:
        case "raw_aligned":
            return values.copy()
        case "baseline":
            return baseline_correction(values, window=101)
        case "baseline_area":
            return area_normalize(baseline_correction(values, window=101))
        case "baseline_dwt":
            return dwt_soft_denoise(baseline_correction(values, window=101))
        case "baseline_area_dwt":
            return dwt_soft_denoise(area_normalize(baseline_correction(values, window=101)))
        case unreachable:
            assert_never(unreachable)


def prepare_pipeline(
    grid: np.ndarray,
    condition: Condition,
) -> tuple[list[runner.PreparedSubject], dict[str, JsonValue]]:
    subjects, raw_axis = runner.load_subjects(grid)
    prepared: list[runner.PreparedSubject] = []
    native_counts: list[int] = []
    finite_counts: list[int] = []
    for subject in subjects:
        native_axis = np.asarray(subject.native_axis, dtype=float)
        mask = (native_axis >= 400.0) & (native_axis <= 2200.0)
        x_trim = native_axis[mask]
        rows = np.vstack(
            [
                transform(
                    np.interp(grid, x_trim, np.asarray(row, dtype=float)[mask]),
                    condition,
                )
                for row in subject.native_replicates
            ]
        ).astype(np.float32)
        finite = np.isfinite(rows).all(axis=1)
        rows = rows[finite]
        if len(rows) == 0:
            raise PipelineDataError(f"subject {subject.ordinal} has no finite transformed spectra")
        native_counts.append(len(subject.native_replicates))
        finite_counts.append(len(rows))
        prepared.append(
            runner.PreparedSubject(
                ordinal=subject.ordinal,
                group=subject.group,
                x=grid.copy(),
                all_spectra=rows,
                qc_keep=np.ones(len(rows), dtype=bool),
                qc_corr=np.full(len(rows), np.nan, dtype=float),
                qc_rsd_pct=float("nan"),
            )
        )
    wavelet = pywt.Wavelet(DWT_WAVELET)
    metadata: dict[str, JsonValue] = {
        "raw_axis": raw_axis,
        "subject_count": len(prepared),
        "group_counts": {group: sum(item.group == group for item in prepared) for group in GROUP_ORDER},
        "input_repeats": int(sum(native_counts)),
        "model_repeats": int(sum(finite_counts)),
        "finite_repeats": int(sum(finite_counts)),
        "nonfinite_repeats_removed": int(sum(native_counts) - sum(finite_counts)),
        "qc_rule": "No spectral QC filtering; finite-value gate only. All finite transformed repeats remain in the locked evaluation.",
        "grid": {
            "min_cm-1": float(grid[0]),
            "max_cm-1": float(grid[-1]),
            "points": int(len(grid)),
            "step_cm-1": float(np.median(np.diff(grid))),
        },
        "condition": condition,
        "preprocessing": {
            "trim": [400.0, 2200.0],
            "alignment": "raw intensity linearly interpolated onto fixed 402-2198 cm-1 grid before signal transforms",
            "baseline": None if condition == "raw_aligned" else {"method": "centered rolling minimum", "window": 101},
            "normalization": "area(sum(abs(y)))" if "area" in condition else None,
            "dwt": None
            if "dwt" not in condition
            else {
                "wavelet": DWT_WAVELET,
                "mode": DWT_MODE,
                "level": int(pywt.dwt_max_level(len(grid), wavelet.dec_len)),
                "threshold": "per-spectrum finest-detail MAD universal threshold",
                "threshold_scale": DWT_THRESHOLD_SCALE,
                "detail_thresholding": "soft",
                "approximation": "preserved",
            },
        },
    }
    return prepared, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--mc-iterations", type=int, default=100)
    parser.add_argument("--n-blocks", type=int, choices=runner.BLOCK_CHOICES, default=3)
    parser.add_argument("--position-aware", action="store_true")
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def write_report(out: Path) -> None:
    metadata = json.loads((out / "run_metadata.json").read_text(encoding="utf-8"))
    with (out / "oof_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("aggregation") == "mean"]

    def value(row: dict[str, str], key: str) -> str:
        number = float(row.get(key, "nan"))
        return "-" if not np.isfinite(number) else f"{number:.4f}"

    lines = [
        "# Mapping aligned → baseline → area → DWT comparison run",
        "",
        f"- condition: `{metadata['condition']}`",
        f"- cohort: {metadata['subject_count']} patients, {metadata['model_repeats']} finite repeats",
        "- raw intensity was aligned to 402–2198 cm⁻¹ on a 933-point grid before optional transforms.",
        "- patient-level StratifiedGroupKFold 5-fold OOF; ResNet used all finite repeat spectra and patient mean was used for OOF aggregation.",
        "- no spectral QC filtering was applied; all finite transformed repeats were retained.",
        "",
        "| Task | Model | ROC-AUC | Macro ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task']} | {row['model']} | {value(row, 'roc_auc')} | {value(row, 'macro_roc_auc')} | "
            f"{value(row, 'balanced_accuracy')} | {value(row, 'sensitivity')} | {value(row, 'specificity')} | {value(row, 'macro_f1')} |"
        )
    lines.extend(
        [
            "",
            "Cancer Screening AUC는 병원/측정 조건 confounding 가능성이 있어 외부 일반화 성능으로 해석하지 않는다.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    condition = args.condition
    output = args.output.resolve()

    def prepare(grid: np.ndarray) -> tuple[list[runner.PreparedSubject], dict[str, JsonValue]]:
        return prepare_pipeline(grid, condition)

    runner.prepare_trim_only = prepare
    runner_args = [
        sys.argv[0],
        "--output",
        str(output),
        "--epochs",
        str(args.epochs),
        "--mc-iterations",
        str(args.mc_iterations),
        "--n-blocks",
        str(args.n_blocks),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
    ]
    if args.position_aware:
        runner_args.append("--position-aware")
    sys.argv = runner_args
    runner.main()
    source = output / "preprocessed_trim_only_arrays.npz"
    target = output / f"preprocessed_{condition}_arrays.npz"
    if source.exists():
        source.replace(target)
    write_report(output)


if __name__ == "__main__":
    main()
