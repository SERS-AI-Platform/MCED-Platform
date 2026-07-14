#!/usr/bin/env python3
"""Generate illustrative SEED 1 QC/preprocessing example package.

The outputs are synthetic, de-identified demonstration data for patent review.
They intentionally exclude cross-instrument calibration examples.
"""

from __future__ import annotations

import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import ListedColormap
from scipy.signal import savgol_filter

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "results" / "patent_seed_examples" / "seed1_qc_preprocessing_2026-06-04"
DATA_DIR = OUT_DIR / "data"
FIG_DIR = OUT_DIR / "figures"
CASE_FIG_DIR = FIG_DIR / "cases"
RAW_INDIV_DIR = DATA_DIR / "individual_raw"
MODEL_READY_INDIV_DIR = DATA_DIR / "individual_calibrated_normalized"

FP_REGION = (400.0, 2200.0)
MODEL_INPUT_AXIS = np.linspace(402.0, 2198.0, 935)

INTENSITY_GATE_RATIO = 0.1
PER_SPEC_CORR_THRESHOLD = 0.925
RSD_THRESHOLD = 5.0
MEAN_CORR_THRESHOLD = 0.95
MIN_VALID_REPLICATES = 3

RNG = np.random.default_rng(20260604)


def configure_matplotlib_fonts() -> None:
    font_candidates = [
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["font.family"] = font_name
            break
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class SpectrumRecord:
    case_id: str
    sample_id: str
    replicate_id: str
    x: np.ndarray
    y: np.ndarray
    expected_issue: str
    metadata_note: str = ""


PEAKS = [
    (620.0, 120.0, 13.0, "band_620"),
    (735.0, 95.0, 10.0, "band_735"),
    (1001.4, 165.0, 11.0, "urea_1001"),
    (1215.0, 80.0, 16.0, "band_1215"),
    (1450.0, 130.0, 20.0, "band_1450"),
    (1660.0, 92.0, 24.0, "band_1660"),
    (2080.0, 55.0, 22.0, "band_2080"),
]


RECOMMENDED_ACTIONS = {
    "none": "전처리 및 모델 분석 단계로 진행.",
    "low_intensity": "초점, 칩 spot, laser power, integration time을 확인한 뒤 재측정.",
    "saturation": "laser power를 낮추거나 integration time을 줄여 재측정.",
    "cosmic_ray": "영향 받은 replicate를 제외하거나 동일 spot을 재측정.",
    "high_rsd": "유효 replicate 수를 늘리거나 추가 chip position에서 재측정.",
    "low_correlation": "불일치 replicate를 제외하고, 유효 replicate 수가 부족하면 재측정.",
    "range_mismatch": "필요 Raman-shift 범위(400-2200 cm-1)를 포함하도록 재측정.",
    "min_valid_replicates": "유효 replicate가 3개 미만이므로 해당 측정 세션을 무효 처리하고 재측정.",
}
PASS_AFTER_EXCLUSION_ACTION = (
    "문제 replicate를 제외하고 유효 replicate가 3개 이상이면 전처리 및 모델 분석 단계로 진행."
)


def gaussian(x: np.ndarray, center: float, amplitude: float, sigma: float) -> np.ndarray:
    return amplitude * np.exp(-0.5 * ((x - center) / sigma) ** 2)


def base_spectrum(
    x: np.ndarray,
    *,
    scale: float = 1.0,
    peak_jitter: float = 0.0,
    peak_multipliers: dict[str, float] | None = None,
    noise: float = 4.0,
) -> np.ndarray:
    peak_multipliers = peak_multipliers or {}
    baseline = 430.0 + 0.035 * (x - x.min()) + 28.0 * np.sin(x / 260.0)
    broad = 70.0 * np.exp(-0.5 * ((x - 1270.0) / 750.0) ** 2)
    y = baseline + broad
    for center, amplitude, sigma, name in PEAKS:
        mult = peak_multipliers.get(name, 1.0)
        y += gaussian(x, center + peak_jitter, amplitude * mult, sigma)
    y = y * scale
    y += RNG.normal(0.0, noise, size=len(x))
    return y


def good_replicate(x: np.ndarray, rep_index: int) -> np.ndarray:
    scale = 1.0 + RNG.normal(0.0, 0.025)
    jitter = RNG.normal(0.0, 0.15)
    multipliers = {name: 1.0 + RNG.normal(0.0, 0.025) for _, _, _, name in PEAKS}
    return base_spectrum(
        x, scale=scale, peak_jitter=jitter, peak_multipliers=multipliers, noise=3.0
    )


def make_records() -> list[SpectrumRecord]:
    records: list[SpectrumRecord] = []
    x_full = np.linspace(50.9232, 3300.39, 1686)

    def add_case(
        case_id: str, sample_id: str, ys: list[tuple[np.ndarray, np.ndarray, str, str]]
    ) -> None:
        for i, (x, y, issue, note) in enumerate(ys, start=1):
            records.append(
                SpectrumRecord(
                    case_id=case_id,
                    sample_id=sample_id,
                    replicate_id=f"R{i}",
                    x=x,
                    y=y,
                    expected_issue=issue,
                    metadata_note=note,
                )
            )

    add_case(
        "PASS",
        "QC-PASS-001",
        [(x_full, good_replicate(x_full, i), "none", "clean replicate") for i in range(5)],
    )

    ys = [(x_full, good_replicate(x_full, i), "none", "clean replicate") for i in range(5)]
    ys[2] = (
        x_full,
        good_replicate(x_full, 3) * 0.025,
        "low_intensity",
        "illustrative focus or laser-power failure",
    )
    add_case("LOW_INTENSITY", "QC-LOWINT-001", ys)

    ys = [(x_full, good_replicate(x_full, i), "none", "clean replicate") for i in range(5)]
    y_sat = good_replicate(x_full, 2) * 1.35
    clip_value = float(np.percentile(y_sat, 91.0))
    y_sat = np.minimum(y_sat, clip_value)
    ys[1] = (x_full, y_sat, "saturation", "illustrative detector clipping")
    add_case("SATURATION", "QC-SAT-001", ys)

    ys = [(x_full, good_replicate(x_full, i), "none", "clean replicate") for i in range(5)]
    y_spike = good_replicate(x_full, 4)
    spike_idx = int(np.argmin(np.abs(x_full - 1328.0)))
    y_spike[spike_idx] = y_spike.max() + 8.0 * (y_spike.max() - y_spike.min())
    ys[3] = (x_full, y_spike, "cosmic_ray", "single-pixel spike inserted")
    add_case("COSMIC_SPIKE", "QC-SPIKE-001", ys)

    high_rsd_scales = [0.35, 0.65, 1.00, 1.45, 1.90]
    base_for_rsd = base_spectrum(x_full, noise=2.5)
    ys = []
    for scale in high_rsd_scales:
        ys.append(
            (
                x_full,
                base_for_rsd * scale + RNG.normal(0.0, 2.5, size=len(x_full)),
                "high_rsd",
                "large replicate-to-replicate intensity variation",
            )
        )
    add_case("HIGH_RSD", "QC-HIGHRSD-001", ys)

    ys = [(x_full, good_replicate(x_full, i), "none", "clean replicate") for i in range(5)]
    discordant = base_spectrum(
        x_full,
        peak_multipliers={
            "band_620": 0.15,
            "band_735": 2.4,
            "urea_1001": 0.20,
            "band_1215": 2.0,
            "band_1450": 0.10,
            "band_1660": 2.2,
            "band_2080": 0.25,
        },
        noise=4.5,
    )
    discordant += gaussian(x_full, 890.0, 170.0, 18.0)
    ys[4] = (x_full, discordant, "low_correlation", "discordant shape / contamination-like profile")
    add_case("LOW_CORRELATION", "QC-LOWCORR-001", ys)

    ys = [(x_full, good_replicate(x_full, i), "none", "clean replicate") for i in range(5)]
    x_short = np.linspace(800.0, 1600.0, 360)
    y_short = base_spectrum(x_short, scale=1.02, noise=3.0)
    ys[4] = (x_short, y_short, "range_mismatch", "acquisition range does not cover 400-2200 cm-1")
    add_case("RANGE_MISMATCH", "QC-RANGE-001", ys)

    ys = [(x_full, good_replicate(x_full, i), "none", "clean replicate") for i in range(5)]
    low_1 = good_replicate(x_full, 1) * 0.025
    low_2 = good_replicate(x_full, 2) * 0.020
    sat = good_replicate(x_full, 3) * 1.35
    sat = np.minimum(sat, float(np.percentile(sat, 91.0)))
    ys[0] = (x_full, low_1, "low_intensity", "low-intensity replicate")
    ys[1] = (x_full, low_2, "low_intensity", "low-intensity replicate")
    ys[2] = (x_full, sat, "saturation", "saturated replicate")
    add_case("INSUFFICIENT_VALID_REPS", "QC-MINVAL-001", ys)

    return records


def trim_spectrum(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = (x >= FP_REGION[0]) & (x <= FP_REGION[1])
    return x[mask], y[mask]


def baseline_correction(y: np.ndarray, window: int = 101) -> np.ndarray:
    baseline = pd.Series(y).rolling(window, center=True, min_periods=1).min().to_numpy()
    return y - baseline


def normalize_snv(y: np.ndarray) -> np.ndarray:
    std = float(np.std(y))
    if std <= 1e-12:
        return y - float(np.mean(y))
    return (y - float(np.mean(y))) / std


def preprocess_stages(x: np.ndarray, y: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    stages: dict[str, tuple[np.ndarray, np.ndarray]] = {"raw": (x, y)}
    xt, yt = trim_spectrum(x, y)
    stages["trimmed"] = (xt, yt)
    if len(yt) >= 11:
        ys = savgol_filter(yt, window_length=11, polyorder=3, mode="interp")
    else:
        ys = yt.copy()
    yb = baseline_correction(ys, window=101)
    stages["baseline_corrected"] = (xt, yb)
    yn = normalize_snv(yb)
    stages["normalized"] = (xt, yn)
    if len(xt) >= 2:
        yr = np.interp(MODEL_INPUT_AXIS, xt, yn)
    else:
        yr = np.full_like(MODEL_INPUT_AXIS, np.nan)
    stages["calibrated_normalized"] = (MODEL_INPUT_AXIS, yr)
    return stages


def fingerprint_mean(x: np.ndarray, y: np.ndarray) -> float:
    mask = (x >= FP_REGION[0]) & (x <= FP_REGION[1])
    if not np.any(mask):
        return float(np.mean(np.abs(y)))
    return float(np.mean(np.abs(y[mask])))


def detect_cosmic_ray(
    y: np.ndarray, isolation_ratio: float = 2.0, height_ratio: float = 0.3
) -> bool:
    if len(y) < 5:
        return False
    yf = y.astype(float)
    imax = int(np.argmax(yf))
    if imax == 0 or imax == len(yf) - 1:
        return False
    peak = yf[imax]
    neighbor_avg = 0.5 * (yf[imax - 1] + yf[imax + 1])
    spectrum_range = float(np.max(yf) - np.min(yf))
    if spectrum_range <= 0 or neighbor_avg <= 0:
        return False
    isolation = (peak - neighbor_avg) / max(neighbor_avg, 1e-9)
    height_frac = (peak - neighbor_avg) / spectrum_range
    return isolation > isolation_ratio and height_frac > height_ratio


def detect_saturation(y: np.ndarray, plateau_min: int = 5) -> bool:
    if len(y) < plateau_min:
        return False
    ymax = float(np.max(y))
    if ymax <= 0:
        return False
    tol = max(abs(ymax) * 1e-3, 1e-9)
    near_max = np.abs(y - ymax) < tol
    longest = current = 0
    for value in near_max:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest >= plateau_min


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    if np.isnan(a).any() or np.isnan(b).any():
        return float("nan")
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def action_for_reasons(reasons: list[str]) -> str:
    if not reasons:
        return RECOMMENDED_ACTIONS["none"]
    priority = [
        "range_mismatch",
        "saturation",
        "cosmic_ray",
        "low_intensity",
        "high_rsd",
        "low_correlation",
        "min_valid_replicates",
    ]
    for key in priority:
        if key in reasons:
            return RECOMMENDED_ACTIONS[key]
    return "Review QC flags and repeat acquisition if needed."


def build_metadata(records: list[SpectrumRecord]) -> pd.DataFrame:
    rows = []
    base_time = pd.Timestamp("2026-06-04 09:00:00")
    for idx, rec in enumerate(records):
        rep_num = int(rec.replicate_id[1:])
        laser_power = 1.0
        integration_time = 0.20
        scan_count = 100
        chip_lot = "LOT-SERS-A2406"
        spot_position = f"dot{rep_num:02d}_center"
        operator = "OP-A"
        site = "SOLUM-LAB"

        if rec.expected_issue == "low_intensity":
            laser_power = 0.10
            integration_time = 0.05
            spot_position = f"dot{rep_num:02d}_edge"
        elif rec.expected_issue == "saturation":
            laser_power = 8.0
            integration_time = 1.00
        elif rec.expected_issue == "high_rsd":
            spot_position = f"dot{rep_num:02d}_variable_hotspot"
        elif rec.expected_issue == "low_correlation":
            spot_position = f"dot{rep_num:02d}_suspected_contamination"
        elif rec.expected_issue == "range_mismatch":
            scan_count = 20

        rows.append(
            {
                "case_id": rec.case_id,
                "sample_id": rec.sample_id,
                "replicate_id": rec.replicate_id,
                "equipment_id": "Thermo_DXR_01",
                "chip_lot": chip_lot,
                "spot_position": spot_position,
                "laser_power_mw": laser_power,
                "integration_time_s": integration_time,
                "scan_count": scan_count,
                "measured_at": (base_time + pd.Timedelta(minutes=idx)).isoformat(),
                "operator": operator,
                "site": site,
                "x_min": float(np.min(rec.x)),
                "x_max": float(np.max(rec.x)),
                "n_points": int(len(rec.x)),
                "expected_issue": rec.expected_issue,
                "metadata_note": rec.metadata_note,
                "raw_file": f"{rec.case_id}_{rec.sample_id}_{rec.replicate_id}_raw.csv",
                "calibrated_normalized_file": f"{rec.case_id}_{rec.sample_id}_{rec.replicate_id}_calibrated_normalized.csv",
            }
        )
    return pd.DataFrame(rows)


def build_data_tables(
    records: list[SpectrumRecord],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str, str], np.ndarray],
    dict[tuple[str, str, str], np.ndarray],
]:
    raw_rows = []
    stage_rows = []
    corr_features: dict[tuple[str, str, str], np.ndarray] = {}
    rsd_features: dict[tuple[str, str, str], np.ndarray] = {}

    for rec in records:
        raw_file = RAW_INDIV_DIR / f"{rec.case_id}_{rec.sample_id}_{rec.replicate_id}_raw.csv"
        pd.DataFrame({"raman_shift": rec.x, "intensity": rec.y}).to_csv(
            raw_file, index=False, encoding="utf-8-sig"
        )

        stages = preprocess_stages(rec.x, rec.y)
        for stage, (x_stage, y_stage) in stages.items():
            if stage == "calibrated_normalized":
                output_file = (
                    MODEL_READY_INDIV_DIR
                    / f"{rec.case_id}_{rec.sample_id}_{rec.replicate_id}_calibrated_normalized.csv"
                )
                pd.DataFrame({"raman_shift": x_stage, "intensity": y_stage}).to_csv(
                    output_file, index=False, encoding="utf-8-sig"
                )
                corr_features[(rec.case_id, rec.sample_id, rec.replicate_id)] = y_stage
                xb, yb = stages["baseline_corrected"]
                if len(xb) >= 2:
                    rsd_features[(rec.case_id, rec.sample_id, rec.replicate_id)] = np.interp(
                        MODEL_INPUT_AXIS, xb, yb
                    )
                else:
                    rsd_features[(rec.case_id, rec.sample_id, rec.replicate_id)] = np.full_like(
                        MODEL_INPUT_AXIS, np.nan
                    )
            for x_val, y_val in zip(x_stage, y_stage):
                stage_rows.append(
                    {
                        "case_id": rec.case_id,
                        "sample_id": rec.sample_id,
                        "replicate_id": rec.replicate_id,
                        "stage": stage,
                        "raman_shift": float(x_val),
                        "intensity": float(y_val) if np.isfinite(y_val) else np.nan,
                    }
                )

        for x_val, y_val in zip(rec.x, rec.y):
            raw_rows.append(
                {
                    "case_id": rec.case_id,
                    "sample_id": rec.sample_id,
                    "replicate_id": rec.replicate_id,
                    "raman_shift": float(x_val),
                    "intensity": float(y_val),
                }
            )

    return pd.DataFrame(raw_rows), pd.DataFrame(stage_rows), corr_features, rsd_features


def build_qc_tables(
    records: list[SpectrumRecord],
    corr_features: dict[tuple[str, str, str], np.ndarray],
    rsd_features: dict[tuple[str, str, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fp_means = [fingerprint_mean(rec.x, rec.y) for rec in records]
    gate_threshold = float(np.median(fp_means) * INTENSITY_GATE_RATIO)

    per_spec_base = {}
    for rec, fp_mean in zip(records, fp_means):
        range_pass = bool(np.min(rec.x) <= FP_REGION[0] and np.max(rec.x) >= FP_REGION[1])
        reasons = []
        if fp_mean < gate_threshold:
            reasons.append("low_intensity")
        if detect_saturation(rec.y):
            reasons.append("saturation")
        if detect_cosmic_ray(rec.y):
            reasons.append("cosmic_ray")
        if not range_pass:
            reasons.append("range_mismatch")
        per_spec_base[(rec.case_id, rec.sample_id, rec.replicate_id)] = {
            "case_id": rec.case_id,
            "sample_id": rec.sample_id,
            "replicate_id": rec.replicate_id,
            "fp_mean": fp_mean,
            "gate_threshold": gate_threshold,
            "intensity_gate_pass": fp_mean >= gate_threshold,
            "cosmic_ray_flag": "cosmic_ray" in reasons,
            "saturation_flag": "saturation" in reasons,
            "range_pass": range_pass,
            "base_reasons": reasons,
        }

    sample_rows = []
    per_spec_corr = {}
    for (case_id, sample_id), group_records in pd.DataFrame(
        [
            {"case_id": r.case_id, "sample_id": r.sample_id, "replicate_id": r.replicate_id}
            for r in records
        ]
    ).groupby(["case_id", "sample_id"]):
        keys = [(case_id, sample_id, rid) for rid in group_records["replicate_id"]]
        corr_mat = np.vstack([corr_features[key] for key in keys])
        rsd_mat = np.vstack([rsd_features[key] for key in keys])

        corr_values = []
        for i, key in enumerate(keys):
            other = np.delete(corr_mat, i, axis=0)
            mean_other = np.nanmean(other, axis=0)
            corr = corrcoef_safe(corr_mat[i], mean_other)
            per_spec_corr[key] = corr
            if np.isfinite(corr):
                corr_values.append(corr)

        pair_corrs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                corr = corrcoef_safe(corr_mat[i], corr_mat[j])
                if np.isfinite(corr):
                    pair_corrs.append(corr)

        mean_spec = np.nanmean(rsd_mat, axis=0)
        std_spec = np.nanstd(rsd_mat, axis=0, ddof=1)
        max_intensity = float(np.nanmax(mean_spec))
        rsd = (std_spec / max(max_intensity, 1e-9)) * 100.0
        mean_rsd = float(np.nanmean(rsd))
        median_rsd = float(np.nanmedian(rsd))
        max_rsd = float(np.nanmax(rsd))
        mean_corr = float(np.mean(pair_corrs)) if pair_corrs else np.nan
        min_corr = float(np.min(pair_corrs)) if pair_corrs else np.nan

        sample_reasons = []
        if mean_rsd > RSD_THRESHOLD:
            sample_reasons.append("high_rsd")
        if np.isfinite(mean_corr) and mean_corr < MEAN_CORR_THRESHOLD:
            sample_reasons.append("low_correlation")

        sample_rows.append(
            {
                "case_id": case_id,
                "sample_id": sample_id,
                "n_replicates": len(keys),
                "mean_rsd": mean_rsd,
                "median_rsd": median_rsd,
                "max_rsd": max_rsd,
                "mean_corr": mean_corr,
                "min_corr": min_corr,
                "sample_level_failures": ";".join(sample_reasons) if sample_reasons else "none",
            }
        )

    qc_rows = []
    pass_counts: Counter[tuple[str, str]] = Counter()
    for key, base in per_spec_base.items():
        reasons = list(base["base_reasons"])
        corr = per_spec_corr.get(key, np.nan)
        has_primary_failure = any(
            reason in {"low_intensity", "saturation", "cosmic_ray", "range_mismatch"}
            for reason in reasons
        )
        if not has_primary_failure and np.isfinite(corr) and corr < PER_SPEC_CORR_THRESHOLD:
            reasons.append("low_correlation")
        prelim = len(reasons) == 0
        if prelim:
            pass_counts[(key[0], key[1])] += 1

        qc_rows.append(
            {
                **{k: v for k, v in base.items() if k != "base_reasons"},
                "corr_to_mean_of_others": corr,
                "corr_threshold": PER_SPEC_CORR_THRESHOLD,
                "prelim_qc_pass": prelim,
                "failure_reason": ";".join(dict.fromkeys(reasons)) if reasons else "none",
                "recommended_action": action_for_reasons(reasons),
            }
        )

    qc_df = pd.DataFrame(qc_rows)
    sample_df = pd.DataFrame(sample_rows)

    qc_df["final_qc_pass"] = qc_df["prelim_qc_pass"]
    qc_df["final_failure_reason"] = qc_df["failure_reason"]
    qc_df["final_recommended_action"] = qc_df["recommended_action"]

    summary_rows = []
    for (case_id, sample_id), sub in qc_df.groupby(["case_id", "sample_id"]):
        stats = sample_df[
            (sample_df["case_id"] == case_id) & (sample_df["sample_id"] == sample_id)
        ].iloc[0]
        replicate_reasons = sorted(
            {
                reason
                for text in sub["final_failure_reason"]
                for reason in str(text).split(";")
                if reason and reason != "none"
            }
        )
        sample_level = [
            reason
            for reason in str(stats["sample_level_failures"]).split(";")
            if reason and reason != "none"
        ]
        n_valid = int(sub["final_qc_pass"].sum())
        n_excluded = int((~sub["final_qc_pass"]).sum())
        sample_failures: list[str] = []
        if n_valid < MIN_VALID_REPLICATES:
            sample_failures.append("min_valid_replicates")
        elif not replicate_reasons:
            sample_failures.extend(sample_level)
        qc_flags = sorted(set(replicate_reasons + sample_failures))
        if sample_failures:
            sample_status = "FAIL"
            recommended_action = action_for_reasons(sample_failures)
        elif replicate_reasons:
            sample_status = "PASS_AFTER_EXCLUSION"
            recommended_action = PASS_AFTER_EXCLUSION_ACTION
        else:
            sample_status = "PASS"
            recommended_action = RECOMMENDED_ACTIONS["none"]
        summary_rows.append(
            {
                "case_id": case_id,
                "sample_id": sample_id,
                "n_replicates": int(len(sub)),
                "n_final_pass": n_valid,
                "n_final_fail": n_excluded,
                "min_valid_replicates": MIN_VALID_REPLICATES,
                "replicate_count_pass": n_valid >= MIN_VALID_REPLICATES,
                "mean_rsd": stats["mean_rsd"],
                "mean_corr": stats["mean_corr"],
                "sample_qc_status": sample_status,
                "qc_flags": ";".join(qc_flags) if qc_flags else "none",
                "sample_failure_reason": ";".join(sample_failures) if sample_failures else "none",
                "failure_reason": ";".join(qc_flags) if qc_flags else "none",
                "recommended_action": recommended_action,
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    return qc_df, summary_df


def save_action_table() -> pd.DataFrame:
    rows = [
        {
            "failure_reason": key,
            "recommended_action": value,
        }
        for key, value in RECOMMENDED_ACTIONS.items()
    ]
    df = pd.DataFrame(rows)
    df.to_csv(DATA_DIR / "recommended_actions.csv", index=False, encoding="utf-8-sig")
    return df


def setup_output_dirs() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    for path in [DATA_DIR, FIG_DIR, CASE_FIG_DIR, RAW_INDIV_DIR, MODEL_READY_INDIV_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def plot_case(records: list[SpectrumRecord], qc_df: pd.DataFrame, case_id: str) -> None:
    case_records = [r for r in records if r.case_id == case_id]
    if not case_records:
        return
    sub_qc = qc_df[qc_df["case_id"] == case_id].set_index("replicate_id")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))

    for color, rec in zip(colors, case_records):
        fail = not bool(sub_qc.loc[rec.replicate_id, "final_qc_pass"])
        label = f"{rec.replicate_id}" + (" fail" if fail else "")
        axes[0].plot(
            rec.x,
            rec.y,
            lw=1.0 if not fail else 1.8,
            color="crimson" if fail else color,
            alpha=0.9,
            label=label,
        )
        stages = preprocess_stages(rec.x, rec.y)
        xg, yg = stages["calibrated_normalized"]
        axes[1].plot(
            xg,
            yg,
            lw=1.0 if not fail else 1.8,
            color="crimson" if fail else color,
            alpha=0.9,
            label=label,
        )

    axes[0].axvspan(
        FP_REGION[0], FP_REGION[1], color="#E5F3FF", alpha=0.35, label="fingerprint range"
    )
    axes[0].set_title(f"{case_id}: raw replicate spectra")
    axes[0].set_xlabel("Raman shift (cm-1)")
    axes[0].set_ylabel("Raw intensity")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].set_title("Spectra after calibration/normalization")
    axes[1].set_xlabel("Raman shift (cm-1)")
    axes[1].set_ylabel("SNV intensity")
    axes[1].legend(ncol=3, fontsize=8)

    reasons = sorted(
        {
            reason
            for text in sub_qc["final_failure_reason"]
            for reason in str(text).split(";")
            if reason and reason != "none"
        }
    )
    reason_text = "PASS" if not reasons else "FAIL: " + ", ".join(reasons)
    fig.text(0.01, 0.01, reason_text, fontsize=10, color="crimson" if reasons else "darkgreen")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(CASE_FIG_DIR / f"{case_id.lower()}_raw_and_preprocessed.png", dpi=220)
    plt.close(fig)


def plot_pass_replicates(records: list[SpectrumRecord]) -> None:
    case_records = [r for r in records if r.case_id == "PASS"]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for rec in case_records:
        ax.plot(rec.x, rec.y, lw=1.1, alpha=0.9, label=rec.replicate_id)
    ax.axvspan(FP_REGION[0], FP_REGION[1], color="#E5F3FF", alpha=0.35)
    ax.set_title("QC PASS example: five raw SERS replicates from one specimen")
    ax.set_xlabel("Raman shift (cm-1)")
    ax.set_ylabel("Raw intensity")
    ax.legend(ncol=5)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_01_qc_pass_raw_replicates.png", dpi=240)
    plt.close(fig)


def plot_failure_gallery(records: list[SpectrumRecord], qc_df: pd.DataFrame) -> None:
    gallery_cases = [
        ("LOW_INTENSITY", "low intensity"),
        ("SATURATION", "saturation"),
        ("COSMIC_SPIKE", "cosmic spike"),
        ("HIGH_RSD", "high RSD"),
        ("LOW_CORRELATION", "low correlation"),
        ("RANGE_MISMATCH", "range mismatch"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.8))
    axes = axes.ravel()
    for ax, (case_id, title) in zip(axes, gallery_cases):
        sub = [r for r in records if r.case_id == case_id]
        qsub = qc_df[qc_df["case_id"] == case_id].set_index("replicate_id")
        for rec in sub:
            fail = not bool(qsub.loc[rec.replicate_id, "final_qc_pass"])
            ax.plot(
                rec.x,
                rec.y,
                lw=1.7 if fail else 0.8,
                color="crimson" if fail else "#6B7280",
                alpha=0.95 if fail else 0.45,
            )
        ax.axvspan(FP_REGION[0], FP_REGION[1], color="#E5F3FF", alpha=0.25)
        ax.set_title(title)
        ax.set_xlabel("cm-1")
        ax.set_ylabel("intensity")
    fig.suptitle("Illustrative QC failure modes for SEED 1", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_02_qc_failure_gallery.png", dpi=240)
    plt.close(fig)


def plot_preprocessing_stages(records: list[SpectrumRecord]) -> None:
    rec = next(r for r in records if r.case_id == "PASS" and r.replicate_id == "R1")
    stages = preprocess_stages(rec.x, rec.y)
    stage_order = ["raw", "trimmed", "baseline_corrected", "normalized", "calibrated_normalized"]
    titles = {
        "raw": "Raw spectrum",
        "trimmed": "Trimmed to fingerprint range",
        "baseline_corrected": "Baseline-corrected",
        "normalized": "SNV normalized",
        "calibrated_normalized": "After calibration/normalization",
    }
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    axes = axes.ravel()
    for ax, stage in zip(axes, stage_order):
        x, y = stages[stage]
        ax.plot(x, y, color="#2563EB", lw=1.1)
        if stage == "raw":
            ax.axvspan(FP_REGION[0], FP_REGION[1], color="#E5F3FF", alpha=0.35)
        ax.set_title(titles[stage])
        ax.set_xlabel("Raman shift (cm-1)")
        ax.set_ylabel("intensity")
    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.70,
        "Pipeline:\nraw -> trim -> smooth -> baseline correction\n-> calibration/normalization output",
        fontsize=11,
        va="top",
    )
    fig.suptitle("Preprocessing before/after example", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / "fig_03_preprocessing_stages.png", dpi=240)
    plt.close(fig)


def plot_qc_metrics(qc_summary: pd.DataFrame, qc_df: pd.DataFrame) -> None:
    order = [
        "PASS",
        "LOW_INTENSITY",
        "SATURATION",
        "COSMIC_SPIKE",
        "HIGH_RSD",
        "LOW_CORRELATION",
        "RANGE_MISMATCH",
        "INSUFFICIENT_VALID_REPS",
    ]
    summary = qc_summary.copy()
    summary["_order"] = summary["case_id"].map({case: idx for idx, case in enumerate(order)})
    summary = summary.sort_values("_order").reset_index(drop=True)

    criteria = [
        ("low_intensity", "Intensity\ngate"),
        ("saturation", "Saturation"),
        ("cosmic_ray", "Cosmic\nspike"),
        ("high_rsd", "RSD"),
        ("low_correlation", "Correlation"),
        ("range_mismatch", "Raman-shift\nrange"),
        ("min_valid_replicates", "Valid reps\n>= 3"),
    ]
    matrix = []
    for _, row in summary.iterrows():
        reasons = {r for r in str(row["qc_flags"]).split(";") if r and r != "none"}
        matrix.append([1 if key in reasons else 0 for key, _ in criteria])
    matrix_arr = np.asarray(matrix)

    fig = plt.figure(figsize=(15.5, 7.2))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.75, 1.0],
        height_ratios=[1.0, 0.85],
        wspace=0.35,
        hspace=0.45,
    )
    matrix_ax = fig.add_subplot(gs[:, 0])
    count_ax = fig.add_subplot(gs[0, 1])
    rsd_ax = fig.add_subplot(gs[1, 1])

    matrix_ax.imshow(
        matrix_arr,
        cmap=ListedColormap(["#E5E7EB", "#DC2626"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    matrix_ax.set_xticks(np.arange(len(criteria)))
    matrix_ax.set_xticklabels([label for _, label in criteria], fontsize=9)
    matrix_ax.set_yticks(np.arange(len(summary)))
    row_labels = [row.case_id.replace("_", " ") for row in summary.itertuples(index=False)]
    matrix_ax.set_yticklabels(row_labels, fontsize=9)
    matrix_ax.set_title("QC flag matrix by example case", fontweight="bold")
    matrix_ax.set_xticks(np.arange(-0.5, len(criteria), 1), minor=True)
    matrix_ax.set_yticks(np.arange(-0.5, len(summary), 1), minor=True)
    matrix_ax.grid(which="minor", color="white", linewidth=2)
    matrix_ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(matrix_arr.shape[0]):
        for j in range(matrix_arr.shape[1]):
            is_fail = bool(matrix_arr[i, j])
            matrix_ax.text(
                j,
                i,
                "FLAG" if is_fail else "OK",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if is_fail else "#374151",
                fontweight="bold" if is_fail else "normal",
            )

    y = np.arange(len(summary))
    pass_counts = summary["n_final_pass"].to_numpy()
    fail_counts = summary["n_final_fail"].to_numpy()
    count_ax.barh(y, pass_counts, color="#059669", label="valid")
    count_ax.barh(y, fail_counts, left=pass_counts, color="#DC2626", label="excluded")
    count_ax.axvline(
        MIN_VALID_REPLICATES,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"min {MIN_VALID_REPLICATES}",
    )
    count_ax.set_yticks(y)
    count_ax.set_yticklabels(summary["case_id"].str.replace("_", " "), fontsize=8)
    count_ax.invert_yaxis()
    count_ax.set_xlim(0, max(summary["n_replicates"]) + 0.75)
    count_ax.set_xlabel("replicate count")
    count_ax.set_title("Valid replicates after per-spectrum QC", fontweight="bold")
    count_ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, fontsize=8, frameon=True
    )
    for idx, row in summary.iterrows():
        count_ax.text(
            row["n_replicates"] + 0.05,
            idx,
            f"{int(row['n_final_pass'])}/{int(row['n_replicates'])}",
            va="center",
            fontsize=8,
        )

    rsd_subset = summary[summary["case_id"].isin(["PASS", "HIGH_RSD"])].copy()
    rsd_subset["_rsd_order"] = rsd_subset["case_id"].map({"PASS": 0, "HIGH_RSD": 1})
    rsd_subset = rsd_subset.sort_values("_rsd_order")
    rsd_colors = [
        "#059669" if status == "PASS" else "#DC2626" for status in rsd_subset["sample_qc_status"]
    ]
    rsd_x = np.arange(len(rsd_subset))
    rsd_ax.bar(rsd_x, rsd_subset["mean_rsd"], color=rsd_colors, width=0.55)
    rsd_ax.axhline(
        RSD_THRESHOLD, color="#111827", linestyle="--", linewidth=1.2, label=f"RSD {RSD_THRESHOLD}%"
    )
    rsd_ax.set_xticks(rsd_x)
    rsd_ax.set_xticklabels(rsd_subset["case_id"].str.replace("_", "\n"), fontsize=9)
    rsd_ax.set_ylabel("Mean RSD (%)")
    rsd_ax.set_ylim(0, max(rsd_subset["mean_rsd"].max() + 1.0, RSD_THRESHOLD + 1.0))
    rsd_ax.set_title("Repeatability criterion example", fontweight="bold")
    rsd_ax.legend(fontsize=8)
    for x_pos, value in zip(rsd_x, rsd_subset["mean_rsd"]):
        rsd_ax.text(x_pos, value + 0.15, f"{value:.1f}%", ha="center", fontsize=8)

    fig.suptitle("SEED 1 QC summary examples", fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.90, bottom=0.12, wspace=0.32, hspace=0.65)
    fig.savefig(FIG_DIR / "fig_04_qc_metrics_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def plot_action_table(action_df: pd.DataFrame) -> None:
    display_df = action_df[action_df["failure_reason"] != "none"].copy()
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.axis("off")
    table = ax.table(
        cellText=display_df.values,
        colLabels=["failure_reason", "recommended_action"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.22, 0.76],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.65)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#E5E7EB")
            cell.set_text_props(weight="bold")
        cell.set_edgecolor("#D1D5DB")
    ax.set_title("QC failure reason to recommended action mapping", fontweight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_05_recommended_actions_table.png", dpi=240)
    plt.close(fig)


def write_readme(qc_summary: pd.DataFrame) -> None:
    case_table = qc_summary[
        [
            "case_id",
            "sample_id",
            "n_final_pass",
            "min_valid_replicates",
            "sample_qc_status",
            "qc_flags",
            "sample_failure_reason",
            "recommended_action",
        ]
    ].to_markdown(index=False)

    readme = f"""# SEED 1 QC/Preprocessing Example Package

작성일: 2026-06-04

이 폴더는 SEED 1 "품질관리 및 장비 독립 정규화 기반 SERS-AI 질환 분석" 설명용 예시 패키지입니다.

중요: 본 패키지는 변리사/내부 설명용 합성 예시 데이터이며 실제 환자 데이터가 아닙니다. 환자 식별자는 포함하지 않았습니다. Cross-instrument calibration 예시는 의도적으로 제외했습니다.

## 포함된 케이스

{case_table}

## QC 기준

| 항목 | 기준 |
|---|---|
| Fingerprint region | 400-2200 cm-1 |
| Intensity gate | fingerprint mean < global median x 0.1 |
| Per-spectrum correlation | corr-to-mean-of-others < {PER_SPEC_CORR_THRESHOLD} |
| Sample-level RSD | mean RSD > {RSD_THRESHOLD}% |
| Sample-level mean correlation | mean pairwise corr < {MEAN_CORR_THRESHOLD} |
| Minimum valid replicates | 5회 반복 측정 중 3개 이상 유효 replicate이면 sample 분석 가능 |

## 주요 파일

| 파일 | 설명 |
|---|---|
| `data/example_metadata.csv` | replicate별 측정 메타데이터 예시 |
| `data/raw_spectra_long.csv` | 모든 raw spectrum long-format |
| `data/preprocessing_stages_long.csv` | raw, trimmed, baseline_corrected, normalized, calibrated_normalized 단계별 데이터 |
| `data/qc_results_per_replicate.csv` | replicate별 QC 산출 결과 |
| `data/qc_summary_by_sample.csv` | sample/case별 QC 요약 |
| `data/recommended_actions.csv` | failure reason별 재측정/처리 권고 |
| `data/individual_raw/` | replicate별 raw CSV |
| `data/individual_calibrated_normalized/` | replicate별 보정·정규화 후 모델 입력 CSV |

## 주요 Figure

| Figure | 설명 |
|---|---|
| `figures/fig_01_qc_pass_raw_replicates.png` | 정상 SERS 5회 반복 측정 raw spectrum |
| `figures/fig_02_qc_failure_gallery.png` | QC 실패 유형 gallery |
| `figures/fig_03_preprocessing_stages.png` | raw -> trimmed -> baseline -> normalized -> 보정·정규화 후 모델 입력 스펙트럼 |
| `figures/fig_04_qc_metrics_summary.png` | QC 실패 기준 matrix, 유효 replicate 수, RSD 기준 예시 |
| `figures/fig_05_recommended_actions_table.png` | failure reason별 권고 조치 |
| `figures/cases/` | case별 raw 및 preprocessed overlay |

## 전달 시 설명 문장

본 예시는 동일 검체의 복수 반복 SERS 신호를 AI 모델에 바로 입력하지 않고, 먼저 품질 지수를 산출하여 정상/불량 신호를 구분하고, 품질 미달 유형에 따라 재측정 또는 제외/보정 조치를 결정한 뒤, 보정·정규화 후 모델 입력 가능한 SERS 신호를 생성하는 과정을 보여준다.

QC 실패 유형은 low intensity, detector saturation, cosmic spike, high RSD, low correlation, range mismatch로 구성하였고, 각 실패 유형에 대해 측정 메타데이터, QC 산출 결과, 전처리 전후 데이터, 권고 조치가 함께 제공된다.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    configure_matplotlib_fonts()
    setup_output_dirs()
    records = make_records()

    metadata = build_metadata(records)
    raw_df, stages_df, corr_features, rsd_features = build_data_tables(records)
    qc_df, qc_summary = build_qc_tables(records, corr_features, rsd_features)
    action_df = save_action_table()

    metadata.to_csv(DATA_DIR / "example_metadata.csv", index=False, encoding="utf-8-sig")
    raw_df.to_csv(DATA_DIR / "raw_spectra_long.csv", index=False, encoding="utf-8-sig")
    stages_df.to_csv(DATA_DIR / "preprocessing_stages_long.csv", index=False, encoding="utf-8-sig")
    qc_df.to_csv(DATA_DIR / "qc_results_per_replicate.csv", index=False, encoding="utf-8-sig")
    qc_summary.to_csv(DATA_DIR / "qc_summary_by_sample.csv", index=False, encoding="utf-8-sig")

    plot_pass_replicates(records)
    plot_failure_gallery(records, qc_df)
    plot_preprocessing_stages(records)
    plot_qc_metrics(qc_summary, qc_df)
    plot_action_table(action_df)
    for case_id in sorted({record.case_id for record in records}):
        plot_case(records, qc_df, case_id)

    write_readme(qc_summary)

    print(f"Wrote SEED 1 QC example package to: {OUT_DIR}")
    print(f"Cases: {len(qc_summary)}")
    print(f"Replicates: {len(records)}")
    print(f"Figures: {len(list(FIG_DIR.rglob('*.png')))}")


if __name__ == "__main__":
    main()
