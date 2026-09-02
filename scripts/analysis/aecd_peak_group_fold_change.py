#!/usr/bin/env python3
"""Per-peak fold change between prostate / prostate-disease-control / control.

이 스크립트는 aecd_api_model_mean_spectrum_clinical_performance.py 가 이미 만들어 둔
두 산출물만 읽는다 (API/DB 재조회 없음):

- mean_representative_spectra.csv     subject별 PS 축보정 + repeat-QC 평균 스펙트럼
- resolution_aware_peak_registry.csv  그룹 median 스펙트럼에서 검출된 공통 피크

각 subject에 대해 파이프라인과 동일한 정의(61-point rolling median)로 local baseline을
빼서 peak contrast를 구하고, subject별 총 면적으로 정규화한 뒤 그룹 간 median ratio를
fold change로 보고한다.

설계 판단 (2026-09-02, 사용자 확인):
- 정규화 기준 = 400-2200 cm^-1 총 면적. subject간 기질 증강 변동(CV 22.5%)을 제거한다.
  정규화하지 않은 원시 contrast fold change도 감사용 컬럼으로 함께 낸다.
- common_peak_02 (1001.9 cm^-1) 는 축 보정 표준물질 Polystyrene 의 ring-breathing
  밴드와 파수가 겹치지만, PS는 축 보정용으로만 별도 측정되고 임상 검체에는 섞이지
  않는다는 사용자 확인에 따라 다른 피크와 동일하게 취급한다.
- local contrast는 clip 하지 않는다. 파이프라인의 peak 검출은 그룹 median에
  max(.,0) 을 쓰지만, subject 단위에서 clip하면 ratio가 0/inf로 붕괴한다.
- peak 세기는 대표 파수 ±1 grid point(총 3점) 평균. registry의
  candidate_centers_by_label 이 그룹 간 최대 ~2 cm^-1 중심 이동을 기록하고 있어
  단일 점 표본은 그룹별로 치우칠 수 있다. 단일 점 값은 민감도 컬럼으로 함께 낸다.
  창 안에서 max를 취하지 않는다 — 그룹마다 n이 달라 상향 편향이 불균등하게 걸린다.

한계: fold change는 subject 수준 기술 통계(descriptive effect size)이며 모델의
판별력과 다르다. 같은 입력에서 나온 screening OOF AUC는 0.6698로 크지 않다.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy.stats import false_discovery_control, kruskal, mannwhitneyu

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DIR = Path(
    os.environ.get(
        "AECD_MEAN_SPECTRUM_OUTDIR",
        str(REPO_ROOT / "notebooks/aecd_api_model_mean_spectrum_outputs"),
    )
)
OUTPUT_DIR = SOURCE_DIR / "peak_fold_change"
FIGURE_DIR = OUTPUT_DIR / "figures"

# 파이프라인과 동일한 local baseline 창 (SNR_PEAK_WLEN)
BASELINE_WINDOW = 61
# 대표 파수 ±1 grid point
PEAK_HALF_WIDTH_POINTS = 1
SPECTRAL_RANGE_CM1 = (400.0, 2200.0)
# 파이프라인이 피크 클러스터링에 쓰는 것과 같은 장비 분해능
INSTRUMENT_RESOLUTION_FWHM_CM1 = 2.0
BOOTSTRAP_DRAWS = 10000
RANDOM_STATE = 20260825

CANCER = "prostate"
DISEASE_CONTROL = "prostate disease control"
NORMAL = "control"
GROUP_ORDER = (NORMAL, DISEASE_CONTROL, CANCER)
GROUP_KOREAN = {NORMAL: "정상", DISEASE_CONTROL: "비암", CANCER: "암"}
GROUP_COLOR = {NORMAL: "#4C72B0", DISEASE_CONTROL: "#DD8452", CANCER: "#C44E52"}
COMPARISONS = (
    (CANCER, NORMAL),
    (CANCER, DISEASE_CONTROL),
    (DISEASE_CONTROL, NORMAL),
)


def configure_korean_font() -> str:
    """축/범례의 한글(암·비암·정상)이 두부로 깨지지 않도록 CJK 폰트를 고른다."""
    available = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in ("Noto Sans CJK KR", "Noto Sans KR", "NanumSquare", "NanumGothic"):
        if candidate in available:
            plt.rcParams["font.family"] = [candidate, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return candidate
    plt.rcParams["axes.unicode_minus"] = False
    return ""


def load_subject_spectra() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (wavenumbers, subject x wavenumber intensities, labels)."""
    path = SOURCE_DIR / "mean_representative_spectra.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. 먼저 "
            "scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py "
            "를 실행하세요."
        )
    frame = pd.read_csv(path, encoding="utf-8-sig")
    spectral_columns = [name for name in frame.columns if name.startswith("wn_")]
    if not spectral_columns:
        raise ValueError(f"{path} 에 wn_ 접두사 스펙트럼 컬럼이 없습니다.")
    wavenumbers = np.asarray([float(name[3:]) for name in spectral_columns])
    order = np.argsort(wavenumbers)
    intensities = frame[spectral_columns].to_numpy(dtype=np.float64)[:, order]
    labels = frame["label"].to_numpy()
    unexpected = sorted(set(labels) - set(GROUP_ORDER))
    if unexpected:
        raise ValueError(f"예상하지 못한 label: {unexpected}")
    if not np.isfinite(intensities).all():
        raise ValueError("subject 평균 스펙트럼에 non-finite 값이 있습니다.")
    return wavenumbers[order], intensities, labels


def load_peak_registry() -> pd.DataFrame:
    path = SOURCE_DIR / "resolution_aware_peak_registry.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다.")
    registry = pd.read_csv(path, encoding="utf-8-sig")
    if registry.empty:
        raise ValueError("peak registry가 비어 있습니다.")
    return registry.sort_values("representative_wavenumber_cm1").reset_index(drop=True)


def local_contrast(intensities: np.ndarray) -> np.ndarray:
    """Subtract the 61-point centred rolling-median baseline, without clipping."""
    baseline = (
        pd.DataFrame(intensities.T)
        .rolling(BASELINE_WINDOW, center=True, min_periods=1)
        .median()
        .to_numpy()
        .T
    )
    return intensities - baseline


def build_peak_features(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    labels: np.ndarray,
    registry: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Per-subject peak intensities, raw and total-area normalised."""
    contrast = local_contrast(intensities)
    total_area = np.trapezoid(intensities, wavenumbers, axis=1)
    if not (total_area > 0).all():
        raise ValueError("총 면적이 0 이하인 subject가 있습니다.")
    reference_area = float(np.median(total_area))
    # 비율에서는 상수배가 상쇄되지만, median 컬럼을 원시 contrast와 같은 자릿수로
    # 읽을 수 있도록 cohort median 면적으로 되돌려 스케일한다.
    normalised = contrast * (reference_area / total_area)[:, None]

    rows: list[dict] = []
    for _, peak in registry.iterrows():
        center = float(peak["representative_wavenumber_cm1"])
        center_index = int(np.argmin(np.abs(wavenumbers - center)))
        low = max(center_index - PEAK_HALF_WIDTH_POINTS, 0)
        high = min(center_index + PEAK_HALF_WIDTH_POINTS + 1, wavenumbers.size)
        for subject_index in range(intensities.shape[0]):
            rows.append(
                {
                    "peak_id": peak["peak_id"],
                    "representative_wavenumber_cm1": center,
                    "window_low_cm1": float(wavenumbers[low]),
                    "window_high_cm1": float(wavenumbers[high - 1]),
                    "subject_index_internal": subject_index,
                    "label": labels[subject_index],
                    "group_korean": GROUP_KOREAN[labels[subject_index]],
                    "total_area_400_2200": float(total_area[subject_index]),
                    "peak_contrast_raw_window": float(
                        contrast[subject_index, low:high].mean()
                    ),
                    "peak_contrast_raw_point": float(
                        contrast[subject_index, center_index]
                    ),
                    "peak_contrast_area_normalised_window": float(
                        normalised[subject_index, low:high].mean()
                    ),
                    "peak_contrast_area_normalised_point": float(
                        normalised[subject_index, center_index]
                    ),
                }
            )

    metadata = {
        "baseline_window_points": BASELINE_WINDOW,
        "peak_window_points": 2 * PEAK_HALF_WIDTH_POINTS + 1,
        "reference_area_400_2200": reference_area,
        "total_area_cv_overall": float(total_area.std(ddof=1) / total_area.mean()),
        "total_area_cv_by_group": {
            label: float(
                total_area[labels == label].std(ddof=1)
                / total_area[labels == label].mean()
            )
            for label in GROUP_ORDER
        },
    }
    return pd.DataFrame(rows), metadata


def bootstrap_median_ratio_ci(
    numerator: np.ndarray,
    denominator: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, float, int]:
    """Percentile CI for the ratio of group medians; also count invalid draws."""
    numerator_draws = rng.choice(
        numerator, size=(BOOTSTRAP_DRAWS, numerator.size), replace=True
    )
    denominator_draws = rng.choice(
        denominator, size=(BOOTSTRAP_DRAWS, denominator.size), replace=True
    )
    numerator_medians = np.median(numerator_draws, axis=1)
    denominator_medians = np.median(denominator_draws, axis=1)
    valid = denominator_medians > 0
    ratios = np.full(BOOTSTRAP_DRAWS, np.nan)
    ratios[valid] = numerator_medians[valid] / denominator_medians[valid]
    finite = ratios[np.isfinite(ratios)]
    invalid = int(BOOTSTRAP_DRAWS - finite.size)
    if finite.size < BOOTSTRAP_DRAWS // 2:
        return float("nan"), float("nan"), invalid
    low, high = np.percentile(finite, [2.5, 97.5])
    return float(low), float(high), invalid


def observed_peak_centers(
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    labels: np.ndarray,
    center: float,
) -> dict[str, float]:
    """Group-median contrast argmax within +/-2 grid points of the registry centre."""
    contrast = local_contrast(intensities)
    center_index = int(np.argmin(np.abs(wavenumbers - center)))
    low = max(center_index - 2, 0)
    high = min(center_index + 3, wavenumbers.size)
    centers: dict[str, float] = {}
    for label in GROUP_ORDER:
        median_profile = np.median(contrast[labels == label, low:high], axis=0)
        centers[label] = float(wavenumbers[low + int(np.argmax(median_profile))])
    return centers


def resolution_groups(registry: pd.DataFrame) -> dict[str, str]:
    """Label peaks that sit closer together than the instrument resolution.

    registry는 그룹별로 따로 검출된 후보를 묶으므로, 어떤 군에서만 검출된 피크가
    이웃 피크와 1 grid point(1.93 cm^-1 < FWHM 2.0 cm^-1) 떨어진 채 별도 행으로
    남을 수 있다. 그런 행은 독립적인 두 피처가 아니라 같은 피처이므로, 결과를
    두 건으로 세지 않도록 같은 resolution_group 을 붙인다.
    """
    ordered = registry.sort_values("representative_wavenumber_cm1")
    groups: dict[str, str] = {}
    group_index = 0
    previous: float | None = None
    for row in ordered.itertuples():
        center = float(row.representative_wavenumber_cm1)
        if (
            previous is None
            or center - previous > INSTRUMENT_RESOLUTION_FWHM_CM1
        ):
            group_index += 1
        groups[row.peak_id] = f"R{group_index:02d}"
        previous = center
    return groups


def build_fold_change_table(
    features: pd.DataFrame,
    registry: pd.DataFrame,
    wavenumbers: np.ndarray,
    intensities: np.ndarray,
    labels: np.ndarray,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    detected_in = dict(
        zip(registry["peak_id"], registry["labels_present"], strict=True)
    )
    rows: list[dict] = []

    for peak_id, peak_frame in features.groupby("peak_id", sort=True):
        center = float(peak_frame["representative_wavenumber_cm1"].iloc[0])
        centers = observed_peak_centers(wavenumbers, intensities, labels, center)
        by_group = {
            label: peak_frame.loc[
                peak_frame["label"] == label,
                "peak_contrast_area_normalised_window",
            ].to_numpy()
            for label in GROUP_ORDER
        }
        raw_by_group = {
            label: peak_frame.loc[
                peak_frame["label"] == label, "peak_contrast_raw_window"
            ].to_numpy()
            for label in GROUP_ORDER
        }
        point_by_group = {
            label: peak_frame.loc[
                peak_frame["label"] == label,
                "peak_contrast_area_normalised_point",
            ].to_numpy()
            for label in GROUP_ORDER
        }
        kruskal_p = float(kruskal(*[by_group[label] for label in GROUP_ORDER]).pvalue)

        for numerator_label, denominator_label in COMPARISONS:
            numerator = by_group[numerator_label]
            denominator = by_group[denominator_label]
            numerator_median = float(np.median(numerator))
            denominator_median = float(np.median(denominator))
            fold_change = (
                numerator_median / denominator_median
                if denominator_median > 0
                else float("nan")
            )
            ci_low, ci_high, invalid = bootstrap_median_ratio_ci(
                numerator, denominator, rng
            )
            test = mannwhitneyu(numerator, denominator, alternative="two-sided")
            # rank-biserial correlation: -1..1, 0 = no stochastic separation
            rank_biserial = float(
                2.0 * test.statistic / (numerator.size * denominator.size) - 1.0
            )
            raw_fold_change = (
                float(np.median(raw_by_group[numerator_label]))
                / float(np.median(raw_by_group[denominator_label]))
                if float(np.median(raw_by_group[denominator_label])) > 0
                else float("nan")
            )
            point_fold_change = (
                float(np.median(point_by_group[numerator_label]))
                / float(np.median(point_by_group[denominator_label]))
                if float(np.median(point_by_group[denominator_label])) > 0
                else float("nan")
            )
            rows.append(
                {
                    "peak_id": peak_id,
                    "representative_wavenumber_cm1": center,
                    "window_low_cm1": float(peak_frame["window_low_cm1"].iloc[0]),
                    "window_high_cm1": float(peak_frame["window_high_cm1"].iloc[0]),
                    "detected_in_labels": detected_in[peak_id],
                    "comparison": (
                        f"{GROUP_KOREAN[numerator_label]} vs "
                        f"{GROUP_KOREAN[denominator_label]}"
                    ),
                    "numerator_label": numerator_label,
                    "denominator_label": denominator_label,
                    "n_numerator": int(numerator.size),
                    "n_denominator": int(denominator.size),
                    "median_numerator": numerator_median,
                    "median_denominator": denominator_median,
                    "fold_change": fold_change,
                    "log2_fold_change": (
                        float(np.log2(fold_change))
                        if np.isfinite(fold_change) and fold_change > 0
                        else float("nan")
                    ),
                    "fold_change_ci_low": ci_low,
                    "fold_change_ci_high": ci_high,
                    "ci_excludes_1": bool(
                        np.isfinite(ci_low)
                        and np.isfinite(ci_high)
                        and (ci_low > 1.0 or ci_high < 1.0)
                    ),
                    "bootstrap_invalid_draws": invalid,
                    "mannwhitney_u": float(test.statistic),
                    "p_value": float(test.pvalue),
                    "rank_biserial_correlation": rank_biserial,
                    "kruskal_p_value_3group": kruskal_p,
                    "observed_center_numerator_cm1": centers[numerator_label],
                    "observed_center_denominator_cm1": centers[denominator_label],
                    "fold_change_raw_unnormalised": raw_fold_change,
                    "fold_change_single_point": point_fold_change,
                }
            )

    table = pd.DataFrame(rows)
    table["resolution_group"] = table["peak_id"].map(
        resolution_groups(registry)
    )

    # 주 검정군 = 비교 1종 안의 피크 9개. 세 비교는 대조군을 공유하므로
    # (암 vs 정상 / 비암 vs 정상 모두 같은 정상 21명) 서로 독립이 아니다.
    table["q_value_bh"] = np.nan
    for comparison in table["comparison"].unique():
        mask = table["comparison"] == comparison
        table.loc[mask, "q_value_bh"] = false_discovery_control(
            table.loc[mask, "p_value"].to_numpy(), method="bh"
        )
    table["significant_q05"] = table["q_value_bh"] < 0.05
    # 보수적 대안 = 표 전체 27개 검정.
    table["q_value_bh_all_tests"] = false_discovery_control(
        table["p_value"].to_numpy(), method="bh"
    )
    table["significant_q05_all_tests"] = table["q_value_bh_all_tests"] < 0.05
    return table.sort_values(
        ["comparison", "representative_wavenumber_cm1"]
    ).reset_index(drop=True)


def plot_log2_fold_change(table: pd.DataFrame, path: Path) -> None:
    comparisons = list(dict.fromkeys(table["comparison"]))
    figure, axes = plt.subplots(
        1, len(comparisons), figsize=(5.2 * len(comparisons), 5.4), sharey=True
    )
    axes = np.atleast_1d(axes)
    for axis, comparison in zip(axes, comparisons, strict=True):
        subset = table[table["comparison"] == comparison].sort_values(
            "representative_wavenumber_cm1"
        )
        positions = np.arange(len(subset))
        values = subset["log2_fold_change"].to_numpy()
        low = np.log2(subset["fold_change_ci_low"].to_numpy())
        high = np.log2(subset["fold_change_ci_high"].to_numpy())
        colors = [
            "#C44E52" if flag else "#B0B0B0"
            for flag in subset["significant_q05"]
        ]
        axis.barh(positions, values, color=colors, height=0.62)
        axis.errorbar(
            values,
            positions,
            xerr=np.vstack([values - low, high - values]),
            fmt="none",
            ecolor="#333333",
            elinewidth=1.0,
            capsize=3,
        )
        axis.axvline(0.0, color="black", linewidth=0.9)
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [
                f"{row.peak_id.replace('common_peak_', 'P')}  "
                f"{row.representative_wavenumber_cm1:.0f}"
                for row in subset.itertuples()
            ]
        )
        n_numerator = int(subset["n_numerator"].iloc[0])
        n_denominator = int(subset["n_denominator"].iloc[0])
        axis.set_title(f"{comparison}\n(n={n_numerator} vs {n_denominator})")
        axis.set_xlabel("log2 fold change (median ratio)")
        axis.invert_yaxis()
    axes[0].set_ylabel("peak (cm$^{-1}$)")
    figure.suptitle(
        "Peak-wise fold change — total-area-normalised local contrast\n"
        "bars: median ratio; whiskers: 95% bootstrap CI; "
        "red: BH-FDR q<0.05 (Mann-Whitney U)",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def plot_peak_distributions(features: pd.DataFrame, path: Path) -> None:
    peaks = list(dict.fromkeys(features["peak_id"]))
    columns = 3
    rows = int(np.ceil(len(peaks) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 3.3 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, peak_id in zip(axes, peaks, strict=False):
        subset = features[features["peak_id"] == peak_id]
        data = [
            subset.loc[
                subset["label"] == label, "peak_contrast_area_normalised_window"
            ].to_numpy()
            for label in GROUP_ORDER
        ]
        boxes = axis.boxplot(
            data, patch_artist=True, widths=0.6, showfliers=False,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for patch, label in zip(boxes["boxes"], GROUP_ORDER, strict=True):
            patch.set_facecolor(GROUP_COLOR[label])
            patch.set_alpha(0.55)
        for position, (values, label) in enumerate(zip(data, GROUP_ORDER, strict=True), 1):
            jitter = np.random.default_rng(RANDOM_STATE + position).normal(
                0.0, 0.05, values.size
            )
            axis.plot(
                position + jitter, values, ".", color=GROUP_COLOR[label],
                markersize=4, alpha=0.7,
            )
        axis.set_xticks([1, 2, 3])
        axis.set_xticklabels(
            [f"{GROUP_KOREAN[label]}\n(n={int((subset['label'] == label).sum())})"
             for label in GROUP_ORDER]
        )
        center = float(subset["representative_wavenumber_cm1"].iloc[0])
        axis.set_title(f"{peak_id.replace('common_peak_', 'P')}  {center:.1f} cm$^{{-1}}$")
        axis.set_ylabel("normalised contrast (a.u.)")
        axis.grid(axis="y", alpha=0.25)
    for axis in axes[len(peaks):]:
        axis.set_visible(False)
    figure.suptitle(
        "Subject-level peak contrast by group — total-area-normalised, "
        "61-point rolling-median baseline removed\n"
        "box: IQR, line: median, whiskers: 1.5xIQR (outliers hidden, points shown)",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=200)
    plt.close(figure)


def write_readme(table: pd.DataFrame, metadata: dict, path: Path) -> None:
    significant = table[table["significant_q05"]]
    lines = [
        "# Peak-wise fold change: 암 / 비암 / 정상",
        "",
        "생성: `python scripts/analysis/aecd_peak_group_fold_change.py`",
        "",
        "## 정의",
        "",
        "- 입력: `../mean_representative_spectra.csv` (subject별 PS 축보정 + repeat-QC 평균),",
        "  `../resolution_aware_peak_registry.csv` (그룹 median에서 검출된 공통 피크)",
        f"- local baseline: {metadata['baseline_window_points']}-point centred rolling median, clip 없음",
        f"- peak 세기: 대표 파수 ±1 grid point ({metadata['peak_window_points']}점) 평균",
        "- 정규화: subject별 400–2200 cm⁻¹ 총 면적. cohort median 면적으로 되돌려 스케일",
        f"  (총 면적 CV = {metadata['total_area_cv_overall']:.3f})",
        "- fold change = 그룹 median 비. 95% CI는 median ratio의 percentile bootstrap "
        f"({BOOTSTRAP_DRAWS:,}회, seed={RANDOM_STATE})",
        "- 검정: Mann-Whitney U (two-sided). 3군 동시비교는 Kruskal-Wallis",
        "- BH-FDR 검정군: 주 지표 `q_value_bh` 는 비교 1종 안의 피크 "
        f"{table['peak_id'].nunique()}개. 세 비교는 서로 독립이 아니다 — "
        "암 vs 정상과 비암 vs 정상이 같은 정상 21명을 공유한다. "
        f"표 전체 {len(table)}개 검정에 건 보수적 값은 `q_value_bh_all_tests`",
        "- `resolution_group`: 대표 파수가 장비 분해능(FWHM "
        f"{INSTRUMENT_RESOLUTION_FWHM_CM1:.1f} cm⁻¹)보다 가까운 피크는 같은 그룹이다. "
        "registry가 군별로 후보를 따로 검출해 같은 피처가 두 행으로 남을 수 있으므로, "
        "같은 그룹의 행을 독립적인 두 결과로 세지 않는다",
        "",
        "## 결과 요약",
        "",
        f"- 피크 {table['peak_id'].nunique()}개 × 비교 3종 = 검정 {len(table)}건",
        f"- BH-FDR q<0.05: {len(significant)}건",
    ]
    if significant.empty:
        lines.append("- 유의한 피크 없음")
    else:
        for row in significant.sort_values("q_value_bh").itertuples():
            lines.append(
                f"  - {row.peak_id} ({row.representative_wavenumber_cm1:.1f} cm⁻¹) "
                f"{row.comparison}: FC={row.fold_change:.3f} "
                f"[{row.fold_change_ci_low:.3f}, {row.fold_change_ci_high:.3f}], "
                f"q={row.q_value_bh:.4f}"
            )
    lines += [
        "",
        "## 해석 시 주의",
        "",
        "- fold change는 subject 수준 기술 통계이며 판별력이 아니다. 동일 입력의 "
        "screening OOF AUC는 0.6698이므로, 큰 FC를 판별 성능으로 읽으면 안 된다.",
        "- 정상군 n=21이 통계적 제약이다. CI가 1을 포함하는 피크는 점추정으로 순위를 매기지 않는다.",
        "- `detected_in_labels`는 그룹 median에서 피크가 검출된 군이다. 검출되지 않은 군에서도 "
        "동일 파수의 값을 측정해 비교하므로, 검출 비대칭 자체가 결과의 일부다.",
        "- common_peak_02 (1001.9 cm⁻¹)는 축 보정 표준물질 Polystyrene의 밴드와 파수가 겹치나, "
        "PS는 축 보정용으로만 별도 측정되고 임상 검체에 섞이지 않는다는 확인에 따라 "
        "다른 피크와 동일하게 취급했다 (2026-09-02).",
        "- common_peak_02 (1001.9 cm⁻¹)의 패턴은 사실대로만 읽어야 한다: 암과 비암이 "
        "모두 정상 대비 약 1.25배 높고(비암 vs 정상 p=0.016, 암 vs 정상 p=0.042), "
        "암과 비암 사이에는 차이가 없다(FC=0.996, p=0.633). 두 환자군이 함께 올라가고 "
        "환자군끼리는 갈리지 않는 형태이므로, 암 관련 신호와 검체 수집·취급 차이를 "
        "이 데이터만으로 구분할 수 없다. 정상군 n=21은 수집 경로가 다를 가능성이 가장 큰 군이다.",
        "- 파수를 분자에 귀속시키지 않는다. 이 산출물은 파수와 fold change만 보고한다.",
        "- 이 코호트는 단일 기관(smcxd07)이며 병원/배치 교란이 제거되지 않았다.",
        "",
        "## 파일",
        "",
        "- `peak_group_fold_change.csv` — 피크 × 비교 주 결과표",
        "- `peak_group_subject_values.csv` — subject별 피크 값 (long format)",
        "- `peak_group_fold_change_metadata.json` — 실행 파라미터",
        "- `figures/peak_log2_fold_change.png` — log2 FC + 95% CI",
        "- `figures/peak_group_distributions.png` — 피크별 그룹 분포",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    korean_font = configure_korean_font()

    wavenumbers, intensities, labels = load_subject_spectra()
    registry = load_peak_registry()
    features, metadata = build_peak_features(
        wavenumbers, intensities, labels, registry
    )
    table = build_fold_change_table(
        features, registry, wavenumbers, intensities, labels
    )

    features.to_csv(
        OUTPUT_DIR / "peak_group_subject_values.csv",
        index=False,
        encoding="utf-8-sig",
    )
    table.to_csv(
        OUTPUT_DIR / "peak_group_fold_change.csv", index=False, encoding="utf-8-sig"
    )
    metadata |= {
        "source_dir": str(SOURCE_DIR),
        "spectral_range_cm1": list(SPECTRAL_RANGE_CM1),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "random_state": RANDOM_STATE,
        "subjects_by_group": {
            label: int((labels == label).sum()) for label in GROUP_ORDER
        },
        "peaks": int(table["peak_id"].nunique()),
        "tests": int(len(table)),
        "significant_q05": int(table["significant_q05"].sum()),
        "korean_font": korean_font or "(없음 — 한글 라벨이 깨질 수 있음)",
        "peak_02_polystyrene_note": (
            "1001.9 cm^-1 는 축 보정 표준물질 PS 밴드와 겹치지만 임상 검체에 PS 잔류가 "
            "없다는 확인에 따라 동일 취급 (2026-09-02)"
        ),
    }
    (OUTPUT_DIR / "peak_group_fold_change_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_log2_fold_change(table, FIGURE_DIR / "peak_log2_fold_change.png")
    plot_peak_distributions(features, FIGURE_DIR / "peak_group_distributions.png")
    write_readme(table, metadata, OUTPUT_DIR / "README.md")

    print("=== Peak-wise fold change (총 면적 정규화 local contrast) ===")
    print(
        f"[data] subjects={intensities.shape[0]} "
        f"peaks={table['peak_id'].nunique()} tests={len(table)}"
    )
    display_columns = [
        "peak_id",
        "representative_wavenumber_cm1",
        "comparison",
        "fold_change",
        "fold_change_ci_low",
        "fold_change_ci_high",
        "p_value",
        "q_value_bh",
    ]
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(table[display_columns].to_string(index=False, float_format="%.4f"))
    print(f"[significant] BH-FDR q<0.05: {int(table['significant_q05'].sum())}건")
    print(f"[output] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
