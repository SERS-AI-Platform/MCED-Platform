from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from boramae_data import (
    COLORS,
    FIG_DIR,
    LABELS,
    OUT,
    ClinicalSample,
    SubjectSpectrum,
    collect_boramae_files,
    group_matrix,
    preprocess_file,
    raw_key,
)
from boramae_peak_modes import PeakModeResult
from scipy.signal import find_peaks
from sklearn.decomposition import PCA


def save_plot(name: str) -> None:
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_preprocessing(subjects: list[SubjectSpectrum], grid: np.ndarray) -> None:
    stages_by_group: dict[str, list[dict[str, tuple[np.ndarray, np.ndarray]]]] = {
        label: [] for label in LABELS
    }
    files = collect_boramae_files()
    for subject in subjects:
        stages, _ = preprocess_file(files[raw_key(subject.sample, files)][0], grid)
        stages_by_group[subject.sample.group].append(stages)
    stage_names = list(next(iter(stages_by_group[LABELS[0]])).keys())
    fig, axes = plt.subplots(len(stage_names), len(LABELS), figsize=(15, 12), sharex=False)
    for row, stage in enumerate(stage_names):
        for col, label in enumerate(LABELS):
            ax = axes[row, col]
            for item in stages_by_group[label][:25]:
                x, y = item[stage]
                ax.plot(x, y, color=COLORS[label], alpha=0.12, lw=0.8)
            xs, _ = stages_by_group[label][0][stage]
            mat = np.vstack([np.interp(xs, *item[stage]) for item in stages_by_group[label]])
            ax.plot(xs, mat.mean(axis=0), color=COLORS[label], lw=1.6)
            ax.set_title(label if row == 0 else "")
            ax.set_ylabel(stage)
            ax.grid(alpha=0.2)
    fig.suptitle("Fig01. Boramae prospective cohort preprocessing audit", y=0.995, fontsize=15)
    save_plot("fig01_boramae_preprocessing_pipeline")


def plot_clean_pro_comparison(
    subjects: list[SubjectSpectrum], clean_pro: np.ndarray, grid: np.ndarray
) -> None:
    boramae = group_matrix(subjects, "Prostate cancer")
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    items = [
        ("Clean PRO", clean_pro, "#4D4D4D"),
        ("Boramae Cancer", boramae, COLORS["Prostate cancer"]),
    ]
    for name, mat, color in items:
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(len(mat))
        axes[0].plot(grid, mean, label=f"{name} (n={len(mat)})", color=color, lw=1.7)
        axes[0].fill_between(grid, mean - 1.96 * sem, mean + 1.96 * sem, color=color, alpha=0.18)
    axes[1].plot(grid, boramae.mean(axis=0) - clean_pro.mean(axis=0), color="#B2182B", lw=1.5)
    axes[0].set_title("Same preprocessing: clean PRO vs Boramae prostate")
    axes[0].legend(frameon=False)
    axes[1].set_title("Mean difference: Boramae - clean PRO")
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    for ax in axes:
        ax.set_ylabel("SNV intensity")
        ax.grid(alpha=0.2)
    save_plot("fig02_boramae_vs_legacy_pro_processed")


def plot_prostate_grade_difference(subjects: list[SubjectSpectrum], grid: np.ndarray) -> None:
    prostate = [
        sub
        for sub in subjects
        if sub.sample.group == "Prostate cancer" and sub.sample.grade_group is not None
    ]
    low = np.vstack([sub.mean_spectrum for sub in prostate if sub.sample.grade_group in {1, 2}])
    high = np.vstack([sub.mean_spectrum for sub in prostate if sub.sample.grade_group in {3, 4, 5}])
    pca = PCA(n_components=2, random_state=42).fit_transform(
        np.vstack([sub.mean_spectrum for sub in prostate])
    )
    labels = ["GG 1-2" if sub.sample.grade_group in {1, 2} else "GG 3-5" for sub in prostate]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for label, color in [("GG 1-2", "#1B9E77"), ("GG 3-5", "#D95F02")]:
        mask = np.array([item == label for item in labels])
        axes[0].scatter(
            pca[mask, 0], pca[mask, 1], label=f"{label} (n={mask.sum()})", color=color, alpha=0.85
        )
    axes[0].set_title("PCA of Cancer spectra")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].legend(frameon=False)
    axes[1].plot(grid, low.mean(axis=0), color="#1B9E77", label=f"GG 1-2 (n={len(low)})")
    axes[1].plot(grid, high.mean(axis=0), color="#D95F02", label=f"GG 3-5 (n={len(high)})")
    axes[1].set_title("Mean spectra by Grade Group")
    axes[1].legend(frameon=False)
    diff = high.mean(axis=0) - low.mean(axis=0)
    axes[2].plot(grid, diff, color="#B2182B")
    peak_idx, _ = find_peaks(
        np.abs(diff), distance=10, prominence=max(0.05, float(np.ptp(diff)) * 0.05)
    )
    top_idx = peak_idx[np.argsort(np.abs(diff[peak_idx]))[-8:]]
    axes[2].scatter(grid[top_idx], diff[top_idx], color="#B2182B", s=18)
    peak_text = ", ".join(
        f"{grid[item]:.0f}" for item in sorted(top_idx, key=lambda item: grid[item])
    )
    axes[2].text(
        0.03,
        0.97,
        f"Top |Δ| peaks\n{peak_text} cm$^{{-1}}$",
        transform=axes[2].transAxes,
        va="top",
        fontsize=8,
    )
    axes[2].set_title("Difference: GG 3-5 - GG 1-2")
    for ax in axes:
        ax.grid(alpha=0.2)
    axes[1].set_xlabel("Raman shift (cm$^{-1}$)")
    axes[2].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.suptitle("Fig05. Cancer internal subgroup check by Grade Group", y=1.02)
    save_plot("fig05_prostate_grade_group_difference")


def write_summary(
    samples: list[ClinicalSample],
    subjects: list[SubjectSpectrum],
    screening_peaks: PeakModeResult,
    three_group_peaks: PeakModeResult,
) -> None:
    excluded = [sample.label for sample in samples if sample.excluded]
    counts = {label: sum(1 for sub in subjects if sub.sample.group == label) for label in LABELS}
    with (OUT / "SUMMARY.md").open("w", encoding="utf-8") as fh:
        fh.write("# 보라매병원 전향검체 SERS 분석 산출물\n\n")
        fh.write("## Cohort\n\n")
        fh.write("- Raw unique: 120명 (`BPRO` 99, `BNOR` 21)\n")
        fh.write(f"- 노란색/형광색 제외: {len(excluded)}명\n")
        fh.write(
            f"- 분석 포함: {len(subjects)}명 — Control {counts['Control']}, Biopsy-negative {counts['Biopsy-negative']}, Prostate {counts['Prostate cancer']}\n"
        )
        fh.write(
            "- `_ave` 파일은 제외하고 `_1.._5` replicate만 사용, sample-level mean으로 분석\n\n"
        )
        fh.write("## Figures\n\n")
        figures = [
            ("fig01_boramae_preprocessing_pipeline", "보라매 3군 preprocessing audit"),
            (
                "fig02_boramae_vs_legacy_pro_processed",
                "Clean PRO 91명 vs 보라매 전향 전립선암 41명",
            ),
            (
                "fig04a_screening_peak_sets_common_differential",
                "Screening peak set: Non-cancer vs Cancer",
            ),
            (
                "fig04b_three_group_peak_sets_common_differential",
                "3그룹 peak set: Control / Biopsy-negative / Cancer",
            ),
            ("fig05_prostate_grade_group_difference", "보라매 전립선암 Grade Group 내부 비교"),
        ]
        for idx, (name, description) in enumerate(figures, start=1):
            fh.write(f"- Fig{idx:02d}: `figures/{name}.png` / `.pdf` — {description}\n")
        fh.write("\n## Peak criteria\n\n")
        fh.write(
            "- 그룹 평균 preprocessed SNV spectrum에서 `scipy.signal.find_peaks`로 local maximum 산출\n"
        )
        fh.write("- prominence 기준: `max(0.08, group mean range의 10%)`, 최소 간격 18 cm^-1\n")
        fh.write("- 공통 peak: 표시된 모든 그룹이 +/-12 cm^-1 안에 peak를 가진 cluster\n")
        fh.write("- shaded area: group mean intensity range가 큰 differential peak cluster top 5\n")
        fh.write(
            f"- Fig04a Screening: 공통 {screening_peaks.common_count}개, "
            f"differential {screening_peaks.differential_count}개, "
            f"shaded peak {screening_peaks.shaded_count}개\n"
        )
        fh.write(
            f"- Fig04b 3-group: 공통 {three_group_peaks.common_count}개, "
            f"differential {three_group_peaks.differential_count}개, "
            f"shaded peak {three_group_peaks.shaded_count}개\n"
        )
