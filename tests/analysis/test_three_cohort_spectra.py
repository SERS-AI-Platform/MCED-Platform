from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SRC = Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "보라매병원" / "src"
sys.path.insert(0, str(SRC))

from three_cohort_spectra import CohortSeries, _mapping_spectra, create_three_cohort_figure

REPO = Path(__file__).resolve().parents[2]


def test_comparison_shows_trimmed_raw_and_processed_panels() -> None:
    # Given: three subject-level cohorts on one Raman-shift grid.
    grid = np.linspace(400.0, 2200.0, 20)
    cohorts = tuple(
        CohortSeries(
            label=label,
            raw_spectra=np.vstack([1000.0 + 100.0 * np.sin(grid / 100.0)] * count),
            processed_spectra=np.vstack([np.sin(grid / 100.0) + offset] * count),
            color=color,
            line_style=line_style,
        )
        for label, count, offset, color, line_style in (
            ("후향 전립선암 데이터 (n=3)", 3, 0.0, "#4D4D4D", "-"),
            ("보라매 검체 (액상 환원제, n=4)", 4, 0.2, "#D95F02", "--"),
            ("보라매 검체 (분말 환원제, n=5)", 5, -0.2, "#2C7FB8", "-."),
        )
    )

    # When: the publication comparison figure is created.
    figure = create_three_cohort_figure(grid, cohorts)

    # Then: raw and processed spectra are shown without a difference subplot.
    assert len(figure.axes) == 2
    assert len(figure.axes[0].lines) == 3
    assert len(figure.axes[1].lines) == 3
    assert figure.axes[0].get_ylabel() == "Raw intensity (a.u.)"
    assert figure.axes[1].get_ylabel() == "SNV intensity"
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
        "후향 전립선암 데이터 (n=3)",
        "보라매 검체 (액상 환원제, n=4)",
        "보라매 검체 (분말 환원제, n=5)",
    ]
    plt.close(figure)


def test_mapping_spectra_include_only_prostate_cancer_on_canonical_grid() -> None:
    # Given: the canonical 935-point comparison grid and verified new-reagent mapping data.
    grid = np.load(REPO / "artifacts" / "usersnet" / "v1.0.0" / "common_grid.npy")

    # When: subject-level mapping spectra are prepared for the comparison.
    raw_spectra, processed_spectra = _mapping_spectra(REPO, grid)

    # Then: both raw and processed data contain the same 43 cancer subjects and grid.
    assert raw_spectra.shape == (43, len(grid))
    assert processed_spectra.shape == (43, len(grid))
