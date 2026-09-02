#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = [
#   "matplotlib>=3.7",
#   "numpy>=1.24",
#   "openpyxl>=3.1",
#   "scipy>=1.10",
# ]
# ///
# How to run:
#   uv run publications/전향검체/보라매병원/src/generate_inverted_band_figure.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[4]
SOURCE_DIR: Final = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(SOURCE_DIR))

from boramae_data import load_clinical_samples
from powder_comparison.cohort import collect_subject_files, pair_subjects
from powder_comparison.data import build_paired_spectra, load_clinical_labels
from powder_comparison.inverted_band_figure import write_inverted_band_figure
from powder_comparison.runner import default_sources


def main() -> None:
    sources = default_sources(REPO)
    labels = load_clinical_labels(sources.clinical_path)
    pairs = pair_subjects(
        labels,
        collect_subject_files(sources.legacy_root),
        collect_subject_files(sources.powder_root),
    )
    spectra = build_paired_spectra(
        pairs,
        np.load(sources.artifact_dir / "common_grid.npy"),
    )
    grades = {int(sample.sample_no): sample.grade_group for sample in load_clinical_samples()}
    grade_bands = np.array(
        [
            "GG1-2"
            if grades[int(sample_id)] in {1, 2}
            else "GG3-5"
            if grades[int(sample_id)] in {3, 4, 5}
            else "Unknown"
            for sample_id in spectra.sample_ids
        ]
    )
    write_inverted_band_figure(spectra, grade_bands, sources.output_dir / "figures")


if __name__ == "__main__":
    main()
