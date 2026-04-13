# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SERS-AI is a SERS (Surface-Enhanced Raman Spectroscopy) analysis pipeline for multi-cancer screening from urine metabolites. Built by SOLUM Healthcare, it processes spectral data from multiple Raman instruments (Thermo, Medical) across 11+ cancer/control groups collected from Korean hospitals.

## Build & Install

```bash
conda env create -f config/environment.yml
conda activate sers-analysis
pip install -e ".[all]"          # editable install with all optional deps
```

## Common Commands

```bash
# Run full test suite
pytest

# Run a single test file or test
pytest tests/test_qc.py
pytest tests/test_signal.py::test_smooth_preserves_peaks -v

# Skip slow tests (those needing real data files)
pytest -m "not slow"

# Lint
ruff check src/ tests/

# Run the preprocessing pipeline
python main.py                       # Thermo data (default)
python main.py --data-source medical # Medical Raman data

# Run QC preprocessing pipeline script
python scripts/pipeline/run_qc_preprocess.py
```

## Architecture

### Source layout (`src/sers/`)

The package follows a pipeline architecture: **I/O → Signal → Preprocessing → QC → Analysis → Visualization**.

- **`config.py`** — Frozen dataclass hierarchy (`Config` → `PreprocessingConfig`, `QCConfig`, `ModelingConfig`, `DisplayConfig`, `MedicalConfig`). Loads from `config/config.yaml`. Handles WSL path auto-detection and Windows↔Linux path conversion via `.env`.
- **`io.py`** — Spectrum file I/O. `read_spectrum()` reads 2-column CSV/TXT files. `parse_filename()` extracts `SpectrumID(group, sample_id, replicate)` from filenames like `CRC 001_3.CSV`. `load_dataset()` handles Medical data with background subtraction and filtering.
- **`signal.py`** — Low-level signal processing primitives: `smooth` (Savitzky-Golay), `baseline_correction` (rolling minimum), `snv` (Standard Normal Variate), `resample` (interpolation to common grid).
- **`preprocessing.py`** — Orchestrates the signal chain: calibration → resample → smooth → baseline → SNV. `preprocess_spectra()` is the main entry point, taking raw spectra dict and config.
- **`qc/`** — Two-stage quality control. Stage 1 (raw): intensity gate, cosmic ray detection, saturation. Stage 2 (post-preprocess): per-replicate correlation-to-mean, minimum surviving replicates. `run_qc_pipeline()` runs both stages.
- **`analysis.py`** — Dataset exploration: group statistics, completeness checks, duplicate detection.
- **`validation.py`** — Input validation: CSV structure, spectrum integrity, batch validation.
- **`visualization/`** — Plotting subpackage: `spectra.py` (spectrum overlays, mean spectra), `distribution.py` (sample distributions), `analysis.py` (variance heatmaps), `shap.py` (SHAP interpretability plots).
- **`scoring.py`** — Model scoring utilities.
- **`calibration_transfer.py`** — Cross-instrument calibration (Piecewise Direct Standardization).

### Key data structure

Spectra are stored as `dict[(group, sample_id, replicate), (x_array, y_array)]` throughout the pipeline. Group codes are 3-4 letter abbreviations (CRC, LUN, NOR, etc.) defined in `config/config.yaml`.

### Dual data sources

- **Thermo**: CSV files in `data/raw_data/`, 5 replicates per sample, already background-subtracted.
- **Medical**: TXT files in `data/raw_data_medical/`, 6 replicates, raw + BG in `Background/` subdir. Requires -28 cm⁻¹ wavenumber shift for Thermo alignment. SG smoothing and calibration are skipped.

### Scripts directory

- `scripts/pipeline/` — Pipeline runners (QC preprocessing, threshold experiments)
- `scripts/analysis/` — One-off analysis scripts (cross-instrument, feature ablation, SHAP, ensemble experiments)
- `scripts/qc_validation/` — Numbered QC validation phases (01-10): literature standards → threshold derivation → hospital confound checks
- `scripts/deployment/` — Clinical webapp (Streamlit), API server, Windows installer
- `scripts/db/` — PostgreSQL/Supabase upload utilities

## Conventions

- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/) (enforced by pre-commit hook): `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `chore:`, `build:`, `ci:`
- Linting: ruff with numpy-style docstrings, 100-char line length, Python 3.10+ target
- Public API is defined in `src/sers/__init__.py` via `__all__`. Internal helpers may change without notice.
- Config is the single source of truth for thresholds and parameters — do not hardcode magic numbers in modules.
- `main.py` imports from `src.sers.*` (development path); installed package uses `from sers import ...`.
