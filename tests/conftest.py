"""Shared fixtures for SERS test suite."""


import numpy as np
import pandas as pd
import pytest

from tests.helpers import make_spectrum

# This test targets a module relocated to scripts/legacy/ during the repo
# reorganization; its import path is stale and breaks collection. Disabled
# until the team decides whether to update the import or retire the test.
collect_ignore = ["test_master_clinical_risk_experiment.py"]


# ---------------------------------------------------------------------------
# Wavenumber grids
# ---------------------------------------------------------------------------
@pytest.fixture
def wavenumber_grid():
    """Common wavenumber grid spanning the fingerprint region (400-2200 cm-1)."""
    return np.linspace(400, 2200, 935)


@pytest.fixture
def full_wavenumber_grid():
    """Full wavenumber grid including regions outside fingerprint (50-3000 cm-1)."""
    return np.linspace(50, 3000, 1500)


@pytest.fixture
def small_grid():
    """Small grid for fast tests."""
    return np.linspace(400, 2200, 100)


# ---------------------------------------------------------------------------
# Synthetic spectra
# ---------------------------------------------------------------------------
@pytest.fixture
def synthetic_spectrum(wavenumber_grid):
    """Single synthetic SERS spectrum (x, y)."""
    return wavenumber_grid, make_spectrum(wavenumber_grid)


@pytest.fixture
def synthetic_spectrum_pair(wavenumber_grid):
    """Two spectra: one clean, one with higher baseline and noise."""
    y_clean = make_spectrum(wavenumber_grid, noise_level=0.005, seed=1)
    y_noisy = make_spectrum(wavenumber_grid, noise_level=0.05, baseline_slope=2.0, seed=2)
    return wavenumber_grid, y_clean, y_noisy


@pytest.fixture
def replicate_spectra(small_grid):
    """Five replicate spectra for the same sample, with slight variation.

    Returns dict with keys (group, sample_id, replicate) -> (x, y).
    """
    spectra = {}
    for rep in range(1, 6):
        y = make_spectrum(small_grid, noise_level=0.01, seed=rep)
        spectra[("CRC", "001", rep)] = (small_grid.copy(), y)
    return spectra


@pytest.fixture
def multi_sample_spectra(small_grid):
    """Multiple samples with replicates for QC testing.

    Two samples (CRC-001 good, CRC-002 noisy) with 3 replicates each.
    """
    spectra = {}
    # Good sample: low noise
    for rep in range(1, 4):
        y = make_spectrum(small_grid, noise_level=0.005, seed=100 + rep)
        spectra[("CRC", "001", rep)] = (small_grid.copy(), y)

    # Noisy sample: high noise
    for rep in range(1, 4):
        y = make_spectrum(small_grid, noise_level=0.15, seed=200 + rep)
        spectra[("CRC", "002", rep)] = (small_grid.copy(), y)

    return spectra


@pytest.fixture
def low_intensity_spectra(small_grid):
    """Spectra with one sample having very low intensity (enhancement failure).

    Returns dict with 3 normal and 3 very low intensity spectra.
    """
    spectra = {}
    # Normal intensity
    for rep in range(1, 4):
        y = make_spectrum(small_grid, noise_level=0.01, seed=rep)
        spectra[("NOR", "001", rep)] = (small_grid.copy(), y)

    # Very low intensity (SERS enhancement failure)
    for rep in range(1, 4):
        y = make_spectrum(small_grid, noise_level=0.01, seed=10 + rep) * 0.001
        spectra[("NOR", "002", rep)] = (small_grid.copy(), y)

    return spectra


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_config_dict():
    """Minimal valid config dictionary."""
    return {
        "preprocessing": {
            "do_smooth": True,
            "smooth_window": 11,
            "smooth_poly": 3,
            "baseline_window": 101,
            "use_snv": True,
        },
        "qc": {
            "rsd_threshold": 5.0,
            "corr_threshold": 0.95,
            "expected_reps": 5,
            "fingerprint_region": [400, 2200],
            "intensity_gate_ratio": 0.1,
        },
        "modeling": {
            "n_splits": 5,
            "use_pca": False,
            "pca_components": 10,
            "random_state": 42,
        },
        "dataset": {
            "folder_to_group": {
                "1. Prostate cancer (100)": "PRO",
                "5. Normal (100)": "NOR",
            },
        },
    }


@pytest.fixture
def config_yaml_path(minimal_config_dict, tmp_path):
    """Write minimal config to a temporary YAML file and return its path."""
    import yaml

    path = tmp_path / "config.yaml"
    with open(path, "w") as f:
        yaml.dump(minimal_config_dict, f)
    return path


# ---------------------------------------------------------------------------
# Temp directory & CSV files
# ---------------------------------------------------------------------------
@pytest.fixture
def tmp_spectrum_csv(tmp_path):
    """Create a temporary 2-column CSV spectrum file."""
    x = np.linspace(400, 2200, 200)
    y = make_spectrum(x)
    df = pd.DataFrame({"raman_shift": x, "intensity": y})
    path = tmp_path / "CRC 001_1.csv"
    df.to_csv(path, index=False, header=False)
    return path


@pytest.fixture
def tmp_spectrum_dir(tmp_path):
    """Create a temporary directory with multiple spectrum CSV files.

    Structure: tmp_path/group_folder/GROUP ID_REP.csv
    """
    group_dir = tmp_path / "1. CRC (10)"
    group_dir.mkdir()
    x = np.linspace(400, 2200, 200)
    for sid in range(1, 4):
        for rep in range(1, 4):
            y = make_spectrum(x, seed=sid * 10 + rep)
            df = pd.DataFrame({"raman_shift": x, "intensity": y})
            fname = f"CRC {sid:03d}_{rep}.csv"
            df.to_csv(group_dir / fname, index=False, header=False)
    return tmp_path
