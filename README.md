# SERS Analysis Pipeline

SERS-based spectral analysis pipeline for multi-cancer screening from urine metabolites.

## Installation

### Using Conda (Recommended)
```bash
# Create environment
conda env create -f config/environment.yml
conda activate sers-analysis
```

### Using pip
```bash
pip install -e .

# With optional dependencies
pip install -e ".[all]"
```

## Quick Start

### Command Line
```bash
# Validate input files
sers validate -i data/raw

# Run pipeline
sers run -i data/raw -o results/
```

### Python API (Stable Public API)
```python
from pathlib import Path

from sers import (
    load_config,
    find_spectra,
    parse_filename,
    read_spectrum,
    make_common_grid,
    preprocess_spectra,
    run_qc_pipeline,
)

config = load_config("config/config.yaml")
files = find_spectra(Path("data/raw"))

spectra = {}
x_arrays = []
for path in files:
    sid = parse_filename(path)
    x, y = read_spectrum(path)
    spectra[(sid.group, sid.sample_id, sid.replicate)] = (x, y)
    x_arrays.append(x)

common_grid = make_common_grid(x_arrays)
processed = preprocess_spectra(spectra, common_grid, config.preprocessing)
gate_df, qc_stats, failures, group_summary = run_qc_pipeline(
    processed,
    common_grid,
    qc_config=config.qc,
)
```

Stable public API is exported from `sers.__all__` and is safe to import from `sers` directly.
Internal helpers are intentionally excluded from `__all__` and may change without notice.

## Configuration

Edit `config/config.yaml` to adjust:

```yaml
dataset:
  folder_to_group:
    colorectal+cancer_SERS: "CRC"
    normal_SERS: "NOR"
    # ...

preprocessing:
  smooth_window: 11
  use_snv: true

qc:
  corr_threshold: 0.95
  min_snr: 3.0
```

### Environment Variables

Override default paths:
```bash
export SERS_DATA_DIR=/path/to/data
export SERS_RESULTS_DIR=/path/to/results
export SERS_MODEL_DIR=/path/to/models
```

## Docker

```bash
# Build
docker build -t sers-analysis -f infra/Dockerfile .

# Run
docker compose -f infra/docker-compose.yml up --build
```

## Project Structure

```
SERS-AI/
├── main.py                           # Pipeline entry point
├── pyproject.toml                    # Package metadata
├── README.md
│
├── src/sers/                         # Core library
│   ├── config.py                     # Centralized path & config management
│   ├── preprocessing.py              # Signal processing
│   ├── io.py                         # File I/O
│   ├── visualization.py
│   ├── qc/                           # Quality control module
│   └── validation/                   # Protocol validation
│
├── models/                           # ML training code
│   ├── train.py                      # Main training script
│   ├── test.py                       # Evaluation
│   ├── model.py                      # Model definitions
│   ├── tune_torch.py                 # Hyperparameter tuning
│   └── results/                      # Model outputs (gitignored)
│       ├── 01_benchmarks/            # Formal model comparisons
│       ├── 02_tuning/                # Hyperparameter search
│       ├── 03_learning_curves/       # Data efficiency analysis
│       ├── 04_comparisons/           # Cross-experiment summaries
│       ├── 05_subset_analysis/       # Targeted experiments
│       └── _archive/                 # Superseded early experiments
│
├── config/                           # Configuration files
│   ├── config.yaml                   # Pipeline & dataset config
│   ├── environment.yml               # Conda environment
│   └── environment.lock.yml          # Locked dependencies
│
├── scripts/                          # Utility & analysis scripts
│   ├── run_qc_preprocess.py          # QC preprocessing runner
│   ├── explore_data.py               # Dataset exploration
│   ├── db_create.py                  # Database setup
│   └── ...
│
├── data/                             # Raw & clinical data (gitignored)
├── results/                          # QC & preprocessing output (gitignored)
├── logs/                             # Log files (gitignored)
├── notebooks/                        # Jupyter notebooks
│
├── docs/                             # Experiment documentation
│   ├── EXPERIMENT_CONTEXT.md         # Full experiment history
│   ├── MODEL_WORKFLOW.md             # Training workflow guide
│   └── CHANGELOG.md
│
├── dashboard/                        # Executive dashboards
│   ├── SERS_AI_Executive_Dashboard.html
│   ├── dashboard_data.json
│   └── img/                          # Dashboard images
│
├── infra/                            # Infrastructure
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── DOCKER_QUICKSTART.sh
│
└── tests/                            # Test suite (planned)
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run available test asset
python models/test.py

# Lint
ruff check src/

# Type check
mypy src/
```

Automated `tests/`-based pytest suite is not yet included in this repository (예정).

## License

MIT

### Team Rule: Import Smoke Check

To prevent accidental public API breakage, run the following smoke-check in CI and before release:

```bash
python -c "from sers import *"
```

This command must succeed without ImportError.
