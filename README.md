# SERS Analysis Pipeline

SERS-based spectral analysis pipeline for multi-cancer screening from urine metabolites.

> 👋 **신규 팀원이신가요? [`ONBOARDING.md`](ONBOARDING.md) 부터 읽으세요.** — 개발 규칙·도메인 규칙 진입점

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
# Inspect available workflows
sers --help

# Validate input files
sers data validate -i data/raw

# Run preprocessing + QC pipeline
sers preprocess --config config/config.yaml --output-dir results/

# Run active uSERS-Net/STK-V2 training smoke path
sers train stacking --dry-run
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
├── main.py                           # Legacy/top-level pipeline entry point
├── pyproject.toml                    # Package metadata
├── README.md
│
├── src/sers/                         # Reusable core library ("engine")
│   ├── config.py                     # Centralized path & config management
│   ├── io.py                         # File I/O
│   ├── preprocessing.py              # Preprocessing pipeline and utilities
│   ├── signal.py                     # Low-level spectral signal helpers
│   ├── qc/                           # Quality control module
│   ├── models/usersnet/              # Stable uSERS-Net/STK-V2 helper import path
│   ├── visualization/                # Plotting and interpretation helpers
│   └── validation/                   # Data/protocol validation
│
├── scripts/                          # Runnable workflows ("buttons")
│   ├── pipeline/                     # Preprocessing, QC, clinical standardization
│   │   ├── run_qc_preprocess.py      # Main QC + preprocessing runner
│   │   └── ...
│   ├── training/                     # Active model training entry points
│   │   ├── train_usersnet.py         # Active uSERS-Net/STK-V2 training
│   │   ├── build_usersnet_production.py
│   │   └── _legacy/                  # Legacy training entry points kept for reference
│   ├── evaluation/                   # Held-out and post-training evaluation
│   ├── analysis/                     # Reproducible analysis/figure support
│   ├── qc_validation/                # QC threshold and validation studies
│   ├── deployment/                   # Inference API, clinical web app, packaging
│   ├── db/                           # Database setup/upload utilities
│   └── visualization/                # R/figure export workflows
│
├── models/                           # Compatibility layer and shared model helpers
│   ├── stacking_utils.py             # STK-V2 helper implementation still used by active code
│   ├── model.py                      # Legacy ResNet compatibility shim
│   ├── build_production_stacking.py  # Alternate/legacy production artifact builder
│   └── legacy/                       # Archived model architectures and old scripts
│
├── artifacts/                        # Production model artifacts (joblib, manifest, grids)
├── figures/                          # Project-level generated figures
├── publications/                     # Publication/poster figure generation
├── metabolite_profiling/             # Metabolite profiling subproject
│
├── config/                           # Configuration files
│   ├── config.yaml                   # Pipeline & dataset config
│   ├── environment.yml               # Conda environment
│   └── environment.lock.yml          # Locked dependencies
│
├── data/                             # Raw & clinical data (gitignored)
├── results/                          # Preprocessing, QC, training, and analysis outputs
├── logs/                             # Log files (gitignored)
├── notebooks/                        # Jupyter notebooks
│
├── docs/                             # Experiment documentation
│   ├── EXPERIMENT_CONTEXT.md         # Full experiment history
│   ├── MODEL_WORKFLOW.md             # Training workflow guide
│   └── CHANGELOG.md
│
├── infra/                            # Infrastructure
│   ├── Dockerfile
│   ├── Dockerfile.api
│   ├── docker-compose.yml
│   └── DOCKER_QUICKSTART.sh
│
└── tests/                            # Pytest suite
```

### Directory Roles

- `src/sers/` contains reusable library code. If multiple scripts need the same logic, move it here.
- `scripts/` contains executable workflows. A script should orchestrate library calls and write outputs, not become the main home for reusable logic.
- `scripts/training/` is the active place for model training and production-model build entry points.
- `models/` is no longer the primary training entry-point directory. It currently holds compatibility shims, shared STK-V2 helpers, and archived model code.
- `artifacts/` stores model artifacts used by inference; `results/` stores reproducible run outputs and analysis products.

Rule of thumb: `scripts/training` is the button, `src/sers` is the engine, and `models` is a transition/compatibility area unless a file explicitly documents otherwise.

## Quality / DGMP Governance

This repository keeps the working technical evidence for DGMP/QMS readiness. Controlled SOP approval copies should live in SharePoint `/02_Regulatory_QMS`; Git tracks the current engineering matrix and evidence links.

| Pillar | Repo evidence | Current status |
|---|---|---|
| Document control | [`docs/compliance/qms/SOP_INDEX.md`](docs/compliance/qms/SOP_INDEX.md), [`docs/compliance/DEVELOPMENT_ASSET_GOVERNANCE.md`](docs/compliance/DEVELOPMENT_ASSET_GOVERNANCE.md) | draft index |
| SDLC / change control | PR workflow, CI, [`CONTRIBUTING.md`](CONTRIBUTING.md) | active internal gate |
| V&V / coverage evidence | [`docs/compliance/COVERAGE_PROCESS_MATRIX.md`](docs/compliance/COVERAGE_PROCESS_MATRIX.md), tests, CI | package baseline gated |
| Data governance | `.gitignore`, [`docs/compliance/DEVELOPMENT_ASSET_GOVERNANCE.md`](docs/compliance/DEVELOPMENT_ASSET_GOVERNANCE.md), clinical-use docs | partial, controlled storage needed |
| Software product / IFU traceability | [`docs/sharepoint/sw_product/`](docs/sharepoint/sw_product/), [`docs/clinical/`](docs/clinical/) | draft working set |
| Security / access / audit | `scripts/deployment/clinical_auth.py`, `clinical_audit.py`, `clinical_db.py` | implementation present, SOP gate open |
| Release / configuration management | `pyproject.toml`, `config/`, `infra/`, `artifacts/` | partial, release SOP open |
| Risk / CAPA / training | [`docs/compliance/qms/DGMP_PILLAR_MATRIX.md`](docs/compliance/qms/DGMP_PILLAR_MATRIX.md) | open gate |

Detailed pillar ownership and SOP status are tracked in [`docs/compliance/qms/DGMP_PILLAR_MATRIX.md`](docs/compliance/qms/DGMP_PILLAR_MATRIX.md). This matrix is a readiness tracker, not a claim of regulatory approval.

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run active uSERS-Net training smoke path
python scripts/training/train_usersnet.py --dry-run

# Run tests
pytest --cov=sers --cov-report=term-missing --cov-report=xml --cov-fail-under=35
python scripts/quality/coverage_by_process.py coverage.xml

# Lint
ruff check src/ tests/ scripts/quality/

# Type check the current CI-gated surface
mypy
```

## License

Proprietary and confidential. See [`LICENSE`](LICENSE).

### Team Rule: Core Import Smoke Check

To prevent accidental core API breakage, run the following smoke-check in CI and before release:

```bash
python - <<'PY'
import sers
from sers import (
    find_spectra,
    load_config,
    make_common_grid,
    parse_filename,
    preprocess_spectra,
    read_spectrum,
    run_qc_pipeline,
)

print("core import smoke ok")
PY
```

This command must succeed without ImportError. Visualization helpers remain optional and require installing `.[viz]`.
