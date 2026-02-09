# SERS Analysis Pipeline

SERS-based spectral analysis pipeline for multi-cancer screening from urine metabolites.

## Installation

### Using Conda (Recommended)
```bash
# Create environment
conda env create -f environment.yml
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

# Run full pipeline
sers run -i data/raw -o results/ -c config.yaml

# Quiet mode
sers run -i data/raw -o results/ -q
```

### Python API
```python
from sers import load_config
from sers.analysis import run_pipeline
from pathlib import Path

# Load custom config
config = load_config("config.yaml")

# Run pipeline
result = run_pipeline(
    data_dir=Path("data/raw"),
    config=config,
)

# Get sklearn-compatible format
X, y, sample_ids = result.to_matrix()

# Access QC report
print(result.qc_report.head())
```

## Configuration

Copy `config.yaml.example` to `config.yaml` and adjust:

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
docker build -t sers-analysis .

# Run
docker run \
  -v $(pwd)/data:/data \
  -v $(pwd)/results:/results \
  sers-analysis run
```

## Project Structure

```
sers-analysis/
├── src/sers/           # Main package
│   ├── config.py       # Configuration management
│   ├── io.py           # File I/O
│   ├── preprocessing.py # Signal processing
│   ├── qc.py           # Quality control
│   ├── analysis.py     # Pipeline orchestration
│   └── cli.py          # Command-line interface
├── tests/              # Unit tests
├── config.yaml         # User configuration
├── pyproject.toml      # Package metadata
└── Dockerfile          # Container deployment
```

## Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=sers

# Lint
ruff check src/

# Type check
mypy src/
```

## License

MIT
