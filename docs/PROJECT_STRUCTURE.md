# SERS Analysis Pipeline - Project Structure

```
sers-analysis/
├── pyproject.toml          # Modern Python packaging (replaces setup.py)
├── environment.yml         # Conda environment
├── config.yaml             # User configuration (git-ignored in production)
├── config.yaml.example     # Template for users
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
├── src/
│   └── sers/               # Main package
│       ├── __init__.py     # Package init + version
│       ├── config.py       # Configuration dataclasses + loader
│       ├── io.py           # File I/O (read_spectrum, parse_filename)
│       ├── preprocessing.py # Signal processing (smooth, baseline, snv)
│       ├── qc.py           # Quality control (correlation, medoid, filters)
│       ├── analysis.py     # High-level analysis pipeline
│       ├── modeling.py     # ML model training + evaluation
│       └── cli.py          # Command-line interface
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # Pytest fixtures
│   ├── test_preprocessing.py
│   ├── test_qc.py
│   └── test_config.py
│
├── notebooks/              # Exploratory analysis
│   └── exploration.ipynb
│
└── data/                   # Local data (git-ignored)
    ├── raw/
    ├── processed/
    └── results/
```

## Key Design Decisions

### 1. `src/` Layout
Using `src/sers/` prevents accidental imports from the working directory during development.
This is the recommended layout for packages that will be distributed.

### 2. Separation of Concerns
- **io.py**: All file reading/writing and filename parsing
- **preprocessing.py**: Pure signal processing functions (stateless)
- **qc.py**: Quality control logic
- **analysis.py**: Orchestrates the pipeline
- **config.py**: Configuration management only

### 3. Configuration Strategy
- **Immutable dataclasses** for type safety
- **Environment variables** for deployment paths (DATA_DIR, MODEL_DIR)
- **YAML** for user-tunable parameters
- **Hardcoded defaults** for QC thresholds that shouldn't change

### 4. Entry Points
The `cli.py` provides command-line access:
```bash
sers-analyze run --config config.yaml --input data/raw --output results/
sers-analyze validate --input data/raw
```
