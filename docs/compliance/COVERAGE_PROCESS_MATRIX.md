# Coverage Process Matrix

This project tracks coverage in two layers:

1. **Merge gate**: pytest coverage for the importable `sers` package must stay above the CI floor.
2. **Process summary**: the generated `coverage.xml` is grouped into MCED-Platform process buckets so status reviews can see which part of the workflow is actually test-backed.

Run locally:

```bash
pytest --cov=sers --cov-report=term-missing --cov-report=xml --cov-fail-under=35
python scripts/quality/coverage_by_process.py coverage.xml
```

## Process Buckets

| Process | Coverage source | Interpretation |
|---|---|---|
| CLI workflow surface | `src/sers/cli/` | User-facing workflow command coverage |
| Config + public package API | `src/sers/__init__.py`, `src/sers/config.py`, `src/sers/logging_config.py` | Stable import/config contract |
| Data I/O + validation | `src/sers/io.py`, `src/sers/validation.py` | Data onboarding and protocol checks |
| Preprocessing + signal processing | `src/sers/preprocessing.py`, `src/sers/signal.py` | Core spectral preprocessing path |
| Quality control | `src/sers/qc/` | Clinical suitability and replicate QC logic |
| SSI/CTI risk scoring | `src/sers/scoring.py` | User-facing score conversion logic |
| Model helper imports | `src/sers/models/` | Model import path and helper scaffolding |
| Analysis + visualization | `src/sers/analysis.py`, `src/sers/visualization*` | Reporting/interpretation surface, currently not a release gate |
| Calibration transfer | `src/sers/calibration_transfer.py` | Instrument-transfer validation surface |
| Legacy staging/ingest utilities | `src/sers/ingest_ypan.py`, `src/sers/reingest_staging.py`, `src/sers/update_lun_dates.py` | Historical reproduction only; forbidden as governed lineage input |

## Current Scope Boundary

The numeric CI coverage gate is intentionally scoped to `--cov=sers`.

Software product code under `scripts/deployment/**` exists in this repository, but it is not yet part of the numeric coverage gate. It should get a separate deployment/software test lane before being described as externally ready.

The `solum-dashboard` workspace is documented as a separate/local dashboard asset, not as code tracked in this repository.

## PPT Wording

Use:

> Coverage baseline is gated at package level, with process-level coverage visibility. Core preprocessing/QC/scoring are test-backed; deployment software and dashboard assets remain separate open gates.

Avoid:

> Coverage is 100%.

That would imply a fully gated release surface across library, CLI, deployment software, dashboard assets, and generated reports, which is not the current repo state.
