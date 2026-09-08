# Moved

The SERS clinical webapp (login, DB, QC, PDF report, Windows `.exe`
packaging) that used to live here has moved to its own repository:

**[SERS-Clinical-App](https://github.com/SERS-AI-Platform/SERS-Clinical-App)**, under `app/`.

That repository is now the sole source for the clinical webapp. It vendors
(syncs, does not fork) `src/sers/` and the trained model artifacts from this
repo via `scripts/sync_vendor.sh` — see `vendor/VENDOR_MANIFEST.json` there
for the exact source commit each vendored copy was synced from.

`src/sers/`, `artifacts/`, and `scripts/analysis/calibration/` stay here —
this repo remains the single source of truth for the preprocessing library
and trained models. Moved 2026-08-31.
