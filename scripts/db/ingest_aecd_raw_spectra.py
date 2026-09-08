#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy>=1.24",
#     "pydantic>=2.0",
#     "psycopg2-binary>=2.9",
#     "typer>=0.12",
#     "typing-extensions>=4.12",
# ]
# ///

# ─── How to run ───
# File-only validation:
#   uv run scripts/db/ingest_aecd_raw_spectra.py
# Validate DB measurement mapping without writes (PG* environment variables):
#   uv run scripts/db/ingest_aecd_raw_spectra.py --mode check-db
# Insert after both validations pass:
#   uv run scripts/db/ingest_aecd_raw_spectra.py --mode write
# ──────────────────

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root))
    _ = runpy.run_module("scripts.db.aecd_raw_spectra_cli", run_name="__main__")


if __name__ == "__main__":
    main()
