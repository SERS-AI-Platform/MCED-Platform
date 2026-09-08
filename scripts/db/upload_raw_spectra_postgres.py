#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy>=1.24", "psycopg2-binary>=2.9"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run upload_raw_spectra_postgres.py --dry-run
# 3. Or make executable and run:
#      chmod +x upload_raw_spectra_postgres.py && ./upload_raw_spectra_postgres.py
# ──────────────────

"""Load original raw CSV spectra into PostgreSQL without preprocessing."""

from __future__ import annotations

import argparse
import hashlib
import os
import re
from collections import Counter
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

FOLDER_GROUPS = {
    "1. Prostate cancer (100개)": "PRO",
    "2. Breast cancer (30개)": "BRE",
    "3. Ovarian cancer (70개)": "OVA",
    "4. Lung cancer (300개)": "LUN",
    "5. Normal (100개)": "NOR",
    "6. Diabetes (100개)": "DIA",
    "7. High blood pressure (100개)": "HBP",
    "8. High blood pressure + Diabetes (100개)": "H.D.",
    "9. Colorectal cancer (300개)": "CRC",
    "10-1. C-Pancreatic cancer (70개)": "CPAN",
    "10-2. S-Pancreatic cancer (72개)": "SPAN",
    "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
    "11 BLC (299개)": "BLC",
    "12. Y-Normal (YNOR)": "YNOR",
}

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS public.raw_spectra (
    id BIGSERIAL PRIMARY KEY,
    dataset_version TEXT NOT NULL,
    source_root TEXT NOT NULL,
    source_path TEXT NOT NULL,
    source_batch TEXT NOT NULL,
    source_kind TEXT NOT NULL CHECK (source_kind IN ('replicate', 'average', 'calibration_control')),
    control_type TEXT,
    group_code VARCHAR(32),
    sample_id VARCHAR(128) NOT NULL,
    subject_key TEXT NOT NULL,
    replicate INTEGER,
    is_averaged BOOLEAN NOT NULL,
    wavenumber DOUBLE PRECISION[] NOT NULL,
    intensities DOUBLE PRECISION[] NOT NULL,
    n_points INTEGER NOT NULL CHECK (n_points = cardinality(wavenumber)),
    x_min DOUBLE PRECISION NOT NULL,
    x_max DOUBLE PRECISION NOT NULL,
    source_sha256 TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (dataset_version, source_path)
);
CREATE INDEX IF NOT EXISTS idx_raw_spectra_dataset ON public.raw_spectra (dataset_version);
CREATE INDEX IF NOT EXISTS idx_raw_spectra_subject ON public.raw_spectra (dataset_version, subject_key);
CREATE INDEX IF NOT EXISTS idx_raw_spectra_kind ON public.raw_spectra (dataset_version, source_kind);
"""


class RawInputError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SpectrumIdentity:
    path: Path
    relative_path: str
    source_batch: str
    source_kind: str
    control_type: str | None
    group_code: str | None
    sample_id: str
    replicate: int | None
    is_averaged: bool


@dataclass(frozen=True, slots=True)
class IngestOptions:
    input_root: Path
    dataset_version: str
    include_averages: bool
    batch_size: int
    host: str | None
    port: int | None
    dbname: str | None
    user: str | None
    dry_run: bool


@dataclass(frozen=True, slots=True)
class ScanSummary:
    identities: tuple[SpectrumIdentity, ...]
    points: int
    point_counts: tuple[tuple[int, int], ...]
    groups: tuple[tuple[str, int], ...]
    manifest_sha256: str


def parse_identity(path: Path, root: Path) -> SpectrumIdentity:
    relative = path.relative_to(root)
    source_batch = relative.parts[0]
    stem = path.stem
    is_averaged = stem.casefold().endswith("_ave")
    calibration = any("calibration_control" in part.casefold() for part in relative.parts)
    if calibration:
        match = re.fullmatch(r"(PS|Sensor|Si[ _]Wafer)\s+(\d+)_(\d+|ave)", stem, re.IGNORECASE)
        if match is None:
            raise RawInputError(f"Cannot parse calibration file: {relative}")
        control_type = re.sub(r"[ _]+", "_", match.group(1).upper())
        sample_id = match.group(2)
        replicate = None if match.group(3).casefold() == "ave" else int(match.group(3))
        return SpectrumIdentity(
            path, relative.as_posix(), source_batch, "calibration_control", control_type,
            None, sample_id, replicate, replicate is None,
        )
    pattern = r"^(?P<group>[A-Za-z]+(?:\.[A-Za-z]+)*\.?)\s*(?P<sample>\d+)_"
    suffix = r"(?:ave|(?P<replicate>\d+))$"
    match = re.fullmatch(pattern + suffix, stem, re.IGNORECASE)
    if match is None:
        numeric = re.fullmatch(r"(?P<sample>\d+)_" + (r"ave" if is_averaged else r"(?P<replicate>\d+)") + r"$", stem, re.IGNORECASE)
        if numeric is None:
            raise RawInputError(f"Cannot parse spectrum file: {relative}")
        group_code = FOLDER_GROUPS.get(source_batch)
        sample_id = numeric.group("sample")
        replicate = None if is_averaged else int(numeric.group("replicate"))
    else:
        group_code = match.group("group").strip().upper()
        sample_id = match.group("sample")
        replicate = None if is_averaged else int(match.group("replicate"))
    if not group_code:
        raise RawInputError(f"No group mapping for spectrum file: {relative}")
    source_kind = "average" if is_averaged else "replicate"
    return SpectrumIdentity(
        path, relative.as_posix(), source_batch, source_kind, None, group_code,
        sample_id, replicate, is_averaged,
    )


def discover_identities(options: IngestOptions) -> tuple[SpectrumIdentity, ...]:
    if not options.input_root.is_dir():
        raise FileNotFoundError(options.input_root)
    paths = sorted({*options.input_root.rglob("*.CSV"), *options.input_root.rglob("*.csv")})
    identities = tuple(
        parse_identity(path, options.input_root)
        for path in paths
        if options.include_averages or not path.stem.casefold().endswith("_ave")
    )
    if not identities:
        raise RawInputError(f"No CSV spectra found under {options.input_root}")
    return identities


def read_raw_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        values = np.loadtxt(path, delimiter=",", usecols=(0, 1), dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as exc:
        raise RawInputError(f"Cannot read {path}: {exc}") from exc
    if values.shape[1] != 2 or values.shape[0] < 2:
        raise RawInputError(f"Spectrum must have at least two numeric rows: {path}")
    x, y = values[:, 0], values[:, 1]
    if not np.isfinite(values).all() or not np.all(np.diff(x) > 0):
        raise RawInputError(f"Non-finite or non-increasing raw spectrum: {path}")
    return x, y


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scan_source(identities: tuple[SpectrumIdentity, ...]) -> ScanSummary:
    point_counts: Counter[int] = Counter()
    groups: Counter[str] = Counter()
    manifest = hashlib.sha256()
    points = 0
    for identity in identities:
        x, _ = read_raw_spectrum(identity.path)
        point_counts[len(x)] += 1
        groups[identity.group_code or identity.control_type or "UNSET"] += 1
        points += len(x)
        manifest.update(identity.relative_path.encode())
        manifest.update(sha256_file(identity.path).encode())
    return ScanSummary(tuple(identities), points, tuple(sorted(point_counts.items())), tuple(sorted(groups.items())), manifest.hexdigest())


def connection_kwargs(options: IngestOptions) -> dict[str, str | int]:
    password = os.environ.get("PGPASSWORD") or getpass("PostgreSQL password (not saved): ")
    if not password:
        raise RuntimeError("A PostgreSQL password is required for database writes")
    return {
        "host": options.host or os.environ.get("PGHOST", "localhost"),
        "port": options.port or int(os.environ.get("PGPORT", "5432")),
        "dbname": options.dbname or os.environ.get("PGDATABASE", "sers_clinical"),
        "user": options.user or os.environ.get("PGUSER", "postgres"),
        "password": password,
    }


def print_summary(summary: ScanSummary, options: IngestOptions) -> None:
    print({
        "input_root": str(options.input_root.resolve()),
        "dataset_version": options.dataset_version,
        "files": len(summary.identities),
        "points": summary.points,
        "point_counts": dict(summary.point_counts),
        "groups_or_controls": dict(summary.groups),
        "manifest_sha256": summary.manifest_sha256,
        "averages_included": options.include_averages,
    })


def write_database(summary: ScanSummary, options: IngestOptions) -> None:
    insert_sql = """INSERT INTO public.raw_spectra
        (dataset_version, source_root, source_path, source_batch, source_kind,
         control_type, group_code, sample_id, subject_key, replicate, is_averaged,
         wavenumber, intensities, n_points, x_min, x_max, source_sha256)
        VALUES %s
        ON CONFLICT (dataset_version, source_path) DO UPDATE SET
            source_sha256 = EXCLUDED.source_sha256, wavenumber = EXCLUDED.wavenumber,
            intensities = EXCLUDED.intensities, n_points = EXCLUDED.n_points,
            x_min = EXCLUDED.x_min, x_max = EXCLUDED.x_max, ingested_at = CURRENT_TIMESTAMP"""
    with psycopg2.connect(**connection_kwargs(options)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
            rows: list[tuple[str, ... | None]] = []
            for identity in summary.identities:
                x, y = read_raw_spectrum(identity.path)
                subject = identity.group_code or identity.control_type or "UNSET"
                rows.append((
                    options.dataset_version, str(options.input_root.resolve()), identity.relative_path,
                    identity.source_batch, identity.source_kind, identity.control_type, identity.group_code,
                    identity.sample_id, f"{identity.source_batch}::{subject}::{identity.sample_id}",
                    identity.replicate, identity.is_averaged, x.tolist(), y.tolist(), len(x),
                    float(x[0]), float(x[-1]), sha256_file(identity.path),
                ))
                if len(rows) >= options.batch_size:
                    execute_values(cursor, insert_sql, rows, page_size=options.batch_size)
                    rows = []
            if rows:
                execute_values(cursor, insert_sql, rows, page_size=options.batch_size)
            cursor.execute(
                "SELECT COUNT(*), COUNT(*) FILTER (WHERE source_kind = 'replicate'), "
                "COUNT(*) FILTER (WHERE source_kind = 'calibration_control'), "
                "COUNT(*) FILTER (WHERE source_kind = 'average') "
                "FROM public.raw_spectra WHERE dataset_version = %s",
                (options.dataset_version,),
            )
            count, replicates, controls, averages = cursor.fetchone()
    print({"db_rows": count, "replicates": replicates, "calibration_controls": controls, "averages": averages})
    if count != len(summary.identities):
        raise RuntimeError(f"Database row count mismatch: expected {len(summary.identities)}, got {count}")


def parse_args() -> IngestOptions:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("data/raw_data"))
    parser.add_argument("--dataset-version", default="raw_data_csv_v1")
    parser.add_argument("--include-averages", action="store_true")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return IngestOptions(args.input_root, args.dataset_version, args.include_averages, args.batch_size, args.host, args.port, args.dbname, args.user, args.dry_run)


def main() -> None:
    options = parse_args()
    summary = scan_source(discover_identities(options))
    print_summary(summary, options)
    if not options.dry_run:
        write_database(summary, options)


if __name__ == "__main__":
    main()
