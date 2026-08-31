from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path

try:
    from scripts.db.aecd_clinical_sources import (
        ClinicalScan,
        numeric_value,
        scan_clinical,
        text_value,
    )
    from scripts.db.aecd_measurement_scan import discover_source_specs, scan_sources
    from scripts.db.aecd_measurement_sources import (
        MeasurementRecord,
        MeasurementScan,
    )
    from scripts.db.aecd_spectrum_formats import read_spectrum
except ModuleNotFoundError:
    from aecd_clinical_sources import ClinicalScan, numeric_value, scan_clinical, text_value
    from aecd_measurement_scan import discover_source_specs, scan_sources
    from aecd_measurement_sources import (
        MeasurementRecord,
        MeasurementScan,
    )
    from aecd_spectrum_formats import read_spectrum


SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS public.aecd_ingest_batches (dataset_version TEXT PRIMARY KEY, repo_root TEXT NOT NULL, measurement_files INTEGER NOT NULL, clinical_records INTEGER NOT NULL, rejected_files INTEGER NOT NULL, manifest_sha256 TEXT NOT NULL, ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP);\n"
    "CREATE TABLE IF NOT EXISTS public.aecd_measurements (id BIGSERIAL PRIMARY KEY, dataset_version TEXT NOT NULL, source_domain TEXT NOT NULL, source_kind TEXT NOT NULL, instrument_key TEXT NOT NULL, preparation TEXT NOT NULL, source_root TEXT NOT NULL, source_batch TEXT NOT NULL, source_path TEXT NOT NULL, source_sha256 TEXT NOT NULL, file_format TEXT NOT NULL, artifact_role TEXT NOT NULL, variant TEXT NOT NULL, is_averaged BOOLEAN NOT NULL, measurement_key TEXT NOT NULL, group_code TEXT, sample_id TEXT NOT NULL, replicate INTEGER, control_type TEXT, acquisition_date DATE, reagent_phase TEXT NOT NULL, reagent_name TEXT, reagent_evidence TEXT NOT NULL, wavenumber DOUBLE PRECISION[] NOT NULL, intensities DOUBLE PRECISION[] NOT NULL, n_points INTEGER NOT NULL CHECK (n_points > 0 AND n_points = cardinality(wavenumber) AND n_points = cardinality(intensities)), x_min DOUBLE PRECISION NOT NULL, x_max DOUBLE PRECISION NOT NULL, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE (dataset_version, source_path));\n"
    "ALTER TABLE public.aecd_measurements ADD COLUMN IF NOT EXISTS reagent_phase TEXT NOT NULL DEFAULT 'not_recorded';\n"
    "ALTER TABLE public.aecd_measurements ADD COLUMN IF NOT EXISTS reagent_name TEXT;\n"
    "ALTER TABLE public.aecd_measurements ADD COLUMN IF NOT EXISTS reagent_evidence TEXT NOT NULL DEFAULT 'reagent phase was not recorded at ingestion';\n"
    "CREATE INDEX IF NOT EXISTS idx_aecd_measurements_domain ON public.aecd_measurements (dataset_version, source_domain, source_kind);\n"
    "CREATE INDEX IF NOT EXISTS idx_aecd_measurements_role ON public.aecd_measurements (dataset_version, artifact_role, is_averaged);\n"
    "CREATE INDEX IF NOT EXISTS idx_aecd_measurements_key ON public.aecd_measurements (dataset_version, measurement_key);\n"
    "CREATE INDEX IF NOT EXISTS idx_aecd_measurements_sample ON public.aecd_measurements (dataset_version, group_code, sample_id);\n"
    "CREATE TABLE IF NOT EXISTS public.aecd_ingest_rejections (id BIGSERIAL PRIMARY KEY, dataset_version TEXT NOT NULL, source_domain TEXT NOT NULL, source_path TEXT NOT NULL, reason_code TEXT NOT NULL, reason TEXT NOT NULL, source_sha256 TEXT, rejected_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE (dataset_version, source_path));\n"
    "CREATE TABLE IF NOT EXISTS public.aecd_clinical_records (id BIGSERIAL PRIMARY KEY, dataset_version TEXT NOT NULL, patient_id TEXT NOT NULL, disease_group TEXT, source_file TEXT NOT NULL, source_row_number INTEGER NOT NULL, source_sha256 TEXT NOT NULL, age DOUBLE PRECISION, sex TEXT, height_cm DOUBLE PRECISION, weight_kg DOUBLE PRECISION, bmi DOUBLE PRECISION, diagnosis TEXT, diagnosis_date TEXT, sample_date TEXT, pathology TEXT, stage TEXT, tnm TEXT, sample_timing TEXT, clinical_json JSONB NOT NULL, ingested_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, UNIQUE (dataset_version, patient_id));\n"
    "CREATE INDEX IF NOT EXISTS idx_aecd_clinical_group ON public.aecd_clinical_records (dataset_version, disease_group);\n"
    "CREATE INDEX IF NOT EXISTS idx_aecd_clinical_sample_date ON public.aecd_clinical_records (dataset_version, sample_date);"
)


@dataclass(frozen=True, slots=True)
class IngestOptions:
    repo_root: Path
    dataset_version: str
    batch_size: int
    dry_run: bool
    exclude_averages: bool
    host: str | None
    port: int | None
    dbname: str | None
    user: str | None


def _connection_kwargs(options: IngestOptions) -> dict[str, str | int]:
    password = os.environ.get("PGPASSWORD")
    if not password:
        if not sys.stdin.isatty():
            raise RuntimeError("Set PGPASSWORD for a non-interactive PostgreSQL write")
        password = getpass("PostgreSQL password (not saved): ")
    if not password:
        raise RuntimeError("A PostgreSQL password is required for database writes")
    return {
        "host": options.host or os.environ.get("PGHOST", "localhost"),
        "port": options.port or int(os.environ.get("PGPORT", "5432")),
        "dbname": options.dbname or os.environ.get("PGDATABASE", "sers_clinical"),
        "user": options.user or os.environ.get("PGUSER", "postgres"),
        "password": password,
    }


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def measurement_source_path(record: MeasurementRecord) -> str:
    return f"{record.source_domain}/{record.source_kind}/{record.relative_path}"


def _measurement_row(options: IngestOptions, record: MeasurementRecord) -> tuple[object, ...]:
    from psycopg2.extras import Json

    data = read_spectrum(record.path)
    if record.source_sha256 != _sha256_file(record.path):
        raise RuntimeError(f"Source changed after scan: {record.path}")
    metadata = {"file_format": data.file_format, **data.metadata}
    source_path = measurement_source_path(record)
    return (
        options.dataset_version,
        record.source_domain,
        record.source_kind,
        record.instrument_key,
        record.preparation,
        str(record.source_root.resolve()),
        record.source_batch,
        source_path,
        record.source_sha256,
        data.file_format,
        record.artifact_role,
        record.variant,
        record.is_averaged,
        record.measurement_key,
        record.group_code,
        record.sample_id,
        record.replicate,
        record.control_type,
        record.acquisition_date,
        record.reagent_phase,
        record.reagent_name,
        record.reagent_evidence,
        data.wavenumber.tolist(),
        data.intensities.tolist(),
        len(data.wavenumber),
        float(data.wavenumber[0]),
        float(data.wavenumber[-1]),
        Json(metadata, dumps=lambda value: json.dumps(value, ensure_ascii=False)),
    )


def _write_measurements(cursor: object, options: IngestOptions, scan: MeasurementScan) -> None:
    from psycopg2.extras import execute_values

    sql = (
        "INSERT INTO public.aecd_measurements (dataset_version, source_domain, source_kind, instrument_key, preparation, source_root, source_batch, source_path, source_sha256, file_format, artifact_role, variant, is_averaged, measurement_key, group_code, sample_id, replicate, control_type, acquisition_date, reagent_phase, reagent_name, reagent_evidence, wavenumber, intensities, n_points, x_min, x_max, metadata) VALUES %s "
        "ON CONFLICT (dataset_version, source_path) DO UPDATE SET source_sha256 = EXCLUDED.source_sha256, source_kind = EXCLUDED.source_kind, instrument_key = EXCLUDED.instrument_key, preparation = EXCLUDED.preparation, artifact_role = EXCLUDED.artifact_role, variant = EXCLUDED.variant, is_averaged = EXCLUDED.is_averaged, measurement_key = EXCLUDED.measurement_key, group_code = EXCLUDED.group_code, sample_id = EXCLUDED.sample_id, replicate = EXCLUDED.replicate, control_type = EXCLUDED.control_type, acquisition_date = EXCLUDED.acquisition_date, reagent_phase = EXCLUDED.reagent_phase, reagent_name = EXCLUDED.reagent_name, reagent_evidence = EXCLUDED.reagent_evidence, wavenumber = EXCLUDED.wavenumber, intensities = EXCLUDED.intensities, n_points = EXCLUDED.n_points, x_min = EXCLUDED.x_min, x_max = EXCLUDED.x_max, metadata = EXCLUDED.metadata, ingested_at = CURRENT_TIMESTAMP"
    )
    rows: list[tuple[object, ...]] = []
    for record in scan.records:
        rows.append(_measurement_row(options, record))
        if len(rows) >= options.batch_size:
            execute_values(cursor, sql, rows, page_size=options.batch_size)
            rows = []
    if rows:
        execute_values(cursor, sql, rows, page_size=options.batch_size)


def _write_rejections(cursor: object, options: IngestOptions, scan: MeasurementScan) -> None:
    from psycopg2.extras import execute_values

    if not scan.rejected:
        return
    sql = (
        "INSERT INTO public.aecd_ingest_rejections (dataset_version, source_domain, source_path, reason_code, reason, source_sha256) VALUES %s "
        "ON CONFLICT (dataset_version, source_path) DO UPDATE SET reason_code = EXCLUDED.reason_code, reason = EXCLUDED.reason, source_sha256 = EXCLUDED.source_sha256, rejected_at = CURRENT_TIMESTAMP"
    )
    rows = [
        (options.dataset_version, item.source_domain, str(item.path.resolve()), item.reason_code, item.reason, item.source_sha256)
        for item in scan.rejected
    ]
    execute_values(cursor, sql, rows, page_size=options.batch_size)


def _write_clinical(cursor: object, options: IngestOptions, scan: ClinicalScan) -> None:
    from psycopg2.extras import Json, execute_values

    sql = (
        "INSERT INTO public.aecd_clinical_records (dataset_version, patient_id, disease_group, source_file, source_row_number, source_sha256, age, sex, height_cm, weight_kg, bmi, diagnosis, diagnosis_date, sample_date, pathology, stage, tnm, sample_timing, clinical_json) VALUES %s "
        "ON CONFLICT (dataset_version, patient_id) DO UPDATE SET disease_group = EXCLUDED.disease_group, source_file = EXCLUDED.source_file, source_row_number = EXCLUDED.source_row_number, source_sha256 = EXCLUDED.source_sha256, age = EXCLUDED.age, sex = EXCLUDED.sex, height_cm = EXCLUDED.height_cm, weight_kg = EXCLUDED.weight_kg, bmi = EXCLUDED.bmi, diagnosis = EXCLUDED.diagnosis, diagnosis_date = EXCLUDED.diagnosis_date, sample_date = EXCLUDED.sample_date, pathology = EXCLUDED.pathology, stage = EXCLUDED.stage, tnm = EXCLUDED.tnm, sample_timing = EXCLUDED.sample_timing, clinical_json = EXCLUDED.clinical_json, ingested_at = CURRENT_TIMESTAMP"
    )
    rows: list[tuple[object, ...]] = []
    for record in scan.records:
        rows.append((
            options.dataset_version,
            record.patient_id,
            text_value(record, "disease_group"),
            record.source_file,
            record.source_row_number,
            record.source_sha256,
            numeric_value(record, "age"),
            text_value(record, "sex"),
            numeric_value(record, "height_cm"),
            numeric_value(record, "weight_kg"),
            numeric_value(record, "bmi"),
            text_value(record, "diagnosis"),
            text_value(record, "diagnosis_date"),
            text_value(record, "sample_date"),
            text_value(record, "pathology"),
            text_value(record, "stage"),
            text_value(record, "tnm"),
            text_value(record, "sample_timing"),
            Json(record.values, dumps=lambda value: json.dumps(value, ensure_ascii=False)),
        ))
        if len(rows) >= options.batch_size:
            execute_values(cursor, sql, rows, page_size=options.batch_size)
            rows = []
    if rows:
        execute_values(cursor, sql, rows, page_size=options.batch_size)


def _print_summary(options: IngestOptions, measurement: MeasurementScan, clinical: ClinicalScan) -> None:
    print({
        "dataset_version": options.dataset_version,
        "repo_root": str(options.repo_root.resolve()),
        "measurement_total_files": measurement.total_files,
        "measurement_valid_files": len(measurement.records),
        "measurement_rejected_files": len(measurement.rejected),
        "measurement_points": measurement.points,
        "measurement_roles": dict(measurement.role_counts),
        "measurement_sources": dict(measurement.source_counts),
        "measurement_formats": dict(measurement.format_counts),
        "clinical_records": len(clinical.records),
        "clinical_source_files": list(clinical.source_files),
        "manifest_sha256": measurement.manifest_sha256,
        "averages_included": any(record.is_averaged for record in measurement.records),
    })
    for item in measurement.rejected:
        print({"rejected": str(item.path), "reason_code": item.reason_code, "reason": item.reason})


def write_database(options: IngestOptions, measurement: MeasurementScan, clinical: ClinicalScan) -> None:
    import psycopg2

    with psycopg2.connect(**_connection_kwargs(options)) as connection:
        with connection.cursor() as cursor:
            cursor.execute(SCHEMA_SQL)
            _write_measurements(cursor, options, measurement)
            _write_rejections(cursor, options, measurement)
            _write_clinical(cursor, options, clinical)
            cursor.execute(
                "INSERT INTO public.aecd_ingest_batches (dataset_version, repo_root, measurement_files, clinical_records, rejected_files, manifest_sha256) VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (dataset_version) DO UPDATE SET repo_root = EXCLUDED.repo_root, measurement_files = EXCLUDED.measurement_files, clinical_records = EXCLUDED.clinical_records, rejected_files = EXCLUDED.rejected_files, manifest_sha256 = EXCLUDED.manifest_sha256, ingested_at = CURRENT_TIMESTAMP",
                (options.dataset_version, str(options.repo_root.resolve()), len(measurement.records), len(clinical.records), len(measurement.rejected), measurement.manifest_sha256),
            )
            cursor.execute('SELECT COUNT(*), COUNT(*) FILTER (WHERE is_averaged), COUNT(DISTINCT measurement_key) FROM public.aecd_measurements WHERE dataset_version = %s', (options.dataset_version,))
            measurement_counts = cursor.fetchone()
            cursor.execute('SELECT COUNT(*) FROM public.aecd_clinical_records WHERE dataset_version = %s', (options.dataset_version,))
            clinical_count = cursor.fetchone()
    if measurement_counts is None or clinical_count is None:
        raise RuntimeError("Database validation query returned no row")
    print({"database_measurements": measurement_counts[0], "database_averages": measurement_counts[1], "database_measurement_keys": measurement_counts[2], "database_clinical_records": clinical_count[0]})
    if measurement_counts[0] != len(measurement.records) or clinical_count[0] != len(clinical.records):
        raise RuntimeError("Database row count validation failed")


def parse_args() -> IngestOptions:
    default_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=default_root)
    parser.add_argument("--dataset-version", default="aecd_all_measurements_20260803")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--exclude-averages", action="store_true")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--dbname")
    parser.add_argument("--user")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    return IngestOptions(args.repo_root, args.dataset_version, args.batch_size, args.dry_run, args.exclude_averages, args.host, args.port, args.dbname, args.user)


def main() -> None:
    options = parse_args()
    specs = discover_source_specs(options.repo_root)
    measurement = scan_sources(options.repo_root, specs, include_averages=not options.exclude_averages)
    clinical = scan_clinical(options.repo_root)
    _print_summary(options, measurement, clinical)
    if not options.dry_run:
        write_database(options, measurement, clinical)


if __name__ == "__main__":
    main()
