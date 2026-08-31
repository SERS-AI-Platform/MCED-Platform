from __future__ import annotations

import os
from collections import Counter
from getpass import getpass
from pathlib import Path

import click
import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extensions import cursor as PgCursor
from psycopg2.extras import execute_values

from scripts.db.canonical_postgres_mapping import (
    ClinicalInput,
    RawInput,
    decide_mapping,
    run_metadata,
    stable_id,
)
from scripts.db.canonical_postgres_rows import (
    ClinicalPlan,
    DbRow,
    RawPlan,
    build_clinical_plan,
    build_raw_plan,
)

SCHEMA_PATH = Path(__file__).with_name("canonical_postgres_schema.sql")
SITE_CODE = "SOLUM_RESEARCH"
ClinicalDbValue = str | int | float | bool | None | dict[str, str | int | float | bool | None]
ClinicalDbRow = tuple[ClinicalDbValue, ...]


class SourceAssetChangedError(ValueError):
    pass


class DuplicateMeasurementError(ValueError):
    pass


def connection_kwargs(
    host: str | None,
    port: int | None,
    dbname: str,
    user: str,
) -> dict[str, str | int]:
    password = os.environ.get("PGPASSWORD") or getpass("PostgreSQL password (not saved): ")
    if not password:
        raise RuntimeError("A PostgreSQL password is required")
    host_value = host or os.environ.get("PGHOST") or "localhost"
    port_value = port or int(os.environ.get("PGPORT", "5432"))
    return {
        "host": host_value,
        "port": port_value,
        "dbname": dbname,
        "user": user,
        "password": password,
    }


def load_inputs(
    connection: PgConnection,
    dataset_version: str,
    clinical_dataset_version: str,
) -> tuple[tuple[RawInput, ...], tuple[ClinicalDbRow, ...]]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT dataset_version, id, source_root, source_path, source_batch, source_kind, "
            "control_type, group_code, sample_id, replicate, source_sha256 "
            "FROM public.raw_spectra WHERE dataset_version = %s ORDER BY id",
            (dataset_version,),
        )
        raw_rows = tuple(RawInput(*row) for row in cursor.fetchall())
        cursor.execute(
            "SELECT dataset_version, id, source_workbook, source_sha256, source_row_number, "
            "solum_label, sample_type, collection_date, raw_record "
            "FROM public.clinical_patient_registry WHERE dataset_version = %s ORDER BY id",
            (clinical_dataset_version,),
        )
        clinical_rows = tuple(cursor.fetchall())
    return raw_rows, clinical_rows


def parse_clinical_row(row: ClinicalDbRow) -> ClinicalInput:
    raw_record = row[8]
    if not isinstance(raw_record, dict):
        raise TypeError("clinical raw_record must be a JSON object")
    normalized: dict[str, str | int | float | bool | None] = {}
    for key, value in raw_record.items():
        if not isinstance(key, str):
            raise TypeError("clinical raw_record keys must be strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(f"clinical raw_record value is not scalar: {key}")
        normalized[key] = value
    registry_id = row[1]
    source_row_number = row[4]
    if isinstance(registry_id, bool) or not isinstance(registry_id, int):
        raise TypeError("clinical registry id must be an integer")
    if isinstance(source_row_number, bool) or not isinstance(source_row_number, int):
        raise TypeError("clinical source row number must be an integer")
    return ClinicalInput(
        str(row[0]),
        registry_id,
        str(row[2]),
        str(row[3]),
        source_row_number,
        str(row[5]),
        str(row[6]) if row[6] is not None else None,
        str(row[7]) if row[7] is not None else None,
        normalized,
    )


def build_plans(
    raw_rows: tuple[RawInput, ...],
    clinical_rows: tuple[ClinicalDbRow, ...],
) -> tuple[tuple[ClinicalPlan, ...], tuple[RawPlan, ...]]:
    site_id = stable_id("site", SITE_CODE)
    clinical_plans = tuple(
        build_clinical_plan(parse_clinical_row(row), site_id)
        for row in clinical_rows
    )
    identities = {plan.identity.solum_label: plan.identity for plan in clinical_plans}
    raw_plans = tuple(
        build_raw_plan(row, decide_mapping(row, identities), run_metadata(row.source_root, row.source_batch), site_id)
        for row in raw_rows
    )
    measurement_keys = [
        (plan.run_id, plan.material_id, plan.measurement[3] if plan.measurement else None)
        for plan in raw_plans
        if plan.measurement is not None
    ]
    duplicates = [key for key, count in Counter(measurement_keys).items() if count > 1]
    if duplicates:
        raise DuplicateMeasurementError(f"duplicate canonical measurement keys: {duplicates[:5]}")
    return clinical_plans, raw_plans


def print_summary(
    raw_plans: tuple[RawPlan, ...],
    clinical_plans: tuple[ClinicalPlan, ...],
    dataset_version: str,
    clinical_dataset_version: str,
    dry_run: bool,
) -> None:
    counts = Counter(plan.decision.mapping_status for plan in raw_plans)
    print({
        "dry_run": dry_run,
        "dataset_version": dataset_version,
        "clinical_dataset_version": clinical_dataset_version,
        "clinical_rows": len(clinical_plans),
        "raw_rows": len(raw_plans),
        "measurement_rows": sum(plan.measurement is not None for plan in raw_plans),
        "measurement_runs": len({plan.run_id for plan in raw_plans}),
        "matched_measurements": counts["matched"],
        "review_measurements": counts["unmatched_spectrum"],
        "qc_rows": sum(plan.qc_artifact is not None for plan in raw_plans),
    })


def insert_values(cursor: PgCursor, sql: str, rows: tuple[DbRow, ...]) -> None:
    if rows:
        execute_values(cursor, sql, rows, page_size=500)


def write_plans(
    connection: PgConnection,
    clinical_plans: tuple[ClinicalPlan, ...],
    raw_plans: tuple[RawPlan, ...],
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        site_id = stable_id("site", SITE_CODE)
        insert_values(cursor, "INSERT INTO master.sites (id, code, name) VALUES %s ON CONFLICT (id) DO NOTHING", ((site_id, SITE_CODE, "SOLUM research registry"),))
        all_assets = tuple(plan.source_asset for plan in clinical_plans) + tuple(plan.source_asset for plan in raw_plans)
        for row in all_assets:
            cursor.execute("SELECT sha256 FROM master.source_assets WHERE uri = %s", (row[2],))
            existing = cursor.fetchone()
            if existing is not None and existing[0] != row[3]:
                raise SourceAssetChangedError(str(row[2]))
        insert_values(cursor, "INSERT INTO master.source_assets (id, site_id, uri, sha256, asset_kind, state, size_bytes, raw_uri) VALUES %s ON CONFLICT (id) DO NOTHING", all_assets)
        batches = tuple(plan.ingest_batch for plan in clinical_plans) + tuple(plan.ingest_batch for plan in raw_plans)
        insert_values(cursor, "INSERT INTO master.ingest_batches (id, source_asset_id, status, parser_name) VALUES %s ON CONFLICT (id) DO NOTHING", batches)
        insert_values(cursor, "INSERT INTO master.subjects (id, site_id, patient_id) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.subject for plan in clinical_plans))
        insert_values(cursor, "INSERT INTO master.samples (id, subject_id, site_id, solum_label, sample_type, collected_at) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.sample for plan in clinical_plans))
        insert_values(cursor, "INSERT INTO master.sample_crosswalk_sources (sample_id, source_asset_id, source_row_locator, identity_field) VALUES %s ON CONFLICT DO NOTHING", tuple(plan.crosswalk for plan in clinical_plans))
        insert_values(cursor, "INSERT INTO master.clinical_events (id, subject_id, site_id, ingest_batch_id, source_asset_id, source_row_locator, event_type, occurred_at) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.event for plan in clinical_plans))
        insert_values(cursor, "INSERT INTO master.clinical_observations (id, clinical_event_id, code, source_field_name, canonical_code, raw_value, value_kind, text_value, numeric_value, date_value, unit, normalization_status) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(row for plan in clinical_plans for row in plan.observations))
        insert_values(cursor, "INSERT INTO master.measurement_runs (id, site_id, ingest_batch_id, instrument_key, status, acquired_at) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.run_row for plan in raw_plans))
        insert_values(cursor, "INSERT INTO master.measurement_run_metadata (measurement_run_id, source_root, source_batch, preparation, reducing_agent, laser_power_mw, integration_time_s, average_count, randomization_block, metadata_status, raw_metadata) VALUES %s ON CONFLICT (measurement_run_id) DO NOTHING", tuple(plan.run_metadata for plan in raw_plans))
        insert_values(cursor, "INSERT INTO master.analytical_materials (id, sample_id, parent_material_id, material_type) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.material for plan in raw_plans if plan.material is not None))
        insert_values(cursor, "INSERT INTO master.spectrum_material_metadata (analytical_material_id, site_id, material_key, canonical_source_code, source_group, material_kind, preparation, fasting_state, specimen_timing, lot_code, identity_status, alias_candidates) VALUES %s ON CONFLICT (analytical_material_id) DO NOTHING", tuple(plan.material_metadata for plan in raw_plans if plan.material_metadata is not None))
        insert_values(cursor, "INSERT INTO master.measurements (id, measurement_run_id, analytical_material_id, replicate_index, status) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.measurement for plan in raw_plans if plan.measurement is not None))
        insert_values(cursor, "INSERT INTO master.measurement_artifacts (id, measurement_id, source_asset_id, artifact_role) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.artifact for plan in raw_plans if plan.artifact is not None))
        insert_values(cursor, "INSERT INTO master.measurement_metadata (measurement_id, raw_spectrum_id, source_kind, material_role, control_type, group_code, source_sample_id, acquisition_order, mapping_status, mapping_method, mapping_confidence, review_note) VALUES %s ON CONFLICT (measurement_id) DO NOTHING", tuple(plan.measurement_metadata for plan in raw_plans if plan.measurement_metadata is not None))
        insert_values(cursor, "INSERT INTO master.spectrum_inventory_records (source_asset_id, source_kind, status, reason_code, acquisition_date, instrument_key, root_key, canonical_source_code, preparation, fasting_state, specimen_timing, lot_code, identity_status, alias_candidates) VALUES %s ON CONFLICT (source_asset_id) DO NOTHING", tuple(plan.inventory for plan in raw_plans))
        insert_values(cursor, "INSERT INTO master.match_candidates (id, site_id, analytical_material_id, sample_id, clinical_event_id, score, status, rule_version, identity_basis, alias_review) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.match_candidate for plan in raw_plans if plan.match_candidate is not None))
        insert_values(cursor, "INSERT INTO master.qc_artifacts (id, measurement_run_id, source_asset_id, raw_spectrum_id, qc_role, qc_status) VALUES %s ON CONFLICT (id) DO NOTHING", tuple(plan.qc_artifact for plan in raw_plans if plan.qc_artifact is not None))


@click.command()
@click.option("--dataset-version", default="raw_data_csv_v1", show_default=True)
@click.option("--clinical-dataset-version", default="clinical_normalized_xlsx_v1", show_default=True)
@click.option("--dbname", default="sers_clinical", show_default=True)
@click.option("--user", "db_user", default="postgres", show_default=True)
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
@click.option("--dry-run", is_flag=True)
def main(
    dataset_version: str,
    clinical_dataset_version: str,
    dbname: str,
    db_user: str,
    host: str | None,
    port: int | None,
    dry_run: bool,
) -> None:
    kwargs = connection_kwargs(host, port, dbname, db_user)
    with psycopg2.connect(**kwargs) as connection:
        raw_rows, clinical_rows = load_inputs(connection, dataset_version, clinical_dataset_version)
        clinical_plans, raw_plans = build_plans(raw_rows, clinical_rows)
        print_summary(raw_plans, clinical_plans, dataset_version, clinical_dataset_version, dry_run)
        if not dry_run:
            write_plans(connection, clinical_plans, raw_plans)
            print({"canonical_schema": "master", "status": "written"})


if __name__ == "__main__":
    main()
