from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from getpass import getpass
from pathlib import Path
from typing import Any

import click
import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import execute_values

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from scripts.db.canonical_postgres_mapping import (
        ClinicalIdentity,
        ClinicalInput,
        normalize_label,
        stable_id,
    )
    from scripts.db.canonical_postgres_rows import build_clinical_plan
except ModuleNotFoundError:
    from canonical_postgres_mapping import (
        ClinicalIdentity,
        ClinicalInput,
        normalize_label,
        stable_id,
    )
    from canonical_postgres_rows import build_clinical_plan


SCHEMA_PATH = Path(__file__).with_name("canonical_postgres_schema.sql")
SITE_CODE = "SOLUM_RESEARCH"
PRIMARY_ROLES = frozenset({"raw", "replicate"})
AVERAGE_ROLE = "average"
QC_ROLES = frozenset({"background", "calibration_control", "equipment_test"})
REFERENCE_ROLES = frozenset({"metabolite_reference"})


@dataclass(frozen=True, slots=True)
class AecdRow:
    row_id: int
    dataset_version: str
    source_domain: str
    source_kind: str
    instrument_key: str
    preparation: str
    source_root: str
    source_batch: str
    source_path: str
    artifact_role: str
    variant: str
    is_averaged: bool
    measurement_key: str
    group_code: str | None
    sample_id: str
    replicate: int | None
    control_type: str | None
    acquisition_date: str | None
    source_sha256: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AecdPlan:
    row: AecdRow
    source_asset_id: str
    ingest_batch_id: str
    run_id: str
    material_id: str | None
    measurement_id: str | None
    solum_label: str | None
    identity: ClinicalIdentity | None
    mapping_status: str
    source_asset: tuple[object, ...]
    ingest_batch: tuple[object, ...]
    run: tuple[object, ...]
    run_metadata: tuple[object, ...]
    material: tuple[object, ...] | None
    material_metadata: tuple[object, ...] | None
    measurement: tuple[object, ...] | None
    artifact: tuple[object, ...] | None
    measurement_metadata: tuple[object, ...] | None
    inventory: tuple[object, ...]
    match_candidate: tuple[object, ...] | None
    qc_artifact: tuple[object, ...] | None


def connection_kwargs(
    host: str | None,
    port: int | None,
    dbname: str,
    user: str,
) -> dict[str, str | int]:
    password = os.environ.get("PGPASSWORD") or getpass(
        "PostgreSQL password (not saved): "
    )
    if not password:
        raise RuntimeError("A PostgreSQL password is required")
    return {
        "host": host or os.environ.get("PGHOST", "localhost"),
        "port": port or int(os.environ.get("PGPORT", "5432")),
        "dbname": dbname,
        "user": user,
        "password": password,
    }


def canonical_aecd_schema() -> str:
    schema = SCHEMA_PATH.read_text(encoding="utf-8")
    schema = schema.split("CREATE OR REPLACE VIEW master.spectrum_measurement_view", 1)[0]
    schema = schema.replace(
        "raw_spectrum_id BIGINT NOT NULL UNIQUE REFERENCES public.raw_spectra(id)",
        "aecd_measurement_id BIGINT NOT NULL UNIQUE REFERENCES public.aecd_measurements(id)",
    )
    schema = schema.replace(
        "idx_master_measurement_metadata_raw ON master.measurement_metadata(raw_spectrum_id)",
        "idx_master_measurement_metadata_aecd ON master.measurement_metadata(aecd_measurement_id)",
    )
    schema += "\n".join(
        (
            "",
            "CREATE TABLE IF NOT EXISTS master.aecd_measurement_links (",
            "    aecd_measurement_id BIGINT PRIMARY KEY REFERENCES public.aecd_measurements(id),",
            "    source_asset_id TEXT NOT NULL REFERENCES master.source_assets(id),",
            "    measurement_id TEXT REFERENCES master.measurements(id),",
            "    artifact_role TEXT NOT NULL,",
            "    canonical_status TEXT NOT NULL CHECK (canonical_status IN (",
            "        'measurement', 'derived_attached', 'derived_unattached',",
            "        'qc_measurement', 'reference_measurement'",
            "    )),",
            "    solum_label TEXT,",
            "    mapping_status TEXT NOT NULL,",
            "    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP",
            ");",
            "",
            "CREATE INDEX IF NOT EXISTS idx_master_aecd_links_status",
            "    ON master.aecd_measurement_links(canonical_status, artifact_role);",
            "CREATE INDEX IF NOT EXISTS idx_master_aecd_links_measurement",
            "    ON master.aecd_measurement_links(measurement_id);",
            "",
            "CREATE OR REPLACE VIEW master.aecd_spectrum_measurement_view AS",
            "SELECT",
            "    source.id AS aecd_measurement_id,",
            "    link.measurement_id,",
            "    link.canonical_status,",
            "    link.mapping_status,",
            "    link.solum_label,",
            "    source.dataset_version,",
            "    source.source_domain,",
            "    source.source_kind,",
            "    source.instrument_key,",
            "    source.preparation,",
            "    source.source_root,",
            "    source.source_batch,",
            "    source.source_path,",
            "    source.artifact_role,",
            "    source.variant,",
            "    source.is_averaged,",
            "    source.measurement_key,",
            "    source.group_code,",
            "    source.sample_id,",
            "    source.replicate,",
            "    source.control_type,",
            "    source.acquisition_date,",
            "    source.wavenumber,",
            "    source.intensities,",
            "    source.n_points,",
            "    source.x_min,",
            "    source.x_max,",
            "    source.source_sha256,",
            "    source.metadata",
            "FROM public.aecd_measurements AS source",
            "JOIN master.aecd_measurement_links AS link",
            "  ON link.aecd_measurement_id = source.id;",
        )
    )
    return schema


def _json_scalars(value: object) -> dict[str, str | int | float | bool | None]:
    if not isinstance(value, dict):
        raise TypeError("clinical raw_record must be a JSON object")
    result: dict[str, str | int | float | bool | None] = {}
    for key, item in value.items():
        if not isinstance(key, str):
            raise TypeError("clinical raw_record keys must be strings")
        if item is not None and not isinstance(item, (str, int, float, bool)):
            raise TypeError(f"clinical raw_record value is not scalar: {key}")
        result[key] = item
    return result


def load_clinical_rows(
    connection: PgConnection,
    dataset_version: str,
) -> tuple[ClinicalInput, ...]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT dataset_version, id, source_workbook, source_sha256, "
            "source_row_number, solum_label, sample_type, collection_date, raw_record "
            "FROM public.clinical_patient_registry WHERE dataset_version = %s "
            "ORDER BY id",
            (dataset_version,),
        )
        rows = cursor.fetchall()
    return tuple(
        ClinicalInput(
            str(row[0]),
            int(row[1]),
            str(row[2]),
            str(row[3]),
            int(row[4]),
            str(row[5]),
            str(row[6]) if row[6] is not None else None,
            str(row[7]) if row[7] is not None else None,
            _json_scalars(row[8]),
        )
        for row in rows
    )


def load_aecd_rows(
    connection: PgConnection,
    dataset_version: str,
) -> tuple[AecdRow, ...]:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT id, dataset_version, source_domain, source_kind, instrument_key, "
            "preparation, source_root, source_batch, source_path, artifact_role, "
            "variant, is_averaged, measurement_key, group_code, sample_id, replicate, "
            "control_type, acquisition_date, source_sha256, metadata "
            "FROM public.aecd_measurements WHERE dataset_version = %s ORDER BY id",
            (dataset_version,),
        )
        rows = cursor.fetchall()
    return tuple(
        AecdRow(
            row_id=int(row[0]),
            dataset_version=str(row[1]),
            source_domain=str(row[2]),
            source_kind=str(row[3]),
            instrument_key=str(row[4]),
            preparation=str(row[5]),
            source_root=str(row[6]),
            source_batch=str(row[7]),
            source_path=str(row[8]),
            artifact_role=str(row[9]),
            variant=str(row[10]),
            is_averaged=bool(row[11]),
            measurement_key=str(row[12]),
            group_code=str(row[13]) if row[13] is not None else None,
            sample_id=str(row[14]),
            replicate=int(row[15]) if row[15] is not None else None,
            control_type=str(row[16]) if row[16] is not None else None,
            acquisition_date=str(row[17]) if row[17] is not None else None,
            source_sha256=str(row[18]),
            metadata=dict(row[19]) if isinstance(row[19], dict) else {},
        )
        for row in rows
    )


def _material_kind(row: AecdRow) -> str:
    if row.artifact_role == "background":
        return "matrix_blank"
    if row.artifact_role == "calibration_control":
        return "blank"
    if row.artifact_role in QC_ROLES | REFERENCE_ROLES:
        return "reference"
    if row.group_code == "BLANK":
        return "blank"
    return "biological"


def _material_key(row: AecdRow) -> str:
    identity = row.control_type or row.group_code or "UNSET"
    return "|".join(
        (
            row.dataset_version,
            row.source_domain,
            row.source_batch,
            row.instrument_key,
            row.preparation,
            identity,
            row.sample_id,
        )
    )


def _run_key(row: AecdRow) -> str:
    return "|".join(
        (
            row.dataset_version,
            row.source_root,
            row.source_batch,
            row.source_kind,
            row.instrument_key,
            row.preparation,
        )
    )


def _solum_label(row: AecdRow) -> str | None:
    if row.group_code is None or row.artifact_role in QC_ROLES | REFERENCE_ROLES:
        return None
    return normalize_label(row.group_code, row.sample_id)


def build_plans(
    clinical_rows: tuple[ClinicalInput, ...],
    aecd_rows: tuple[AecdRow, ...],
) -> tuple[tuple[object, ...], tuple[AecdPlan, ...]]:
    site_id = stable_id("site", SITE_CODE)
    clinical_plans = tuple(
        build_clinical_plan(record, site_id) for record in clinical_rows
    )
    identities = {plan.identity.solum_label: plan.identity for plan in clinical_plans}

    material_ids: dict[str, str] = {}
    run_ids: dict[str, str] = {}
    primary_measurements: dict[str, str] = {}
    primary_by_material: dict[str, str] = {}
    for row in aecd_rows:
        run_ids.setdefault(
            _run_key(row),
            stable_id("measurement-run", f"{site_id}:{_run_key(row)}"),
        )
        if row.artifact_role != AVERAGE_ROLE:
            material_ids.setdefault(
                _material_key(row),
                stable_id("analytical-material", f"{site_id}:{_material_key(row)}"),
            )
            measurement_id = stable_id(
                "measurement", f"{row.dataset_version}:aecd:{row.row_id}"
            )
            primary_measurements[str(row.row_id)] = measurement_id
            primary_by_material.setdefault(_material_key(row), measurement_id)

    plans: list[AecdPlan] = []
    for row in aecd_rows:
        source_uri = f"raw://{row.dataset_version}/{row.source_path}"
        source_asset_id = stable_id("source-asset", source_uri)
        ingest_batch_id = stable_id("ingest-batch", source_asset_id)
        run_id = run_ids[_run_key(row)]
        material_key = _material_key(row)
        material_id = material_ids.get(material_key)
        label = _solum_label(row)
        identity = identities.get(label) if label is not None else None
        mapping_status = (
            "not_applicable"
            if label is None
            else "matched"
            if identity is not None
            else "unmatched_spectrum"
        )
        measurement_id = primary_measurements.get(str(row.row_id))
        if row.artifact_role == AVERAGE_ROLE:
            measurement_id = primary_by_material.get(material_key)

        source_asset = (
            source_asset_id,
            site_id,
            source_uri,
            row.source_sha256,
            "spectrum",
            "ingested",
            None,
            None,
        )
        ingest_batch = (
            ingest_batch_id,
            source_asset_id,
            "completed",
            "postgres-aecd-bridge-v1",
        )
        run = (
            run_id,
            site_id,
            None,
            row.instrument_key,
            "complete",
            row.acquisition_date,
        )
        raw_metadata = {
            "source_domain": row.source_domain,
            "source_kind": row.source_kind,
            "artifact_role": row.artifact_role,
            "variant": row.variant,
            "measurement_key": row.measurement_key,
            "aecd_metadata": row.metadata,
        }
        run_metadata = (
            run_id,
            row.source_root,
            row.source_batch,
            row.preparation,
            None,
            None,
            None,
            None,
            None,
            "source_table",
            json.dumps(raw_metadata, ensure_ascii=False, sort_keys=True),
        )

        material = None
        material_metadata = None
        if material_id is not None:
            material = (
                material_id,
                identity.sample_id if identity is not None else None,
                None,
                "primary" if material_key in primary_by_material else "average",
            )
            material_metadata = (
                material_id,
                site_id,
                material_key,
                label or row.control_type or row.group_code or "UNSET",
                row.group_code or row.control_type or "UNSET",
                _material_kind(row),
                row.preparation,
                "unspecified",
                "unspecified",
                None,
                "canonical" if label is not None else "alias_review",
                "",
            )

        measurement = None
        artifact = None
        measurement_metadata = None
        if row.artifact_role != AVERAGE_ROLE and measurement_id is not None:
            replicate_index = row.replicate or 1
            measurement = (
                measurement_id,
                run_id,
                material_id,
                replicate_index,
                "acquired",
            )
        if measurement_id is not None:
            artifact_role = "processed" if row.artifact_role == AVERAGE_ROLE else "raw"
            artifact = (
                stable_id(
                    "artifact",
                    f"{row.dataset_version}:aecd:{row.row_id}:{artifact_role}",
                ),
                measurement_id,
                source_asset_id,
                artifact_role,
            )
        if row.artifact_role != AVERAGE_ROLE and measurement_id is not None:
            measurement_metadata = (
                measurement_id,
                row.row_id,
                row.source_kind,
                "qc" if row.artifact_role in QC_ROLES else "reference" if row.artifact_role in REFERENCE_ROLES else "patient_sample",
                row.control_type,
                row.group_code,
                row.sample_id,
                row.replicate,
                mapping_status,
                "solum_label_exact" if label is not None else "source_role",
                1.0 if identity is not None else 0.0,
                None if identity is not None or label is None else "no clinical Solum_label match",
            )

        match_candidate = None
        if measurement_id is not None and row.artifact_role not in QC_ROLES | REFERENCE_ROLES:
            match_candidate = (
                stable_id("match-candidate", f"{row.dataset_version}:aecd:{row.row_id}"),
                site_id,
                material_id,
                identity.sample_id if identity is not None else None,
                identity.clinical_event_id if identity is not None else None,
                1.0 if identity is not None else 0.0,
                "matched" if identity is not None else "unmatched_spectrum",
                "solum_label_exact_v1",
                "solum_label",
                False,
            )

        inventory_status = "derived" if row.artifact_role == AVERAGE_ROLE else "ready"
        if row.artifact_role == AVERAGE_ROLE and measurement_id is None:
            inventory_status = "quarantined"
        inventory = (
            source_asset_id,
            row.artifact_role,
            inventory_status,
            "average_without_measurement"
            if row.artifact_role == AVERAGE_ROLE and measurement_id is None
            else None,
            row.acquisition_date,
            row.instrument_key,
            row.source_root,
            label or row.control_type or row.group_code,
            row.preparation,
            "unspecified",
            "unspecified",
            None,
            "canonical" if label is not None else "alias_review",
            "",
        )
        qc_artifact = None
        if row.artifact_role in QC_ROLES:
            qc_artifact = (
                stable_id("qc-artifact", f"{row.dataset_version}:aecd:{row.row_id}"),
                run_id,
                source_asset_id,
                row.row_id,
                row.artifact_role,
                "unreviewed",
            )
        plans.append(
            AecdPlan(
                row,
                source_asset_id,
                ingest_batch_id,
                run_id,
                material_id,
                measurement_id,
                label,
                identity,
                mapping_status,
                source_asset,
                ingest_batch,
                run,
                run_metadata,
                material,
                material_metadata,
                measurement,
                artifact,
                measurement_metadata,
                inventory,
                match_candidate,
                qc_artifact,
            )
        )
    return tuple(clinical_plans), tuple(plans)


def _insert_values(cursor: object, sql: str, rows: tuple[tuple[object, ...], ...]) -> None:
    if rows:
        execute_values(cursor, sql, rows, page_size=500)


def _condition_summary(rows: tuple[AecdRow, ...]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                f"{row.source_domain}:{row.source_kind}:{row.instrument_key}:"
                f"{row.preparation}:{row.artifact_role}:{row.variant}"
                for row in rows
            ).items()
        )
    )


def _condition_field_missing(rows: tuple[AecdRow, ...]) -> dict[str, int]:
    fields = (
        "source_domain",
        "source_kind",
        "instrument_key",
        "preparation",
        "source_batch",
        "artifact_role",
        "variant",
        "source_sha256",
    )
    return {
        field: sum(not getattr(row, field) for row in rows)
        for field in fields
    }


def _unmatched_breakdown(plans: tuple[AecdPlan, ...]) -> dict[str, dict[str, int]]:
    unmatched = tuple(
        plan
        for plan in plans
        if plan.row.artifact_role in PRIMARY_ROLES
        and plan.mapping_status == "unmatched_spectrum"
    )
    by_source = Counter(
        f"{plan.row.source_domain}:{plan.row.source_kind}:"
        f"{plan.row.instrument_key}:{plan.row.preparation}"
        for plan in unmatched
    )
    by_group = Counter(plan.row.group_code or "<NO_GROUP>" for plan in unmatched)
    by_reason = Counter(
        "no_clinical_Solum_label_match"
        if plan.solum_label is not None
        else "no_group_or_sample_key"
        for plan in unmatched
    )
    return {
        "by_source": dict(sorted(by_source.items())),
        "by_group": dict(sorted(by_group.items())),
        "by_reason": dict(sorted(by_reason.items())),
    }


def print_summary(
    clinical_rows: tuple[ClinicalInput, ...],
    aecd_rows: tuple[AecdRow, ...],
    plans: tuple[AecdPlan, ...],
    dry_run: bool,
) -> None:
    patient_status = Counter(
        plan.mapping_status
        for plan in plans
        if plan.row.artifact_role in PRIMARY_ROLES
    )
    average_status = Counter(
        plan.mapping_status
        for plan in plans
        if plan.row.artifact_role == AVERAGE_ROLE
    )
    print(
        {
            "dry_run": dry_run,
            "clinical_rows": len(clinical_rows),
            "aecd_rows": len(aecd_rows),
            "non_average_measurements": sum(
                row.artifact_role != AVERAGE_ROLE for row in aecd_rows
            ),
            "patient_measurements": sum(
                row.artifact_role in PRIMARY_ROLES for row in aecd_rows
            ),
            "patient_mapping_status": dict(sorted(patient_status.items())),
            "matched_patient_measurements": patient_status["matched"],
            "unmatched_patient_measurements": patient_status["unmatched_spectrum"],
            "patient_not_applicable": patient_status["not_applicable"],
            "average_mapping_status": dict(sorted(average_status.items())),
            "unmatched_patient_breakdown": _unmatched_breakdown(plans),
            "condition_groups": _condition_summary(aecd_rows),
            "condition_field_missing": _condition_field_missing(aecd_rows),
        }
    )


def write_database(
    connection: PgConnection,
    clinical_plans: tuple[object, ...],
    plans: tuple[AecdPlan, ...],
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT table_name, column_name FROM information_schema.columns "
            "WHERE table_schema = 'master' AND table_name IN "
            "('measurement_metadata', 'qc_artifacts')"
        )
        existing_columns = defaultdict(set)
        for table_name, column_name in cursor.fetchall():
            existing_columns[table_name].add(column_name)
        if (
            "raw_spectrum_id" in existing_columns["measurement_metadata"]
            and "aecd_measurement_id" not in existing_columns["measurement_metadata"]
        ) or (
            "raw_spectrum_id" in existing_columns["qc_artifacts"]
            and "aecd_measurement_id" not in existing_columns["qc_artifacts"]
        ):
            raise RuntimeError(
                "master schema is already tied to public.raw_spectra; "
                "use a fresh master schema or migrate it before this bridge"
            )
        cursor.execute(canonical_aecd_schema())
        site_id = stable_id("site", SITE_CODE)
        cursor.execute(
            "INSERT INTO master.sites (id, code, name) VALUES (%s, %s, %s) "
            "ON CONFLICT (id) DO NOTHING",
            (site_id, SITE_CODE, "SOLUM research registry"),
        )
        clinical_assets = tuple(plan.source_asset for plan in clinical_plans)
        spectrum_assets = tuple(plan.source_asset for plan in plans)
        all_assets = clinical_assets + spectrum_assets
        for asset in all_assets:
            cursor.execute(
                "SELECT sha256 FROM master.source_assets WHERE uri = %s",
                (asset[2],),
            )
            existing = cursor.fetchone()
            if existing is not None and existing[0] != asset[3]:
                raise RuntimeError(f"Source asset changed: {asset[2]}")
        _insert_values(
            cursor,
            "INSERT INTO master.source_assets "
            "(id, site_id, uri, sha256, asset_kind, state, size_bytes, raw_uri) "
            "VALUES %s ON CONFLICT (id) DO NOTHING",
            all_assets,
        )
        _insert_values(
            cursor,
            "INSERT INTO master.ingest_batches "
            "(id, source_asset_id, status, parser_name) VALUES %s "
            "ON CONFLICT (id) DO NOTHING",
            tuple(plan.ingest_batch for plan in clinical_plans)
            + tuple(plan.ingest_batch for plan in plans),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.subjects (id, site_id, patient_id) VALUES %s "
            "ON CONFLICT (id) DO NOTHING",
            tuple(plan.subject for plan in clinical_plans),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.samples "
            "(id, subject_id, site_id, solum_label, sample_type, collected_at) "
            "VALUES %s ON CONFLICT (id) DO NOTHING",
            tuple(plan.sample for plan in clinical_plans),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.sample_crosswalk_sources "
            "(sample_id, source_asset_id, source_row_locator, identity_field) "
            "VALUES %s ON CONFLICT DO NOTHING",
            tuple(plan.crosswalk for plan in clinical_plans),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.clinical_events "
            "(id, subject_id, site_id, ingest_batch_id, source_asset_id, "
            "source_row_locator, event_type, occurred_at) VALUES %s "
            "ON CONFLICT (id) DO NOTHING",
            tuple(plan.event for plan in clinical_plans),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.clinical_observations "
            "(id, clinical_event_id, code, source_field_name, canonical_code, "
            "raw_value, value_kind, text_value, numeric_value, date_value, unit, "
            "normalization_status) VALUES %s ON CONFLICT (id) DO NOTHING",
            tuple(row for plan in clinical_plans for row in plan.observations),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.measurement_runs "
            "(id, site_id, ingest_batch_id, instrument_key, status, acquired_at) "
            "VALUES %s ON CONFLICT (id) DO NOTHING",
            tuple(plan.run for plan in plans),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.measurement_run_metadata "
            "(measurement_run_id, source_root, source_batch, preparation, "
            "reducing_agent, laser_power_mw, integration_time_s, average_count, "
            "randomization_block, metadata_status, raw_metadata) VALUES %s "
            "ON CONFLICT (measurement_run_id) DO NOTHING",
            tuple(plan.run_metadata for plan in plans),
        )
        materials = tuple(
            plan.material for plan in plans if plan.material is not None
        )
        material_metadata = tuple(
            plan.material_metadata
            for plan in plans
            if plan.material_metadata is not None
        )
        _insert_values(
            cursor,
            "INSERT INTO master.analytical_materials "
            "(id, sample_id, parent_material_id, material_type) VALUES %s "
            "ON CONFLICT (id) DO NOTHING",
            materials,
        )
        _insert_values(
            cursor,
            "INSERT INTO master.spectrum_material_metadata "
            "(analytical_material_id, site_id, material_key, canonical_source_code, "
            "source_group, material_kind, preparation, fasting_state, "
            "specimen_timing, lot_code, identity_status, alias_candidates) "
            "VALUES %s ON CONFLICT (analytical_material_id) DO NOTHING",
            material_metadata,
        )
        measurements = tuple(
            plan.measurement for plan in plans if plan.measurement is not None
        )
        _insert_values(
            cursor,
            "INSERT INTO master.measurements "
            "(id, measurement_run_id, analytical_material_id, replicate_index, status) "
            "VALUES %s ON CONFLICT (id) DO NOTHING",
            measurements,
        )
        _insert_values(
            cursor,
            "INSERT INTO master.measurement_artifacts "
            "(id, measurement_id, source_asset_id, artifact_role) VALUES %s "
            "ON CONFLICT DO NOTHING",
            tuple(plan.artifact for plan in plans if plan.artifact is not None),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.measurement_metadata "
            "(measurement_id, aecd_measurement_id, source_kind, material_role, "
            "control_type, group_code, source_sample_id, acquisition_order, "
            "mapping_status, mapping_method, mapping_confidence, review_note) "
            "VALUES %s ON CONFLICT (measurement_id) DO NOTHING",
            tuple(
                plan.measurement_metadata
                for plan in plans
                if plan.measurement_metadata is not None
            ),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.spectrum_inventory_records "
            "(source_asset_id, source_kind, status, reason_code, acquisition_date, "
            "instrument_key, root_key, canonical_source_code, preparation, "
            "fasting_state, specimen_timing, lot_code, identity_status, "
            "alias_candidates) VALUES %s ON CONFLICT (source_asset_id) DO NOTHING",
            tuple(plan.inventory for plan in plans),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.match_candidates "
            "(id, site_id, analytical_material_id, sample_id, clinical_event_id, "
            "score, status, rule_version, identity_basis, alias_review) VALUES %s "
            "ON CONFLICT (id) DO NOTHING",
            tuple(plan.match_candidate for plan in plans if plan.match_candidate is not None),
        )
        _insert_values(
            cursor,
            "INSERT INTO master.qc_artifacts "
            "(id, measurement_run_id, source_asset_id, aecd_measurement_id, "
            "qc_role, qc_status) VALUES %s ON CONFLICT (id) DO NOTHING",
            tuple(plan.qc_artifact for plan in plans if plan.qc_artifact is not None),
        )
        links = tuple(
            (
                plan.row.row_id,
                plan.source_asset_id,
                plan.measurement_id,
                plan.row.artifact_role,
                "derived_attached"
                if plan.row.artifact_role == AVERAGE_ROLE and plan.measurement_id is not None
                else "derived_unattached"
                if plan.row.artifact_role == AVERAGE_ROLE
                else "qc_measurement"
                if plan.row.artifact_role in QC_ROLES
                else "reference_measurement"
                if plan.row.artifact_role in REFERENCE_ROLES
                else "measurement",
                plan.solum_label,
                plan.mapping_status,
            )
            for plan in plans
        )
        _insert_values(
            cursor,
            "INSERT INTO master.aecd_measurement_links "
            "(aecd_measurement_id, source_asset_id, measurement_id, artifact_role, "
            "canonical_status, solum_label, mapping_status) VALUES %s "
            "ON CONFLICT (aecd_measurement_id) DO UPDATE SET "
            "source_asset_id = EXCLUDED.source_asset_id, "
            "measurement_id = EXCLUDED.measurement_id, "
            "artifact_role = EXCLUDED.artifact_role, "
            "canonical_status = EXCLUDED.canonical_status, "
            "solum_label = EXCLUDED.solum_label, "
            "mapping_status = EXCLUDED.mapping_status",
            links,
        )


def report_database(
    connection: PgConnection,
    dataset_version: str,
) -> None:
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT artifact_role, source_domain, source_kind, instrument_key, "
            "preparation, variant, COUNT(*), COUNT(DISTINCT group_code), "
            "COUNT(DISTINCT sample_id) "
            "FROM master.aecd_spectrum_measurement_view "
            "WHERE dataset_version = %s "
            "GROUP BY artifact_role, source_domain, source_kind, instrument_key, "
            "preparation, variant "
            "ORDER BY source_domain, source_kind, instrument_key, preparation, "
            "artifact_role, variant",
            (dataset_version,),
        )
        conditions = cursor.fetchall()
        cursor.execute(
            "SELECT canonical_status, artifact_role, COUNT(*) "
            "FROM master.aecd_measurement_links l "
            "JOIN public.aecd_measurements a ON a.id = l.aecd_measurement_id "
            "WHERE a.dataset_version = %s "
            "GROUP BY canonical_status, artifact_role "
            "ORDER BY canonical_status, artifact_role",
            (dataset_version,),
        )
        links = cursor.fetchall()
        cursor.execute(
            "SELECT COUNT(*) FILTER (WHERE l.mapping_status = 'matched'), "
            "COUNT(*) FILTER (WHERE l.mapping_status = 'unmatched_spectrum'), "
            "COUNT(DISTINCT l.solum_label) "
            "FROM master.aecd_measurement_links l "
            "JOIN public.aecd_measurements a ON a.id = l.aecd_measurement_id "
            "WHERE a.dataset_version = %s AND l.artifact_role <> 'average'",
            (dataset_version,),
        )
        matching = cursor.fetchone()
    print({
        "canonical_conditions": [tuple(row) for row in conditions],
        "canonical_links": [tuple(row) for row in links],
        "canonical_matching": tuple(matching) if matching is not None else None,
    })


@click.command()
@click.option("--dataset-version", default="aecd_all_measurements_20260803", show_default=True)
@click.option("--clinical-dataset-version", default="clinical_normalized_xlsx_v1", show_default=True)
@click.option("--dbname", default="sers_clinical", show_default=True)
@click.option("--user", "db_user", default="postgres", show_default=True)
@click.option("--host", default=None)
@click.option("--port", type=int, default=None)
@click.option("--dry-run", is_flag=True)
@click.option("--report-only", is_flag=True)
def main(
    dataset_version: str,
    clinical_dataset_version: str,
    dbname: str,
    db_user: str,
    host: str | None,
    port: int | None,
    dry_run: bool,
    report_only: bool,
) -> None:
    kwargs = connection_kwargs(host, port, dbname, db_user)
    with psycopg2.connect(**kwargs) as connection:
        if report_only:
            report_database(connection, dataset_version)
            return
        clinical_rows = load_clinical_rows(connection, clinical_dataset_version)
        aecd_rows = load_aecd_rows(connection, dataset_version)
        if not clinical_rows:
            raise click.ClickException(
                "No clinical_patient_registry rows found. The bridge requires "
                "clinical Solum_label values."
            )
        if not aecd_rows:
            raise click.ClickException(
                "No public.aecd_measurements rows found for the requested dataset."
            )
        clinical_plans, plans = build_plans(clinical_rows, aecd_rows)
        print_summary(clinical_rows, aecd_rows, plans, dry_run)
        if dry_run:
            return
        write_database(connection, clinical_plans, plans)
        report_database(connection, dataset_version)
        print({"canonical_schema": "master", "status": "written"})


if __name__ == "__main__":
    main()
