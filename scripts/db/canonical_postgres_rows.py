from __future__ import annotations

import json
import re
from dataclasses import dataclass

from scripts.db.canonical_postgres_mapping import (
    ClinicalIdentity,
    ClinicalInput,
    MappingDecision,
    RawInput,
    RunMetadata,
    stable_id,
)

DbScalar = str | int | float | bool | None
DbRow = tuple[DbScalar, ...]


@dataclass(frozen=True, slots=True)
class ClinicalPlan:
    identity: ClinicalIdentity
    source_asset: DbRow
    ingest_batch: DbRow
    subject: DbRow
    sample: DbRow
    crosswalk: DbRow
    event: DbRow
    observations: tuple[DbRow, ...]


@dataclass(frozen=True, slots=True)
class RawPlan:
    raw: RawInput
    decision: MappingDecision
    run: RunMetadata
    source_asset_id: str
    ingest_batch_id: str
    run_id: str
    material_id: str | None
    measurement_id: str | None
    source_asset: DbRow
    ingest_batch: DbRow
    run_row: DbRow
    run_metadata: DbRow
    material: DbRow | None
    material_metadata: DbRow | None
    measurement: DbRow | None
    artifact: DbRow | None
    measurement_metadata: DbRow | None
    inventory: DbRow
    match_candidate: DbRow | None
    qc_artifact: DbRow | None


def _sample_type(value: str | None) -> str:
    normalized = (value or "").strip().casefold()
    if normalized in {"urine", "serum", "plasma"}:
        return normalized
    return "other"


def _observation(field: str, value: DbScalar, observation_id: str, event_id: str) -> DbRow | None:
    if value is None or value == "":
        return None
    raw_value = str(value).lower() if isinstance(value, bool) else str(value)
    if isinstance(value, bool):
        return (observation_id, event_id, field, field, None, raw_value, "text", raw_value, None, None, None, "raw")
    if isinstance(value, (int, float)):
        return (observation_id, event_id, field, field, None, raw_value, "numeric", None, float(value), None, None, "raw")
    is_date = field.casefold().endswith("date") or re.fullmatch(r"\d{4}-\d{2}-\d{2}", value) is not None
    if is_date:
        return (observation_id, event_id, field, field, None, raw_value, "date", None, None, raw_value, None, "raw")
    return (observation_id, event_id, field, field, None, raw_value, "text", raw_value, None, None, None, "raw")


def build_clinical_plan(record: ClinicalInput, site_id: str) -> ClinicalPlan:
    source_uri = record.source_workbook
    source_asset_id = stable_id("source-asset", f"clinical:{source_uri}")
    batch_id = stable_id("ingest-batch", f"clinical:{record.dataset_version}:{source_asset_id}")
    subject_id = stable_id("subject", f"{site_id}:{record.solum_label}")
    sample_id = stable_id("sample", f"{site_id}:{record.solum_label}")
    event_id = stable_id("clinical-event", f"{source_asset_id}:{record.source_row_number}")
    identity = ClinicalIdentity(record.registry_id, record.solum_label, subject_id, sample_id, event_id)
    observations = tuple(
        row
        for field, value in sorted(record.raw_record.items())
        for row in (
            _observation(field, value, stable_id("clinical-observation", f"{event_id}:{field}"), event_id),
        )
        if row is not None
    )
    source_asset = (source_asset_id, site_id, source_uri, record.source_sha256, "clinical", "ingested", None, None)
    ingest_batch = (batch_id, source_asset_id, "completed", "postgres-clinical-registry-v1")
    subject = (subject_id, site_id, None)
    sample = (sample_id, subject_id, site_id, record.solum_label, _sample_type(record.sample_type), record.collection_date)
    crosswalk = (sample_id, source_asset_id, str(record.source_row_number), "solum_label")
    event_type = "collection" if record.collection_date else "other"
    event = (event_id, subject_id, site_id, batch_id, source_asset_id, str(record.source_row_number), event_type, record.collection_date)
    return ClinicalPlan(identity, source_asset, ingest_batch, subject, sample, crosswalk, event, observations)


def build_raw_plan(
    raw: RawInput,
    decision: MappingDecision,
    run: RunMetadata,
    site_id: str,
) -> RawPlan:
    source_uri = f"raw://{raw.dataset_version}/{raw.source_path}"
    source_asset_id = stable_id("source-asset", source_uri)
    ingest_batch_id = stable_id("ingest-batch", source_asset_id)
    run_id = stable_id("measurement-run", f"{site_id}:{raw.dataset_version}:{run.run_key}")
    material_id = None
    measurement_id = None
    if decision.material_role != "qc":
        material_id = stable_id("analytical-material", f"{run_id}:{decision.solum_label or raw.raw_id}")
    if raw.source_kind == "replicate" and material_id is not None:
        measurement_id = stable_id("measurement", f"{raw.dataset_version}:{raw.raw_id}")
    source_asset = (source_asset_id, site_id, source_uri, raw.source_sha256, "spectrum", "ingested", None, None)
    ingest_batch = (ingest_batch_id, source_asset_id, "completed", "postgres-raw-spectra-v1")
    run_row = (run_id, site_id, None, "unknown", "complete", run.acquisition_date)
    run_metadata = (
        run_id,
        raw.source_root,
        raw.source_batch,
        None,
        None,
        run.laser_power_mw,
        run.integration_time_s,
        run.average_count,
        None,
        run.metadata_status,
        json.dumps({"source_kind": raw.source_kind, "control_type": raw.control_type}),
    )
    material = None
    material_metadata = None
    if material_id is not None:
        material_type = "average" if raw.source_kind == "average" else "primary"
        material = (material_id, decision.identity.sample_id if decision.identity else None, None, material_type)
        source_group = raw.group_code or raw.control_type or "unknown"
        material_metadata = (
            material_id,
            site_id,
            f"{run.run_key}:{decision.solum_label or raw.raw_id}",
            decision.solum_label or f"{source_group}_{raw.sample_id}",
            source_group,
            "biological",
            "unknown",
            "unspecified",
            "unspecified",
            None,
            "canonical" if decision.identity else "alias_review",
            "",
        )
    measurement = None
    artifact = None
    measurement_metadata = None
    match_candidate = None
    if measurement_id is not None and material_id is not None:
        replicate_index = raw.replicate or 1
        measurement = (measurement_id, run_id, material_id, replicate_index, "acquired")
        artifact = (stable_id("artifact", f"{measurement_id}:raw"), measurement_id, source_asset_id, "raw")
        measurement_metadata = (
            measurement_id,
            raw.raw_id,
            raw.source_kind,
            decision.material_role,
            raw.control_type,
            raw.group_code,
            raw.sample_id,
            None,
            decision.mapping_status,
            decision.mapping_method,
            1.0 if decision.identity else 0.0,
            decision.review_note,
        )
        match_status = "matched" if decision.identity else "unmatched_spectrum"
        match_candidate = (
            stable_id("match-candidate", f"{raw.dataset_version}:{raw.raw_id}"),
            site_id,
            material_id,
            decision.identity.sample_id if decision.identity else None,
            decision.identity.clinical_event_id if decision.identity else None,
            1.0 if decision.identity else 0.0,
            match_status,
            "canonical_group_sample_v1",
            "solum_label",
            False,
        )
    inventory_status = "derived" if raw.source_kind == "average" else "ready"
    inventory = (
        source_asset_id,
        raw.source_kind,
        inventory_status,
        None,
        run.acquisition_date,
        "unknown",
        raw.source_root,
        decision.solum_label,
        None,
        "unspecified",
        "unspecified",
        None,
        "canonical" if decision.identity else "alias_review",
        "",
    )
    qc_artifact = None
    if decision.material_role == "qc":
        qc_artifact = (
            stable_id("qc-artifact", f"{raw.dataset_version}:{raw.raw_id}"),
            run_id,
            source_asset_id,
            raw.raw_id,
            decision.qc_role or "calibration_control",
            "unreviewed",
        )
    return RawPlan(
        raw, decision, run, source_asset_id, ingest_batch_id, run_id, material_id, measurement_id,
        source_asset, ingest_batch, run_row, run_metadata, material, material_metadata,
        measurement, artifact, measurement_metadata, inventory, match_candidate, qc_artifact,
    )
