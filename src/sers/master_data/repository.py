from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from string import hexdigits
from uuid import uuid4

from .types import (
    AnalyticalMaterialDraft,
    AnalyticalMaterialId,
    IngestBatchDraft,
    IngestBatchId,
    MeasurementDraft,
    MeasurementId,
    MeasurementRunDraft,
    MeasurementRunId,
    SampleDraft,
    SampleId,
    Site,
    SiteDraft,
    SiteId,
    SourceAssetDraft,
    SourceAssetId,
    SubjectId,
    SubjectIdentity,
)


@dataclass(frozen=True, slots=True)
class InvalidPatientIdError(ValueError):
    patient_id: str

    def __str__(self) -> str:
        return "patient_id must be non-empty"


@dataclass(frozen=True, slots=True)
class InvalidSha256Error(ValueError):
    sha256: str

    def __str__(self) -> str:
        return "sha256 must contain exactly 64 hexadecimal characters"


def upsert_site(connection: sqlite3.Connection, draft: SiteDraft) -> Site:
    site_id = SiteId(str(uuid4()))
    row = connection.execute(
        """INSERT INTO sites (id, code, name, status)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(code) DO UPDATE
           SET name = excluded.name, status = excluded.status
           RETURNING id, code, name, status""",
        (site_id, draft.code, draft.name, draft.status),
    ).fetchone()
    assert row is not None
    return Site(id=SiteId(row[0]), code=row[1], name=row[2], status=row[3])


def get_site(connection: sqlite3.Connection, site_id: SiteId) -> Site | None:
    row = connection.execute(
        "SELECT id, code, name, status FROM sites WHERE id = ?",
        (site_id,),
    ).fetchone()
    if row is None:
        return None
    return Site(id=SiteId(row[0]), code=row[1], name=row[2], status=row[3])


def resolve_or_create_subject(
    connection: sqlite3.Connection,
    site_id: SiteId,
    patient_id: str,
) -> SubjectIdentity:
    if not patient_id:
        raise InvalidPatientIdError(patient_id)
    row = connection.execute(
        """SELECT id
           FROM subjects
           WHERE site_id = ? AND patient_id = ?""",
        (site_id, patient_id),
    ).fetchone()
    if row is not None:
        return SubjectIdentity(
            subject_id=SubjectId(row[0]),
            site_id=site_id,
            patient_id=patient_id,
        )
    subject_id = SubjectId(str(uuid4()))
    connection.execute(
        """INSERT INTO subjects (id, site_id, patient_id)
           VALUES (?, ?, ?)""",
        (subject_id, site_id, patient_id),
    )
    return SubjectIdentity(
        subject_id=subject_id,
        site_id=site_id,
        patient_id=patient_id,
    )


def create_source_asset(
    connection: sqlite3.Connection,
    draft: SourceAssetDraft,
) -> SourceAssetId:
    if len(draft.sha256) != 64 or any(character not in hexdigits for character in draft.sha256):
        raise InvalidSha256Error(draft.sha256)
    source_asset_id = SourceAssetId(str(uuid4()))
    connection.execute(
        """INSERT INTO source_assets (
               id, site_id, uri, sha256, asset_kind, size_bytes, raw_uri
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            source_asset_id,
            draft.site_id,
            draft.uri,
            draft.sha256,
            draft.asset_kind,
            draft.size_bytes,
            draft.raw_uri,
        ),
    )
    return source_asset_id


def create_ingest_batch(
    connection: sqlite3.Connection,
    draft: IngestBatchDraft,
) -> IngestBatchId:
    ingest_batch_id = IngestBatchId(str(uuid4()))
    connection.execute(
        """INSERT INTO ingest_batches (id, source_asset_id, parser_name)
           VALUES (?, ?, ?)""",
        (ingest_batch_id, draft.source_asset_id, draft.parser_name),
    )
    return ingest_batch_id


def create_sample(connection: sqlite3.Connection, draft: SampleDraft) -> SampleId:
    sample_id = SampleId(str(uuid4()))
    connection.execute(
        """INSERT INTO samples (
               id, subject_id, site_id, solum_label, sample_type, collected_at
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            sample_id,
            draft.subject_id,
            draft.site_id,
            draft.solum_label,
            draft.sample_type,
            draft.collected_at,
        ),
    )
    return sample_id


def create_analytical_material(
    connection: sqlite3.Connection,
    draft: AnalyticalMaterialDraft,
) -> AnalyticalMaterialId:
    material_id = AnalyticalMaterialId(str(uuid4()))
    connection.execute(
        """INSERT INTO analytical_materials (
               id, sample_id, parent_material_id, material_type
           ) VALUES (?, ?, ?, ?)""",
        (
            material_id,
            draft.sample_id,
            draft.parent_material_id,
            draft.material_type,
        ),
    )
    return material_id


def create_measurement_run(
    connection: sqlite3.Connection,
    draft: MeasurementRunDraft,
) -> MeasurementRunId:
    run_id = MeasurementRunId(str(uuid4()))
    connection.execute(
        """INSERT INTO measurement_runs (
               id, site_id, ingest_batch_id, instrument_key, acquired_at
           ) VALUES (?, ?, ?, ?, ?)""",
        (
            run_id,
            draft.site_id,
            draft.ingest_batch_id,
            draft.instrument_key,
            draft.acquired_at,
        ),
    )
    return run_id


def create_measurement(
    connection: sqlite3.Connection,
    draft: MeasurementDraft,
) -> MeasurementId:
    measurement_id = MeasurementId(str(uuid4()))
    connection.execute(
        """INSERT INTO measurements (
               id, measurement_run_id, analytical_material_id, replicate_index
           ) VALUES (?, ?, ?, ?)""",
        (
            measurement_id,
            draft.measurement_run_id,
            draft.analytical_material_id,
            draft.replicate_index,
        ),
    )
    return measurement_id
