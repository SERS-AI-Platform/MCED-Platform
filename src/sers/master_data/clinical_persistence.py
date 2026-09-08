from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from uuid import UUID, uuid5

from .clinical_types import ClinicalRow, ClinicalSource
from .repository import resolve_or_create_subject, upsert_site
from .spectrum_types import StoredRaw
from .types import SiteDraft, SiteId, SubjectId, SubjectIdentity

_ID_NAMESPACE = UUID("e3942d80-815e-4662-a0f5-ecae44a95bc4")


def _stable_id(kind: str, key: str) -> str:
    return str(uuid5(_ID_NAMESPACE, f"{kind}:{key}"))


@dataclass(frozen=True, slots=True)
class ConflictingSolumLabelError(ValueError):
    site_id: SiteId

    def __str__(self) -> str:
        return "solum_label is already linked to another subject at this site"


def _subject(
    connection: sqlite3.Connection,
    site_id: SiteId,
    row: ClinicalRow,
) -> SubjectIdentity:
    if row.patient_id is not None:
        return resolve_or_create_subject(connection, site_id, row.patient_id)
    assert row.solum_label is not None
    subject_id = SubjectId(
        _stable_id("subject-without-patient-id", f"{site_id}:{row.solum_label}")
    )
    connection.execute(
        """INSERT OR IGNORE INTO subjects (id, site_id, patient_id)
           VALUES (?, ?, NULL)""",
        (subject_id, site_id),
    )
    return SubjectIdentity(subject_id, site_id, None)


def _sample(
    connection: sqlite3.Connection,
    identity: SubjectIdentity,
    solum_label: str,
) -> None:
    existing = connection.execute(
        """SELECT subject_id FROM samples
           WHERE site_id = ? AND solum_label = ?""",
        (identity.site_id, solum_label),
    ).fetchone()
    if existing is not None:
        if str(existing[0]) != identity.subject_id:
            raise ConflictingSolumLabelError(identity.site_id)
        return
    connection.execute(
        """INSERT INTO samples (
               id, subject_id, site_id, solum_label, sample_type
           ) VALUES (?, ?, ?, ?, 'urine')""",
        (
            _stable_id("clinical-solum-sample", f"{identity.site_id}:{solum_label}"),
            identity.subject_id,
            identity.site_id,
            solum_label,
        ),
    )


def persist_clinical_source(
    connection: sqlite3.Connection,
    source: ClinicalSource,
    stored: StoredRaw,
    rows: tuple[ClinicalRow, ...],
) -> None:
    source_uri = source.path.resolve().as_uri()
    site = upsert_site(
        connection,
        SiteDraft(code=source.site_code, name=source.site_name),
    )
    asset_id = _stable_id("source-asset", source_uri)
    connection.execute(
        """INSERT OR IGNORE INTO source_assets (
               id, site_id, uri, sha256, asset_kind, state, size_bytes, raw_uri
           ) VALUES (?, ?, ?, ?, 'clinical', 'discovered', ?, ?)""",
        (
            asset_id,
            site.id,
            source_uri,
            stored.sha256,
            stored.size_bytes,
            stored.raw_uri,
        ),
    )
    connection.execute(
        """INSERT OR IGNORE INTO clinical_source_metadata (
               source_asset_id, protocol_code, source_group, canonical_alias,
               identity_status, source_version
           ) VALUES (?, ?, ?, ?, ?, ?)""",
        (
            asset_id,
            source.protocol_code,
            source.source_group,
            source.canonical_alias,
            source.identity_status.value,
            source.source_version.value,
        ),
    )
    batch_id = _stable_id("ingest-batch", asset_id)
    connection.execute(
        """INSERT OR IGNORE INTO ingest_batches (
               id, source_asset_id, status, parser_name
           ) VALUES (?, ?, 'started', 'clinical-tabular-v1')""",
        (batch_id, asset_id),
    )
    for row in rows:
        identity = _subject(connection, site.id, row)
        if row.source_patient_id_field is not None:
            connection.execute(
                """INSERT OR IGNORE INTO subject_source_aliases (
                       subject_id, source_asset_id, source_field_name
                   ) VALUES (?, ?, ?)""",
                (identity.subject_id, asset_id, row.source_patient_id_field),
            )
        if row.solum_label is not None:
            _sample(connection, identity, row.solum_label)
        event_id = _stable_id("clinical-event", f"{asset_id}:{row.locator}")
        connection.execute(
            """INSERT OR IGNORE INTO clinical_events (
                   id, subject_id, site_id, ingest_batch_id, source_asset_id,
                   source_row_locator, event_type, occurred_at
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id,
                identity.subject_id,
                site.id,
                batch_id,
                asset_id,
                row.locator,
                row.event_type,
                row.occurred_at,
            ),
        )
        for observation in row.observations:
            value_kind = (
                "numeric"
                if observation.numeric_value is not None
                else "date"
                if observation.date_value is not None
                else "text"
            )
            code = observation.canonical_code or observation.source_field_name
            connection.execute(
                """INSERT OR IGNORE INTO clinical_observations (
                       id, clinical_event_id, code, source_field_name,
                       canonical_code, raw_value, value_kind, text_value,
                       numeric_value, date_value, unit, normalization_status
                   ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    _stable_id(
                        "clinical-observation",
                        f"{event_id}:{observation.source_field_name}",
                    ),
                    event_id,
                    code,
                    observation.source_field_name,
                    observation.canonical_code,
                    observation.raw_value,
                    value_kind,
                    (
                        observation.date_value
                        if value_kind == "date"
                        else observation.text_value
                    ),
                    observation.numeric_value,
                    observation.date_value,
                    observation.unit,
                    observation.normalization_status,
                ),
            )
    connection.execute(
        """UPDATE ingest_batches
           SET status = 'completed', completed_at = CURRENT_TIMESTAMP
           WHERE id = ?""",
        (batch_id,),
    )
    connection.execute(
        "UPDATE source_assets SET state = 'ingested' WHERE id = ?",
        (asset_id,),
    )
