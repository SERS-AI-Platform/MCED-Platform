from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass

_SAMPLE_NAMESPACE = uuid.UUID("d82d683a-f203-47b4-a328-fd8712d13641")
_NORMALIZED_FIELD = (
    "lower(replace(replace(trim(COALESCE(source_field_name, '')), ' ', ''), '_', ''))"
)


@dataclass(frozen=True, slots=True)
class AmbiguousSolumLabelError(RuntimeError):
    site_id: str
    solum_label: str

    def __str__(self) -> str:
        return "solum_label resolves to more than one subject within a site"


def _tables(connection: sqlite3.Connection) -> frozenset[str]:
    return frozenset(
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    )


def _columns(connection: sqlite3.Connection, table: str) -> frozenset[str]:
    return frozenset(
        str(row[1]) for row in connection.execute(f"PRAGMA table_info({table})")
    )


def _migrate_subjects(connection: sqlite3.Connection) -> None:
    if "subjects" not in _tables(connection):
        return
    if "source_subject_key" not in _columns(connection, "subjects"):
        return
    connection.execute(
        """CREATE TABLE subjects_v7 (
               id TEXT PRIMARY KEY,
               site_id TEXT NOT NULL REFERENCES sites(id),
               patient_id TEXT CHECK(patient_id IS NULL OR length(patient_id) > 0),
               created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
               UNIQUE(site_id, patient_id)
           )"""
    )
    connection.execute(
        f"""INSERT INTO subjects_v7 (id, site_id, patient_id, created_at)
            WITH solum_subjects AS (
                SELECT DISTINCT subject_id
                FROM subject_source_aliases
                WHERE {_NORMALIZED_FIELD} = 'solumlabel'
            ),
            hospital_patient_ids AS (
                SELECT event.subject_id,
                       MIN(trim(COALESCE(
                           NULLIF(observation.text_value, ''),
                           observation.raw_value
                       ))) AS patient_id
                FROM clinical_events AS event
                JOIN clinical_observations AS observation
                  ON observation.clinical_event_id = event.id
                WHERE lower(replace(replace(
                          trim(COALESCE(observation.source_field_name, '')),
                          ' ', ''
                      ), '_', '')) = '제공자:제공자bcode'
                GROUP BY event.subject_id
                HAVING COUNT(DISTINCT trim(COALESCE(
                           NULLIF(observation.text_value, ''),
                           observation.raw_value
                       ))) = 1
            )
            SELECT subject.id, subject.site_id,
                   CASE WHEN solum.subject_id IS NULL
                        THEN subject.source_subject_key
                        ELSE hospital.patient_id
                   END,
                   subject.created_at
            FROM subjects AS subject
            LEFT JOIN solum_subjects AS solum ON solum.subject_id = subject.id
            LEFT JOIN hospital_patient_ids AS hospital
              ON hospital.subject_id = subject.id"""
    )
    connection.execute("DROP TABLE subjects")
    connection.execute("ALTER TABLE subjects_v7 RENAME TO subjects")


def _migrate_samples(connection: sqlite3.Connection) -> None:
    if "samples" not in _tables(connection):
        return
    if "source_sample_key" not in _columns(connection, "samples"):
        return
    connection.execute(
        """CREATE TABLE samples_v7 (
               id TEXT PRIMARY KEY,
               subject_id TEXT NOT NULL REFERENCES subjects(id),
               site_id TEXT NOT NULL REFERENCES sites(id),
               solum_label TEXT,
               sample_type TEXT NOT NULL
                   CHECK(sample_type IN ('urine', 'serum', 'plasma', 'other')),
               collected_at TEXT,
               created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
               UNIQUE(site_id, solum_label)
           )"""
    )
    connection.execute(
        """INSERT INTO samples_v7 (
               id, subject_id, site_id, solum_label,
               sample_type, collected_at, created_at
           )
           SELECT id, subject_id, site_id, source_sample_key,
                  sample_type, collected_at, created_at
           FROM samples"""
    )
    connection.execute("DROP TABLE samples")
    connection.execute("ALTER TABLE samples_v7 RENAME TO samples")


def _clinical_solum_labels(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str, str], ...]:
    if not {"clinical_events", "clinical_observations"} <= _tables(connection):
        return ()
    rows = connection.execute(
        f"""SELECT event.site_id, event.subject_id,
                   trim(COALESCE(
                       NULLIF(observation.text_value, ''),
                       observation.raw_value
                   )) AS solum_label
            FROM clinical_events AS event
            JOIN clinical_observations AS observation
              ON observation.clinical_event_id = event.id
            WHERE {_NORMALIZED_FIELD.replace("source_field_name", "observation.source_field_name")}
                  = 'solumlabel'
              AND trim(COALESCE(
                      NULLIF(observation.text_value, ''),
                      observation.raw_value
                  )) <> ''
            GROUP BY event.site_id, event.subject_id, solum_label
            ORDER BY event.site_id, solum_label"""
    ).fetchall()
    return tuple((str(row[0]), str(row[1]), str(row[2])) for row in rows)


def _backfill_samples(connection: sqlite3.Connection) -> None:
    labels = _clinical_solum_labels(connection)
    owners: dict[tuple[str, str], str] = {}
    for site_id, subject_id, solum_label in labels:
        key = (site_id, solum_label)
        owner = owners.setdefault(key, subject_id)
        if owner != subject_id:
            raise AmbiguousSolumLabelError(site_id, solum_label)
        existing = connection.execute(
            """SELECT subject_id FROM samples
               WHERE site_id = ? AND solum_label = ?""",
            key,
        ).fetchone()
        if existing is not None and str(existing[0]) != subject_id:
            raise AmbiguousSolumLabelError(site_id, solum_label)
        sample_id = str(uuid.uuid5(_SAMPLE_NAMESPACE, f"{site_id}:{solum_label}"))
        connection.execute(
            """INSERT OR IGNORE INTO samples (
                   id, subject_id, site_id, solum_label, sample_type
               ) VALUES (?, ?, ?, ?, 'urine')""",
            (sample_id, subject_id, site_id, solum_label),
        )


def _migrate_match_candidates(connection: sqlite3.Connection) -> None:
    if "match_candidates" not in _tables(connection):
        return
    table_sql = connection.execute(
        """SELECT sql FROM sqlite_master
           WHERE type = 'table' AND name = 'match_candidates'"""
    ).fetchone()
    if table_sql is None or "solum_label" in str(table_sql[0]):
        return
    connection.execute(
        """CREATE TABLE match_candidates_v7 (
               id TEXT PRIMARY KEY,
               site_id TEXT NOT NULL REFERENCES sites(id),
               analytical_material_id TEXT REFERENCES analytical_materials(id),
               sample_id TEXT REFERENCES samples(id),
               clinical_event_id TEXT REFERENCES clinical_events(id),
               score REAL NOT NULL CHECK(score >= 0 AND score <= 1),
               status TEXT NOT NULL CHECK(status IN (
                   'matched', 'unmatched_clinical', 'unmatched_spectrum',
                   'ambiguous', 'duplicate'
               )),
               rule_version TEXT NOT NULL CHECK(length(trim(rule_version)) > 0),
               identity_basis TEXT NOT NULL CHECK(identity_basis IN (
                   'solum_label', 'manual_review', 'inventory',
                   'legacy_site_subject_key', 'legacy_site_sample_code'
               )),
               alias_review INTEGER NOT NULL DEFAULT 0
                   CHECK(alias_review IN (0, 1)),
               created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
               CHECK(
                   (status = 'matched' AND analytical_material_id IS NOT NULL
                       AND sample_id IS NOT NULL AND clinical_event_id IS NOT NULL)
                   OR (status = 'unmatched_clinical'
                       AND analytical_material_id IS NULL
                       AND clinical_event_id IS NOT NULL)
                   OR (status = 'unmatched_spectrum'
                       AND analytical_material_id IS NOT NULL
                       AND clinical_event_id IS NULL)
                   OR status IN ('ambiguous', 'duplicate')
               )
           )"""
    )
    connection.execute(
        """INSERT INTO match_candidates_v7 (
               id, site_id, analytical_material_id, sample_id,
               clinical_event_id, score, status, rule_version,
               identity_basis, alias_review, created_at
           )
           SELECT id, site_id, analytical_material_id, sample_id,
                  clinical_event_id, score, status, rule_version,
                  CASE identity_basis
                      WHEN 'site_subject_key' THEN 'legacy_site_subject_key'
                      WHEN 'site_sample_code' THEN 'legacy_site_sample_code'
                      ELSE identity_basis
                  END,
                  alias_review, created_at
           FROM match_candidates"""
    )
    connection.execute("DROP TABLE match_candidates")
    connection.execute(
        "ALTER TABLE match_candidates_v7 RENAME TO match_candidates"
    )


def migrate_identifiers_v7(connection: sqlite3.Connection) -> None:
    _migrate_subjects(connection)
    _migrate_samples(connection)
    if "samples" in _tables(connection):
        _backfill_samples(connection)
    _migrate_match_candidates(connection)
