from __future__ import annotations

import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class AmbiguousSubjectIdentityError(RuntimeError):
    subject_count: int
    key_count: int

    def __str__(self) -> str:
        return (
            "subjects can be simplified only when every subject has exactly "
            "one site-scoped source key"
        )


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


def _merge_subject_identity(connection: sqlite3.Connection) -> None:
    tables = _tables(connection)
    if "subject_source_keys" not in tables:
        return
    subject_count = int(
        connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
    )
    key_count, keyed_subject_count = connection.execute(
        "SELECT COUNT(*), COUNT(DISTINCT subject_id) FROM subject_source_keys"
    ).fetchone()
    if subject_count != key_count or subject_count != keyed_subject_count:
        raise AmbiguousSubjectIdentityError(subject_count, int(key_count))
    connection.execute(
        """CREATE TABLE subjects_merged (
               id TEXT PRIMARY KEY,
               site_id TEXT NOT NULL REFERENCES sites(id),
               source_subject_key TEXT NOT NULL CHECK(length(source_subject_key) > 0),
               created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
               UNIQUE(site_id, source_subject_key)
           )"""
    )
    connection.execute(
        """INSERT INTO subjects_merged (
               id, site_id, source_subject_key, created_at
           )
           SELECT s.id, k.site_id, k.source_subject_key, s.created_at
           FROM subjects AS s
           JOIN subject_source_keys AS k ON k.subject_id = s.id"""
    )
    if "subject_source_aliases" in tables:
        connection.execute(
            """CREATE TABLE subject_source_aliases_merged (
                   subject_id TEXT NOT NULL REFERENCES subjects_merged(id),
                   source_asset_id TEXT NOT NULL REFERENCES source_assets(id),
                   source_field_name TEXT NOT NULL CHECK(length(source_field_name) > 0),
                   created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                   PRIMARY KEY(subject_id, source_asset_id)
               )"""
        )
        connection.execute(
            """INSERT INTO subject_source_aliases_merged (
                   subject_id, source_asset_id, source_field_name, created_at
               )
               SELECT k.subject_id, a.source_asset_id,
                      a.source_field_name, a.created_at
               FROM subject_source_aliases AS a
               JOIN subject_source_keys AS k
                 ON k.id = a.subject_source_key_id"""
        )
        connection.execute("DROP TABLE subject_source_aliases")
    connection.execute("DROP TABLE subject_source_keys")
    connection.execute("DROP TABLE subjects")
    connection.execute("ALTER TABLE subjects_merged RENAME TO subjects")
    if "subject_source_aliases" in tables:
        connection.execute(
            "ALTER TABLE subject_source_aliases_merged RENAME TO subject_source_aliases"
        )


def _merge_asset_location(connection: sqlite3.Connection) -> None:
    tables = _tables(connection)
    if "source_assets" not in tables:
        return
    if "raw_uri" not in _columns(connection, "source_assets"):
        connection.execute(
            """ALTER TABLE source_assets ADD COLUMN raw_uri TEXT
               CHECK(raw_uri IS NULL OR length(trim(raw_uri)) > 0)"""
        )
    if "source_asset_locations" in tables:
        connection.execute(
            """UPDATE source_assets
               SET raw_uri = (
                   SELECT location.raw_uri
                   FROM source_asset_locations AS location
                   WHERE location.source_asset_id = source_assets.id
               )
               WHERE raw_uri IS NULL"""
        )
        connection.execute("DROP TABLE source_asset_locations")


def _merge_spectrum_uri(connection: sqlite3.Connection) -> None:
    if "spectrum_inventory_records" not in _tables(connection):
        return
    if "source_uri" not in _columns(connection, "spectrum_inventory_records"):
        return
    connection.execute(
        """CREATE TABLE spectrum_inventory_records_merged (
               source_asset_id TEXT PRIMARY KEY REFERENCES source_assets(id),
               source_kind TEXT NOT NULL,
               status TEXT NOT NULL CHECK(
                   status IN ('ready', 'derived', 'excluded', 'quarantined')
               ),
               reason_code TEXT,
               acquisition_date TEXT NOT NULL,
               instrument_key TEXT NOT NULL,
               root_key TEXT NOT NULL,
               canonical_source_code TEXT,
               preparation TEXT,
               identity_status TEXT,
               alias_candidates TEXT NOT NULL DEFAULT ''
           )"""
    )
    connection.execute(
        """INSERT INTO spectrum_inventory_records_merged (
               source_asset_id, source_kind, status, reason_code,
               acquisition_date, instrument_key, root_key,
               canonical_source_code, preparation, identity_status,
               alias_candidates
           )
           SELECT source_asset_id, source_kind, status, reason_code,
                  acquisition_date, instrument_key, root_key,
                  canonical_source_code, preparation, identity_status,
                  alias_candidates
           FROM spectrum_inventory_records"""
    )
    connection.execute("DROP TABLE spectrum_inventory_records")
    connection.execute(
        """ALTER TABLE spectrum_inventory_records_merged
           RENAME TO spectrum_inventory_records"""
    )


def _merge_qc_version(connection: sqlite3.Connection) -> None:
    if "qc_evaluations" not in _tables(connection):
        return
    if "evaluator_version" not in _columns(connection, "qc_evaluations"):
        return
    connection.execute(
        """CREATE TABLE qc_evaluations_merged (
               id TEXT PRIMARY KEY,
               measurement_id TEXT NOT NULL REFERENCES measurements(id),
               rule_version TEXT NOT NULL CHECK(length(trim(rule_version)) > 0),
               evaluation_version TEXT NOT NULL
                   CHECK(length(trim(evaluation_version)) > 0),
               outcome TEXT NOT NULL CHECK(outcome IN ('pass', 'fail', 'review')),
               metrics_json TEXT NOT NULL CHECK(json_valid(metrics_json)),
               evaluated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    connection.execute(
        """INSERT INTO qc_evaluations_merged (
               id, measurement_id, rule_version, evaluation_version,
               outcome, metrics_json, evaluated_at
           )
           SELECT id, measurement_id,
                  COALESCE(rule_version, evaluator_version),
                  COALESCE(evaluation_version, evaluator_version),
                  outcome, metrics_json, evaluated_at
           FROM qc_evaluations"""
    )
    connection.execute("DROP TABLE qc_evaluations")
    connection.execute(
        "ALTER TABLE qc_evaluations_merged RENAME TO qc_evaluations"
    )


def simplify_legacy_schema(
    connection: sqlite3.Connection,
    *,
    keep_operational_links: bool,
) -> None:
    _merge_asset_location(connection)
    _merge_subject_identity(connection)
    _merge_spectrum_uri(connection)
    _merge_qc_version(connection)
    if not keep_operational_links and "operational_session_links" in _tables(connection):
        connection.execute("DROP TABLE operational_session_links")
