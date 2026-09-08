from __future__ import annotations

import sqlite3
from pathlib import Path

from sers.master_data.schema import initialize_schema


def _legacy_database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE sites (
            id TEXT PRIMARY KEY,
            code TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE source_assets (
            id TEXT PRIMARY KEY,
            site_id TEXT REFERENCES sites(id),
            uri TEXT NOT NULL UNIQUE,
            sha256 TEXT NOT NULL,
            asset_kind TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT 'discovered',
            size_bytes INTEGER,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE source_asset_locations (
            source_asset_id TEXT PRIMARY KEY REFERENCES source_assets(id),
            raw_uri TEXT NOT NULL
        );
        CREATE TABLE subjects (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE subject_source_keys (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL REFERENCES subjects(id),
            site_id TEXT NOT NULL REFERENCES sites(id),
            source_subject_key TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(site_id, source_subject_key)
        );
        CREATE TABLE subject_source_aliases (
            subject_source_key_id TEXT NOT NULL REFERENCES subject_source_keys(id),
            source_asset_id TEXT NOT NULL REFERENCES source_assets(id),
            source_field_name TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY(subject_source_key_id, source_asset_id)
        );
        CREATE TABLE clinical_events (
            id TEXT PRIMARY KEY,
            subject_id TEXT NOT NULL REFERENCES subjects(id),
            site_id TEXT NOT NULL REFERENCES sites(id),
            ingest_batch_id TEXT,
            source_asset_id TEXT REFERENCES source_assets(id),
            source_row_locator TEXT,
            event_type TEXT NOT NULL,
            occurred_at TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE clinical_observations (
            id TEXT PRIMARY KEY,
            clinical_event_id TEXT NOT NULL REFERENCES clinical_events(id),
            code TEXT NOT NULL,
            source_field_name TEXT,
            canonical_code TEXT,
            raw_value TEXT,
            value_kind TEXT NOT NULL,
            text_value TEXT,
            numeric_value REAL,
            date_value TEXT,
            unit TEXT,
            normalization_status TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE spectrum_inventory_records (
            source_uri TEXT PRIMARY KEY,
            source_asset_id TEXT NOT NULL REFERENCES source_assets(id),
            source_kind TEXT NOT NULL,
            status TEXT NOT NULL,
            reason_code TEXT,
            acquisition_date TEXT NOT NULL,
            instrument_key TEXT NOT NULL,
            root_key TEXT NOT NULL,
            canonical_source_code TEXT,
            preparation TEXT,
            identity_status TEXT,
            alias_candidates TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE match_candidates (
            id TEXT PRIMARY KEY,
            site_id TEXT NOT NULL REFERENCES sites(id),
            analytical_material_id TEXT,
            sample_id TEXT,
            clinical_event_id TEXT REFERENCES clinical_events(id),
            score REAL NOT NULL,
            status TEXT NOT NULL,
            rule_version TEXT NOT NULL,
            identity_basis TEXT NOT NULL CHECK(identity_basis IN (
                'site_subject_key', 'site_sample_code', 'manual_review', 'inventory'
            )),
            alias_review INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE measurements (
            id TEXT PRIMARY KEY,
            measurement_run_id TEXT NOT NULL,
            analytical_material_id TEXT NOT NULL,
            replicate_index INTEGER NOT NULL,
            status TEXT NOT NULL DEFAULT 'acquired',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE qc_evaluations (
            id TEXT PRIMARY KEY,
            measurement_id TEXT NOT NULL REFERENCES measurements(id),
            evaluator_version TEXT NOT NULL,
            rule_version TEXT,
            evaluation_version TEXT,
            outcome TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            evaluated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE operational_session_links (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES sessions(id),
            sample_id TEXT NOT NULL,
            relation TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );

        INSERT INTO sites (id, code, name) VALUES ('site-1', 'SITE', 'Site');
        INSERT INTO source_assets (
            id, site_id, uri, sha256, asset_kind
        ) VALUES (
            'asset-1', 'site-1', 'file:///source.csv',
            'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',
            'clinical'
        );
        INSERT INTO source_asset_locations (
            source_asset_id, raw_uri
        ) VALUES ('asset-1', '/raw/aa/aa');
        INSERT INTO subjects (id) VALUES ('subject-1');
        INSERT INTO subject_source_keys (
            id, subject_id, site_id, source_subject_key
        ) VALUES ('key-1', 'subject-1', 'site-1', 'SOLUM-001');
        INSERT INTO subject_source_aliases (
            subject_source_key_id, source_asset_id, source_field_name
        ) VALUES ('key-1', 'asset-1', 'SoluM Label');
        INSERT INTO clinical_events (
            id, subject_id, site_id, source_asset_id, source_row_locator, event_type
        ) VALUES ('event-1', 'subject-1', 'site-1', 'asset-1', 'row:1', 'lab');
        INSERT INTO clinical_observations (
            id, clinical_event_id, code, source_field_name, raw_value,
            value_kind, text_value
        ) VALUES (
            'observation-patient', 'event-1', 'patient_id',
            '제공자:제공자bCODE', 'PATIENT-001', 'text', 'PATIENT-001'
        );
        INSERT INTO clinical_observations (
            id, clinical_event_id, code, source_field_name, raw_value,
            value_kind, text_value
        ) VALUES (
            'observation-solum', 'event-1', 'solum_label',
            'SoluM Label', 'SOLUM-001', 'text', 'SOLUM-001'
        );
        INSERT INTO match_candidates (
            id, site_id, clinical_event_id, score, status,
            rule_version, identity_basis
        ) VALUES (
            'match-1', 'site-1', 'event-1', 0,
            'unmatched_clinical', 'legacy-v1', 'site_subject_key'
        );
        INSERT INTO spectrum_inventory_records (
            source_uri, source_asset_id, source_kind, status,
            acquisition_date, instrument_key, root_key
        ) VALUES (
            'file:///source.csv', 'asset-1', 'remeasurement', 'ready',
            '2026-07-29', 'Thermo', 'root'
        );
        INSERT INTO measurements (
            id, measurement_run_id, analytical_material_id, replicate_index
        ) VALUES ('measurement-1', 'run-1', 'material-1', 1);
        INSERT INTO qc_evaluations (
            id, measurement_id, evaluator_version, rule_version,
            evaluation_version, outcome, metrics_json
        ) VALUES (
            'qc-1', 'measurement-1', 'engine-v1', 'rules-v1',
            'engine-v1', 'pass', '{}'
        );
        """
    )
    return connection


def test_initialize_schema_merges_one_to_one_legacy_tables(tmp_path: Path) -> None:
    # Given: a legacy master database with four redundant representations.
    with _legacy_database(tmp_path / "legacy.db") as connection:
        # When: the current master schema is initialized.
        initialize_schema(connection)

        # Then: values survive in their owning rows and redundant tables disappear.
        assert tuple(
            connection.execute(
                """SELECT id, site_id, patient_id
                   FROM subjects"""
            ).fetchone()
        ) == ("subject-1", "site-1", "PATIENT-001")
        assert connection.execute(
            "SELECT raw_uri FROM source_assets WHERE id = 'asset-1'"
        ).fetchone()[0] == "/raw/aa/aa"
        assert connection.execute(
            "SELECT subject_id FROM subject_source_aliases"
        ).fetchone()[0] == "subject-1"
        assert connection.execute(
            "SELECT solum_label FROM samples"
        ).fetchone()[0] == "SOLUM-001"
        assert connection.execute(
            "SELECT identity_basis FROM match_candidates"
        ).fetchone()[0] == "legacy_site_subject_key"
        assert connection.execute(
            "SELECT source_asset_id FROM spectrum_inventory_records"
        ).fetchone()[0] == "asset-1"
        assert tuple(
            connection.execute(
                """SELECT rule_version, evaluation_version
                   FROM qc_evaluations"""
            ).fetchone()
        ) == ("rules-v1", "engine-v1")

        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "subject_source_keys" not in tables
        assert "source_asset_locations" not in tables
        assert "operational_session_links" not in tables
        assert "evaluator_version" not in {
            row[1] for row in connection.execute("PRAGMA table_info(qc_evaluations)")
        }
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 8
