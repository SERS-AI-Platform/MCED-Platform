from __future__ import annotations

import sqlite3
from typing import Final

MANIFEST_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS dataset_manifest_specs (
    dataset_manifest_id TEXT PRIMARY KEY REFERENCES dataset_manifests(id),
    preprocessing_version TEXT NOT NULL CHECK(length(trim(preprocessing_version)) > 0),
    feature_schema_version TEXT NOT NULL CHECK(length(trim(feature_schema_version)) > 0),
    decision_policy_version TEXT NOT NULL CHECK(length(trim(decision_policy_version)) > 0),
    qc_policy_version TEXT NOT NULL CHECK(length(trim(qc_policy_version)) > 0),
    qc_rule_version TEXT NOT NULL CHECK(length(trim(qc_rule_version)) > 0),
    inclusion_policy_json TEXT NOT NULL CHECK(json_valid(inclusion_policy_json)),
    exclusion_policy_json TEXT NOT NULL CHECK(json_valid(exclusion_policy_json)),
    provenance_json TEXT NOT NULL CHECK(json_valid(provenance_json))
);

CREATE TABLE IF NOT EXISTS dataset_manifest_exclusions (
    id TEXT PRIMARY KEY,
    dataset_manifest_id TEXT NOT NULL REFERENCES dataset_manifests(id),
    measurement_id TEXT NOT NULL REFERENCES measurements(id),
    sample_label_id TEXT NOT NULL REFERENCES sample_labels(id),
    artifact_sha256 TEXT NOT NULL CHECK(
        length(artifact_sha256) = 64
        AND artifact_sha256 NOT GLOB '*[^0-9A-Fa-f]*'
    ),
    qc_evaluation_id TEXT REFERENCES qc_evaluations(id),
    reason TEXT NOT NULL CHECK(reason IN (
        'measurement_invalid', 'qc_missing', 'qc_outcome_rejected'
    )),
    UNIQUE(dataset_manifest_id, measurement_id)
);

CREATE TABLE IF NOT EXISTS dataset_manifest_label_evidence (
    dataset_manifest_item_id TEXT NOT NULL REFERENCES dataset_manifest_items(id),
    sample_label_evidence_id TEXT NOT NULL REFERENCES sample_label_evidence(id),
    PRIMARY KEY(dataset_manifest_item_id, sample_label_evidence_id)
);

CREATE INDEX IF NOT EXISTS idx_manifest_items_manifest
    ON dataset_manifest_items(dataset_manifest_id);
CREATE INDEX IF NOT EXISTS idx_prediction_manifest
    ON prediction_runs(dataset_manifest_id);

CREATE TRIGGER IF NOT EXISTS immutable_qc_evaluations_update
BEFORE UPDATE ON qc_evaluations
BEGIN
    SELECT RAISE(ABORT, 'qc_evaluations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS immutable_qc_evaluations_delete
BEFORE DELETE ON qc_evaluations
BEGIN
    SELECT RAISE(ABORT, 'qc_evaluations are append-only');
END;
CREATE TRIGGER IF NOT EXISTS immutable_prediction_runs_update
BEFORE UPDATE ON prediction_runs
BEGIN
    SELECT RAISE(ABORT, 'prediction_runs are append-only');
END;
CREATE TRIGGER IF NOT EXISTS immutable_prediction_runs_delete
BEFORE DELETE ON prediction_runs
BEGIN
    SELECT RAISE(ABORT, 'prediction_runs are append-only');
END;
CREATE TRIGGER IF NOT EXISTS immutable_prediction_inputs_update
BEFORE UPDATE ON prediction_input_measurements
BEGIN
    SELECT RAISE(ABORT, 'prediction inputs are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_prediction_inputs_delete
BEFORE DELETE ON prediction_input_measurements
BEGIN
    SELECT RAISE(ABORT, 'prediction inputs are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_frozen_manifests_update
BEFORE UPDATE ON dataset_manifests
WHEN OLD.status = 'frozen'
BEGIN
    SELECT RAISE(ABORT, 'frozen dataset manifests are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_frozen_manifests_delete
BEFORE DELETE ON dataset_manifests
WHEN OLD.status = 'frozen'
BEGIN
    SELECT RAISE(ABORT, 'frozen dataset manifests are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_frozen_manifest_specs_insert
BEFORE INSERT ON dataset_manifest_specs
WHEN EXISTS (
    SELECT 1 FROM dataset_manifests
    WHERE id = NEW.dataset_manifest_id AND status = 'frozen'
)
BEGIN
    SELECT RAISE(ABORT, 'frozen dataset manifests are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_frozen_manifest_items_insert
BEFORE INSERT ON dataset_manifest_items
WHEN EXISTS (
    SELECT 1 FROM dataset_manifests
    WHERE id = NEW.dataset_manifest_id AND status = 'frozen'
)
BEGIN
    SELECT RAISE(ABORT, 'frozen dataset manifests are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_frozen_manifest_exclusions_insert
BEFORE INSERT ON dataset_manifest_exclusions
WHEN EXISTS (
    SELECT 1 FROM dataset_manifests
    WHERE id = NEW.dataset_manifest_id AND status = 'frozen'
)
BEGIN
    SELECT RAISE(ABORT, 'frozen dataset manifests are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_frozen_manifest_evidence_insert
BEFORE INSERT ON dataset_manifest_label_evidence
WHEN EXISTS (
    SELECT 1
    FROM dataset_manifest_items AS item
    JOIN dataset_manifests AS manifest
      ON manifest.id = item.dataset_manifest_id
    WHERE item.id = NEW.dataset_manifest_item_id
      AND manifest.status = 'frozen'
)
BEGIN
    SELECT RAISE(ABORT, 'frozen dataset manifests are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_specs_update
BEFORE UPDATE ON dataset_manifest_specs
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest specs are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_specs_delete
BEFORE DELETE ON dataset_manifest_specs
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest specs are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_items_update
BEFORE UPDATE ON dataset_manifest_items
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest items are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_items_delete
BEFORE DELETE ON dataset_manifest_items
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest items are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_exclusions_update
BEFORE UPDATE ON dataset_manifest_exclusions
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest exclusions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_exclusions_delete
BEFORE DELETE ON dataset_manifest_exclusions
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest exclusions are immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_evidence_update
BEFORE UPDATE ON dataset_manifest_label_evidence
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest label evidence is immutable');
END;
CREATE TRIGGER IF NOT EXISTS immutable_manifest_evidence_delete
BEFORE DELETE ON dataset_manifest_label_evidence
BEGIN
    SELECT RAISE(ABORT, 'dataset manifest label evidence is immutable');
END;
"""

ITEM_COLUMNS: Final = {
    "artifact_sha256": "TEXT",
    "label_definition_version": "TEXT",
    "qc_rule_version": "TEXT",
    "qc_evaluation_version": "TEXT",
    "split_name": "TEXT",
    "fold_index": "INTEGER",
}

QC_COLUMNS: Final = {
    "rule_version": "TEXT",
    "evaluation_version": "TEXT",
}

PREDICTION_COLUMNS: Final = {
    "dataset_manifest_id": "TEXT REFERENCES dataset_manifests(id)",
    "preprocessing_version": "TEXT",
    "feature_schema_version": "TEXT",
    "decision_policy_version": "TEXT",
    "input_set_sha256": "TEXT",
}


def _add_missing_columns(
    connection: sqlite3.Connection,
    table: str,
    columns: dict[str, str],
) -> None:
    existing = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    for name, definition in columns.items():
        if name not in existing:
            connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def initialize_manifest_schema(connection: sqlite3.Connection) -> None:
    _add_missing_columns(connection, "dataset_manifest_items", ITEM_COLUMNS)
    _add_missing_columns(connection, "qc_evaluations", QC_COLUMNS)
    _add_missing_columns(connection, "prediction_runs", PREDICTION_COLUMNS)
    connection.executescript(MANIFEST_SCHEMA)
