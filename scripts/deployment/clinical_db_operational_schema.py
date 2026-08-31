from typing import Final

OPERATIONAL_LINEAGE_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS legacy_prediction_lineage_status (
    id TEXT PRIMARY KEY, legacy_prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    result_sha256 TEXT NOT NULL CHECK(length(result_sha256) = 64),
    status TEXT NOT NULL CHECK(status IN ('pending_context', 'linked')),
    linked_prediction_run_id TEXT REFERENCES prediction_runs(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, linked_at TEXT,
    UNIQUE(legacy_prediction_id, result_sha256),
    CHECK((status = 'pending_context' AND linked_prediction_run_id IS NULL)
        OR (status = 'linked' AND linked_prediction_run_id IS NOT NULL))
);
CREATE TABLE IF NOT EXISTS operational_prediction_run_links (
    id TEXT PRIMARY KEY, legacy_prediction_id INTEGER NOT NULL REFERENCES predictions(id),
    session_id TEXT NOT NULL REFERENCES sessions(id),
    result_sha256 TEXT NOT NULL CHECK(length(result_sha256) = 64),
    prediction_run_id TEXT NOT NULL UNIQUE REFERENCES prediction_runs(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS operational_session_links (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    sample_id TEXT NOT NULL REFERENCES samples(id),
    relation TEXT NOT NULL DEFAULT 'source'
        CHECK(relation IN ('source', 'retest', 'derived')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(session_id, sample_id, relation)
);
"""
