from typing import Final

CORE_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS sites (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL UNIQUE CHECK(length(trim(code)) > 0),
    name TEXT NOT NULL CHECK(length(trim(name)) > 0),
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'inactive')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS source_assets (
    id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES sites(id),
    uri TEXT NOT NULL UNIQUE CHECK(length(trim(uri)) > 0),
    sha256 TEXT NOT NULL CHECK(
        length(sha256) = 64 AND sha256 NOT GLOB '*[^0-9A-Fa-f]*'
    ),
    asset_kind TEXT NOT NULL CHECK(asset_kind IN ('clinical', 'spectrum', 'manifest', 'other')),
    state TEXT NOT NULL DEFAULT 'discovered'
        CHECK(state IN ('discovered', 'ingested', 'failed')),
    size_bytes INTEGER CHECK(size_bytes IS NULL OR size_bytes >= 0),
    raw_uri TEXT CHECK(raw_uri IS NULL OR length(trim(raw_uri)) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingest_batches (
    id TEXT PRIMARY KEY,
    source_asset_id TEXT NOT NULL REFERENCES source_assets(id),
    status TEXT NOT NULL DEFAULT 'started'
        CHECK(status IN ('started', 'completed', 'failed')),
    parser_name TEXT NOT NULL CHECK(length(trim(parser_name)) > 0),
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS subjects (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES sites(id),
    patient_id TEXT CHECK(patient_id IS NULL OR length(patient_id) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(site_id, patient_id)
);

CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    solum_label TEXT,
    sample_type TEXT NOT NULL CHECK(sample_type IN ('urine', 'serum', 'plasma', 'other')),
    collected_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(site_id, solum_label)
);

CREATE TABLE IF NOT EXISTS sample_crosswalk_sources (
    sample_id TEXT NOT NULL REFERENCES samples(id),
    source_asset_id TEXT NOT NULL REFERENCES source_assets(id),
    source_row_locator TEXT NOT NULL CHECK(length(trim(source_row_locator)) > 0),
    identity_field TEXT NOT NULL CHECK(length(trim(identity_field)) > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY(sample_id, source_asset_id),
    UNIQUE(source_asset_id, source_row_locator)
);

CREATE TABLE IF NOT EXISTS clinical_events (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES subjects(id),
    site_id TEXT NOT NULL REFERENCES sites(id),
    ingest_batch_id TEXT REFERENCES ingest_batches(id),
    source_asset_id TEXT REFERENCES source_assets(id),
    source_row_locator TEXT,
    event_type TEXT NOT NULL
        CHECK(event_type IN ('collection', 'diagnosis', 'procedure', 'lab', 'other')),
    occurred_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clinical_observations (
    id TEXT PRIMARY KEY,
    clinical_event_id TEXT NOT NULL REFERENCES clinical_events(id),
    code TEXT NOT NULL CHECK(length(trim(code)) > 0),
    source_field_name TEXT,
    canonical_code TEXT,
    raw_value TEXT,
    value_kind TEXT NOT NULL CHECK(value_kind IN ('text', 'numeric', 'date')),
    text_value TEXT,
    numeric_value REAL,
    date_value TEXT,
    unit TEXT,
    normalization_status TEXT CHECK(
        normalization_status IS NULL
        OR normalization_status IN ('normalized', 'raw', 'unmapped')
    ),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(
        (value_kind = 'text' AND text_value IS NOT NULL AND numeric_value IS NULL)
        OR (value_kind = 'date' AND text_value IS NOT NULL AND numeric_value IS NULL)
        OR (value_kind = 'numeric' AND numeric_value IS NOT NULL AND text_value IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS analytical_materials (
    id TEXT PRIMARY KEY,
    sample_id TEXT REFERENCES samples(id),
    parent_material_id TEXT REFERENCES analytical_materials(id),
    material_type TEXT NOT NULL
        CHECK(material_type IN ('primary', 'aliquot', 'extract', 'average')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS measurement_runs (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES sites(id),
    ingest_batch_id TEXT REFERENCES ingest_batches(id),
    instrument_key TEXT NOT NULL CHECK(length(trim(instrument_key)) > 0),
    status TEXT NOT NULL DEFAULT 'acquired'
        CHECK(status IN ('acquired', 'complete', 'failed')),
    acquired_at TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS measurements (
    id TEXT PRIMARY KEY,
    measurement_run_id TEXT NOT NULL REFERENCES measurement_runs(id),
    analytical_material_id TEXT NOT NULL REFERENCES analytical_materials(id),
    replicate_index INTEGER NOT NULL CHECK(replicate_index > 0),
    status TEXT NOT NULL DEFAULT 'acquired' CHECK(status IN ('acquired', 'invalid')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(measurement_run_id, analytical_material_id, replicate_index)
);

CREATE TABLE IF NOT EXISTS measurement_artifacts (
    id TEXT PRIMARY KEY,
    measurement_id TEXT NOT NULL REFERENCES measurements(id),
    source_asset_id TEXT NOT NULL REFERENCES source_assets(id),
    artifact_role TEXT NOT NULL CHECK(artifact_role IN ('raw', 'processed', 'preview')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(measurement_id, artifact_role)
);
"""
