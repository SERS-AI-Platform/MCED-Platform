from typing import Final

LINEAGE_SCHEMA: Final = """
CREATE TABLE IF NOT EXISTS match_candidates (
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
    alias_review INTEGER NOT NULL DEFAULT 0 CHECK(alias_review IN (0, 1)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(
        (status = 'matched' AND analytical_material_id IS NOT NULL
            AND sample_id IS NOT NULL AND clinical_event_id IS NOT NULL)
        OR (status = 'unmatched_clinical' AND analytical_material_id IS NULL
            AND clinical_event_id IS NOT NULL)
        OR (status = 'unmatched_spectrum' AND analytical_material_id IS NOT NULL
            AND clinical_event_id IS NULL)
        OR status IN ('ambiguous', 'duplicate')
    )
);

CREATE TABLE IF NOT EXISTS match_resolutions (
    id TEXT PRIMARY KEY,
    match_candidate_id TEXT NOT NULL REFERENCES match_candidates(id),
    resolution TEXT NOT NULL CHECK(resolution IN ('accepted', 'rejected')),
    resolver TEXT NOT NULL CHECK(length(trim(resolver)) > 0),
    rationale TEXT,
    resolved_sample_id TEXT REFERENCES samples(id),
    resolved_clinical_event_id TEXT REFERENCES clinical_events(id),
    resolved_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sample_labels (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL REFERENCES samples(id),
    task_name TEXT NOT NULL CHECK(length(trim(task_name)) > 0),
    label_definition_version TEXT NOT NULL
        CHECK(length(trim(label_definition_version)) > 0),
    label_value TEXT NOT NULL CHECK(length(trim(label_value)) > 0),
    label_source TEXT NOT NULL CHECK(length(trim(label_source)) > 0),
    status TEXT NOT NULL DEFAULT 'proposed'
        CHECK(status IN ('proposed', 'confirmed', 'rejected')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(
        sample_id, task_name, label_definition_version, label_value, label_source
    )
);

CREATE TABLE IF NOT EXISTS sample_label_evidence (
    id TEXT PRIMARY KEY,
    sample_label_id TEXT NOT NULL REFERENCES sample_labels(id),
    clinical_observation_id TEXT REFERENCES clinical_observations(id),
    match_resolution_id TEXT REFERENCES match_resolutions(id),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK(
        clinical_observation_id IS NOT NULL OR match_resolution_id IS NOT NULL
    )
);

CREATE TABLE IF NOT EXISTS qc_evaluations (
    id TEXT PRIMARY KEY,
    measurement_id TEXT NOT NULL REFERENCES measurements(id),
    rule_version TEXT NOT NULL CHECK(length(trim(rule_version)) > 0),
    evaluation_version TEXT NOT NULL CHECK(length(trim(evaluation_version)) > 0),
    outcome TEXT NOT NULL CHECK(outcome IN ('pass', 'fail', 'review')),
    metrics_json TEXT NOT NULL CHECK(json_valid(metrics_json)),
    evaluated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_runs (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL CHECK(length(trim(model_name)) > 0),
    model_version TEXT NOT NULL CHECK(length(trim(model_version)) > 0),
    status TEXT NOT NULL CHECK(status IN ('completed', 'failed')),
    output_json TEXT CHECK(output_json IS NULL OR json_valid(output_json)),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prediction_input_measurements (
    prediction_run_id TEXT NOT NULL REFERENCES prediction_runs(id),
    measurement_id TEXT NOT NULL REFERENCES measurements(id),
    PRIMARY KEY(prediction_run_id, measurement_id)
);

CREATE TABLE IF NOT EXISTS dataset_manifests (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL CHECK(length(trim(name)) > 0),
    version TEXT NOT NULL CHECK(length(trim(version)) > 0),
    content_sha256 TEXT NOT NULL CHECK(length(content_sha256) = 64),
    status TEXT NOT NULL DEFAULT 'draft' CHECK(status IN ('draft', 'frozen')),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

CREATE TABLE IF NOT EXISTS dataset_manifest_items (
    id TEXT PRIMARY KEY,
    dataset_manifest_id TEXT NOT NULL REFERENCES dataset_manifests(id),
    measurement_id TEXT NOT NULL REFERENCES measurements(id),
    sample_label_id TEXT REFERENCES sample_labels(id),
    qc_evaluation_id TEXT REFERENCES qc_evaluations(id),
    ordinal INTEGER NOT NULL CHECK(ordinal >= 0),
    UNIQUE(dataset_manifest_id, measurement_id),
    UNIQUE(dataset_manifest_id, ordinal)
);

CREATE INDEX IF NOT EXISTS idx_samples_subject ON samples(subject_id);
CREATE INDEX IF NOT EXISTS idx_clinical_events_subject ON clinical_events(subject_id);
CREATE INDEX IF NOT EXISTS idx_measurements_material ON measurements(analytical_material_id);
CREATE INDEX IF NOT EXISTS idx_qc_evaluations_measurement ON qc_evaluations(measurement_id);
"""
