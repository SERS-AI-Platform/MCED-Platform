CREATE SCHEMA IF NOT EXISTS master;

CREATE TABLE IF NOT EXISTS master.sites (
    id TEXT PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'inactive')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.source_assets (
    id TEXT PRIMARY KEY,
    site_id TEXT REFERENCES master.sites(id),
    uri TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64),
    asset_kind TEXT NOT NULL CHECK (asset_kind IN ('clinical', 'spectrum', 'manifest', 'other')),
    state TEXT NOT NULL DEFAULT 'discovered'
        CHECK (state IN ('discovered', 'ingested', 'failed')),
    size_bytes BIGINT CHECK (size_bytes IS NULL OR size_bytes >= 0),
    raw_uri TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.ingest_batches (
    id TEXT PRIMARY KEY,
    source_asset_id TEXT NOT NULL REFERENCES master.source_assets(id),
    status TEXT NOT NULL DEFAULT 'started'
        CHECK (status IN ('started', 'completed', 'failed')),
    parser_name TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMPTZ,
    UNIQUE (source_asset_id, parser_name)
);

CREATE TABLE IF NOT EXISTS master.subjects (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES master.sites(id),
    patient_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (site_id, patient_id)
);

CREATE TABLE IF NOT EXISTS master.samples (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES master.subjects(id),
    site_id TEXT NOT NULL REFERENCES master.sites(id),
    solum_label TEXT NOT NULL,
    sample_type TEXT NOT NULL CHECK (sample_type IN ('urine', 'serum', 'plasma', 'other')),
    collected_at DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (site_id, solum_label)
);

CREATE TABLE IF NOT EXISTS master.sample_crosswalk_sources (
    sample_id TEXT NOT NULL REFERENCES master.samples(id),
    source_asset_id TEXT NOT NULL REFERENCES master.source_assets(id),
    source_row_locator TEXT NOT NULL,
    identity_field TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (sample_id, source_asset_id),
    UNIQUE (source_asset_id, source_row_locator)
);

CREATE TABLE IF NOT EXISTS master.clinical_events (
    id TEXT PRIMARY KEY,
    subject_id TEXT NOT NULL REFERENCES master.subjects(id),
    site_id TEXT NOT NULL REFERENCES master.sites(id),
    ingest_batch_id TEXT REFERENCES master.ingest_batches(id),
    source_asset_id TEXT REFERENCES master.source_assets(id),
    source_row_locator TEXT,
    event_type TEXT NOT NULL CHECK (event_type IN ('collection', 'diagnosis', 'procedure', 'lab', 'other')),
    occurred_at DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.clinical_observations (
    id TEXT PRIMARY KEY,
    clinical_event_id TEXT NOT NULL REFERENCES master.clinical_events(id),
    code TEXT NOT NULL,
    source_field_name TEXT,
    canonical_code TEXT,
    raw_value TEXT,
    value_kind TEXT NOT NULL CHECK (value_kind IN ('text', 'numeric', 'date')),
    text_value TEXT,
    numeric_value DOUBLE PRECISION,
    date_value TEXT,
    unit TEXT,
    normalization_status TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (
        (value_kind = 'text' AND text_value IS NOT NULL AND numeric_value IS NULL AND date_value IS NULL)
        OR (value_kind = 'numeric' AND numeric_value IS NOT NULL AND text_value IS NULL AND date_value IS NULL)
        OR (value_kind = 'date' AND date_value IS NOT NULL AND text_value IS NULL AND numeric_value IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS master.analytical_materials (
    id TEXT PRIMARY KEY,
    sample_id TEXT REFERENCES master.samples(id),
    parent_material_id TEXT REFERENCES master.analytical_materials(id),
    material_type TEXT NOT NULL CHECK (material_type IN ('primary', 'aliquot', 'extract', 'average')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.measurement_runs (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES master.sites(id),
    ingest_batch_id TEXT REFERENCES master.ingest_batches(id),
    instrument_key TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'acquired' CHECK (status IN ('acquired', 'complete', 'failed')),
    acquired_at DATE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.measurement_run_metadata (
    measurement_run_id TEXT PRIMARY KEY REFERENCES master.measurement_runs(id),
    source_root TEXT NOT NULL,
    source_batch TEXT NOT NULL,
    preparation TEXT,
    reducing_agent TEXT,
    laser_power_mw DOUBLE PRECISION,
    integration_time_s DOUBLE PRECISION,
    average_count INTEGER,
    randomization_block TEXT,
    metadata_status TEXT NOT NULL DEFAULT 'unknown',
    raw_metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS master.measurements (
    id TEXT PRIMARY KEY,
    measurement_run_id TEXT NOT NULL REFERENCES master.measurement_runs(id),
    analytical_material_id TEXT NOT NULL REFERENCES master.analytical_materials(id),
    replicate_index INTEGER NOT NULL CHECK (replicate_index > 0),
    status TEXT NOT NULL DEFAULT 'acquired' CHECK (status IN ('acquired', 'invalid')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (measurement_run_id, analytical_material_id, replicate_index)
);

CREATE TABLE IF NOT EXISTS master.measurement_artifacts (
    id TEXT PRIMARY KEY,
    measurement_id TEXT NOT NULL REFERENCES master.measurements(id),
    source_asset_id TEXT NOT NULL REFERENCES master.source_assets(id),
    artifact_role TEXT NOT NULL CHECK (artifact_role IN ('raw', 'processed', 'preview')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (measurement_id, artifact_role)
);

CREATE TABLE IF NOT EXISTS master.measurement_metadata (
    measurement_id TEXT PRIMARY KEY REFERENCES master.measurements(id),
    raw_spectrum_id BIGINT NOT NULL UNIQUE REFERENCES public.raw_spectra(id),
    source_kind TEXT NOT NULL,
    material_role TEXT NOT NULL,
    control_type TEXT,
    group_code TEXT,
    source_sample_id TEXT,
    acquisition_order INTEGER,
    mapping_status TEXT NOT NULL,
    mapping_method TEXT NOT NULL,
    mapping_confidence DOUBLE PRECISION,
    review_note TEXT
);

CREATE TABLE IF NOT EXISTS master.spectrum_inventory_records (
    source_asset_id TEXT PRIMARY KEY REFERENCES master.source_assets(id),
    source_kind TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('ready', 'derived', 'excluded', 'quarantined')),
    reason_code TEXT,
    acquisition_date DATE,
    instrument_key TEXT,
    root_key TEXT NOT NULL,
    canonical_source_code TEXT,
    preparation TEXT,
    fasting_state TEXT NOT NULL DEFAULT 'unspecified'
        CHECK (fasting_state IN ('fasting', 'non_fasting', 'unspecified')),
    specimen_timing TEXT NOT NULL DEFAULT 'unspecified'
        CHECK (specimen_timing IN ('post_operative', 'unspecified')),
    lot_code TEXT,
    identity_status TEXT,
    alias_candidates TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS master.spectrum_material_metadata (
    analytical_material_id TEXT PRIMARY KEY REFERENCES master.analytical_materials(id),
    site_id TEXT NOT NULL REFERENCES master.sites(id),
    material_key TEXT NOT NULL,
    canonical_source_code TEXT NOT NULL,
    source_group TEXT NOT NULL,
    material_kind TEXT NOT NULL CHECK (material_kind IN ('biological', 'blank', 'reference', 'matrix_blank')),
    preparation TEXT NOT NULL,
    fasting_state TEXT NOT NULL DEFAULT 'unspecified'
        CHECK (fasting_state IN ('fasting', 'non_fasting', 'unspecified')),
    specimen_timing TEXT NOT NULL DEFAULT 'unspecified'
        CHECK (specimen_timing IN ('post_operative', 'unspecified')),
    lot_code TEXT,
    identity_status TEXT NOT NULL CHECK (identity_status IN ('canonical', 'alias_review')),
    alias_candidates TEXT NOT NULL DEFAULT '',
    UNIQUE (site_id, material_key)
);

CREATE TABLE IF NOT EXISTS master.match_candidates (
    id TEXT PRIMARY KEY,
    site_id TEXT NOT NULL REFERENCES master.sites(id),
    analytical_material_id TEXT REFERENCES master.analytical_materials(id),
    sample_id TEXT REFERENCES master.samples(id),
    clinical_event_id TEXT REFERENCES master.clinical_events(id),
    score DOUBLE PRECISION NOT NULL CHECK (score >= 0 AND score <= 1),
    status TEXT NOT NULL CHECK (status IN ('matched', 'unmatched_spectrum', 'ambiguous', 'duplicate')),
    rule_version TEXT NOT NULL,
    identity_basis TEXT NOT NULL CHECK (identity_basis IN ('solum_label', 'manual_review', 'inventory')),
    alias_review BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.qc_artifacts (
    id TEXT PRIMARY KEY,
    measurement_run_id TEXT NOT NULL REFERENCES master.measurement_runs(id),
    source_asset_id TEXT NOT NULL REFERENCES master.source_assets(id),
    raw_spectrum_id BIGINT NOT NULL UNIQUE REFERENCES public.raw_spectra(id),
    qc_role TEXT NOT NULL,
    qc_status TEXT NOT NULL DEFAULT 'unreviewed'
        CHECK (qc_status IN ('unreviewed', 'pass', 'fail', 'review')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.qc_evaluations (
    id TEXT PRIMARY KEY,
    measurement_id TEXT NOT NULL REFERENCES master.measurements(id),
    rule_version TEXT NOT NULL,
    evaluation_version TEXT NOT NULL,
    outcome TEXT NOT NULL CHECK (outcome IN ('pass', 'fail', 'review')),
    metrics_json JSONB NOT NULL,
    evaluated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.sample_labels (
    id TEXT PRIMARY KEY,
    sample_id TEXT NOT NULL REFERENCES master.samples(id),
    task_name TEXT NOT NULL,
    label_definition_version TEXT NOT NULL,
    label_value TEXT NOT NULL,
    label_source TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'proposed' CHECK (status IN ('proposed', 'confirmed', 'rejected')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (sample_id, task_name, label_definition_version, label_value, label_source)
);

CREATE TABLE IF NOT EXISTS master.sample_label_evidence (
    id TEXT PRIMARY KEY,
    sample_label_id TEXT NOT NULL REFERENCES master.sample_labels(id),
    clinical_observation_id TEXT REFERENCES master.clinical_observations(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CHECK (clinical_observation_id IS NOT NULL)
);

CREATE TABLE IF NOT EXISTS master.dataset_manifests (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    content_sha256 TEXT NOT NULL CHECK (length(content_sha256) = 64),
    status TEXT NOT NULL DEFAULT 'draft' CHECK (status IN ('draft', 'frozen')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (name, version)
);

CREATE TABLE IF NOT EXISTS master.dataset_manifest_items (
    id TEXT PRIMARY KEY,
    dataset_manifest_id TEXT NOT NULL REFERENCES master.dataset_manifests(id),
    measurement_id TEXT NOT NULL REFERENCES master.measurements(id),
    sample_label_id TEXT REFERENCES master.sample_labels(id),
    qc_evaluation_id TEXT REFERENCES master.qc_evaluations(id),
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0),
    UNIQUE (dataset_manifest_id, measurement_id),
    UNIQUE (dataset_manifest_id, ordinal)
);

CREATE TABLE IF NOT EXISTS master.dataset_manifest_specs (
    dataset_manifest_id TEXT PRIMARY KEY REFERENCES master.dataset_manifests(id),
    preprocessing_version TEXT NOT NULL,
    feature_schema_version TEXT NOT NULL,
    decision_policy_version TEXT NOT NULL,
    qc_policy_version TEXT NOT NULL,
    qc_rule_version TEXT NOT NULL,
    inclusion_policy JSONB NOT NULL,
    exclusion_policy JSONB NOT NULL,
    provenance JSONB NOT NULL
);

CREATE TABLE IF NOT EXISTS master.dataset_manifest_exclusions (
    id TEXT PRIMARY KEY,
    dataset_manifest_id TEXT NOT NULL REFERENCES master.dataset_manifests(id),
    measurement_id TEXT NOT NULL REFERENCES master.measurements(id),
    qc_evaluation_id TEXT REFERENCES master.qc_evaluations(id),
    reason TEXT NOT NULL CHECK (reason IN ('measurement_invalid', 'qc_missing', 'qc_outcome_rejected')),
    UNIQUE (dataset_manifest_id, measurement_id)
);

CREATE TABLE IF NOT EXISTS master.dataset_manifest_label_evidence (
    dataset_manifest_item_id TEXT NOT NULL REFERENCES master.dataset_manifest_items(id),
    sample_label_evidence_id TEXT NOT NULL REFERENCES master.sample_label_evidence(id),
    PRIMARY KEY (dataset_manifest_item_id, sample_label_evidence_id)
);

CREATE TABLE IF NOT EXISTS master.prediction_runs (
    id TEXT PRIMARY KEY,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('completed', 'failed')),
    output_json JSONB,
    dataset_manifest_id TEXT REFERENCES master.dataset_manifests(id),
    preprocessing_version TEXT,
    feature_schema_version TEXT,
    decision_policy_version TEXT,
    input_set_sha256 TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS master.prediction_input_measurements (
    prediction_run_id TEXT NOT NULL REFERENCES master.prediction_runs(id),
    measurement_id TEXT NOT NULL REFERENCES master.measurements(id),
    PRIMARY KEY (prediction_run_id, measurement_id)
);

CREATE INDEX IF NOT EXISTS idx_master_samples_label ON master.samples(site_id, solum_label);
CREATE INDEX IF NOT EXISTS idx_master_measurements_material ON master.measurements(analytical_material_id);
CREATE INDEX IF NOT EXISTS idx_master_measurement_metadata_raw ON master.measurement_metadata(raw_spectrum_id);
CREATE INDEX IF NOT EXISTS idx_master_match_status ON master.match_candidates(status);
CREATE INDEX IF NOT EXISTS idx_master_qc_run ON master.qc_artifacts(measurement_run_id);
CREATE INDEX IF NOT EXISTS idx_master_manifest_items_manifest ON master.dataset_manifest_items(dataset_manifest_id);
CREATE INDEX IF NOT EXISTS idx_master_prediction_manifest ON master.prediction_runs(dataset_manifest_id);

CREATE OR REPLACE VIEW master.spectrum_measurement_view AS
SELECT
    measurement.id AS measurement_id,
    measurement.measurement_run_id,
    measurement.replicate_index,
    measurement.status AS measurement_status,
    metadata.raw_spectrum_id,
    metadata.mapping_status,
    metadata.mapping_method,
    metadata.mapping_confidence,
    metadata.review_note,
    samples.solum_label,
    samples.sample_type,
    raw.source_batch,
    raw.source_path,
    raw.source_kind,
    raw.group_code,
    raw.sample_id,
    raw.wavenumber,
    raw.intensities,
    raw.n_points,
    raw.x_min,
    raw.x_max,
    raw.source_sha256
FROM master.measurements AS measurement
JOIN master.measurement_metadata AS metadata
  ON metadata.measurement_id = measurement.id
JOIN public.raw_spectra AS raw
  ON raw.id = metadata.raw_spectrum_id
LEFT JOIN master.analytical_materials AS materials
  ON materials.id = measurement.analytical_material_id
LEFT JOIN master.samples AS samples
  ON samples.id = materials.sample_id;

CREATE OR REPLACE FUNCTION master.reject_append_only_update() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END;
$$;

DROP TRIGGER IF EXISTS qc_evaluations_append_only_update ON master.qc_evaluations;
CREATE TRIGGER qc_evaluations_append_only_update
BEFORE UPDATE OR DELETE ON master.qc_evaluations
FOR EACH ROW EXECUTE FUNCTION master.reject_append_only_update();

DROP TRIGGER IF EXISTS prediction_runs_append_only_update ON master.prediction_runs;
CREATE TRIGGER prediction_runs_append_only_update
BEFORE UPDATE OR DELETE ON master.prediction_runs
FOR EACH ROW EXECUTE FUNCTION master.reject_append_only_update();

DROP TRIGGER IF EXISTS frozen_manifest_update ON master.dataset_manifests;
CREATE TRIGGER frozen_manifest_update
BEFORE UPDATE OR DELETE ON master.dataset_manifests
FOR EACH ROW WHEN (OLD.status = 'frozen')
EXECUTE FUNCTION master.reject_append_only_update();
