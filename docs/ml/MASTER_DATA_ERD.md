# SERS Master Data ERD

이 문서는 `sers_master.db`를 한 장에 모두 펼치지 않고 세 업무 경계로 나눈다.
같은 테이블이 경계 사이의 연결점일 때는 두 그림에 반복 표시한다.

## 1. 임상정보 ERD

```mermaid
erDiagram
    sites ||--o{ subjects : identifies
    sites ||--o{ source_assets : owns
    subjects ||--o{ samples : provides
    subjects ||--o{ subject_source_aliases : appears_in
    source_assets ||--o{ subject_source_aliases : contains
    source_assets ||--|| clinical_source_metadata : describes
    source_assets ||--o{ ingest_batches : ingested_as
    subjects ||--o{ clinical_events : has
    sites ||--o{ clinical_events : records
    source_assets ||--o{ clinical_events : sourced_from
    ingest_batches ||--o{ clinical_events : loads
    clinical_events ||--o{ clinical_observations : contains

    sites {
        text id PK
        text code UK
        text name
        text status
    }
    subjects {
        text id PK
        text site_id FK
        text patient_id "nullable hospital pseudonym"
        text created_at
    }
    samples {
        text id PK
        text subject_id FK
        text site_id FK
        text solum_label "spectrum matching ID"
        text sample_type
    }
    source_assets {
        text id PK
        text site_id FK
        text uri UK
        text sha256
        text raw_uri
        text asset_kind
        text state
    }
    subject_source_aliases {
        text subject_id PK,FK
        text source_asset_id PK,FK
        text source_field_name
    }
    clinical_source_metadata {
        text source_asset_id PK,FK
        text protocol_code
        text source_group
        text identity_status
        text source_version
    }
    ingest_batches {
        text id PK
        text source_asset_id FK
        text parser_name
        text status
    }
    clinical_events {
        text id PK
        text subject_id FK
        text site_id FK
        text source_asset_id FK
        text ingest_batch_id FK
        text event_type
        text occurred_at
    }
    clinical_observations {
        text id PK
        text clinical_event_id FK
        text canonical_code
        text raw_value
        text value_kind
        text unit
    }
```

`subjects.patient_id`는 병원이 제공한 가명 환자번호이며 없으면 `NULL`이다.
`solum_label`로 빈 환자번호를 채우지 않는다. `samples.solum_label`은 임상자료와
spectrum 파일을 연결하는 별도 ID다. 원본 위치는 `source_assets.raw_uri`에
보관한다.

## 2. Spectrum 및 임상 매칭 ERD

```mermaid
erDiagram
    sites ||--o{ subjects : identifies
    subjects ||--o{ samples : provides
    samples ||--o{ analytical_materials : yields
    analytical_materials ||--o{ analytical_materials : parent_of
    analytical_materials ||--|| spectrum_material_metadata : describes
    sites ||--o{ measurement_runs : performs
    measurement_runs ||--o{ measurements : contains
    analytical_materials ||--o{ measurements : measured_as
    measurements ||--o{ measurement_artifacts : produces
    source_assets ||--o{ measurement_artifacts : stores
    source_assets ||--o| spectrum_inventory_records : inventoried_as
    analytical_materials ||--o{ match_candidates : candidate
    samples ||--o{ match_candidates : candidate
    clinical_events ||--o{ match_candidates : candidate
    match_candidates ||--o{ match_resolutions : reviewed_by

    subjects {
        text id PK
        text site_id FK
        text patient_id "nullable hospital pseudonym"
    }
    samples {
        text id PK
        text subject_id FK
        text site_id FK
        text solum_label UK
        text sample_type
        text collected_at
    }
    analytical_materials {
        text id PK
        text sample_id FK
        text parent_material_id FK
        text material_type
    }
    spectrum_material_metadata {
        text analytical_material_id PK,FK
        text site_id FK
        text material_key UK
        text canonical_source_code
        text preparation
        text material_kind
        text identity_status
    }
    measurement_runs {
        text id PK
        text site_id FK
        text ingest_batch_id FK
        text instrument_key
        text acquired_at
    }
    measurements {
        text id PK
        text measurement_run_id FK
        text analytical_material_id FK
        int replicate_index
        text status
    }
    measurement_artifacts {
        text id PK
        text measurement_id FK
        text source_asset_id FK
        text artifact_role
    }
    source_assets {
        text id PK
        text uri UK
        text sha256
        text raw_uri
    }
    spectrum_inventory_records {
        text source_asset_id PK,FK
        text source_kind
        text status
        text reason_code
        text acquisition_date
        text instrument_key
    }
    match_candidates {
        text id PK
        text analytical_material_id FK
        text sample_id FK
        text clinical_event_id FK
        real score
        text status
        text rule_version
    }
    match_resolutions {
        text id PK
        text match_candidate_id FK
        text resolution
        text resolver
        text resolved_sample_id FK
        text resolved_clinical_event_id FK
    }
    clinical_events {
        text id PK
        text subject_id FK
    }
    sites {
        text id PK
        text code UK
    }
```

한 CSV technical replicate는 `measurements` 한 행이며 실제 파일은
`measurement_artifacts`를 통해 `source_assets`에 연결된다. Liquid/Powder는
동일 검체에서 파생된 별도 `analytical_materials`로 표현한다.
Spectrum의 `canonical_source_code`는 같은 site의 `samples.solum_label`과 정확히
일치할 때만 자동 연결한다. `subjects.patient_id`는 spectrum 매칭에 사용하지 않는다.

## 3. QC, Dataset, Prediction MLOps ERD

```mermaid
erDiagram
    measurements ||--o{ qc_evaluations : evaluated_by
    samples ||--o{ sample_labels : labeled_as
    sample_labels ||--o{ sample_label_evidence : supported_by
    clinical_observations ||--o{ sample_label_evidence : evidence
    match_resolutions ||--o{ sample_label_evidence : evidence
    dataset_manifests ||--|| dataset_manifest_specs : configured_by
    dataset_manifests ||--o{ dataset_manifest_items : includes
    dataset_manifests ||--o{ dataset_manifest_exclusions : excludes
    measurements ||--o{ dataset_manifest_items : selected
    measurements ||--o{ dataset_manifest_exclusions : considered
    sample_labels ||--o{ dataset_manifest_items : targets
    qc_evaluations ||--o{ dataset_manifest_items : qualifies
    dataset_manifest_items ||--o{ dataset_manifest_label_evidence : pins
    sample_label_evidence ||--o{ dataset_manifest_label_evidence : pinned_by
    dataset_manifests ||--o{ prediction_runs : drives
    prediction_runs ||--o{ prediction_input_measurements : consumes
    measurements ||--o{ prediction_input_measurements : input_to

    samples {
        text id PK
        text subject_id FK
    }
    measurements {
        text id PK
        text analytical_material_id FK
        int replicate_index
    }
    qc_evaluations {
        text id PK
        text measurement_id FK
        text rule_version
        text evaluation_version
        text outcome
        json metrics_json
    }
    sample_labels {
        text id PK
        text sample_id FK
        text task_name
        text label_definition_version
        text label_value
        text label_source
        text status
    }
    sample_label_evidence {
        text id PK
        text sample_label_id FK
        text clinical_observation_id FK
        text match_resolution_id FK
    }
    dataset_manifests {
        text id PK
        text name
        text version
        text content_sha256
        text status
    }
    dataset_manifest_specs {
        text dataset_manifest_id PK,FK
        text preprocessing_version
        text feature_schema_version
        text decision_policy_version
        text qc_policy_version
        json provenance_json
    }
    dataset_manifest_items {
        text id PK
        text dataset_manifest_id FK
        text measurement_id FK
        text sample_label_id FK
        text qc_evaluation_id FK
        int ordinal
    }
    dataset_manifest_exclusions {
        text id PK
        text dataset_manifest_id FK
        text measurement_id FK
        text reason
    }
    dataset_manifest_label_evidence {
        text dataset_manifest_item_id PK,FK
        text sample_label_evidence_id PK,FK
    }
    prediction_runs {
        text id PK
        text dataset_manifest_id FK
        text model_name
        text model_version
        text preprocessing_version
        text feature_schema_version
        text decision_policy_version
        text input_set_sha256
        text status
    }
    prediction_input_measurements {
        text prediction_run_id PK,FK
        text measurement_id PK,FK
    }
    clinical_observations {
        text id PK
    }
    match_resolutions {
        text id PK
    }
```

QC는 `rule_version`과 `evaluation_version`만 사용한다. Dataset과 Prediction은
결과를 덮어쓰지 않고 정확한 label, QC, measurement 조합을 append-only로
추적한다. 운영 앱의 legacy `sessions`/`predictions` 연결 테이블은
`sers_master.db`가 아니라 임상 운영 DB 경계에 둔다.
