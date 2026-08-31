from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType

SiteId = NewType("SiteId", str)
SubjectId = NewType("SubjectId", str)
SourceAssetId = NewType("SourceAssetId", str)
IngestBatchId = NewType("IngestBatchId", str)
SampleId = NewType("SampleId", str)
AnalyticalMaterialId = NewType("AnalyticalMaterialId", str)
MeasurementRunId = NewType("MeasurementRunId", str)
MeasurementId = NewType("MeasurementId", str)

SiteStatus = Literal["active", "inactive"]
AssetKind = Literal["clinical", "spectrum", "manifest", "other"]
SampleType = Literal["urine", "serum", "plasma", "other"]
MaterialType = Literal["primary", "aliquot", "extract", "average"]


@dataclass(frozen=True, slots=True)
class SiteDraft:
    code: str
    name: str
    status: SiteStatus = "active"


@dataclass(frozen=True, slots=True)
class Site:
    id: SiteId
    code: str
    name: str
    status: SiteStatus


@dataclass(frozen=True, slots=True)
class SubjectIdentity:
    subject_id: SubjectId
    site_id: SiteId
    patient_id: str | None


@dataclass(frozen=True, slots=True)
class SourceAssetDraft:
    site_id: SiteId | None
    uri: str
    sha256: str
    asset_kind: AssetKind
    size_bytes: int | None = None
    raw_uri: str | None = None


@dataclass(frozen=True, slots=True)
class IngestBatchDraft:
    source_asset_id: SourceAssetId
    parser_name: str


@dataclass(frozen=True, slots=True)
class SampleDraft:
    subject_id: SubjectId
    site_id: SiteId
    sample_type: SampleType
    solum_label: str | None = None
    collected_at: str | None = None


@dataclass(frozen=True, slots=True)
class AnalyticalMaterialDraft:
    sample_id: SampleId | None
    material_type: MaterialType
    parent_material_id: AnalyticalMaterialId | None = None


@dataclass(frozen=True, slots=True)
class MeasurementRunDraft:
    site_id: SiteId
    instrument_key: str
    ingest_batch_id: IngestBatchId | None = None
    acquired_at: str | None = None


@dataclass(frozen=True, slots=True)
class MeasurementDraft:
    measurement_run_id: MeasurementRunId
    analytical_material_id: AnalyticalMaterialId
    replicate_index: int
