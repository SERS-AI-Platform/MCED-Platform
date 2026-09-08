from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final
from uuid import UUID, uuid5

_ID_NAMESPACE: Final = UUID("6b8a6b34-b8dc-4c7e-8717-6a8dbf7a8c7c")


@dataclass(frozen=True, slots=True)
class RawInput:
    dataset_version: str
    raw_id: int
    source_root: str
    source_path: str
    source_batch: str
    source_kind: str
    control_type: str | None
    group_code: str | None
    sample_id: str
    replicate: int | None
    source_sha256: str


@dataclass(frozen=True, slots=True)
class ClinicalInput:
    dataset_version: str
    registry_id: int
    source_workbook: str
    source_sha256: str
    source_row_number: int
    solum_label: str
    sample_type: str | None
    collection_date: str | None
    raw_record: dict[str, str | int | float | bool | None]


@dataclass(frozen=True, slots=True)
class ClinicalIdentity:
    registry_id: int
    solum_label: str
    subject_id: str
    sample_id: str
    clinical_event_id: str


@dataclass(frozen=True, slots=True)
class RunMetadata:
    run_key: str
    acquisition_date: str | None
    laser_power_mw: float | None
    integration_time_s: float | None
    average_count: int | None
    metadata_status: str


@dataclass(frozen=True, slots=True)
class MappingDecision:
    solum_label: str | None
    identity: ClinicalIdentity | None
    mapping_status: str
    material_role: str
    qc_role: str | None
    mapping_method: str
    review_note: str | None


def stable_id(kind: str, key: str) -> str:
    return str(uuid5(_ID_NAMESPACE, f"{kind}:{key}"))


def normalize_label(group_code: str | None, sample_id: str) -> str | None:
    if not group_code or not sample_id.isdigit():
        return None
    group = {"H.D.": "H. D."}.get(group_code.strip().upper(), group_code.strip().upper())
    return f"{group}_{int(sample_id)}"


def run_metadata(source_root: str, source_batch: str) -> RunMetadata:
    date_match = re.match(r"(20\d{6,})", source_batch)
    power_match = re.search(r"(\d+(?:\.\d+)?)\s*mW", source_batch, re.IGNORECASE)
    time_match = re.search(r"(\d+(?:\.\d+)?)\s*s(?:ec)?", source_batch, re.IGNORECASE)
    average_match = re.search(r"Ave\s*(\d+)", source_batch, re.IGNORECASE)
    acquisition_date = None
    if date_match and len(date_match.group(1)) >= 8:
        digits = date_match.group(1)
        acquisition_date = f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    metadata_status = "inferred_from_path" if any(
        match is not None for match in (date_match, power_match, time_match, average_match)
    ) else "unknown"
    return RunMetadata(
        run_key=f"{source_root}::{source_batch}",
        acquisition_date=acquisition_date,
        laser_power_mw=float(power_match.group(1)) if power_match else None,
        integration_time_s=float(time_match.group(1)) if time_match else None,
        average_count=int(average_match.group(1)) if average_match else None,
        metadata_status=metadata_status,
    )


def decide_mapping(
    raw: RawInput,
    identities: dict[str, ClinicalIdentity],
) -> MappingDecision:
    if raw.source_kind == "calibration_control":
        return MappingDecision(
            solum_label=None,
            identity=None,
            mapping_status="not_applicable",
            material_role="qc",
            qc_role=raw.control_type or "calibration_control",
            mapping_method="source_kind",
            review_note=None,
        )
    label = normalize_label(raw.group_code, raw.sample_id)
    identity = identities.get(label) if label else None
    if identity is not None:
        return MappingDecision(
            solum_label=label,
            identity=identity,
            mapping_status="matched",
            material_role="average" if raw.source_kind == "average" else "patient_sample",
            qc_role=None,
            mapping_method="canonical_group_sample",
            review_note=None,
        )
    return MappingDecision(
        solum_label=label,
        identity=None,
        mapping_status="unmatched_spectrum",
        material_role="average" if raw.source_kind == "average" else "patient_sample",
        qc_role=None,
        mapping_method="canonical_group_sample" if label else "missing_group_or_numeric_sample",
        review_note="no_clinical_registry_match" if label else "cannot_build_solum_label",
    )
