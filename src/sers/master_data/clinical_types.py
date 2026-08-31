from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, NewType

from ._compat import StrEnum

ClinicalRowLocator = NewType("ClinicalRowLocator", str)
ClinicalEventType = Literal["collection", "diagnosis", "procedure", "lab", "other"]
ClinicalInventoryIssueReason = Literal[
    "missing_source",
    "unreadable_source",
    "expected_group_empty",
]


class ClinicalFormat(StrEnum):
    CSV = "csv"
    EXCEL = "excel"


class ClinicalSourceVersion(StrEnum):
    CURRENT = "current"
    HISTORICAL = "historical"


class ClinicalIdentityStatus(StrEnum):
    CANONICAL = "canonical"
    ALIAS_REVIEW = "alias_review"


@dataclass(frozen=True, slots=True)
class ClinicalSheetContract:
    sheet_name: str | None
    header_row: int
    patient_id_fields: tuple[str, ...]
    event_type: ClinicalEventType
    subject_key_alias_field: str | None = None
    data_start_row: int | None = None


@dataclass(frozen=True, slots=True)
class ClinicalSource:
    path: Path
    site_code: str
    site_name: str
    protocol_code: str
    source_group: str
    source_format: ClinicalFormat
    patient_id_fields: tuple[str, ...]
    event_type: ClinicalEventType
    sheet_name: str | None = None
    header_row: int = 1
    canonical_alias: str | None = None
    identity_status: ClinicalIdentityStatus = ClinicalIdentityStatus("canonical")
    source_version: ClinicalSourceVersion = ClinicalSourceVersion("current")
    sheet_contracts: tuple[ClinicalSheetContract, ...] = ()


@dataclass(frozen=True, slots=True)
class ClinicalInventoryCounts:
    current: int
    historical: int


@dataclass(frozen=True, slots=True)
class ClinicalInventoryIssue:
    path: Path
    site_code: str
    protocol_code: str
    source_group: str
    reason_code: ClinicalInventoryIssueReason


@dataclass(frozen=True, slots=True)
class ClinicalInventory:
    sources: tuple[ClinicalSource, ...]
    counts: ClinicalInventoryCounts
    issues: tuple[ClinicalInventoryIssue, ...] = ()

    def redacted_summary(self) -> str:
        parts = [
            f"current={self.counts.current}",
            f"historical={self.counts.historical}",
            f"issues={len(self.issues)}",
        ]
        for reason in (
            "missing_source",
            "unreadable_source",
            "expected_group_empty",
        ):
            count = sum(issue.reason_code == reason for issue in self.issues)
            if count:
                parts.append(f"{reason}={count}")
        return ",".join(parts)


@dataclass(frozen=True, slots=True)
class ClinicalObservationValue:
    source_field_name: str
    canonical_code: str | None
    raw_value: str
    text_value: str | None
    numeric_value: float | None
    date_value: str | None
    unit: str | None
    normalization_status: Literal["normalized", "raw", "unmapped"]


@dataclass(frozen=True, slots=True)
class ClinicalRow:
    locator: ClinicalRowLocator
    patient_id: str | None
    solum_label: str | None
    occurred_at: str | None
    observations: tuple[ClinicalObservationValue, ...]
    event_type: ClinicalEventType = "other"
    source_patient_id_field: str | None = None


@dataclass(frozen=True, slots=True)
class ClinicalIngestionCounts:
    source_assets: int
    subjects: int
    events: int
    observations: int
    quarantined_rows: int
