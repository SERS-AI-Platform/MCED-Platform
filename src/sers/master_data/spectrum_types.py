from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NewType

from ._compat import StrEnum
from .types import SourceAssetId

Sha256 = NewType("Sha256", str)
RawUri = NewType("RawUri", str)


class SourceKind(StrEnum):
    THERMO = "thermo"
    MEDICAL = "medical"
    REMEASUREMENT = "remeasurement"
    BORAMAE_LIQUID = "boramae_liquid"
    BORAMAE_POWDER = "boramae_powder"
    POWDER_REPRODUCIBILITY = "powder_reproducibility"


class InventoryStatus(StrEnum):
    READY = "ready"
    DERIVED = "derived"
    EXCLUDED = "excluded"
    QUARANTINED = "quarantined"


class MaterialKind(StrEnum):
    BIOLOGICAL = "biological"
    BLANK = "blank"
    REFERENCE = "reference"
    MATRIX_BLANK = "matrix_blank"


class IdentityStatus(StrEnum):
    CANONICAL = "canonical"
    ALIAS_REVIEW = "alias_review"


class FastingState(StrEnum):
    FASTING = "fasting"
    NON_FASTING = "non_fasting"
    UNSPECIFIED = "unspecified"


class SpecimenTiming(StrEnum):
    POST_OPERATIVE = "post_operative"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True, slots=True)
class InventoryRoot:
    path: Path
    source_kind: SourceKind
    instrument_key: str
    preparation: str
    acquisition_date: str | None = None


@dataclass(frozen=True, slots=True)
class ParsedSpectrumName:
    group_code: str
    source_sample_code: str
    replicate_index: int | None
    preparation: str
    status: InventoryStatus
    material_kind: MaterialKind
    identity_status: IdentityStatus
    fasting_state: FastingState = FastingState.UNSPECIFIED
    specimen_timing: SpecimenTiming = SpecimenTiming.UNSPECIFIED
    lot_code: str | None = None
    alias_candidates: tuple[str, ...] = ()

    @property
    def solum_label(self) -> str:
        return f"{self.group_code} {self.source_sample_code}"


@dataclass(frozen=True, slots=True)
class InventoryItem:
    source_path: Path
    source_kind: SourceKind
    instrument_key: str
    acquisition_date: str
    root_key: str
    parsed: ParsedSpectrumName | None
    status: InventoryStatus
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class InventoryCounts:
    ready: int
    derived: int
    excluded: int
    quarantined: int


@dataclass(frozen=True, slots=True)
class InventoryReport:
    items: tuple[InventoryItem, ...]
    counts: InventoryCounts

    def redacted_summary(self) -> str:
        return (
            f"ready={self.counts.ready},derived={self.counts.derived},"
            f"excluded={self.counts.excluded},quarantined={self.counts.quarantined}"
        )


@dataclass(frozen=True, slots=True)
class StoredRaw:
    sha256: Sha256
    raw_uri: RawUri
    size_bytes: int


@dataclass(frozen=True, slots=True)
class IngestionCounts:
    source_assets: int
    measurements: int
    derived_artifacts: int
    quarantined: int


@dataclass(frozen=True, slots=True)
class SourceAssetRecord:
    id: SourceAssetId
    sha256: Sha256
    raw_uri: RawUri
