from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NewType

from ._compat import StrEnum

MatchCandidateId = NewType("MatchCandidateId", str)
MatchResolutionId = NewType("MatchResolutionId", str)


class MatchState(StrEnum):
    MATCHED = "matched"
    UNMATCHED_CLINICAL = "unmatched_clinical"
    UNMATCHED_SPECTRUM = "unmatched_spectrum"
    AMBIGUOUS = "ambiguous"
    DUPLICATE = "duplicate"


class IdentityBasis(StrEnum):
    SOLUM_LABEL = "solum_label"
    MANUAL_REVIEW = "manual_review"
    INVENTORY = "inventory"


@dataclass(frozen=True, slots=True)
class MatchingCounts:
    matched: int
    unmatched_clinical: int
    unmatched_spectrum: int
    ambiguous: int
    duplicate: int


@dataclass(frozen=True, slots=True)
class ManualResolution:
    candidate_id: str
    resolution: Literal["accepted", "rejected"]
    resolver: str
    rationale: str | None
    sample_id: str | None
    clinical_event_id: str | None


@dataclass(frozen=True, slots=True)
class MaterialIdentity:
    material_id: str
    site_id: str
    source_code: str
    alias_review: bool
    linked_sample_id: str | None


@dataclass(frozen=True, slots=True)
class IdentityTarget:
    subject_id: str
    basis: IdentityBasis
    sample_id: str | None


@dataclass(frozen=True, slots=True)
class CandidateDraft:
    site_id: str
    material_id: str | None
    sample_id: str | None
    clinical_event_id: str | None
    state: MatchState
    rule_version: str
    basis: IdentityBasis
    alias_review: bool
