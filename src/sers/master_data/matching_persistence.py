from __future__ import annotations

import sqlite3
from uuid import UUID, uuid4, uuid5

from .matching_types import (
    CandidateDraft,
    ManualResolution,
    MatchCandidateId,
    MatchResolutionId,
    MatchState,
)

_ID_NAMESPACE = UUID("d3416d56-4ee7-4111-9291-8233164188a9")


def stable_id(kind: str, key: str) -> str:
    return str(uuid5(_ID_NAMESPACE, f"{kind}:{key}"))


def persist_candidate(
    connection: sqlite3.Connection,
    draft: CandidateDraft,
) -> MatchCandidateId:
    identity = "|".join(
        (
            draft.rule_version,
            draft.site_id,
            draft.material_id or "",
            draft.sample_id or "",
            draft.clinical_event_id or "",
            draft.state.value,
            draft.basis.value,
        )
    )
    candidate_id = MatchCandidateId(stable_id("match-candidate", identity))
    connection.execute(
        """INSERT OR IGNORE INTO match_candidates (
               id, site_id, analytical_material_id, sample_id,
               clinical_event_id, score, status, rule_version,
               identity_basis, alias_review
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            candidate_id,
            draft.site_id,
            draft.material_id,
            draft.sample_id,
            draft.clinical_event_id,
            1.0 if draft.state is MatchState.MATCHED else 0.0,
            draft.state.value,
            draft.rule_version,
            draft.basis.value,
            int(draft.alias_review),
        ),
    )
    return candidate_id


def append_resolution(
    connection: sqlite3.Connection,
    draft: ManualResolution,
) -> MatchResolutionId:
    resolution_id = MatchResolutionId(str(uuid4()))
    connection.execute(
        """INSERT INTO match_resolutions (
               id, match_candidate_id, resolution, resolver, rationale,
               resolved_sample_id, resolved_clinical_event_id
           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (
            resolution_id,
            draft.candidate_id,
            draft.resolution,
            draft.resolver,
            draft.rationale,
            draft.sample_id,
            draft.clinical_event_id,
        ),
    )
    return resolution_id
