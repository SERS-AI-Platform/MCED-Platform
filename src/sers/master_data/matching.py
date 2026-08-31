from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from . import matching_queries
from ._compat import assert_never
from .matching_persistence import append_resolution, persist_candidate, stable_id
from .matching_types import (
    CandidateDraft,
    IdentityBasis,
    IdentityTarget,
    ManualResolution,
    MatchingCounts,
    MatchResolutionId,
    MatchState,
    MaterialIdentity,
)


@dataclass(frozen=True, slots=True)
class InvalidReviewStateError(ValueError):
    state: MatchState

    def __str__(self) -> str:
        return f"{self.state.value} is not a manual review state"


def _sample(
    connection: sqlite3.Connection,
    material: MaterialIdentity,
    target: IdentityTarget,
) -> str:
    if target.sample_id is not None:
        return target.sample_id
    sample_id = stable_id(
        "matched-sample",
        f"{material.site_id}:{target.subject_id}:{material.source_code}",
    )
    connection.execute(
        """INSERT OR IGNORE INTO samples (
               id, subject_id, site_id, solum_label, sample_type
           ) VALUES (?, ?, ?, ?, 'urine')""",
        (sample_id, target.subject_id, material.site_id, material.source_code),
    )
    return sample_id


def _record_material(
    connection: sqlite3.Connection,
    material: MaterialIdentity,
    rule_version: str,
) -> tuple[str, ...]:
    targets = matching_queries.identity_targets(connection, material)
    if len(targets) == 0:
        persist_candidate(
            connection,
            CandidateDraft(
                material.site_id,
                material.material_id,
                None,
                None,
                MatchState.UNMATCHED_SPECTRUM,
                rule_version,
                IdentityBasis.SOLUM_LABEL,
                material.alias_review,
            ),
        )
        return ()
    if len(targets) > 1:
        persist_candidate(
            connection,
            CandidateDraft(
                material.site_id,
                material.material_id,
                None,
                None,
                MatchState.AMBIGUOUS,
                rule_version,
                IdentityBasis.MANUAL_REVIEW,
                material.alias_review,
            ),
        )
        return ()
    target = targets[0]
    sample_id = _sample(connection, material, target)
    if (
        material.linked_sample_id is not None
        and material.linked_sample_id != sample_id
    ):
        persist_candidate(
            connection,
            CandidateDraft(
                material.site_id,
                material.material_id,
                None,
                None,
                MatchState.AMBIGUOUS,
                rule_version,
                IdentityBasis.MANUAL_REVIEW,
                material.alias_review,
            ),
        )
        return ()
    connection.execute(
        """UPDATE analytical_materials SET sample_id = ?
           WHERE id = ? AND sample_id IS NULL""",
        (sample_id, material.material_id),
    )
    events = matching_queries.clinical_event_ids(
        connection,
        target.subject_id,
        material.site_id,
    )
    if len(events) == 0:
        persist_candidate(
            connection,
            CandidateDraft(
                material.site_id,
                material.material_id,
                sample_id,
                None,
                MatchState.UNMATCHED_SPECTRUM,
                rule_version,
                target.basis,
                material.alias_review,
            ),
        )
        return ()
    for event_id in events:
        persist_candidate(
            connection,
            CandidateDraft(
                material.site_id,
                material.material_id,
                sample_id,
                event_id,
                MatchState.MATCHED,
                rule_version,
                target.basis,
                material.alias_review,
            ),
        )
    return events


def _record_unmatched_clinical(
    connection: sqlite3.Connection,
    matched_events: set[str],
    rule_version: str,
) -> None:
    for event_id, site_id in matching_queries.clinical_event_sites(connection):
        if event_id not in matched_events:
            persist_candidate(
                connection,
                CandidateDraft(
                    site_id,
                    None,
                    None,
                    event_id,
                    MatchState.UNMATCHED_CLINICAL,
                    rule_version,
                    IdentityBasis.INVENTORY,
                    False,
                ),
            )
def run_deterministic_matching(
    connection: sqlite3.Connection,
    rule_version: str,
) -> MatchingCounts:
    matched_events: set[str] = set()
    for material in matching_queries.materials(connection):
        matched_events.update(_record_material(connection, material, rule_version))
    _record_unmatched_clinical(connection, matched_events, rule_version)
    return matching_queries.matching_counts(connection, rule_version)


def record_review_state(
    connection: sqlite3.Connection,
    *,
    material_id: str,
    state: MatchState,
    rule_version: str,
) -> None:
    match state:
        case MatchState.AMBIGUOUS | MatchState.DUPLICATE:
            pass
        case (
            MatchState.MATCHED
            | MatchState.UNMATCHED_CLINICAL
            | MatchState.UNMATCHED_SPECTRUM
        ):
            raise InvalidReviewStateError(state)
        case unreachable:
            assert_never(unreachable)
    metadata = matching_queries.review_metadata(connection, material_id)
    assert metadata is not None
    site_id, alias_review = metadata
    persist_candidate(
        connection,
        CandidateDraft(
            site_id,
            material_id,
            None,
            None,
            state,
            rule_version,
            IdentityBasis.INVENTORY,
            alias_review,
        ),
    )


def append_manual_resolution(
    connection: sqlite3.Connection,
    draft: ManualResolution,
) -> MatchResolutionId:
    return append_resolution(connection, draft)
