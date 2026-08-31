from __future__ import annotations

import sqlite3

from .matching_types import (
    IdentityBasis,
    IdentityTarget,
    MatchingCounts,
    MatchState,
    MaterialIdentity,
)


def materials(connection: sqlite3.Connection) -> tuple[MaterialIdentity, ...]:
    rows = connection.execute(
        """SELECT sm.analytical_material_id, sm.site_id,
                  sm.canonical_source_code, sm.identity_status, am.sample_id
           FROM spectrum_material_metadata AS sm
           JOIN analytical_materials AS am ON am.id = sm.analytical_material_id
           WHERE sm.material_kind = 'biological'
           ORDER BY sm.site_id, sm.analytical_material_id"""
    )
    return tuple(
        MaterialIdentity(
            material_id=row[0],
            site_id=row[1],
            source_code=row[2],
            alias_review=row[3] == "alias_review",
            linked_sample_id=row[4],
        )
        for row in rows
    )


def identity_targets(
    connection: sqlite3.Connection,
    material: MaterialIdentity,
) -> tuple[IdentityTarget, ...]:
    targets: dict[str, IdentityTarget] = {}
    sample_rows = connection.execute(
        """SELECT subject_id, id
           FROM samples
           WHERE site_id = ? AND solum_label = ?""",
        (material.site_id, material.source_code),
    )
    for row in sample_rows:
        targets[row[0]] = IdentityTarget(
            row[0],
            IdentityBasis.SOLUM_LABEL,
            row[1],
        )
    return tuple(targets.values())


def clinical_event_ids(
    connection: sqlite3.Connection,
    subject_id: str,
    site_id: str,
) -> tuple[str, ...]:
    return tuple(
        row[0]
        for row in connection.execute(
            """SELECT id FROM clinical_events
               WHERE subject_id = ? AND site_id = ?
               ORDER BY id""",
            (subject_id, site_id),
        )
    )


def clinical_event_sites(
    connection: sqlite3.Connection,
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (row[0], row[1])
        for row in connection.execute(
            "SELECT id, site_id FROM clinical_events ORDER BY site_id, id"
        )
    )


def matching_counts(
    connection: sqlite3.Connection,
    rule_version: str,
) -> MatchingCounts:
    values = {
        row[0]: row[1]
        for row in connection.execute(
            """SELECT status, COUNT(*) FROM match_candidates
               WHERE rule_version = ? GROUP BY status""",
            (rule_version,),
        )
    }
    return MatchingCounts(
        matched=values.get(MatchState.MATCHED.value, 0),
        unmatched_clinical=values.get(MatchState.UNMATCHED_CLINICAL.value, 0),
        unmatched_spectrum=values.get(MatchState.UNMATCHED_SPECTRUM.value, 0),
        ambiguous=values.get(MatchState.AMBIGUOUS.value, 0),
        duplicate=values.get(MatchState.DUPLICATE.value, 0),
    )


def review_metadata(
    connection: sqlite3.Connection,
    material_id: str,
) -> tuple[str, bool] | None:
    row = connection.execute(
        """SELECT sm.site_id, sm.identity_status
           FROM spectrum_material_metadata AS sm
           WHERE sm.analytical_material_id = ?""",
        (material_id,),
    ).fetchone()
    if row is None:
        return None
    return row[0], row[1] == "alias_review"
