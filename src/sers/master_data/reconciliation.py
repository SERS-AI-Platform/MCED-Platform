from __future__ import annotations

import sqlite3

from .matching_types import MatchingCounts, MatchState


def reconciliation_counts(
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
