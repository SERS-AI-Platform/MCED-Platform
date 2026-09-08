from __future__ import annotations

from pathlib import Path

import pytest

from sers.master_data.matching import (
    ManualResolution,
    MatchState,
    append_manual_resolution,
    record_review_state,
    run_deterministic_matching,
)
from sers.master_data.spectrum_parser import parse_spectrum_name
from tests.master_data.matching_test_support import database, seed_identity, seed_material


def test_matching_is_site_scoped_exact_and_idempotent(tmp_path: Path) -> None:
    # Given: the same pseudonymous key at two sites and exact spectra at both sites.
    with database(tmp_path / "matching.db") as connection:
        first_identity = seed_identity(connection, "SITE-A", "PRO 001")
        second_identity = seed_identity(connection, "SITE-B", "PRO 001")
        first_material = seed_material(connection, first_identity.site_id, "PRO 001")
        second_material = seed_material(connection, second_identity.site_id, "PRO 001")

        # When: deterministic matching is repeated.
        first = run_deterministic_matching(connection, "exact-v1")
        second = run_deterministic_matching(connection, "exact-v1")
        linked_subjects = tuple(
            row[0]
            for row in connection.execute(
                """SELECT s.subject_id
                   FROM analytical_materials AS am
                   JOIN samples AS s ON s.id = am.sample_id
                   ORDER BY s.site_id"""
            )
        )

        # Then: both sites remain separated and no duplicate lineage is written.
        assert first == second
        assert first.matched == 2
        assert linked_subjects == (
            first_identity.subject_id,
            second_identity.subject_id,
        )
        assert connection.execute(
            "SELECT COUNT(*) FROM match_candidates"
        ).fetchone()[0] == 2
        assert {first_material, second_material} == {
            row[0]
            for row in connection.execute(
                "SELECT analytical_material_id FROM match_candidates"
            )
        }


def test_matching_persists_review_states_without_alias_rewrite(tmp_path: Path) -> None:
    # Given: exact unmatched records, a conflicting explicit sample, and alias metadata.
    with database(tmp_path / "review.db") as connection:
        matched = seed_identity(connection, "SITE-A", "PRO 001")
        unmatched_clinical = seed_identity(connection, "SITE-A", "PRO 404")
        conflicting = seed_identity(connection, "SITE-A", "PRO 777")
        other = seed_identity(connection, "SITE-A", "OTHER 777")
        seed_material(connection, matched.site_id, "PRO 001")
        seed_material(connection, matched.site_id, "PRO 999")
        seed_material(
            connection,
            matched.site_id,
            "CPAN 001",
            source_group="CPAN",
            canonical_alias="PAN",
        )
        ambiguous_material = seed_material(connection, matched.site_id, "PRO 777")
        connection.execute(
            """UPDATE analytical_materials SET sample_id = ?
               WHERE id = ?""",
            (other.sample_id, ambiguous_material),
        )

        # When: exact matching and an externally detected duplicate review are recorded.
        result = run_deterministic_matching(connection, "exact-v1")
        record_review_state(
            connection,
            material_id=ambiguous_material,
            state=MatchState.DUPLICATE,
            rule_version="inventory-v1",
        )
        states = {
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT status FROM match_candidates"
            )
        }

        # Then: every review state is explicit and aliases never become identity.
        assert result.ambiguous == 1
        assert result.unmatched_spectrum == 2
        assert result.unmatched_clinical >= 2
        assert states == {
            "matched",
            "unmatched_clinical",
            "unmatched_spectrum",
            "ambiguous",
            "duplicate",
        }
        assert connection.execute(
            """SELECT COUNT(*) FROM match_candidates AS mc
               JOIN spectrum_material_metadata AS sm
                 ON sm.analytical_material_id = mc.analytical_material_id
               WHERE sm.canonical_source_code = 'CPAN 001'
                 AND mc.status = 'matched'"""
        ).fetchone()[0] == 0
        assert conflicting.subject_id != other.subject_id
        assert unmatched_clinical.subject_id


@pytest.mark.parametrize(
    "path",
    [
        Path("2. Breast cancer/Order 02/Box 03/BRE 017_2.CSV"),
        Path("3. Ovarian cancer/Order 09/Box 04/OVA A070_5.CSV"),
    ],
)
def test_real_shape_numeric_positions_remain_unmatched_without_exact_key(
    tmp_path: Path,
    path: Path,
) -> None:
    # Given: order/box identities exist, but no institution key matches the filename.
    parsed = parse_spectrum_name(path, preparation="liquid")
    with database(tmp_path / f"{parsed.group_code}.db") as connection:
        identity = seed_identity(connection, "SITE-A", "03")
        seed_material(
            connection,
            identity.site_id,
            parsed.source_sample_code,
            source_group=parsed.group_code,
        )

        # When: deterministic site-scoped matching runs.
        result = run_deterministic_matching(connection, "exact-v1")

        # Then: numeric path positions never establish patient identity.
        assert result.matched == 0
        assert result.unmatched_spectrum == 1
        assert result.unmatched_clinical == 1


def test_manual_resolutions_are_append_only(tmp_path: Path) -> None:
    # Given: one ambiguous deterministic candidate.
    with database(tmp_path / "resolution.db") as connection:
        identity = seed_identity(connection, "SITE-A", "PRO 777")
        other = seed_identity(connection, "SITE-A", "OTHER 777")
        material_id = seed_material(connection, identity.site_id, "PRO 777")
        connection.execute(
            """UPDATE analytical_materials SET sample_id = ?
               WHERE id = ?""",
            (other.sample_id, material_id),
        )
        run_deterministic_matching(connection, "exact-v1")
        candidate_id = connection.execute(
            """SELECT id FROM match_candidates
               WHERE analytical_material_id = ? AND status = 'ambiguous'""",
            (material_id,),
        ).fetchone()[0]
        resolution = ManualResolution(
            candidate_id=candidate_id,
            resolution="accepted",
            resolver="reviewer-role",
            rationale="verified clinical source",
            sample_id=other.sample_id,
            clinical_event_id=other.event_id,
        )

        # When: two review decisions are appended.
        first_id = append_manual_resolution(connection, resolution)
        second_id = append_manual_resolution(connection, resolution)

        # Then: neither decision overwrites the other or mutates the candidate.
        assert first_id != second_id
        assert connection.execute(
            "SELECT COUNT(*) FROM match_resolutions"
        ).fetchone()[0] == 2
        assert connection.execute(
            "SELECT status FROM match_candidates WHERE id = ?",
            (candidate_id,),
        ).fetchone()[0] == "ambiguous"
