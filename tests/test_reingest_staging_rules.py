from __future__ import annotations

import pytest

from sers.reingest_staging_rules import gleason_to_grade_group


@pytest.mark.parametrize(
    ("gleason", "expected_group"),
    [
        ("6(3+3)", 1),
        ("7(3+4)", 2),
        ("7(4+3)", 3),
        ("8(4+4)", 4),
        ("8(3+5)", 4),
        ("8(5+3)", 4),
        ("9(4+5)", 5),
        ("9(5+4)", 5),
        ("10(5+5)", 5),
    ],
)
def test_gleason_patterns_map_to_standard_isup_grade_groups(
    gleason: str,
    expected_group: int,
) -> None:
    # Given: a total Gleason score with its primary and secondary patterns.
    # When: the value is converted to an ISUP Grade Group.
    actual_group = gleason_to_grade_group(gleason)

    # Then: pattern order and high-grade scores use the standard 1-5 mapping.
    assert actual_group == expected_group


@pytest.mark.parametrize("gleason", [None, "", "7", "7(4+4)"])
def test_gleason_grade_group_is_missing_when_patterns_are_not_usable(
    gleason: str | None,
) -> None:
    # Given: a missing, ambiguous, or internally inconsistent Gleason value.
    # When: grade-group conversion is attempted.
    actual_group = gleason_to_grade_group(gleason)

    # Then: the converter does not invent a clinical grade.
    assert actual_group is None
