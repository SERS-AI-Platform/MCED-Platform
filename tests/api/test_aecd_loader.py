from __future__ import annotations

from datetime import datetime, timezone

from sers.aecd_api.loader import load_spectra_from_aecd
from sers.aecd_api.models import CohortSummary, Spectrum, SpectrumFilters, SpectrumPage


def _spectrum(measurement_id: int, sample_key: str, cohort_group: str | None, replicate: int) -> Spectrum:
    return Spectrum(
        measurement_id=measurement_id,
        subject_key="subject:1",
        sample_key=sample_key,
        site_code="SITE-A",
        cohort_group=cohort_group,
        cancer_type="prostate",
        study_cancer_type="prostate",
        diagnosed_cancer_type=None,
        case_status="healthy_control",
        measured_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        replicate_number=replicate,
        instrument_name="SERS-01",
        n_points=3,
        x_min=400.0,
        x_max=402.0,
        wavenumber=[400.0, 401.0, 402.0],
        intensities=[0.1, 0.2, 0.3],
    )


class PagedFakeRepository:
    """Two-page fake: exercises pagination, and one record with a missing cohort_group."""

    def __init__(self) -> None:
        self._pages = [
            [
                _spectrum(101, "sample:11", "PRO", 1),
                _spectrum(102, "sample:11", "PRO", 2),
            ],
            [
                _spectrum(103, "sample:12", None, 1),
            ],
        ]

    def health(self) -> bool:
        return True

    def cohort_summary(self) -> list[CohortSummary]:
        return []

    def spectra(self, filters: SpectrumFilters) -> SpectrumPage:
        total = sum(len(page) for page in self._pages)
        page_index = filters.offset // 2
        items = self._pages[page_index] if page_index < len(self._pages) else []
        return SpectrumPage(total=total, limit=filters.limit, offset=filters.offset, items=items)


def test_load_spectra_from_aecd_paginates_and_builds_expected_key_shape() -> None:
    # Given
    repository = PagedFakeRepository()
    filters = SpectrumFilters(cohort_group="PRO", limit=2, offset=0)

    # When
    result = load_spectra_from_aecd(repository, filters, page_size=2)

    # Then
    assert set(result.spectra.keys()) == {("PRO", 11, 1), ("PRO", 11, 2)}
    x, y = result.spectra[("PRO", 11, 1)]
    assert x.tolist() == [400.0, 401.0, 402.0]
    assert y.tolist() == [0.1, 0.2, 0.3]
    assert result.measurement_ids == [101, 102]


def test_load_spectra_from_aecd_skips_missing_cohort_group_without_fabricating_one() -> None:
    # Given
    repository = PagedFakeRepository()
    filters = SpectrumFilters(limit=2, offset=0)

    # When
    result = load_spectra_from_aecd(repository, filters, page_size=2)

    # Then
    assert 103 in result.skipped_missing_group
    assert all(key[0] != "UNK" for key in result.spectra)
    assert 103 not in result.measurement_ids
