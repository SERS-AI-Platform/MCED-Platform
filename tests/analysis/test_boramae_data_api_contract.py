"""Contract test: the Boramae publication loader against the real AECD API app.

No database and no HTTP server: `create_app(FakeRepository())` is served through
FastAPI's TestClient, which the publication's `AecdApiClient` accepts as its
`http` client. Anything the loader assumes about the API (auth header, paging
fields, response keys, cohort codes) is therefore checked against the app itself.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sers.aecd_api.app import create_app
from sers.aecd_api.models import (
    CohortSummary,
    ReferencePeaksPage,
    Spectrum,
    SpectrumFilters,
    SpectrumPage,
)

SRC = Path(__file__).resolve().parents[2] / "publications" / "전향검체" / "보라매병원" / "src"
sys.path.insert(0, str(SRC))

import boramae_data  # noqa: E402

GRID = np.linspace(600.0, 1700.0, 12)
X = np.linspace(400.0, 1800.0, 30)

# (subject id, cohort_group, grade_group)
SUBJECTS = [
    (1, "control", None),
    (2, "prostate disease control", None),
    (3, "prostate", 3),
    (4, "Drop", None),
]
REPLICATES = 2


def _spectrum(subject: int, cohort_group: str, grade: int | None, replicate: int) -> Spectrum:
    return Spectrum(
        measurement_id=subject * 10 + replicate,
        subject_key=f"subject:{subject}",
        sample_key=f"sample:{subject}",
        site_code="BORAMAE",
        cohort_group=cohort_group,
        cancer_type="prostate",
        study_cancer_type="prostate",
        diagnosed_cancer_type=None,
        case_status=None,
        measured_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
        replicate_number=replicate,
        instrument_name="SERS-01",
        n_points=len(X),
        x_min=float(X[0]),
        x_max=float(X[-1]),
        wavenumber=X.tolist(),
        intensities=(np.sin(X / 100.0) + subject + replicate * 0.1).tolist(),
        grade_group=grade,
    )


class FakeRepository:
    def __init__(self) -> None:
        self.items = [
            _spectrum(subject, group, grade, replicate)
            for subject, group, grade in SUBJECTS
            for replicate in range(1, REPLICATES + 1)
        ]
        self.requests: list[SpectrumFilters] = []

    def health(self) -> bool:
        return True

    def cohort_summary(self) -> list[CohortSummary]:
        return []

    def reference_peaks(self, standard_material: str) -> ReferencePeaksPage:
        return ReferencePeaksPage(items=[])

    def spectra(self, filters: SpectrumFilters) -> SpectrumPage:
        self.requests.append(filters)
        selected = [
            item
            for item in self.items
            if (filters.site_code is None or item.site_code == filters.site_code)
            and (filters.cohort_group is None or item.cohort_group == filters.cohort_group)
        ]
        window = selected[filters.offset : filters.offset + filters.limit]
        return SpectrumPage(total=len(selected), limit=filters.limit, offset=filters.offset, items=window)


@pytest.fixture
def repository() -> FakeRepository:
    return FakeRepository()


@pytest.fixture
def client(repository: FakeRepository) -> boramae_data.AecdApiClient:
    return boramae_data.AecdApiClient(http=TestClient(create_app(repository)))


def test_load_clinical_samples_maps_every_cohort_code(client: boramae_data.AecdApiClient) -> None:
    samples = boramae_data.load_clinical_samples(client=client)

    assert [(s.label, s.group, s.grade_group, s.excluded) for s in samples] == [
        ("subject:1", "Control", None, False),
        ("subject:2", "Biopsy-negative", None, False),
        ("subject:3", "Prostate cancer", 3, False),
        ("subject:4", "Excluded", None, True),
    ]
    # the API is deidentified: no source sample code can leak into publication outputs
    assert all(not s.label.startswith(("BPRO", "BNOR")) for s in samples)


def test_build_subjects_skips_dropped_and_averages_replicates(
    client: boramae_data.AecdApiClient,
) -> None:
    samples = boramae_data.load_clinical_samples(client=client)

    subjects = boramae_data.build_boramae_subjects(samples, GRID, client=client)

    assert [s.sample.label for s in subjects] == ["subject:1", "subject:2", "subject:3"]
    for subject in subjects:
        assert subject.replicate_spectra.shape == (REPLICATES, len(GRID))
        assert subject.mean_spectrum.shape == (len(GRID),)
        np.testing.assert_allclose(subject.mean_spectrum, subject.replicate_spectra.mean(axis=0))


def test_client_pages_with_limit_and_offset(
    repository: FakeRepository, client: boramae_data.AecdApiClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(boramae_data, "PAGE_SIZE", 3)

    records = list(client.iter_spectra(site_code="BORAMAE"))

    assert len(records) == len(SUBJECTS) * REPLICATES
    assert [(f.limit, f.offset) for f in repository.requests] == [(3, 0), (3, 3), (3, 6)]
    assert all(f.site_code == "BORAMAE" for f in repository.requests)


def test_client_sends_the_api_key_header_the_app_expects(repository: FakeRepository) -> None:
    app = create_app(repository, api_key="test-secret")

    authorized = boramae_data.AecdApiClient(api_key="test-secret", http=TestClient(app))
    anonymous = boramae_data.AecdApiClient(api_key=None, http=TestClient(app))

    assert len(list(authorized.iter_spectra())) == len(SUBJECTS) * REPLICATES
    with pytest.raises(Exception, match="401"):
        list(anonymous.iter_spectra())


def test_unknown_cohort_code_fails_loudly(client: boramae_data.AecdApiClient, repository: FakeRepository) -> None:
    repository.items.append(_spectrum(9, "biopsy_negative", None, 1))

    with pytest.raises(RuntimeError, match="biopsy_negative"):
        boramae_data.load_clinical_samples(client=client)
