from __future__ import annotations

from datetime import datetime, timezone
from types import TracebackType
from typing import cast

import psycopg2
import pytest
from fastapi.testclient import TestClient

from sers.aecd_api.app import create_app
from sers.aecd_api.models import CohortSummary, Spectrum, SpectrumFilters, SpectrumPage
from sers.aecd_api.repository import DatabaseSettings, PostgresAecdRepository


class DeployedSchemaCursor:
    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple[str | int, ...]]] = []

    def __enter__(self) -> DeployedSchemaCursor:
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        return None

    def execute(self, query: str, parameters: tuple[str | int, ...]) -> None:
        self.executions.append((query, parameters))
        if "to_jsonb(clinical.diagnoses)" in query:
            raise psycopg2.errors.UndefinedTable("schema-qualified row reference is invalid")
        if "run.measured_at" in query or "measurement.replicate_number" in query:
            raise psycopg2.errors.UndefinedColumn("deployed measurement column is missing")

    def fetchone(self) -> tuple[int]:
        return (1,)

    def fetchall(
        self,
    ) -> list[
        tuple[
            int,
            int,
            int,
            str,
            str,
            str,
            str | None,
            str,
            datetime,
            int,
            str,
            int,
            float,
            float,
            list[float],
            list[float],
            int | None,
        ]
    ]:
        return [
            (
                101,
                7,
                11,
                "SITE-A",
                "PRO",
                "prostate",
                None,
                "healthy_control",
                datetime(2026, 8, 19, tzinfo=timezone.utc),
                1,
                "SERS-01",
                3,
                400.0,
                402.0,
                [400.0, 401.0, 402.0],
                [0.1, 0.2, 0.3],
                3,
            )
        ]


class DeployedSchemaConnection:
    def __init__(self) -> None:
        self.cursor_instance: DeployedSchemaCursor = DeployedSchemaCursor()

    def cursor(self) -> DeployedSchemaCursor:
        return self.cursor_instance

    def close(self) -> None:
        return None


class FakeRepository:
    def health(self) -> bool:
        return True

    def cohort_summary(self) -> list[CohortSummary]:
        return [
            CohortSummary(
                site_code="SITE-A",
                cohort_group="PRO",
                cancer_type="prostate",
                study_cancer_type="prostate",
                diagnosed_cancer_type=None,
                case_status="healthy_control",
                subjects=2,
                samples=3,
                spectra=15,
            )
        ]

    def spectra(self, filters: SpectrumFilters) -> SpectrumPage:
        assert filters.cohort_group == "PRO"
        assert filters.cancer_type is None
        assert filters.site_code is None
        assert filters.limit == 10
        assert filters.offset == 0
        return SpectrumPage(
            total=1,
            limit=filters.limit,
            offset=filters.offset,
            items=[
                Spectrum(
                    measurement_id=101,
                    subject_key="subject:7",
                    sample_key="sample:11",
                    site_code="SITE-A",
                    cohort_group="PRO",
                    cancer_type="prostate",
                    study_cancer_type="prostate",
                    diagnosed_cancer_type=None,
                    case_status="healthy_control",
                    measured_at=datetime(2026, 8, 19, tzinfo=timezone.utc),
                    replicate_number=1,
                    instrument_name="SERS-01",
                    n_points=3,
                    x_min=400.0,
                    x_max=402.0,
                    wavenumber=[400.0, 401.0, 402.0],
                    intensities=[0.1, 0.2, 0.3],
                )
            ],
        )


def test_spectra_returns_page_with_deployed_measurement_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    connection = DeployedSchemaConnection()

    def connect(**_kwargs: str | int) -> DeployedSchemaConnection:
        return connection

    monkeypatch.setattr(psycopg2, "connect", connect)
    repository = PostgresAecdRepository(
        DatabaseSettings(
            host="localhost",
            port=5432,
            database="aecd_platform",
            user="postgres",
            password="test-secret",
        )
    )

    # When
    page = repository.spectra(SpectrumFilters(limit=1, offset=0))

    # Then
    assert page.total == 1
    assert page.items[0].replicate_number == 1
    assert page.items[0].grade_group == 3


def test_health_when_repository_is_available() -> None:
    # Given
    client = TestClient(create_app(FakeRepository()))

    # When
    response = client.get("/health")

    # Then
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "aecd_platform"}


def test_cohort_summary_returns_modeling_counts() -> None:
    # Given
    client = TestClient(create_app(FakeRepository()))

    # When
    response = client.get("/v1/cohorts")

    # Then
    assert response.status_code == 200
    assert response.json()[0]["spectra"] == 15
    assert response.json()[0]["study_cancer_type"] == "prostate"
    assert response.json()[0]["diagnosed_cancer_type"] is None
    assert response.json()[0]["case_status"] == "healthy_control"


def test_spectra_returns_deidentified_filtered_batch() -> None:
    # Given
    client = TestClient(create_app(FakeRepository()))

    # When
    response = client.get("/v1/spectra", params={"cohort_group": "PRO", "limit": 10})

    # Then
    assert response.status_code == 200
    payload = cast(dict[str, object], response.json())
    items = cast(list[dict[str, object]], payload["items"])
    assert payload["total"] == 1
    assert items[0]["subject_key"] == "subject:7"
    assert items[0]["study_cancer_type"] == "prostate"
    assert items[0]["diagnosed_cancer_type"] is None
    assert items[0]["case_status"] == "healthy_control"
    assert "patient_code" not in items[0]
    assert "source_uri" not in items[0]


def test_spectra_rejects_oversized_batch() -> None:
    # Given
    client = TestClient(create_app(FakeRepository()))

    # When: the request exceeds SpectrumFilters.limit's upper bound (le=5000).
    response = client.get("/v1/spectra", params={"limit": 5001})

    # Then
    assert response.status_code == 422


def test_data_routes_require_configured_api_key() -> None:
    # Given
    client = TestClient(create_app(FakeRepository(), api_key="test-secret"))

    # When
    unauthorized = client.get("/v1/cohorts")
    authorized = client.get("/v1/cohorts", headers={"X-API-Key": "test-secret"})

    # Then
    assert unauthorized.status_code == 401
    assert authorized.status_code == 200
