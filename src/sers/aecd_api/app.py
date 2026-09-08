import os
import secrets
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Security, status
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader

from sers.aecd_api.models import (
    CohortSummary,
    HealthResponse,
    ReferencePeaksPage,
    SpectrumFilters,
    SpectrumPage,
)
from sers.aecd_api.repository import (
    AecdRepository,
    DatabaseSettings,
    DatabaseUnavailableError,
    PostgresAecdRepository,
)


def create_app(
    repository: AecdRepository | None = None,
    api_key: str | None = None,
) -> FastAPI:
    data_repository = repository or PostgresAecdRepository(DatabaseSettings.from_environment())
    expected_api_key = api_key or os.environ.get("AECD_API_KEY") or None
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
    application = FastAPI(
        title="AECD Platform Data API",
        description="Read-only, deidentified SERS data access for research notebooks",
        version="0.1.0",
    )

    @application.exception_handler(DatabaseUnavailableError)
    def database_unavailable(
        _request: Request, error: DatabaseUnavailableError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(error)},
        )

    def authorize(provided_key: Annotated[str | None, Security(api_key_header)]) -> None:
        if expected_api_key is None:
            return
        if provided_key is None or not secrets.compare_digest(provided_key, expected_api_key):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    @application.get("/health")
    def health() -> HealthResponse:
        if not data_repository.health():
            raise DatabaseUnavailableError("health check")
        return HealthResponse(status="ok", database="aecd_platform")

    @application.get("/v1/cohorts", dependencies=[Depends(authorize)])
    def cohorts() -> list[CohortSummary]:
        return data_repository.cohort_summary()

    @application.get("/v1/spectra", dependencies=[Depends(authorize)])
    def spectra(filters: Annotated[SpectrumFilters, Query()]) -> SpectrumPage:
        return data_repository.spectra(filters)

    @application.get("/v1/reference_peaks", dependencies=[Depends(authorize)])
    def reference_peaks(
        standard_material: Annotated[str, Query()] = "Polystyrene (PS)",
    ) -> ReferencePeaksPage:
        return data_repository.reference_peaks(standard_material)

    return application


app = create_app()
