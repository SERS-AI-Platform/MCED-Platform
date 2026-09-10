from __future__ import annotations

from datetime import datetime
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class HealthResponse(ApiModel):
    status: str
    database: str


class CohortSummary(ApiModel):
    site_code: str
    cohort_group: str | None
    cancer_type: str | None
    study_cancer_type: str | None
    diagnosed_cancer_type: str | None
    case_status: str | None
    subjects: int
    samples: int
    spectra: int


class Spectrum(ApiModel):
    measurement_id: int
    subject_key: str
    sample_key: str
    site_code: str
    cohort_group: str | None
    cancer_type: str | None
    study_cancer_type: str | None
    diagnosed_cancer_type: str | None
    case_status: str | None
    measured_at: datetime
    replicate_number: int
    instrument_name: str
    n_points: int
    x_min: float
    x_max: float
    wavenumber: list[float]
    intensities: list[float]
    # ISUP grade group (1-5) from the subject's latest diagnosis; None when not
    # graded (controls, biopsy-negative, sites that do not record it).
    grade_group: int | None = None


class ReferencePeaks(ApiModel):
    standard_material: str
    calibration_type: str
    tolerance_cm1: float
    reference_peaks_cm1: list[float]
    calibration_date: datetime | None = None


class ReferencePeaksPage(ApiModel):
    items: list[ReferencePeaks]

class SpectrumPage(ApiModel):
    total: int
    limit: int
    offset: int
    items: list[Spectrum]


class SpectrumFilters(ApiModel):
    cohort_group: str | None = Field(default=None, min_length=1, max_length=64)
    cancer_type: str | None = Field(default=None, min_length=1, max_length=128)
    study_cancer_type: str | None = Field(default=None, min_length=1, max_length=128)
    diagnosed_cancer_type: str | None = Field(default=None, min_length=1, max_length=128)
    case_status: str | None = Field(default=None, min_length=1, max_length=64)
    site_code: str | None = Field(default=None, min_length=1, max_length=32)
    limit: int = Field(default=100, ge=1, le=5000)
    offset: int = Field(default=0, ge=0)
