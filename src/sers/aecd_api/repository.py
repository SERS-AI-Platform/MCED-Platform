from __future__ import annotations

import os
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, TypeAlias

import psycopg2
from psycopg2.extensions import connection as PgConnection
from pydantic import TypeAdapter
from typing_extensions import override

from sers.aecd_api.models import (
    CohortSummary,
    ReferencePeaks,
    ReferencePeaksPage,
    Spectrum,
    SpectrumFilters,
    SpectrumPage,
)

REFERENCE_PEAK_ROWS: Final = TypeAdapter(list[tuple[str, str, float, list[float], datetime]])


CohortRow: TypeAlias = tuple[str, str | None, str | None, str | None, str | None, int, int, int]
SpectrumRow: TypeAlias = tuple[
    int,
    int,
    int,
    str,
    str | None,
    str | None,
    str | None,
    str | None,
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
COHORT_ROWS: Final = TypeAdapter(list[CohortRow])
SPECTRUM_ROWS: Final = TypeAdapter(list[SpectrumRow])
COUNT_ROW: Final = TypeAdapter(tuple[int])


@dataclass(frozen=True, slots=True)
class DatabaseSettings:
    host: str
    port: int
    database: str
    user: str
    password: str

    @classmethod
    def from_environment(cls) -> DatabaseSettings:
        return cls(
            host=os.environ.get("PGHOST", "localhost"),
            port=int(os.environ.get("PGPORT", "5432")),
            database=os.environ.get("PGDATABASE", "aecd_platform"),
            user=os.environ.get("PGUSER", "postgres"),
            password=os.environ.get("PGPASSWORD", ""),
        )


@dataclass(frozen=True, slots=True)
class DatabaseUnavailableError(Exception):
    operation: str

    @override
    def __str__(self) -> str:
        return f"aecd_platform database unavailable during {self.operation}"


class AecdRepository(Protocol):
    def health(self) -> bool: ...

    def cohort_summary(self) -> list[CohortSummary]: ...

    def spectra(self, filters: SpectrumFilters) -> SpectrumPage: ...


class PostgresAecdRepository:
    def __init__(self, settings: DatabaseSettings) -> None:
        self._settings: DatabaseSettings = settings

    def _connect(self) -> PgConnection:
        return psycopg2.connect(
            host=self._settings.host,
            port=self._settings.port,
            dbname=self._settings.database,
            user=self._settings.user,
            password=self._settings.password,
            connect_timeout=5,
            options="-c default_transaction_read_only=on -c statement_timeout=30000",
        )

    def health(self) -> bool:
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                cursor.execute("SELECT current_database() = 'aecd_platform'")
                row = cursor.fetchone()
        except psycopg2.Error as error:
            raise DatabaseUnavailableError("health check") from error
        return bool(row and row[0])

    def cohort_summary(self) -> list[CohortSummary]:
        study_type = "COALESCE(to_jsonb(diagnosis)->>'study_cancer_type', diagnosis.cancer_type)"
        diagnosed_type = (
            "CASE WHEN to_jsonb(diagnosis) ? 'diagnosed_cancer_type' "
            "THEN to_jsonb(diagnosis)->>'diagnosed_cancer_type' "
            "ELSE diagnosis.cancer_type END"
        )
        query = (
            f"SELECT site.site_code, diagnosis.cohort_group, {study_type}, "
            f"{diagnosed_type}, to_jsonb(diagnosis)->>'case_status', "
            "COUNT(DISTINCT subject.subject_id), COUNT(DISTINCT sample.sample_id), "
            "COUNT(DISTINCT measurement.measurement_id) FROM master.sites AS site "
            "JOIN master.subjects AS subject USING (site_id) "
            "JOIN master.samples AS sample USING (subject_id) "
            "JOIN measurement.measurements AS measurement USING (sample_id) "
            "JOIN measurement.raw_spectra AS spectrum USING (measurement_id) "
            "LEFT JOIN clinical.diagnoses AS diagnosis USING (subject_id) "
            "WHERE measurement.status = 'acquired' "
            "AND measurement.measurement_role = 'clinical' "
            "GROUP BY 1, 2, 3, 4, 5 ORDER BY 1, 2, 3, 4, 5"
        )
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                cursor.execute(query)
                rows = COHORT_ROWS.validate_python(cursor.fetchall())
        except psycopg2.Error as error:
            raise DatabaseUnavailableError("cohort summary") from error
        return [
            CohortSummary(
                site_code=row[0],
                cohort_group=row[1],
                cancer_type=row[2],
                study_cancer_type=row[2],
                diagnosed_cancer_type=row[3],
                case_status=row[4],
                subjects=row[5],
                samples=row[6],
                spectra=row[7],
            )
            for row in rows
        ]

    def reference_peaks(self, standard_material: str) -> ReferencePeaksPage:
        query = (
            "SELECT standard_material, calibration_type, tolerance_cm1, "
            "reference_peaks_cm1, calibration_date "
            "FROM measurement.calibrations "
            "WHERE standard_material = %s "
            "AND calibration_type = 'raman_shift' "
            "AND result = 'pass' "
            "ORDER BY calibration_date DESC "
            "LIMIT 1"
        )
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                cursor.execute(query, (standard_material,))
                rows = REFERENCE_PEAK_ROWS.validate_python(cursor.fetchall())
        except psycopg2.Error as error:
            raise DatabaseUnavailableError("reference peaks") from error
        return ReferencePeaksPage(
            items=[
                ReferencePeaks(
                    standard_material=row[0],
                    calibration_type=row[1],
                    tolerance_cm1=row[2],
                    reference_peaks_cm1=list(row[3]),
                    calibration_date=row[4],
                )
                for row in rows
            ]
        )
    def spectra(self, filters: SpectrumFilters) -> SpectrumPage:
        study_type = "COALESCE(diagnosis.payload->>'study_cancer_type', diagnosis.cancer_type)"
        diagnosed_type = (
            "CASE WHEN diagnosis.payload ? 'diagnosed_cancer_type' "
            "THEN diagnosis.payload->>'diagnosed_cancer_type' "
            "ELSE diagnosis.cancer_type END"
        )
        clauses = [
            "measurement.status = 'acquired'",
            "measurement.measurement_role = 'clinical'",
        ]
        parameters: list[str | int] = []
        for column, value in (
            ("diagnosis.cohort_group", filters.cohort_group),
            (study_type, filters.cancer_type),
            (study_type, filters.study_cancer_type),
            (diagnosed_type, filters.diagnosed_cancer_type),
            ("diagnosis.payload->>'case_status'", filters.case_status),
            ("site.site_code", filters.site_code),
        ):
            if value is not None:
                clauses.append(f"{column} = %s")
                parameters.append(value)
        from_sql = (
            " FROM measurement.raw_spectra AS spectrum "
            "JOIN measurement.measurements AS measurement USING (measurement_id) "
            "JOIN measurement.runs AS run USING (measurement_run_id) "
            "JOIN measurement.instruments AS instrument USING (instrument_id) "
            "JOIN master.samples AS sample USING (sample_id) "
            "JOIN master.subjects AS subject USING (subject_id) "
            "JOIN master.sites AS site USING (site_id) "
            "LEFT JOIN LATERAL (SELECT latest.cohort_group, latest.cancer_type, "
            "to_jsonb(latest) AS payload FROM clinical.diagnoses AS latest "
            "WHERE latest.subject_id = subject.subject_id "
            "ORDER BY latest.diagnosis_id DESC LIMIT 1) AS diagnosis ON TRUE WHERE "
        )
        where_sql = " AND ".join(clauses)
        count_query = "SELECT COUNT(*) " + from_sql + where_sql
        data_query = f"""SELECT measurement.measurement_id, subject.subject_id,
            sample.sample_id, site.site_code, diagnosis.cohort_group, {study_type},
            {diagnosed_type}, diagnosis.payload->>'case_status', run.measurement_date,
            measurement.point_no, instrument.instrument_name, spectrum.n_points,
            spectrum.x_min, spectrum.x_max, spectrum.wavenumber, spectrum.intensities,
            (diagnosis.payload->>'grade_group')::int
            {from_sql}{where_sql}
            ORDER BY measurement.measurement_id LIMIT %s OFFSET %s"""
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                cursor.execute(count_query, tuple(parameters))
                total = COUNT_ROW.validate_python(cursor.fetchone())[0]
                cursor.execute(data_query, (*parameters, filters.limit, filters.offset))
                rows = SPECTRUM_ROWS.validate_python(cursor.fetchall())
        except psycopg2.Error as error:
            raise DatabaseUnavailableError("spectrum query") from error
        items = [
            Spectrum(
                measurement_id=row[0],
                subject_key=f"subject:{row[1]}",
                sample_key=f"sample:{row[2]}",
                site_code=row[3],
                cohort_group=row[4],
                cancer_type=row[5],
                study_cancer_type=row[5],
                diagnosed_cancer_type=row[6],
                case_status=row[7],
                measured_at=row[8],
                replicate_number=row[9],
                instrument_name=row[10],
                n_points=row[11],
                x_min=row[12],
                x_max=row[13],
                wavenumber=row[14],
                intensities=row[15],
                grade_group=row[16],
            )
            for row in rows
        ]
        return SpectrumPage(total=total, limit=filters.limit, offset=filters.offset, items=items)
