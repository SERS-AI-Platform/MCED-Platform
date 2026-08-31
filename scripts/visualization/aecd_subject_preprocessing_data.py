from __future__ import annotations

import json
import os
import socket
from collections import defaultdict
from contextlib import closing
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar, Final
from zoneinfo import ZoneInfo

import httpx2
import numpy as np
import psycopg2
from pydantic import BaseModel, ConfigDict, TypeAdapter

API_URL: Final = os.environ.get("AECD_API_BASE_URL", "http://127.0.0.1:8000")
RANGE_CM1: Final = (400.0, 2200.0)
QC_MAD_THRESHOLD: Final = 3.5
CALIBRATION_ROWS: Final = TypeAdapter(list[tuple[date, str, Decimal]])


class Spectrum(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    measurement_id: int
    subject_key: str
    measured_at: datetime
    replicate_number: int
    instrument_name: str
    wavenumber: list[float]
    intensities: list[float]


class SpectrumPage(BaseModel):
    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    items: list[Spectrum]
    total: int


class DataUnavailableError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SubjectSeries:
    rows: tuple[Spectrum, ...]
    raw_grids: tuple[np.ndarray, ...]
    raw_values: np.ndarray
    corrected_grids: tuple[np.ndarray, ...]
    common_grid: np.ndarray
    aligned_values: np.ndarray
    shifts: np.ndarray
    keep: np.ndarray
    distances: np.ndarray
    qc_limit: float


def api_client() -> httpx2.Client:
    limits = httpx2.Limits(
        max_connections=200,
        max_keepalive_connections=40,
        keepalive_expiry=30.0,
    )
    timeout = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=limits,
        socket_options=[(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)],
    )
    api_key = os.environ.get("AECD_API_KEY")
    headers = {"X-API-Key": api_key} if api_key else {}
    return httpx2.Client(
        base_url=API_URL,
        headers=headers,
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
    )


def load_spectra() -> SpectrumPage:
    with api_client() as client:
        response = client.get("/v1/spectra", params={"limit": 5000, "offset": 0})
        response.raise_for_status()
        return SpectrumPage.model_validate_json(response.content)


def load_calibrations() -> dict[tuple[str, str], float]:
    password = os.environ.get("PGPASSWORD", "")
    if not password:
        raise DataUnavailableError("PGPASSWORD is required for calibration lookup.")
    connection = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "aecd_platform"),
        user=os.environ.get("PGUSER", "postgres"),
        password=password,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    )
    sql = """
        SELECT calibration.calibration_date, instrument.instrument_name,
               calibration.global_shift_cm1
        FROM measurement.calibrations AS calibration
        JOIN measurement.instruments AS instrument USING (instrument_id)
        WHERE calibration.standard_material = 'Polystyrene (PS)'
          AND calibration.calibration_type = 'raman_shift'
          AND calibration.result = 'pass'
          AND calibration.global_shift_cm1 IS NOT NULL
    """
    with closing(connection), connection.cursor() as cursor:
        cursor.execute(sql)
        rows = CALIBRATION_ROWS.validate_python(cursor.fetchall())
    return {(row[0].isoformat(), row[1]): float(row[2]) for row in rows}


def robust_qc(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Calculate standardized RMS distance and a median + 3.5 MAD cutoff."""
    center = np.median(values, axis=0)
    residuals = values - center
    mad_scale = np.median(np.abs(residuals), axis=0)
    scale = np.where(mad_scale > np.finfo(float).eps, 1.4826 * mad_scale, 1.0)
    distances = np.sqrt(np.mean((residuals / scale) ** 2, axis=1))
    distance_center = float(np.median(distances))
    distance_mad = float(np.median(np.abs(distances - distance_center)))
    spread = (
        1.4826 * distance_mad if distance_mad > np.finfo(float).eps else float(np.std(distances))
    )
    limit = distance_center + QC_MAD_THRESHOLD * spread
    keep = distances <= max(limit, distance_center)
    if int(np.count_nonzero(keep)) < 2:
        keep = np.zeros(len(values), dtype=bool)
        keep[np.argsort(distances)[:2]] = True
    return keep, distances, float(limit)


def prepare_subject(
    rows: list[Spectrum],
    calibrations: dict[tuple[str, str], float],
) -> SubjectSeries | None:
    ordered = sorted(rows, key=lambda row: (row.replicate_number, row.measurement_id))
    raw_grids = tuple(np.asarray(row.wavenumber, dtype=np.float64) for row in ordered)
    raw_values = np.vstack([np.asarray(row.intensities, dtype=np.float64) for row in ordered])
    step = float(np.median([np.median(np.diff(grid)) for grid in raw_grids]))
    points = int(round((RANGE_CM1[1] - RANGE_CM1[0]) / step)) + 1
    common_grid = np.linspace(*RANGE_CM1, points, dtype=np.float64)
    corrected: list[np.ndarray] = []
    shifts: list[float] = []
    aligned: list[np.ndarray] = []
    for row, grid, intensity in zip(ordered, raw_grids, raw_values, strict=True):
        shift = calibrations.get((row.measured_at.date().isoformat(), row.instrument_name))
        if shift is None:
            return None
        corrected_grid = grid - shift
        if corrected_grid[0] > common_grid[0] or corrected_grid[-1] < common_grid[-1]:
            return None
        corrected.append(corrected_grid)
        shifts.append(shift)
        aligned.append(np.interp(common_grid, corrected_grid, intensity))
    aligned_values = np.vstack(aligned)
    keep, distances, limit = robust_qc(aligned_values)
    return SubjectSeries(
        rows=tuple(ordered),
        raw_grids=raw_grids,
        raw_values=raw_values,
        corrected_grids=tuple(corrected),
        common_grid=common_grid,
        aligned_values=aligned_values,
        shifts=np.asarray(shifts),
        keep=keep,
        distances=distances,
        qc_limit=limit,
    )


def select_subject(
    page: SpectrumPage,
    calibrations: dict[tuple[str, str], float],
) -> SubjectSeries:
    grouped: defaultdict[str, list[Spectrum]] = defaultdict(list)
    for row in page.items:
        grouped[row.subject_key].append(row)
    complete_count = max(len(rows) for rows in grouped.values())
    candidates = [
        prepared
        for rows in grouped.values()
        if len(rows) == complete_count
        for prepared in [prepare_subject(rows, calibrations)]
        if prepared is not None
    ]
    if not candidates:
        raise DataUnavailableError("No complete subject has matching PS calibration.")
    exclusions = np.asarray([np.count_nonzero(~item.keep) for item in candidates])
    target = float(np.median(exclusions))
    return min(candidates, key=lambda item: abs(np.count_nonzero(~item.keep) - target))


def write_manifest(series: SubjectSeries, api_total: int, output_dir: Path) -> None:
    count = len(series.rows)
    passed = int(np.count_nonzero(series.keep))
    manifest = {
        "source": "AECD API / aecd_platform",
        "observed_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
        "real_clinical_spectra": True,
        "deidentified_subject": True,
        "subject_selection": "complete subject with QC exclusions nearest the candidate median",
        "api_total_observed": api_total,
        "repeats_total": count,
        "repeats_qc_passed": passed,
        "repeats_excluded": count - passed,
        "common_grid_points": len(series.common_grid),
        "common_grid_range_cm1": list(RANGE_CM1),
        "qc_rule": "standardized RMS distance from within-subject median; median + 3.5 × 1.4826 × MAD",
    }
    _ = (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
