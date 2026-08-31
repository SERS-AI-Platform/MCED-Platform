from __future__ import annotations

# pyright: reportUnnecessaryComparison=false
import os
from dataclasses import dataclass
from enum import Enum
from getpass import getpass
from pathlib import Path
from typing import Annotated, Final, TypeAlias

import typer
from psycopg2 import connect
from psycopg2.extensions import cursor as PgCursor
from pydantic import TypeAdapter
from typing_extensions import assert_never

from .aecd_raw_spectra_ingest import (
    Discovery,
    IngestError,
    SpectrumKey,
    discover_spectrum_files,
    read_spectrum,
    sha256_file,
    source_mapping_from_notes,
    validate_files,
)

DEFAULT_MAPPING_ROOT: Final = Path("/home/user/SERS-AI/data/mapping")
DEFAULT_BATCH_SIZE: Final = 25
INSERT_SQL: Final = """
    INSERT INTO measurement.raw_spectra (
        measurement_id, source_uri, raw_filename, source_kind,
        wavenumber, intensities, n_points, x_min, x_max, source_sha256
    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
"""
RawSpectrumRow: TypeAlias = tuple[
    int,
    str,
    str,
    str,
    list[float],
    list[float],
    int,
    float,
    float,
    str,
]
MeasurementRow: TypeAlias = tuple[int, str, str, int]
MEASUREMENT_ROWS_ADAPTER: Final = TypeAdapter(list[MeasurementRow])
COUNT_ROW_ADAPTER: Final = TypeAdapter(tuple[int])
FLOAT_LIST_ADAPTER: Final = TypeAdapter(list[float])


class RunMode(str, Enum):
    FILES = "files"
    CHECK_DB = "check-db"
    WRITE = "write"


@dataclass(frozen=True, slots=True)
class DatabaseConfig:
    host: str
    port: int
    database: str
    user: str
    password: str


def database_config_from_env() -> DatabaseConfig:
    raw_port = os.environ.get("PGPORT", "5432")
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise IngestError(f"PGPORT must be an integer: {raw_port}") from exc
    password = os.environ.get("PGPASSWORD") or getpass("PostgreSQL password: ")
    if not password:
        raise IngestError("PostgreSQL password is required")
    return DatabaseConfig(
        host=os.environ.get("PGHOST", "localhost"),
        port=port,
        database=os.environ.get("PGDATABASE", "aecd_platform"),
        user=os.environ.get("PGUSER", "postgres"),
        password=password,
    )


def _load_measurement_lookup(cursor: PgCursor) -> dict[SpectrumKey, int]:
    cursor.execute(
        """
        SELECT m.measurement_id, r.notes, s.solum_label, m.point_no
        FROM measurement.measurements AS m
        JOIN measurement.runs AS r
          ON r.measurement_run_id = m.measurement_run_id
        JOIN master.samples AS s
          ON s.sample_id = m.sample_id
        WHERE m.measurement_role = 'clinical'
        """
    )
    lookup: dict[SpectrumKey, int] = {}
    rows = MEASUREMENT_ROWS_ADAPTER.validate_python(cursor.fetchall())
    for measurement_id, notes, solum_label, point_no in rows:
        key = SpectrumKey(source_mapping_from_notes(notes), solum_label, point_no)
        if key in lookup:
            raise IngestError(f"duplicate DB measurement key: {key}")
        lookup[key] = measurement_id
    return lookup


def check_or_write_database(
    discovery: Discovery,
    config: DatabaseConfig,
    write: bool,
) -> tuple[int, int]:
    with connect(
        host=config.host,
        port=config.port,
        dbname=config.database,
        user=config.user,
        password=config.password,
        connect_timeout=10,
    ) as connection:
        with connection.cursor() as cursor:
            lookup = _load_measurement_lookup(cursor)
            missing = tuple(item.key for item in discovery.files if item.key not in lookup)
            if missing:
                raise IngestError(f"files without DB measurement: {missing[:5]}")
            measurement_ids = [lookup[item.key] for item in discovery.files]
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM measurement.raw_spectra
                WHERE measurement_id = ANY(%s)
                """,
                (measurement_ids,),
            )
            count_row = cursor.fetchone()
            if count_row is None:
                raise IngestError("raw spectrum count query returned no row")
            existing_count = COUNT_ROW_ADAPTER.validate_python(count_row)[0]
            if not write:
                connection.rollback()
                return len(measurement_ids), existing_count
            if existing_count:
                raise IngestError(f"raw spectra already exist for {existing_count} measurements")
            for offset in range(0, len(discovery.files), DEFAULT_BATCH_SIZE):
                rows: list[RawSpectrumRow] = []
                for item in discovery.files[offset : offset + DEFAULT_BATCH_SIZE]:
                    data = read_spectrum(item.path)
                    wavenumbers = FLOAT_LIST_ADAPTER.validate_python(
                        data.wavenumber.tolist()
                    )
                    intensities = FLOAT_LIST_ADAPTER.validate_python(
                        data.intensities.tolist()
                    )
                    rows.append(
                        (
                            lookup[item.key],
                            item.path.resolve().as_uri(),
                            item.path.name,
                            "replicate",
                            wavenumbers,
                            intensities,
                            len(wavenumbers),
                            wavenumbers[0],
                            wavenumbers[-1],
                            sha256_file(item.path),
                        )
                    )
                cursor.executemany(INSERT_SQL, rows)
    return len(discovery.files), 0


def main(
    mapping_root: Annotated[Path, typer.Option("--mapping-root")] = DEFAULT_MAPPING_ROOT,
    expected_points: Annotated[int, typer.Option("--expected-points-per-sample")] = 121,
    mode: Annotated[RunMode, typer.Option("--mode")] = RunMode.FILES,
) -> None:
    discovery = discover_spectrum_files(mapping_root, expected_points)
    validation = validate_files(discovery, mapping_root)
    typer.echo(
        {
            "samples": discovery.sample_count,
            "point_files": len(discovery.files),
            "averages_excluded": discovery.averages_excluded,
            "spectrum_lengths": dict(validation.point_lengths),
            "manifest_sha256": validation.manifest_sha256,
        }
    )
    match mode:
        case RunMode.FILES:
            return
        case RunMode.CHECK_DB | RunMode.WRITE:
            matched, existing = check_or_write_database(
                discovery,
                database_config_from_env(),
                write=mode is RunMode.WRITE,
            )
            typer.echo(
                {"db_measurements_matched": matched, "existing_raw_spectra": existing}
            )
        case unreachable:
            assert_never(unreachable)


if __name__ == "__main__":
    typer.run(main)
