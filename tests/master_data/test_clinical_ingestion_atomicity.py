from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import pytest

from sers.master_data import clinical_ingestion, clinical_persistence
from sers.master_data.clinical_ingestion import (
    ClinicalIngestionRequest,
    ingest_clinical_registry,
    ingest_clinical_source,
)
from sers.master_data.clinical_parser import MissingSubjectKeyError
from sers.master_data.clinical_types import ClinicalFormat
from sers.master_data.raw_store import store_raw_file
from tests.master_data.clinical_test_support import (
    database,
    raw_blobs,
    source,
    write_csv,
)


@dataclass(frozen=True, slots=True)
class InjectedClinicalFailure(RuntimeError):
    stage: str

    def __str__(self) -> str:
        return f"synthetic {self.stage} interruption"


def _interrupt_persistence(_connection, _site_id, _source_key) -> NoReturn:
    raise InjectedClinicalFailure("persistence")


def _interrupt_parser(_source) -> NoReturn:
    raise InjectedClinicalFailure("parser")


def test_database_failure_rolls_back_the_whole_source(tmp_path: Path) -> None:
    # Given: a valid source and a database failure during observation persistence.
    source_path = tmp_path / "atomic.csv"
    write_csv(source_path, ("SYNTHETIC-005,V1,1.0,x",))
    raw_store = tmp_path / "raw"
    blobs_before = raw_blobs(raw_store)
    with database(tmp_path / "clinical.db") as connection:
        connection.execute(
            """CREATE TRIGGER reject_synthetic_observation
               BEFORE INSERT ON clinical_observations
               BEGIN
                   SELECT RAISE(ABORT, 'synthetic persistence failure');
               END"""
        )

        # When: ingestion is interrupted after parent rows would have been created.
        with pytest.raises(sqlite3.IntegrityError):
            ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(
                    source(source_path, "SITE-A", ClinicalFormat.CSV),
                    raw_store,
                ),
            )

        # Then: the savepoint exposes no partial site, asset, subject, event, or batch.
        assert tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "sites",
                "source_assets",
                "ingest_batches",
                "subjects",
                "clinical_events",
                "clinical_observations",
            )
        ) == (0, 0, 0, 0, 0, 0)
        assert raw_blobs(raw_store) == blobs_before


def test_runtime_failure_closes_savepoint_and_removes_new_raw_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a typed runtime interruption after asset persistence begins.
    source_path = tmp_path / "runtime.csv"
    raw_store = tmp_path / "raw"
    write_csv(source_path, ("SYNTHETIC-006,V1,1.0,x",))
    blobs_before = raw_blobs(raw_store)
    monkeypatch.setattr(
        clinical_persistence,
        "resolve_or_create_subject",
        _interrupt_persistence,
    )

    # When: the caller catches the typed persistence exception.
    with database(tmp_path / "clinical.db") as connection:
        with pytest.raises(InjectedClinicalFailure, match="persistence"):
            ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(
                    source(source_path, "SITE-A", ClinicalFormat.CSV),
                    raw_store,
                ),
            )

        # Then: no savepoint, partial database row, or new raw blob remains.
        assert not connection.in_transaction
        assert tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "sites",
                "source_assets",
                "ingest_batches",
                "subjects",
                "clinical_events",
                "clinical_observations",
            )
        ) == (0, 0, 0, 0, 0, 0)
        assert raw_blobs(raw_store) == blobs_before


def test_runtime_failure_preserves_preexisting_content_addressed_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the source bytes already exist in the shared content-addressed store.
    source_path = tmp_path / "shared.csv"
    raw_store = tmp_path / "raw"
    write_csv(source_path, ("SYNTHETIC-007,V1,1.0,x",))
    store_raw_file(source_path, raw_store)
    blobs_before = raw_blobs(raw_store)
    monkeypatch.setattr(
        clinical_persistence,
        "resolve_or_create_subject",
        _interrupt_persistence,
    )

    # When: persistence fails after reusing the existing raw blob.
    with database(tmp_path / "clinical.db") as connection:
        with pytest.raises(InjectedClinicalFailure, match="persistence"):
            ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(
                    source(source_path, "SITE-A", ClinicalFormat.CSV),
                    raw_store,
                ),
            )

        # Then: rollback removes database rows without deleting the shared blob.
        assert not connection.in_transaction
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0
        assert raw_blobs(raw_store) == blobs_before


def test_parser_runtime_failure_releases_savepoint_before_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a parser hook that raises a typed runtime exception.
    source_path = tmp_path / "parser.csv"
    raw_store = tmp_path / "raw"
    write_csv(source_path, ("SYNTHETIC-008,V1,1.0,x",))
    monkeypatch.setattr(
        clinical_ingestion,
        "load_clinical_rows",
        _interrupt_parser,
    )

    # When: the parser exception propagates to the caller.
    with database(tmp_path / "clinical.db") as connection:
        with pytest.raises(InjectedClinicalFailure, match="parser"):
            ingest_clinical_source(
                connection,
                ClinicalIngestionRequest(
                    source(source_path, "SITE-A", ClinicalFormat.CSV),
                    raw_store,
                ),
            )

        # Then: its savepoint is closed and raw publication never started.
        assert not connection.in_transaction
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0
        assert raw_blobs(raw_store) == set()


def test_registry_invalid_second_source_rolls_back_first_source_and_blob(
    tmp_path: Path,
) -> None:
    # Given: a valid first source followed by a source with no subject key.
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    raw_store = tmp_path / "raw"
    write_csv(first_path, ("SYNTHETIC-REG-1,V1,1.0,valid",))
    write_csv(second_path, (",V1,2.0,invalid",))

    # When: both sources cross the direct registry ingestion boundary.
    with database(tmp_path / "clinical.db") as connection:
        statements: list[str] = []
        connection.set_trace_callback(statements.append)
        with pytest.raises(MissingSubjectKeyError):
            ingest_clinical_registry(
                connection,
                (
                    ClinicalIngestionRequest(
                        source(first_path, "SITE-A", ClinicalFormat.CSV),
                        raw_store,
                    ),
                    ClinicalIngestionRequest(
                        source(second_path, "SITE-B", ClinicalFormat.CSV),
                        raw_store,
                    ),
                ),
            )

        # Then: one outer savepoint rolls back every row and the first raw blob.
        assert sum(
            statement.startswith("SAVEPOINT clinical_registry_")
            for statement in statements
        ) == 1
        assert tuple(
            connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in (
                "sites",
                "source_assets",
                "ingest_batches",
                "subjects",
                "clinical_events",
                "clinical_observations",
            )
        ) == (0, 0, 0, 0, 0, 0)
        assert raw_blobs(raw_store) == set()


def test_registry_interruption_rolls_back_all_and_preserves_shared_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: two valid sources and a preexisting blob shared with the first source.
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    raw_store = tmp_path / "raw"
    write_csv(first_path, ("SYNTHETIC-REG-2,V1,1.0,first",))
    write_csv(second_path, ("SYNTHETIC-REG-3,V1,2.0,second",))
    store_raw_file(first_path, raw_store)
    blobs_before = raw_blobs(raw_store)
    real_persist = clinical_ingestion.persist_clinical_source
    calls = 0

    def interrupt_second(*args, **kwargs) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InjectedClinicalFailure("second source")
        real_persist(*args, **kwargs)

    monkeypatch.setattr(
        clinical_ingestion,
        "persist_clinical_source",
        interrupt_second,
    )

    # When: persistence is interrupted on the second source.
    with database(tmp_path / "clinical.db") as connection:
        with pytest.raises(InjectedClinicalFailure, match="second source"):
            ingest_clinical_registry(
                connection,
                (
                    ClinicalIngestionRequest(
                        source(first_path, "SITE-A", ClinicalFormat.CSV),
                        raw_store,
                    ),
                    ClinicalIngestionRequest(
                        source(second_path, "SITE-B", ClinicalFormat.CSV),
                        raw_store,
                    ),
                ),
            )

        # Then: all database rows and only the newly created second blob disappear.
        assert connection.execute("SELECT COUNT(*) FROM source_assets").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM clinical_events").fetchone()[0] == 0
        assert raw_blobs(raw_store) == blobs_before
