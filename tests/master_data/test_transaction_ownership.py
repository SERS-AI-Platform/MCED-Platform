from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

import pytest

from sers.master_data.clinical_ingestion import (
    ClinicalIngestionRequest,
    ingest_clinical_registry,
    ingest_clinical_source,
)
from sers.master_data.clinical_types import ClinicalFormat
from sers.master_data.inventory import inventory_spectra
from sers.master_data.repository import upsert_site
from sers.master_data.schema import initialize_schema
from sers.master_data.spectrum_ingestion import (
    SpectrumIngestionRequest,
    ingest_spectra,
)
from sers.master_data.spectrum_schema import initialize_spectrum_schema
from sers.master_data.spectrum_types import InventoryRoot, SourceKind
from sers.master_data.transaction import ActiveCallerTransactionError
from sers.master_data.types import SiteDraft
from tests.master_data.clinical_test_support import (
    database,
    source,
    write_csv,
)
from tests.master_data.test_spectrum_ingestion import (
    _database,
    _write_spectrum,
)


def _schema_state(connection: sqlite3.Connection) -> tuple[int, tuple[str, ...]]:
    version = connection.execute("PRAGMA schema_version").fetchone()[0]
    tables = tuple(
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        )
    )
    return version, tables


def _raw_blobs(raw_store: Path) -> set[Path]:
    return {
        path.relative_to(raw_store)
        for path in raw_store.rglob("*")
        if path.is_file()
    }


def _assert_caller_transaction_refused(
    connection: sqlite3.Connection,
    raw_store: Path,
    action: Callable[[], None],
) -> None:
    connection.execute(
        "CREATE TABLE IF NOT EXISTS caller_rows (value TEXT NOT NULL)"
    )
    connection.commit()
    connection.execute("INSERT INTO caller_rows VALUES ('caller-owned')")
    schema_before = _schema_state(connection)
    blobs_before = _raw_blobs(raw_store)

    with pytest.raises(ActiveCallerTransactionError):
        action()

    assert connection.in_transaction
    assert _schema_state(connection) == schema_before
    assert _raw_blobs(raw_store) == blobs_before
    connection.rollback()
    assert connection.execute("SELECT COUNT(*) FROM caller_rows").fetchone()[0] == 0


def test_initialize_schema_refuses_caller_owned_transaction(tmp_path: Path) -> None:
    # Given: an uncommitted caller row on a connection without master-data tables.
    with sqlite3.connect(tmp_path / "schema.db") as connection:
        # When/Then: initialization fails before executescript can commit the caller row.
        _assert_caller_transaction_refused(
            connection,
            tmp_path / "raw",
            lambda: initialize_schema(connection),
        )


def test_initialize_spectrum_schema_refuses_caller_owned_transaction(
    tmp_path: Path,
) -> None:
    # Given: an uncommitted caller row on an initialized core database.
    with database(tmp_path / "spectrum-schema.db") as connection:
        # When/Then: spectrum schema initialization preserves caller rollback ownership.
        _assert_caller_transaction_refused(
            connection,
            tmp_path / "raw",
            lambda: initialize_spectrum_schema(connection),
        )


def test_single_clinical_ingestion_refuses_active_outer_transaction(
    tmp_path: Path,
) -> None:
    # Given: a valid source and an unrelated uncommitted caller row.
    source_path = tmp_path / "single.csv"
    write_csv(source_path, ("TX-SINGLE,V1,1.0,valid",))
    raw_store = tmp_path / "raw"
    with database(tmp_path / "single.db") as connection:
        request = ClinicalIngestionRequest(
            source(source_path, "SITE-A", ClinicalFormat.CSV),
            raw_store,
        )

        # When/Then: public single-source ingestion refuses before raw publication.
        _assert_caller_transaction_refused(
            connection,
            raw_store,
            lambda: ingest_clinical_source(connection, request),
        )


def test_clinical_registry_refuses_active_outer_transaction(
    tmp_path: Path,
) -> None:
    # Given: a valid registry request and an unrelated uncommitted caller row.
    source_path = tmp_path / "registry.csv"
    write_csv(source_path, ("TX-REGISTRY,V1,1.0,valid",))
    raw_store = tmp_path / "raw"
    with database(tmp_path / "registry.db") as connection:
        requests = (
            ClinicalIngestionRequest(
                source(source_path, "SITE-A", ClinicalFormat.CSV),
                raw_store,
            ),
        )

        # When/Then: registry orchestration refuses before schema or raw mutation.
        _assert_caller_transaction_refused(
            connection,
            raw_store,
            lambda: ingest_clinical_registry(connection, requests),
        )


def test_spectrum_ingestion_refuses_active_outer_transaction(
    tmp_path: Path,
) -> None:
    # Given: a valid spectrum report, committed site, and uncommitted caller row.
    root = tmp_path / "spectra"
    _write_spectrum(root / "PRO 1_1.CSV")
    report = inventory_spectra(
        (
            InventoryRoot(
                root,
                SourceKind.REMEASUREMENT,
                "Thermo",
                "liquid",
            ),
        )
    )
    raw_store = tmp_path / "raw"
    with _database(tmp_path / "spectra.db") as connection:
        site = upsert_site(connection, SiteDraft(code="TX", name="Transaction"))
        connection.commit()
        request = SpectrumIngestionRequest(report, site.id, raw_store)

        # When/Then: public spectrum ingestion refuses before raw publication.
        _assert_caller_transaction_refused(
            connection,
            raw_store,
            lambda: ingest_spectra(connection, request),
        )
