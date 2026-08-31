from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pytest

from sers.master_data import spectrum_ingestion
from sers.master_data.inventory import inventory_spectra
from sers.master_data.raw_store import RawStoreWrite, store_raw_file
from sers.master_data.repository import upsert_site
from sers.master_data.spectrum_ingestion import (
    SpectrumIngestionRequest,
    ingest_spectra,
)
from sers.master_data.spectrum_persistence import PersistItem
from sers.master_data.spectrum_types import InventoryRoot, Sha256, SourceKind
from sers.master_data.types import SiteDraft
from tests.master_data.test_spectrum_ingestion import _database, _write_spectrum


@dataclass(frozen=True, slots=True)
class InjectedSpectrumFailure(RuntimeError):
    stage: str

    def __str__(self) -> str:
        return f"synthetic spectrum {self.stage} failure"


def _request(
    connection: sqlite3.Connection,
    tmp_path: Path,
) -> tuple[SpectrumIngestionRequest, tuple[Path, Path]]:
    root = tmp_path / "20260429_Urine test"
    sources = (root / "PRO 1_1.CSV", root / "PRO 2_1.CSV")
    for source in sources:
        _write_spectrum(source)
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
    site = upsert_site(connection, SiteDraft(code="ATOMIC", name="Atomic"))
    connection.commit()
    return SpectrumIngestionRequest(report, site.id, tmp_path / "raw"), sources


def _raw_blobs(raw_store: Path) -> set[Path]:
    return {
        path.relative_to(raw_store)
        for path in raw_store.rglob("*")
        if path.is_file()
    }


def _ingestion_counts(connection: sqlite3.Connection) -> tuple[int, ...]:
    return tuple(
        connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in (
            "source_assets",
            "ingest_batches",
            "measurement_runs",
            "measurements",
            "measurement_artifacts",
        )
    )


def test_sqlite_failure_after_first_item_rolls_back_batch_and_blobs(
    tmp_path: Path,
) -> None:
    # Given: SQLite rejects the second source after the first can persist.
    with _database(tmp_path / "atomic.db") as connection:
        request, _sources = _request(connection, tmp_path)
        connection.execute(
            """CREATE TRIGGER reject_second_spectrum
               BEFORE INSERT ON source_assets
               WHEN NEW.uri LIKE '%PRO%202_1.CSV'
               BEGIN
                   SELECT RAISE(ABORT, 'synthetic second-item failure');
               END"""
        )
        connection.commit()

        # When: public batch ingestion reaches that second item.
        with pytest.raises(sqlite3.IntegrityError, match="second-item"):
            ingest_spectra(connection, request)

        # Then: its savepoint is closed and no batch rows or blobs survive.
        assert not connection.in_transaction
        assert _ingestion_counts(connection) == (0, 0, 0, 0, 0)
        assert _raw_blobs(request.raw_store) == set()


def test_runtime_failure_preserves_preexisting_blob_and_removes_new_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: item one is shared, then runtime persistence fails on item two.
    with _database(tmp_path / "runtime.db") as connection:
        request, sources = _request(connection, tmp_path)
        store_raw_file(sources[0], request.raw_store)
        blobs_before = _raw_blobs(request.raw_store)
        original = spectrum_ingestion.persist_item
        calls = 0

        def fail_second(
            target: sqlite3.Connection,
            draft: PersistItem,
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise InjectedSpectrumFailure("runtime")
            original(target, draft)

        monkeypatch.setattr(spectrum_ingestion, "persist_item", fail_second)

        # When: the typed interruption occurs after one successful item.
        with pytest.raises(InjectedSpectrumFailure, match="runtime"):
            ingest_spectra(connection, request)

        # Then: shared bytes remain while new rows and the new blob are removed.
        assert not connection.in_transaction
        assert _ingestion_counts(connection) == (0, 0, 0, 0, 0)
        assert _raw_blobs(request.raw_store) == blobs_before


def test_filesystem_failure_after_first_item_rolls_back_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: raw publication fails while opening the second source.
    with _database(tmp_path / "filesystem.db") as connection:
        request, _sources = _request(connection, tmp_path)
        original = spectrum_ingestion.store_raw_file_with_status
        calls = 0

        def fail_second(
            source: Path,
            raw_store: Path,
            expected_sha256: Sha256 | None = None,
        ) -> RawStoreWrite:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("synthetic filesystem failure")
            return original(source, raw_store, expected_sha256)

        monkeypatch.setattr(
            spectrum_ingestion,
            "store_raw_file_with_status",
            fail_second,
        )

        # When: the filesystem error occurs after the first item.
        with pytest.raises(OSError, match="filesystem"):
            ingest_spectra(connection, request)

        # Then: no savepoint, database row, or newly created blob remains.
        assert not connection.in_transaction
        assert _ingestion_counts(connection) == (0, 0, 0, 0, 0)
        assert _raw_blobs(request.raw_store) == set()
