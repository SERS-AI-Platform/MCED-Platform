from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .clinical_parser import load_clinical_rows
from .clinical_persistence import persist_clinical_source
from .clinical_types import ClinicalIngestionCounts, ClinicalSource
from .raw_store import RawStoreWrite, hash_source_file, store_raw_file_with_status
from .schema import initialize_schema
from .spectrum_persistence import assert_source_version
from .transaction import require_clean_connection


@dataclass(frozen=True, slots=True)
class ClinicalIngestionRequest:
    source: ClinicalSource
    raw_store: Path


@dataclass(frozen=True, slots=True)
class EmptyClinicalSourceError(ValueError):
    source_path: Path

    def __str__(self) -> str:
        return f"clinical source contains no ingestible rows: {self.source_path.name}"


def _counts(connection: sqlite3.Connection) -> ClinicalIngestionCounts:
    return ClinicalIngestionCounts(
        source_assets=connection.execute(
            "SELECT COUNT(*) FROM source_assets WHERE asset_kind = 'clinical'"
        ).fetchone()[0],
        subjects=connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0],
        events=connection.execute("SELECT COUNT(*) FROM clinical_events").fetchone()[0],
        observations=connection.execute(
            "SELECT COUNT(*) FROM clinical_observations"
        ).fetchone()[0],
        quarantined_rows=0,
    )


def _remove_unreferenced_raw(
    connection: sqlite3.Connection,
    raw_writes: tuple[RawStoreWrite, ...],
) -> None:
    for raw_write in raw_writes:
        if not raw_write.created:
            continue
        referenced = connection.execute(
            "SELECT 1 FROM source_assets WHERE raw_uri = ? LIMIT 1",
            (raw_write.stored.raw_uri,),
        ).fetchone()
        if referenced is None:
            Path(raw_write.stored.raw_uri).unlink(missing_ok=True)


def ingest_clinical_registry(
    connection: sqlite3.Connection,
    requests: tuple[ClinicalIngestionRequest, ...],
) -> ClinicalIngestionCounts:
    """Own one registry savepoint and refuse an active caller transaction."""
    require_clean_connection(connection, "clinical registry ingestion")
    initialize_schema(connection)
    savepoint = f"clinical_registry_{uuid4().hex}"
    connection.execute(f"SAVEPOINT {savepoint}")
    raw_writes: list[RawStoreWrite] = []
    succeeded = False
    try:
        for request in requests:
            source_uri = request.source.path.resolve().as_uri()
            source_hash = hash_source_file(request.source.path)
            assert_source_version(connection, source_uri, source_hash)
            rows = load_clinical_rows(request.source)
            if not rows:
                raise EmptyClinicalSourceError(request.source.path)
            raw_write = store_raw_file_with_status(
                request.source.path,
                request.raw_store,
                source_hash,
            )
            raw_writes.append(raw_write)
            persist_clinical_source(
                connection,
                request.source,
                raw_write.stored,
                rows,
            )
        succeeded = True
    finally:
        if succeeded:
            connection.execute(f"RELEASE {savepoint}")
        else:
            connection.execute(f"ROLLBACK TO {savepoint}")
            connection.execute(f"RELEASE {savepoint}")
            _remove_unreferenced_raw(connection, tuple(raw_writes))
    return _counts(connection)


def ingest_clinical_source(
    connection: sqlite3.Connection,
    request: ClinicalIngestionRequest,
) -> ClinicalIngestionCounts:
    """Ingest one source only when no caller transaction is active."""
    return ingest_clinical_registry(connection, (request,))
