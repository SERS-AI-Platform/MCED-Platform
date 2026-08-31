from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sers.io import read_spectrum

from .raw_store import (
    RawStoreWrite,
    hash_source_file,
    store_raw_file_with_status,
)
from .spectrum_persistence import (
    PersistItem,
    assert_source_version,
    persist_item,
)
from .spectrum_queries import ingestion_counts
from .spectrum_schema import initialize_spectrum_schema
from .spectrum_types import IngestionCounts, InventoryReport, InventoryStatus
from .transaction import require_clean_connection
from .types import SiteId


@dataclass(frozen=True, slots=True)
class SpectrumIngestionRequest:
    report: InventoryReport
    site_id: SiteId
    raw_store: Path


def _remove_unreferenced_blobs(
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


def ingest_spectra(
    connection: sqlite3.Connection,
    request: SpectrumIngestionRequest,
) -> IngestionCounts:
    """Own the batch savepoint and refuse an active caller transaction."""
    require_clean_connection(connection, "spectrum ingestion")
    initialize_spectrum_schema(connection)
    ordered_items = sorted(
        (
            item
            for item in request.report.items
            if item.status is not InventoryStatus.EXCLUDED
        ),
        key=lambda item: (
            item.status is not InventoryStatus.READY,
            item.status is InventoryStatus.QUARANTINED,
            str(item.source_path),
        ),
    )
    savepoint = f"spectra_{uuid4().hex}"
    connection.execute(f"SAVEPOINT {savepoint}")
    raw_writes: list[RawStoreWrite] = []
    completed = False
    try:
        for item in ordered_items:
            source_hash = hash_source_file(item.source_path)
            assert_source_version(
                connection,
                item.source_path.resolve().as_uri(),
                source_hash,
            )
            raw_write = store_raw_file_with_status(
                item.source_path,
                request.raw_store,
                source_hash,
            )
            raw_writes.append(raw_write)
            status = item.status
            reason_code = item.reason_code
            if item.parsed is not None:
                try:
                    read_spectrum(item.source_path)
                except (OSError, ValueError):
                    status = InventoryStatus.QUARANTINED
                    reason_code = "malformed_spectrum"
            persist_item(
                connection,
                PersistItem(
                    item=item,
                    stored=raw_write.stored,
                    site_id=request.site_id,
                    status=status,
                    reason_code=reason_code,
                ),
            )
        completed = True
    finally:
        if not completed:
            connection.execute(f"ROLLBACK TO {savepoint}")
            connection.execute(f"RELEASE {savepoint}")
            _remove_unreferenced_blobs(connection, tuple(raw_writes))
    connection.execute(f"RELEASE {savepoint}")
    return ingestion_counts(connection)
