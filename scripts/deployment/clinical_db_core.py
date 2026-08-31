"""SQLite connection and frozen-application storage helpers."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def connection(db_path: Path) -> Iterator[sqlite3.Connection]:
    database = sqlite3.connect(db_path, timeout=30)
    database.row_factory = sqlite3.Row
    database.execute("PRAGMA busy_timeout=30000")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("PRAGMA foreign_keys=ON")
    try:
        with database:
            yield database
    finally:
        database.close()


def migrate_legacy_database(target: Path, legacy: Path) -> Path | None:
    """Copy a legacy internal database to external storage with a backup."""
    if target.exists() or not legacy.exists() or target.resolve() == legacy.resolve():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name("clinical_data.legacy-backup.db")
    if not backup.exists():
        shutil.copy2(legacy, backup)
    shutil.copy2(backup, target)
    return backup
