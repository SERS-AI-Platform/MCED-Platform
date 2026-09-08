from __future__ import annotations

import sqlite3
from pathlib import Path

from sers.master_data.schema import initialize_schema
from sers.master_data.spectrum_schema import initialize_spectrum_schema


def open_master_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    initialize_schema(connection)
    initialize_spectrum_schema(connection)
    return connection
