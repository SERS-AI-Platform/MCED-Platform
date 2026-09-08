from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .spectrum_types import Sha256


@dataclass(frozen=True, slots=True)
class SourceAssetChangedError(RuntimeError):
    source_uri: str

    def __str__(self) -> str:
        return f"source asset changed after initial ingestion: {self.source_uri}"


def assert_source_version(
    connection: sqlite3.Connection,
    source_uri: str,
    sha256: Sha256,
) -> None:
    row = connection.execute(
        "SELECT sha256 FROM source_assets WHERE uri = ?",
        (source_uri,),
    ).fetchone()
    if row is not None and row[0] != sha256:
        raise SourceAssetChangedError(source_uri)
