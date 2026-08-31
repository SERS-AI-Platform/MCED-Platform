from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from .lineage_types import DatasetBuildRequest, DatasetManifest
from .manifest_canonical import canonical_manifest
from .manifest_persistence import persist_manifest
from .manifest_queries import (
    ManifestMemberMismatchError,
    resolve_member,
)

__all__ = ["ManifestMemberMismatchError", "build_dataset_manifest"]


@dataclass(frozen=True, slots=True)
class DuplicateManifestMeasurementError(ValueError):
    measurement_id: str

    def __str__(self) -> str:
        return f"measurement appears more than once: {self.measurement_id}"


def build_dataset_manifest(
    connection: sqlite3.Connection,
    request: DatasetBuildRequest,
) -> DatasetManifest:
    sorted_drafts = tuple(
        sorted(request.members, key=lambda member: member.measurement_id)
    )
    seen: set[str] = set()
    for draft in sorted_drafts:
        if draft.measurement_id in seen:
            raise DuplicateManifestMeasurementError(draft.measurement_id)
        seen.add(draft.measurement_id)
    members = tuple(
        resolve_member(connection, draft, request.policy)
        for draft in sorted_drafts
    )
    return persist_manifest(
        connection,
        request,
        members,
        canonical_manifest(request, members),
    )
