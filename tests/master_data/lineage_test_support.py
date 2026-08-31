from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from uuid import uuid4

from sers.master_data.labels import LabelMapping, LabelRule, derive_sample_labels
from sers.master_data.matching import run_deterministic_matching
from tests.master_data.matching_test_support import seed_identity, seed_material


@dataclass(frozen=True, slots=True)
class SeededLineage:
    measurement_id: str
    sample_label_id: str
    evidence_id: str
    artifact_sha256: str


def seed_lineage(
    connection: sqlite3.Connection,
    site_code: str,
    source_key: str,
    *,
    measurement_status: str = "acquired",
) -> SeededLineage:
    identity = seed_identity(connection, site_code, source_key)
    material_id = seed_material(connection, identity.site_id, source_key)
    run_deterministic_matching(connection, "exact-v1")
    derive_sample_labels(
        connection,
        LabelRule(
            task_name="prostate_screening",
            label_definition_version="pathology-v1",
            observation_code="pathology_result",
            label_source="pathology",
            mappings=(LabelMapping("Cancer", "Cancer"),),
        ),
    )
    measurement_run_id = str(uuid4())
    measurement_id = str(uuid4())
    source_asset_id = str(uuid4())
    artifact_sha256 = hashlib.sha256(measurement_id.encode()).hexdigest()
    connection.execute(
        """INSERT INTO measurement_runs (
               id, site_id, instrument_key, status
           ) VALUES (?, ?, 'synthetic-instrument', 'complete')""",
        (measurement_run_id, identity.site_id),
    )
    connection.execute(
        """INSERT INTO measurements (
               id, measurement_run_id, analytical_material_id,
               replicate_index, status
           ) VALUES (?, ?, ?, 1, ?)""",
        (measurement_id, measurement_run_id, material_id, measurement_status),
    )
    connection.execute(
        """INSERT INTO source_assets (
               id, site_id, uri, sha256, asset_kind, state
           ) VALUES (?, ?, ?, ?, 'spectrum', 'ingested')""",
        (
            source_asset_id,
            identity.site_id,
            f"synthetic://artifact/{source_asset_id}",
            artifact_sha256,
        ),
    )
    connection.execute(
        """INSERT INTO measurement_artifacts (
               id, measurement_id, source_asset_id, artifact_role
           ) VALUES (?, ?, ?, 'raw')""",
        (str(uuid4()), measurement_id, source_asset_id),
    )
    label_row = connection.execute(
        """SELECT sl.id, sle.id
           FROM sample_labels AS sl
           JOIN sample_label_evidence AS sle ON sle.sample_label_id = sl.id
           JOIN analytical_materials AS am ON am.sample_id = sl.sample_id
           WHERE am.id = ?""",
        (material_id,),
    ).fetchone()
    assert label_row is not None
    return SeededLineage(
        measurement_id=measurement_id,
        sample_label_id=label_row[0],
        evidence_id=label_row[1],
        artifact_sha256=artifact_sha256,
    )
