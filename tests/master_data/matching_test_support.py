from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from sers.master_data.schema import initialize_schema
from sers.master_data.spectrum_schema import initialize_spectrum_schema


@dataclass(frozen=True, slots=True)
class SeededIdentity:
    site_id: str
    subject_id: str
    sample_id: str
    event_id: str
    observation_id: str


def database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY)")
    initialize_schema(connection)
    initialize_spectrum_schema(connection)
    return connection


def seed_identity(
    connection: sqlite3.Connection,
    site_code: str,
    source_key: str,
    observation_code: str = "pathology_result",
    observation_value: str = "Cancer",
) -> SeededIdentity:
    site_id = f"site-{site_code}"
    subject_id = str(uuid4())
    sample_id = str(uuid4())
    event_id = str(uuid4())
    observation_id = str(uuid4())
    connection.execute(
        "INSERT OR IGNORE INTO sites (id, code, name) VALUES (?, ?, ?)",
        (site_id, site_code, f"Synthetic {site_code}"),
    )
    connection.execute(
        """INSERT INTO subjects (
               id, site_id, patient_id
           ) VALUES (?, ?, ?)""",
        (subject_id, site_id, source_key),
    )
    connection.execute(
        """INSERT INTO samples (
               id, subject_id, site_id, solum_label, sample_type
           ) VALUES (?, ?, ?, ?, 'urine')""",
        (sample_id, subject_id, site_id, source_key),
    )
    connection.execute(
        """INSERT INTO clinical_events (
               id, subject_id, site_id, event_type
           ) VALUES (?, ?, ?, 'diagnosis')""",
        (event_id, subject_id, site_id),
    )
    connection.execute(
        """INSERT INTO clinical_observations (
               id, clinical_event_id, code, source_field_name, canonical_code,
               raw_value, value_kind, text_value, normalization_status
           ) VALUES (?, ?, ?, ?, ?, ?, 'text', ?, 'normalized')""",
        (
            observation_id,
            event_id,
            observation_code,
            observation_code,
            observation_code,
            observation_value,
            observation_value,
        ),
    )
    return SeededIdentity(site_id, subject_id, sample_id, event_id, observation_id)


def seed_material(
    connection: sqlite3.Connection,
    site_id: str,
    source_code: str,
    *,
    source_group: str = "PRO",
    canonical_alias: str = "",
) -> str:
    material_id = str(uuid4())
    connection.execute(
        """INSERT INTO analytical_materials (id, material_type)
           VALUES (?, 'primary')""",
        (material_id,),
    )
    connection.execute(
        """INSERT INTO spectrum_material_metadata (
               analytical_material_id, site_id, material_key,
               canonical_source_code, source_group, material_kind,
               preparation, identity_status, alias_candidates
           ) VALUES (?, ?, ?, ?, ?, 'biological', 'liquid', ?, ?)""",
        (
            material_id,
            site_id,
            f"material:{material_id}",
            source_code,
            source_group,
            "alias_review" if canonical_alias else "canonical",
            canonical_alias,
        ),
    )
    return material_id
