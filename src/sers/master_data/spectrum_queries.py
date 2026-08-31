from __future__ import annotations

import sqlite3

from .spectrum_types import IngestionCounts


def ingestion_counts(connection: sqlite3.Connection) -> IngestionCounts:
    return IngestionCounts(
        source_assets=connection.execute(
            "SELECT COUNT(*) FROM spectrum_inventory_records"
        ).fetchone()[0],
        measurements=connection.execute(
            """SELECT COUNT(*) FROM measurements AS m
               JOIN spectrum_material_metadata AS sm
                 ON sm.analytical_material_id = m.analytical_material_id"""
        ).fetchone()[0],
        derived_artifacts=connection.execute(
            """SELECT COUNT(*) FROM measurement_artifacts
               WHERE artifact_role = 'processed'"""
        ).fetchone()[0],
        quarantined=connection.execute(
            """SELECT COUNT(*) FROM spectrum_inventory_records
               WHERE status = 'quarantined'"""
        ).fetchone()[0],
    )
