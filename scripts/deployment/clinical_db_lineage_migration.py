from __future__ import annotations

import hashlib
import json
import sqlite3

from sers.master_data.lineage_privacy import JsonValue


def seed_legacy_prediction_pending_status(
    connection: sqlite3.Connection,
) -> None:
    rows = connection.execute(
        """SELECT id, session_id, result_json
           FROM predictions ORDER BY id"""
    )
    for prediction_id, session_id, raw_result in rows:
        result: JsonValue = json.loads(raw_result or "{}")
        canonical_result = json.dumps(
            result,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        result_sha256 = hashlib.sha256(canonical_result.encode()).hexdigest()
        status_id = hashlib.sha256(
            f"{prediction_id}\0{result_sha256}".encode()
        ).hexdigest()
        connection.execute(
            """INSERT OR IGNORE INTO legacy_prediction_lineage_status (
                   id, legacy_prediction_id, session_id, result_sha256, status
               ) VALUES (?, ?, ?, ?, 'pending_context')""",
            (status_id, prediction_id, session_id, result_sha256),
        )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_legacy_lineage_status
           ON legacy_prediction_lineage_status(status, created_at)"""
    )
    connection.execute(
        """CREATE INDEX IF NOT EXISTS idx_operational_run_session
           ON operational_prediction_run_links(session_id, created_at)"""
    )
