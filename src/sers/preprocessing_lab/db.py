"""Read/write access to the aecd_platform `experiment` schema.

Schema DDL: scripts/db/experiment_tracking/01_schema.sql

Unlike `sers.aecd_api.repository`, which is deliberately read-only, this module
writes — it is the only place in the preprocessing lab that does. It touches the
`experiment` schema exclusively; `master`, `clinical`, and `measurement` are read
through foreign keys only and are never written to from here.
"""

from __future__ import annotations

import json
from contextlib import closing
from dataclasses import dataclass
from typing import Any, Final, Sequence

import psycopg2
from psycopg2.extensions import connection as PgConnection
from psycopg2.extras import execute_values

from sers.aecd_api.repository import DatabaseSettings

WRITE_OPTIONS: Final = "-c statement_timeout=60000 -c lock_timeout=5000"


@dataclass(frozen=True, slots=True)
class ExperimentTrackingError(Exception):
    operation: str

    def __str__(self) -> str:
        return f"aecd_platform experiment schema unavailable during {self.operation}"


class ExperimentTracker:
    """Writes preprocessing-lab runs into aecd_platform.experiment.*"""

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self._settings: DatabaseSettings = settings or DatabaseSettings.from_environment()

    def _connect(self) -> PgConnection:
        return psycopg2.connect(
            host=self._settings.host,
            port=self._settings.port,
            dbname=self._settings.database,
            user=self._settings.user,
            password=self._settings.password,
            connect_timeout=5,
            options=WRITE_OPTIONS,
        )

    def upsert_paper(self, citation: str, **fields: Any) -> int:
        query = (
            "INSERT INTO experiment.papers (citation, title, doi_or_url, source_tool, notes) "
            "VALUES (%s, %s, %s, %s, %s) RETURNING paper_id"
        )
        parameters = (
            citation,
            fields.get("title"),
            fields.get("doi_or_url"),
            fields.get("source_tool"),
            fields.get("notes"),
        )
        return self._insert_returning_id(query, parameters, "paper insert")

    def upsert_method(self, method_key: str, stage: str, **fields: Any) -> int:
        query = (
            "INSERT INTO experiment.preprocessing_methods "
            "(method_key, stage, display_name, paper_id, implementation_path, config_overrides) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (method_key) DO UPDATE SET "
            "display_name = EXCLUDED.display_name, "
            "implementation_path = EXCLUDED.implementation_path, "
            "config_overrides = EXCLUDED.config_overrides "
            "RETURNING method_id"
        )
        overrides = fields.get("config_overrides")
        parameters = (
            method_key,
            stage,
            fields.get("display_name"),
            fields.get("paper_id"),
            fields.get("implementation_path"),
            json.dumps(overrides) if overrides is not None else None,
        )
        return self._insert_returning_id(query, parameters, "method upsert")

    def select_method_id(self, method_key: str) -> int | None:
        """Look up a registered method's id, or None when it is not registered."""
        query = "SELECT method_id FROM experiment.preprocessing_methods WHERE method_key = %s"
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                cursor.execute(query, (method_key,))
                row = cursor.fetchone()
        except psycopg2.Error as error:
            raise ExperimentTrackingError("method lookup") from error
        return int(row[0]) if row else None

    def create_run(self, run_name: str, method_id: int, **fields: Any) -> int:
        query = (
            "INSERT INTO experiment.runs "
            "(run_name, method_id, git_commit, config_snapshot, data_query_filters, "
            "n_subjects, n_samples, n_spectra, qc_passed_n, qc_failed_n, started_at, status, notes) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING run_id"
        )
        snapshot = fields.get("config_snapshot")
        filters = fields.get("data_query_filters")
        parameters = (
            run_name,
            method_id,
            fields.get("git_commit"),
            json.dumps(snapshot) if snapshot is not None else None,
            json.dumps(filters) if filters is not None else None,
            fields.get("n_subjects"),
            fields.get("n_samples"),
            fields.get("n_spectra"),
            fields.get("qc_passed_n"),
            fields.get("qc_failed_n"),
            fields.get("started_at"),
            fields.get("status", "running"),
            fields.get("notes"),
        )
        return self._insert_returning_id(query, parameters, "run insert")

    def link_measurements(self, run_id: int, measurement_ids: Sequence[int]) -> int:
        """Record exactly which aecd_platform measurements this run consumed.

        The foreign key means an unknown measurement_id is rejected by the
        database rather than silently stored — this is the integrity guarantee a
        cross-database design could not provide.
        """
        if not measurement_ids:
            return 0
        rows = [(run_id, int(measurement_id)) for measurement_id in measurement_ids]
        query = (
            "INSERT INTO experiment.run_measurements (run_id, measurement_id) VALUES %s "
            "ON CONFLICT DO NOTHING"
        )
        return self._execute_values(query, rows, "measurement link")

    def record_metrics(self, run_id: int, metrics: Sequence[dict[str, Any]]) -> int:
        if not metrics:
            return 0
        rows = [
            (
                run_id,
                metric["metric_name"],
                metric.get("metric_value"),
                metric.get("split"),
                metric.get("model_name"),
            )
            for metric in metrics
        ]
        query = (
            "INSERT INTO experiment.run_metrics "
            "(run_id, metric_name, metric_value, split, model_name) VALUES %s "
            "ON CONFLICT (run_id, metric_name, split, model_name) DO UPDATE "
            "SET metric_value = EXCLUDED.metric_value"
        )
        return self._execute_values(query, rows, "metric insert")

    def finish_run(self, run_id: int, status: str, finished_at: Any) -> None:
        query = (
            "UPDATE experiment.runs "
            "SET status = %s, finished_at = %s WHERE run_id = %s"
        )
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                cursor.execute(query, (status, finished_at, run_id))
                connection.commit()
        except psycopg2.Error as error:
            raise ExperimentTrackingError("run finish") from error

    def _insert_returning_id(
        self, query: str, parameters: tuple[Any, ...], operation: str
    ) -> int:
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                cursor.execute(query, parameters)
                row = cursor.fetchone()
                connection.commit()
        except psycopg2.Error as error:
            raise ExperimentTrackingError(operation) from error
        if row is None:
            raise ExperimentTrackingError(operation)
        return int(row[0])

    def _execute_values(
        self, query: str, rows: list[tuple[Any, ...]], operation: str
    ) -> int:
        try:
            with closing(self._connect()) as connection, connection.cursor() as cursor:
                execute_values(cursor, query, rows)
                connection.commit()
        except psycopg2.Error as error:
            raise ExperimentTrackingError(operation) from error
        return len(rows)
