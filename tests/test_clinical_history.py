import shutil
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import pytest

from scripts.deployment import clinical_db as db
from scripts.deployment.clinical_db_core import migrate_legacy_database

SOURCE_DB = Path(__file__).resolve().parents[1] / "scripts" / "deployment" / "clinical_data.db"


def _counts(db_path: Path) -> dict[str, int]:
    table_names = (
        "users",
        "sessions",
        "spectra",
        "predictions",
        "reports",
        "audit_log",
    )
    with sqlite3.connect(db_path) as connection:
        return {
            table_name: connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            for table_name in table_names
        }


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "clinical_data.db"
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", db_path)
    db.init_db(db_path)
    db.ensure_default_users()
    return db_path


def test_init_db_migrates_legacy_database_without_losing_rows(tmp_path: Path) -> None:
    # Given: a copy of the pre-history clinical database with patient artifacts.
    migrated_path = tmp_path / "legacy.db"
    shutil.copy2(SOURCE_DB, migrated_path)
    before = _counts(migrated_path)

    # When: startup migration runs twice.
    db.init_db(migrated_path)
    db.init_db(migrated_path)

    # Then: all records remain and the ownership/test lineage schema is present.
    assert _counts(migrated_path) == before
    with sqlite3.connect(migrated_path) as connection:
        session_columns = {row[1] for row in connection.execute("PRAGMA table_info(sessions)")}
        prediction_columns = {
            row[1] for row in connection.execute("PRAGMA table_info(predictions)")
        }
        indexes = {row[1] for row in connection.execute("PRAGMA index_list(sessions)")}
        backfilled = connection.execute(
            "SELECT test_id, test_date, test_sequence, owner_user_id FROM sessions"
        ).fetchall()
    assert {
        "test_id",
        "test_date",
        "test_sequence",
        "owner_user_id",
        "retest_of_session_id",
        "qc_valid",
    } <= session_columns
    assert "idx_sessions_test_id_unique" in indexes
    assert {"decision_level", "decision_policy"} <= prediction_columns
    assert all(all(value is not None for value in row) for row in backfilled)


def test_same_patient_tests_receive_sequential_ids_for_the_day(
    isolated_db: Path,
) -> None:
    # Given: one account creates two tests for the same patient on one day.
    first_id = db.create_session(
        patient_id="PATIENT-01",
        age=55,
        sex="M",
        bmi=24.0,
        operating_mode="balanced",
        created_by=1,
    )

    # When: the second session is created.
    second_id = db.create_session(
        patient_id="PATIENT-01",
        age=55,
        sex="M",
        bmi=24.0,
        operating_mode="balanced",
        created_by=1,
    )

    # Then: the date component is stable and the sequence increases.
    test_date = date.today().strftime("%Y%m%d")
    assert db.get_session(first_id)["test_id"] == f"PATIENT-01-{test_date}-01"
    assert db.get_session(second_id)["test_id"] == f"PATIENT-01-{test_date}-02"


def test_concurrent_session_creation_allocates_unique_test_sequences(
    isolated_db: Path,
) -> None:
    # Given: six same-patient tests begin concurrently on the shared PC.
    def create_one(_: int) -> str:
        return db.create_session(
            patient_id="PATIENT-CONCURRENT",
            age=48,
            sex="F",
            bmi=22.5,
            operating_mode="balanced",
            created_by=1,
        )

    # When: SQLite allocates their session identifiers.
    with ThreadPoolExecutor(max_workers=6) as executor:
        session_ids = list(executor.map(create_one, range(6)))

    # Then: every sequence from 01 through 06 is allocated exactly once.
    test_ids = {db.get_session(session_id)["test_id"] for session_id in session_ids}
    assert {test_id.rsplit("-", 1)[1] for test_id in test_ids} == {
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
    }


def test_report_id_is_stable_across_repeated_downloads(isolated_db: Path) -> None:
    # Given: one newly created test.
    session_id = db.create_session(
        patient_id="PATIENT-REPORT",
        age=61,
        sex="M",
        bmi=26.4,
        operating_mode="balanced",
        created_by=1,
    )

    # When: report metadata is requested twice.
    first_report_id = db.create_report(session_id, 1)
    second_report_id = db.create_report(session_id, 1)

    # Then: both downloads reuse one report identity and one row.
    assert first_report_id == second_report_id
    assert first_report_id == f"{db.get_session(session_id)['test_id']}-R01"
    assert len(db.get_reports_for_session(session_id)) == 1


def test_existing_random_report_id_is_preserved(isolated_db: Path) -> None:
    session_id = db.create_session(
        patient_id="PATIENT-LEGACY-REPORT",
        age=61,
        sex="M",
        bmi=26.4,
        operating_mode="balanced",
        created_by=1,
    )
    with db.get_connection() as database:
        database.execute(
            "INSERT INTO reports (id, session_id, generated_by) VALUES (?, ?, ?)",
            ("historical-random-id", session_id, 1),
        )

    assert db.create_report(session_id, 1) == "historical-random-id"
    assert len(db.get_reports_for_session(session_id)) == 1


def test_owner_lookup_hides_foreign_session_but_allows_admin(isolated_db: Path) -> None:
    # Given: a test owned by the first account.
    session_id = db.create_session(
        patient_id="PATIENT-OWNER",
        age=42,
        sex="F",
        bmi=20.1,
        operating_mode="balanced",
        created_by=1,
    )

    # When: the owner, another account, and an admin resolve the direct URL.
    owner_session = db.get_session_for_user(session_id, user_id=1, is_admin=False)
    foreign_session = db.get_session_for_user(session_id, user_id=2, is_admin=False)
    admin_session = db.get_session_for_user(session_id, user_id=2, is_admin=True)

    # Then: only the owner and admin can observe the record.
    assert owner_session["id"] == session_id
    assert foreign_session is None
    assert admin_session["id"] == session_id


def test_retest_preserves_failed_session_artifacts_and_owner(isolated_db: Path) -> None:
    # Given: a QC-failed test with spectra, prediction, and a historical report.
    source_id = db.create_session(
        patient_id="PATIENT-RETEST",
        age=57,
        sex="M",
        bmi=25.2,
        operating_mode="balanced",
        created_by=2,
    )
    spectrum_id = db.add_spectrum(source_id, "failed.csv")
    db.update_spectrum_qc(spectrum_id, qc_pass=False, qc_flags=["low_correlation"])
    db.save_prediction(
        source_id,
        {
            "cancer_detected": False,
            "screening_index": 2.0,
            "patient_decision": {"cancer_detected": False, "ssi_score": 2.0},
        },
    )
    source_report_id = db.create_report(source_id, 2)
    db.update_session_status(source_id, "qc_done", qc_valid=False)

    # When: an administrator starts a retest for the original owner.
    retest_id = db.create_retest_session(source_id, performed_by=1)

    # Then: a linked empty attempt is created without changing the failed evidence.
    source = db.get_session(source_id)
    retest = db.get_session(retest_id)
    assert retest["retest_of_session_id"] == source_id
    assert retest["owner_user_id"] == source["owner_user_id"] == 2
    assert retest["created_by"] == 1
    assert retest["patient_id"] == source["patient_id"]
    assert retest["test_sequence"] == source["test_sequence"] + 1
    assert db.get_spectra_for_session(source_id)[0]["filename"] == "failed.csv"
    assert db.get_prediction(source_id) is not None
    assert db.get_reports_for_session(source_id)[0]["id"] == source_report_id
    assert db.get_spectra_for_session(retest_id) == []


def test_history_query_scopes_rows_to_owner_and_admin(isolated_db: Path) -> None:
    # Given: two users each own one test.
    first_id = db.create_session(
        patient_id="OWNER-A",
        age=31,
        sex="F",
        bmi=21.0,
        operating_mode="balanced",
        created_by=1,
    )
    second_id = db.create_session(
        patient_id="OWNER-B",
        age=32,
        sex="M",
        bmi=22.0,
        operating_mode="balanced",
        created_by=2,
    )

    # When: a normal user and an admin load history.
    owner_rows, owner_total = db.get_session_history(user_id=1, is_admin=False)
    admin_rows, admin_total = db.get_session_history(user_id=1, is_admin=True)

    # Then: the normal user sees only owned rows while admin sees both.
    assert owner_total == 1
    assert [row["id"] for row in owner_rows] == [first_id]
    assert admin_total == 2
    assert {row["id"] for row in admin_rows} == {first_id, second_id}


def test_history_uses_fifty_row_pages_and_combined_filters(isolated_db: Path) -> None:
    for index in range(51):
        session_id = db.create_session(
            patient_id=f"PAGE-{index:02d}",
            age=40,
            sex="F",
            bmi=22.0,
            operating_mode="balanced",
            created_by=1,
        )
        if index == 50:
            db.update_session_status(session_id, "qc_done", qc_valid=False)

    first_page, total = db.get_session_history(user_id=1, is_admin=False)
    second_page, second_total = db.get_session_history(
        user_id=1,
        is_admin=False,
        filters=db.HistoryFilters(page=2),
    )
    filtered, filtered_total = db.get_session_history(
        user_id=1,
        is_admin=False,
        filters=db.HistoryFilters(patient_id="PAGE-50", qc_status="failed"),
    )

    assert total == second_total == 51
    assert len(first_page) == 50
    assert len(second_page) == 1
    assert filtered_total == 1
    assert filtered[0]["patient_id"] == "PAGE-50"


def test_legacy_internal_database_is_backed_up_before_external_copy(
    tmp_path: Path,
) -> None:
    # Given: an internal database from an older one-folder executable.
    legacy_path = tmp_path / "_internal" / "scripts" / "deployment" / "clinical_data.db"
    legacy_path.parent.mkdir(parents=True)
    legacy_path.write_bytes(b"legacy-clinical-data")
    external_path = tmp_path / "clinical_data.db"

    # When: first-run storage migration moves data beside the executable.
    backup_path = migrate_legacy_database(external_path, legacy_path)
    second_attempt = migrate_legacy_database(external_path, legacy_path)

    # Then: both a backup and the external working copy preserve the original bytes.
    assert backup_path == tmp_path / "clinical_data.legacy-backup.db"
    assert backup_path.read_bytes() == b"legacy-clinical-data"
    assert external_path.read_bytes() == b"legacy-clinical-data"
    assert legacy_path.read_bytes() == b"legacy-clinical-data"
    assert second_attempt is None


def test_historical_missing_bmi_requires_override_for_retest(isolated_db: Path) -> None:
    # Given: a migrated QC-failed session whose original BMI was not recorded.
    source_id = db.create_session(
        patient_id="LEGACY-NO-BMI",
        age=68,
        sex="F",
        bmi=None,
        operating_mode="balanced",
        created_by=2,
    )
    db.update_session_status(source_id, "qc_done", qc_valid=False)

    # When/Then: retest creation pauses until a valid BMI is supplied.
    with pytest.raises(db.RetestBmiRequiredError):
        db.create_retest_session(source_id, performed_by=2)

    retest_id = db.create_retest_session(source_id, performed_by=2, bmi_override=23.7)
    assert db.get_session(retest_id)["bmi"] == 23.7


def test_frozen_database_path_is_beside_executable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: the application is running from a PyInstaller executable.
    executable = tmp_path / "SERS_Clinical.exe"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(executable))

    # When: the runtime database path is resolved.
    resolved = db.get_db_path()

    # Then: rebuildable _internal assets are not used for clinical data.
    assert resolved == tmp_path / "clinical_data.db"
