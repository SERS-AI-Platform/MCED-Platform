from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from scripts.deployment import clinical_auth, clinical_db, launcher
from scripts.deployment import sers_clinical_webapp as webapp
from scripts.deployment.clinical_i18n import STRINGS


def test_default_users_exclude_obsolete_technician_account(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a newly initialized clinical database.
    database_path = tmp_path / "clinical_data.db"
    monkeypatch.setattr(clinical_db, "DEFAULT_DB_PATH", database_path)
    clinical_db.init_db(database_path)

    # When: startup creates the default accounts.
    clinical_db.ensure_default_users()
    usernames = {user["username"] for user in clinical_db.get_active_users()}

    # Then: only the administrator and clinician defaults exist.
    assert usernames == {"admin", "doctor1"}


def test_schema_migrates_legacy_technician_role_to_clinician(tmp_path: Path) -> None:
    # Given: a legacy database containing a technician account.
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('technician', 'clinician', 'admin')),
                display_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )"""
        )
        connection.execute(
            """INSERT INTO users (username, password_hash, role, display_name)
               VALUES ('legacy-tech', 'hash', 'technician', 'Legacy User')"""
        )

    # When: the current startup migration runs.
    clinical_db.init_db(database_path)

    # Then: the account is preserved as a clinician.
    with sqlite3.connect(database_path) as connection:
        role = connection.execute(
            "SELECT role FROM users WHERE username = 'legacy-tech'"
        ).fetchone()[0]
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                """INSERT INTO users (username, password_hash, role, display_name)
                   VALUES ('new-tech', 'hash', 'technician', 'New Technician')"""
            )
    assert role == "clinician"


def test_registration_creates_clinician_without_role_selector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a fresh application database.
    database_path = tmp_path / "clinical_data.db"
    monkeypatch.setattr(clinical_db, "DEFAULT_DB_PATH", database_path)
    clinical_db.init_db(database_path)
    client = TestClient(webapp.app)

    # When: a user opens and submits the registration form.
    page = client.get("/register")
    response = client.post(
        "/register",
        data={
            "username": "doctor-new",
            "display_name": "New Doctor",
            "password": "safe-password",
            "password_confirm": "safe-password",
        },
        follow_redirects=False,
    )
    invalid = client.post(
        "/register",
        data={
            "username": "doctor-invalid",
            "display_name": "Invalid Doctor",
            "password": "safe-password",
            "password_confirm": "different-password",
        },
    )
    registered = client.get("/login?registered=1")

    # Then: no role control is exposed and the account is a clinician.
    assert 'name="role"' not in page.text
    assert 'role="alert"' in invalid.text
    assert 'role="status"' in registered.text
    assert response.status_code == 303
    assert clinical_db.get_user_by_username("doctor-new")["role"] == "clinician"


def test_product_name_is_aecd_software_in_both_languages(capsys: pytest.CaptureFixture[str]) -> None:
    # Given: the shared bilingual product labels.
    # When: the application and launcher names are rendered.
    launcher.print_banner()

    # Then: every user-facing product title uses AECD Software.
    assert STRINGS["ko"]["app_title"] == "AECD Software"
    assert STRINGS["en"]["app_title"] == "AECD Software"
    assert STRINGS["ko"]["report_title"] == "AECD Software 보고서"
    assert STRINGS["en"]["report_title"] == "AECD Software Report"
    assert "role_technician" not in STRINGS["ko"]
    assert "role_technician" not in STRINGS["en"]
    assert webapp.app.title == "AECD Software"
    assert "AECD Software" in capsys.readouterr().out


def test_hash_password_uses_pbkdf2_and_verifies() -> None:
    encoded = clinical_auth.hash_password("safe-password")

    assert encoded.startswith("pbkdf2_sha256$")
    assert clinical_auth.verify_password("safe-password", encoded)
    assert not clinical_auth.verify_password("wrong-password", encoded)


def test_verify_password_accepts_existing_pbkdf2_format() -> None:
    iterations = 240_000
    salt_hex = "c0cf16fd336261439267c968924608d7"
    expected = hashlib.pbkdf2_hmac(
        "sha256",
        b"existing-password",
        salt_hex.encode("ascii"),
        iterations,
    ).hex()
    encoded = f"pbkdf2_sha256${iterations}${salt_hex}${expected}"

    assert clinical_auth.verify_password("existing-password", encoded)


def test_verify_password_accepts_legacy_sha256() -> None:
    encoded = hashlib.sha256(b"legacy-password").hexdigest()

    assert clinical_auth.verify_password("legacy-password", encoded)
    assert not clinical_auth.verify_password("wrong-password", encoded)


@pytest.mark.parametrize(
    "encoded",
    ["", "pbkdf2_sha256", "pbkdf2_sha256$bad$salt$digest", "not-a-hash"],
)
def test_verify_password_rejects_malformed_hashes(encoded: str) -> None:
    assert not clinical_auth.verify_password("password", encoded)
