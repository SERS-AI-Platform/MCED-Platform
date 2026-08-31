from __future__ import annotations

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from scripts.deployment import clinical_auth as auth
from scripts.deployment import clinical_db as db
from scripts.deployment import sers_clinical_webapp as webapp


@pytest.fixture
def account_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    database_path = tmp_path / "clinical_data.db"
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", database_path)
    db.init_db(database_path)
    db.ensure_default_users()
    auth._sessions.clear()
    return TestClient(webapp.app)


def _login(client: TestClient, username: str, password: str) -> None:
    response = client.post(
        "/login",
        data={"username": username, "password": password},
        follow_redirects=False,
    )
    assert response.status_code == 303


def test_admin_account_page_exposes_password_reset_form(account_app: TestClient) -> None:
    # Given: an authenticated administrator.
    _login(account_app, "admin", "admin123")

    # When: the administrator opens account management.
    response = account_app.get("/account")

    # Then: the reset endpoint and target selector are rendered.
    assert response.status_code == 200
    assert 'action="/account/password-reset"' in response.text
    assert 'name="target_user_id"' in response.text
    assert 'value="2"' in response.text


def test_non_admin_cannot_open_account_management(account_app: TestClient) -> None:
    # Given: an authenticated clinician.
    _login(account_app, "doctor1", "admin123")

    # When: the clinician requests the administrator page directly.
    response = account_app.get("/account", follow_redirects=False)

    # Then: the request is redirected to the clinical workflow.
    assert response.status_code == 303
    assert response.headers["location"] == "/patient/new"


def test_admin_password_reset_replaces_login_and_revokes_existing_session(
    account_app: TestClient,
) -> None:
    # Given: the clinician and administrator both have active sessions.
    clinician_client = TestClient(webapp.app)
    _login(clinician_client, "doctor1", "admin123")
    _login(account_app, "admin", "admin123")

    # When: the administrator assigns a temporary password.
    response = account_app.post(
        "/account/password-reset",
        data={
            "target_user_id": "2",
            "temporary_password": "Temporary2026!",
            "password_confirm": "Temporary2026!",
        },
        follow_redirects=False,
    )

    # Then: the old session is revoked and only the temporary password logs in.
    assert response.status_code == 303
    assert response.headers["location"] == "/account?reset=1"
    revoked = clinician_client.get("/patient/new", follow_redirects=False)
    assert revoked.status_code == 303
    assert revoked.headers["location"] == "/login"
    audit = db.get_audit_logs(limit=1)[0]
    assert audit["action"] == "password_reset"
    assert "Temporary2026!" not in (audit["detail"] or "")
    assert auth.login("doctor1", "admin123") is None
    assert auth.login("doctor1", "Temporary2026!") is not None


def test_admin_password_reset_rejects_mismatched_confirmation(
    account_app: TestClient,
) -> None:
    # Given: an authenticated administrator and the target's current credential.
    _login(account_app, "admin", "admin123")

    # When: the password confirmation differs.
    response = account_app.post(
        "/account/password-reset",
        data={
            "target_user_id": "2",
            "temporary_password": "Temporary2026!",
            "password_confirm": "Different2026!",
        },
    )

    # Then: the reset is rejected and the original password still works.
    assert response.status_code == 400
    assert auth.login("doctor1", "admin123") is not None
