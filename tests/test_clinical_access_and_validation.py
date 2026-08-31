import io
from pathlib import Path

import anyio
import pytest
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.testclient import TestClient

from scripts.deployment import clinical_db as db
from scripts.deployment import sers_clinical_webapp as webapp
from scripts.deployment.clinical_i18n import STRINGS


def _request(path: str, method: str = "GET") -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": method,
            "scheme": "http",
            "path": path,
            "raw_path": path.encode(),
            "query_string": b"",
            "headers": [],
            "client": ("testclient", 50000),
            "server": ("testserver", 80),
            "root_path": "",
            "app": webapp.app,
        }
    )


@pytest.fixture
def foreign_session_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {
            "user_id": 2,
            "username": "other",
            "display_name": "Other",
            "role": "technician",
        },
    )
    monkeypatch.setattr(
        webapp.db,
        "get_session_for_user",
        lambda session_id, user_id, is_admin: None,
    )
    monkeypatch.setattr(
        webapp.db,
        "get_session",
        lambda session_id: {
            "id": session_id,
            "patient_id": "FOREIGN",
            "age": 55,
            "sex": "M",
            "bmi": 24.0,
            "threshold": 0.6,
            "owner_user_id": 1,
        },
    )
    monkeypatch.setattr(webapp.db, "get_spectra_for_session", lambda session_id: [])
    monkeypatch.setattr(webapp.db, "get_prediction", lambda session_id: None)
    monkeypatch.setattr(webapp.db, "reset_session_measurements", lambda session_id: None)


@pytest.mark.parametrize(
    ("route", "path"),
    [
        (webapp.upload_page, "/patient/foreign/upload"),
        (webapp.qc_page, "/patient/foreign/qc"),
        (webapp.results_page, "/patient/foreign/results"),
        (webapp.report_page, "/patient/foreign/report"),
        (webapp.report_pdf, "/patient/foreign/report.pdf"),
        (webapp.report_csv, "/patient/foreign/report.csv"),
        (webapp.remeasure_patient, "/patient/foreign/remeasure"),
    ],
)
def test_direct_session_routes_hide_foreign_records(
    foreign_session_guard: None,
    route,
    path: str,
) -> None:
    # Given: a logged-in user requests a session owned by another account.
    # When: the route resolves the session identifier.
    response = anyio.run(route, _request(path), "foreign")

    # Then: it is indistinguishable from an unknown identifier.
    assert response.status_code == 404


def test_upload_post_hides_foreign_session_before_file_validation(
    foreign_session_guard: None,
) -> None:
    # Given: another user's upload endpoint and an invalid empty upload set.
    # When: the POST route resolves ownership.
    response = anyio.run(
        webapp.upload_submit,
        _request("/patient/foreign/upload", method="POST"),
        "foreign",
        [],
    )

    # Then: ownership fails before multipart validation reveals workflow details.
    assert response.status_code == 404


def test_qc_failure_offers_new_patient_path_in_addition_to_retest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an authenticated owner viewing a failed QC session.
    session = {
        "id": "failed-session",
        "patient_id": "PATIENT-FAILED",
        "age": 55,
        "sex": "M",
        "bmi": 24.0,
        "owner_user_id": 1,
    }
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {"user_id": 1, "role": "clinician", "display_name": "Doctor"},
    )
    monkeypatch.setattr(webapp, "_authorized_session", lambda request, session_id: session)
    monkeypatch.setattr(
        webapp.db,
        "get_spectra_for_session",
        lambda session_id: [{"qc_pass": False, "filename": "failed.csv"}],
    )

    # When: the QC page is rendered.
    response = anyio.run(webapp.qc_page, _request("/patient/failed-session/qc"), "failed-session")

    # Then: the user can start either a same-patient retest or a new-patient flow.
    body = response.body.decode("utf-8")
    assert response.status_code == 200
    assert 'action="/patient/failed-session/retest"' in body
    assert 'href="/patient/new"' in body
    assert "새 환자로 진행" in body


@pytest.mark.parametrize("bmi", ["", "nan", "inf", "9.9", "60.1"])
def test_patient_registration_rejects_missing_or_out_of_range_bmi(
    monkeypatch: pytest.MonkeyPatch,
    bmi: str,
) -> None:
    # Given: an authenticated new-patient submission with invalid BMI.
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {"user_id": 1, "role": "admin"},
    )
    monkeypatch.setattr(
        webapp,
        "get_predictor",
        lambda: type("Predictor", (), {"threshold": 0.6})(),
    )

    # When: the server parses the form boundary.
    response = anyio.run(
        webapp.patient_submit,
        _request("/patient/new", method="POST"),
        "PATIENT-VALID",
        55,
        "M",
        bmi,
    )

    # Then: the form is rejected without creating a test.
    assert response.status_code == 400


def test_patient_registration_rejects_unsafe_patient_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a patient ID that cannot safely appear in a report filename.
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {"user_id": 1, "role": "admin"},
    )
    monkeypatch.setattr(
        webapp,
        "get_predictor",
        lambda: type("Predictor", (), {"threshold": 0.6})(),
    )

    # When: the form is submitted.
    response = anyio.run(
        webapp.patient_submit,
        _request("/patient/new", method="POST"),
        "../환자 01",
        55,
        "M",
        "24.0",
    )

    # Then: the server rejects it at the input boundary.
    assert response.status_code == 400


def test_upload_persists_explicit_cancer_signal_and_qc_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a valid owned test and five QC-passing replicate predictions.
    db_path = tmp_path / "clinical_data.db"
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", db_path)
    db.init_db(db_path)
    db.ensure_default_users()
    session_id = db.create_session(
        patient_id="PATIENT-COUNTS",
        age=50,
        sex="F",
        bmi=23.0,
        operating_mode="balanced",
        created_by=1,
    )
    filenames = [f"replicate_{index}.csv" for index in range(5)]
    per_replicate = [
        {
            "file": filename,
            "cancer_detected": index < 3,
            "ssi_score": 5.0 if index < 3 else 2.0,
            "replicate_correlation": 0.99,
        }
        for index, filename in enumerate(filenames)
    ]
    result = {
        "status": "ok",
        "model_variant": "test",
        "qc_summary": {
            "passed": 5,
            "passes": [{"file": filename, "replicate_correlation": 0.99} for filename in filenames],
            "failures": [],
        },
        "patient_decision": {
            "cancer_detected": True,
            "ssi_score": 5.0,
            "ssi_threshold": 4.0,
            "majority_vote": "3/5",
        },
        "per_replicate": per_replicate,
        "cancer_type_probabilities": {},
    }
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {"user_id": 1, "role": "admin"},
    )
    monkeypatch.setattr(
        webapp,
        "get_predictor",
        lambda: type("Predictor", (), {"predict_patient": lambda self, **kwargs: result})(),
    )
    uploads = [UploadFile(file=io.BytesIO(b"1,2\n"), filename=filename) for filename in filenames]

    # When: analysis output crosses the persistence boundary.
    response = anyio.run(
        webapp.upload_submit,
        _request(f"/patient/{session_id}/upload", method="POST"),
        session_id,
        uploads,
    )

    # Then: explicit counts replace majority-vote parsing for current records.
    saved = db.get_prediction(session_id)["result_json"]
    assert response.status_code == 303
    assert saved["cancer_signal_spectra_count"] == 3
    assert saved["qc_valid_spectra_count"] == 5
    assert db.get_session(session_id)["qc_valid"] == 1


def test_manual_is_packaged_and_downloaded_as_attachment() -> None:
    # Given: the canonical packaged software IFU.
    manual_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "deployment"
        / "manuals"
        / "software_ifu.pdf"
    )

    # When: an unauthenticated user downloads the manual.
    response = anyio.run(webapp.manual_download)

    # Then: the bundled PDF is available as a download.
    assert manual_path.is_file()
    assert response.media_type == "application/pdf"
    assert response.headers["content-disposition"].startswith("attachment;")


def test_ssi_explanation_and_signal_labels_are_bilingual() -> None:
    # Given: the two supported output languages.
    # When: shared result/report/export terminology is selected.
    # Then: each language explains the same three-level mean-SSI action rule.
    assert STRINGS["ko"]["ssi_calculation_text"] == (
        "검사가 유효한 경우 QC를 통과한 모든 스펙트럼의 암 관련 모델 신호값을 "
        "평균한 뒤 0~10점의 평균 SSI로 변환합니다. 평균 SSI가 4.0을 초과하면 "
        "‘추가 확인 권고’, 1.0 이상 4.0 이하이면 ‘추가 평가 고려’, 1.0 미만이면 "
        "‘기준 미만’으로 판정합니다. 암 신호 스펙트럼 수는 반복측정 참고값이며 "
        "최종 판정에 사용하지 않습니다."
    )
    assert STRINGS["en"]["ssi_calculation_text"] == (
        "For a valid test, the cancer-related model signals from all QC-passing spectra "
        "are averaged and converted to a 0–10 mean SSI. A mean SSI above 4.0 results in "
        "‘Further Evaluation Recommended’; 1.0 through 4.0 inclusive results in "
        "‘Consider Further Evaluation’; and below 1.0 results in ‘Below Decision "
        "Threshold’. The cancer-signal spectrum count is a replicate-level reference "
        "and is not used for the final decision."
    )
    assert STRINGS["ko"]["majority_vote_label"] == "암 신호 스펙트럼(참고)"
    assert STRINGS["ko"]["spectra_unit"] == "개"
    assert STRINGS["en"]["majority_vote_label"] == "Cancer-signal spectra (reference)"
    assert STRINGS["ko"]["replicate_above_internal_threshold"] == "암 신호 있음"
    assert STRINGS["en"]["replicate_above_internal_threshold"] == "Cancer signal present"


def test_legacy_outputs_use_historical_copy_instead_of_mean_ssi_action_copy() -> None:
    deployment = Path(__file__).resolve().parents[1] / "scripts" / "deployment"
    results_template = (deployment / "templates" / "results.html").read_text(
        encoding="utf-8"
    )
    report_template = (deployment / "templates" / "report_template.html").read_text(
        encoding="utf-8"
    )
    csv_route = (deployment / "sers_clinical_webapp.py").read_text(encoding="utf-8")

    assert "legacy_mean_ssi_reference_title if is_legacy_decision" in results_template
    assert "{% elif is_legacy_decision %}" in report_template
    assert "s.legacy_recommendation" in report_template
    assert 'strings["legacy_recommendation"]' in csv_route
    assert "소급 해석하지 마세요" in STRINGS["ko"]["legacy_recommendation"]


def test_history_route_passes_owner_scope_and_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a normal user searches their own QC-failed records.
    captured: dict = {}
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {
            "user_id": 7,
            "role": "technician",
            "display_name": "Owner",
        },
    )

    def fake_history(user_id: int, is_admin: bool, filters):
        captured.update(
            user_id=user_id,
            is_admin=is_admin,
            patient_id=filters.patient_id,
            qc_status=filters.qc_status,
        )
        return [], 0

    monkeypatch.setattr(webapp.db, "get_session_history", fake_history)

    # When: the authenticated history route is rendered.
    response = anyio.run(
        webapp.history_page,
        _request("/history"),
        "PATIENT-7",
        "",
        "failed",
        "",
        None,
        1,
    )

    # Then: account scope reaches the DB query and the history template is used.
    assert captured == {
        "user_id": 7,
        "is_admin": False,
        "patient_id": "PATIENT-7",
        "qc_status": "failed",
    }
    assert response.template.name == "history.html"


def test_admin_history_accepts_blank_owner_filter_from_browser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an admin submits the native HTML search form with no owner selected.
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {
            "user_id": 1,
            "username": "admin",
            "role": "admin",
            "display_name": "Administrator",
        },
    )
    monkeypatch.setattr(webapp.db, "get_session_history", lambda **kwargs: ([], 0))
    monkeypatch.setattr(webapp.db, "get_active_users", lambda: [])

    # When: the browser includes the empty select value in the query string.
    response = TestClient(webapp.app).get(
        "/history?patient_id=PATIENT-1&test_date=&qc_status=&result_status=&owner_user_id="
    )

    # Then: the empty optional filter is accepted instead of returning HTTP 422.
    assert response.status_code == 200
