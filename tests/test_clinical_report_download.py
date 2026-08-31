import anyio
import pytest
from starlette.requests import Request

from scripts.deployment import clinical_db as db
from scripts.deployment import sers_clinical_webapp as webapp


def _request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "http_version": "1.1",
            "method": "GET",
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


def _mock_authorized_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {"user_id": 1, "role": "admin"},
    )
    monkeypatch.setattr(
        webapp.db,
        "get_session_for_user",
        lambda session_id, *, user_id, is_admin: {"id": session_id},
    )
    monkeypatch.setattr(webapp.db, "log_audit", lambda *args, **kwargs: None)


def test_pdf_report_download_uses_attachment_content_disposition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an authenticated patient session with a generated PDF report.
    _mock_authorized_session(monkeypatch)
    monkeypatch.setattr(webapp.db, "get_spectra_for_session", lambda session_id: [])
    monkeypatch.setattr(webapp.db, "log_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        webapp.report,
        "create_and_save_report",
        lambda **kwargs: ("ABCD1234", b"%PDF-1.7\n"),
    )

    # When: the user selects report generation.
    response = anyio.run(webapp.report_pdf, _request("/patient/test/report.pdf"), "test")

    # Then: the browser receives a file download instead of an inline PDF view.
    assert response.media_type == "application/pdf"
    assert response.headers["content-disposition"] == (
        'attachment; filename="SERS_Report_ABCD1234.pdf"'
    )


def test_report_generation_page_contains_explicit_pdf_download(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an authenticated patient session that can produce a report.
    _mock_authorized_session(monkeypatch)
    monkeypatch.setattr(webapp.db, "get_spectra_for_session", lambda session_id: [])

    # When: the user opens report generation.
    response = anyio.run(webapp.report_page, _request("/patient/test/report"), "test")

    # Then: the page exposes a separate PDF download action.
    assert response.template.name == "report_ready.html"
    assert response.context["session"]["id"] == "test"


def test_csv_download_reuses_report_id_and_logs_each_download(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "clinical_data.db"
    monkeypatch.setattr(db, "DEFAULT_DB_PATH", db_path)
    db.init_db(db_path)
    db.ensure_default_users()
    session_id = db.create_session(
        patient_id="PATIENT-CSV",
        age=50,
        sex="F",
        bmi=23.0,
        operating_mode="balanced",
        created_by=1,
        threshold=0.6,
    )
    for index in range(5):
        spectrum_id = db.add_spectrum(session_id, f"replicate-{index}.csv")
        db.update_spectrum_qc(spectrum_id, True, [], 0.99, 100.0)
    db.update_session_status(session_id, "analyzed", qc_valid=True)
    db.save_prediction(
        session_id,
        {
            "cancer_detected": True,
            "screening_index": 5.0,
            "decision_level": "positive",
            "decision_policy": "mean_ssi_three_band_v1",
            "cancer_signal_spectra_count": 3,
            "qc_valid_spectra_count": 5,
            "patient_decision": {
                "cancer_detected": True,
                "ssi_score": 5.0,
                "decision_level": "positive",
                "decision_policy": "mean_ssi_three_band_v1",
                "cancer_signal_spectra_count": 3,
                "qc_valid_spectra_count": 5,
            },
            "cancer_type_probabilities": {},
        },
    )
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {"user_id": 1, "role": "admin"},
    )

    first = anyio.run(
        webapp.report_csv,
        _request(f"/patient/{session_id}/report.csv"),
        session_id,
    )
    second = anyio.run(
        webapp.report_csv,
        _request(f"/patient/{session_id}/report.csv"),
        session_id,
    )

    report_id = db.get_reports_for_session(session_id)[0]["id"]
    csv_text = first.body.decode("utf-8-sig")
    assert first.headers["content-disposition"] == (
        f'attachment; filename="SERS_Report_{report_id}.csv"'
    )
    assert second.headers["content-disposition"] == first.headers["content-disposition"]
    assert "암 신호 스펙트럼(참고),3" in csv_text
    assert "QC 통과 스펙트럼,5" in csv_text
    assert "최종 판정,추가 확인 권고" in csv_text
    assert "판정 정책,평균 SSI 3단계 판정" in csv_text
    assert "risk_band_interpretation," in csv_text
    assert sum(log["action"] == "report_download" for log in db.get_audit_logs()) == 2
