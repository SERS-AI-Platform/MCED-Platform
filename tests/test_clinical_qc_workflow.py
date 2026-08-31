import io
from collections.abc import Mapping
from pathlib import Path

import anyio
import numpy as np
import pytest
from starlette.datastructures import UploadFile
from starlette.requests import Request

from scripts.deployment import clinical_qc, sers_predict
from scripts.deployment import sers_clinical_webapp as webapp
from scripts.deployment.clinical_report import generate_report_html
from scripts.deployment.sers_predict import ProductionPredictor

SpectrumFixtureValue = bool | str | float | list[str]


def _spectrum(qc_pass: bool, *flags: str) -> Mapping[str, SpectrumFixtureValue]:
    return {
        "qc_pass": qc_pass,
        "qc_flags": list(flags),
        "filename": "replicate.csv",
        "fp_mean": 100.0,
        "replicate_correlation": 0.99,
    }


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


def _mock_authorized_session(monkeypatch: pytest.MonkeyPatch, session: dict | None) -> None:
    monkeypatch.setattr(webapp.auth, "require_auth", lambda request: None)
    monkeypatch.setattr(
        webapp.auth,
        "get_current_user",
        lambda request: {"user_id": 1, "role": "admin"},
    )
    monkeypatch.setattr(
        webapp.db,
        "get_session_for_user",
        lambda session_id, *, user_id, is_admin: session,
    )


@pytest.mark.parametrize(
    "critical_flag",
    ["intensity_gate_fail", "cosmic_spike", "detector_saturation"],
)
def test_critical_qc_failure_invalidates_test_even_with_three_passing_spectra(
    critical_flag: str,
) -> None:
    # Given: enough passing replicates by count, but one acquisition-critical failure.
    spectra = [
        _spectrum(True),
        _spectrum(True),
        _spectrum(True),
        _spectrum(False, critical_flag),
        _spectrum(False, "low_correlation"),
    ]

    # When: patient-level QC validity is calculated.
    summary = clinical_qc.build_qc_summary(spectra, min_valid_count=3)

    # Then: the critical failure blocks interpretation despite the passing count.
    assert summary["passed"] == 3
    assert summary["has_critical_failure"] is True
    assert summary["valid"] is False


def test_noncritical_qc_failures_allow_results_when_three_spectra_pass() -> None:
    # Given: three passing replicates and only noncritical correlation failures.
    spectra = [
        _spectrum(True),
        _spectrum(True),
        _spectrum(True),
        _spectrum(False, "low_correlation"),
        _spectrum(False, "corr_below_threshold"),
    ]

    # When: patient-level QC validity is calculated.
    summary = clinical_qc.build_qc_summary(spectra, min_valid_count=3)

    # Then: the existing minimum-count rule still permits interpretation.
    assert summary["has_critical_failure"] is False
    assert summary["valid"] is True


def test_persisted_critical_flag_invalidates_even_if_pass_state_is_inconsistent() -> None:
    # Given: a persisted row incorrectly marked as passing despite saturation.
    spectra = [_spectrum(True, "detector_saturation") for _ in range(5)]

    # When: patient-level QC validity is recalculated from persisted evidence.
    summary = clinical_qc.build_qc_summary(spectra, min_valid_count=3)

    # Then: the critical flag fails closed and blocks result interpretation.
    assert summary["passed"] == 5
    assert summary["reason_counts"]["saturation"] == 5
    assert summary["valid"] is False


@pytest.mark.parametrize(
    ("defect", "expected_flag"),
    [("spike", "cosmic_ray_spike"), ("saturation", "detector_saturation")],
)
def test_predictor_qc_detects_critical_raw_spectrum_defects(
    defect: str, expected_flag: str
) -> None:
    # Given: a raw spectrum with either one isolated spike or a clipped plateau.
    x = np.arange(50, dtype=float)
    y = np.linspace(10.0, 20.0, 50)
    if defect == "spike":
        y[25] = 100.0
    else:
        y[22:27] = 100.0
    predictor = ProductionPredictor.__new__(ProductionPredictor)
    predictor.prep = {"trim_region": [0.0, 49.0]}
    spectra = [{"x": x, "y": y, "features": np.linspace(0.0, 1.0, 50)}]

    # When: the same QC producer used by patient prediction evaluates the spectrum.
    checked = predictor._qc_replicate_set(spectra)

    # Then: the raw defect is persisted as a critical QC failure flag.
    assert checked[0]["qc_pass"] is False
    assert expected_flag in checked[0]["qc_flags"]


def test_patient_prediction_stops_before_inference_on_critical_qc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: four normal files and one raw file containing a cosmic-ray spike.
    def fake_read_spectrum(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        x = np.arange(50, dtype=float)
        y = np.linspace(10.0, 20.0, 50)
        if "critical" in filepath.name:
            y[25] = 100.0
        return x, y

    monkeypatch.setattr(sers_predict, "read_spectrum", fake_read_spectrum)
    predictor = ProductionPredictor.__new__(ProductionPredictor)
    predictor.prep = {"trim_region": [0.0, 49.0]}
    predictor.preprocess = lambda x, y, instrument: np.linspace(0.0, 1.0, 50)
    filepaths = [Path(f"normal_{index}.csv") for index in range(4)] + [Path("critical.csv")]

    # When: patient prediction reaches the production QC boundary.
    result = predictor.predict_patient(filepaths)

    # Then: inference output is absent and only the invalid QC summary is returned.
    assert result["status"] == "qc_invalid"
    assert result["qc_summary"]["passed"] == 4
    assert len(result["qc_summary"]["passes"]) == 4
    assert "patient_decision" not in result

    # When: the upload persistence boundary maps the early-return QC summary.
    persisted = {
        filepath.name: webapp._spectrum_qc_update(filepath.name, result) for filepath in filepaths
    }

    # Then: four passing files remain passes and only the critical file fails.
    assert sum(update[0] for update in persisted.values()) == 4
    assert persisted["critical.csv"][0] is False
    assert "cosmic_ray_spike" in persisted["critical.csv"][1]


def test_patient_prediction_stops_before_inference_below_minimum_pass_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: five readable spectra, but only two pass correlation QC.
    monkeypatch.setattr(
        sers_predict,
        "read_spectrum",
        lambda filepath: (np.arange(50, dtype=float), np.linspace(10.0, 20.0, 50)),
    )
    predictor = ProductionPredictor.__new__(ProductionPredictor)
    predictor.prep = {"trim_region": [0.0, 49.0]}
    predictor.preprocess = lambda x, y, instrument: np.linspace(0.0, 1.0, 50)
    predictor._qc_replicate_set = lambda spectra: [
        {
            **spectrum,
            "qc_pass": index < 2,
            "qc_flags": [] if index < 2 else ["low_correlation"],
        }
        for index, spectrum in enumerate(spectra)
    ]

    # When: prediction reaches the patient-level minimum-count boundary.
    result = predictor.predict_patient([Path(f"replicate_{index}.csv") for index in range(5)])

    # Then: correlation-only failure below 3-of-5 still blocks inference.
    assert result["status"] == "qc_invalid"
    assert result["qc_summary"]["passed"] == 2
    assert "patient_decision" not in result


def test_read_errors_are_counted_as_qc_failures_before_inference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: only one of five uploaded spectra can be read.
    def fake_read_spectrum(filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        if filepath.name != "readable.csv":
            raise ValueError("unreadable")
        return np.arange(50, dtype=float), np.linspace(10.0, 20.0, 50)

    monkeypatch.setattr(sers_predict, "read_spectrum", fake_read_spectrum)
    predictor = ProductionPredictor.__new__(ProductionPredictor)
    predictor.prep = {"trim_region": [0.0, 49.0]}
    predictor.preprocess = lambda x, y, instrument: np.linspace(0.0, 1.0, 50)

    # When: the five-file patient prediction is attempted.
    result = predictor.predict_patient(
        [Path("readable.csv")] + [Path(f"unreadable_{index}.csv") for index in range(4)]
    )

    # Then: unreadable files are failures and no model output is produced.
    assert result["status"] == "qc_invalid"
    assert result["qc_summary"]["total"] == 5
    assert result["qc_summary"]["passed"] == 1
    assert result["qc_summary"]["failed"] == 4
    assert all(failure["flags"] == ["read_error"] for failure in result["qc_summary"]["failures"])
    assert "patient_decision" not in result


def test_invalid_results_route_redirects_to_qc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an authenticated session with a saved prediction and saturation failure.
    _mock_authorized_session(monkeypatch, {"id": "test"})
    monkeypatch.setattr(webapp.db, "get_prediction", lambda session_id: {"result_json": {}})
    monkeypatch.setattr(
        webapp.db,
        "get_spectra_for_session",
        lambda session_id: [_spectrum(True) for _ in range(4)]
        + [_spectrum(False, "detector_saturation")],
    )

    # When: the invalid result URL is requested directly.
    response = anyio.run(webapp.results_page, _request("/patient/test/results"), "test")

    # Then: the route redirects to QC before reading prediction values.
    assert response.status_code == 303
    assert response.headers["location"] == "/patient/test/qc"


def test_invalid_csv_route_omits_prediction_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a saved positive prediction paired with a saturation-invalid test.
    prediction = {
        "patient_decision": {
            "cancer_detected": True,
            "ssi_score": 9.55,
            "majority_vote": "4/4",
            "model_probability_mean": 0.9697,
        },
        "cancer_type_probabilities": {"BLC": 0.6545},
    }
    _mock_authorized_session(
        monkeypatch,
        {
            "id": "test",
            "patient_id": "QA",
            "age": 55,
            "sex": "M",
            "bmi": 24.0,
            "test_id": "QA-20260722-01",
            "threshold": 0.6,
        },
    )
    monkeypatch.setattr(
        webapp.db,
        "create_report",
        lambda session_id, user_id: "QA-20260722-01-R01",
    )
    monkeypatch.setattr(webapp.db, "log_audit", lambda *args, **kwargs: None)
    monkeypatch.setattr(webapp.db, "get_prediction", lambda session_id: {"result_json": prediction})
    monkeypatch.setattr(
        webapp.db,
        "get_spectra_for_session",
        lambda session_id: [_spectrum(True) for _ in range(4)]
        + [_spectrum(False, "detector_saturation")],
    )

    # When: the invalid CSV report is requested.
    response = anyio.run(webapp.report_csv, _request("/patient/test/report.csv"), "test")
    csv_text = response.body.decode("utf-8-sig")

    # Then: QC and retest guidance remain without any prediction output rows.
    assert "screening_index" not in csv_text
    assert "risk_level" not in csv_text
    assert "top_type" not in csv_text
    assert "권고 재검 유형" in csv_text


def test_report_route_redirects_unknown_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an authenticated request for a session that does not exist.
    _mock_authorized_session(monkeypatch, None)

    # When: the PDF report route is requested directly.
    response = anyio.run(webapp.report_page, _request("/patient/missing/report"), "missing")

    # Then: the route does not disclose whether the inaccessible record exists.
    assert response.status_code == 404


def test_valid_qc_without_prediction_cannot_generate_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: an authenticated session with valid QC rows but no saved prediction.
    _mock_authorized_session(monkeypatch, {"id": "test"})
    monkeypatch.setattr(
        webapp.db,
        "get_spectra_for_session",
        lambda session_id: [_spectrum(True) for _ in range(5)],
    )
    monkeypatch.setattr(webapp.db, "get_prediction", lambda session_id: None)

    # When: the PDF endpoint is requested directly.
    response = anyio.run(webapp.report_page, _request("/patient/test/report"), "test")

    # Then: no empty negative report is fabricated.
    assert response.status_code == 303
    assert response.headers["location"] == "/patient/test/qc"


def test_duplicate_upload_filenames_are_rejected_before_prediction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: five uploaded parts that reuse the same basename.
    _mock_authorized_session(
        monkeypatch,
        {"id": "test", "age": 55, "sex": "M", "bmi": None},
    )
    uploads = [UploadFile(file=io.BytesIO(b"1,2\n"), filename="same.csv") for _ in range(5)]

    # When: the upload endpoint validates the multipart set.
    response = anyio.run(
        webapp.upload_submit,
        _request("/patient/test/upload"),
        "test",
        uploads,
    )

    # Then: the overwrite-prone set is rejected before model loading or QC.
    assert response.status_code == 400
    assert "unique filename" in response.body.decode()


def test_invalid_report_omits_prediction_outputs() -> None:
    # Given: a prediction with distinctive outputs paired with a critical QC failure.
    prediction = {
        "cancer_detected": True,
        "screening_index": 9.55,
        "ssi_score": 9.55,
        "ssi_threshold": 4.0,
        "model_probability_mean": 0.9697,
        "model_probability_threshold": 0.6,
        "majority_vote": "1/1",
        "cancer_type_probabilities": {"BLC": 0.6545, "LUN": 0.192},
    }
    spectra = [
        _spectrum(True),
        _spectrum(True),
        _spectrum(True),
        _spectrum(False, "detector_saturation"),
        _spectrum(False, "low_correlation"),
    ]

    # When: the invalid clinical report is rendered.
    html = generate_report_html(
        session={"patient_id": "QA-INVALID", "age": 55, "sex": "M", "bmi": None},
        prediction=prediction,
        spectra=spectra,
        report_id="REPORT-INVALID",
        operator_name="QA",
        lang="ko",
    )

    # Then: only invalid-test guidance remains, without any model output values.
    assert "9.55" not in html
    assert "0.9697" not in html
    assert "0.6545" not in html
    assert "BLC" not in html
    assert "1/1" not in html
    assert "REPORT-INVALID" in html
