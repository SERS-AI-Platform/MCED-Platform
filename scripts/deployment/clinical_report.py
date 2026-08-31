"""
SERS Clinical Webapp - PDF Report Generation
WeasyPrint 기반 임상 보고서 생성
"""

import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup, escape

from . import clinical_db as db
from . import clinical_decision as decision_utils
from . import clinical_qc as qc_utils
from . import clinical_result_data as result_data
from .clinical_i18n import STRINGS
from .sers_predict import SSI_DECISION_CUTOFF, probability_to_ssi

TEMPLATE_DIR = Path(__file__).parent / "templates"


class PdfGenerationError(RuntimeError):
    pass


class MissingPredictionError(RuntimeError):
    """Raised when a QC-valid session cannot be reported without a prediction."""


def _keep_words_together(value: str) -> Markup:
    parts = re.split(r"(\s+)", value)
    joined = "".join(part if not part or part.isspace() else "\u2060".join(part) for part in parts)
    return escape(joined)


def get_patient_ssi_score(
    patient_decision: dict, probability_threshold: float | None = None
) -> float:
    """Return SSI score, converting legacy 0-1 saved probabilities when needed."""
    if patient_decision.get("ssi_score") is not None:
        return patient_decision["ssi_score"]

    score = patient_decision.get(
        "screening_index", patient_decision.get("cancer_signal_score", 0.0)
    )
    if (
        patient_decision.get("model_probability_mean") is None
        and score is not None
        and float(score) <= 1.0
        and probability_threshold is not None
    ):
        return round(probability_to_ssi(float(score), probability_threshold), 2)

    return score


def generate_report_html(
    session: dict,
    prediction: dict,
    spectra: list[dict],
    report_id: str,
    operator_name: str,
    lang: str = "ko",
) -> str:
    """Render report HTML from template."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
    env.filters["keep_words_together"] = _keep_words_together
    template = env.get_template("report_template.html")

    s = STRINGS.get(lang, STRINGS["ko"])

    # Prepare cancer types sorted by probability
    cancer_types_sorted = []
    probs = prediction.get("cancer_type_probabilities") or {}
    for code, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        cancer_types_sorted.append({"code": code, "prob": prob})

    qc_summary = qc_utils.build_qc_summary(spectra, qc_utils.DEFAULT_MIN_VALID_COUNT)
    type_confidence = decision_utils.type_confidence(cancer_types_sorted)
    # Use get_patient_ssi_score (not the bare screening_index_to_ssi shim): current-format
    # predictions always carry ssi_score, which must be returned as-is. The bare shim
    # reinterprets any value <=1.0 as a legacy 0-1 probability and rescales it ×10 — for
    # current predictions that collides with genuine low SSI values (the new LOW band is
    # 0-1), silently inflating a real low-risk patient's displayed risk.
    ssi = get_patient_ssi_score(prediction, prediction.get("model_probability_threshold"))
    final_decision = decision_utils.final_decision(prediction, qc_summary["valid"])
    risk_info = decision_utils.ssi_risk_info(ssi)

    return template.render(
        s=s,
        session=session,
        prediction=prediction,
        spectra=spectra,
        cancer_types_sorted=cancer_types_sorted,
        type_confidence=type_confidence,
        ssi=ssi,
        risk_info=risk_info,
        final_decision=final_decision,
        is_legacy_decision=prediction.get("decision_policy")
        != decision_utils.MEAN_SSI_DECISION_POLICY,
        qc_summary=qc_summary,
        qc_passed=qc_summary["passed"],
        qc_total=qc_summary["total"],
        report_id=report_id,
        operator_name=operator_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        lang=lang,
    )


def generate_report_pdf(
    session: dict,
    prediction: dict,
    spectra: list[dict],
    report_id: str,
    operator_name: str,
    lang: str = "ko",
) -> bytes:
    """Generate PDF report. Returns PDF bytes.

    Uses an installed Chromium browser to render PDF when WeasyPrint is unavailable.
    """
    html = generate_report_html(session, prediction, spectra, report_id, operator_name, lang)

    try:
        from weasyprint import HTML

        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    except (ImportError, OSError):
        return _generate_pdf_with_browser(html)


def _generate_pdf_with_browser(html: str) -> bytes:
    browser = next(
        (Path(found) for name in ("msedge", "chrome") if (found := shutil.which(name)) is not None),
        None,
    )
    if browser is None:
        candidates = (
            Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
            Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
            Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        )
        browser = next((candidate for candidate in candidates if candidate.exists()), None)
    if browser is None:
        raise PdfGenerationError("PDF browser renderer is not available")

    with tempfile.TemporaryDirectory(prefix="sers_report_") as directory:
        work_dir = Path(directory)
        html_path = work_dir / "report.html"
        pdf_path = work_dir / "report.pdf"
        profile_path = work_dir / "browser-profile"
        html_path.write_text(html, encoding="utf-8")
        command = [
            str(browser),
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--user-data-dir={profile_path}",
            f"--print-to-pdf={pdf_path}",
            html_path.as_uri(),
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except subprocess.TimeoutExpired as error:
            raise PdfGenerationError("PDF browser renderer timed out") from error
        if completed.returncode != 0 or not pdf_path.exists():
            browser_output = completed.stderr.strip() or completed.stdout.strip()
            raise PdfGenerationError(
                f"PDF browser renderer failed (exit {completed.returncode}): "
                f"{browser_output or 'no browser output'}"
            )
        pdf_bytes = pdf_path.read_bytes()
        if not pdf_bytes.startswith(b"%PDF-"):
            raise PdfGenerationError("PDF browser renderer returned invalid data")
        return pdf_bytes


def create_and_save_report(
    session: dict,
    user_id: int,
    lang: str = "ko",
) -> tuple[str, bytes]:
    """Create report record in DB and generate PDF.

    Returns (report_id, pdf_bytes).
    """
    session_id = session["id"]
    prediction_row = db.get_prediction(session_id)
    spectra = db.get_spectra_for_session(session_id)
    user = db.get_user_by_id(user_id)
    qc_summary = qc_utils.build_qc_summary(spectra, qc_utils.DEFAULT_MIN_VALID_COUNT)
    if qc_summary["valid"] and not prediction_row:
        raise MissingPredictionError("A valid QC session has no prediction result")

    # Flatten prediction for template
    pred = prediction_row["result_json"] if prediction_row else {}
    patient_decision = pred.get("patient_decision", {})
    cancer_signal_count, qc_valid_count = result_data.get_signal_counts(pred)
    ssi_score = get_patient_ssi_score(patient_decision, session.get("threshold"))
    prediction_data = {
        "cancer_detected": patient_decision.get("cancer_detected", False),
        "screening_index": ssi_score,
        "ssi_score": ssi_score,
        "ssi_threshold": patient_decision.get("ssi_threshold", SSI_DECISION_CUTOFF),
        "model_probability_mean": patient_decision.get("model_probability_mean"),
        "model_probability_threshold": patient_decision.get("model_probability_threshold"),
        "decision_level": patient_decision.get("decision_level")
        or (prediction_row.get("decision_level") if prediction_row else None),
        "decision_policy": patient_decision.get("decision_policy")
        or (prediction_row.get("decision_policy") if prediction_row else None),
        "majority_vote": patient_decision.get("majority_vote"),
        "cancer_signal_spectra_count": cancer_signal_count,
        "qc_valid_spectra_count": qc_valid_count,
        "cancer_type_prediction": pred.get("cancer_type_prediction"),
        "cancer_type_confidence": pred.get("cancer_type_confidence"),
        "cancer_type_probabilities": pred.get("cancer_type_probabilities", {}),
    }

    report_id = db.create_report(session_id, user_id)
    db.update_session_status(session_id, "reported")

    operator_name = user["display_name"] if user else "Unknown"

    pdf_bytes = generate_report_pdf(
        session=session,
        prediction=prediction_data,
        spectra=spectra,
        report_id=report_id,
        operator_name=operator_name,
        lang=lang,
    )

    return report_id, pdf_bytes
