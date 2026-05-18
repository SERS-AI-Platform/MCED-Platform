"""
SERS Clinical Webapp - PDF Report Generation
WeasyPrint 기반 임상 보고서 생성
"""

import io
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from . import clinical_db as db
from . import clinical_qc as qc_utils
from . import clinical_decision as decision_utils
from .clinical_i18n import STRINGS
from .sers_predict import SSI_DECISION_CUTOFF, probability_to_ssi

TEMPLATE_DIR = Path(__file__).parent / "templates"


def get_patient_ssi_score(patient_decision: dict, probability_threshold: float | None = None) -> float:
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
    template = env.get_template("report_template.html")

    s = STRINGS.get(lang, STRINGS["ko"])

    # Prepare cancer types sorted by probability
    cancer_types_sorted = []
    probs = prediction.get("cancer_type_probabilities") or {}
    for code, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        cancer_types_sorted.append({"code": code, "prob": prob})

    qc_summary = qc_utils.build_qc_summary(spectra, qc_utils.DEFAULT_MIN_VALID_COUNT)
    type_confidence = decision_utils.type_confidence(cancer_types_sorted)
    ssi = decision_utils.screening_index_to_ssi(prediction.get("screening_index"))
    final_decision = decision_utils.final_decision(prediction, qc_summary["valid"])

    return template.render(
        s=s,
        session=session,
        prediction=prediction,
        spectra=spectra,
        cancer_types_sorted=cancer_types_sorted,
        type_confidence=type_confidence,
        ssi=ssi,
        final_decision=final_decision,
        qc_summary=qc_summary,
        qc_passed=qc_summary["passed"],
        qc_total=qc_summary["total"],
        report_id=report_id,
        operator_name=operator_name,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
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

    Falls back to HTML-only if WeasyPrint is not installed.
    """
    html = generate_report_html(session, prediction, spectra, report_id, operator_name, lang)

    try:
        from weasyprint import HTML
        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    except ImportError:
        # WeasyPrint not installed — return HTML bytes as fallback
        return html.encode("utf-8")


def create_and_save_report(
    session_id: str,
    user_id: int,
    lang: str = "ko",
) -> tuple[str, bytes]:
    """Create report record in DB and generate PDF.

    Returns (report_id, pdf_bytes).
    """
    session = db.get_session(session_id)
    prediction_row = db.get_prediction(session_id)
    spectra = db.get_spectra_for_session(session_id)
    user = db.get_user_by_id(user_id)

    # Flatten prediction for template
    pred = prediction_row["result_json"] if prediction_row else {}
    patient_decision = pred.get("patient_decision", {})
    ssi_score = get_patient_ssi_score(patient_decision, session.get("threshold"))
    prediction_data = {
        "cancer_detected": patient_decision.get("cancer_detected", False),
        "screening_index": ssi_score,
        "ssi_score": ssi_score,
        "ssi_threshold": patient_decision.get("ssi_threshold", SSI_DECISION_CUTOFF),
        "model_probability_mean": patient_decision.get("model_probability_mean"),
        "model_probability_threshold": patient_decision.get("model_probability_threshold"),
        "majority_vote": patient_decision.get("majority_vote"),
        "cancer_type_prediction": pred.get("cancer_type_prediction"),
        "cancer_type_confidence": pred.get("cancer_type_confidence"),
        "cancer_type_probabilities": pred.get("cancer_type_probabilities", {}),
    }

    report_id = db.create_report(session_id, user_id)
    db.update_session_status(session_id, "reported")
    db.log_audit("report_generate", user_id=user_id, session_id=session_id, detail={"report_id": report_id})

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
