"""
SERS Clinical Webapp - PDF Report Generation
WeasyPrint 기반 임상 보고서 생성
"""

import io
from datetime import datetime
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from . import clinical_db as db
from .clinical_i18n import STRINGS

TEMPLATE_DIR = Path(__file__).parent / "templates"


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

    qc_passed = sum(1 for sp in spectra if sp.get("qc_pass"))
    qc_total = len(spectra)

    return template.render(
        s=s,
        session=session,
        prediction=prediction,
        spectra=spectra,
        cancer_types_sorted=cancer_types_sorted,
        qc_passed=qc_passed,
        qc_total=qc_total,
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
    prediction_data = {
        "cancer_detected": patient_decision.get("cancer_detected", False),
        "screening_index": patient_decision.get("screening_index", 0.0),
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
