#!/usr/bin/env python3
"""
SERS Clinical Webapp — IEC 62366 Compliant Clinical Interface

임상용 사용적합성 테스트 대응 웹앱. 환자 중심 step-by-step 워크플로우.

Usage:
    python scripts/deployment/sers_clinical_webapp.py [--port 8080]

Requires:
    pip install fastapi uvicorn jinja2 python-multipart
"""

from __future__ import annotations

import json
import csv
import io
import sys
import tempfile
import argparse
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment import clinical_db as db
from scripts.deployment import clinical_auth as auth
from scripts.deployment import clinical_report as report
from scripts.deployment import clinical_qc as qc_utils
from scripts.deployment import clinical_decision as decision_utils
from scripts.deployment.clinical_i18n import get_strings, get_lang, LANG_COOKIE
from scripts.deployment.sers_predict import (
    ProductionPredictor,
    StackingPredictor,
    SSI_DECISION_CUTOFF,
    probability_to_ssi,
)

logger = logging.getLogger(__name__)

# --- App Setup ---
app = FastAPI(title="SERS Clinical Screening", version="1.0")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global predictor (loaded once)
predictor: ProductionPredictor | None = None
MODEL_DISPLAY_NAME = "uSERS-Net Ver1"
DECISION_PROFILE = "standard_balanced"
DB_OPERATING_MODE = "balanced"


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

# Usability-test clinical workflow constants.
REQUIRED_SPECTRA_COUNT = 5
MIN_QC_PASS_COUNT = qc_utils.DEFAULT_MIN_VALID_COUNT
ALLOWED_SPECTRUM_EXTENSIONS = {".csv", ".txt"}

MANUAL_CANDIDATES = [
    BASE_DIR / "manuals" / "software_ifu.pdf",
    Path("/mnt/c/Users/user/Downloads/SERS_Software_IFU.pdf"),
    Path("/mnt/c/Users/user/Downloads/SERS_Software_IFU.docx"),
    Path("/mnt/c/Users/user/OneDrive - solum/바탕 화면/AI BD/사용적합성/소프트웨어 사용설명서_IFU.pdf"),
    Path("/mnt/c/Users/user/OneDrive - solum/바탕 화면/AI BD/사용적합성/소프트웨어 사용설명서_IFU.docx"),
]


def get_predictor() -> ProductionPredictor:
    global predictor
    if predictor is None:
        # Prefer stacking model if available, fall back to LR
        stacking_dir = PROJECT_ROOT / "artifacts" / "usersnet" / "current"
        if stacking_dir.exists():
            predictor = StackingPredictor(stacking_dir)
            logger.info(f"Stacking V2 model loaded: {predictor.cancer_types}, "
                        f"standard decision profile: {DECISION_PROFILE}")
        else:
            predictor = ProductionPredictor()
            logger.info(f"LR model loaded: {predictor.cancer_types}, "
                        f"standard decision profile: {DECISION_PROFILE}")
    return predictor


def template_context(request: Request, **kwargs) -> dict:
    """Build common template context."""
    user = auth.get_current_user(request)
    lang = get_lang(request)
    s = get_strings(request)
    auto_logout_message = s["auto_logout_notice"].format(
        minutes=auth.SESSION_EXPIRY_MINUTES
    )
    return {
        "request": request,
        "user": user,
        "lang": lang,
        "s": s,
        "decision_profile": DECISION_PROFILE,
        "idle_timeout_minutes": auth.SESSION_EXPIRY_MINUTES,
        "auto_logout_message": auto_logout_message,
        **kwargs,
    }


# --- Routes ---

@app.on_event("startup")
async def startup():
    db.init_db()
    db.ensure_default_users()
    get_predictor()  # Pre-load model
    logger.info("Clinical webapp started")


@app.get("/", response_class=HTMLResponse)
async def root():
    return RedirectResponse("/login")


# --- Login/Logout ---

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    user = auth.get_current_user(request)
    if user:
        return RedirectResponse("/patient/new", status_code=303)
    ctx = template_context(request, error=False, registered=request.query_params.get("registered") == "1")
    return templates.TemplateResponse("login.html", ctx)


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    user = auth.get_current_user(request)
    if user:
        return RedirectResponse("/patient/new", status_code=303)
    ctx = template_context(request, error=False, success=False)
    return templates.TemplateResponse("register.html", ctx)


@app.post("/register")
async def register_submit(
    request: Request,
    username: str = Form(...),
    display_name: str = Form(...),
    role: str = Form(...),
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    s = get_strings(request)

    # Validation
    if not all([username.strip(), display_name.strip(), role, password]):
        ctx = template_context(
            request, error=True, success=False,
            error_message=s["register_error_fields"],
            form_username=username, form_display_name=display_name, form_role=role,
        )
        return templates.TemplateResponse("register.html", ctx)

    if password != password_confirm:
        ctx = template_context(
            request, error=True, success=False,
            error_message=s["register_error_password"],
            form_username=username, form_display_name=display_name, form_role=role,
        )
        return templates.TemplateResponse("register.html", ctx)

    if role not in ("technician", "clinician"):
        role = "technician"

    # Check duplicate
    existing = db.get_user_by_username(username.strip())
    if existing:
        ctx = template_context(
            request, error=True, success=False,
            error_message=s["register_error_exists"],
            form_username=username, form_display_name=display_name, form_role=role,
        )
        return templates.TemplateResponse("register.html", ctx)

    # Create user
    password_hash = auth.hash_password(password)
    db.create_user(username.strip(), password_hash, role, display_name.strip())
    db.log_audit("user_register", detail={"username": username.strip(), "role": role})

    return RedirectResponse("/login?registered=1", status_code=303)


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    token = auth.login(username, password)
    if not token:
        ctx = template_context(request, error=True)
        return templates.TemplateResponse("login.html", ctx)

    response = RedirectResponse("/patient/new", status_code=303)
    response.set_cookie(auth.SESSION_COOKIE, token, httponly=True, max_age=auth.SESSION_EXPIRY_MINUTES * 60)
    return response


@app.get("/logout")
async def logout(request: Request):
    token = auth.get_session_token(request)
    if token:
        auth.logout(token)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


# --- Operating Mode ---

@app.get("/mode", response_class=HTMLResponse)
async def mode_page(request: Request):
    return RedirectResponse("/patient/new", status_code=303)


@app.post("/mode")
async def mode_submit(request: Request, mode: str = Form(...)):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect
    return RedirectResponse("/patient/new", status_code=303)


# --- Patient Registration ---

@app.get("/patient/new", response_class=HTMLResponse)
async def patient_page(request: Request):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    ctx = template_context(request, show_stepper=True, current_step="patient", completed_steps=[])
    return templates.TemplateResponse("patient_register.html", ctx)


@app.post("/patient/new")
async def patient_submit(
    request: Request,
    patient_id: str = Form(...),
    age: float = Form(...),
    sex: str = Form(...),
    bmi: float = Form(None),
):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    pred = get_predictor()
    threshold = pred.threshold

    session_id = db.create_session(
        patient_id=patient_id,
        age=age,
        sex=sex,
        bmi=bmi,
        operating_mode=DB_OPERATING_MODE,
        created_by=user["user_id"],
        threshold=threshold,
    )

    db.log_audit(
        "patient_create",
        user_id=user["user_id"],
        session_id=session_id,
        detail={"patient_id": patient_id, "age": age, "sex": sex, "bmi": bmi},
    )

    return RedirectResponse(f"/patient/{session_id}/upload", status_code=303)


# --- Spectrum Upload & Analysis ---

@app.get("/patient/{session_id}/upload", response_class=HTMLResponse)
async def upload_page(request: Request, session_id: str):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    session = db.get_session(session_id)
    if not session:
        return RedirectResponse("/patient/new", status_code=303)

    ctx = template_context(
        request,
        session=session,
        show_stepper=True,
        current_step="upload",
        completed_steps=["patient"],
        remeasure=request.query_params.get("remeasure") == "1",
    )
    return templates.TemplateResponse("spectrum_upload.html", ctx)


@app.post("/patient/{session_id}/upload")
async def upload_submit(request: Request, session_id: str, files: list[UploadFile] = File(...)):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    session = db.get_session(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, 404)

    pred = get_predictor()

    valid_uploads = []
    rejected_files = []
    for f in files:
        filename = Path(f.filename or "").name
        suffix = Path(filename).suffix.lower()
        if suffix not in ALLOWED_SPECTRUM_EXTENSIONS:
            rejected_files.append(filename or "(unnamed)")
            continue
        content = await f.read()
        valid_uploads.append((filename, content))

    if rejected_files:
        return JSONResponse(
            {
                "error": (
                    "Only CSV or TXT spectrum files are allowed. "
                    f"Rejected: {', '.join(rejected_files)}"
                )
            },
            400,
        )
    if len(valid_uploads) != REQUIRED_SPECTRA_COUNT:
        return JSONResponse(
            {
                "error": (
                    f"Exactly {REQUIRED_SPECTRA_COUNT} CSV/TXT files are required "
                    "for the usability-test workflow."
                )
            },
            400,
        )

    # Reuploading in the same patient session must replace, not append, spectra.
    # This prevents the formative-evaluation failure mode where browser Back +
    # reupload produced 10 files for one patient.
    if db.get_spectra_for_session(session_id):
        db.reset_session_measurements(session_id)
        db.log_audit(
            "measurement_reupload_replace",
            user_id=user["user_id"],
            session_id=session_id,
            detail={"reason": "same_patient_reupload"},
        )

    # Save uploaded files to temp directory and record in DB
    temp_dir = tempfile.mkdtemp(prefix="sers_clinical_")
    filepaths = []
    spectrum_ids = []

    for filename, content in valid_uploads:
        temp_path = Path(temp_dir) / filename
        temp_path.write_bytes(content)
        filepaths.append(temp_path)
        spec_id = db.add_spectrum(session_id, filename)
        spectrum_ids.append((spec_id, filename))

    db.update_session_status(session_id, "uploaded")
    db.log_audit(
        "spectrum_upload",
        user_id=user["user_id"],
        session_id=session_id,
        detail={"n_files": len(filepaths), "filenames": [f.name for f in filepaths]},
    )

    # Run prediction
    result = pred.predict_patient(
        filepaths=filepaths,
        age=session["age"],
        sex=session["sex"],
        bmi=session["bmi"],
    )

    # Update spectra QC info
    per_rep = result.get("per_replicate", [])
    qc_summary = result.get("qc_summary", {})
    failures = {f["file"]: f["flags"] for f in qc_summary.get("failures", [])}

    for spec_id, filename in spectrum_ids:
        # Find matching replicate info
        rep_info = next((r for r in per_rep if r["file"] == filename), None)
        is_failed = filename in failures

        if is_failed:
            db.update_spectrum_qc(spec_id, qc_pass=False, qc_flags=failures[filename])
        elif rep_info:
            db.update_spectrum_qc(
                spec_id,
                qc_pass=True,
                qc_flags=[],
                correlation=rep_info.get("replicate_correlation"),
                fp_mean=None,
            )
        else:
            db.update_spectrum_qc(spec_id, qc_pass=True, qc_flags=[])

    db.update_session_status(session_id, "qc_done")

    passed_count = result.get("qc_summary", {}).get("passed", 0)

    # Save available prediction output. The results/report pages decide whether
    # it is usable for interpretation based on the clinical QC validity rule.
    if result.get("status") == "ok":
        patient_decision = result.get("patient_decision", {})
        ssi_score = patient_decision.get(
            "ssi_score", patient_decision.get("screening_index", 0.0)
        )
        prediction_data = {
            "cancer_detected": patient_decision.get("cancer_detected", False),
            "cancer_signal_score": ssi_score,
            "screening_index": ssi_score,
            "ssi_score": ssi_score,
            "ssi_threshold": patient_decision.get("ssi_threshold", SSI_DECISION_CUTOFF),
            "model_probability_mean": patient_decision.get("model_probability_mean"),
            "model_probability_threshold": patient_decision.get("model_probability_threshold"),
            "majority_vote": patient_decision.get("majority_vote"),
            "cancer_type_prediction": result.get("cancer_type_prediction"),
            "cancer_type_confidence": result.get("cancer_type_confidence"),
            "cancer_type_probabilities": result.get("cancer_type_probabilities", {}),
            "per_replicate": per_rep,
            "patient_decision": patient_decision,
        }
        db.save_prediction(session_id, prediction_data)
        db.update_session_status(
            session_id, "analyzed",
            model_variant=result.get("model_variant"),
        )

    db.log_audit(
        "analysis_complete",
        user_id=user["user_id"],
        session_id=session_id,
        detail={
            "status": result.get("status"),
            "qc_passed": passed_count,
            "qc_required": MIN_QC_PASS_COUNT,
            "cancer_detected": result.get("patient_decision", {}).get("cancer_detected"),
            "ssi_score": result.get("patient_decision", {}).get("ssi_score"),
            "model_probability_mean": result.get("patient_decision", {}).get("model_probability_mean"),
        },
    )

    # Redirect to QC page
    return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)


@app.post("/patient/{session_id}/remeasure")
async def remeasure_patient(request: Request, session_id: str):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    session = db.get_session(session_id)
    if not session:
        return RedirectResponse("/patient/new", status_code=303)

    db.reset_session_measurements(session_id)
    db.log_audit(
        "measurement_reupload_start",
        user_id=user["user_id"],
        session_id=session_id,
        detail={"patient_id": session["patient_id"]},
    )
    return RedirectResponse(f"/patient/{session_id}/upload?remeasure=1", status_code=303)


# --- QC Review ---

@app.get("/patient/{session_id}/qc", response_class=HTMLResponse)
async def qc_page(request: Request, session_id: str):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    session = db.get_session(session_id)
    if not session:
        return RedirectResponse("/patient/new", status_code=303)

    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)
    strings = get_strings(request)
    qc_summary_label = strings["qc_summary"].format(
        passed=qc_summary["passed"],
        total=qc_summary["total"],
    )
    qc_status_template = (
        strings["qc_valid_message"]
        if qc_summary["valid"]
        else strings["qc_invalid_message"]
    )
    qc_status_message = qc_status_template.format(
        total=qc_summary["total"],
        passed=qc_summary["passed"],
        rate=qc_summary["pass_rate"],
        min_count=qc_summary["min_valid_count"],
    )

    ctx = template_context(
        request,
        session=session,
        spectra=spectra,
        qc_summary=qc_summary,
        qc_summary_label=qc_summary_label,
        qc_status_message=qc_status_message,
        qc_passed=qc_summary["passed"],
        qc_total=qc_summary["total"],
        min_qc_pass_count=MIN_QC_PASS_COUNT,
        show_stepper=True,
        current_step="qc",
        completed_steps=["patient", "upload"],
    )
    return templates.TemplateResponse("qc_review.html", ctx)


# --- Results ---

@app.get("/patient/{session_id}/results", response_class=HTMLResponse)
async def results_page(request: Request, session_id: str):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    session = db.get_session(session_id)
    if not session:
        return RedirectResponse("/patient/new", status_code=303)

    prediction_row = db.get_prediction(session_id)
    if not prediction_row:
        return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)

    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)

    # Flatten prediction for template
    result_json = prediction_row["result_json"]
    patient_decision = result_json.get("patient_decision", {})
    ssi_score = get_patient_ssi_score(patient_decision, session.get("threshold"))

    prediction = {
        "cancer_detected": patient_decision.get("cancer_detected", False),
        "screening_index": ssi_score,
        "ssi_score": ssi_score,
        "ssi_threshold": patient_decision.get("ssi_threshold", SSI_DECISION_CUTOFF),
        "model_probability_mean": patient_decision.get("model_probability_mean"),
        "model_probability_threshold": patient_decision.get("model_probability_threshold"),
        "majority_vote": patient_decision.get("majority_vote"),
        "cancer_type_prediction": result_json.get("cancer_type_prediction"),
        "cancer_type_confidence": result_json.get("cancer_type_confidence"),
        "cancer_type_probabilities": result_json.get("cancer_type_probabilities", {}),
    }

    # Sort cancer types by probability
    cancer_types_sorted = []
    probs = prediction.get("cancer_type_probabilities") or {}
    for code, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
        cancer_types_sorted.append({"code": code, "prob": prob})

    per_replicate = result_json.get("per_replicate", [])
    type_confidence = decision_utils.type_confidence(cancer_types_sorted)
    ssi = get_patient_ssi_score(prediction, session.get("threshold"))
    final_decision = decision_utils.final_decision(prediction, qc_summary["valid"])
    risk_info = decision_utils.ssi_risk_info(ssi)

    ctx = template_context(
        request,
        session=session,
        prediction=prediction,
        cancer_types_sorted=cancer_types_sorted,
        type_confidence=type_confidence,
        ssi=ssi,
        risk_info=risk_info,
        final_decision=final_decision,
        per_replicate=per_replicate,
        spectra=spectra,
        qc_summary=qc_summary,
        qc_passed=qc_summary["passed"],
        qc_total=qc_summary["total"],
        show_stepper=True,
        current_step="results",
        completed_steps=["patient", "upload", "qc"],
    )
    return templates.TemplateResponse("results.html", ctx)


# --- Report ---

@app.get("/patient/{session_id}/report", response_class=HTMLResponse)
async def report_page(request: Request, session_id: str):
    """Generate and return PDF report."""
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    lang = get_lang(request)

    try:
        report_id, pdf_bytes = report.create_and_save_report(
            session_id=session_id,
            user_id=user["user_id"],
            lang=lang,
        )
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        return HTMLResponse(f"<h2>Report generation error: {e}</h2>", status_code=500)

    # Try returning PDF, fall back to HTML
    try:
        from weasyprint import HTML  # noqa: F401
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'inline; filename="SERS_Report_{report_id}.pdf"'},
        )
    except ImportError:
        # WeasyPrint not installed — return HTML
        return HTMLResponse(content=pdf_bytes.decode("utf-8"))


@app.get("/patient/{session_id}/report.csv")
async def report_csv(request: Request, session_id: str):
    """Download a CSV summary of the current report result."""
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    session = db.get_session(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, 404)

    prediction_row = db.get_prediction(session_id)
    pred = prediction_row["result_json"] if prediction_row else {}
    patient_decision = pred.get("patient_decision", {})
    ssi_score = get_patient_ssi_score(patient_decision, session.get("threshold"))
    prediction = {
        "cancer_detected": patient_decision.get("cancer_detected", False),
        "screening_index": ssi_score,
        "ssi_score": ssi_score,
        "majority_vote": patient_decision.get("majority_vote"),
        "cancer_type_probabilities": pred.get("cancer_type_probabilities", {}) or {},
    }

    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)
    ssi = get_patient_ssi_score(prediction, session.get("threshold"))
    final_decision = decision_utils.final_decision(prediction, qc_summary["valid"])
    cancer_types_sorted = [
        {"code": code, "prob": prob}
        for code, prob in sorted(
            prediction["cancer_type_probabilities"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ]
    type_confidence = decision_utils.type_confidence(cancer_types_sorted)
    risk_info = decision_utils.ssi_risk_info(ssi)

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    strings = get_strings(request)
    writer.writerow(["section", "item", "value"])
    writer.writerow(["검사 정보", "patient_id", session["patient_id"]])
    writer.writerow(["검사 정보", "age", session["age"]])
    writer.writerow(["검사 정보", "sex", session["sex"]])
    writer.writerow(["검사 정보", "model", MODEL_DISPLAY_NAME])
    writer.writerow(["QC 요약", "total", qc_summary["total"]])
    writer.writerow(["QC 요약", "passed", qc_summary["passed"]])
    writer.writerow(["QC 요약", "failed", qc_summary["failed"]])
    writer.writerow(["QC 요약", "pass_rate", f"{qc_summary['pass_rate']:.1f}%"])
    writer.writerow(["QC 요약", "min_valid_count", qc_summary["min_valid_count"]])
    writer.writerow(["QC 요약", "valid", qc_summary["valid"]])
    for reason, count in qc_summary["reason_counts"].items():
        writer.writerow(["QC 실패 사유", reason, count])
    writer.writerow(["검사 결과", "screening_index", prediction["screening_index"]])
    writer.writerow(["검사 결과", "SSI", ssi])
    writer.writerow(["검사 결과", "majority_vote", prediction["majority_vote"]])
    writer.writerow(["검사 결과", "final_decision", final_decision])
    writer.writerow(["검사 결과", "risk_level", risk_info["level"]])
    writer.writerow(["검사 결과", "risk_band_range", risk_info["range_label"]])
    writer.writerow(["검사 결과", "risk_observed_cancer_rate", f"{risk_info['observed_cancer_rate']:.1f}%"])
    writer.writerow(["검사 결과", "risk_band_n", risk_info["n"]])
    writer.writerow(["검사 결과", "risk_band_note", strings["risk_band_note"]])
    writer.writerow(["점수 산출", "ssi_calculation", strings["ssi_calculation_text"]])
    if patient_decision.get("model_probability_mean") is not None:
        writer.writerow(["점수 산출", "model_mean_probability", patient_decision.get("model_probability_mean")])
        writer.writerow(["점수 산출", "model_probability_threshold", patient_decision.get("model_probability_threshold")])
        writer.writerow(["점수 산출", "ssi_threshold", patient_decision.get("ssi_threshold", SSI_DECISION_CUTOFF)])
    if type_confidence["top"]:
        writer.writerow(["암종별 확률", "top_type", type_confidence["top"]["code"]])
        writer.writerow(["암종별 확률", "top_type_confidence", type_confidence["level"]])
        writer.writerow(["암종별 확률", "top_second_gap", f"{type_confidence['gap']:.4f}"])
        writer.writerow(["점수 산출", "type_score_calculation", strings["type_score_calculation_text"]])
        writer.writerow(["점수 산출", "top_type_probability", f"{type_confidence['top']['prob']:.4f}"])
        writer.writerow(["점수 산출", "type_score", f"{type_confidence['top']['prob'] * 10:.1f}/10.0"])
        if type_confidence["level"] == "high":
            writer.writerow([
                "암종별 확률",
                "high_classification_guide",
                strings["high_classification_guide"],
            ])
    for ct in cancer_types_sorted:
        code = ct["code"]
        prob = ct["prob"]
        writer.writerow(["암종별 확률", code, prob])
    if not qc_summary["valid"]:
        writer.writerow(["권고", "recommendation", "본 검사는 최종 판정에 사용하지 않으며, 재검을 권고합니다."])
        writer.writerow(["권고", "recommended_retest_type", "검체/기판 재준비 후 재측정"])

    content = "\ufeff" + buffer.getvalue()
    return Response(
        content=content.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="SERS_Report_{session_id[:8]}.csv"'},
    )


@app.get("/manual")
async def manual_download():
    """Download the current software IFU/manual if available."""
    for path in MANUAL_CANDIDATES:
        if path.exists():
            filename = path.name
            media_type = "application/pdf" if path.suffix.lower() == ".pdf" else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            return FileResponse(
                path,
                media_type=media_type,
                filename=filename,
                content_disposition_type="inline",
            )
    return HTMLResponse(
        "<h2>사용설명서 파일을 찾을 수 없습니다.</h2>"
        "<p>배포 전 software_ifu.pdf를 scripts/deployment/manuals/에 배치하세요.</p>",
        status_code=404,
    )


# --- Audit Log ---

@app.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    redirect = auth.require_role(request, ["admin"])
    if redirect:
        return redirect

    logs = db.get_audit_logs(limit=200)
    ctx = template_context(request, logs=logs)
    return templates.TemplateResponse("audit_log.html", ctx)


# --- API Endpoints ---

@app.get("/health")
async def health():
    pred = get_predictor()
    return {
        "status": "healthy",
        "model_loaded": pred is not None,
        "model_name": MODEL_DISPLAY_NAME,
        "cancer_types": pred.cancer_types if pred else [],
        "decision_profile": DECISION_PROFILE,
        "ssi_threshold": SSI_DECISION_CUTOFF,
        "model_probability_threshold": pred.threshold if pred else None,
        "required_spectra": REQUIRED_SPECTRA_COUNT,
        "min_qc_pass_count": MIN_QC_PASS_COUNT,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/model-info")
async def model_info():
    pred = get_predictor()
    return pred.manifest if pred else {"error": "Model not loaded"}


# --- Main ---

def main():
    import uvicorn

    parser = argparse.ArgumentParser(description="SERS Clinical Webapp")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info(f"Starting SERS Clinical Webapp on {args.host}:{args.port}")

    uvicorn.run(
        "scripts.deployment.sers_clinical_webapp:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
