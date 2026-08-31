#!/usr/bin/env python3
"""
# noqa: SIZE_OK — legacy FastAPI deployment entry point retained for packaging compatibility.
AECD Software — IEC 62366 Compliant Clinical Interface

임상용 사용적합성 테스트 대응 웹앱. 환자 중심 step-by-step 워크플로우.

Usage:
    python scripts/deployment/sers_clinical_webapp.py [--port 8080]

Requires:
    pip install fastapi uvicorn jinja2 python-multipart
"""

from __future__ import annotations

import argparse
import csv
import io
import logging
import sys
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment import clinical_access as access
from scripts.deployment import clinical_auth as auth
from scripts.deployment import clinical_db as db
from scripts.deployment import clinical_decision as decision_utils
from scripts.deployment import clinical_qc as qc_utils
from scripts.deployment import clinical_report as report
from scripts.deployment import clinical_result_data as result_data
from scripts.deployment import clinical_validation as validation
from scripts.deployment.clinical_i18n import get_lang, get_strings
from scripts.deployment.sers_predict import (
    SSI_DECISION_CUTOFF,
    ProductionPredictor,
    StackingPredictor,
    probability_to_ssi,
)

logger = logging.getLogger(__name__)

# --- App Setup ---
app = FastAPI(title="AECD Software", version="1.0")

BASE_DIR = Path(__file__).parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Global predictor (loaded once)
predictor: ProductionPredictor | None = None
MODEL_DISPLAY_NAME = "uSERS-Net Ver1"
DECISION_PROFILE = "standard_balanced"
DB_OPERATING_MODE = "balanced"


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


def _spectrum_qc_update(
    filename: str, result: dict
) -> tuple[bool, list[str], float | None, float | None]:
    qc_summary = result.get("qc_summary", {})
    failure = next(
        (item for item in qc_summary.get("failures", []) if item["file"] == filename),
        None,
    )
    if failure:
        return False, list(failure["flags"]), None, None

    passed = next(
        (item for item in qc_summary.get("passes", []) if item["file"] == filename),
        None,
    )
    replicate = next(
        (item for item in result.get("per_replicate", []) if item["file"] == filename),
        None,
    )
    evaluated = passed or replicate
    if evaluated:
        return (
            True,
            [],
            evaluated.get("replicate_correlation"),
            evaluated.get("fp_mean"),
        )
    return False, ["read_error"], None, None


# Usability-test clinical workflow constants.
REQUIRED_SPECTRA_COUNT = 5
MIN_QC_PASS_COUNT = qc_utils.DEFAULT_MIN_VALID_COUNT
ALLOWED_SPECTRUM_EXTENSIONS = {".csv", ".txt"}

MANUAL_PATH = BASE_DIR / "manuals" / "software_ifu.pdf"


def get_predictor() -> ProductionPredictor:
    global predictor
    if predictor is None:
        # Prefer stacking model if available, fall back to LR
        stacking_dir = PROJECT_ROOT / "artifacts" / "usersnet" / "current"
        if stacking_dir.exists():
            predictor = StackingPredictor(stacking_dir)
            logger.info(
                f"Stacking V2 model loaded: {predictor.cancer_types}, "
                f"standard decision profile: {DECISION_PROFILE}"
            )
        else:
            predictor = ProductionPredictor()
            logger.info(
                f"LR model loaded: {predictor.cancer_types}, "
                f"standard decision profile: {DECISION_PROFILE}"
            )
    return predictor


def template_context(request: Request, **kwargs: Any) -> Mapping[str, Any]:
    """Build common template context."""
    user = auth.get_current_user(request)
    lang = get_lang(request)
    s = get_strings(request)
    auto_logout_message = s["auto_logout_notice"].format(minutes=auth.SESSION_EXPIRY_MINUTES)
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


def _authorized_session(request: Request, session_id: str) -> dict | None:
    user = auth.get_current_user(request)
    if user is None:
        return None
    return access.get_authorized_session(user, session_id)


def _session_not_found(request: Request) -> HTMLResponse:
    return HTMLResponse(f"<h2>{get_strings(request)['session_not_found']}</h2>", status_code=404)


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
    ctx = template_context(
        request, error=False, registered=request.query_params.get("registered") == "1"
    )
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
    password: str = Form(...),
    password_confirm: str = Form(...),
):
    s = get_strings(request)
    role = "clinician"

    # Validation
    if not all([username.strip(), display_name.strip(), password]):
        ctx = template_context(
            request,
            error=True,
            success=False,
            error_message=s["register_error_fields"],
            form_username=username,
            form_display_name=display_name,
        )
        return templates.TemplateResponse("register.html", ctx)

    if password != password_confirm:
        ctx = template_context(
            request,
            error=True,
            success=False,
            error_message=s["register_error_password"],
            form_username=username,
            form_display_name=display_name,
        )
        return templates.TemplateResponse("register.html", ctx)

    # Check duplicate
    existing = db.get_user_by_username(username.strip())
    if existing:
        ctx = template_context(
            request,
            error=True,
            success=False,
            error_message=s["register_error_exists"],
            form_username=username,
            form_display_name=display_name,
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
    response.set_cookie(
        auth.SESSION_COOKIE, token, httponly=True, max_age=auth.SESSION_EXPIRY_MINUTES * 60
    )
    return response


@app.get("/logout")
async def logout(request: Request):
    token = auth.get_session_token(request)
    if token:
        auth.logout(token)
    response = RedirectResponse("/login", status_code=303)
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


# --- Account Management ---


@app.get("/account", response_class=HTMLResponse)
async def account_page(request: Request):
    redirect = auth.require_role(request, ["admin"])
    if redirect:
        return redirect

    managed_users = [
        user for user in db.get_active_users() if user["role"] != "admin"
    ]
    ctx = template_context(
        request,
        managed_users=managed_users,
        reset_success=request.query_params.get("reset") == "1",
        error_message=None,
        selected_user_id=None,
    )
    return templates.TemplateResponse("account.html", ctx)


@app.post("/account/password-reset", response_class=HTMLResponse)
async def account_password_reset(
    request: Request,
    target_user_id: int = Form(...),
    temporary_password: str = Form(...),
    password_confirm: str = Form(...),
):
    redirect = auth.require_role(request, ["admin"])
    if redirect:
        return redirect

    administrator = auth.get_current_user(request)
    target_user = db.get_user_by_id(target_user_id)
    managed_users = [
        user for user in db.get_active_users() if user["role"] != "admin"
    ]
    strings = get_strings(request)
    error_message = None
    if (
        target_user is None
        or target_user["role"] == "admin"
        or not target_user["is_active"]
    ):
        error_message = strings["account_reset_invalid_target"]
    elif temporary_password != password_confirm:
        error_message = strings["account_reset_password_mismatch"]
    elif len(temporary_password) < 8 or not temporary_password.strip():
        error_message = strings["account_password_requirement"]

    if error_message:
        ctx = template_context(
            request,
            managed_users=managed_users,
            reset_success=False,
            error_message=error_message,
            selected_user_id=target_user_id,
        )
        return templates.TemplateResponse("account.html", ctx, status_code=400)

    password_hash = auth.hash_password(temporary_password)
    if not db.update_user_password(target_user_id, password_hash):
        ctx = template_context(
            request,
            managed_users=managed_users,
            reset_success=False,
            error_message=strings["account_reset_invalid_target"],
            selected_user_id=target_user_id,
        )
        return templates.TemplateResponse("account.html", ctx, status_code=400)

    auth.invalidate_user_sessions(target_user_id)
    db.log_audit(
        "password_reset",
        user_id=administrator["user_id"],
        detail={
            "target_user_id": target_user_id,
            "target_username": target_user["username"],
        },
    )
    return RedirectResponse("/account?reset=1", status_code=303)


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


@app.get("/history", response_class=HTMLResponse)
async def history_page(
    request: Request,
    patient_id: str = "",
    test_date: str = "",
    qc_status: str = "",
    result_status: str = "",
    owner_user_id: str = "",
    page: int = 1,
):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect
    user = auth.get_current_user(request)
    is_admin = user["role"] == "admin"
    selected_owner_id = int(owner_user_id) if is_admin and owner_user_id.isdecimal() else None
    filters = db.HistoryFilters(
        patient_id=patient_id.strip(),
        test_date=test_date,
        qc_status=qc_status,
        result_status=result_status,
        owner_user_id=selected_owner_id,
        page=max(page, 1),
    )
    rows, total = db.get_session_history(
        user_id=user["user_id"], is_admin=is_admin, filters=filters
    )
    ctx = template_context(
        request,
        rows=rows,
        total=total,
        pages=max(1, (total + 49) // 50),
        filters=filters,
        is_admin=is_admin,
        owners=db.get_active_users() if is_admin else [],
    )
    return templates.TemplateResponse("history.html", ctx)


# --- Patient Registration ---


@app.get("/patient/new", response_class=HTMLResponse)
async def patient_page(request: Request):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    ctx = template_context(
        request,
        show_stepper=True,
        current_step="patient",
        completed_steps=[],
        error_message=None,
        form_values={},
    )
    return templates.TemplateResponse("patient_register.html", ctx)


@app.post("/patient/new")
async def patient_submit(
    request: Request,
    patient_id: str = Form(...),
    age: float = Form(...),
    sex: str = Form(...),
    bmi: str = Form(...),
):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    try:
        parsed_patient_id = validation.parse_patient_id(patient_id)
        parsed_bmi = validation.parse_bmi(bmi)
    except validation.ClinicalInputError as error:
        strings = get_strings(request)
        error_key = (
            "patient_id_format_error" if error.field == "patient_id" else "bmi_validation_error"
        )
        ctx = template_context(
            request,
            show_stepper=True,
            current_step="patient",
            completed_steps=[],
            error_message=strings[error_key],
            form_values={
                "patient_id": patient_id,
                "age": age,
                "sex": sex,
                "bmi": bmi,
            },
        )
        return templates.TemplateResponse("patient_register.html", ctx, status_code=400)
    pred = get_predictor()
    threshold = pred.threshold

    session_id = db.create_session(
        patient_id=parsed_patient_id,
        age=age,
        sex=sex,
        bmi=parsed_bmi,
        operating_mode=DB_OPERATING_MODE,
        created_by=user["user_id"],
        threshold=threshold,
    )

    db.log_audit(
        "patient_create",
        user_id=user["user_id"],
        session_id=session_id,
        detail={
            "patient_id": parsed_patient_id,
            "age": age,
            "sex": sex,
            "bmi": parsed_bmi,
        },
    )

    return RedirectResponse(f"/patient/{session_id}/upload", status_code=303)


# --- Spectrum Upload & Analysis ---


@app.get("/patient/{session_id}/upload", response_class=HTMLResponse)
async def upload_page(request: Request, session_id: str):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)

    ctx = template_context(
        request,
        session=session,
        show_stepper=True,
        current_step="upload",
        completed_steps=["patient"],
        remeasure=(
            request.query_params.get("retest") == "1"
            or request.query_params.get("remeasure") == "1"
        ),
    )
    return templates.TemplateResponse("spectrum_upload.html", ctx)


@app.post("/patient/{session_id}/upload")
async def upload_submit(request: Request, session_id: str, files: list[UploadFile] = File(...)):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)

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
    upload_names = [filename.casefold() for filename, _ in valid_uploads]
    if len(upload_names) != len(set(upload_names)):
        return JSONResponse(
            {"error": "Each uploaded spectrum must have a unique filename."},
            400,
        )
    pred = get_predictor()

    if db.get_spectra_for_session(session_id):
        return JSONResponse(
            {
                "error": (
                    "This test already contains spectra. Start a linked retest "
                    "from its QC or history page."
                )
            },
            409,
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

    for spec_id, filename in spectrum_ids:
        qc_pass, qc_flags, correlation, fp_mean = _spectrum_qc_update(filename, result)
        db.update_spectrum_qc(
            spec_id,
            qc_pass=qc_pass,
            qc_flags=qc_flags,
            correlation=correlation,
            fp_mean=fp_mean,
        )

    passed_count = result.get("qc_summary", {}).get("passed", 0)
    persisted_qc = qc_utils.build_qc_summary(
        db.get_spectra_for_session(session_id), MIN_QC_PASS_COUNT
    )
    db.update_session_status(session_id, "qc_done", qc_valid=persisted_qc["valid"])

    # Save available prediction output. The results/report pages decide whether
    # it is usable for interpretation based on the clinical QC validity rule.
    if result.get("status") == "ok":
        prediction_data = result_data.build_persisted_prediction(result)
        db.save_prediction(session_id, prediction_data)
        db.update_session_status(
            session_id,
            "analyzed",
            model_variant=result.get("model_variant"),
        )

    audit_detail = {
        "status": result.get("status"),
        "qc_passed": passed_count,
        "qc_required": MIN_QC_PASS_COUNT,
    }
    if result.get("status") == "ok":
        audit_detail.update(
            {
                "cancer_detected": result.get("patient_decision", {}).get("cancer_detected"),
                "ssi_score": result.get("patient_decision", {}).get("ssi_score"),
                "model_probability_mean": result.get("patient_decision", {}).get(
                    "model_probability_mean"
                ),
            }
        )
    db.log_audit(
        "analysis_complete",
        user_id=user["user_id"],
        session_id=session_id,
        detail=audit_detail,
    )

    # Redirect to QC page
    return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)


@app.post("/patient/{session_id}/remeasure")
@app.post("/patient/{session_id}/retest")
async def remeasure_patient(
    request: Request,
    session_id: str,
    bmi: Annotated[str | None, Form()] = None,
):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)

    bmi_override = None
    if session["bmi"] is None:
        if bmi is None:
            ctx = template_context(request, session=session, error_message=None)
            return templates.TemplateResponse("retest_bmi.html", ctx, status_code=422)
        try:
            bmi_override = validation.parse_bmi(bmi)
        except validation.ClinicalInputError:
            ctx = template_context(
                request,
                session=session,
                error_message=get_strings(request)["bmi_validation_error"],
            )
            return templates.TemplateResponse("retest_bmi.html", ctx, status_code=400)
    try:
        retest_id = db.create_retest_session(
            session_id,
            performed_by=user["user_id"],
            bmi_override=bmi_override,
        )
    except db.RetestNotAllowedError:
        return _session_not_found(request)
    db.log_audit(
        "retest_create",
        user_id=user["user_id"],
        session_id=retest_id,
        detail={
            "patient_id": session["patient_id"],
            "retest_of_session_id": session_id,
            "owner_user_id": session["owner_user_id"],
            "performed_by": user["user_id"],
        },
    )
    return RedirectResponse(f"/patient/{retest_id}/upload?retest=1", status_code=303)


# --- QC Review ---


@app.get("/patient/{session_id}/qc", response_class=HTMLResponse)
async def qc_page(request: Request, session_id: str):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)

    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)
    strings = get_strings(request)
    qc_summary_label = strings["qc_summary"].format(
        passed=qc_summary["passed"],
        total=qc_summary["total"],
    )
    if qc_summary["has_critical_failure"]:
        qc_status_template = strings["qc_critical_failure_message"]
    elif qc_summary["valid"]:
        qc_status_template = strings["qc_valid_message"]
    else:
        qc_status_template = strings["qc_invalid_message"]
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

    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)

    prediction_row = db.get_prediction(session_id)
    if not prediction_row:
        return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)

    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)
    if not qc_summary["valid"]:
        return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)

    # Flatten prediction for template
    result_json = prediction_row["result_json"]
    patient_decision = result_json.get("patient_decision", {})
    cancer_signal_count, qc_valid_count = result_data.get_signal_counts(result_json)
    ssi_score = get_patient_ssi_score(patient_decision, session.get("threshold"))

    prediction = {
        "cancer_detected": patient_decision.get("cancer_detected", False),
        "screening_index": ssi_score,
        "ssi_score": ssi_score,
        "ssi_threshold": patient_decision.get("ssi_threshold", SSI_DECISION_CUTOFF),
        "model_probability_mean": patient_decision.get("model_probability_mean"),
        "model_probability_threshold": patient_decision.get("model_probability_threshold"),
        "decision_level": patient_decision.get("decision_level")
        or prediction_row.get("decision_level"),
        "decision_policy": patient_decision.get("decision_policy")
        or prediction_row.get("decision_policy"),
        "majority_vote": patient_decision.get("majority_vote"),
        "cancer_signal_spectra_count": cancer_signal_count,
        "qc_valid_spectra_count": qc_valid_count,
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
        is_legacy_decision=prediction["decision_policy"]
        != decision_utils.MEAN_SSI_DECISION_POLICY,
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
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)
    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)
    if qc_summary["valid"] and not db.get_prediction(session_id):
        return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)

    ctx = template_context(
        request,
        session=session,
        show_stepper=True,
        current_step="report",
        completed_steps=["patient", "upload", "qc", "results"],
    )
    return templates.TemplateResponse("report_ready.html", ctx)


@app.get("/patient/{session_id}/report.pdf")
async def report_pdf(request: Request, session_id: str):
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    lang = get_lang(request)
    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)
    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)
    if qc_summary["valid"] and not db.get_prediction(session_id):
        return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)

    try:
        report_id, pdf_bytes = report.create_and_save_report(
            session=session,
            user_id=user["user_id"],
            lang=lang,
        )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as e:
        logger.error(f"Report generation failed: {e}")
        return HTMLResponse(f"<h2>Report generation error: {e}</h2>", status_code=500)

    db.log_audit(
        "report_download",
        user_id=user["user_id"],
        session_id=session_id,
        detail={"report_id": report_id, "format": "pdf"},
    )
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="SERS_Report_{report_id}.pdf"'},
    )


@app.get("/patient/{session_id}/report.csv")
async def report_csv(request: Request, session_id: str):
    """Download a CSV summary of the current report result."""
    redirect = auth.require_auth(request)
    if redirect:
        return redirect

    user = auth.get_current_user(request)
    session = _authorized_session(request, session_id)
    if not session:
        return _session_not_found(request)

    prediction_row = db.get_prediction(session_id)
    spectra = db.get_spectra_for_session(session_id)
    qc_summary = qc_utils.build_qc_summary(spectra, MIN_QC_PASS_COUNT)
    if qc_summary["valid"] and not prediction_row:
        return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)
    pred = prediction_row["result_json"] if prediction_row else {}
    patient_decision = pred.get("patient_decision", {})
    cancer_signal_count, qc_valid_count = result_data.get_signal_counts(pred)
    ssi_score = get_patient_ssi_score(patient_decision, session.get("threshold"))
    prediction = {
        "cancer_detected": patient_decision.get("cancer_detected", False),
        "screening_index": ssi_score,
        "ssi_score": ssi_score,
        "decision_level": patient_decision.get("decision_level")
        or (prediction_row.get("decision_level") if prediction_row else None),
        "decision_policy": patient_decision.get("decision_policy")
        or (prediction_row.get("decision_policy") if prediction_row else None),
        "majority_vote": patient_decision.get("majority_vote"),
        "cancer_signal_spectra_count": cancer_signal_count,
        "qc_valid_spectra_count": qc_valid_count,
        "cancer_type_probabilities": pred.get("cancer_type_probabilities", {}) or {},
    }

    ssi = get_patient_ssi_score(prediction, session.get("threshold"))
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
    final_decision = decision_utils.final_decision(prediction, qc_summary["valid"])
    displayed_decision = strings[f"{final_decision}_decision"]
    patient_section = strings["patient_info"]
    qc_section = strings["qc_summary_label"]
    result_section = strings["test_result"]
    score_section = strings["score_interpretation"]
    writer.writerow(["section", "item", "value"])
    writer.writerow([patient_section, "patient_id", session["patient_id"]])
    writer.writerow([patient_section, "age", session["age"]])
    writer.writerow([patient_section, "sex", session["sex"]])
    writer.writerow([patient_section, "bmi", session["bmi"]])
    writer.writerow([patient_section, "test_id", session["test_id"]])
    writer.writerow([patient_section, "model", MODEL_DISPLAY_NAME])
    writer.writerow([qc_section, "total", qc_summary["total"]])
    writer.writerow([qc_section, "passed", qc_summary["passed"]])
    writer.writerow([qc_section, "failed", qc_summary["failed"]])
    writer.writerow([qc_section, "pass_rate", f"{qc_summary['pass_rate']:.1f}%"])
    writer.writerow([qc_section, "min_valid_count", qc_summary["min_valid_count"]])
    writer.writerow([qc_section, "valid", qc_summary["valid"]])
    for reason, count in qc_summary["reason_counts"].items():
        writer.writerow([strings["qc_failure_reasons"], reason, count])
    if qc_summary["valid"]:
        writer.writerow([result_section, strings["screening_index"], f"{ssi:.2f}/10.0"])
        if prediction["cancer_signal_spectra_count"] is not None:
            writer.writerow(
                [
                    result_section,
                    strings["majority_vote_label"],
                    prediction["cancer_signal_spectra_count"],
                ]
            )
            writer.writerow(
                [
                    result_section,
                    strings["qc_valid_spectra_label"],
                    prediction["qc_valid_spectra_count"],
                ]
            )
        elif prediction["majority_vote"] is not None:
            writer.writerow(
                [
                    result_section,
                    strings["majority_vote_label"],
                    prediction["majority_vote"],
                ]
            )
        writer.writerow([result_section, strings["final_decision"], displayed_decision])
        writer.writerow(
            [
                result_section,
                strings["decision_policy_label"],
                (
                    strings["legacy_decision_policy"]
                    if prediction["decision_policy"]
                    != decision_utils.MEAN_SSI_DECISION_POLICY
                    else strings["mean_ssi_decision_policy"]
                ),
            ]
        )
        is_legacy_decision = (
            prediction["decision_policy"] != decision_utils.MEAN_SSI_DECISION_POLICY
        )
        if not is_legacy_decision:
            writer.writerow(
                [
                    result_section,
                    strings["risk_level"],
                    strings[f"risk_{risk_info['level'].lower()}"],
                ]
            )
            writer.writerow(
                [result_section, strings["risk_band_range"], risk_info["range_label"]]
            )
            writer.writerow(
                [
                    result_section,
                    "risk_band_interpretation",
                    strings[f"risk_{risk_info['level'].lower()}_interpretation"],
                ]
            )
        writer.writerow(
            [
                result_section,
                strings["risk_band_note"],
                (
                    strings["legacy_ssi_calculation_text"]
                    if is_legacy_decision
                    else strings["risk_band_note"]
                ),
            ]
        )
        writer.writerow(
            [
                score_section,
                strings["ssi_calculation_title"],
                (
                    strings["legacy_ssi_calculation_text"]
                    if is_legacy_decision
                    else strings["ssi_calculation_text"]
                ),
            ]
        )
        writer.writerow(
            [
                strings["recommendation"],
                "recommendation",
                (
                    strings["legacy_recommendation"]
                    if is_legacy_decision
                    else strings[f"valid_recommendation_{final_decision}"]
                ),
            ]
        )
        if final_decision == "positive" and type_confidence["top"]:
            pattern_section = strings["type_probabilities"]
            writer.writerow(
                [
                    pattern_section,
                    strings["estimated_type"],
                    (
                        f"{strings['cancer_' + type_confidence['top']['code']]} "
                        f"({type_confidence['top']['code']})"
                    ),
                ]
            )
            writer.writerow(
                [
                    pattern_section,
                    strings["confidence"],
                    strings[f"confidence_{type_confidence['level']}"],
                ]
            )
            writer.writerow(
                [
                    pattern_section,
                    strings["probability_gap"],
                    f"{type_confidence['gap'] * 100:.1f}{strings['percentage_point_unit']}",
                ]
            )
            writer.writerow(
                [
                    score_section,
                    strings["type_score_calculation_title"],
                    strings["type_score_calculation_text"],
                ]
            )
            writer.writerow(
                [
                    pattern_section,
                    strings["type_score"],
                    f"{type_confidence['top']['prob'] * 10:.1f}/10.0",
                ]
            )
            if type_confidence["level"] == "high":
                writer.writerow(
                    [
                        pattern_section,
                        strings["confidence_high"],
                        strings["high_classification_guide"],
                    ]
                )
        for ct in cancer_types_sorted:
            code = ct["code"]
            prob = ct["prob"]
            writer.writerow(
                [
                    strings["type_probabilities"],
                    f"{strings['cancer_' + code]} ({code})",
                    f"{prob * 100:.1f}%",
                ]
            )
    else:
        writer.writerow(
            [strings["recommendation"], "recommendation", strings["invalid_recommendation"]]
        )
        writer.writerow(
            [
                strings["recommendation"],
                strings["recommended_retest_type"],
                strings["retest_specimen_substrate"],
            ]
        )

    report_id = db.create_report(session_id, user["user_id"])
    db.log_audit(
        "report_download",
        user_id=user["user_id"],
        session_id=session_id,
        detail={"report_id": report_id, "format": "csv"},
    )
    content = "\ufeff" + buffer.getvalue()
    return Response(
        content=content.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="SERS_Report_{report_id}.csv"'},
    )


@app.get("/manual")
async def manual_download():
    """Download the current software IFU/manual if available."""
    if MANUAL_PATH.exists():
        return FileResponse(
            MANUAL_PATH,
            media_type="application/pdf",
            filename="software_ifu.pdf",
            content_disposition_type="attachment",
        )
    return HTMLResponse(
        "<h2>사용설명서 파일을 찾을 수 없습니다.</h2>"
        "<p>관리자에게 software_ifu.pdf 설치 상태를 확인해 달라고 요청하세요.</p>",
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

    parser = argparse.ArgumentParser(description="AECD Software")
    parser.add_argument("--host", default="0.0.0.0", help="Host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="Port (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logger.info(f"Starting AECD Software on {args.host}:{args.port}")

    uvicorn.run(
        "scripts.deployment.sers_clinical_webapp:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
