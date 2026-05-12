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
import sys
import tempfile
import argparse
import logging
from pathlib import Path
from datetime import datetime

from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, RedirectResponse, Response, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment import clinical_db as db
from scripts.deployment import clinical_auth as auth
from scripts.deployment import clinical_report as report
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
STANDARD_DECISION_PROFILE = "balanced"


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


def get_predictor() -> ProductionPredictor:
    global predictor
    if predictor is None:
        # Prefer stacking model if available, fall back to LR
        stacking_dir = PROJECT_ROOT / "models" / "production_stacking"
        if stacking_dir.exists():
            predictor = StackingPredictor(stacking_dir)
            logger.info(f"Stacking V2 model loaded: {predictor.cancer_types}, "
                        f"standard decision profile: {STANDARD_DECISION_PROFILE}")
        else:
            predictor = ProductionPredictor()
            logger.info(f"LR model loaded: {predictor.cancer_types}, "
                        f"standard decision profile: {STANDARD_DECISION_PROFILE}")
    return predictor


def template_context(request: Request, **kwargs) -> dict:
    """Build common template context."""
    user = auth.get_current_user(request)
    lang = get_lang(request)
    s = get_strings(request)
    return {
        "request": request,
        "user": user,
        "lang": lang,
        "s": s,
        "decision_profile": STANDARD_DECISION_PROFILE,
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
    ctx = template_context(request, error=False)
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

    ctx = template_context(request, error=False, success=True)
    return templates.TemplateResponse("register.html", ctx)


@app.post("/login")
async def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    token = auth.login(username, password)
    if not token:
        ctx = template_context(request, error=True)
        return templates.TemplateResponse("login.html", ctx)

    response = RedirectResponse("/patient/new", status_code=303)
    response.set_cookie(auth.SESSION_COOKIE, token, httponly=True, max_age=8 * 3600)
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
    threshold = pred.operating_modes[STANDARD_DECISION_PROFILE]["threshold"]

    session_id = db.create_session(
        patient_id=patient_id,
        age=age,
        sex=sex,
        bmi=bmi,
        operating_mode=STANDARD_DECISION_PROFILE,
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

    # Save uploaded files to temp directory and record in DB
    temp_dir = tempfile.mkdtemp(prefix="sers_clinical_")
    filepaths = []
    spectrum_ids = []

    for f in files:
        if not f.filename.lower().endswith(".csv"):
            continue
        content = await f.read()
        temp_path = Path(temp_dir) / f.filename
        temp_path.write_bytes(content)
        filepaths.append(temp_path)
        spec_id = db.add_spectrum(session_id, f.filename)
        spectrum_ids.append((spec_id, f.filename))

    if not filepaths:
        return JSONResponse({"error": "No valid CSV files"}, 400)

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
        mode=STANDARD_DECISION_PROFILE,
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

    # Save prediction if status is OK
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
            "cancer_detected": result.get("patient_decision", {}).get("cancer_detected"),
            "ssi_score": result.get("patient_decision", {}).get("ssi_score"),
            "model_probability_mean": result.get("patient_decision", {}).get("model_probability_mean"),
        },
    )

    # Redirect to QC page
    return RedirectResponse(f"/patient/{session_id}/qc", status_code=303)


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
    qc_passed = sum(1 for sp in spectra if sp.get("qc_pass"))
    qc_total = len(spectra)

    ctx = template_context(
        request,
        session=session,
        spectra=spectra,
        qc_passed=qc_passed,
        qc_total=qc_total,
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
    qc_passed = sum(1 for sp in spectra if sp.get("qc_pass"))

    ctx = template_context(
        request,
        session=session,
        prediction=prediction,
        cancer_types_sorted=cancer_types_sorted,
        per_replicate=per_replicate,
        spectra=spectra,
        qc_passed=qc_passed,
        qc_total=len(spectra),
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
        "cancer_types": pred.cancer_types if pred else [],
        "decision_profile": STANDARD_DECISION_PROFILE,
        "ssi_threshold": SSI_DECISION_CUTOFF,
        "model_probability_threshold": pred.operating_modes[STANDARD_DECISION_PROFILE]["threshold"] if pred else None,
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
