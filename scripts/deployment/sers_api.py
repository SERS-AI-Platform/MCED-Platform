"""
SERS Cancer Screening — REST API Server

Endpoints:
    POST /predict          Upload spectrum CSV, get cancer prediction
    POST /predict/batch    Upload multiple spectra
    GET  /health           Health check
    GET  /model-info       Model metadata

Usage:
    uvicorn scripts.deployment.sers_api:app --host 0.0.0.0 --port 8000
    # or via Docker:
    docker compose -f infra/docker-compose.yml up sers-api
"""

from __future__ import annotations

import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.deployment.sers_predict import ProductionPredictor

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sers-api")

# Load predictor at startup
try:
    predictor = ProductionPredictor()
    logger.info(f"Model loaded: {len(predictor.cancer_types)} cancer types, "
                f"grid={len(predictor.grid)} points")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    predictor = None

app = FastAPI(
    title="SERS Cancer Screening API",
    description="Predict cancer from SERS urine spectra",
    version="1.0.0",
)


@app.get("/health")
def health():
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "cancer_types": predictor.cancer_types,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model-info")
def model_info():
    if predictor is None:
        raise HTTPException(503, "Model not loaded")
    return predictor.manifest


@app.post("/predict")
async def predict(
    file: UploadFile = File(..., description="Spectrum CSV file"),
    age: float | None = Query(None, description="Patient age"),
    sex: str | None = Query(None, description="Patient sex (M/F)"),
    bmi: float | None = Query(None, description="Patient BMI"),
):
    """Predict cancer from a single SERS spectrum."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = predictor.predict_single(tmp_path, age=age, sex=sex, bmi=bmi)
        result["original_filename"] = file.filename
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(..., description="Spectrum CSV files"),
    age: float | None = Query(None),
    sex: str | None = Query(None),
    bmi: float | None = Query(None),
):
    """Predict cancer from multiple SERS spectra (same patient)."""
    if predictor is None:
        raise HTTPException(503, "Model not loaded")

    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            result = predictor.predict_single(tmp_path, age=age, sex=sex, bmi=bmi)
            result["original_filename"] = file.filename
            results.append(result)
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "message": str(e)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    ok = [r for r in results if r.get("status") == "ok"]
    import numpy as np
    summary = {}
    if ok:
        n_detected = sum(1 for r in ok if r["cancer_detected"])
        summary = {
            "total": len(ok),
            "cancer_detected": n_detected,
            "non_cancer": len(ok) - n_detected,
            "mean_cancer_probability": round(float(np.mean([r["cancer_probability"] for r in ok])), 4),
        }

    return JSONResponse(content={
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "n_files": len(results),
        "results": results,
        "summary": summary,
    })
