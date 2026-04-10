# SERS Cancer Screening API Reference

## Overview

Two server applications are provided:

| Application | File | Purpose |
|---|---|---|
| REST API | `scripts/sers_api.py` | Headless JSON API for programmatic access |
| Web App | `scripts/sers_webapp.py` | Browser UI + API endpoints for interactive use |

Both use **FastAPI** and load the `ProductionPredictor` model at startup.

---

## REST API (`sers_api.py`)

### Start

```bash
uvicorn scripts.sers_api:app --host 0.0.0.0 --port 8000
# or via Docker:
docker compose -f infra/docker-compose.yml up sers-api
```

### Authentication

None. The API is designed for LAN access only. No API keys or tokens are required.

### Endpoints

#### `GET /health`

Health check and model status.

**Response** `200 OK`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "cancer_types": ["CRC", "GC", "HCC", "OVA", "PRO"],
  "timestamp": "2026-03-24T10:30:00.000000"
}
```

**Error** `503 Service Unavailable` — model failed to load.

---

#### `GET /model-info`

Returns the full model manifest (version, training config, performance metrics).

**Response** `200 OK` — contents of the model's `manifest.json`.

**Error** `503 Service Unavailable` — model not loaded.

---

#### `POST /predict`

Predict cancer from a single SERS spectrum.

**Parameters**

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `file` | body (multipart) | file | Yes | Spectrum CSV (wavenumber, intensity) |
| `age` | query | float | No | Patient age |
| `sex` | query | string | No | Patient sex (`M` or `F`) |
| `bmi` | query | float | No | Patient BMI |

**Example Request**
```bash
curl -X POST "http://localhost:8000/predict?age=55&sex=M" \
  -F "file=@spectrum_001.csv"
```

**Response** `200 OK`
```json
{
  "status": "ok",
  "cancer_detected": true,
  "cancer_probability": 0.873,
  "cancer_type_prediction": "CRC",
  "cancer_type_confidence": 0.65,
  "cancer_type_probabilities": {
    "CRC": 0.65,
    "GC": 0.12,
    "HCC": 0.10,
    "OVA": 0.08,
    "PRO": 0.05
  },
  "model_variant": "fusion",
  "warnings": [],
  "original_filename": "spectrum_001.csv"
}
```

If clinical data (age/sex/bmi) is provided, `model_variant` will be `"fusion"` (SERS + clinical features). Otherwise it will be `"sers"` (SERS only).

**Errors**
- `500` — prediction failed (invalid file format, processing error)
- `503` — model not loaded

---

#### `POST /predict/batch`

Predict cancer from multiple SERS spectra (typically replicates from the same patient).

**Parameters**

| Name | In | Type | Required | Description |
|---|---|---|---|---|
| `files` | body (multipart) | file[] | Yes | Multiple spectrum CSV files |
| `age` | query | float | No | Patient age |
| `sex` | query | string | No | Patient sex (`M` or `F`) |
| `bmi` | query | float | No | Patient BMI |

**Example Request**
```bash
curl -X POST "http://localhost:8000/predict/batch?sex=F" \
  -F "files=@spectrum_001_1.csv" \
  -F "files=@spectrum_001_2.csv" \
  -F "files=@spectrum_001_3.csv"
```

**Response** `200 OK`
```json
{
  "version": "1.0",
  "timestamp": "2026-03-24T10:30:00.000000",
  "n_files": 3,
  "results": [
    { "status": "ok", "cancer_detected": true, "cancer_probability": 0.87, "...": "..." },
    { "status": "ok", "cancer_detected": true, "cancer_probability": 0.91, "...": "..." },
    { "status": "error", "file": "bad_file.csv", "message": "Cannot parse..." }
  ],
  "summary": {
    "total": 2,
    "cancer_detected": 2,
    "non_cancer": 0,
    "mean_cancer_probability": 0.89
  }
}
```

---

## Web App (`sers_webapp.py`)

### Start

```bash
python scripts/sers_webapp.py
python scripts/sers_webapp.py --host 0.0.0.0 --port 8000
```

### Endpoints

#### `GET /`

Serves the single-page web UI (HTML/CSS/JS embedded in the response).

---

#### `GET /health`

Same as REST API `/health`.

---

#### `POST /api/predict`

Single spectrum prediction. Same behavior as REST API `/predict` but at path `/api/predict`.

**Parameters** — same as REST API `/predict`.

---

#### `POST /api/predict/batch`

Batch prediction. Same behavior as REST API `/predict/batch` but at path `/api/predict/batch`.

**Additional constraints:**
- Maximum **50 files** per batch request (returns `400` if exceeded)

---

### Rate Limiting

The web app enforces **60 requests per minute per IP** (in-memory counter). Exceeding this returns `429 Too Many Requests`.

### Upload Constraints

| Constraint | Value |
|---|---|
| Max file size | 10 MB per file |
| Allowed extensions | `.csv` only |
| Max batch size | 50 files |

Invalid extensions return `400`. Oversized files return `413`.

---

## Spectrum CSV Format

Input files must be two-column CSV (no header required):

```
400.0,1234.5
401.2,1245.8
402.4,1267.3
...
```

- Column 1: Raman shift / wavenumber (cm⁻¹)
- Column 2: Intensity
- Any separator auto-detected (comma, tab, space)
- Metrohm Mira P handheld key-value format is also supported
