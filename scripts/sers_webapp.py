"""
SERS Cancer Screening — Web Application

A self-contained web UI for uploading SERS spectrum CSV files
and getting cancer predictions. No external dependencies beyond
FastAPI and the production model.

Usage:
    python scripts/sers_webapp.py
    # Open http://localhost:8000 in browser

    python scripts/sers_webapp.py --port 9000 --host 0.0.0.0
"""

from __future__ import annotations

import sys
import tempfile
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sers_predict import ProductionPredictor

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    import uvicorn
except ImportError:
    print("ERROR: FastAPI not installed. Run: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sers-webapp")

# Load model
predictor = ProductionPredictor()
logger.info(f"Model loaded: {len(predictor.cancer_types)} cancer types")

app = FastAPI(title="SERS Cancer Screening", version="1.0.0")


# ── HTML UI ──
HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SERS Cancer Screening</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', -apple-system, sans-serif; background: #f0f2f5; color: #1a1a2e; }

.header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white; padding: 28px 40px; text-align: center;
}
.header h1 { font-size: 28px; font-weight: 600; letter-spacing: 0.5px; }
.header p { color: #a0b4d0; margin-top: 6px; font-size: 14px; }

.container { max-width: 960px; margin: 30px auto; padding: 0 20px; }

.card {
    background: white; border-radius: 12px; padding: 28px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); margin-bottom: 24px;
}
.card h2 { font-size: 18px; margin-bottom: 16px; color: #16213e; }

/* Upload area */
.upload-area {
    border: 2px dashed #c0c8d8; border-radius: 12px; padding: 48px 24px;
    text-align: center; cursor: pointer; transition: all 0.3s;
    background: #f8f9fc;
}
.upload-area:hover, .upload-area.dragover {
    border-color: #0f3460; background: #e8edf6;
}
.upload-area .icon { font-size: 48px; margin-bottom: 12px; }
.upload-area p { color: #666; font-size: 14px; }
.upload-area .formats { color: #999; font-size: 12px; margin-top: 8px; }

/* Clinical inputs */
.clinical-section { margin-top: 20px; }
.clinical-toggle {
    cursor: pointer; color: #0f3460; font-size: 14px;
    user-select: none; display: flex; align-items: center; gap: 6px;
}
.clinical-fields {
    display: none; margin-top: 16px;
    grid-template-columns: repeat(3, 1fr); gap: 12px;
}
.clinical-fields.show { display: grid; }
.field label { display: block; font-size: 12px; color: #666; margin-bottom: 4px; font-weight: 500; }
.field input, .field select {
    width: 100%; padding: 10px 12px; border: 1px solid #d0d5dd; border-radius: 8px;
    font-size: 14px; outline: none; transition: border 0.2s;
}
.field input:focus, .field select:focus { border-color: #0f3460; }

/* File list */
.file-list { margin-top: 16px; }
.file-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; background: #f8f9fc; border-radius: 8px; margin-bottom: 6px;
    font-size: 13px;
}
.file-item .name { font-weight: 500; }
.file-item .remove { cursor: pointer; color: #e53935; font-size: 18px; }

/* Button */
.btn-predict {
    width: 100%; padding: 14px; border: none; border-radius: 10px;
    background: linear-gradient(135deg, #0f3460, #1a5276);
    color: white; font-size: 16px; font-weight: 600; cursor: pointer;
    margin-top: 20px; transition: transform 0.1s, box-shadow 0.2s;
}
.btn-predict:hover { transform: translateY(-1px); box-shadow: 0 4px 16px rgba(15,52,96,0.3); }
.btn-predict:disabled { background: #999; cursor: not-allowed; transform: none; }
.btn-predict.loading { animation: pulse 1.5s infinite; }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.7; } }

/* Results */
.results { display: none; }
.results.show { display: block; }

.result-summary {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;
    margin-bottom: 20px;
}
.stat-card {
    text-align: center; padding: 20px; border-radius: 10px;
}
.stat-card .value { font-size: 32px; font-weight: 700; }
.stat-card .label { font-size: 12px; color: #666; margin-top: 4px; }
.stat-card.detected { background: #ffeaea; color: #c62828; }
.stat-card.clear { background: #e8f5e9; color: #2e7d32; }
.stat-card.prob { background: #e3f2fd; color: #1565c0; }

.result-item {
    border: 1px solid #e0e4ea; border-radius: 10px; padding: 18px;
    margin-bottom: 12px; transition: all 0.2s;
}
.result-item:hover { box-shadow: 0 2px 8px rgba(0,0,0,0.06); }
.result-item .file-name { font-weight: 600; font-size: 14px; margin-bottom: 10px; }

.prob-bar-container {
    background: #e8ecf2; border-radius: 20px; height: 28px; overflow: hidden;
    position: relative; margin: 8px 0;
}
.prob-bar {
    height: 100%; border-radius: 20px; transition: width 0.8s ease-out;
    display: flex; align-items: center; padding-left: 12px;
    font-size: 13px; font-weight: 600; color: white;
}
.prob-bar.cancer { background: linear-gradient(90deg, #e53935, #c62828); }
.prob-bar.normal { background: linear-gradient(90deg, #43a047, #2e7d32); }

.type-prediction {
    margin-top: 10px; padding: 10px 14px;
    background: #f8f9fc; border-radius: 8px; font-size: 13px;
}
.type-probs {
    display: flex; gap: 8px; flex-wrap: wrap; margin-top: 8px;
}
.type-prob-tag {
    padding: 4px 10px; border-radius: 6px; font-size: 12px;
    background: #e8ecf2; color: #333;
}
.type-prob-tag.top { background: #0f3460; color: white; font-weight: 600; }

.warnings { margin-top: 8px; }
.warning-tag {
    display: inline-block; padding: 3px 8px; border-radius: 4px;
    background: #fff3e0; color: #e65100; font-size: 11px; margin: 2px;
}

.model-badge {
    display: inline-block; padding: 3px 8px; border-radius: 4px;
    font-size: 11px; font-weight: 500; margin-left: 8px;
}
.model-badge.fusion { background: #e8eaf6; color: #283593; }
.model-badge.sers { background: #e0f2f1; color: #00695c; }

.footer { text-align: center; padding: 20px; color: #999; font-size: 12px; }
</style>
</head>
<body>

<div class="header">
    <h1>SERS Cancer Screening System</h1>
    <p>Upload SERS urine spectrum CSV files for multi-cancer detection</p>
</div>

<div class="container">
    <!-- Upload Card -->
    <div class="card">
        <h2>Upload Spectra</h2>
        <div class="upload-area" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">📂</div>
            <p><strong>Click or drag & drop</strong> spectrum CSV files here</p>
            <p class="formats">Supported: .CSV files (wavenumber, intensity format)</p>
        </div>
        <input type="file" id="fileInput" multiple accept=".csv,.CSV" style="display:none">

        <div class="file-list" id="fileList"></div>

        <!-- Clinical features (optional) -->
        <div class="clinical-section">
            <div class="clinical-toggle" onclick="toggleClinical()">
                <span id="clinicalArrow">▶</span>
                <span>Add patient info (optional, improves accuracy)</span>
            </div>
            <div class="clinical-fields" id="clinicalFields">
                <div class="field">
                    <label>Age</label>
                    <input type="number" id="age" placeholder="e.g. 55" min="1" max="120">
                </div>
                <div class="field">
                    <label>Sex</label>
                    <select id="sex">
                        <option value="">-- Select --</option>
                        <option value="M">Male</option>
                        <option value="F">Female</option>
                    </select>
                </div>
                <div class="field">
                    <label>BMI</label>
                    <input type="number" id="bmi" placeholder="e.g. 24.3" step="0.1" min="10" max="60">
                </div>
            </div>
        </div>

        <button class="btn-predict" id="predictBtn" onclick="runPrediction()" disabled>
            Analyze Spectra
        </button>
    </div>

    <!-- Results Card -->
    <div class="card results" id="resultsCard">
        <h2>Results</h2>
        <div class="result-summary" id="resultSummary"></div>
        <div id="resultDetails"></div>
    </div>
</div>

<div class="footer">
    SERS Cancer Screening v1.0 | SOLUM Healthcare | Model: LR Early Fusion
</div>

<script>
let selectedFiles = [];

// Drag & drop
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
    e.preventDefault(); dropZone.classList.remove('dragover');
    addFiles(e.dataTransfer.files);
});

document.getElementById('fileInput').addEventListener('change', e => addFiles(e.target.files));

function addFiles(files) {
    for (const f of files) {
        if (!selectedFiles.some(sf => sf.name === f.name)) {
            selectedFiles.push(f);
        }
    }
    renderFileList();
}

function removeFile(idx) {
    selectedFiles.splice(idx, 1);
    renderFileList();
}

function renderFileList() {
    const list = document.getElementById('fileList');
    const btn = document.getElementById('predictBtn');
    if (selectedFiles.length === 0) {
        list.innerHTML = '';
        btn.disabled = true;
        return;
    }
    btn.disabled = false;
    list.innerHTML = selectedFiles.map((f, i) =>
        `<div class="file-item">
            <span class="name">${f.name}</span>
            <span class="remove" onclick="removeFile(${i})">×</span>
        </div>`
    ).join('');
}

function toggleClinical() {
    const fields = document.getElementById('clinicalFields');
    const arrow = document.getElementById('clinicalArrow');
    fields.classList.toggle('show');
    arrow.textContent = fields.classList.contains('show') ? '▼' : '▶';
}

async function runPrediction() {
    const btn = document.getElementById('predictBtn');
    btn.disabled = true;
    btn.textContent = 'Analyzing...';
    btn.classList.add('loading');

    const formData = new FormData();
    selectedFiles.forEach(f => formData.append('files', f));

    const age = document.getElementById('age').value;
    const sex = document.getElementById('sex').value;
    const bmi = document.getElementById('bmi').value;

    let url = '/api/predict/batch';
    const params = new URLSearchParams();
    if (age) params.append('age', age);
    if (sex) params.append('sex', sex);
    if (bmi) params.append('bmi', bmi);
    if (params.toString()) url += '?' + params.toString();

    try {
        const resp = await fetch(url, { method: 'POST', body: formData });
        const data = await resp.json();
        showResults(data);
    } catch (err) {
        alert('Error: ' + err.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze Spectra';
        btn.classList.remove('loading');
    }
}

function showResults(data) {
    const card = document.getElementById('resultsCard');
    card.classList.add('show');

    // Summary
    const s = data.summary || {};
    document.getElementById('resultSummary').innerHTML = `
        <div class="stat-card detected">
            <div class="value">${s.cancer_detected || 0}</div>
            <div class="label">Cancer Detected</div>
        </div>
        <div class="stat-card clear">
            <div class="value">${s.non_cancer || 0}</div>
            <div class="label">Non-cancer</div>
        </div>
        <div class="stat-card prob">
            <div class="value">${((s.mean_cancer_probability || 0) * 100).toFixed(1)}%</div>
            <div class="label">Mean Probability</div>
        </div>
    `;

    // Details
    const details = document.getElementById('resultDetails');
    details.innerHTML = data.results.map(r => {
        if (r.status === 'error') {
            return `<div class="result-item">
                <div class="file-name">${r.file} <span style="color:#e53935">ERROR: ${r.message}</span></div>
            </div>`;
        }

        const prob = r.cancer_probability;
        const pct = (prob * 100).toFixed(1);
        const barClass = r.cancer_detected ? 'cancer' : 'normal';
        const barWidth = Math.max(prob * 100, 5);
        const variant = r.model_variant === 'fusion' ? 'fusion' : 'sers';
        const badgeText = variant === 'fusion' ? 'SERS + Clinical' : 'SERS Only';

        let typeHtml = '';
        if (r.cancer_detected && r.cancer_type_probabilities) {
            const sorted = Object.entries(r.cancer_type_probabilities).sort((a,b) => b[1]-a[1]);
            const tags = sorted.map(([name, p]) =>
                `<span class="type-prob-tag ${name === r.cancer_type_prediction ? 'top' : ''}">${name}: ${(p*100).toFixed(1)}%</span>`
            ).join('');
            typeHtml = `<div class="type-prediction">
                <strong>Predicted type: ${r.cancer_type_prediction}</strong> (confidence: ${(r.cancer_type_confidence*100).toFixed(1)}%)
                <div class="type-probs">${tags}</div>
            </div>`;
        }

        let warningHtml = '';
        if (r.warnings && r.warnings.length > 0) {
            warningHtml = `<div class="warnings">${r.warnings.map(w => `<span class="warning-tag">${w}</span>`).join('')}</div>`;
        }

        return `<div class="result-item">
            <div class="file-name">${r.file} <span class="model-badge ${variant}">${badgeText}</span></div>
            <div class="prob-bar-container">
                <div class="prob-bar ${barClass}" style="width:${barWidth}%">${pct}% ${r.cancer_detected ? 'Cancer' : 'Normal'}</div>
            </div>
            ${typeHtml}
            ${warningHtml}
        </div>`;
    }).join('');

    card.scrollIntoView({ behavior: 'smooth' });
}
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML_PAGE


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True,
            "cancer_types": predictor.cancer_types,
            "timestamp": datetime.now().isoformat()}


@app.get("/api/model-info")
def model_info():
    return predictor.manifest


@app.post("/api/predict")
async def predict_single(
    file: UploadFile = File(...),
    age: float | None = None, sex: str | None = None, bmi: float | None = None,
):
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        result = predictor.predict_single(tmp_path, age=age, sex=sex, bmi=bmi)
        result["original_filename"] = file.filename
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/predict/batch")
async def predict_batch(
    files: list[UploadFile] = File(...),
    age: float | None = None, sex: str | None = None, bmi: float | None = None,
):
    results = []
    for file in files:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            r = predictor.predict_single(tmp_path, age=age, sex=sex, bmi=bmi)
            r["original_filename"] = file.filename
            results.append(r)
        except Exception as e:
            results.append({"file": file.filename, "status": "error", "message": str(e)})
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    ok = [r for r in results if r.get("status") == "ok"]
    summary = {}
    if ok:
        summary = {
            "total": len(ok),
            "cancer_detected": sum(1 for r in ok if r["cancer_detected"]),
            "non_cancer": sum(1 for r in ok if not r["cancer_detected"]),
            "mean_cancer_probability": round(float(np.mean([r["cancer_probability"] for r in ok])), 4),
        }

    return {"version": "1.0", "timestamp": datetime.now().isoformat(),
            "n_files": len(results), "results": results, "summary": summary}


def main():
    p = argparse.ArgumentParser(description="SERS Cancer Screening Web App")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    print(f"\n{'='*50}")
    print(f"  SERS Cancer Screening Web App")
    print(f"  Open: http://{args.host}:{args.port}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
