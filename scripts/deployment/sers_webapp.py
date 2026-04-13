"""
SERS Cancer Screening — Web Application

Usage:
    python scripts/sers_webapp.py
    python scripts/sers_webapp.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import os
import secrets
import tempfile
import argparse
import logging
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.sers_predict import ProductionPredictor

try:
    from fastapi import FastAPI, UploadFile, File, Request, HTTPException
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    import uvicorn
except ImportError:
    print("ERROR: Run: pip install fastapi uvicorn python-multipart")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("sers-webapp")

predictor = ProductionPredictor()
logger.info(f"Model loaded: {len(predictor.cancer_types)} cancer types")

app = FastAPI(title="SERS Cancer Screening", version="1.0.0", docs_url=None, redoc_url=None)

# ── Security ──
# Rate limiting: simple in-memory counter
_request_counts: dict[str, list[float]] = {}
MAX_REQUESTS_PER_MINUTE = 60
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB per file
ALLOWED_EXTENSIONS = {".csv"}


def check_rate_limit(client_ip: str):
    import time
    now = time.time()
    if client_ip not in _request_counts:
        _request_counts[client_ip] = []
    _request_counts[client_ip] = [t for t in _request_counts[client_ip] if now - t < 60]
    if len(_request_counts[client_ip]) >= MAX_REQUESTS_PER_MINUTE:
        raise HTTPException(429, "Too many requests. Please wait.")
    _request_counts[client_ip].append(now)


def validate_upload(file: UploadFile):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Only CSV files allowed, got: {ext}")


HTML_PAGE = """
<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SERS Cancer Screening</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
:root {
    --bg: #0a0e1a;
    --card: #111827;
    --card-hover: #1a2237;
    --border: #1e293b;
    --accent: #3b82f6;
    --accent-glow: rgba(59,130,246,0.15);
    --green: #10b981;
    --red: #ef4444;
    --orange: #f59e0b;
    --text: #e2e8f0;
    --text-dim: #94a3b8;
    --text-bright: #f8fafc;
}
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Inter', -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    min-height: 100vh;
}

/* Header */
.header {
    background: linear-gradient(180deg, #111827 0%, var(--bg) 100%);
    border-bottom: 1px solid var(--border);
    padding: 32px 40px; text-align: center;
}
.header .logo {
    display: inline-flex; align-items: center; gap: 12px; margin-bottom: 8px;
}
.header .logo-icon {
    width: 42px; height: 42px; background: var(--accent);
    border-radius: 10px; display: flex; align-items: center; justify-content: center;
    font-size: 22px; box-shadow: 0 0 20px var(--accent-glow);
}
.header h1 {
    font-size: 26px; font-weight: 700; color: var(--text-bright);
    letter-spacing: -0.5px;
}
.header .subtitle {
    color: var(--text-dim); font-size: 14px; margin-top: 6px; font-weight: 300;
}
.header .badge {
    display: inline-block; margin-top: 10px; padding: 4px 12px;
    background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.3);
    border-radius: 20px; font-size: 11px; color: var(--green); font-weight: 500;
}

.container { max-width: 900px; margin: 0 auto; padding: 28px 20px; }

/* Cards */
.card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: 16px; padding: 28px; margin-bottom: 20px;
    transition: border-color 0.3s;
}
.card:hover { border-color: #2d3a52; }
.card-title {
    font-size: 15px; font-weight: 600; color: var(--text-bright);
    margin-bottom: 18px; display: flex; align-items: center; gap: 8px;
}
.card-title .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); }

/* Upload */
.upload-zone {
    border: 2px dashed #2d3a52; border-radius: 14px;
    padding: 50px 24px; text-align: center; cursor: pointer;
    transition: all 0.3s; background: rgba(59,130,246,0.02);
}
.upload-zone:hover, .upload-zone.active {
    border-color: var(--accent); background: var(--accent-glow);
}
.upload-zone .icon { font-size: 44px; margin-bottom: 14px; filter: grayscale(0.3); }
.upload-zone p { color: var(--text-dim); font-size: 14px; }
.upload-zone .hint { font-size: 12px; color: #475569; margin-top: 8px; }
.upload-zone .browse {
    color: var(--accent); font-weight: 500; cursor: pointer;
    text-decoration: underline; text-underline-offset: 2px;
}

/* File list */
.files { margin-top: 14px; }
.file-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 6px 12px; background: #1e293b; border-radius: 8px;
    font-size: 12px; margin: 3px; font-weight: 500; color: var(--text);
    border: 1px solid var(--border);
}
.file-chip .x {
    cursor: pointer; color: var(--red); font-size: 14px;
    margin-left: 2px; opacity: 0.7; transition: opacity 0.2s;
}
.file-chip .x:hover { opacity: 1; }

/* Clinical toggle */
.clinical-toggle {
    margin-top: 18px; cursor: pointer; user-select: none;
    display: flex; align-items: center; gap: 8px;
    color: var(--text-dim); font-size: 13px; font-weight: 500;
    transition: color 0.2s;
}
.clinical-toggle:hover { color: var(--accent); }
.clinical-toggle .arrow { transition: transform 0.3s; font-size: 10px; }
.clinical-toggle .arrow.open { transform: rotate(90deg); }

.clinical-grid {
    display: none; margin-top: 14px;
    grid-template-columns: repeat(3, 1fr); gap: 10px;
}
.clinical-grid.show { display: grid; }
.input-group label {
    display: block; font-size: 11px; color: var(--text-dim);
    margin-bottom: 5px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;
}
.input-group input, .input-group select {
    width: 100%; padding: 10px 12px; border: 1px solid var(--border);
    border-radius: 8px; font-size: 13px; background: #0f172a;
    color: var(--text); outline: none; transition: border-color 0.2s;
    font-family: inherit;
}
.input-group input:focus, .input-group select:focus {
    border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-glow);
}
.input-group select { appearance: none; cursor: pointer; }

/* Button */
.btn-analyze {
    width: 100%; padding: 14px; border: none; border-radius: 12px;
    background: var(--accent); color: white; font-size: 15px;
    font-weight: 600; cursor: pointer; margin-top: 20px;
    transition: all 0.2s; font-family: inherit;
    box-shadow: 0 4px 14px rgba(59,130,246,0.25);
}
.btn-analyze:hover:not(:disabled) {
    background: #2563eb; transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(59,130,246,0.35);
}
.btn-analyze:disabled { background: #1e293b; color: #475569; cursor: default; box-shadow: none; }
.btn-analyze.loading::after {
    content: ''; display: inline-block; width: 16px; height: 16px;
    border: 2px solid rgba(255,255,255,0.3); border-top-color: white;
    border-radius: 50%; margin-left: 10px; animation: spin 0.8s linear infinite;
    vertical-align: middle;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* Results */
.results-section { display: none; }
.results-section.show { display: block; animation: fadeUp 0.4s ease-out; }
@keyframes fadeUp { from { opacity: 0; transform: translateY(16px); } to { opacity: 1; transform: none; } }

.summary-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 20px; }
.summary-stat {
    padding: 20px; border-radius: 12px; text-align: center;
    border: 1px solid var(--border);
}
.summary-stat .num { font-size: 36px; font-weight: 700; line-height: 1; }
.summary-stat .lbl { font-size: 11px; color: var(--text-dim); margin-top: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.summary-stat.cancer { background: rgba(239,68,68,0.08); border-color: rgba(239,68,68,0.2); }
.summary-stat.cancer .num { color: var(--red); }
.summary-stat.clear { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.2); }
.summary-stat.clear .num { color: var(--green); }
.summary-stat.avg { background: rgba(59,130,246,0.08); border-color: rgba(59,130,246,0.2); }
.summary-stat.avg .num { color: var(--accent); }

/* Result items */
.result-row {
    border: 1px solid var(--border); border-radius: 12px;
    padding: 18px 20px; margin-bottom: 10px;
    background: var(--card); transition: border-color 0.2s;
}
.result-row:hover { border-color: #334155; }
.result-row .top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.result-row .fname { font-weight: 600; font-size: 13px; color: var(--text-bright); }
.result-row .variant-tag {
    padding: 3px 8px; border-radius: 6px; font-size: 10px;
    font-weight: 600; text-transform: uppercase; letter-spacing: 0.3px;
}
.variant-tag.fusion { background: rgba(59,130,246,0.15); color: var(--accent); }
.variant-tag.sers { background: rgba(16,185,129,0.15); color: var(--green); }

.prob-track {
    height: 32px; background: #1e293b; border-radius: 10px;
    overflow: hidden; position: relative;
}
.prob-fill {
    height: 100%; border-radius: 10px; display: flex;
    align-items: center; padding: 0 14px; font-size: 13px;
    font-weight: 600; color: white; transition: width 1s cubic-bezier(0.22,1,0.36,1);
    min-width: 50px;
}
.prob-fill.cancer { background: linear-gradient(90deg, #dc2626, #ef4444); }
.prob-fill.normal { background: linear-gradient(90deg, #059669, #10b981); }
.prob-label {
    position: absolute; right: 14px; top: 50%; transform: translateY(-50%);
    font-size: 12px; color: var(--text-dim); font-weight: 500;
}

.type-box {
    margin-top: 12px; padding: 12px 14px; background: #0f172a;
    border-radius: 10px; border: 1px solid var(--border);
}
.type-box .predicted {
    font-size: 13px; font-weight: 600; color: var(--text-bright); margin-bottom: 8px;
}
.type-tags { display: flex; gap: 6px; flex-wrap: wrap; }
.type-tag {
    padding: 4px 10px; border-radius: 6px; font-size: 11px;
    background: #1e293b; color: var(--text-dim); font-weight: 500;
    border: 1px solid transparent;
}
.type-tag.best {
    background: rgba(59,130,246,0.15); color: var(--accent);
    border-color: rgba(59,130,246,0.3); font-weight: 600;
}

.warn-row { margin-top: 8px; }
.warn-pill {
    display: inline-block; padding: 3px 8px; border-radius: 5px;
    background: rgba(245,158,11,0.1); color: var(--orange);
    font-size: 10px; margin: 2px; font-weight: 500;
}

.footer {
    text-align: center; padding: 28px 20px; color: #334155;
    font-size: 11px; border-top: 1px solid var(--border); margin-top: 20px;
}
.footer a { color: var(--text-dim); text-decoration: none; }
</style>
</head>
<body>

<div class="header">
    <div class="logo">
        <div class="logo-icon">S</div>
        <h1>SERS Cancer Screening</h1>
    </div>
    <div class="subtitle">Multi-cancer detection from urine SERS spectroscopy</div>
    <div class="badge">Model Ready &middot; 5 Cancer Types &middot; LR Fusion v1.0</div>
</div>

<div class="container">
    <div class="card">
        <div class="card-title"><span class="dot"></span> Upload Spectra</div>
        <div class="upload-zone" id="dropZone" onclick="document.getElementById('fileInput').click()">
            <div class="icon">&#128300;</div>
            <p>Drop spectrum CSV files here or <span class="browse">browse</span></p>
            <p class="hint">Accepts .CSV files &middot; Wavenumber + Intensity format</p>
        </div>
        <input type="file" id="fileInput" multiple accept=".csv,.CSV" style="display:none">

        <div class="files" id="fileChips"></div>

        <div class="clinical-toggle" onclick="toggleClinical()">
            <span class="arrow" id="clinArrow">&#9654;</span>
            Patient information (optional &middot; improves accuracy)
        </div>
        <div class="clinical-grid" id="clinGrid">
            <div class="input-group">
                <label>Age</label>
                <input type="number" id="age" placeholder="e.g. 55" min="1" max="120">
            </div>
            <div class="input-group">
                <label>Sex</label>
                <select id="sex">
                    <option value="">Select</option>
                    <option value="M">Male</option>
                    <option value="F">Female</option>
                </select>
            </div>
            <div class="input-group">
                <label>BMI</label>
                <input type="number" id="bmi" placeholder="e.g. 24.3" step="0.1">
            </div>
        </div>

        <button class="btn-analyze" id="btnAnalyze" onclick="analyze()" disabled>
            Analyze Spectra
        </button>
    </div>

    <div class="results-section" id="resultsSection">
        <div class="card">
            <div class="card-title"><span class="dot"></span> Results</div>
            <div class="summary-grid" id="summaryGrid"></div>
            <div id="resultRows"></div>
        </div>
    </div>
</div>

<div class="footer">
    SERS Cancer Screening v1.0 &middot; SOLUM Healthcare<br>
    <span style="color:#475569">LAN access only &middot; No data stored on server</span>
</div>

<script>
let files = [];
const dz = document.getElementById('dropZone');

dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('active'); });
dz.addEventListener('dragleave', () => dz.classList.remove('active'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('active'); addFiles(e.dataTransfer.files); });
document.getElementById('fileInput').addEventListener('change', e => addFiles(e.target.files));

function addFiles(fl) {
    for (const f of fl) if (!files.some(x => x.name === f.name)) files.push(f);
    render();
}
function removeFile(i) { files.splice(i, 1); render(); }
function render() {
    const c = document.getElementById('fileChips');
    document.getElementById('btnAnalyze').disabled = files.length === 0;
    c.innerHTML = files.map((f, i) =>
        '<span class="file-chip">' + f.name + ' <span class="x" onclick="removeFile(' + i + ')">&times;</span></span>'
    ).join('');
}
function toggleClinical() {
    const g = document.getElementById('clinGrid');
    const a = document.getElementById('clinArrow');
    g.classList.toggle('show');
    a.classList.toggle('open');
}

async function analyze() {
    const btn = document.getElementById('btnAnalyze');
    btn.disabled = true; btn.textContent = 'Analyzing'; btn.classList.add('loading');

    const fd = new FormData();
    files.forEach(f => fd.append('files', f));

    const age = document.getElementById('age').value;
    const sex = document.getElementById('sex').value;
    const bmi = document.getElementById('bmi').value;
    const p = new URLSearchParams();
    if (age) p.append('age', age);
    if (sex) p.append('sex', sex);
    if (bmi) p.append('bmi', bmi);
    let url = '/api/predict/batch';
    if (p.toString()) url += '?' + p;

    try {
        const r = await fetch(url, { method: 'POST', body: fd });
        if (!r.ok) throw new Error('Server error: ' + r.status);
        showResults(await r.json());
    } catch (e) { alert('Error: ' + e.message); }
    finally { btn.disabled = false; btn.textContent = 'Analyze Spectra'; btn.classList.remove('loading'); }
}

function showResults(data) {
    document.getElementById('resultsSection').classList.add('show');
    const s = data.summary || {};

    document.getElementById('summaryGrid').innerHTML =
        '<div class="summary-stat cancer"><div class="num">' + (s.cancer_detected||0) + '</div><div class="lbl">Cancer Detected</div></div>' +
        '<div class="summary-stat clear"><div class="num">' + (s.non_cancer||0) + '</div><div class="lbl">Non-cancer</div></div>' +
        '<div class="summary-stat avg"><div class="num">' + ((s.mean_cancer_probability||0)*100).toFixed(1) + '%</div><div class="lbl">Mean Probability</div></div>';

    document.getElementById('resultRows').innerHTML = data.results.map(r => {
        if (r.status === 'error') return '<div class="result-row"><div class="fname">' + r.file + ' <span style="color:var(--red)">Error: ' + r.message + '</span></div></div>';

        const pct = (r.cancer_probability * 100).toFixed(1);
        const w = Math.max(r.cancer_probability * 100, 4);
        const cls = r.cancer_detected ? 'cancer' : 'normal';
        const status = r.cancer_detected ? 'Cancer Detected' : 'Normal';
        const vt = r.model_variant === 'fusion' ? 'fusion' : 'sers';
        const vtLabel = vt === 'fusion' ? 'SERS + Clinical' : 'SERS Only';

        let typeHtml = '';
        if (r.cancer_detected && r.cancer_type_probabilities) {
            const sorted = Object.entries(r.cancer_type_probabilities).sort((a,b)=>b[1]-a[1]);
            const tags = sorted.map(([n,p]) =>
                '<span class="type-tag ' + (n===r.cancer_type_prediction?'best':'') + '">' + n + ' ' + (p*100).toFixed(1) + '%</span>'
            ).join('');
            typeHtml = '<div class="type-box"><div class="predicted">Predicted: ' + r.cancer_type_prediction + ' (' + (r.cancer_type_confidence*100).toFixed(1) + '% confidence)</div><div class="type-tags">' + tags + '</div></div>';
        }

        let warnHtml = '';
        if (r.warnings && r.warnings.length) warnHtml = '<div class="warn-row">' + r.warnings.map(w=>'<span class="warn-pill">' + w + '</span>').join('') + '</div>';

        return '<div class="result-row"><div class="top"><span class="fname">' + (r.original_filename||r.file) + '</span><span class="variant-tag ' + vt + '">' + vtLabel + '</span></div><div class="prob-track"><div class="prob-fill ' + cls + '" style="width:' + w + '%">' + pct + '%</div><span class="prob-label">' + status + '</span></div>' + typeHtml + warnHtml + '</div>';
    }).join('');

    document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth', block: 'start' });
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


@app.post("/api/predict")
async def predict_single(request: Request, file: UploadFile = File(...),
                         age: float | None = None, sex: str | None = None, bmi: float | None = None):
    check_rate_limit(request.client.host)
    validate_upload(file)

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        content = await file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(413, f"File too large (max {MAX_UPLOAD_SIZE//1024//1024}MB)")
        tmp.write(content)
        tmp_path = tmp.name
    try:
        result = predictor.predict_single(tmp_path, age=age, sex=sex, bmi=bmi)
        result["original_filename"] = file.filename
        return result
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@app.post("/api/predict/batch")
async def predict_batch(request: Request, files: list[UploadFile] = File(...),
                        age: float | None = None, sex: str | None = None, bmi: float | None = None):
    check_rate_limit(request.client.host)
    if len(files) > 50:
        raise HTTPException(400, "Maximum 50 files per batch")

    results = []
    for file in files:
        validate_upload(file)
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            content = await file.read()
            if len(content) > MAX_UPLOAD_SIZE:
                results.append({"file": file.filename, "status": "error", "message": "File too large"})
                continue
            tmp.write(content)
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
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    print(f"\n{'='*50}")
    print(f"  SERS Cancer Screening Web App")
    print(f"  Local:   http://localhost:{args.port}")
    print(f"  Network: http://0.0.0.0:{args.port}")
    print(f"{'='*50}\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
