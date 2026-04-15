"""
BLC Subgroup Interactive HTML Dashboard.

Plotly-based interactive UMAP/t-SNE with:
  - Hover: 환자 ID + 전체 clinical info
  - 검색: 특정 sample_id 하이라이트
  - Color-by: cluster / 임상변수 선택 가능
  - Click: 개별 환자 상세 패널

Usage:
    python scripts/analysis/blc_subgroup_interactive.py
"""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SPECTRA_PATH = ROOT / "results" / "processed_spectra.csv"
CLINICAL_PATH = ROOT / "data" / "clinical_data" / "by_cancer" / "bladder.xlsx"
OUTPUT_DIR = ROOT / "results" / "blc_subgroup_analysis"
OUTPUT_HTML = OUTPUT_DIR / "blc_subgroup_interactive.html"


# ─── Data loading (reuse from static analysis) ───────────────────────────────

def load_data():
    spec = pd.read_csv(SPECTRA_PATH)
    blc = spec[spec["group"] == "BLC"].copy()
    feat_cols = [c for c in blc.columns if c.startswith("x_")]

    blc_agg = blc.groupby("sample_id")[feat_cols].mean().reset_index()

    clin = pd.read_excel(CLINICAL_PATH, sheet_name="Clean", engine="openpyxl")
    clin["sample_id"] = clin["patient_id"].astype(str).apply(
        lambda x: int(m.group(1)) if (m := re.search(r"-(\d+)$", x)) else None
    )
    clin = clin.dropna(subset=["sample_id"])
    clin["sample_id"] = clin["sample_id"].astype(int)

    # Simplify pathology
    def simp_path(p):
        if pd.isna(p): return "Unknown"
        p = str(p).upper().strip().rstrip(",")
        if "NONINVASIVE" in p or "NON-INVASIVE" in p: return "Non-invasive Papillary UC"
        if "PAPILLARY" in p: return "Papillary UC"
        if "UROTHELIAL" in p: return "UC (non-papillary)"
        return "Other"

    def simp_t(t):
        if pd.isna(t): return "Unknown"
        t = str(t).upper().strip()
        if t.startswith("TA"): return "Ta"
        if t.startswith("T1"): return "T1"
        if t.startswith("T2"): return "T2"
        if t.startswith("T3") or t.startswith("T4"): return "T3/T4"
        return t

    clin["pathology_group"] = clin["pathology"].apply(simp_path)
    clin["t_stage"] = clin["t_stage"].apply(simp_t)

    merged = blc_agg.merge(clin, on="sample_id", how="inner")
    print(f"Loaded {len(merged)} subjects")
    return merged, feat_cols


def compute_embeddings(df, feat_cols):
    from umap import UMAP
    from sklearn.manifold import TSNE
    from hdbscan import HDBSCAN

    X = df[feat_cols].values

    # UMAP
    reducer = UMAP(n_components=2, n_neighbors=20, min_dist=0.1,
                   metric="euclidean", random_state=42)
    emb_umap = reducer.fit_transform(X)
    df["umap_1"] = emb_umap[:, 0]
    df["umap_2"] = emb_umap[:, 1]

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, random_state=42,
                learning_rate="auto", init="pca")
    emb_tsne = tsne.fit_transform(X)
    df["tsne_1"] = emb_tsne[:, 0]
    df["tsne_2"] = emb_tsne[:, 1]

    # HDBSCAN on UMAP
    labels = HDBSCAN(min_cluster_size=15, min_samples=5).fit_predict(emb_umap)
    df["cluster"] = labels

    print(f"Embeddings: UMAP + t-SNE done, {len(set(labels) - {-1})} clusters")
    return df


def build_html(df):
    """Build self-contained interactive HTML."""

    # Prepare data for JS
    numeric_vars = [
        ("age", "Age"), ("bmi", "BMI"),
        ("bp_systolic", "Systolic BP"), ("bp_diastolic", "Diastolic BP"),
        ("wbc", "WBC"), ("rbc", "RBC"), ("hb", "Hemoglobin"), ("hct", "Hematocrit"),
        ("platelet", "Platelet"), ("neutrophil_pct", "Neutrophil %"),
        ("lymphocyte_pct", "Lymphocyte %"),
        ("ast", "AST"), ("alt", "ALT"), ("alp", "ALP"), ("ggt", "GGT"),
        ("bun", "BUN"), ("creatinine", "Creatinine"), ("uric_acid", "Uric Acid"),
        ("glucose", "Glucose"), ("total_bilirubin", "Bilirubin"),
        ("calcium", "Calcium"), ("total_cholesterol", "Cholesterol"),
        ("triglyceride", "Triglyceride"),
        ("ua_sg", "Urine SG"), ("ua_ph", "Urine pH"),
        ("potassium", "K"), ("chloride", "Cl"),
    ]
    categorical_vars = [
        ("cluster", "Cluster"),
        ("sex", "Sex"), ("smoking_status", "Smoking"),
        ("drinking_status", "Drinking"), ("t_stage", "T-Stage"),
        ("sample_timing", "Sample Timing"),
        ("pathology_group", "Pathology"),
        ("ua_protein", "Urine Protein"), ("ua_blood", "Urine Blood"),
        ("ua_glucose", "Urine Glucose"),
    ]

    # Build records
    records = []
    for _, row in df.iterrows():
        rec = {
            "id": int(row["sample_id"]),
            "umap_1": round(float(row["umap_1"]), 4),
            "umap_2": round(float(row["umap_2"]), 4),
            "tsne_1": round(float(row["tsne_1"]), 4),
            "tsne_2": round(float(row["tsne_2"]), 4),
            "cluster": int(row["cluster"]),
        }
        for col, _ in numeric_vars:
            if col in row and pd.notna(row[col]):
                rec[col] = round(float(row[col]), 2)
            else:
                rec[col] = None
        for col, _ in categorical_vars:
            if col == "cluster":
                continue
            if col in row and pd.notna(row[col]):
                val = row[col]
                rec[col] = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
            else:
                rec[col] = None
        records.append(rec)

    data_json = json.dumps(records)
    numeric_json = json.dumps(numeric_vars)
    categorical_json = json.dumps(categorical_vars)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BLC Subgroup Explorer</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0f172a; color: #e2e8f0; }}

  .header {{
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid #334155;
    padding: 16px 24px;
    display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
  }}
  .header h1 {{ font-size: 18px; font-weight: 700; color: #f8fafc; white-space: nowrap; }}
  .header .badge {{
    background: #7c3aed; color: white; padding: 2px 10px; border-radius: 12px;
    font-size: 12px; font-weight: 600;
  }}

  .controls {{
    display: flex; gap: 12px; padding: 12px 24px;
    background: #1e293b; border-bottom: 1px solid #334155;
    flex-wrap: wrap; align-items: center;
  }}
  .control-group {{ display: flex; align-items: center; gap: 6px; }}
  .control-group label {{ font-size: 12px; color: #94a3b8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }}
  select, input {{
    background: #0f172a; border: 1px solid #475569; color: #e2e8f0;
    padding: 6px 10px; border-radius: 6px; font-size: 13px;
    outline: none; transition: border-color 0.2s;
  }}
  select:focus, input:focus {{ border-color: #7c3aed; }}
  input::placeholder {{ color: #64748b; }}

  .main {{ display: flex; height: calc(100vh - 110px); }}

  .plot-area {{ flex: 1; position: relative; }}
  #scatter {{ width: 100%; height: 100%; }}

  .sidebar {{
    width: 340px; background: #1e293b; border-left: 1px solid #334155;
    overflow-y: auto; padding: 0;
    transition: width 0.3s;
  }}
  .sidebar.collapsed {{ width: 0; padding: 0; overflow: hidden; }}

  .sidebar-header {{
    padding: 14px 16px; background: #334155;
    font-size: 13px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.5px; color: #94a3b8;
    position: sticky; top: 0; z-index: 1;
  }}

  .patient-card {{
    padding: 12px 16px; border-bottom: 1px solid #1e293b;
    cursor: pointer; transition: background 0.15s;
  }}
  .patient-card:hover {{ background: #334155; }}
  .patient-card.active {{ background: #7c3aed22; border-left: 3px solid #7c3aed; }}

  .patient-card .pid {{ font-size: 15px; font-weight: 700; color: #f8fafc; }}
  .patient-card .cluster-tag {{
    display: inline-block; padding: 1px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600; margin-left: 6px;
  }}
  .patient-card .meta {{ font-size: 12px; color: #94a3b8; margin-top: 4px; }}

  .detail-panel {{
    position: fixed; right: 340px; top: 110px;
    width: 380px; max-height: calc(100vh - 130px);
    background: #1e293b; border: 1px solid #475569;
    border-radius: 12px; box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    overflow-y: auto; display: none; z-index: 100;
  }}
  .detail-panel.show {{ display: block; }}
  .detail-header {{
    padding: 16px; background: #334155; border-radius: 12px 12px 0 0;
    display: flex; justify-content: space-between; align-items: center;
  }}
  .detail-header h3 {{ font-size: 16px; font-weight: 700; }}
  .detail-close {{ cursor: pointer; color: #94a3b8; font-size: 20px; }}
  .detail-close:hover {{ color: #f8fafc; }}
  .detail-body {{ padding: 16px; }}
  .detail-section {{ margin-bottom: 14px; }}
  .detail-section h4 {{
    font-size: 11px; text-transform: uppercase; color: #7c3aed;
    letter-spacing: 0.5px; margin-bottom: 6px; font-weight: 700;
  }}
  .detail-row {{ display: flex; justify-content: space-between; padding: 3px 0; font-size: 13px; }}
  .detail-row .label {{ color: #94a3b8; }}
  .detail-row .value {{ color: #f8fafc; font-weight: 600; }}

  .highlight-ring {{
    position: absolute; pointer-events: none;
    border: 3px solid #fbbf24; border-radius: 50%;
    width: 28px; height: 28px;
    animation: pulse 1.5s ease-in-out infinite;
    display: none;
  }}
  @keyframes pulse {{
    0%, 100% {{ transform: scale(1); opacity: 1; }}
    50% {{ transform: scale(1.6); opacity: 0.3; }}
  }}

  .legend {{
    position: absolute; bottom: 16px; left: 16px;
    background: #1e293bdd; backdrop-filter: blur(8px);
    border: 1px solid #475569; border-radius: 10px;
    padding: 10px 14px; font-size: 12px;
    max-height: 200px; overflow-y: auto;
  }}
  .legend-title {{ font-weight: 700; margin-bottom: 6px; color: #94a3b8; text-transform: uppercase; font-size: 11px; }}
  .legend-item {{ display: flex; align-items: center; gap: 6px; padding: 2px 0; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  .stats-bar {{
    position: absolute; top: 12px; left: 12px;
    display: flex; gap: 8px;
  }}
  .stat-chip {{
    background: #1e293bdd; backdrop-filter: blur(8px);
    border: 1px solid #475569; border-radius: 8px;
    padding: 6px 12px; font-size: 12px;
  }}
  .stat-chip .num {{ font-weight: 700; color: #7c3aed; }}

  .colorbar {{
    position: absolute; bottom: 16px; right: 360px;
    background: #1e293bdd; backdrop-filter: blur(8px);
    border: 1px solid #475569; border-radius: 10px;
    padding: 10px 14px; display: none;
  }}
  .colorbar-gradient {{
    width: 20px; height: 150px; border-radius: 4px;
  }}
  .colorbar-labels {{ font-size: 11px; color: #94a3b8; }}
</style>
</head>
<body>

<div class="header">
  <h1>BLC Spectral Subgroup Explorer</h1>
  <span class="badge">{len(records)} subjects</span>
</div>

<div class="controls">
  <div class="control-group">
    <label>Embedding</label>
    <select id="embSelect">
      <option value="umap" selected>UMAP</option>
      <option value="tsne">t-SNE</option>
    </select>
  </div>
  <div class="control-group">
    <label>Color by</label>
    <select id="colorSelect">
      <optgroup label="Categorical">
      </optgroup>
      <optgroup label="Numeric">
      </optgroup>
    </select>
  </div>
  <div class="control-group">
    <label>Search Patient</label>
    <input id="searchInput" type="text" placeholder="Sample ID (e.g. 42)" style="width:140px;">
  </div>
  <div class="control-group">
    <label>Point Size</label>
    <input id="sizeSlider" type="range" min="4" max="20" value="9" style="width:100px;">
  </div>
</div>

<div class="main">
  <div class="plot-area">
    <canvas id="scatter"></canvas>
    <div class="stats-bar" id="statsBar"></div>
    <div class="legend" id="legend"></div>
    <div class="colorbar" id="colorbar">
      <div class="colorbar-labels" id="cbMax" style="text-align:center;margin-bottom:4px;"></div>
      <canvas id="cbGradient" width="20" height="150" style="border-radius:4px;"></canvas>
      <div class="colorbar-labels" id="cbMin" style="text-align:center;margin-top:4px;"></div>
    </div>
  </div>
  <div class="sidebar" id="sidebar">
    <div class="sidebar-header">Patients <span id="listCount"></span></div>
    <div id="patientList"></div>
  </div>
</div>

<div class="detail-panel" id="detailPanel">
  <div class="detail-header">
    <h3 id="detailTitle">Patient #--</h3>
    <span class="detail-close" id="detailClose">&times;</span>
  </div>
  <div class="detail-body" id="detailBody"></div>
</div>

<script>
const DATA = {data_json};
const NUMERIC_VARS = {numeric_json};
const CATEGORICAL_VARS = {categorical_json};

const CLUSTER_COLORS = ['#EF4444','#3B82F6','#22C55E','#F97316','#A855F7','#06B6D4','#EC4899','#EAB308'];
const NOISE_COLOR = '#6B7280';
const VIRIDIS = [
  [68,1,84],[72,26,108],[71,47,125],[65,68,135],[57,86,140],
  [48,103,141],[40,120,142],[33,137,141],[26,153,136],[30,169,120],
  [53,183,95],[94,196,60],[143,205,26],[194,211,30],[241,229,29],[253,231,37]
];

function interpViridis(t) {{
  t = Math.max(0, Math.min(1, t));
  const idx = t * (VIRIDIS.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  const f = idx - lo;
  const c = VIRIDIS[lo].map((v, i) => Math.round(v + f * (VIRIDIS[hi][i] - v)));
  return `rgb(${{c[0]}},${{c[1]}},${{c[2]}})`;
}}

let currentEmb = 'umap';
let currentColor = 'cluster';
let highlightId = null;
let selectedId = null;
let pointSize = 9;

// ── Build color selector ─────────────────────────────────────────────────
const colorSel = document.getElementById('colorSelect');
const catGroup = colorSel.querySelector('optgroup[label="Categorical"]');
const numGroup = colorSel.querySelector('optgroup[label="Numeric"]');
CATEGORICAL_VARS.forEach(([k, v]) => {{
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = v;
  if (k === 'cluster') opt.selected = true;
  catGroup.appendChild(opt);
}});
NUMERIC_VARS.forEach(([k, v]) => {{
  const opt = document.createElement('option');
  opt.value = k; opt.textContent = v;
  numGroup.appendChild(opt);
}});

// ── Canvas setup ─────────────────────────────────────────────────────────
const canvas = document.getElementById('scatter');
const ctx = canvas.getContext('2d');
let W, H, transform;

function resize() {{
  const rect = canvas.parentElement.getBoundingClientRect();
  W = rect.width; H = rect.height;
  canvas.width = W * devicePixelRatio;
  canvas.height = H * devicePixelRatio;
  canvas.style.width = W + 'px';
  canvas.style.height = H + 'px';
  ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
  computeTransform();
  draw();
}}

function computeTransform() {{
  const xKey = currentEmb === 'umap' ? 'umap_1' : 'tsne_1';
  const yKey = currentEmb === 'umap' ? 'umap_2' : 'tsne_2';
  const xs = DATA.map(d => d[xKey]);
  const ys = DATA.map(d => d[yKey]);
  const pad = 50;
  const xMin = Math.min(...xs), xMax = Math.max(...xs);
  const yMin = Math.min(...ys), yMax = Math.max(...ys);
  const xRange = xMax - xMin || 1;
  const yRange = yMax - yMin || 1;
  const scale = Math.min((W - 2*pad) / xRange, (H - 2*pad) / yRange);
  const offX = (W - xRange * scale) / 2;
  const offY = (H - yRange * scale) / 2;
  transform = {{ xMin, yMin, scale, offX, offY }};
}}

function toScreen(d) {{
  const xKey = currentEmb === 'umap' ? 'umap_1' : 'tsne_1';
  const yKey = currentEmb === 'umap' ? 'umap_2' : 'tsne_2';
  return {{
    x: (d[xKey] - transform.xMin) * transform.scale + transform.offX,
    y: H - ((d[yKey] - transform.yMin) * transform.scale + transform.offY),
  }};
}}

// ── Color logic ──────────────────────────────────────────────────────────
function isNumericVar(v) {{ return NUMERIC_VARS.some(([k]) => k === v); }}

function getColor(d) {{
  const v = currentColor;
  if (v === 'cluster') {{
    const c = d.cluster;
    return c < 0 ? NOISE_COLOR : CLUSTER_COLORS[c % CLUSTER_COLORS.length];
  }}
  if (isNumericVar(v)) {{
    const val = d[v];
    if (val == null) return '#374151';
    const vals = DATA.map(r => r[v]).filter(x => x != null);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const t = hi > lo ? (val - lo) / (hi - lo) : 0.5;
    return interpViridis(t);
  }}
  // categorical
  const cats = [...new Set(DATA.map(r => r[v]).filter(x => x != null))].sort();
  const idx = cats.indexOf(d[v]);
  if (idx < 0) return '#374151';
  const palette = ['#EF4444','#3B82F6','#22C55E','#F97316','#A855F7','#06B6D4','#EC4899','#EAB308','#84CC16','#F43F5E'];
  return palette[idx % palette.length];
}}

// ── Draw ─────────────────────────────────────────────────────────────────
function draw() {{
  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#1e293b';
  ctx.lineWidth = 1;
  for (let x = 0; x < W; x += 60) {{ ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }}
  for (let y = 0; y < H; y += 60) {{ ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }}

  // Points
  const sorted = [...DATA].sort((a, b) => {{
    if (a.id === selectedId) return 1;
    if (b.id === selectedId) return -1;
    if (a.id === highlightId) return 1;
    if (b.id === highlightId) return -1;
    return 0;
  }});

  sorted.forEach(d => {{
    const {{ x, y }} = toScreen(d);
    const isHL = d.id === highlightId;
    const isSel = d.id === selectedId;
    const r = isHL || isSel ? pointSize * 1.8 : pointSize;

    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = getColor(d);
    ctx.globalAlpha = (highlightId && !isHL && !isSel) ? 0.15 : 0.8;
    ctx.fill();

    if (isHL || isSel) {{
      ctx.strokeStyle = '#fbbf24';
      ctx.lineWidth = 3;
      ctx.stroke();
      // Label
      ctx.globalAlpha = 1;
      ctx.fillStyle = '#fbbf24';
      ctx.font = 'bold 13px system-ui';
      ctx.fillText(`#${{d.id}}`, x + r + 5, y - r);
    }}
    ctx.globalAlpha = 1;
  }});

  updateLegend();
  updateStats();
}}

// ── Legend ────────────────────────────────────────────────────────────────
function updateLegend() {{
  const legend = document.getElementById('legend');
  const colorbar = document.getElementById('colorbar');
  const v = currentColor;

  if (isNumericVar(v)) {{
    legend.style.display = 'none';
    colorbar.style.display = 'block';
    const vals = DATA.map(r => r[v]).filter(x => x != null);
    const lo = Math.min(...vals), hi = Math.max(...vals);
    document.getElementById('cbMax').textContent = hi.toFixed(1);
    document.getElementById('cbMin').textContent = lo.toFixed(1);
    const cbCanvas = document.getElementById('cbGradient');
    const cbCtx = cbCanvas.getContext('2d');
    for (let i = 0; i < 150; i++) {{
      cbCtx.fillStyle = interpViridis(1 - i / 149);
      cbCtx.fillRect(0, i, 20, 1);
    }}
    return;
  }}

  colorbar.style.display = 'none';
  legend.style.display = 'block';
  const label = [...CATEGORICAL_VARS, ...NUMERIC_VARS].find(([k]) => k === v)?.[1] || v;
  let items = '';

  if (v === 'cluster') {{
    const clusters = [...new Set(DATA.map(d => d.cluster))].sort((a,b) => a-b);
    items = clusters.map(c => {{
      const col = c < 0 ? NOISE_COLOR : CLUSTER_COLORS[c % CLUSTER_COLORS.length];
      const n = DATA.filter(d => d.cluster === c).length;
      const lbl = c < 0 ? `Noise (${{n}})` : `C${{c}} (${{n}})`;
      return `<div class="legend-item"><span class="legend-dot" style="background:${{col}}"></span>${{lbl}}</div>`;
    }}).join('');
  }} else {{
    const cats = [...new Set(DATA.map(d => d[v]).filter(x => x != null))].sort();
    const palette = ['#EF4444','#3B82F6','#22C55E','#F97316','#A855F7','#06B6D4','#EC4899','#EAB308','#84CC16','#F43F5E'];
    items = cats.map((c, i) => {{
      const n = DATA.filter(d => d[v] === c).length;
      return `<div class="legend-item"><span class="legend-dot" style="background:${{palette[i % palette.length]}}"></span>${{c}} (${{n}})</div>`;
    }}).join('');
  }}
  legend.innerHTML = `<div class="legend-title">${{label}}</div>${{items}}`;
}}

function updateStats() {{
  const bar = document.getElementById('statsBar');
  const nClusters = new Set(DATA.filter(d => d.cluster >= 0).map(d => d.cluster)).size;
  const nNoise = DATA.filter(d => d.cluster < 0).length;
  bar.innerHTML = `
    <div class="stat-chip"><span class="num">${{DATA.length}}</span> subjects</div>
    <div class="stat-chip"><span class="num">${{nClusters}}</span> clusters</div>
    <div class="stat-chip"><span class="num">${{nNoise}}</span> noise</div>
    <div class="stat-chip">${{currentEmb.toUpperCase()}}</div>
  `;
}}

// ── Patient list ─────────────────────────────────────────────────────────
function buildPatientList(filter) {{
  const list = document.getElementById('patientList');
  let items = DATA;
  if (filter) {{
    const q = filter.toLowerCase();
    items = items.filter(d => String(d.id).includes(q));
  }}
  items = items.sort((a,b) => a.id - b.id);
  document.getElementById('listCount').textContent = `(${{items.length}})`;

  list.innerHTML = items.map(d => {{
    const clr = d.cluster < 0 ? NOISE_COLOR : CLUSTER_COLORS[d.cluster % CLUSTER_COLORS.length];
    const tag = d.cluster < 0 ? 'Noise' : `C${{d.cluster}}`;
    const meta = [d.sex, d.age ? d.age+'y' : '', d.t_stage].filter(Boolean).join(' · ');
    return `<div class="patient-card" data-id="${{d.id}}"
      onmouseenter="hoverPatient(${{d.id}})" onmouseleave="unhoverPatient()"
      onclick="selectPatient(${{d.id}})">
      <span class="pid">#${{d.id}}</span>
      <span class="cluster-tag" style="background:${{clr}}33;color:${{clr}}">${{tag}}</span>
      <div class="meta">${{meta}}</div>
    </div>`;
  }}).join('');
}}

function hoverPatient(id) {{
  highlightId = id;
  draw();
}}
function unhoverPatient() {{
  highlightId = null;
  draw();
}}

function selectPatient(id) {{
  selectedId = id;
  highlightId = id;
  draw();
  showDetail(id);
  // Highlight in list
  document.querySelectorAll('.patient-card').forEach(el => {{
    el.classList.toggle('active', parseInt(el.dataset.id) === id);
  }});
}}

// ── Detail panel ─────────────────────────────────────────────────────────
function showDetail(id) {{
  const d = DATA.find(r => r.id === id);
  if (!d) return;
  const panel = document.getElementById('detailPanel');
  document.getElementById('detailTitle').innerHTML =
    `Patient #${{d.id}} <span class="cluster-tag" style="background:${{
      d.cluster < 0 ? NOISE_COLOR : CLUSTER_COLORS[d.cluster % CLUSTER_COLORS.length]
    }}33;color:${{
      d.cluster < 0 ? NOISE_COLOR : CLUSTER_COLORS[d.cluster % CLUSTER_COLORS.length]
    }}">${{d.cluster < 0 ? 'Noise' : 'C'+d.cluster}}</span>`;

  const sections = [
    ['Demographics', [['Age', d.age], ['Sex', d.sex], ['BMI', d.bmi], ['Systolic BP', d.bp_systolic], ['Diastolic BP', d.bp_diastolic], ['Smoking', d.smoking_status], ['Drinking', d.drinking_status]]],
    ['Cancer', [['T-Stage', d.t_stage], ['Pathology', d.pathology_group], ['Sample Timing', d.sample_timing]]],
    ['CBC', [['WBC', d.wbc], ['RBC', d.rbc], ['Hb', d.hb], ['Hct', d.hct], ['Platelet', d.platelet], ['Neutrophil %', d.neutrophil_pct], ['Lymphocyte %', d.lymphocyte_pct]]],
    ['Chemistry', [['AST', d.ast], ['ALT', d.alt], ['ALP', d.alp], ['GGT', d.ggt], ['BUN', d.bun], ['Creatinine', d.creatinine], ['Uric Acid', d.uric_acid], ['Glucose', d.glucose], ['Bilirubin', d.total_bilirubin], ['Calcium', d.calcium], ['Cholesterol', d.total_cholesterol], ['Triglyceride', d.triglyceride], ['K', d.potassium], ['Cl', d.chloride]]],
    ['Urinalysis', [['SG', d.ua_sg], ['pH', d.ua_ph], ['Protein', d.ua_protein], ['Blood', d.ua_blood], ['Glucose', d.ua_glucose]]],
  ];

  let html = '';
  sections.forEach(([title, rows]) => {{
    html += `<div class="detail-section"><h4>${{title}}</h4>`;
    rows.forEach(([lbl, val]) => {{
      const v = val != null ? val : '—';
      html += `<div class="detail-row"><span class="label">${{lbl}}</span><span class="value">${{v}}</span></div>`;
    }});
    html += '</div>';
  }});

  // Coordinates
  html += `<div class="detail-section"><h4>Embedding</h4>
    <div class="detail-row"><span class="label">UMAP</span><span class="value">(${{d.umap_1.toFixed(2)}}, ${{d.umap_2.toFixed(2)}})</span></div>
    <div class="detail-row"><span class="label">t-SNE</span><span class="value">(${{d.tsne_1.toFixed(2)}}, ${{d.tsne_2.toFixed(2)}})</span></div>
  </div>`;

  document.getElementById('detailBody').innerHTML = html;
  panel.classList.add('show');
}}

document.getElementById('detailClose').onclick = () => {{
  document.getElementById('detailPanel').classList.remove('show');
  selectedId = null;
  draw();
}};

// ── Canvas interactions ──────────────────────────────────────────────────
canvas.addEventListener('mousemove', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let closest = null, minDist = Infinity;
  DATA.forEach(d => {{
    const {{ x, y }} = toScreen(d);
    const dist = Math.hypot(mx - x, my - y);
    if (dist < minDist) {{ minDist = dist; closest = d; }}
  }});
  if (minDist < 20 && closest) {{
    canvas.style.cursor = 'pointer';
    if (highlightId !== closest.id) {{ highlightId = closest.id; draw(); }}
  }} else {{
    canvas.style.cursor = 'default';
    if (highlightId && !selectedId) {{ highlightId = null; draw(); }}
  }}
}});

canvas.addEventListener('click', e => {{
  const rect = canvas.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;
  let closest = null, minDist = Infinity;
  DATA.forEach(d => {{
    const {{ x, y }} = toScreen(d);
    const dist = Math.hypot(mx - x, my - y);
    if (dist < minDist) {{ minDist = dist; closest = d; }}
  }});
  if (minDist < 20 && closest) selectPatient(closest.id);
}});

// ── Controls ─────────────────────────────────────────────────────────────
document.getElementById('embSelect').onchange = e => {{
  currentEmb = e.target.value;
  computeTransform();
  draw();
}};
document.getElementById('colorSelect').onchange = e => {{
  currentColor = e.target.value;
  draw();
}};
document.getElementById('searchInput').oninput = e => {{
  const q = e.target.value.trim();
  buildPatientList(q);
  if (q && DATA.some(d => String(d.id) === q)) {{
    selectPatient(parseInt(q));
  }} else {{
    highlightId = null;
    selectedId = null;
    document.getElementById('detailPanel').classList.remove('show');
    draw();
  }}
}};
document.getElementById('sizeSlider').oninput = e => {{
  pointSize = parseInt(e.target.value);
  draw();
}};

// ── Init ─────────────────────────────────────────────────────────────────
window.addEventListener('resize', resize);
buildPatientList();
resize();
</script>
</body>
</html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"HTML saved to {OUTPUT_HTML}")
    print(f"File size: {OUTPUT_HTML.stat().st_size / 1024:.0f} KB")


def main():
    df, feat_cols = load_data()
    df = compute_embeddings(df, feat_cols)
    build_html(df)


if __name__ == "__main__":
    main()
