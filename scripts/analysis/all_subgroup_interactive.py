"""
All-group Subgroup Interactive HTML Dashboard.

Generates a single HTML with tab-based navigation for every disease group.
Each tab: UMAP + t-SNE, HDBSCAN clustering, clinical variable overlay, patient search.

Usage:
    python scripts/analysis/all_subgroup_interactive.py
"""

import json
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]
SPECTRA_PATH = ROOT / "results" / "processed_spectra.csv"
CLINICAL_BY_CANCER = ROOT / "data" / "clinical_data" / "by_cancer"
CLINICAL_STD = ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv"
OUTPUT_DIR = ROOT / "results" / "subgroup_analysis"
OUTPUT_HTML = OUTPUT_DIR / "subgroup_explorer.html"

# ── Group definitions ─────────────────────────────────────────────────────
# (by_cancer_file, clinical_disease_group, spectra_groups, display_name)
GROUP_DEFS = [
    ("bladder",    "BLA", ["BLC"],          "BLC (Bladder)"),
    ("colorectal", "CRC", ["CRC"],          "CRC (Colorectal)"),
    ("lung",       "LUN", ["LUN"],          "LUN (Lung)"),
    ("prostate",   "PRO", ["PRO"],          "PRO (Prostate)"),
    ("pancreatic", "PAN", ["CPAN", "YPAN"], "PAN (Pancreatic)"),
    ("ovarian",    "OVA", ["OVA"],          "OVA (Ovarian)"),
    ("breast",     "BRE", ["BRE"],          "BRE (Breast)"),
    ("controls",   "NOR", ["NOR", "YNOR"],  "NOR (Control)"),
]

# Superset of all clinical variables
ALL_NUMERIC = [
    "age", "bmi", "bp_systolic", "bp_diastolic",
    "wbc", "rbc", "hb", "hct", "platelet",
    "neutrophil_pct", "lymphocyte_pct",
    "ast", "alt", "alp", "ggt",
    "bun", "creatinine", "uric_acid", "glucose",
    "total_bilirubin", "calcium", "total_cholesterol", "triglyceride",
    "hba1c", "afp", "cea", "ca19_9", "psa",
    "ua_sg", "ua_ph", "potassium", "chloride",
    "total_protein", "albumin", "ldh", "hs_crp", "sodium", "hdl_c", "ldl_c",
]
ALL_CATEGORICAL = [
    "sex", "smoking_status", "drinking_status",
    "stage", "t_stage", "sample_timing",
    "pathology_group",
    "ua_protein", "ua_blood", "ua_glucose",
]

VAR_LABELS = {
    "age": "Age", "bmi": "BMI",
    "bp_systolic": "Systolic BP", "bp_diastolic": "Diastolic BP",
    "wbc": "WBC", "rbc": "RBC", "hb": "Hemoglobin", "hct": "Hematocrit",
    "platelet": "Platelet", "neutrophil_pct": "Neutrophil %",
    "lymphocyte_pct": "Lymphocyte %",
    "ast": "AST", "alt": "ALT", "alp": "ALP", "ggt": "GGT",
    "bun": "BUN", "creatinine": "Creatinine", "uric_acid": "Uric Acid",
    "glucose": "Glucose", "total_bilirubin": "Bilirubin",
    "calcium": "Calcium", "total_cholesterol": "Cholesterol",
    "triglyceride": "Triglyceride", "hba1c": "HbA1c",
    "afp": "AFP", "cea": "CEA", "ca19_9": "CA19-9", "psa": "PSA",
    "ua_sg": "Urine SG", "ua_ph": "Urine pH",
    "potassium": "K", "chloride": "Cl",
    "total_protein": "Total Protein", "albumin": "Albumin",
    "ldh": "LDH", "hs_crp": "hs-CRP", "sodium": "Na",
    "hdl_c": "HDL-C", "ldl_c": "LDL-C",
    "sex": "Sex", "smoking_status": "Smoking",
    "drinking_status": "Drinking", "stage": "Stage",
    "t_stage": "T-Stage", "sample_timing": "Sample Timing",
    "pathology_group": "Pathology",
    "ua_protein": "Urine Protein", "ua_blood": "Urine Blood",
    "ua_glucose": "Urine Glucose",
    "cluster": "Cluster", "subgroup": "Subgroup",
}

# ── ID mapping (reuse logic from models/clinical_utils.py) ────────────────
def build_id_mappings():
    """Build solum_label → clinical patient_id for non-standard groups."""
    mapping = {}

    # BRE: hospital ID from raw Excel
    bre_path = ROOT / "data" / "clinical_data" / "2. 유방암" / "SMCXD01_유방암.xlsx"
    if bre_path.exists():
        try:
            xl = pd.read_excel(bre_path, header=None, skiprows=2)
            m = xl[[1, 4]].dropna()
            m.columns = ["hospital_id", "solum_label"]
            for _, row in m.iterrows():
                label = str(row["solum_label"]).strip()
                hid = str(int(row["hospital_id"]))
                mapping[("BRE", label)] = hid
        except Exception:
            pass

    # OVA: from two Excel files
    for fpath, cols, skip in [
        (ROOT / "data" / "clinical_data" / "3. 난소암" / "SMCXD01_난소암 1.xlsx", [1, 4], 2),
        (ROOT / "data" / "clinical_data" / "3. 난소암" / "SMCXD01_난소암 2.xlsx", [0, 3], 1),
    ]:
        if fpath.exists():
            try:
                xl = pd.read_excel(fpath, header=None, skiprows=skip)
                m = xl[cols].dropna()
                m.columns = ["hospital_id", "solum_label"] if skip == 2 else ["solum_label", "hospital_id"]
                for _, row in m.iterrows():
                    label = str(row["solum_label"]).strip()
                    hid = str(int(row["hospital_id"]))
                    mapping[("OVA", label)] = hid
            except Exception:
                pass

    # BLC: sequential numbering from staging Excel
    blc_path = ROOT / "data" / "clinical_data" / "11. 방광암" / "SMCXD06_방광암.xlsm"
    if blc_path.exists():
        try:
            xl = pd.read_excel(blc_path, header=None, skiprows=2)
            m = xl[[0, 1]].dropna()
            for i, (_, row) in enumerate(m.iterrows(), start=1):
                mapping[("BLC", str(i))] = str(row.iloc[0])
        except Exception:
            pass

    return mapping


def simplify_pathology(p):
    if pd.isna(p):
        return None
    p = str(p).upper().strip().rstrip(",")
    if "NONINVASIVE" in p or "NON-INVASIVE" in p:
        return "Non-invasive"
    if "PAPILLARY" in p and "UROTHELIAL" in p:
        return "Papillary UC"
    if "UROTHELIAL" in p:
        return "UC"
    if "ADENOCARCINOMA" in p:
        return "Adenocarcinoma"
    if "SQUAMOUS" in p:
        return "Squamous"
    if "SMALL CELL" in p:
        return "Small Cell"
    if "MUCINOUS" in p:
        return "Mucinous"
    if "SEROUS" in p:
        return "Serous"
    if "CLEAR CELL" in p:
        return "Clear Cell"
    if len(p) > 40:
        return p[:35] + "..."
    return p


def simplify_t_stage(t):
    if pd.isna(t):
        return None
    t = str(t).upper().strip()
    if t in ("TX", "UNKNOWN"):
        return None
    if t.startswith("TA"):
        return "Ta"
    for prefix in ["T1", "T2", "T3", "T4"]:
        if t.startswith(prefix):
            return prefix
    return t


# ── Per-group processing ──────────────────────────────────────────────────
def process_group(bc_file, clin_grp, spec_groups, display_name, spectra_df, id_mappings):
    """Process one disease group: link, embed, cluster, return records."""
    from umap import UMAP
    from sklearn.manifold import TSNE
    from hdbscan import HDBSCAN

    feat_cols = [c for c in spectra_df.columns if c.startswith("x_")]

    # Filter spectra
    blc = spectra_df[spectra_df["group"].isin(spec_groups)].copy()
    if len(blc) == 0:
        print(f"  {display_name}: no spectra found, skipping")
        return None

    # Subject-level aggregation
    blc_agg = blc.groupby("sample_id")[feat_cols].mean().reset_index()
    blc_agg["spec_group"] = blc.groupby("sample_id")["group"].first().values

    # Load clinical
    bc_path = CLINICAL_BY_CANCER / f"{bc_file}.xlsx"
    clin = pd.read_excel(bc_path, sheet_name="Clean", engine="openpyxl")

    # Build patient_id → row lookup
    clin_lookup = {}
    for _, row in clin.iterrows():
        pid = str(row["patient_id"])
        clin_lookup[pid] = row

    # Match spectra to clinical
    merged_rows = []
    for _, spec_row in blc_agg.iterrows():
        sid = int(spec_row["sample_id"])
        sg = spec_row["spec_group"]

        # Strategy 1: "{GROUP} {N}" pattern
        matched_clin = None
        for key_fmt in [f"{sg} {sid}", f"{clin_grp} {sid}"]:
            if key_fmt in clin_lookup:
                matched_clin = clin_lookup[key_fmt]
                break

        # Strategy 2: numeric tail match
        if matched_clin is None:
            for pid, crow in clin_lookup.items():
                m = re.search(r"(\d+)$", str(pid))
                if m and int(m.group(1)) == sid:
                    matched_clin = crow
                    break

        # Strategy 3: ID mapping (BRE/OVA hospital IDs)
        if matched_clin is None:
            solum_label = f"{sg} {sid}"
            mapped_id = id_mappings.get((sg, solum_label)) or id_mappings.get((sg, str(sid)))
            if mapped_id and mapped_id in clin_lookup:
                matched_clin = clin_lookup[mapped_id]

        if matched_clin is not None:
            combined = {**spec_row.to_dict(), **matched_clin.to_dict()}
            combined["sample_id"] = sid
            merged_rows.append(combined)

    if len(merged_rows) < 10:
        print(f"  {display_name}: only {len(merged_rows)} matched, skipping")
        return None

    df = pd.DataFrame(merged_rows)
    df["pathology_group"] = df.get("pathology", pd.Series()).apply(simplify_pathology)
    df["t_stage"] = df.get("t_stage", pd.Series()).apply(simplify_t_stage)
    print(f"  {display_name}: {len(df)} subjects linked")

    # Embeddings
    X = df[feat_cols].values

    reducer = UMAP(n_components=2, n_neighbors=min(20, len(df) - 1),
                   min_dist=0.1, metric="euclidean", random_state=42)
    emb_umap = reducer.fit_transform(X)
    df["umap_1"] = emb_umap[:, 0]
    df["umap_2"] = emb_umap[:, 1]

    perp = min(30, max(5, len(df) // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                learning_rate="auto", init="pca")
    emb_tsne = tsne.fit_transform(X)
    df["tsne_1"] = emb_tsne[:, 0]
    df["tsne_2"] = emb_tsne[:, 1]

    # HDBSCAN
    mcs = max(8, len(df) // 15)
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3).fit_predict(emb_umap)
    df["cluster"] = labels
    n_cl = len(set(labels) - {-1})
    print(f"    {n_cl} clusters, {(labels == -1).sum()} noise")

    # Determine which variables have data
    avail_numeric = []
    for v in ALL_NUMERIC:
        if v in df.columns and df[v].notna().mean() > 0.25:
            avail_numeric.append([v, VAR_LABELS.get(v, v)])
    avail_categorical = [["cluster", "Cluster"]]
    for v in ALL_CATEGORICAL:
        if v in df.columns and df[v].notna().mean() > 0.25 and df[v].dropna().nunique() >= 2:
            avail_categorical.append([v, VAR_LABELS.get(v, v)])

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
        if "spec_group" in row and pd.notna(row["spec_group"]):
            rec["subgroup"] = str(row["spec_group"])
        for v, _ in avail_numeric:
            rec[v] = round(float(row[v]), 2) if pd.notna(row.get(v)) else None
        for v, _ in avail_categorical:
            if v == "cluster":
                continue
            val = row.get(v)
            if pd.notna(val):
                rec[v] = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
            else:
                rec[v] = None
        records.append(rec)

    return {
        "key": spec_groups[0] if len(spec_groups) == 1 else bc_file.upper(),
        "display_name": display_name,
        "n_subjects": len(records),
        "n_clusters": n_cl,
        "records": records,
        "numeric_vars": avail_numeric,
        "categorical_vars": avail_categorical,
    }


# ── HTML generation ───────────────────────────────────────────────────────
def build_html(all_groups):
    """Generate single-page HTML with tab navigation across groups."""
    groups_json = json.dumps(all_groups)

    html = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SERS Subgroup Explorer — All Groups</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0}

.header{background:linear-gradient(135deg,#1e293b,#0f172a);border-bottom:1px solid #334155;padding:14px 24px;display:flex;align-items:center;gap:16px}
.header h1{font-size:18px;font-weight:700;color:#f8fafc}

.tabs{display:flex;gap:2px;padding:0 24px;background:#1e293b;border-bottom:1px solid #334155;overflow-x:auto}
.tab{padding:10px 18px;font-size:13px;font-weight:600;color:#94a3b8;cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:all .2s}
.tab:hover{color:#e2e8f0;background:#334155}
.tab.active{color:#a78bfa;border-bottom-color:#7c3aed}
.tab .badge{background:#334155;color:#94a3b8;padding:1px 7px;border-radius:10px;font-size:11px;margin-left:5px}
.tab.active .badge{background:#7c3aed33;color:#a78bfa}

.controls{display:flex;gap:12px;padding:10px 24px;background:#1e293b;border-bottom:1px solid #334155;flex-wrap:wrap;align-items:center}
.cg{display:flex;align-items:center;gap:5px}
.cg label{font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:.5px}
select,input{background:#0f172a;border:1px solid #475569;color:#e2e8f0;padding:5px 8px;border-radius:5px;font-size:12px;outline:none}
select:focus,input:focus{border-color:#7c3aed}
input::placeholder{color:#64748b}

.main{display:flex;height:calc(100vh - 140px)}
.plot-area{flex:1;position:relative}
#scatter{width:100%;height:100%}

.sidebar{width:300px;background:#1e293b;border-left:1px solid #334155;overflow-y:auto}
.sidebar-header{padding:12px 14px;background:#334155;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8;position:sticky;top:0;z-index:1}
.pcard{padding:10px 14px;border-bottom:1px solid #0f172a;cursor:pointer;transition:background .15s}
.pcard:hover{background:#334155}
.pcard.active{background:#7c3aed22;border-left:3px solid #7c3aed}
.pcard .pid{font-size:14px;font-weight:700;color:#f8fafc}
.pcard .ctag{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:600;margin-left:5px}
.pcard .meta{font-size:11px;color:#94a3b8;margin-top:3px}

.detail{position:fixed;right:300px;top:140px;width:360px;max-height:calc(100vh - 160px);background:#1e293b;border:1px solid #475569;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.5);overflow-y:auto;display:none;z-index:100}
.detail.show{display:block}
.detail-hd{padding:14px;background:#334155;border-radius:12px 12px 0 0;display:flex;justify-content:space-between;align-items:center}
.detail-hd h3{font-size:15px;font-weight:700}
.detail-close{cursor:pointer;color:#94a3b8;font-size:20px}
.detail-close:hover{color:#f8fafc}
.detail-body{padding:14px}
.dsec{margin-bottom:12px}
.dsec h4{font-size:10px;text-transform:uppercase;color:#7c3aed;letter-spacing:.5px;margin-bottom:5px;font-weight:700}
.drow{display:flex;justify-content:space-between;padding:2px 0;font-size:12px}
.drow .lb{color:#94a3b8}
.drow .vl{color:#f8fafc;font-weight:600}

.legend{position:absolute;bottom:14px;left:14px;background:#1e293bdd;backdrop-filter:blur(8px);border:1px solid #475569;border-radius:10px;padding:8px 12px;font-size:11px;max-height:180px;overflow-y:auto}
.legend-title{font-weight:700;margin-bottom:4px;color:#94a3b8;text-transform:uppercase;font-size:10px}
.litem{display:flex;align-items:center;gap:5px;padding:1px 0}
.ldot{width:9px;height:9px;border-radius:50%;flex-shrink:0}

.stats-bar{position:absolute;top:10px;left:10px;display:flex;gap:6px}
.schip{background:#1e293bdd;backdrop-filter:blur(8px);border:1px solid #475569;border-radius:7px;padding:5px 10px;font-size:11px}
.schip .n{font-weight:700;color:#7c3aed}

.cbar{position:absolute;bottom:14px;right:320px;background:#1e293bdd;backdrop-filter:blur(8px);border:1px solid #475569;border-radius:10px;padding:8px 12px;display:none}
</style>
</head>
<body>

<div class="header">
  <h1>SERS Spectral Subgroup Explorer</h1>
</div>
<div class="tabs" id="tabs"></div>
<div class="controls">
  <div class="cg"><label>Embedding</label><select id="embSel"><option value="umap" selected>UMAP</option><option value="tsne">t-SNE</option></select></div>
  <div class="cg"><label>Color by</label><select id="colorSel"></select></div>
  <div class="cg"><label>Search</label><input id="searchIn" type="text" placeholder="Sample ID" style="width:120px"></div>
  <div class="cg"><label>Size</label><input id="sizeSl" type="range" min="3" max="18" value="8" style="width:80px"></div>
</div>

<div class="main">
  <div class="plot-area">
    <canvas id="scatter"></canvas>
    <div class="stats-bar" id="statsBar"></div>
    <div class="legend" id="legend"></div>
    <div class="cbar" id="cbar">
      <div id="cbMax" style="text-align:center;font-size:10px;color:#94a3b8;margin-bottom:3px"></div>
      <canvas id="cbG" width="18" height="120" style="border-radius:3px"></canvas>
      <div id="cbMin" style="text-align:center;font-size:10px;color:#94a3b8;margin-top:3px"></div>
    </div>
  </div>
  <div class="sidebar" id="sidebar">
    <div class="sidebar-header">Patients <span id="listCnt"></span></div>
    <div id="pList"></div>
  </div>
</div>

<div class="detail" id="detailP">
  <div class="detail-hd"><h3 id="dTitle">Patient</h3><span class="detail-close" id="dClose">&times;</span></div>
  <div class="detail-body" id="dBody"></div>
</div>

<script>
const GROUPS = """ + groups_json + """;

const CC = ['#EF4444','#3B82F6','#22C55E','#F97316','#A855F7','#06B6D4','#EC4899','#EAB308','#84CC16','#F43F5E'];
const NC = '#6B7280';
const VIR = [[68,1,84],[72,26,108],[71,47,125],[65,68,135],[57,86,140],[48,103,141],[40,120,142],[33,137,141],[26,153,136],[30,169,120],[53,183,95],[94,196,60],[143,205,26],[194,211,30],[241,229,29],[253,231,37]];
function vir(t){t=Math.max(0,Math.min(1,t));const i=t*(VIR.length-1),l=Math.floor(i),h=Math.ceil(i),f=i-l;const c=VIR[l].map((v,j)=>Math.round(v+f*(VIR[h][j]-v)));return`rgb(${c[0]},${c[1]},${c[2]})`}

const LABELS = """ + json.dumps(VAR_LABELS) + """;
function lb(v){return LABELS[v]||v}

let gIdx=0, emb='umap', colorBy='cluster', hlId=null, selId=null, ptSize=8;
let G, D, numV, catV;

function setGroup(i){
  gIdx=i; G=GROUPS[i]; D=G.records; numV=G.numeric_vars; catV=G.categorical_vars;
  document.querySelectorAll('.tab').forEach((t,j)=>t.classList.toggle('active',j===i));
  hlId=null; selId=null;
  document.getElementById('detailP').classList.remove('show');
  buildColorSel();
  buildList();
  computeXf();
  draw();
}

// Tabs
const tabsEl=document.getElementById('tabs');
GROUPS.forEach((g,i)=>{
  const t=document.createElement('div');
  t.className='tab'+(i===0?' active':'');
  t.innerHTML=`${g.display_name}<span class="badge">${g.n_subjects}</span>`;
  t.onclick=()=>setGroup(i);
  tabsEl.appendChild(t);
});

// Color selector
function buildColorSel(){
  const s=document.getElementById('colorSel');
  s.innerHTML='';
  const og1=document.createElement('optgroup');og1.label='Categorical';
  catV.forEach(([k,v])=>{const o=document.createElement('option');o.value=k;o.textContent=v;og1.appendChild(o)});
  s.appendChild(og1);
  if(numV.length){
    const og2=document.createElement('optgroup');og2.label='Numeric';
    numV.forEach(([k,v])=>{const o=document.createElement('option');o.value=k;o.textContent=v;og2.appendChild(o)});
    s.appendChild(og2);
  }
  colorBy='cluster';s.value='cluster';
}

// Canvas
const canvas=document.getElementById('scatter'), ctx=canvas.getContext('2d');
let W,H,xf;

function resize(){
  const r=canvas.parentElement.getBoundingClientRect();
  W=r.width;H=r.height;
  canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;
  canvas.style.width=W+'px';canvas.style.height=H+'px';
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  computeXf();draw();
}

function computeXf(){
  const xk=emb==='umap'?'umap_1':'tsne_1', yk=emb==='umap'?'umap_2':'tsne_2';
  const xs=D.map(d=>d[xk]),ys=D.map(d=>d[yk]);
  const p=50,xMn=Math.min(...xs),xMx=Math.max(...xs),yMn=Math.min(...ys),yMx=Math.max(...ys);
  const xR=xMx-xMn||1,yR=yMx-yMn||1;
  const sc=Math.min((W-2*p)/xR,(H-2*p)/yR);
  xf={xMn,yMn,sc,ox:(W-xR*sc)/2,oy:(H-yR*sc)/2};
}

function toS(d){
  const xk=emb==='umap'?'umap_1':'tsne_1', yk=emb==='umap'?'umap_2':'tsne_2';
  return{x:(d[xk]-xf.xMn)*xf.sc+xf.ox, y:H-((d[yk]-xf.yMn)*xf.sc+xf.oy)};
}

function isNum(v){return numV.some(([k])=>k===v)}

function getCol(d){
  const v=colorBy;
  if(v==='cluster'){return d.cluster<0?NC:CC[d.cluster%CC.length]}
  if(v==='subgroup'&&d.subgroup){
    const cats=[...new Set(D.map(r=>r.subgroup).filter(Boolean))].sort();
    const i=cats.indexOf(d.subgroup);return i<0?'#374151':CC[i%CC.length];
  }
  if(isNum(v)){
    const val=d[v];if(val==null)return'#374151';
    const vs=D.map(r=>r[v]).filter(x=>x!=null);
    const lo=Math.min(...vs),hi=Math.max(...vs);
    return vir(hi>lo?(val-lo)/(hi-lo):.5);
  }
  const cats=[...new Set(D.map(r=>r[v]).filter(x=>x!=null))].sort();
  const i=cats.indexOf(d[v]);return i<0?'#374151':CC[i%CC.length];
}

function draw(){
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#1e293b';ctx.lineWidth=1;
  for(let x=0;x<W;x+=60){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}
  for(let y=0;y<H;y+=60){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}

  const sorted=[...D].sort((a,b)=>{
    if(a.id===selId)return 1;if(b.id===selId)return-1;
    if(a.id===hlId)return 1;if(b.id===hlId)return-1;return 0;
  });

  sorted.forEach(d=>{
    const{x,y}=toS(d);
    const isH=d.id===hlId,isS=d.id===selId;
    const r=isH||isS?ptSize*1.8:ptSize;
    ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fillStyle=getCol(d);
    ctx.globalAlpha=(hlId&&!isH&&!isS)?.12:.8;
    ctx.fill();
    if(isH||isS){
      ctx.strokeStyle='#fbbf24';ctx.lineWidth=3;ctx.stroke();
      ctx.globalAlpha=1;ctx.fillStyle='#fbbf24';ctx.font='bold 12px system-ui';
      ctx.fillText('#'+d.id,x+r+4,y-r);
    }
    ctx.globalAlpha=1;
  });
  updLegend();updStats();
}

function updLegend(){
  const leg=document.getElementById('legend'),cb=document.getElementById('cbar'),v=colorBy;
  if(isNum(v)){
    leg.style.display='none';cb.style.display='block';
    const vs=D.map(r=>r[v]).filter(x=>x!=null);
    document.getElementById('cbMax').textContent=Math.max(...vs).toFixed(1);
    document.getElementById('cbMin').textContent=Math.min(...vs).toFixed(1);
    const c2=document.getElementById('cbG').getContext('2d');
    for(let i=0;i<120;i++){c2.fillStyle=vir(1-i/119);c2.fillRect(0,i,18,1)}
    return;
  }
  cb.style.display='none';leg.style.display='block';
  if(v==='cluster'){
    const cls=[...new Set(D.map(d=>d.cluster))].sort((a,b)=>a-b);
    leg.innerHTML='<div class="legend-title">Cluster</div>'+cls.map(c=>{
      const col=c<0?NC:CC[c%CC.length];const n=D.filter(d=>d.cluster===c).length;
      return`<div class="litem"><span class="ldot" style="background:${col}"></span>${c<0?'Noise':('C'+c)} (${n})</div>`;
    }).join('');
  } else {
    const cats=[...new Set(D.map(d=>d[v]).filter(x=>x!=null))].sort();
    leg.innerHTML=`<div class="legend-title">${lb(v)}</div>`+cats.map((c,i)=>{
      const n=D.filter(d=>d[v]===c).length;
      return`<div class="litem"><span class="ldot" style="background:${CC[i%CC.length]}"></span>${c} (${n})</div>`;
    }).join('');
  }
}

function updStats(){
  const nc=new Set(D.filter(d=>d.cluster>=0).map(d=>d.cluster)).size;
  const nn=D.filter(d=>d.cluster<0).length;
  document.getElementById('statsBar').innerHTML=
    `<div class="schip"><span class="n">${D.length}</span> subjects</div>`+
    `<div class="schip"><span class="n">${nc}</span> clusters</div>`+
    `<div class="schip"><span class="n">${nn}</span> noise</div>`+
    `<div class="schip">${emb.toUpperCase()}</div>`;
}

// Patient list
function buildList(filter){
  const el=document.getElementById('pList');
  let items=D;
  if(filter){const q=filter.toLowerCase();items=items.filter(d=>String(d.id).includes(q))}
  items=[...items].sort((a,b)=>a.id-b.id);
  document.getElementById('listCnt').textContent='('+items.length+')';
  el.innerHTML=items.map(d=>{
    const clr=d.cluster<0?NC:CC[d.cluster%CC.length];
    const tag=d.cluster<0?'Noise':'C'+d.cluster;
    const meta=[d.sex,d.age?d.age+'y':'',d.t_stage||d.stage||'',d.subgroup||''].filter(Boolean).join(' · ');
    return`<div class="pcard" data-id="${d.id}" onmouseenter="hov(${d.id})" onmouseleave="unhov()" onclick="sel(${d.id})">
      <span class="pid">#${d.id}</span><span class="ctag" style="background:${clr}33;color:${clr}">${tag}</span>
      <div class="meta">${meta}</div></div>`;
  }).join('');
}

function hov(id){hlId=id;draw()}
function unhov(){hlId=null;draw()}
function sel(id){
  selId=id;hlId=id;draw();showDetail(id);
  document.querySelectorAll('.pcard').forEach(e=>e.classList.toggle('active',+e.dataset.id===id));
}

function showDetail(id){
  const d=D.find(r=>r.id===id);if(!d)return;
  const p=document.getElementById('detailP');
  const clr=d.cluster<0?NC:CC[d.cluster%CC.length];
  document.getElementById('dTitle').innerHTML=
    `#${d.id} <span class="ctag" style="background:${clr}33;color:${clr}">${d.cluster<0?'Noise':'C'+d.cluster}</span>`;

  // Build sections dynamically from available vars
  let html='';
  const sections=[
    ['Demographics', ['age','sex','bmi','bp_systolic','bp_diastolic','smoking_status','drinking_status']],
    ['Cancer', ['stage','t_stage','pathology_group','sample_timing','subgroup']],
    ['CBC', ['wbc','rbc','hb','hct','platelet','neutrophil_pct','lymphocyte_pct']],
    ['Chemistry', ['ast','alt','alp','ggt','bun','creatinine','uric_acid','glucose','total_bilirubin','calcium','total_cholesterol','triglyceride','total_protein','albumin','ldh','hs_crp','sodium','potassium','chloride','hdl_c','ldl_c','hba1c']],
    ['Tumor Markers', ['afp','cea','ca19_9','psa']],
    ['Urinalysis', ['ua_sg','ua_ph','ua_protein','ua_blood','ua_glucose']],
  ];
  sections.forEach(([title, vars])=>{
    const rows=vars.filter(v=>d[v]!=null).map(v=>
      `<div class="drow"><span class="lb">${lb(v)}</span><span class="vl">${d[v]}</span></div>`
    );
    if(rows.length){html+=`<div class="dsec"><h4>${title}</h4>${rows.join('')}</div>`}
  });

  html+=`<div class="dsec"><h4>Embedding</h4>
    <div class="drow"><span class="lb">UMAP</span><span class="vl">(${d.umap_1.toFixed(2)}, ${d.umap_2.toFixed(2)})</span></div>
    <div class="drow"><span class="lb">t-SNE</span><span class="vl">(${d.tsne_1.toFixed(2)}, ${d.tsne_2.toFixed(2)})</span></div></div>`;

  document.getElementById('dBody').innerHTML=html;
  p.classList.add('show');
}

document.getElementById('dClose').onclick=()=>{
  document.getElementById('detailP').classList.remove('show');selId=null;draw();
};

// Canvas interactions
canvas.addEventListener('mousemove',e=>{
  const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=null,md=Infinity;
  D.forEach(d=>{const{x,y}=toS(d);const dist=Math.hypot(mx-x,my-y);if(dist<md){md=dist;best=d}});
  if(md<18&&best){canvas.style.cursor='pointer';if(hlId!==best.id){hlId=best.id;draw()}}
  else{canvas.style.cursor='default';if(hlId&&!selId){hlId=null;draw()}}
});
canvas.addEventListener('click',e=>{
  const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=null,md=Infinity;
  D.forEach(d=>{const{x,y}=toS(d);const dist=Math.hypot(mx-x,my-y);if(dist<md){md=dist;best=d}});
  if(md<18&&best)sel(best.id);
});

// Controls
document.getElementById('embSel').onchange=e=>{emb=e.target.value;computeXf();draw()};
document.getElementById('colorSel').onchange=e=>{colorBy=e.target.value;draw()};
document.getElementById('searchIn').oninput=e=>{
  const q=e.target.value.trim();buildList(q);
  if(q&&D.some(d=>String(d.id)===q)){sel(parseInt(q))}
  else{hlId=null;selId=null;document.getElementById('detailP').classList.remove('show');draw()}
};
document.getElementById('sizeSl').oninput=e=>{ptSize=+e.target.value;draw()};

// Init
window.addEventListener('resize',resize);
setGroup(0);
resize();
</script>
</body>
</html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"\nHTML saved to {OUTPUT_HTML}")
    print(f"File size: {OUTPUT_HTML.stat().st_size / 1024:.0f} KB")


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("All-Group Subgroup Interactive Explorer")
    print("=" * 60)

    spectra_df = pd.read_csv(SPECTRA_PATH)
    print(f"Loaded spectra: {len(spectra_df)} rows")

    id_mappings = build_id_mappings()
    print(f"ID mappings: {len(id_mappings)} entries\n")

    all_groups = []
    for bc_file, clin_grp, spec_groups, display_name in GROUP_DEFS:
        result = process_group(bc_file, clin_grp, spec_groups, display_name,
                               spectra_df, id_mappings)
        if result:
            all_groups.append(result)

    # Also add non-cancer disease groups from standardized data
    # DIA, HBP, H.D. — use standardized CSV directly
    std_clin = pd.read_csv(CLINICAL_STD)
    for spec_grp, clin_grps, display_name in [
        ("DIA", ["DIA"], "DIA (Diabetes)"),
        ("HBP", ["HBP"], "HBP (Hypertension)"),
        ("H.D.", ["H.D."], "H.D. (DIA+HBP)"),
    ]:
        result = process_noncancer_group(spec_grp, clin_grps, display_name,
                                         spectra_df, std_clin)
        if result:
            all_groups.append(result)

    print(f"\n{'='*60}")
    print(f"Processed {len(all_groups)} groups:")
    for g in all_groups:
        print(f"  {g['display_name']:25s} — {g['n_subjects']:4d} subjects, {g['n_clusters']} clusters")

    build_html(all_groups)


def process_noncancer_group(spec_grp, clin_grps, display_name, spectra_df, std_clin):
    """Process non-cancer groups using standardized clinical CSV."""
    from umap import UMAP
    from sklearn.manifold import TSNE
    from hdbscan import HDBSCAN

    feat_cols = [c for c in spectra_df.columns if c.startswith("x_")]

    blc = spectra_df[spectra_df["group"] == spec_grp].copy()
    if len(blc) == 0:
        return None

    blc_agg = blc.groupby("sample_id")[feat_cols].mean().reset_index()

    # Match: "{disease_group} {sample_id}" format in standardized
    clin_sub = std_clin[std_clin["disease_group"].isin(clin_grps)].copy()
    clin_sub["sex_display"] = clin_sub["sex"]

    # Extract numeric from patient_id
    clin_lookup = {}
    for _, row in clin_sub.iterrows():
        pid = str(row["patient_id"])
        m = re.search(r"(\d+)$", pid)
        if m:
            clin_lookup[int(m.group(1))] = row

    merged_rows = []
    for _, spec_row in blc_agg.iterrows():
        sid = int(spec_row["sample_id"])
        if sid in clin_lookup:
            crow = clin_lookup[sid]
            combined = {**spec_row.to_dict(), **crow.to_dict()}
            combined["sample_id"] = sid
            merged_rows.append(combined)

    if len(merged_rows) < 10:
        print(f"  {display_name}: only {len(merged_rows)} matched, skipping")
        return None

    df = pd.DataFrame(merged_rows)
    df["pathology_group"] = df.get("pathology", pd.Series()).apply(simplify_pathology)
    print(f"  {display_name}: {len(df)} subjects linked")

    X = df[feat_cols].values

    reducer = UMAP(n_components=2, n_neighbors=min(20, len(df) - 1),
                   min_dist=0.1, metric="euclidean", random_state=42)
    emb_umap = reducer.fit_transform(X)
    df["umap_1"] = emb_umap[:, 0]
    df["umap_2"] = emb_umap[:, 1]

    perp = min(30, max(5, len(df) // 5))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42,
                learning_rate="auto", init="pca")
    emb_tsne = tsne.fit_transform(X)
    df["tsne_1"] = emb_tsne[:, 0]
    df["tsne_2"] = emb_tsne[:, 1]

    mcs = max(8, len(df) // 15)
    labels = HDBSCAN(min_cluster_size=mcs, min_samples=3).fit_predict(emb_umap)
    df["cluster"] = labels
    n_cl = len(set(labels) - {-1})
    print(f"    {n_cl} clusters, {(labels == -1).sum()} noise")

    avail_numeric = []
    for v in ALL_NUMERIC:
        if v in df.columns and df[v].notna().mean() > 0.25:
            avail_numeric.append([v, VAR_LABELS.get(v, v)])
    avail_categorical = [["cluster", "Cluster"]]
    for v in ALL_CATEGORICAL:
        if v in df.columns and df[v].notna().mean() > 0.25 and df[v].dropna().nunique() >= 2:
            avail_categorical.append([v, VAR_LABELS.get(v, v)])

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
        for v, _ in avail_numeric:
            rec[v] = round(float(row[v]), 2) if pd.notna(row.get(v)) else None
        for v, _ in avail_categorical:
            if v == "cluster":
                continue
            val = row.get(v)
            if pd.notna(val):
                rec[v] = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
            else:
                rec[v] = None
        records.append(rec)

    return {
        "key": spec_grp,
        "display_name": display_name,
        "n_subjects": len(records),
        "n_clusters": n_cl,
        "records": records,
        "numeric_vars": avail_numeric,
        "categorical_vars": avail_categorical,
    }


if __name__ == "__main__":
    main()
