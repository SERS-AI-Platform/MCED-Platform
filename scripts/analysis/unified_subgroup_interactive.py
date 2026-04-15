"""
Unified Subgroup Interactive HTML — All patients on one UMAP/t-SNE.

All disease groups embedded together. Color by group, cluster, or any clinical variable.

Usage:
    python scripts/analysis/unified_subgroup_interactive.py
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
CLINICAL_BY_CANCER = ROOT / "data" / "clinical_data" / "by_cancer"
CLINICAL_STD = ROOT / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv"
OUTPUT_DIR = ROOT / "results" / "subgroup_analysis"
OUTPUT_HTML = OUTPUT_DIR / "unified_explorer.html"

# ── Clinical variable superset ────────────────────────────────────────────
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
    "group", "category", "cluster", "intra_cluster",
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
    "group": "Disease Group", "category": "Category", "cluster": "Global Cluster",
    "intra_cluster": "Intra-group Cluster",
}

# Group display config
GROUP_ORDER = ["NOR", "YNOR", "DIA", "HBP", "H.D.", "PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
GROUP_COLORS = {
    "PRO": "#E91E63", "BRE": "#FF69B4", "OVA": "#AB47BC",
    "LUN": "#42A5F5", "CRC": "#EF5350", "PAN": "#FFA726",
    "BLC": "#7E57C2",
    "DIA": "#66BB6A", "HBP": "#26A69A", "H.D.": "#78909C",
    "NOR": "#8D6E63", "YNOR": "#A1887F",
}
CATEGORY_MAP = {
    "PRO": "cancer", "BRE": "cancer", "OVA": "cancer",
    "LUN": "cancer", "CRC": "cancer", "PAN": "cancer", "BLC": "cancer",
    "DIA": "non-cancer", "HBP": "non-cancer", "H.D.": "non-cancer",
    "NOR": "control", "YNOR": "control",
}
CATEGORY_COLORS = {"cancer": "#E53935", "non-cancer": "#43A047", "control": "#5C6BC0"}


# ── ID mapping (from models/clinical_utils.py) ───────────────────────────
def build_id_mappings():
    mapping = {}
    # BRE
    p = ROOT / "data" / "clinical_data" / "2. 유방암" / "SMCXD01_유방암.xlsx"
    if p.exists():
        try:
            xl = pd.read_excel(p, header=None, skiprows=2)
            m = xl[[1, 4]].dropna(); m.columns = ["hid", "label"]
            for _, r in m.iterrows():
                mapping[("BRE", str(r["label"]).strip())] = str(int(r["hid"]))
        except Exception:
            pass
    # OVA
    for fp, cols, skip in [
        (ROOT / "data" / "clinical_data" / "3. 난소암" / "SMCXD01_난소암 1.xlsx", [1, 4], 2),
        (ROOT / "data" / "clinical_data" / "3. 난소암" / "SMCXD01_난소암 2.xlsx", [0, 3], 1),
    ]:
        if fp.exists():
            try:
                xl = pd.read_excel(fp, header=None, skiprows=skip)
                m = xl[cols].dropna()
                m.columns = ["hid", "label"] if skip == 2 else ["label", "hid"]
                for _, r in m.iterrows():
                    mapping[("OVA", str(r["label"]).strip())] = str(int(r["hid"]))
            except Exception:
                pass
    # BLC
    p = ROOT / "data" / "clinical_data" / "11. 방광암" / "SMCXD06_방광암.xlsm"
    if p.exists():
        try:
            xl = pd.read_excel(p, header=None, skiprows=2)
            m = xl[[0, 1]].dropna()
            for i, (_, r) in enumerate(m.iterrows(), start=1):
                mapping[("BLC", str(i))] = str(r.iloc[0])
        except Exception:
            pass
    return mapping


def simplify_pathology(p):
    if pd.isna(p): return None
    p = str(p).upper().strip().rstrip(",")
    if "NONINVASIVE" in p or "NON-INVASIVE" in p: return "Non-invasive"
    if "PAPILLARY" in p and "UROTHELIAL" in p: return "Papillary UC"
    if "UROTHELIAL" in p: return "UC"
    if "ADENOCARCINOMA" in p: return "Adenocarcinoma"
    if "SQUAMOUS" in p: return "Squamous"
    if "SMALL CELL" in p: return "Small Cell"
    if "MUCINOUS" in p: return "Mucinous"
    if "SEROUS" in p: return "Serous"
    if "CLEAR CELL" in p: return "Clear Cell"
    if len(p) > 35: return p[:30] + "..."
    return p


def simplify_t_stage(t):
    if pd.isna(t): return None
    t = str(t).upper().strip()
    if t in ("TX", "UNKNOWN"): return None
    for pf in ["T1", "T2", "T3", "T4"]:
        if t.startswith(pf): return pf
    return t


# ── Load & link all data ─────────────────────────────────────────────────
def load_all():
    spectra = pd.read_csv(SPECTRA_PATH)
    feat_cols = [c for c in spectra.columns if c.startswith("x_")]

    # by_cancer clinical files
    BC_DEFS = [
        ("bladder",    "BLA", ["BLC"]),
        ("colorectal", "CRC", ["CRC"]),
        ("lung",       "LUN", ["LUN"]),
        ("prostate",   "PRO", ["PRO"]),
        ("pancreatic", "PAN", ["CPAN", "YPAN"]),
        ("ovarian",    "OVA", ["OVA"]),
        ("breast",     "BRE", ["BRE"]),
        ("controls",   "NOR", ["NOR"]),  # YNOR handled separately via STD_DEFS
    ]
    # Non-cancer + YNOR from standardized
    STD_DEFS = [
        ("DIA", ["DIA"]),
        ("HBP", ["HBP"]),
        ("H.D.", ["H.D."]),
        ("YNOR", ["YNOR"]),
    ]

    id_mappings = build_id_mappings()
    std_clin = pd.read_csv(CLINICAL_STD)

    all_rows = []

    # ── by_cancer groups ──
    for bc_file, clin_grp, spec_groups in BC_DEFS:
        sub = spectra[spectra["group"].isin(spec_groups)].copy()
        if len(sub) == 0:
            continue
        agg = sub.groupby("sample_id")[feat_cols].mean().reset_index()
        agg["spec_group"] = sub.groupby("sample_id")["group"].first().values

        clin = pd.read_excel(
            CLINICAL_BY_CANCER / f"{bc_file}.xlsx",
            sheet_name="Clean", engine="openpyxl"
        )
        clin_lookup = {str(r["patient_id"]): r for _, r in clin.iterrows()}

        # Unified display group
        display_grp = spec_groups[0] if len(spec_groups) == 1 else "PAN"
        # For PAN: remap CPAN/YPAN → PAN
        if display_grp == "CPAN":
            display_grp = "PAN"

        for _, spec_row in agg.iterrows():
            sid = int(spec_row["sample_id"])
            sg = spec_row["spec_group"]
            matched = None

            # Strategy 1: direct key
            for fmt in [f"{sg} {sid}", f"{clin_grp} {sid}"]:
                if fmt in clin_lookup:
                    matched = clin_lookup[fmt]; break

            # Strategy 2: numeric tail
            if matched is None:
                for pid, crow in clin_lookup.items():
                    m = re.search(r"(\d+)$", str(pid))
                    if m and int(m.group(1)) == sid:
                        matched = crow; break

            # Strategy 3: ID mapping
            if matched is None:
                for label in [f"{sg} {sid}", str(sid)]:
                    mapped = id_mappings.get((sg, label))
                    if mapped and mapped in clin_lookup:
                        matched = clin_lookup[mapped]; break

            if matched is not None:
                row = {**spec_row.to_dict(), **matched.to_dict()}
                row["sample_id"] = sid
                row["group"] = display_grp
                row["category"] = CATEGORY_MAP.get(display_grp, "unknown")
                all_rows.append(row)

    # ── Standardized non-cancer ──
    for spec_grp, clin_grps, in STD_DEFS:
        sub = spectra[spectra["group"] == spec_grp].copy()
        if len(sub) == 0:
            continue
        agg = sub.groupby("sample_id")[feat_cols].mean().reset_index()
        clin_sub = std_clin[std_clin["disease_group"].isin(clin_grps)]
        cl = {}
        for _, r in clin_sub.iterrows():
            m = re.search(r"(\d+)$", str(r["patient_id"]))
            if m: cl[int(m.group(1))] = r

        for _, spec_row in agg.iterrows():
            sid = int(spec_row["sample_id"])
            if sid in cl:
                row = {**spec_row.to_dict(), **cl[sid].to_dict()}
                row["sample_id"] = sid
                row["group"] = spec_grp
                row["category"] = CATEGORY_MAP.get(spec_grp, "unknown")
                all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df["pathology_group"] = df.get("pathology", pd.Series()).apply(simplify_pathology)
    df["t_stage"] = df.get("t_stage", pd.Series()).apply(simplify_t_stage)
    print(f"Total linked: {len(df)} subjects across {df['group'].nunique()} groups")
    print(df["group"].value_counts().reindex(GROUP_ORDER).dropna().to_string())
    return df, feat_cols


def embed_and_cluster(df, feat_cols):
    from umap import UMAP
    from sklearn.manifold import TSNE
    from hdbscan import HDBSCAN

    X = df[feat_cols].values

    print("\nComputing UMAP...")
    emb_u = UMAP(n_components=2, n_neighbors=30, min_dist=0.1,
                 metric="euclidean", random_state=42).fit_transform(X)
    df["umap_1"] = emb_u[:, 0]
    df["umap_2"] = emb_u[:, 1]

    print("Computing t-SNE...")
    emb_t = TSNE(n_components=2, perplexity=50, random_state=42,
                 learning_rate="auto", init="pca").fit_transform(X)
    df["tsne_1"] = emb_t[:, 0]
    df["tsne_2"] = emb_t[:, 1]

    # Global HDBSCAN
    print("Computing global HDBSCAN...")
    labels = HDBSCAN(min_cluster_size=25, min_samples=5).fit_predict(emb_u)
    df["cluster"] = labels
    n_cl = len(set(labels) - {-1})
    print(f"  Global: {n_cl} clusters, {(labels == -1).sum()} noise")

    # Per-group HDBSCAN (more granular subgroups within each disease)
    print("Computing per-group HDBSCAN...")
    df["intra_cluster"] = -1
    for grp, gdf in df.groupby("group"):
        if len(gdf) < 20:
            continue
        idx = gdf.index
        X_g = emb_u[idx]
        mcs = max(8, len(gdf) // 15)
        gl = HDBSCAN(min_cluster_size=mcs, min_samples=3).fit_predict(X_g)
        df.loc[idx, "intra_cluster"] = gl
        nc = len(set(gl) - {-1})
        if nc > 0:
            print(f"  {grp}: {nc} intra-clusters")
    print()
    return df


def build_html(df):
    # Available variables (>20% coverage)
    avail_num = []
    for v in ALL_NUMERIC:
        if v in df.columns and df[v].notna().mean() > 0.2:
            avail_num.append([v, VAR_LABELS.get(v, v)])
    avail_cat = []
    for v in ALL_CATEGORICAL:
        if v in df.columns and df[v].notna().mean() > 0.2 and df[v].dropna().nunique() >= 2:
            avail_cat.append([v, VAR_LABELS.get(v, v)])

    # Build records
    records = []
    for _, row in df.iterrows():
        rec = {
            "id": int(row["sample_id"]),
            "group": str(row["group"]),
            "category": str(row["category"]),
            "umap_1": round(float(row["umap_1"]), 4),
            "umap_2": round(float(row["umap_2"]), 4),
            "tsne_1": round(float(row["tsne_1"]), 4),
            "tsne_2": round(float(row["tsne_2"]), 4),
            "cluster": int(row["cluster"]),
            "intra_cluster": int(row["intra_cluster"]),
        }
        for v, _ in avail_num:
            rec[v] = round(float(row[v]), 2) if pd.notna(row.get(v)) else None
        for v, _ in avail_cat:
            if v in ("group", "category", "cluster", "intra_cluster"):
                continue
            val = row.get(v)
            if pd.notna(val):
                rec[v] = str(int(val)) if isinstance(val, float) and val == int(val) else str(val)
            else:
                rec[v] = None
        records.append(rec)

    # Group counts for filter panel
    group_counts = df["group"].value_counts().reindex(GROUP_ORDER).dropna().astype(int).to_dict()

    data_json = json.dumps(records)
    num_json = json.dumps(avail_num)
    cat_json = json.dumps(avail_cat)
    gc_json = json.dumps(GROUP_COLORS)
    go_json = json.dumps(GROUP_ORDER)
    gcnt_json = json.dumps(group_counts)
    cc_json = json.dumps(CATEGORY_COLORS)

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SERS Unified Subgroup Explorer</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,-apple-system,sans-serif;background:#0f172a;color:#e2e8f0}}

.header{{background:linear-gradient(135deg,#1e293b,#0f172a);border-bottom:1px solid #334155;padding:14px 24px;display:flex;align-items:center;gap:16px}}
.header h1{{font-size:18px;font-weight:700;color:#f8fafc}}
.header .badge{{background:#7c3aed;color:white;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600}}

.controls{{display:flex;gap:10px;padding:10px 24px;background:#1e293b;border-bottom:1px solid #334155;flex-wrap:wrap;align-items:center}}
.cg{{display:flex;align-items:center;gap:5px}}
.cg label{{font-size:11px;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:.5px}}
select,input[type=text]{{background:#0f172a;border:1px solid #475569;color:#e2e8f0;padding:5px 8px;border-radius:5px;font-size:12px;outline:none}}
select:focus,input:focus{{border-color:#7c3aed}}
input::placeholder{{color:#64748b}}
input[type=range]{{accent-color:#7c3aed}}

.main{{display:flex;height:calc(100vh - 105px)}}
.plot-area{{flex:1;position:relative}}
#scatter{{width:100%;height:100%}}

/* Filter panel */
.filter-panel{{
  position:absolute;top:10px;right:10px;
  background:#1e293bee;backdrop-filter:blur(8px);
  border:1px solid #475569;border-radius:10px;
  padding:10px 12px;width:180px;font-size:11px;z-index:10;
}}
.filter-panel h4{{font-size:10px;text-transform:uppercase;color:#94a3b8;letter-spacing:.5px;margin-bottom:6px;font-weight:700}}
.fg{{display:flex;align-items:center;gap:6px;padding:2px 0;cursor:pointer}}
.fg input{{accent-color:#7c3aed;cursor:pointer}}
.fg .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
.fg .lbl{{flex:1;color:#e2e8f0}}
.fg .cnt{{color:#64748b;font-size:10px}}
.filter-btns{{display:flex;gap:4px;margin-top:6px}}
.fbtn{{background:#334155;border:none;color:#94a3b8;padding:3px 8px;border-radius:4px;font-size:10px;cursor:pointer}}
.fbtn:hover{{background:#475569;color:#e2e8f0}}

.sidebar{{width:300px;background:#1e293b;border-left:1px solid #334155;overflow-y:auto}}
.sidebar-header{{padding:12px 14px;background:#334155;font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8;position:sticky;top:0;z-index:1}}
.pcard{{padding:8px 14px;border-bottom:1px solid #0f172a;cursor:pointer;transition:background .15s}}
.pcard:hover{{background:#334155}}
.pcard.active{{background:#7c3aed22;border-left:3px solid #7c3aed}}
.pcard .pid{{font-size:13px;font-weight:700;color:#f8fafc}}
.pcard .gtag{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:10px;font-weight:600;margin-left:4px}}
.pcard .meta{{font-size:10px;color:#94a3b8;margin-top:2px}}

.detail{{position:fixed;right:300px;top:105px;width:360px;max-height:calc(100vh - 125px);background:#1e293b;border:1px solid #475569;border-radius:12px;box-shadow:0 20px 60px rgba(0,0,0,.5);overflow-y:auto;display:none;z-index:100}}
.detail.show{{display:block}}
.detail-hd{{padding:14px;background:#334155;border-radius:12px 12px 0 0;display:flex;justify-content:space-between;align-items:center}}
.detail-hd h3{{font-size:15px;font-weight:700}}
.detail-close{{cursor:pointer;color:#94a3b8;font-size:20px}}
.detail-close:hover{{color:#f8fafc}}
.detail-body{{padding:14px}}
.dsec{{margin-bottom:12px}}
.dsec h4{{font-size:10px;text-transform:uppercase;color:#7c3aed;letter-spacing:.5px;margin-bottom:5px;font-weight:700}}
.drow{{display:flex;justify-content:space-between;padding:2px 0;font-size:12px}}
.drow .lb{{color:#94a3b8}}
.drow .vl{{color:#f8fafc;font-weight:600}}

.legend{{position:absolute;bottom:14px;left:14px;background:#1e293bdd;backdrop-filter:blur(8px);border:1px solid #475569;border-radius:10px;padding:8px 12px;font-size:11px;max-height:220px;overflow-y:auto}}
.legend-title{{font-weight:700;margin-bottom:4px;color:#94a3b8;text-transform:uppercase;font-size:10px}}
.litem{{display:flex;align-items:center;gap:5px;padding:1px 0}}
.ldot{{width:9px;height:9px;border-radius:50%;flex-shrink:0}}

.stats-bar{{position:absolute;top:10px;left:10px;display:flex;gap:6px;flex-wrap:wrap}}
.schip{{background:#1e293bdd;backdrop-filter:blur(8px);border:1px solid #475569;border-radius:7px;padding:5px 10px;font-size:11px}}
.schip .n{{font-weight:700;color:#7c3aed}}

.cbar{{position:absolute;bottom:14px;right:320px;background:#1e293bdd;backdrop-filter:blur(8px);border:1px solid #475569;border-radius:10px;padding:8px 12px;display:none}}
</style>
</head>
<body>

<div class="header">
  <h1>SERS Unified Subgroup Explorer</h1>
  <span class="badge">{len(records)} subjects · {df['group'].nunique()} groups</span>
</div>
<div class="controls">
  <div class="cg"><label>Embedding</label><select id="embSel"><option value="umap" selected>UMAP</option><option value="tsne">t-SNE</option></select></div>
  <div class="cg"><label>Color by</label><select id="colorSel"></select></div>
  <div class="cg"><label>Search</label><input id="searchIn" type="text" placeholder="Group ID (e.g. CRC 42)" style="width:160px"></div>
  <div class="cg"><label>Size</label><input id="sizeSl" type="range" min="2" max="16" value="6" style="width:80px"></div>
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
    <div class="filter-panel" id="filterPanel">
      <h4>Filter Groups</h4>
      <div id="filterList"></div>
      <div class="filter-btns">
        <button class="fbtn" onclick="toggleAll(true)">All</button>
        <button class="fbtn" onclick="toggleAll(false)">None</button>
        <button class="fbtn" onclick="toggleCat('cancer')">Cancer</button>
        <button class="fbtn" onclick="toggleCat('non-cancer')">Non-cancer</button>
        <button class="fbtn" onclick="toggleCat('control')">Control</button>
      </div>
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
const ALL={data_json};
const NUM_V={num_json};
const CAT_V={cat_json};
const GC={gc_json};
const GO={go_json};
const GCNT={gcnt_json};
const CAT_COL={cc_json};
const CAT_MAP={json.dumps(CATEGORY_MAP)};
const LABELS={json.dumps(VAR_LABELS)};

const CLUST_C=['#EF4444','#3B82F6','#22C55E','#F97316','#A855F7','#06B6D4','#EC4899','#EAB308','#84CC16','#F43F5E','#14B8A6','#F472B6','#A3E635','#FB923C'];
const NC='#4B5563';
const VIR=[[68,1,84],[72,26,108],[71,47,125],[65,68,135],[57,86,140],[48,103,141],[40,120,142],[33,137,141],[26,153,136],[30,169,120],[53,183,95],[94,196,60],[143,205,26],[194,211,30],[241,229,29],[253,231,37]];
function vir(t){{t=Math.max(0,Math.min(1,t));const i=t*(VIR.length-1),l=Math.floor(i),h=Math.ceil(i),f=i-l;const c=VIR[l].map((v,j)=>Math.round(v+f*(VIR[h][j]-v)));return`rgb(${{c[0]}},${{c[1]}},${{c[2]}})`}}
function lb(v){{return LABELS[v]||v}}

let emb='umap',colorBy='group',hlId=null,selId=null,hlGroup=null,ptSize=6;
let activeGroups=new Set(GO);
let D=ALL; // filtered data

// Color selector
const cSel=document.getElementById('colorSel');
(function(){{
  const og1=document.createElement('optgroup');og1.label='Categorical';
  CAT_V.forEach(([k,v])=>{{const o=document.createElement('option');o.value=k;o.textContent=v;if(k==='group')o.selected=true;og1.appendChild(o)}});
  cSel.appendChild(og1);
  const og2=document.createElement('optgroup');og2.label='Numeric';
  NUM_V.forEach(([k,v])=>{{const o=document.createElement('option');o.value=k;o.textContent=v;og2.appendChild(o)}});
  cSel.appendChild(og2);
}})();

// Filter panel
function buildFilter(){{
  const el=document.getElementById('filterList');
  el.innerHTML=GO.map(g=>{{
    const n=GCNT[g]||0;
    const col=GC[g]||'#666';
    const chk=activeGroups.has(g)?'checked':'';
    return`<label class="fg"><input type="checkbox" ${{chk}} onchange="toggleGroup('${{g}}',this.checked)"><span class="dot" style="background:${{col}}"></span><span class="lbl">${{g}}</span><span class="cnt">${{n}}</span></label>`;
  }}).join('');
}}
function toggleGroup(g,on){{if(on)activeGroups.add(g);else activeGroups.delete(g);filterData();}}
function toggleAll(on){{GO.forEach(g=>{{if(on)activeGroups.add(g);else activeGroups.delete(g)}});buildFilter();filterData();}}
function toggleCat(cat){{activeGroups.clear();GO.forEach(g=>{{if(CAT_MAP[g]===cat)activeGroups.add(g)}});buildFilter();filterData();}}
function filterData(){{D=ALL.filter(d=>activeGroups.has(d.group));computeXf();buildList();draw();}}

// Canvas
const canvas=document.getElementById('scatter'),ctx=canvas.getContext('2d');
let W,H,xf;

function resize(){{
  const r=canvas.parentElement.getBoundingClientRect();
  W=r.width;H=r.height;
  canvas.width=W*devicePixelRatio;canvas.height=H*devicePixelRatio;
  canvas.style.width=W+'px';canvas.style.height=H+'px';
  ctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  computeXf();draw();
}}

function computeXf(){{
  if(!D.length)return;
  const xk=emb==='umap'?'umap_1':'tsne_1',yk=emb==='umap'?'umap_2':'tsne_2';
  const xs=D.map(d=>d[xk]),ys=D.map(d=>d[yk]);
  const p=50,xMn=Math.min(...xs),xMx=Math.max(...xs),yMn=Math.min(...ys),yMx=Math.max(...ys);
  const xR=xMx-xMn||1,yR=yMx-yMn||1;
  const sc=Math.min((W-2*p)/xR,(H-2*p)/yR);
  xf={{xMn,yMn,sc,ox:(W-xR*sc)/2,oy:(H-yR*sc)/2}};
}}

function toS(d){{
  const xk=emb==='umap'?'umap_1':'tsne_1',yk=emb==='umap'?'umap_2':'tsne_2';
  return{{x:(d[xk]-xf.xMn)*xf.sc+xf.ox,y:H-((d[yk]-xf.yMn)*xf.sc+xf.oy)}};
}}

function isNum(v){{return NUM_V.some(([k])=>k===v)}}

function getCol(d){{
  const v=colorBy;
  if(v==='group')return GC[d.group]||'#666';
  if(v==='category')return CAT_COL[d.category]||'#666';
  if(v==='cluster')return d.cluster<0?NC:CLUST_C[d.cluster%CLUST_C.length];
  if(v==='intra_cluster')return d.intra_cluster<0?NC:CLUST_C[d.intra_cluster%CLUST_C.length];
  if(isNum(v)){{
    const val=d[v];if(val==null)return'#1e293b';
    const vs=D.map(r=>r[v]).filter(x=>x!=null);
    const lo=Math.min(...vs),hi=Math.max(...vs);
    return vir(hi>lo?(val-lo)/(hi-lo):.5);
  }}
  const cats=[...new Set(D.map(r=>r[v]).filter(x=>x!=null))].sort();
  const i=cats.indexOf(d[v]);return i<0?'#1e293b':CLUST_C[i%CLUST_C.length];
}}

function draw(){{
  ctx.clearRect(0,0,W,H);
  if(!D.length||!xf)return;
  // Grid
  ctx.strokeStyle='#1e293b';ctx.lineWidth=1;
  for(let x=0;x<W;x+=60){{ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke()}}
  for(let y=0;y<H;y+=60){{ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke()}}

  const sorted=[...D].sort((a,b)=>{{
    if(a.id===selId&&a.group===(hlGroup||a.group))return 1;
    if(b.id===selId&&b.group===(hlGroup||b.group))return-1;
    if(a.id===hlId&&a.group===(hlGroup||a.group))return 1;
    if(b.id===hlId&&b.group===(hlGroup||b.group))return-1;
    return 0;
  }});

  sorted.forEach(d=>{{
    const{{x,y}}=toS(d);
    const isH=(d.id===hlId&&(!hlGroup||d.group===hlGroup));
    const isS=(d.id===selId&&(!hlGroup||d.group===hlGroup));
    const r=isH||isS?ptSize*2:ptSize;

    ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);
    ctx.fillStyle=getCol(d);
    ctx.globalAlpha=(hlId&&!isH&&!isS)?.1:.75;
    ctx.fill();
    if(isH||isS){{
      ctx.strokeStyle='#fbbf24';ctx.lineWidth=2.5;ctx.stroke();
      ctx.globalAlpha=1;ctx.fillStyle='#fbbf24';ctx.font='bold 12px system-ui';
      ctx.fillText(`${{d.group}} #${{d.id}}`,x+r+4,y-r);
    }}
    ctx.globalAlpha=1;
  }});
  updLegend();updStats();
}}

function updLegend(){{
  const leg=document.getElementById('legend'),cb=document.getElementById('cbar'),v=colorBy;
  if(isNum(v)){{
    leg.style.display='none';cb.style.display='block';
    const vs=D.map(r=>r[v]).filter(x=>x!=null);
    if(!vs.length)return;
    document.getElementById('cbMax').textContent=Math.max(...vs).toFixed(1);
    document.getElementById('cbMin').textContent=Math.min(...vs).toFixed(1);
    const c2=document.getElementById('cbG').getContext('2d');
    for(let i=0;i<120;i++){{c2.fillStyle=vir(1-i/119);c2.fillRect(0,i,18,1)}}
    return;
  }}
  cb.style.display='none';leg.style.display='block';
  if(v==='group'){{
    leg.innerHTML='<div class="legend-title">Disease Group</div>'+GO.filter(g=>activeGroups.has(g)).map(g=>{{
      const n=D.filter(d=>d.group===g).length;
      return`<div class="litem"><span class="ldot" style="background:${{GC[g]||'#666'}}"></span>${{g}} (${{n}})</div>`;
    }}).join('');
  }}else if(v==='category'){{
    leg.innerHTML='<div class="legend-title">Category</div>'+['cancer','non-cancer','control'].map(c=>{{
      const n=D.filter(d=>d.category===c).length;
      return`<div class="litem"><span class="ldot" style="background:${{CAT_COL[c]}}"></span>${{c}} (${{n}})</div>`;
    }}).join('');
  }}else if(v==='cluster'||v==='intra_cluster'){{
    const cls=[...new Set(D.map(d=>d[v]))].sort((a,b)=>a-b);
    const title=v==='cluster'?'Global Cluster':'Intra-group Cluster';
    leg.innerHTML=`<div class="legend-title">${{title}}</div>`+cls.map(c=>{{
      const col=c<0?NC:CLUST_C[c%CLUST_C.length];const n=D.filter(d=>d[v]===c).length;
      return`<div class="litem"><span class="ldot" style="background:${{col}}"></span>${{c<0?'Noise':'C'+c}} (${{n}})</div>`;
    }}).join('');
  }}else{{
    const cats=[...new Set(D.map(d=>d[v]).filter(x=>x!=null))].sort();
    leg.innerHTML=`<div class="legend-title">${{lb(v)}}</div>`+cats.map((c,i)=>{{
      const n=D.filter(d=>d[v]===c).length;
      return`<div class="litem"><span class="ldot" style="background:${{CLUST_C[i%CLUST_C.length]}}"></span>${{c}} (${{n}})</div>`;
    }}).join('');
  }}
}}

function updStats(){{
  const groups=[...new Set(D.map(d=>d.group))];
  const nc=new Set(D.filter(d=>d.cluster>=0).map(d=>d.cluster)).size;
  document.getElementById('statsBar').innerHTML=
    `<div class="schip"><span class="n">${{D.length}}</span> subjects</div>`+
    `<div class="schip"><span class="n">${{groups.length}}</span> groups</div>`+
    `<div class="schip"><span class="n">${{nc}}</span> clusters</div>`+
    `<div class="schip">${{emb.toUpperCase()}}</div>`;
}}

// Patient list
function buildList(filter){{
  const el=document.getElementById('pList');
  let items=D;
  if(filter){{
    const q=filter.toLowerCase().trim();
    items=items.filter(d=>{{
      const key=d.group+' '+d.id;
      return key.toLowerCase().includes(q)||String(d.id).includes(q);
    }});
  }}
  items=[...items].sort((a,b)=>a.group.localeCompare(b.group)||a.id-b.id);
  document.getElementById('listCnt').textContent='('+items.length+')';
  // Limit for performance
  const show=items.slice(0,500);
  el.innerHTML=show.map(d=>{{
    const col=GC[d.group]||'#666';
    const meta=[d.sex,d.age?d.age+'y':'',d.t_stage||d.stage||''].filter(Boolean).join(' · ');
    return`<div class="pcard" data-id="${{d.id}}" data-grp="${{d.group}}" onmouseenter="hov(${{d.id}},'${{d.group}}')" onmouseleave="unhov()" onclick="sel(${{d.id}},'${{d.group}}')">
      <span class="pid">#${{d.id}}</span><span class="gtag" style="background:${{col}}33;color:${{col}}">${{d.group}}</span>
      <div class="meta">${{meta}}</div></div>`;
  }}).join('')+(items.length>500?`<div style="padding:10px;color:#64748b;text-align:center">+${{items.length-500}} more...</div>`:'');
}}

function hov(id,grp){{hlId=id;hlGroup=grp;draw()}}
function unhov(){{hlId=null;hlGroup=null;draw()}}
function sel(id,grp){{
  selId=id;hlId=id;hlGroup=grp;draw();showDetail(id,grp);
  document.querySelectorAll('.pcard').forEach(e=>e.classList.toggle('active',+e.dataset.id===id&&e.dataset.grp===grp));
}}

function showDetail(id,grp){{
  const d=D.find(r=>r.id===id&&r.group===grp);if(!d)return;
  const p=document.getElementById('detailP');
  const col=GC[d.group]||'#666';
  document.getElementById('dTitle').innerHTML=
    `${{d.group}} #${{d.id}} <span class="gtag" style="background:${{col}}33;color:${{col}}">${{d.group}}</span>`;

  let html='';
  const sections=[
    ['Demographics',['age','sex','bmi','bp_systolic','bp_diastolic','smoking_status','drinking_status']],
    ['Cancer',['group','category','cluster','stage','t_stage','pathology_group','sample_timing']],
    ['CBC',['wbc','rbc','hb','hct','platelet','neutrophil_pct','lymphocyte_pct']],
    ['Chemistry',['ast','alt','alp','ggt','bun','creatinine','uric_acid','glucose','total_bilirubin','calcium','total_cholesterol','triglyceride','total_protein','albumin','ldh','hs_crp','sodium','potassium','chloride','hdl_c','ldl_c','hba1c']],
    ['Tumor Markers',['afp','cea','ca19_9','psa']],
    ['Urinalysis',['ua_sg','ua_ph','ua_protein','ua_blood','ua_glucose']],
  ];
  sections.forEach(([title,vars])=>{{
    const rows=vars.filter(v=>d[v]!=null&&d[v]!==undefined).map(v=>
      `<div class="drow"><span class="lb">${{lb(v)}}</span><span class="vl">${{d[v]}}</span></div>`
    );
    if(rows.length)html+=`<div class="dsec"><h4>${{title}}</h4>${{rows.join('')}}</div>`;
  }});
  html+=`<div class="dsec"><h4>Embedding</h4>
    <div class="drow"><span class="lb">UMAP</span><span class="vl">(${{d.umap_1.toFixed(2)}}, ${{d.umap_2.toFixed(2)}})</span></div>
    <div class="drow"><span class="lb">t-SNE</span><span class="vl">(${{d.tsne_1.toFixed(2)}}, ${{d.tsne_2.toFixed(2)}})</span></div></div>`;

  document.getElementById('dBody').innerHTML=html;
  p.classList.add('show');
}}

document.getElementById('dClose').onclick=()=>{{
  document.getElementById('detailP').classList.remove('show');selId=null;hlGroup=null;draw();
}};

// Canvas interactions
canvas.addEventListener('mousemove',e=>{{
  const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=null,md=Infinity;
  D.forEach(d=>{{const{{x,y}}=toS(d);const dist=Math.hypot(mx-x,my-y);if(dist<md){{md=dist;best=d}}}});
  if(md<15&&best){{canvas.style.cursor='pointer';if(hlId!==best.id||hlGroup!==best.group){{hlId=best.id;hlGroup=best.group;draw()}}}}
  else{{canvas.style.cursor='default';if(hlId&&!selId){{hlId=null;hlGroup=null;draw()}}}}
}});
canvas.addEventListener('click',e=>{{
  const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
  let best=null,md=Infinity;
  D.forEach(d=>{{const{{x,y}}=toS(d);const dist=Math.hypot(mx-x,my-y);if(dist<md){{md=dist;best=d}}}});
  if(md<15&&best)sel(best.id,best.group);
}});

// Controls
document.getElementById('embSel').onchange=e=>{{emb=e.target.value;computeXf();draw()}};
document.getElementById('colorSel').onchange=e=>{{colorBy=e.target.value;draw()}};
document.getElementById('searchIn').oninput=e=>{{
  const q=e.target.value.trim();buildList(q);
  // Try exact match "GRP ID"
  const m=q.match(/^([A-Z.]+)\\s*(\\d+)$/i);
  if(m){{
    const grp=m[1].toUpperCase(),id=parseInt(m[2]);
    const found=D.find(d=>d.group===grp&&d.id===id);
    if(found){{sel(found.id,found.group);return}}
  }}
  // Numeric only — highlight first match
  if(/^\\d+$/.test(q)){{
    const id=parseInt(q);
    const found=D.find(d=>d.id===id);
    if(found){{sel(found.id,found.group);return}}
  }}
  hlId=null;selId=null;hlGroup=null;
  document.getElementById('detailP').classList.remove('show');draw();
}};
document.getElementById('sizeSl').oninput=e=>{{ptSize=+e.target.value;draw()}};

// Init
window.addEventListener('resize',resize);
buildFilter();
filterData();
resize();
</script>
</body>
</html>"""

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"HTML saved to {OUTPUT_HTML}")
    print(f"File size: {OUTPUT_HTML.stat().st_size / 1024:.0f} KB")


def main():
    print("=" * 60)
    print("Unified Subgroup Interactive Explorer")
    print("=" * 60)

    df, feat_cols = load_all()
    df = embed_and_cluster(df, feat_cols)
    build_html(df)


if __name__ == "__main__":
    main()
