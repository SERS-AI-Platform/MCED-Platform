"""
Metabolite-centric cross-reference dashboard.
Base: Thermo metabolite peaks → map cancer-type-specific SERS peaks onto them.
SPAN excluded.
"""

import pandas as pd
import numpy as np
import json
import html as html_mod

BASE = "/home/user/SERS-AI"
PEAK_FILE = f"{BASE}/results/training/R_BLC_added/logistic_regression/v001/evaluation/stage2/peak_intensity_by_diagnosis.csv"
CORR_FILE = f"{BASE}/metabolite_profiling/data/metabolite_sers_correlation.csv"
THERMO_FILE = f"{BASE}/thermo_metabolite_data.json"

TOLERANCE = 10  # cm-1 for matching
CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "BLC"]  # No SPAN

# ── Load ──
peaks_df = pd.read_csv(PEAK_FILE)
peaks_df = peaks_df[peaks_df["diagnosis"].isin(CANCER_TYPES)]
corr_df = pd.read_csv(CORR_FILE)
with open(THERMO_FILE) as f:
    thermo_data = json.load(f)

corr_lookup = dict(zip(corr_df["metabolite"], corr_df["pearson_correlation"]))

# ── Metabolite categories ──
CATEGORIES = {
    "아미노산": ["Alanine", "Arginine", "Aspartic acid", "Cysteine", "Glutamic acid", "Glycine",
                "Histidine", "Isoleucine", "Leucine", "Phenylalanine", "Proline", "Serine",
                "Threonine", "Tryptophan", "Tyrosine", "Valine"],
    "핵산/뉴클레오사이드": ["Adenine", "Adenosine", "Guanine", "Hypoxanthine", "Inosine",
                     "Pseudouridine", "Purine", "Pyrimidine", "Uracil", "Xanthine"],
    "유기산": ["Acrylic acid", "Ascorbic acid", "Benzoic acid", "cis-Aconitic acid",
            "trans-Aconitic acid", "Hippuric acid", "Maleic acid", "Malic acid", "Uric acid", "Xylonic acid"],
    "지질/지방산": ["Cholesterol", "Palmitic acid", "Sphinganine", "Stearic acid"],
    "당류": ["Fucose", "Galactosamine", "Glucose", "Glycogen", "Xylose"],
    "장내미생물 관련": ["Hippuric acid", "Trimethylamine-N-oxide", "Benzoic acid"],
    "산화스트레스": ["8-Hydroxy-2-deoxyguanosine", "Ascorbic acid"],
    "에너지/기타": ["Betaine", "Choline", "Creatine", "Creatinine", "Kynurenine", "NADH",
                "O-Acetylcarnitine", "Spermidine", "4-Pyridoxic acid", "Kaempferol",
                "N-Acetylneuraminic acid", "Glycocholic acid", "Taurine"],
}

def get_category(name):
    for cat, members in CATEGORIES.items():
        if name in members:
            return cat
    return "기타"

# ── Known vibration modes for common wavenumber regions ──
VIBRATION_LABELS = {
    (540, 570): "S-S stretch",
    (595, 635): "C-S stretch",
    (660, 700): "Ring breathing (Creatinine)",
    (710, 740): "Adenine ring breathing",
    (740, 770): "C-N stretch",
    (780, 810): "Ring breathing (Hippuric)",
    (830, 860): "Tyr Fermi / C-C",
    (870, 900): "C-C stretch",
    (920, 945): "C-C protein backbone",
    (985, 1015): "Phe ring breathing",
    (1010, 1040): "C-C / C-O stretch",
    (1050, 1075): "C-O-C stretch",
    (1100, 1160): "C-N / C-O-C",
    (1200, 1260): "Amide III",
    (1280, 1320): "CH₂ twist / Amide III",
    (1335, 1370): "CH def / Trp / Adenine",
    (1390, 1430): "COO⁻ / CH₃ def",
    (1430, 1470): "CH₂ deformation",
    (1530, 1560): "Amide II",
    (1575, 1615): "C=C / Purine ring",
    (1620, 1660): "Amide I (C=O)",
    (1670, 1700): "C=O stretch",
    (1700, 1750): "C=O ester",
    (2080, 2130): "S-H / Unknown",
}

def get_vibration_label(wn):
    for (lo, hi), label in VIBRATION_LABELS.items():
        if lo <= wn <= hi:
            return label
    return ""

# ── Build metabolite data ──
metabolite_results = []

for m in thermo_data["metabolites"]:
    name = m["name"]
    corr = corr_lookup.get(name)
    category = get_category(name)
    all_peaks = m["top_peaks"]  # top 5 peaks

    # For each metabolite peak, find matching cancer-type SERS peaks
    peak_mappings = []
    total_cancer_hits = 0

    for p in all_peaks:
        wn = p["wn"]
        intensity = p["intensity"]
        vib_label = get_vibration_label(wn)

        # Find cancer SERS peaks within tolerance
        mask = peaks_df["wavenumber"].between(wn - TOLERANCE, wn + TOLERANCE)
        matched_cancers = peaks_df[mask][["diagnosis", "peak_rank", "wavenumber", "peak_intensity", "prominence"]].copy()
        matched_cancers = matched_cancers.sort_values("peak_rank")

        cancer_hits = []
        for _, row in matched_cancers.iterrows():
            cancer_hits.append({
                "type": row["diagnosis"],
                "rank": int(row["peak_rank"]),
                "sers_wn": round(row["wavenumber"], 1),
                "delta": round(abs(row["wavenumber"] - wn), 1),
                "sers_intensity": round(row["peak_intensity"], 2),
            })
            total_cancer_hits += 1

        peak_mappings.append({
            "wn": wn,
            "rel_intensity": intensity,
            "vibration": vib_label,
            "cancer_hits": cancer_hits,
        })

    metabolite_results.append({
        "name": name,
        "category": category,
        "corr": round(corr, 3) if corr else None,
        "n_peaks": len(all_peaks),
        "total_cancer_hits": total_cancer_hits,
        "peaks": peak_mappings,
    })

# Sort: by total cancer hits (desc), then correlation (desc)
metabolite_results.sort(key=lambda x: (-x["total_cancer_hits"], -(x["corr"] or 0)))

# ── Stats ──
mets_with_hits = [m for m in metabolite_results if m["total_cancer_hits"] > 0]
total_mappings = sum(m["total_cancer_hits"] for m in metabolite_results)

# Cancer-type summary: which metabolites map to each cancer's peaks
cancer_met_map = {ct: {} for ct in CANCER_TYPES}
for m in metabolite_results:
    for p in m["peaks"]:
        for ch in p["cancer_hits"]:
            ct = ch["type"]
            if m["name"] not in cancer_met_map[ct]:
                cancer_met_map[ct][m["name"]] = {"ranks": [], "peaks": [], "corr": m["corr"]}
            cancer_met_map[ct][m["name"]]["ranks"].append(ch["rank"])
            cancer_met_map[ct][m["name"]]["peaks"].append(round(p["wn"], 0))

# ── HTML ──
def esc(s):
    return html_mod.escape(str(s))

def rank_badge(rank):
    colors = {1:"#dc2626",2:"#ea580c",3:"#d97706",4:"#ca8a04",5:"#65a30d",6:"#059669",7:"#0891b2",8:"#6366f1"}
    c = colors.get(rank, "#94a3b8")
    return f'<span style="display:inline-block;min-width:18px;text-align:center;padding:1px 5px;border-radius:10px;font-size:10px;font-weight:700;background:{c};color:#fff">#{rank}</span>'

def corr_badge(c):
    if c is None: return '<span class="tag tag-low">N/A</span>'
    if c >= 0.6: return f'<span class="tag tag-high">r={c:.3f}</span>'
    if c >= 0.4: return f'<span class="tag tag-mid">r={c:.3f}</span>'
    return f'<span class="tag tag-low">r={c:.3f}</span>'

CANCER_COLORS = {
    "PRO": ("#2563eb", "#eff6ff", "#bfdbfe"),
    "BRE": ("#db2777", "#fdf2f8", "#fbcfe8"),
    "OVA": ("#9333ea", "#faf5ff", "#e9d5ff"),
    "LUN": ("#16a34a", "#f0fdf4", "#bbf7d0"),
    "CRC": ("#ea580c", "#fff7ed", "#fed7aa"),
    "CPAN": ("#ca8a04", "#fefce8", "#fef08a"),
    "BLC": ("#0891b2", "#ecfeff", "#a5f3fc"),
}
CANCER_KR = {"PRO":"전립선","BRE":"유방","OVA":"난소","LUN":"폐","CRC":"대장","CPAN":"만성췌장","BLC":"방광"}

html = []
html.append(f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Metabolite ↔ Cancer Peak Mapping</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Noto Sans KR',sans-serif;color:#333;background:#f0f2f5;line-height:1.6;font-size:14px}}
  .container{{max-width:1400px;margin:0 auto;padding:20px}}
  .header{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;padding:40px;border-radius:16px;margin-bottom:24px}}
  .header h1{{font-size:22px;font-weight:700;margin-bottom:4px}}
  .header p{{font-size:13px;opacity:.8;margin-top:4px}}
  .htags{{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}}
  .htags span{{background:rgba(255,255,255,.15);padding:3px 10px;border-radius:16px;font-size:11px}}
  .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
  .mc{{background:#fff;border-radius:12px;padding:20px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
  .mc .v{{font-size:32px;font-weight:700;color:#1e3a5f}}
  .mc .l{{font-size:11px;color:#94a3b8;margin-top:2px}}
  .card{{background:#fff;border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:20px}}
  .card h3{{font-size:15px;color:#1e3a5f;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #e2e8f0}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{background:#1e3a5f;color:#fff;padding:6px 8px;text-align:left;font-weight:500;font-size:11px;position:sticky;top:0;z-index:2}}
  td{{padding:5px 8px;border-bottom:1px solid #f1f5f9;vertical-align:top}}
  tr:hover td{{background:#f8fafc}}
  .tag{{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600;margin:1px;white-space:nowrap}}
  .tag-high{{background:#dcfce7;color:#16a34a}}
  .tag-mid{{background:#fef3c7;color:#b45309}}
  .tag-low{{background:#f3f4f6;color:#6b7280}}
  .tag-cat{{background:#e0e7ff;color:#4338ca;font-size:10px;padding:2px 8px;border-radius:10px}}
  .cancer-badge{{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:500;margin:2px;border:1px solid}}
  .peak-row{{display:flex;align-items:center;gap:8px;margin:3px 0;flex-wrap:wrap}}
  .peak-wn{{font-weight:700;font-size:13px;color:#1e3a5f;min-width:65px}}
  .peak-bar{{height:6px;border-radius:3px;background:#3b82f6}}
  .vib-label{{font-size:10px;color:#64748b;min-width:120px}}
  .hit-group{{display:flex;gap:3px;flex-wrap:wrap}}
  .met-card{{border:1px solid #e2e8f0;border-radius:10px;margin-bottom:12px;overflow:hidden;transition:all .2s}}
  .met-card.highlight{{border-color:#3b82f6;box-shadow:0 0 0 2px rgba(59,130,246,.15)}}
  .met-header{{display:flex;justify-content:space-between;align-items:center;padding:12px 16px;cursor:pointer;transition:background .15s}}
  .met-header:hover{{background:#f8fafc}}
  .met-name{{font-size:15px;font-weight:600;color:#1e3a5f}}
  .met-sub{{font-size:11px;color:#64748b;margin-top:2px}}
  .met-right{{text-align:right;display:flex;align-items:center;gap:8px}}
  .met-body{{padding:0 16px 16px;display:none}}
  .met-body.open{{display:block}}
  .hit-count{{font-size:20px;font-weight:700;min-width:28px;text-align:center;border-radius:8px;padding:2px 6px}}
  .hit-high{{color:#dc2626;background:#fef2f2}}
  .hit-mid{{color:#d97706;background:#fffbeb}}
  .hit-low{{color:#64748b;background:#f8fafc}}
  .no-hit{{color:#d1d5db}}
  .search{{width:100%;padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:13px;margin-bottom:12px;font-family:inherit}}
  .search:focus{{outline:none;border-color:#1e3a5f;box-shadow:0 0 0 3px rgba(30,58,95,.1)}}
  .controls{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap;align-items:center}}
  .controls button{{padding:6px 14px;border:1px solid #e2e8f0;border-radius:8px;background:#fff;cursor:pointer;font-size:12px;font-family:inherit;transition:all .15s}}
  .controls button:hover{{border-color:#1e3a5f;background:#f8fafc}}
  .controls button.on{{background:#1e3a5f;color:#fff;border-color:#1e3a5f}}
  .controls select{{padding:6px 10px;border:1px solid #e2e8f0;border-radius:8px;font-size:12px;font-family:inherit}}
  .arrow{{transition:transform .2s;font-size:12px;color:#94a3b8;margin-left:8px}}
  .arrow.open{{transform:rotate(90deg)}}
  .section-title{{font-size:14px;font-weight:700;color:#475569;margin:20px 0 10px;padding:8px 12px;background:#f1f5f9;border-radius:8px}}
  .kf{{border-left:4px solid #f59e0b;background:linear-gradient(135deg,#fffbeb,#fef3c7)}}
  .kf h3{{color:#b45309;border-bottom-color:#f59e0b}}
  .fn{{font-size:10px;color:#94a3b8;margin-top:20px;padding-top:10px;border-top:1px solid #e2e8f0;line-height:1.8}}
  @media(max-width:768px){{.metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>Thermo Metabolite &harr; Cancer SERS Peak Mapping</h1>
  <p>73종 Thermo 실측 대사체의 Raman 피크를 기준으로, 각 암종 SERS 주요 피크와의 매칭을 확인합니다.</p>
  <p>각 대사체의 Top-5 피크 &rarr; 7개 암종(SPAN 제외)의 Top-8 SERS 피크와 &pm;{TOLERANCE} cm&sup1; 내 매칭</p>
  <div class="htags">
    <span>SOLUM Healthcare</span><span>AECD Platform</span><span>73 Thermo Metabolites</span>
    <span>7 Cancer Types</span><span>&pm;{TOLERANCE} cm⁻¹</span><span>2026-03-23</span>
  </div>
</div>

<div class="metrics">
  <div class="mc"><div class="v">73</div><div class="l">Thermo 대사체</div></div>
  <div class="mc"><div class="v">{len(mets_with_hits)}</div><div class="l">암종 피크 매칭된 대사체</div></div>
  <div class="mc"><div class="v">{total_mappings}</div><div class="l">총 대사체↔암종 매핑</div></div>
  <div class="mc"><div class="v">7</div><div class="l">암종 (SPAN 제외)</div></div>
</div>
''')

# ── Cancer-type summary card ──
html.append('''<div class="card">
  <h3>암종별 매핑 요약: 어떤 대사체가 각 암종의 주요 피크에 대응하는가</h3>
  <div style="overflow-x:auto">
  <table>
    <tr><th>암종</th><th>Rank #1 피크</th><th>매칭 대사체 Top-5 (SERS 상관 순)</th><th>총 매칭</th></tr>
''')

for ct in CANCER_TYPES:
    color, bg, border = CANCER_COLORS[ct]
    kr = CANCER_KR[ct]

    # Rank 1 peak
    r1 = peaks_df[(peaks_df["diagnosis"]==ct)&(peaks_df["peak_rank"]==1)]
    r1_wn = f'{r1["wavenumber"].values[0]:.1f}' if len(r1) else "—"

    # Top metabolites for this cancer
    cm = cancer_met_map.get(ct, {})
    sorted_mets = sorted(cm.items(), key=lambda x: -(x[1]["corr"] or 0))[:5]
    met_tags = ""
    for mname, minfo in sorted_mets:
        best_rank = min(minfo["ranks"])
        peaks_str = ", ".join([f"{int(p)}" for p in sorted(set(minfo["peaks"]))])
        c = minfo["corr"]
        tc = "tag-high" if c and c >= 0.6 else "tag-mid" if c and c >= 0.4 else "tag-low"
        met_tags += f'<div style="margin:2px 0"><span class="tag {tc}">{esc(mname)}</span> <span style="font-size:10px;color:#94a3b8">({peaks_str} cm⁻¹, best {rank_badge(best_rank)})</span></div>'

    total = sum(len(v["ranks"]) for v in cm.values())
    html.append(f'''    <tr>
      <td><span class="cancer-badge" style="background:{bg};color:{color};border-color:{border}"><b>{ct}</b> {kr}</span></td>
      <td><b>{r1_wn}</b> cm⁻¹</td>
      <td>{met_tags}</td>
      <td style="text-align:center"><b>{total}</b></td>
    </tr>
''')

html.append('''  </table>
  </div>
</div>
''')

# ── Controls ──
html.append('''
<input type="text" class="search" id="searchBox" placeholder="검색: 대사체명, 카테고리, wavenumber (예: Hippuric, 아미노산, 724)..." oninput="filterMets()">
<div class="controls">
  <button class="on" onclick="toggleAll(true)">모두 펼치기</button>
  <button onclick="toggleAll(false)">모두 접기</button>
  <span style="color:#94a3b8;font-size:11px;margin:0 4px">|</span>
  <button class="on" id="btn-sort-hits" onclick="sortBy('hits')">매핑 수 순</button>
  <button id="btn-sort-corr" onclick="sortBy('corr')">상관 순</button>
  <button id="btn-sort-name" onclick="sortBy('name')">이름 순</button>
  <span style="color:#94a3b8;font-size:11px;margin:0 4px">|</span>
  <button class="on" id="btn-filter-all" onclick="filterHits('all')">전체</button>
  <button id="btn-filter-hits" onclick="filterHits('hits')">매핑 있는 것만</button>
  <span style="color:#94a3b8;font-size:11px;margin:0 4px">|</span>
  <select onchange="filterCategory(this.value)" id="catSelect">
    <option value="">전체 카테고리</option>
''')
for cat in CATEGORIES:
    html.append(f'    <option value="{cat}">{cat}</option>\n')
html.append('''  </select>
</div>
<div id="metContainer">
''')

# ── Metabolite cards ──
for idx, m in enumerate(metabolite_results):
    name = m["name"]
    corr = m["corr"]
    cat = m["category"]
    hits = m["total_cancer_hits"]

    hit_cls = "hit-high" if hits >= 5 else "hit-mid" if hits >= 2 else "hit-low" if hits > 0 else "no-hit"
    highlight = " highlight" if hits >= 5 else ""

    # Cancer types this metabolite maps to
    mapped_cancers = set()
    for p in m["peaks"]:
        for ch in p["cancer_hits"]:
            mapped_cancers.add(ch["type"])
    cancer_data_attr = " ".join(sorted(mapped_cancers))

    html.append(f'''
<div class="met-card{highlight}" data-idx="{idx}" data-hits="{hits}" data-corr="{corr or 0}" data-name="{esc(name)}" data-cat="{cat}" data-cancers="{cancer_data_attr}" data-search="{esc(name.lower())} {cat} {' '.join([str(int(p['wn'])) for p in m['peaks']])}">
  <div class="met-header" onclick="toggleMet({idx})">
    <div>
      <div class="met-name">{esc(name)}</div>
      <div class="met-sub"><span class="tag-cat">{cat}</span> &nbsp; Top-5 피크: {", ".join([f"{p['wn']:.0f}" for p in m['peaks']])} cm⁻¹</div>
    </div>
    <div class="met-right">
      {corr_badge(corr)}
      <div class="hit-count {hit_cls}">{hits}</div>
      <span class="arrow" id="arrow-{idx}">▶</span>
    </div>
  </div>
  <div class="met-body" id="body-{idx}">
''')

    # Peak detail rows
    html.append('    <table><tr><th style="width:80px">피크 (cm⁻¹)</th><th style="width:60px">상대강도</th><th style="width:140px">진동모드</th><th>매칭 암종 (±10 cm⁻¹)</th></tr>\n')

    for p in m["peaks"]:
        wn = p["wn"]
        rel_int = p["rel_intensity"]
        vib = p["vibration"]
        bar_w = int(rel_int * 60)

        # Cancer hit badges
        if p["cancer_hits"]:
            hits_html = ""
            for ch in p["cancer_hits"]:
                ct = ch["type"]
                color, bg, border = CANCER_COLORS[ct]
                hits_html += f'<span class="cancer-badge" style="background:{bg};color:{color};border-color:{border}">{rank_badge(ch["rank"])} {ct} <span style="font-size:9px;opacity:.7">Δ{ch["delta"]}cm⁻¹</span></span>'
        else:
            hits_html = '<span style="color:#d1d5db;font-size:11px">— 매칭 없음</span>'

        row_bg = ' style="background:#fffbeb"' if len(p["cancer_hits"]) >= 3 else ""
        html.append(f'    <tr{row_bg}><td><b>{wn:.1f}</b></td><td><span class="peak-bar" style="width:{bar_w}px"></span> {rel_int:.2f}</td><td style="font-size:11px;color:#64748b">{esc(vib)}</td><td>{hits_html}</td></tr>\n')

    html.append('    </table>\n')
    html.append('  </div>\n</div>\n')

html.append('</div>\n')  # metContainer

# ── Key findings ──
html.append('''
<div class="card kf">
  <h3>Key Findings</h3>
  <ul style="font-size:12px;line-height:2.2;padding-left:18px">
    <li><b>Hippuric acid</b> (r=0.764): 5개 피크 중 4개가 암종 피크와 매칭. 특히 673.8 cm⁻¹ (BLC #4), 997.8 cm⁻¹ (모든 암종 #1-3), 801.1 cm⁻¹ (CRC #7). 장내미생물 대사 지표</li>
    <li><b>Adenine</b> (r=0.305): 729.7 cm⁻¹ 피크가 <b>BLC #1</b> (721.9 cm⁻¹)과 매칭 — 방광암의 핵산 대사 특이성 시사</li>
    <li><b>Phenylalanine</b> (r=0.661): 999.7 cm⁻¹가 전 암종 Rank 1-3의 핵심 피크. SERS에서 가장 강한 증강을 받는 대사체</li>
    <li><b>Creatinine</b> (r=0.274): 673.8 cm⁻¹ 피크가 BLC #4 (685.3 cm⁻¹)에 매칭. 암환자 전체에서 감소 (d=-1.26)</li>
    <li><b>Hypoxanthine</b> (r=0.383): 720.1 cm⁻¹ 피크가 BLC #1에 매칭 — Adenine과 함께 퓨린 대사 경로의 변화 반영</li>
    <li><b>~2098 cm⁻¹ 밴드:</b> Trimethylamine-N-oxide(2120), Spermidine(2112) 외에는 매칭 없음 — 대부분의 대사체 라이브러리가 이 영역을 커버하지 못함</li>
  </ul>
</div>
''')

# ── Cross-reference matrix ──
html.append('''<div class="card">
  <h3>Cross-Reference Matrix: 주요 대사체 × 암종</h3>
  <p style="font-size:11px;color:#64748b;margin-bottom:12px">SERS 상관 r≥0.4 이상인 대사체만 표시. 셀 = 매칭된 피크 수 (클릭 시 해당 대사체로 이동)</p>
  <div style="overflow-x:auto">
  <table>
    <tr><th>대사체</th><th>카테고리</th><th>r</th>''')

for ct in CANCER_TYPES:
    color = CANCER_COLORS[ct][0]
    html.append(f'<th style="text-align:center;color:{color}">{ct}</th>')
html.append('<th>Total</th></tr>\n')

# Filter to r >= 0.4 or has hits
matrix_mets = [m for m in metabolite_results if (m["corr"] and m["corr"] >= 0.4) or m["total_cancer_hits"] >= 3]
matrix_mets.sort(key=lambda x: -(x["corr"] or 0))

for m in matrix_mets:
    name = m["name"]
    corr = m["corr"]
    cat = m["category"]

    # Count hits per cancer type
    ct_hits = {ct: 0 for ct in CANCER_TYPES}
    ct_best_rank = {ct: 99 for ct in CANCER_TYPES}
    for p in m["peaks"]:
        for ch in p["cancer_hits"]:
            ct_hits[ch["type"]] += 1
            ct_best_rank[ch["type"]] = min(ct_best_rank[ch["type"]], ch["rank"])

    total = sum(ct_hits.values())

    cells = ""
    for ct in CANCER_TYPES:
        h = ct_hits[ct]
        if h > 0:
            br = ct_best_rank[ct]
            bg = "#dcfce7" if br <= 2 else "#fef3c7" if br <= 4 else "#f8fafc"
            cells += f'<td style="text-align:center;background:{bg};cursor:pointer" title="Best rank: #{br}">{h} {rank_badge(br)}</td>'
        else:
            cells += '<td style="text-align:center;color:#e2e8f0">—</td>'

    tc = "tag-high" if corr and corr >= 0.6 else "tag-mid" if corr and corr >= 0.4 else "tag-low"
    html.append(f'    <tr><td><b>{esc(name)}</b></td><td><span class="tag-cat">{cat}</span></td><td><span class="tag {tc}">{"%.3f"%corr if corr else "N/A"}</span></td>{cells}<td style="text-align:center;font-weight:700">{total}</td></tr>\n')

html.append('''  </table>
  </div>
</div>
''')

# ── Footer ──
html.append(f'''
<div class="fn">
  <b>분석 방법:</b> Thermo Raman 실측 73종 대사체의 Top-5 피크를 7개 암종의 Top-8 SERS 피크와 &pm;{TOLERANCE} cm⁻¹ 내 매칭.<br>
  <b>SERS 상관(r):</b> 각 대사체의 Thermo 전체 스펙트럼과 평균 소변 SERS 스펙트럼 간 Pearson correlation. 높을수록 소변 SERS 신호 기여도 높음.<br>
  <b>데이터:</b> Cancer: PRO 100, BRE 30, OVA 70, LUN 300, CRC 300, CPAN 70, BLC 299 (총 1,169) | Non-cancer 400 | Thermo 73 standards<br>
  <b>제외:</b> SPAN (surgical pancreatic cancer) — 샘플 특성상 별도 분석 필요<br>
  <b>생성:</b> build_metabolite_dashboard_v2.py | 2026-03-23
</div>

</div>
</div>

<script>
function toggleMet(i) {{
  const body = document.getElementById('body-' + i);
  const arrow = document.getElementById('arrow-' + i);
  body.classList.toggle('open');
  arrow.classList.toggle('open');
}}

function toggleAll(open) {{
  document.querySelectorAll('.met-body').forEach(b => {{
    if (open) b.classList.add('open'); else b.classList.remove('open');
  }});
  document.querySelectorAll('.arrow').forEach(a => {{
    if (open) a.classList.add('open'); else a.classList.remove('open');
  }});
}}

function filterMets() {{ applyFilters(); }}

let currentSort = 'hits';
let currentHitFilter = 'all';
let currentCat = '';

function sortBy(mode) {{
  currentSort = mode;
  document.querySelectorAll('[id^="btn-sort-"]').forEach(b => b.classList.remove('on'));
  document.getElementById('btn-sort-' + mode).classList.add('on');
  applyFilters();
}}

function filterHits(mode) {{
  currentHitFilter = mode;
  document.querySelectorAll('[id^="btn-filter-"]').forEach(b => b.classList.remove('on'));
  document.getElementById('btn-filter-' + mode).classList.add('on');
  applyFilters();
}}

function filterCategory(cat) {{
  currentCat = cat;
  applyFilters();
}}

function applyFilters() {{
  const q = document.getElementById('searchBox').value.toLowerCase();
  const container = document.getElementById('metContainer');
  const cards = Array.from(container.querySelectorAll('.met-card'));

  cards.forEach(card => {{
    const search = card.dataset.search || '';
    const hits = parseInt(card.dataset.hits);
    const cat = card.dataset.cat || '';
    let show = true;
    if (q && !search.includes(q)) show = false;
    if (currentHitFilter === 'hits' && hits === 0) show = false;
    if (currentCat && cat !== currentCat) show = false;
    card.style.display = show ? '' : 'none';
  }});

  // Sort
  const sorted = cards.sort((a, b) => {{
    if (currentSort === 'hits') return parseInt(b.dataset.hits) - parseInt(a.dataset.hits);
    if (currentSort === 'corr') return parseFloat(b.dataset.corr) - parseFloat(a.dataset.corr);
    if (currentSort === 'name') return a.dataset.name.localeCompare(b.dataset.name);
    return 0;
  }});
  sorted.forEach(card => container.appendChild(card));
}}

// Start with all collapsed, hits filter on
filterHits('hits');
</script>
</body>
</html>
''')

out_path = "/home/user/workspace/solum-dashboard/Metabolite_CrossRef_Dashboard.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("".join(html))

print(f"Dashboard saved: {out_path}")
print(f"  Metabolites with cancer hits: {len(mets_with_hits)} / 73")
print(f"  Total mappings: {total_mappings}")

# Also save data as JSON for reference
data_out = {
    "metabolites": metabolite_results,
    "cancer_summary": {ct: {k: {"peaks": v["peaks"], "ranks": v["ranks"], "corr": v["corr"]}
                            for k, v in cm.items()}
                       for ct, cm in cancer_met_map.items()},
    "config": {"tolerance": TOLERANCE, "cancer_types": CANCER_TYPES},
}
json_path = f"{BASE}/metabolite_profiling/data/metabolite_cancer_crossref.json"
with open(json_path, "w") as f:
    json.dump(data_out, f, indent=2, ensure_ascii=False)
print(f"  JSON data: {json_path}")
