"""
Build wavenumber-centric metabolite cross-reference dashboard.
For each SERS peak position: which metabolites match + which cancer types show it as important.
"""

import pandas as pd
import numpy as np
import json
import html as html_mod

BASE = "/home/user/SERS-AI"
PEAK_FILE = f"{BASE}/results/training/R_BLC_added/logistic_regression/v001/evaluation/stage2/peak_intensity_by_diagnosis.csv"
ASSIGN_FILE = f"{BASE}/metabolite_profiling/data/metabolite_peak_assignments.csv"
CORR_FILE = f"{BASE}/metabolite_profiling/data/metabolite_sers_correlation.csv"
THERMO_FILE = f"{BASE}/thermo_metabolite_data.json"
BAND_FILE = f"{BASE}/metabolite_profiling/data/metabolite_band_statistics.csv"

# ── Load ──
peaks_df = pd.read_csv(PEAK_FILE)
assign_df = pd.read_csv(ASSIGN_FILE)
corr_df = pd.read_csv(CORR_FILE)
with open(THERMO_FILE) as f:
    thermo_data = json.load(f)

corr_lookup = dict(zip(corr_df["metabolite"], corr_df["pearson_correlation"]))

# ── Build wavenumber grid ──
# Collect all unique SERS peak wavenumbers from cancer-type peaks
all_wns = sorted(peaks_df["wavenumber"].unique())

# Group nearby wavenumbers into bands (within 8 cm-1)
bands = []
used = set()
for wn in all_wns:
    if wn in used:
        continue
    group = [wn]
    for wn2 in all_wns:
        if wn2 != wn and abs(wn2 - wn) < 8 and wn2 not in used:
            group.append(wn2)
            used.add(wn2)
    used.add(wn)
    bands.append({
        "center": round(np.mean(group), 1),
        "members": sorted(group),
        "range": (min(group) - 5, max(group) + 5),
    })

# Known vibration assignments
VIBRATION_MAP = {
    446: ("C-C-O / Ring def", "단백질/당 골격 진동"),
    538: ("S-S stretch", "이황화 결합 (시스틴)"),
    618: ("C-S stretch", "시스테인, 메티오닌 (산화 스트레스)"),
    683: ("Ring breathing", "크레아티닌, 구아닌, 퓨린"),
    722: ("Adenine ring", "아데닌 고리 진동 (핵산 대사)"),
    797: ("Ring breathing", "히푸르산, 키뉴레닌 (장내미생물)"),
    847: ("Tyr Fermi", "타이로신/트립토판 (방향족 아미노산)"),
    895: ("C-C stretch", "히푸르산, TMAO, 요산"),
    934: ("C-C stretch", "단백질 골격 (비특이적)"),
    999: ("Phe ring breathing", "페닐알라닌 (가장 강한 SERS 피크)"),
    1148: ("C-N / C-O-C", "글리코겐, 포도당, 자일로스"),
    1294: ("Amide III / CH₂ twist", "O-아세틸카르니틴, 지방산"),
    1350: ("CH def / Trp", "아데닌, 구아닌, 트립토판"),
    1428: ("CH₂ scissor", "지질, 지방산"),
    1449: ("CH₂ deformation", "지질/지방산 (비특이적, 거의 모든 대사체)"),
    1597: ("C=C / Purine ring", "아데닌, 키뉴레닌, 타이로신"),
    1645: ("Amide I (C=O)", "말레산, 글리코겐, 키뉴레닌"),
    1680: ("C=O stretch", "구아닌, 우라실, NADH"),
    2098: ("S-H / Unknown", "미확인 (S-H stretch 또는 기기 artifact)"),
}

def get_vibration(center):
    best_key = min(VIBRATION_MAP.keys(), key=lambda k: abs(k - center))
    if abs(best_key - center) < 15:
        return VIBRATION_MAP[best_key]
    return ("unassigned", "미확인")

TOLERANCE = 12  # cm-1

# ── For each band: find metabolites + cancer types ──
band_data = []
for band in bands:
    center = band["center"]
    lo, hi = band["range"]
    vib_en, vib_kr = get_vibration(center)

    # Cancer types that have this peak
    cancer_peaks = peaks_df[peaks_df["wavenumber"].isin(band["members"])].copy()
    cancer_info = []
    for _, row in cancer_peaks.sort_values("peak_rank").iterrows():
        cancer_info.append({
            "type": row["diagnosis"],
            "rank": int(row["peak_rank"]),
            "intensity": round(row["peak_intensity"], 2),
            "prominence": round(row["prominence"], 2),
        })

    # Metabolite candidates from assignment table
    mask = assign_df["sers_peak_cm1"].between(lo, hi)
    metabolites_raw = assign_df[mask][["metabolite", "metabolite_peak_cm1", "delta_cm1", "vibration_mode"]].copy()
    metabolites_raw["sers_corr"] = metabolites_raw["metabolite"].map(corr_lookup)

    # Also check Thermo peak data directly
    for m in thermo_data["metabolites"]:
        for p in m["top_peaks"]:
            if lo <= p["wn"] <= hi:
                name = m["name"]
                if name not in metabolites_raw["metabolite"].values:
                    metabolites_raw = pd.concat([metabolites_raw, pd.DataFrame([{
                        "metabolite": name,
                        "metabolite_peak_cm1": p["wn"],
                        "delta_cm1": abs(p["wn"] - center),
                        "vibration_mode": "from Thermo top peaks",
                        "sers_corr": corr_lookup.get(name, None),
                    }])], ignore_index=True)

    # Deduplicate by metabolite, keep closest match
    if len(metabolites_raw) > 0:
        metabolites_raw = metabolites_raw.sort_values("delta_cm1").drop_duplicates("metabolite", keep="first")
        metabolites_raw = metabolites_raw.sort_values("sers_corr", ascending=False)

    met_list = []
    for _, m in metabolites_raw.iterrows():
        met_list.append({
            "name": m["metabolite"],
            "thermo_peak": round(m["metabolite_peak_cm1"], 1),
            "delta": round(m["delta_cm1"], 1),
            "corr": round(m["sers_corr"], 3) if pd.notna(m["sers_corr"]) else None,
            "vibration": m["vibration_mode"] if pd.notna(m["vibration_mode"]) else "",
        })

    band_data.append({
        "center": center,
        "range_lo": round(lo, 1),
        "range_hi": round(hi, 1),
        "vibration_en": vib_en,
        "vibration_kr": vib_kr,
        "cancer_types": cancer_info,
        "metabolites": met_list,
        "n_metabolites": len(met_list),
        "n_cancers": len(set(c["type"] for c in cancer_info)),
    })

# Sort by center
band_data.sort(key=lambda x: x["center"])

# ── Statistics ──
total_bands = len(band_data)
total_met_matches = sum(b["n_metabolites"] for b in band_data)
all_mets = set()
for b in band_data:
    for m in b["metabolites"]:
        all_mets.add(m["name"])
total_unique_mets = len(all_mets)

# ── Build HTML ──
def esc(s):
    return html_mod.escape(str(s))

def corr_class(c):
    if c is None: return "peak-low"
    if c >= 0.6: return "peak-high"
    if c >= 0.4: return "peak-mid"
    return "peak-low"

def rank_badge(rank):
    colors = {1: "#dc2626", 2: "#ea580c", 3: "#d97706", 4: "#ca8a04", 5: "#65a30d", 6: "#059669", 7: "#0891b2", 8: "#6366f1"}
    c = colors.get(rank, "#94a3b8")
    return f'<span style="display:inline-block;min-width:18px;text-align:center;padding:1px 5px;border-radius:10px;font-size:10px;font-weight:700;background:{c};color:#fff">#{rank}</span>'

html_parts = []

# Header
html_parts.append(f'''<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Metabolite Peak Cross-Reference Dashboard</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Noto Sans KR',sans-serif;color:#333;background:#f0f2f5;line-height:1.6;font-size:14px}}
  .container{{max-width:1400px;margin:0 auto;padding:20px}}
  .header{{background:linear-gradient(135deg,#0f172a,#1e3a5f);color:#fff;padding:40px;border-radius:16px;margin-bottom:24px}}
  .header h1{{font-size:22px;font-weight:700;margin-bottom:4px}}
  .header p{{font-size:13px;opacity:.8}}
  .htags{{display:flex;gap:10px;margin-top:14px;flex-wrap:wrap}}
  .htags span{{background:rgba(255,255,255,.15);padding:3px 10px;border-radius:16px;font-size:11px}}
  .metrics{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px}}
  .mc{{background:#fff;border-radius:12px;padding:20px;text-align:center;box-shadow:0 1px 3px rgba(0,0,0,.06)}}
  .mc .v{{font-size:32px;font-weight:700;color:#1e3a5f}}
  .mc .l{{font-size:11px;color:#94a3b8;margin-top:2px}}
  .card{{background:#fff;border-radius:12px;padding:24px;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:20px}}
  .card h3{{font-size:15px;color:#1e3a5f;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid #e2e8f0}}
  .band-card{{background:#fff;border-radius:12px;padding:0;box-shadow:0 1px 3px rgba(0,0,0,.06);margin-bottom:16px;overflow:hidden;border:1px solid #e2e8f0}}
  .band-header{{display:flex;justify-content:space-between;align-items:center;padding:16px 20px;cursor:pointer;transition:background .15s}}
  .band-header:hover{{background:#f8fafc}}
  .band-header .wn{{font-size:20px;font-weight:700;color:#1e3a5f}}
  .band-header .wn small{{font-size:12px;font-weight:400;color:#64748b;margin-left:6px}}
  .band-header .vib{{font-size:12px;color:#64748b;margin-top:2px}}
  .band-header .badges{{display:flex;gap:4px;flex-wrap:wrap;align-items:center}}
  .band-body{{padding:0 20px 20px;display:none}}
  .band-body.open{{display:block}}
  .sub-grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}}
  .sub-section h4{{font-size:13px;color:#475569;margin-bottom:8px;font-weight:600}}
  table{{width:100%;border-collapse:collapse;font-size:12px}}
  th{{background:#1e3a5f;color:#fff;padding:6px 8px;text-align:left;font-weight:500;font-size:11px}}
  td{{padding:5px 8px;border-bottom:1px solid #f1f5f9}}
  tr:hover td{{background:#f8fafc}}
  .peak-tag{{display:inline-block;padding:1px 6px;border-radius:8px;font-size:10px;font-weight:600;margin:1px;white-space:nowrap}}
  .peak-high{{background:#dcfce7;color:#16a34a}}
  .peak-mid{{background:#fef3c7;color:#b45309}}
  .peak-low{{background:#f3f4f6;color:#6b7280}}
  .corr-bar{{display:inline-block;height:8px;border-radius:4px;vertical-align:middle;margin-left:4px}}
  .cancer-badge{{display:inline-flex;align-items:center;gap:3px;padding:2px 8px;border-radius:12px;font-size:11px;font-weight:500;margin:2px;border:1px solid}}
  .cb-PRO{{background:#eff6ff;color:#2563eb;border-color:#bfdbfe}}
  .cb-BRE{{background:#fdf2f8;color:#db2777;border-color:#fbcfe8}}
  .cb-OVA{{background:#faf5ff;color:#9333ea;border-color:#e9d5ff}}
  .cb-LUN{{background:#f0fdf4;color:#16a34a;border-color:#bbf7d0}}
  .cb-CRC{{background:#fff7ed;color:#ea580c;border-color:#fed7aa}}
  .cb-CPAN{{background:#fefce8;color:#ca8a04;border-color:#fef08a}}
  .cb-SPAN{{background:#fef2f2;color:#dc2626;border-color:#fecaca}}
  .cb-BLC{{background:#ecfeff;color:#0891b2;border-color:#a5f3fc}}
  .legend{{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:16px;padding:12px;background:#f8fafc;border-radius:8px;font-size:11px}}
  .legend-item{{display:flex;align-items:center;gap:4px}}
  .arrow{{transition:transform .2s;font-size:12px;color:#94a3b8}}
  .arrow.open{{transform:rotate(90deg)}}
  .controls{{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}}
  .controls button{{padding:6px 14px;border:1px solid #e2e8f0;border-radius:8px;background:#fff;cursor:pointer;font-size:12px;font-family:inherit;transition:all .15s}}
  .controls button:hover{{border-color:#1e3a5f;background:#f8fafc}}
  .controls button.on{{background:#1e3a5f;color:#fff;border-color:#1e3a5f}}
  .search{{width:100%;padding:10px 14px;border:1px solid #e2e8f0;border-radius:8px;font-size:13px;margin-bottom:12px;font-family:inherit}}
  .search:focus{{outline:none;border-color:#1e3a5f;box-shadow:0 0 0 3px rgba(30,58,95,.1)}}
  .sig-row{{background:#fffbeb!important}}
  .bio-note{{font-size:11px;color:#64748b;margin-top:6px;padding:8px 12px;background:#f8fafc;border-radius:6px;border-left:3px solid #3b82f6}}
  .fn{{font-size:10px;color:#94a3b8;margin-top:20px;padding-top:10px;border-top:1px solid #e2e8f0;line-height:1.8}}
  @media(max-width:900px){{.sub-grid{{grid-template-columns:1fr}}.metrics{{grid-template-columns:repeat(2,1fr)}}}}
</style>
</head>
<body>
<div class="container">

<div class="header">
  <h1>SERS Peak &harr; Metabolite Cross-Reference</h1>
  <p>Wavenumber Grid 기준: 각 SERS 피크 위치에서 Thermo 실측 대사체 후보군 &amp; 암종별 피크 중요도</p>
  <div class="htags">
    <span>SOLUM Healthcare</span><span>AECD Platform</span><span>Thermo &times; SERS</span>
    <span>{total_bands} Wavenumber Bands</span><span>8 Cancer Types</span><span>2026-03-23</span>
  </div>
</div>

<div class="metrics">
  <div class="mc"><div class="v">{total_bands}</div><div class="l">Wavenumber Bands</div></div>
  <div class="mc"><div class="v">{total_unique_mets}</div><div class="l">매칭 대사체 (Unique)</div></div>
  <div class="mc"><div class="v">{total_met_matches}</div><div class="l">총 피크-대사체 매칭</div></div>
  <div class="mc"><div class="v">8</div><div class="l">암종</div></div>
</div>

<div class="card">
  <h3>사용 방법</h3>
  <p style="font-size:12px;color:#64748b">각 wavenumber band를 클릭하면 상세 정보가 펼쳐집니다. 왼쪽에는 매칭된 Thermo 대사체 후보군, 오른쪽에는 해당 피크를 주요 피크로 가진 암종 목록이 표시됩니다.</p>
  <div class="legend" style="margin-top:12px">
    <div class="legend-item"><span class="peak-tag peak-high">r &ge; 0.6</span> 높은 SERS 상관</div>
    <div class="legend-item"><span class="peak-tag peak-mid">r 0.4&ndash;0.6</span> 중간</div>
    <div class="legend-item"><span class="peak-tag peak-low">r &lt; 0.4</span> 낮음</div>
    <div class="legend-item" style="margin-left:16px">Rank: 해당 암종 내 피크 중요도 순위</div>
  </div>
</div>

<input type="text" class="search" id="searchBox" placeholder="검색: 대사체명, wavenumber, 암종코드 (예: Hippuric, 724, BLC)..." oninput="filterBands()">

<div class="controls">
  <button class="on" onclick="toggleAll(true)">모두 펼치기</button>
  <button onclick="toggleAll(false)">모두 접기</button>
  <button onclick="filterCancer('')" class="on" id="btn-all">전체 암종</button>
  <button onclick="filterCancer('PRO')" id="btn-PRO" class="cancer-badge cb-PRO" style="border-radius:8px">PRO</button>
  <button onclick="filterCancer('BRE')" id="btn-BRE" class="cancer-badge cb-BRE" style="border-radius:8px">BRE</button>
  <button onclick="filterCancer('OVA')" id="btn-OVA" class="cancer-badge cb-OVA" style="border-radius:8px">OVA</button>
  <button onclick="filterCancer('LUN')" id="btn-LUN" class="cancer-badge cb-LUN" style="border-radius:8px">LUN</button>
  <button onclick="filterCancer('CRC')" id="btn-CRC" class="cancer-badge cb-CRC" style="border-radius:8px">CRC</button>
  <button onclick="filterCancer('CPAN')" id="btn-CPAN" class="cancer-badge cb-CPAN" style="border-radius:8px">CPAN</button>
  <button onclick="filterCancer('SPAN')" id="btn-SPAN" class="cancer-badge cb-SPAN" style="border-radius:8px">SPAN</button>
  <button onclick="filterCancer('BLC')" id="btn-BLC" class="cancer-badge cb-BLC" style="border-radius:8px">BLC</button>
</div>

<div id="bandContainer">
''')

# ── Band cards ──
for i, bd in enumerate(band_data):
    center = bd["center"]
    vib_en = bd["vibration_en"]
    vib_kr = bd["vibration_kr"]

    # Cancer badges
    cancer_badges = ""
    cancer_codes = set()
    for c in sorted(bd["cancer_types"], key=lambda x: x["rank"]):
        ct = c["type"]
        cancer_codes.add(ct)
        cancer_badges += f'<span class="cancer-badge cb-{ct}">{rank_badge(c["rank"])} {ct}</span>'

    cancer_data_attr = " ".join(sorted(cancer_codes))

    # Met names for search
    met_names = " ".join(m["name"] for m in bd["metabolites"])

    html_parts.append(f'''
<div class="band-card" data-cancers="{cancer_data_attr}" data-search="{center} {vib_en} {vib_kr} {met_names} {cancer_data_attr}">
  <div class="band-header" onclick="toggleBand({i})">
    <div>
      <div class="wn">{center} cm⁻¹<small>({bd["range_lo"]}–{bd["range_hi"]})</small></div>
      <div class="vib">{esc(vib_en)} — {esc(vib_kr)}</div>
    </div>
    <div style="text-align:right">
      <div class="badges">{cancer_badges}</div>
      <div style="font-size:11px;color:#94a3b8;margin-top:4px">{bd["n_metabolites"]}개 대사체 매칭</div>
    </div>
    <span class="arrow" id="arrow-{i}">▶</span>
  </div>
  <div class="band-body" id="body-{i}">
    <div class="sub-grid">
      <div class="sub-section">
        <h4>Thermo 대사체 후보군 (SERS 상관 순)</h4>
        <div style="max-height:350px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:6px">
        <table>
          <tr><th>대사체</th><th>Thermo 피크</th><th>Δ cm⁻¹</th><th>SERS 상관 (r)</th></tr>
''')

    for m in bd["metabolites"][:20]:
        corr_val = f'{m["corr"]:.3f}' if m["corr"] is not None else "N/A"
        cc = corr_class(m["corr"])
        bar_w = int((m["corr"] or 0) * 80)
        bar_color = "#16a34a" if (m["corr"] or 0) >= 0.6 else "#d97706" if (m["corr"] or 0) >= 0.4 else "#94a3b8"
        sig_class = ' class="sig-row"' if (m["corr"] or 0) >= 0.6 else ""
        html_parts.append(f'''          <tr{sig_class}>
            <td><b>{esc(m["name"])}</b></td>
            <td>{m["thermo_peak"]}</td>
            <td>{m["delta"]}</td>
            <td><span class="peak-tag {cc}">{corr_val}</span><span class="corr-bar" style="width:{bar_w}px;background:{bar_color}"></span></td>
          </tr>
''')

    if bd["n_metabolites"] > 20:
        html_parts.append(f'          <tr><td colspan="4" style="color:#94a3b8;text-align:center;font-style:italic">+{bd["n_metabolites"]-20}개 추가 대사체 (생략)</td></tr>\n')

    html_parts.append('''        </table>
        </div>
      </div>
      <div class="sub-section">
        <h4>암종별 피크 중요도</h4>
        <div style="max-height:350px;overflow-y:auto;border:1px solid #e2e8f0;border-radius:6px">
        <table>
          <tr><th>암종</th><th>Rank</th><th>Intensity</th><th>Prominence</th></tr>
''')

    for c in sorted(bd["cancer_types"], key=lambda x: x["rank"]):
        html_parts.append(f'''          <tr>
            <td><span class="cancer-badge cb-{c["type"]}">{c["type"]}</span></td>
            <td>{rank_badge(c["rank"])}</td>
            <td>{c["intensity"]}</td>
            <td>{c["prominence"]}</td>
          </tr>
''')

    if not bd["cancer_types"]:
        html_parts.append('          <tr><td colspan="4" style="color:#94a3b8;text-align:center">해당 없음</td></tr>\n')

    # Bio note
    bio_notes = {
        "Phe ring breathing": "Phenylalanine의 벤젠 고리 진동. 가장 강한 SERS 피크로, 모든 암종에서 Rank 1–3. 2-Phenylacetamide, Hippuric acid도 이 영역에서 강한 신호를 보임.",
        "Adenine ring": "핵산 대사 지표. Adenine, Hypoxanthine 등 퓨린 대사체. 암환자에서 감소 (d=-1.13) — 종양 세포의 퓨린 소비 증가 반영. BLC에서 특이적으로 Rank 1.",
        "C-S stretch": "시스테인, 글루타치온 등 티올 화합물. 암환자에서 증가 (d=+0.76) — 산화 스트레스 반응. LUN에서 특히 두드러짐.",
        "Ring breathing": "크레아티닌 관련. 암환자에서 가장 큰 감소 (d=-1.26) — 신장 기능 변화 또는 악액질(cachexia). BLC에서 Rank 4.",
        "Tyr Fermi": "타이로신 Fermi 공명. PRO, BRE에서 높음 (단백질 대사 우위 패턴).",
        "C-C stretch": "히푸르산, TMAO 관련. 장내미생물 대사. CRC, SPAN에서 두드러짐.",
        "CH₂ deformation": "지질/지방산의 비특이적 진동. 거의 모든 대사체가 이 영역에 피크를 가짐. 단독으로 대사체 특정 불가.",
        "C=C / Purine ring": "아데닌, 키뉴레닌, 타이로신. 암환자에서 감소 (d=-0.93). 퓨린 대사 경로 변화 반영.",
        "Amide I (C=O)": "단백질 2차 구조 (α-helix, β-sheet). PRO, OVA, BRE에서 높음 — 단백질 대사 우위 클러스터.",
        "C=O stretch": "구아닌, 우라실, NADH. BLC에서 특이적으로 Rank 6에 등장 (다른 암종에는 없음).",
        "S-H / Unknown": "Thermo 대사체 라이브러리에서 매칭 없음. S-H stretch 또는 기기 artifact 가능성. 추가 검증 필요.",
        "CH def / Trp": "아데닌, 구아닌, 트립토판의 CH 변형 진동. CRC에서 Rank 3으로 높음.",
        "Amide III / CH₂ twist": "O-아세틸카르니틴, 지방산. OVA에서 Rank 7.",
        "CH₂ scissor": "지질 관련. BRE에서 Rank 4.",
    }
    note = bio_notes.get(vib_en, "")
    if note:
        html_parts.append(f'''        </table>
        </div>
        <div class="bio-note">{note}</div>
''')
    else:
        html_parts.append('''        </table>
        </div>
''')

    html_parts.append('''      </div>
    </div>
  </div>
</div>
''')

# ── Summary table card ──
html_parts.append('''
<div class="card">
  <h3>요약: 주요 SERS 밴드 &times; 암종 매트릭스</h3>
  <p style="font-size:11px;color:#64748b;margin-bottom:12px">셀 내 숫자 = 해당 암종에서의 피크 순위 (Rank). 색상 = 순위 (1위 빨강 → 8위 보라)</p>
  <div style="overflow-x:auto">
  <table>
    <tr><th>Wavenumber</th><th>Vibration</th><th>PRO</th><th>BRE</th><th>OVA</th><th>LUN</th><th>CRC</th><th>CPAN</th><th>SPAN</th><th>BLC</th><th>#대사체</th></tr>
''')

for bd in band_data:
    center = bd["center"]
    vib = bd["vibration_en"]

    # Build rank cells
    cancer_rank = {}
    for c in bd["cancer_types"]:
        cancer_rank[c["type"]] = c["rank"]

    cells = ""
    for ct in ["PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "SPAN", "BLC"]:
        if ct in cancer_rank:
            cells += f'<td style="text-align:center">{rank_badge(cancer_rank[ct])}</td>'
        else:
            cells += '<td style="text-align:center;color:#e2e8f0">—</td>'

    top_met = bd["metabolites"][0]["name"] if bd["metabolites"] else "—"
    html_parts.append(f'    <tr><td><b>{center}</b></td><td style="font-size:11px">{esc(vib)}</td>{cells}<td style="text-align:center">{bd["n_metabolites"]}</td></tr>\n')

html_parts.append('''  </table>
  </div>
</div>

<div class="card" style="border-left:4px solid #f59e0b;background:linear-gradient(135deg,#fffbeb,#fef3c7)">
  <h3 style="color:#b45309;border-bottom-color:#f59e0b">Key Findings</h3>
  <ul style="font-size:12px;line-height:2;padding-left:18px">
    <li><b>BLC 특이점:</b> 721.9 cm⁻¹ (Adenine) 피크가 Rank 1 — 다른 암종은 모두 ~1001 cm⁻¹ (Phe)이 최상위. 또한 1680 cm⁻¹ (C=O, Guanine/Uracil/NADH) 피크가 BLC에서만 Rank 6에 등장</li>
    <li><b>Gut-microbiome 클러스터</b> (CRC, SPAN): 895 cm⁻¹ (Hippuric/TMAO) 피크 활성 + 797 cm⁻¹ (Hippuric) 피크가 CRC Rank 7</li>
    <li><b>Protein-metabolism 클러스터</b> (PRO, BRE, OVA): 849 cm⁻¹ (Tyr/Trp) + 1647 cm⁻¹ (Amide I) 피크가 Rank 5–6에 공통 등장</li>
    <li><b>~2098 cm⁻¹:</b> 모든 암종에서 높은 순위를 차지하나, Thermo 73종 대사체 중 매칭 없음 → S-H stretch 또는 기기 artifact 가능성, 추가 검증 필요</li>
    <li><b>CH₂ def (1449 cm⁻¹):</b> 거의 모든 대사체가 피크를 가지는 비특이적 밴드 → 대사체 특정에는 부적합하나, 전체 지질 수준 지표로 활용 가능</li>
  </ul>
</div>

<div class="fn">
  <b>분석 방법:</b> Thermo Raman 실측 73종 대사체 피크를 SERS 피크 위치에 &pm;12 cm&sup1; 내 매칭. SERS 상관계수(r)는 각 대사체의 Thermo 스펙트럼과 평균 소변 SERS 스펙트럼 간 Pearson correlation.<br>
  <b>피크 순위:</b> 각 암종의 평균 스펙트럼에서 검출된 피크를 강도(intensity)와 돌출도(prominence) 기준으로 정렬한 순위.<br>
  <b>데이터:</b> Cancer n=1,141 (PRO 100, BRE 30, OVA 70, LUN 300, CRC 300, CPAN 70, SPAN 72, BLC 299) | Non-cancer n=400 (NOR 100, H.D. 100, HBP 100, DIA 100) | Metabolites: 73 Thermo standards<br>
  <b>생성:</b> build_crossref_dashboard.py | 2026-03-23
</div>

</div>
</div>

<script>
function toggleBand(i) {
  const body = document.getElementById('body-' + i);
  const arrow = document.getElementById('arrow-' + i);
  body.classList.toggle('open');
  arrow.classList.toggle('open');
}

function toggleAll(open) {
  document.querySelectorAll('.band-body').forEach(b => {
    if (open) b.classList.add('open'); else b.classList.remove('open');
  });
  document.querySelectorAll('.arrow').forEach(a => {
    if (open) a.classList.add('open'); else a.classList.remove('open');
  });
}

let activeFilter = '';
function filterCancer(ct) {
  activeFilter = ct;
  document.querySelectorAll('.controls button').forEach(b => b.classList.remove('on'));
  if (ct === '') {
    document.getElementById('btn-all').classList.add('on');
  } else {
    const btn = document.getElementById('btn-' + ct);
    if (btn) btn.classList.add('on');
  }
  applyFilters();
}

function filterBands() {
  applyFilters();
}

function applyFilters() {
  const q = document.getElementById('searchBox').value.toLowerCase();
  document.querySelectorAll('.band-card').forEach(card => {
    const cancers = card.dataset.cancers || '';
    const search = (card.dataset.search || '').toLowerCase();
    let show = true;
    if (activeFilter && !cancers.includes(activeFilter)) show = false;
    if (q && !search.includes(q)) show = false;
    card.style.display = show ? '' : 'none';
  });
}

// Start with all expanded
toggleAll(true);
</script>
</body>
</html>
''')

# Write
out_path = f"/home/user/workspace/solum-dashboard/Metabolite_CrossRef_Dashboard.html"
with open(out_path, "w", encoding="utf-8") as f:
    f.write("".join(html_parts))
print(f"Dashboard saved: {out_path}")
print(f"  Bands: {total_bands}")
print(f"  Unique metabolites: {total_unique_mets}")
print(f"  Total matches: {total_met_matches}")
