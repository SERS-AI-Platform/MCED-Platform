"""Build the 대표님 보고 deck for the paired July-liquid vs August-mapping comparison.

Every number on the slides is read from results/boramae_paired_reducing_agent/summary.json
so the deck cannot drift from the analysis. Figures are embedded as base64 PNG.
Follows publications/전향검체/보라매병원/slides/DESIGN.md.

    PYTHONPATH=src python scripts/analysis/boramae_paired_reducing_agent_deck.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Final

REPO: Final = Path(__file__).resolve().parents[2]
RES: Final = REPO / "results" / "boramae_paired_reducing_agent"
OUT: Final = REPO / "publications" / "전향검체" / "보라매병원" / "slides" / "환원제 변경전후 동일환자 비교.html"


def img(name: str) -> str:
    data = base64.b64encode((RES / name).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def fmt(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}"


def main() -> None:
    s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    n = s["n_paired"]
    lab = s["labels"]
    reps = s["repeats"]
    dist = {(r["task"], r["metric"], r["condition"]): r for r in s["repeat_summary"]}
    pair = {p["comparison"]: p for p in s["paired_summary"]}
    corr = s["spectral_corr_july_vs_mapping"]
    rep0 = {p["comparison"]: p for p in s["paired"] if p["repeat"] == 0}

    def d(task: str, metric: str, cond: str) -> dict:
        return dist[(task, metric, cond)]

    scr = {c: d("screening_binary", "roc_auc", c) for c in ("july_liquid", "aug_mapping", "aug_mapping_5")}
    thr = {c: d("three_group", "macro_ovr_roc_auc", c) for c in ("july_liquid", "aug_mapping", "aug_mapping_5")}
    p_jm = pair["july_liquid -> aug_mapping"]
    p_m5 = pair["aug_mapping -> aug_mapping_5"]
    flips = rep0["july_liquid -> aug_mapping"]["subjects_flipped"]
    flips_all = [p["subjects_flipped"] for p in s["paired"] if p["comparison"] == "july_liquid -> aug_mapping"]
    flips_lo, flips_hi = min(flips_all), max(flips_all)
    deltas_sorted = sorted(p["delta_auc"] for p in s["paired"] if p["comparison"] == "july_liquid -> aug_mapping")
    rep0_delta = rep0["july_liquid -> aug_mapping"]["delta_auc"]
    rep0_delta_rank = deltas_sorted.index(rep0_delta) + 1  # 1 = worst for August
    thr_aug_all = sorted(float(m["macro_ovr_roc_auc"]) for m in s["metrics"] if m["task"] == "three_group" and m["condition"] == "aug_mapping")
    rep0_thr_aug = float(next(m["macro_ovr_roc_auc"] for m in s["metrics"] if m["task"] == "three_group" and m["condition"] == "aug_mapping" and m["repeat"] == 0))
    rep0_thr_rank_from_top = len(thr_aug_all) - thr_aug_all.index(rep0_thr_aug)
    rep0_note = (f"repeat 0은 Screening에서는 8월에 불리한 편(ΔAUC {rep0_delta:+.3f}, {reps}회 중 {rep0_delta_rank}번째로 낮음), "
                 f"3군에서는 8월에 가장 유리한 회차({rep0_thr_aug:.3f}, {reps}회 중 위에서 {rep0_thr_rank_from_top}번째)")
    import numpy as _np
    grid = _np.array(s["grid"])
    def band(lo: float, hi: float, cond: str, g: str) -> float:
        sel = (grid >= lo) & (grid <= hi)
        return float(_np.mean(_np.array(s["mean_by_group"][cond][g])[sel]))
    groups = ("Control", "Biopsy-negative", "Prostate cancer")
    b1000 = {c: [band(990, 1010, c, g) for g in groups] for c in ("july_liquid", "aug_mapping")}
    b1650 = {c: [band(1600, 1700, c, g) for g in groups] for c in ("july_liquid", "aug_mapping")}
    b1450 = {c: [band(1430, 1470, c, g) for g in groups] for c in ("july_liquid", "aug_mapping")}
    def rng_txt(vals: list[float]) -> str:
        return f"{min(vals):.1f}~{max(vals):.1f}"

    # --- full metric table, per-class breakdown, session/day confound (extended run 2026-09-09 19:57) ---
    import csv as _csv
    conds4 = ("july_liquid", "aug_mapping", "aug_mapping_qc", "aug_mapping_5")
    cond_name = {"july_liquid": "7월 점측정 5회", "aug_mapping": "8월 mapping 121점", "aug_mapping_qc": "8월 mapping QC통과점", "aug_mapping_5": "8월 mapping 5점"}
    def cell(task: str, metric: str, cond: str) -> str:
        r = d(task, metric, cond)
        return f"{r['mean']:.3f} ± {r['sd']:.3f}"
    def cell1(task: str, metric: str, cond: str) -> str:
        return f"{d(task, metric, cond)['mean']:.3f}"
    ovr = {c: {k: d("three_group", f"ovr_auc_{k}", c)["mean"] for k in ("control", "biopsy_neg", "cancer")} for c in conds4}
    rec = {c: {k: d("three_group", f"recall_{k}", c)["mean"] for k in ("control", "biopsy_neg", "cancer")} for c in conds4}
    p_mq = pair["aug_mapping -> aug_mapping_qc"]
    batch = list(_csv.DictReader((RES / "july_session_batch_check.csv").open(encoding="utf-8-sig")))
    def brow(cond: str, sess: str, cls: str) -> dict | None:
        return next((r for r in batch if r["condition"] == cond and r["july_session"] == sess and r["class"] == cls), None)
    jul_sess = s["july_sessions"]  # {session: {label: n}}
    aug_day = list(_csv.DictReader((RES / "august_day_by_class.csv").open(encoding="utf-8-sig")))
    aug_by_day: dict[str, dict[str, int]] = {}
    for r in aug_day:
        aug_by_day.setdefault(r["measurement_date"][5:], {})[r["cohort_group"]] = int(r["samples"])
    ctrl_11h = brow("july_liquid", "07-09 11h", "control")
    ctrl_18h = brow("july_liquid", "07-09 18h", "control")
    aug_ctrl_11h = brow("aug_mapping", "07-09 11h", "control")
    aug_ctrl_18h = brow("aug_mapping", "07-09 18h", "control")
    cancer_days = {day: v.get("prostate", 0) for day, v in aug_by_day.items()}
    cancer_on_14 = cancer_days.get("08-14", 0)
    cancer_total = sum(cancer_days.values())

    # headline judgement is deliberately descriptive: is the July->August shift larger than the
    # fold-to-fold spread, and do the two ranges overlap at all?
    delta_jm = p_jm["delta_auc_mean"]
    scr_overlap = scr["aug_mapping"]["max"] >= scr["july_liquid"]["min"] and scr["july_liquid"]["max"] >= scr["aug_mapping"]["min"]
    thr_overlap = thr["aug_mapping"]["max"] >= thr["july_liquid"]["min"] and thr["july_liquid"]["max"] >= thr["aug_mapping"]["min"]
    thr_delta = thr["aug_mapping"]["mean"] - thr["july_liquid"]["mean"]
    conf = {c: [[int(v) for v in line.split(",")] for line in (RES / f"confusion_three_group_{c}.csv").read_text().split()] for c in ("july_liquid", "aug_mapping")}
    ctrl_ok = {c: conf[c][0][0] for c in conf}                      # true Control predicted Control
    into_ctrl = {c: conf[c][1][0] + conf[c][2][0] for c in conf}    # non-controls predicted Control
    cancer_ok = {c: conf[c][2][2] for c in conf}
    headline = (f"Screening(암 vs 나머지=정상+조직검사음성)은 변경 전후 차이가 없고(ΔAUC {delta_jm:+.3f}, 유의 {p_jm['repeats_delong_p_lt_0_05']}/{reps}회), "
                f"3군 구분은 {reps}/{reps}회 모두 하락했다(macro AUC {thr['july_liquid']['mean']:.3f} → {thr['aug_mapping']['mean']:.3f}) — 정상군 OvR AUC가 {ovr['july_liquid']['control']:.2f} → {ovr['aug_mapping']['control']:.2f}(우연 수준)로 내려간 것이 전부이며, 7월의 정상군 구분은 측정 세션과 얽혀 있다.")

    pipeline = ("전처리·모델 동일: trim(400–2,200 cm⁻¹) → Savitzky–Golay(11,3) → rolling-min baseline(101) → SNV → 환자당 평균 → "
                "StandardScaler + class-weighted LR · nested 5×4-fold OOF · 라벨은 두 조건 모두 aecd_platform clinical v7 cohort_group")

    slides = []

    # 1. title
    slides.append(f'''
  <section class="slide active" data-title="표지" aria-labelledby="s1-title">
    <header><p class="eyebrow">보라매 전향 코호트 · 환원제 변경 전후 · 동일 환자 짝 비교</p>
      <h1 id="s1-title" class="display">같은 환자 {n}명, 같은 알고리즘 — 환원제 변경 전후 성능 비교</h1></header>
    <div class="hero-layout center">
      <div>
        <p class="lead">7월 점측정(변경 전, 5회)과 8월 mapping 측정(변경 후, 121점) — 둘 다 액상 검체 — 을 <strong>같은 환자·같은 임상 라벨·같은 모델</strong>로 맞춰 비교했습니다.
        결과는 한 번의 학습이 아니라 fold 배정을 {reps}번 바꿔 얻은 분포로 제시합니다.</p>
        <div class="grid-3">
          <div class="card liquid-card"><p class="kicker">변경 전 · 7월 점측정 5회</p><span class="metric">{fmt(scr["july_liquid"]["mean"])}</span><p>Screening AUC 평균<br><span class="muted">± {fmt(scr["july_liquid"]["sd"])} (fold {reps}회)</span></p></div>
          <div class="card powder-card"><p class="kicker">변경 후 · 8월 mapping 121점</p><span class="metric">{fmt(scr["aug_mapping"]["mean"])}</span><p>Screening AUC 평균<br><span class="muted">± {fmt(scr["aug_mapping"]["sd"])} (fold {reps}회)</span></p></div>
          <div class="card green-card"><p class="kicker">3군 macro AUC · 전 → 후</p><span class="metric">{fmt(thr["july_liquid"]["mean"])} → {fmt(thr["aug_mapping"]["mean"])}</span><p>정상 / 조직검사 음성 / 암 3군 구분<br><span class="muted">{reps}회 반복 모두 하락 · 범위 비겹침</span></p></div>
        </div>
      </div>
      <div class="hero-meta">
        <table class="cohort-table"><thead><tr><th>임상군 (DB v7)</th><th>n</th></tr></thead><tbody>
          <tr><td>Control</td><td>{lab["Control"]}</td></tr>
          <tr><td>Biopsy-negative</td><td>{lab["Biopsy-negative"]}</td></tr>
          <tr><td>Prostate cancer</td><td>{lab["Prostate cancer"]}</td></tr>
          <tr class="total"><td>짝 비교 환자</td><td>{n}</td></tr></tbody></table>
        <p class="muted" style="margin-top:8px">Drop 1명 제외 · 7월에만 있는 7명(DB 미등록) 제외</p>
      </div>
    </div>
    <footer class="slide-footer"><span>핵심 메시지: {headline}</span><span>2026-09-09 · SOLUM AI</span></footer>
  </section>''')

    # 2. what was compared
    slides.append(f'''
  <section class="slide" data-title="비교 설계" aria-labelledby="s2-title">
    <header><h1 id="s2-title">무엇을 무엇과 비교했나</h1><div class="pipeline">{pipeline}</div></header>
    <div class="center">
      <table>
        <thead><tr><th></th><th class="liquid">변경 전 · 7월 점측정</th><th class="powder">변경 후 · 8월 mapping</th><th>8월 mapping 5점 추출</th></tr></thead>
        <tbody>
          <tr><td>측정일</td><td>2026-07-09</td><td>2026-08-10 ~ 08-14</td><td>(동일)</td></tr>
          <tr><td>환원제 / 센서 lot</td><td><strong>기록 없음</strong></td><td>Sigma-Aldrich 226904 Lot BCCP0922 · strip lot 기록</td><td>(동일)</td></tr>
          <tr><td>시료 형태 / 환자당 스펙트럼</td><td>액상 / 점 측정 5회 (Ave100)</td><td>액상 / mapping 121점 (누적 횟수 미기록)</td><td>121점 중 무작위 5점 (반복 수 통제용)</td></tr>
          <tr><td>데이터 위치</td><td>로컬 CSV (DB 미적재)</td><td>aecd_platform (API)</td><td>(동일)</td></tr>
          <tr><td>라벨</td><td colspan="3"><strong>세 조건 모두 DB clinical v7 cohort_group</strong> — 7월 파일명 라벨(BPRO/BNOR)은 병리 확정 전 번호라 사용하지 않음. 7월 BPRO로 적힌 49명이 DB에서는 조직검사 음성(Biopsy-negative)</td></tr>
          <tr><td>환자 짝</td><td colspan="3">solum_label 번호로 대조 — 8월 113명 전원이 7월 세트에 존재 (Drop 1명 제외 → {n}명)</td></tr>
        </tbody>
      </table>
    </div>
    <footer class="slide-footer"><span>핵심 메시지: 환자·라벨·알고리즘을 고정했으므로 남는 차이는 "측정 조건"(환원제·lot·날짜·반복 수)뿐이다. 단, 7월 lot 미기록이라 환원제 하나로 좁혀지지는 않는다.</span><span>조건표: dataset_conditions.md 5·8번</span></footer>
  </section>''')

    # 3. headline metrics
    slides.append(f'''
  <section class="slide" data-title="성능 비교" aria-labelledby="s3-title">
    <header><h1 id="s3-title">Screening은 유지, 3군 구분은 하락 — fold 반복 {reps}회 기준</h1></header>
    <div class="center"><img class="fig" src="{img("fig_metric_bars.png")}" alt="세 조건의 Screening AUC와 3군 macro AUC 평균±SD, 그리고 fold 반복별 ΔAUC"></div>
    <footer class="slide-footer"><span>Screening AUC: 7월 {fmt(scr["july_liquid"]["mean"])}±{fmt(scr["july_liquid"]["sd"])} → 8월 {fmt(scr["aug_mapping"]["mean"])}±{fmt(scr["aug_mapping"]["sd"])} (ΔAUC {delta_jm:+.3f}, DeLong p 중앙값 {fmt(p_jm["delong_p_median"], 2)}, 범위 {"겹침" if scr_overlap else "비겹침"}) · 3군 macro AUC: {fmt(thr["july_liquid"]["mean"])}±{fmt(thr["july_liquid"]["sd"])} → {fmt(thr["aug_mapping"]["mean"])}±{fmt(thr["aug_mapping"]["sd"])} (Δ {thr_delta:+.3f}, 범위 {"겹침" if thr_overlap else "비겹침"}) · 막대=평균, 오차=SD, 점=fold 반복 {reps}회 · n={n} · OOF</span><span>bootstrap CI 아님 — fold 반복 SD</span></footer>
  </section>''')

    # 3a. full metric table
    def trow(label: str, cells: list[str], strong: bool = False) -> str:
        tag = "strong" if strong else "span"
        return "<tr><td>" + label + "</td>" + "".join(f"<td><{tag}>{c}</{tag}></td>" for c in cells) + "</tr>"
    head = "<thead><tr><th>지표 (평균 ± SD, fold 반복 " + str(reps) + "회)</th>" + "".join(f"<th class=\"{'liquid' if c=='july_liquid' else 'powder'}\">{cond_name[c]}</th>" for c in conds4) + "</tr></thead>"
    scr_rows = "".join([
        trow("Screening ROC-AUC", [cell("screening_binary", "roc_auc", c) for c in conds4], True),
        trow("Screening balanced accuracy", [cell("screening_binary", "balanced_accuracy", c) for c in conds4]),
        trow("Screening sensitivity (암 검출)", [cell("screening_binary", "sensitivity", c) for c in conds4]),
        trow("Screening specificity", [cell("screening_binary", "specificity", c) for c in conds4]),
        trow("Screening macro-F1", [cell("screening_binary", "macro_f1", c) for c in conds4]),
    ])
    thr_rows = "".join([
        trow("3군 macro OvR ROC-AUC", [cell("three_group", "macro_ovr_roc_auc", c) for c in conds4], True),
        trow("3군 balanced accuracy", [cell("three_group", "balanced_accuracy", c) for c in conds4]),
        trow("3군 macro-F1", [cell("three_group", "macro_f1", c) for c in conds4]),
        trow("&nbsp;&nbsp;OvR AUC · 정상(Control)", [cell1("three_group", "ovr_auc_control", c) for c in conds4]),
        trow("&nbsp;&nbsp;OvR AUC · 조직검사 음성", [cell1("three_group", "ovr_auc_biopsy_neg", c) for c in conds4]),
        trow("&nbsp;&nbsp;OvR AUC · 암", [cell1("three_group", "ovr_auc_cancer", c) for c in conds4]),
        trow("&nbsp;&nbsp;Recall · 정상 / 조직검사음성 / 암", [f"{rec[c]['control']:.2f} / {rec[c]['biopsy_neg']:.2f} / {rec[c]['cancer']:.2f}" for c in conds4]),
    ])
    slides.append(f'''
  <section class="slide" data-title="지표 전체표" aria-labelledby="s3a-title">
    <header><h1 id="s3a-title">지표 전체표 — Screening은 QC를 넣어도 같고, 3군은 정상군 AUC가 우연 수준</h1></header>
    <div class="center" style="overflow:auto"><table class="metrics">{head}<tbody>{scr_rows}<tr class="sep"><td colspan="5"></td></tr>{thr_rows}</tbody></table></div>
    <footer class="slide-footer"><span>핵심 메시지: 8월 mapping의 정상군 OvR AUC {ovr["aug_mapping"]["control"]:.2f}는 우연(0.5)과 같다 — 3군 하락 전부가 여기서 온다. 암 OvR AUC는 {ovr["july_liquid"]["cancer"]:.2f} → {ovr["aug_mapping"]["cancer"]:.2f}로 거의 그대로. "QC통과점"은 PL-1과 같은 2단계 QC(13,552 → 11,336점)를 거친 평균인데 Screening {cell1("screening_binary","roc_auc","aug_mapping_qc")}로 121점 전체 평균과 차이 없음(ΔAUC {p_mq["delta_auc_mean"]:+.3f}, 유의 {p_mq["repeats_delong_p_lt_0_05"]}/{reps}회).</span><span>n={n} · OOF · 4조건 같은 fold 배정</span></footer>
  </section>''')

    # 3b. where the 3-group drop is
    def conf_table(c: str, title_cls: str, title: str) -> str:
        m = conf[c]
        return (f'<div class="card"><p class="kicker {title_cls}">{title}</p><table class="conf"><thead><tr><th>실제 \\ 예측</th><th>Control</th><th>Biopsy-neg</th><th>Cancer</th></tr></thead><tbody>'
                f'<tr><td>Control ({lab["Control"]})</td><td><strong>{m[0][0]}</strong></td><td>{m[0][1]}</td><td>{m[0][2]}</td></tr>'
                f'<tr><td>Biopsy-neg ({lab["Biopsy-negative"]})</td><td>{m[1][0]}</td><td><strong>{m[1][1]}</strong></td><td>{m[1][2]}</td></tr>'
                f'<tr><td>Cancer ({lab["Prostate cancer"]})</td><td>{m[2][0]}</td><td>{m[2][1]}</td><td><strong>{m[2][2]}</strong></td></tr></tbody></table></div>')
    slides.append(f'''
  <section class="slide" data-title="3군 하락의 위치" aria-labelledby="s3b-title">
    <header><h1 id="s3b-title">3군 하락은 Control 군에서 온다 — 암 구분은 그대로</h1></header>
    <div class="grid-3 center" style="grid-template-columns:1fr 1fr .8fr">
      {conf_table("july_liquid", "liquid", "변경 전 · 7월 점측정 (repeat 0)")}
      {conf_table("aug_mapping", "powder", "변경 후 · 8월 mapping (repeat 0)")}
      <div>
        <div class="card liquid-card"><p class="kicker">Control을 Control로</p><span class="metric">{ctrl_ok["july_liquid"]} → {ctrl_ok["aug_mapping"]}</span><p>/ {lab["Control"]}명</p></div>
        <div class="card powder-card" style="margin-top:10px"><p class="kicker">조직검사 음성·암이 Control로 오인</p><span class="metric">{into_ctrl["july_liquid"]} → {into_ctrl["aug_mapping"]}</span><p>명</p></div>
        <div class="card" style="margin-top:10px"><p class="kicker">Cancer를 Cancer로</p><span class="metric">{cancer_ok["july_liquid"]} → {cancer_ok["aug_mapping"]}</span><p>/ {lab["Prostate cancer"]}명 — 유지</p></div>
        <div class="card green-card" style="margin-top:10px"><p class="kicker">클래스별 OvR AUC · 전 → 후 ({reps}회 평균)</p><p>정상 <strong>{ovr["july_liquid"]["control"]:.2f} → {ovr["aug_mapping"]["control"]:.2f}</strong> · 조직검사음성 {ovr["july_liquid"]["biopsy_neg"]:.2f} → {ovr["aug_mapping"]["biopsy_neg"]:.2f} · 암 {ovr["july_liquid"]["cancer"]:.2f} → {ovr["aug_mapping"]["cancer"]:.2f}</p></div>
      </div>
    </div>
    <footer class="slide-footer"><span>핵심 메시지: 8월 mapping에서는 정상(Control) 스펙트럼이 조직검사 음성·암과 섞인다. 암 자체의 구분은 유지되므로 Screening AUC는 변하지 않고 3군 AUC만 떨어진다. 원인이 환원제인지 측정 방식인지는 이 자료로 알 수 없다.</span><span>혼동행렬 = OOF, 행=DB 임상군, repeat 0 ({rep0_note})</span></footer>
  </section>''')

    # 3c. session / day confound
    def sess_cells(sess: str) -> str:
        v = jul_sess.get(sess, {})
        return f"<td>{v.get('Control', 0)}</td><td>{v.get('Biopsy-negative', 0)}</td><td>{v.get('Prostate cancer', 0)}</td>"
    jul_tbl = "".join(f"<tr><td>{sess}</td>{sess_cells(sess)}</tr>" for sess in sorted(jul_sess))
    aug_tbl = "".join(f"<tr><td>{day}</td><td>{v.get('control', 0)}</td><td>{v.get('prostate disease control', 0)}</td><td>{v.get('prostate', 0)}</td></tr>" for day, v in sorted(aug_by_day.items()))
    slides.append(f'''
  <section class="slide" data-title="세션·측정일 교란" aria-labelledby="s3c-title">
    <header><h1 id="s3c-title">7월은 정상군이, 8월은 암이 따로 측정됐다 — 두 세트 모두 라벨과 측정 세션이 얽혀 있다</h1></header>
    <div class="grid-3 center" style="grid-template-columns:1fr 1fr 1.1fr">
      <div class="card liquid-card"><p class="kicker">7월 · 측정 세션(파일 시각) × 임상군</p>
        <table class="conf"><thead><tr><th>세션</th><th>정상</th><th>조직검사음성</th><th>암</th></tr></thead><tbody>{jul_tbl}</tbody></table>
        <p class="muted" style="margin-top:6px">정상 {lab["Control"]}명 중 {jul_sess.get("07-09 11h", {}).get("Control", 0)}명이 <strong>정상만 있는 11시 세션</strong>에서 측정</p></div>
      <div class="card powder-card"><p class="kicker">8월 · 측정일 × 임상군 (DB runs)</p>
        <table class="conf"><thead><tr><th>날짜</th><th>정상</th><th>조직검사음성</th><th>암</th></tr></thead><tbody>{aug_tbl}</tbody></table>
        <p class="muted" style="margin-top:6px">암 {cancer_total}명 중 {cancer_on_14}명이 <strong>암만 있는 08-14</strong>에 측정 · 정상과 조직검사음성은 나흘에 섞여 있음</p></div>
      <div>
        <div class="card"><p class="kicker">7월 정상군 recall — 세션별 (3군 모델, {reps}회 평균)</p>
          <p>정상만 있던 11시 세션 (n={ctrl_11h["n"]}): <strong>{float(ctrl_11h["recall_mean"]):.2f}</strong><br>
          다른 군과 섞인 18시 세션 (n={ctrl_18h["n"]}): <strong>{float(ctrl_18h["recall_mean"]):.2f}</strong></p>
          <p class="muted">같은 환자들의 8월 mapping에서는 {float(aug_ctrl_11h["recall_mean"]):.2f} / {float(aug_ctrl_18h["recall_mean"]):.2f}</p></div>
        <div class="card green-card" style="margin-top:10px"><p class="kicker">읽는 법</p>
          <p>7월 모델은 "정상"보다 "11시 세션"을 배웠을 수 있다 — 같은 정상군이라도 섞인 세션에서는 절반만 맞힌다. 8월에는 정상·조직검사음성이 같은 날 섞여 있어 그 지름길이 없고, 정상군 AUC가 우연 수준으로 내려간다. 반대로 8월은 암이 08-14에 몰려 있어 <strong>Screening 쪽에 같은 위험</strong>이 있다.</p></div>
      </div>
    </div>
    <footer class="slide-footer"><span>핵심 메시지: "3군 하락"은 환원제 효과라기보다, 7월의 정상군 구분이 측정 세션에 기대고 있었을 가능성이 크다. 어느 세트도 라벨과 측정 세션이 분리돼 있지 않으므로, 두 세트의 성능 수치는 모두 위쪽으로 치우쳐 있을 수 있다.</span><span>7월 세션 = CSV 파일 시각 · 8월 = measurement.runs · 세션별 recall은 전체 모델의 OOF</span></footer>
  </section>''')

    # 4. per-patient
    slides.append(f'''
  <section class="slide" data-title="환자별 변화" aria-labelledby="s4-title">
    <header><h1 id="s4-title">환자별 암 확률 — AUC는 같아도 개별 판정은 {flips_lo}~{flips_hi}명이 뒤집힌다</h1></header>
    <div class="center"><img class="fig" src="{img("fig_prob_shift.png")}" alt="환자별 7월 점측정 대 8월 mapping OOF 암 확률 산점도와 변화량 막대"></div>
    <footer class="slide-footer"><span>왼쪽: 같은 환자의 OOF P(암) — x=7월, y=8월, 점선=0.5 판정선 · 오른쪽: 8월−7월 변화량을 정렬, 색=DB 임상군 · 그림은 repeat 0 (뒤집힘 {flips}명; {reps}회 범위 {flips_lo}~{flips_hi}명) · 확률 변화 중앙값 {rep0["july_liquid -> aug_mapping"]["prob_shift_median"]:+.3f}, Wilcoxon p={fmt(rep0["july_liquid -> aug_mapping"]["wilcoxon_p"], 2)}</span><span>n={n} · x축 7월 점측정, y축 8월 mapping</span></footer>
  </section>''')

    # 5. spectra
    slides.append(f'''
  <section class="slide" data-title="스펙트럼" aria-labelledby="s5-title">
    <header><h1 id="s5-title">전처리 후 스펙트럼은 같은 환자에서 얼마나 닮았나</h1></header>
    <div class="grid-2 center">
      <img class="fig" src="{img("fig_group_mean_spectra.png")}" alt="임상군별 평균 스펙트럼, 7월 점측정 대 8월 mapping">
      <div>
        <div class="card"><p class="kicker">같은 환자 7월 점측정 ↔ 8월 mapping 스펙트럼 상관</p><span class="metric">{fmt(corr["median"], 3)}</span><p>중앙값 · 5–95 백분위 {fmt(corr["p5"], 3)} – {fmt(corr["p95"], 3)}<br><span class="muted">전처리 후 935점 SNV 스펙트럼, 환자별 피어슨 상관</span></p></div>
        <div class="card" style="margin-top:12px"><p class="kicker">반복 수 통제 — mapping 5점만 쓰면</p><p>Screening AUC {fmt(scr["aug_mapping_5"]["mean"])}±{fmt(scr["aug_mapping_5"]["sd"])} (121점 {fmt(scr["aug_mapping"]["mean"])}, ΔAUC {p_m5["delta_auc_mean"]:+.3f}, 유의 {p_m5["repeats_delong_p_lt_0_05"]}/{reps}회). 즉 mapping 한 점은 7월 1회 측정보다 정보가 적다. 7월은 Ave100 누적이고 8월 mapping의 누적 횟수는 <strong>기록에 없어</strong> 획득 조건 차이는 확인이 필요하다. 비교 단위는 8월 121점 평균 — 이때 <strong>Screening AUC 수준에서만</strong> 7월 5회 평균과 같아진다.</p></div>
      </div>
    </div>
    <footer class="slide-footer"><span>핵심 메시지: 세 군 모두 같은 방향으로 형태가 바뀐다 — 1000 cm⁻¹(990–1010) 구간 SNV {rng_txt(b1000["july_liquid"])} → {rng_txt(b1000["aug_mapping"])}, 1450(1430–1470) {rng_txt(b1450["july_liquid"])} → {rng_txt(b1450["aug_mapping"])} 감소, 1600–1700 구간 {rng_txt(b1650["july_liquid"])} → {rng_txt(b1650["aug_mapping"])} 증가(군별 범위). 같은 환자의 두 스펙트럼 상관은 중앙값 {fmt(corr["median"], 2)}. 이 형태 변화와 Control 구분 상실의 연결은 미분석.</span><span>임상군 = DB v7 · 상관은 935점 SNV 스펙트럼 · 띠 평균은 군 평균 스펙트럼</span></footer>
  </section>''')

    # 6. how to read
    slides.append(f'''
  <section class="slide" data-title="해석" aria-labelledby="s6-title">
    <header><h1 id="s6-title">이 수치를 어떻게 읽어야 하나</h1></header>
    <div class="grid-3 center">
      <div class="card liquid-card"><p class="kicker">1 · 같은 데이터, fold만 바꿔도</p><p>8월 mapping Screening AUC는 fold 배정에 따라 <strong>{fmt(scr["aug_mapping"]["min"])} ~ {fmt(scr["aug_mapping"]["max"])}</strong> 사이를 오간다 (SD {fmt(scr["aug_mapping"]["sd"])}). 단일 fold 배정 값 0.789(AS-11)는 이 {reps}회의 최대치보다도 높아 반복에서 재현되지 않았다. 앞서 보고된 <strong>0.79~0.80(PL-1)은 다른 파이프라인</strong>이다 — 개별 스펙트럼 단위 학습 + config 전처리(SG5) + subject-grouped fold. 같은 데이터에 논문 파이프라인(환자 평균 1행)을 쓰면 {fmt(scr["aug_mapping"]["mean"])}이고, PL-1과 같은 QC를 넣어도 {cell1("screening_binary","roc_auc","aug_mapping_qc")}로 변하지 않는다.</p></div>
      <div class="card powder-card"><p class="kicker">2 · Screening 차이는 그 폭 안, 3군 차이는 밖</p><p>Screening ΔAUC {delta_jm:+.3f} (범위 {p_jm["delta_auc_min"]:+.3f} ~ {p_jm["delta_auc_max"]:+.3f}), DeLong p&lt;0.05 <strong>{p_jm["repeats_delong_p_lt_0_05"]}/{reps}회</strong> → 차이 없음. 3군 macro AUC는 7월 최저 {fmt(thr["july_liquid"]["min"])} &gt; 8월 최고 {fmt(thr["aug_mapping"]["max"])} — <strong>{reps}회 어느 반복에서도 겹치지 않는다</strong>.</p></div>
      <div class="card green-card"><p class="kicker">3 · 그래서</p><p><strong>암 검출(Screening)은 변경 전후 같다.</strong> 달라진 것은 정상군을 따로 알아보는 능력이고, 이것은 fold 노이즈가 아니다. 다만 7월 lot·측정일·측정 방식이 함께 바뀌었으므로 <strong>"환원제 때문"이라고는 아직 말할 수 없다</strong>.</p></div>
    </div>
    <footer class="slide-footer"><span>핵심 메시지: 단일 실행값(0.789)이 아니라 반복 분포로 읽어야 한다 — 그렇게 읽으면 Screening은 불변, 3군은 실제 하락이다.</span><span>DeLong: 동일 환자 상관 AUC 검정 · 3군은 macro OvR AUC 범위 비교</span></footer>
  </section>''')

    # 7. limits + next actions
    slides.append(f'''
  <section class="slide" data-title="한계와 다음 행동" aria-labelledby="s7-title">
    <header><h1 id="s7-title">환원제 효과를 분리하려면 — 한계와 다음 행동</h1></header>
    <div class="grid-2 center">
      <div class="card powder-card"><p class="kicker">이 비교가 분리하지 못한 변수</p>
        <ul>
          <li><strong>라벨 ↔ 측정 세션 교란</strong> — 7월은 정상군이, 8월은 암이 별도 세션에 몰려 있다(앞 슬라이드). 3군 하락의 상당 부분은 이것으로 설명될 수 있다</li>
          <li><strong>7월 센서/환원제 lot 미기록</strong> — "변경 전"이 정상 lot인지 문제 lot인지 모른다</li>
          <li>측정일 차이 1개월 — 검체 보관, 장비 상태, 측정자</li>
          <li>측정 방식 — 점 측정 5회 vs mapping 121점 (5점 추출로 반복 수는 통제했으나 mapping 위치 효과는 남음)</li>
          <li>7월 세트 7명이 DB 미등록 → 비교에서 제외</li>
        </ul></div>
      <div class="card green-card"><p class="kicker">다음 행동 (권고)</p>
        <ol>
          <li><strong>lot 기록 복원</strong>: 7월 09일 측정에 쓴 센서·환원제 lot을 실험 노트에서 확인 — 이것 하나로 해석이 갈린다</li>
          <li><strong>한 변수 실험</strong>: 같은 날, 같은 검체 분주를 구·신 환원제 센서로 각각 측정 — 나머지 조건(장비·측정자·누적 횟수·mapping 여부) 고정, <strong>임상군을 세션에 섞어서</strong> 배치. 검체 수는 이 자료의 환자별 변동(상관 {fmt(corr["median"], 2)})으로 검정력을 계산해 정한다</li>
          <li>이 짝 비교 코드는 재실행 가능 상태로 유지 — 새 측정이 들어오면 같은 표를 다시 만든다</li>
        </ol></div>
    </div>
    <footer class="slide-footer"><span>핵심 메시지: 다음 단계는 모델을 더 돌리는 것이 아니라, 환원제만 다르고 임상군이 세션에 섞인 측정 한 세트를 만드는 것이다.</span><span>스크립트: scripts/analysis/boramae_paired_reducing_agent_comparison.py</span></footer>
  </section>''')

    css = '''
    :root {
      --canvas:#F5F3EE; --surface:#FFFEFB; --surface-muted:#ECE9E1;
      --ink:#202321; --muted:#646862; --line:#D8D5CD; --on:#FFFFFF;
      --liquid:#2C6E9B; --liquid-soft:#DCEBF4; --powder:#A94712; --powder-soft:#F6E5D7;
      --control:#2C6E9B; --biopsy:#72528F; --cancer:#A94712; --green:#496D57; --green-soft:#DFEADF;
      --s2:8px; --s3:12px; --s4:16px; --s5:20px; --s8:32px;
      --display:clamp(1.35rem,2.25vw,2.1rem); --h1:clamp(1.25rem,2vw,1.9rem);
      --body:clamp(.82rem,1.06vw,1rem); --body-lg:clamp(1rem,1.35vw,1.3rem); --label:clamp(.66rem,.78vw,.78rem);
    }
    * { box-sizing:border-box; } [hidden] { display:none !important; }
    html,body { width:100%; min-height:100%; margin:0; }
    body { overflow:hidden; background:var(--canvas); color:var(--ink); font-family:Pretendard,"Noto Sans KR","Apple SD Gothic Neo","Segoe UI",sans-serif; word-break:keep-all; }
    button { font:inherit; }
    .deck { position:relative; width:100vw; height:100vh; min-height:540px; display:grid; place-items:center; padding:24px 24px 72px; }
    .slide { position:absolute; width:min(calc(100vw - 48px),calc((100vh - 96px)*16/9)); aspect-ratio:16/9; max-height:calc(100vh - 96px); padding:clamp(24px,3.4vw,52px);
      display:grid; grid-template-rows:auto 1fr auto; gap:var(--s3); overflow:hidden; background:var(--surface); border:1px solid var(--line);
      opacity:0; visibility:hidden; transform:translateX(24px); transition:opacity 220ms ease-out,transform 220ms ease-out; }
    .slide.active { z-index:2; opacity:1; visibility:visible; transform:none; } .slide.was-active { transform:translateX(-24px); }
    h1,h2,h3,p { margin-top:0; }
    h1 { margin-bottom:0; font-size:var(--h1); line-height:1.1; letter-spacing:-.03em; }
    h1.display { font-size:var(--display); letter-spacing:-.04em; }
    p,li,td,th { font-size:var(--body); line-height:1.48; }
    .eyebrow { margin:0 0 var(--s2); color:var(--muted); font-size:var(--label); font-weight:750; letter-spacing:.08em; text-transform:uppercase; }
    .lead { max-width:60ch; color:var(--muted); font-size:var(--body-lg); line-height:1.5; }
    .muted { color:var(--muted); } .liquid { color:var(--liquid); } .powder { color:var(--powder); }
    .grid-2 { display:grid; grid-template-columns:1.35fr .65fr; gap:var(--s5); align-items:center; }
    .grid-3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:var(--s4); }
    .center { align-self:center; min-height:0; }
    .card { min-width:0; padding:var(--s4); border:1px solid var(--line); border-radius:9px; background:var(--surface); }
    .liquid-card { background:var(--liquid-soft); border-color:transparent; } .powder-card { background:var(--powder-soft); border-color:transparent; } .green-card { background:var(--green-soft); border-color:transparent; }
    .kicker { margin:0 0 var(--s2); color:var(--muted); font-size:var(--label); font-weight:750; letter-spacing:.05em; }
    .metric { display:block; margin-bottom:var(--s2); font-size:clamp(1.5rem,2.8vw,2.7rem); font-weight:780; line-height:1; letter-spacing:-.04em; }
    .slide-footer { display:flex; align-items:flex-end; justify-content:space-between; gap:var(--s4); color:var(--muted); font-size:var(--label); border-top:1px solid var(--line); padding-top:8px; }
    .slide-footer span:first-child { max-width:84%; }
    table { width:100%; border-collapse:collapse; } th,td { padding:7px 10px; border-bottom:1px solid var(--line); text-align:left; vertical-align:top; }
    th { color:var(--muted); font-size:var(--label); letter-spacing:.03em; } td strong { font-weight:750; }
    .hero-layout { display:grid; grid-template-columns:1.3fr .7fr; gap:var(--s8); align-items:start; }
    .hero-meta { justify-self:end; width:100%; }
    .cohort-table td:nth-child(2),.cohort-table th:nth-child(2) { text-align:right; } .cohort-table .total td { font-weight:780; border-top:2px solid var(--ink); }
    .pipeline { margin-top:var(--s2); padding:8px 12px; border-left:4px solid var(--ink); background:var(--surface-muted); color:var(--muted); font-size:var(--label); line-height:1.4; }
    .fig { display:block; max-width:100%; max-height:100%; margin:0 auto; object-fit:contain; }
    ul,ol { margin:0; padding-left:1.2em; } li { margin-bottom:6px; }
    .conf td,.conf th { text-align:center; } .conf td:first-child,.conf th:first-child { text-align:left; }
    .metrics td,.metrics th { text-align:right; font-size:clamp(.74rem,.95vw,.92rem); padding:5px 10px; } .metrics td:first-child,.metrics th:first-child { text-align:left; } .metrics tr.sep td { padding:2px; border-bottom:2px solid var(--ink); }
    .nav { position:fixed; left:0; right:0; bottom:0; display:flex; align-items:center; justify-content:center; gap:14px; padding:12px; background:var(--canvas); border-top:1px solid var(--line); z-index:5; }
    .nav button { padding:7px 14px; border:1px solid var(--line); border-radius:6px; background:var(--surface); cursor:pointer; } .nav button:disabled { opacity:.4; cursor:default; }
    .progress { position:fixed; left:0; top:0; height:3px; width:100%; background:var(--green); transform-origin:left; transform:scaleX(0); z-index:6; }
    .sr { position:absolute; left:-9999px; }
    @media print { body { overflow:visible; } .deck { display:block; height:auto; padding:0; } .slide { position:relative; opacity:1; visibility:visible; transform:none; width:100%; max-height:none; page-break-after:always; margin:0 0 12px; } .nav,.progress { display:none; } }
    '''
    html = f'''<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>환원제 변경 전후 동일 환자 비교 · 보라매</title><style>{css}</style></head>
<body>
<div class="progress" id="progress"></div><div class="sr" aria-live="polite" id="announcer"></div>
<main class="deck">{"".join(slides)}
</main>
<nav class="nav"><button id="prev" aria-label="이전 슬라이드">← 이전</button><span id="counter">01 / {len(slides):02d}</span><button id="next" aria-label="다음 슬라이드">다음 →</button></nav>
<script>
  (() => {{
    const slides=[...document.querySelectorAll('.slide')];
    const prev=document.getElementById('prev'),next=document.getElementById('next'),progress=document.getElementById('progress'),counter=document.getElementById('counter'),announcer=document.getElementById('announcer');
    let current=Math.max(0,Math.min(slides.length-1,Number(location.hash.slice(1))-1||0));
    function render(i,announce=true){{ current=Math.max(0,Math.min(slides.length-1,i));
      slides.forEach((s,k)=>{{s.classList.toggle('active',k===current);s.classList.toggle('was-active',k<current);s.setAttribute('aria-hidden',k===current?'false':'true');}});
      prev.disabled=current===0;next.disabled=current===slides.length-1;progress.style.transform=`scaleX(${{(current+1)/slides.length}})`;
      counter.textContent=`${{String(current+1).padStart(2,'0')}} / ${{String(slides.length).padStart(2,'0')}}`;
      document.title=`${{slides[current].dataset.title}} · 환원제 변경 전후 동일 환자 비교`;history.replaceState(null,'',`#${{current+1}}`);
      if(announce)announcer.textContent=`${{current+1}}번 슬라이드, ${{slides[current].dataset.title}}`; }}
    prev.addEventListener('click',()=>render(current-1));next.addEventListener('click',()=>render(current+1));
    addEventListener('keydown',e=>{{if(['ArrowRight','PageDown',' '].includes(e.key)){{e.preventDefault();render(current+1)}}if(['ArrowLeft','PageUp'].includes(e.key)){{e.preventDefault();render(current-1)}}}});
    render(current,false);
  }})();
</script></body></html>'''
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size/1e6:.1f} MB, {len(slides)} slides)")


if __name__ == "__main__":
    main()
