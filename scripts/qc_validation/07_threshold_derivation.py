"""
Phase 7 — QC v2 임계값 도출

목적
----
2단 QC v2 의 두 가지 임계값을 데이터 기반으로 결정한다:
    - per_spectrum_corr_threshold (Stage 2a)
    - min_reps_after_qc           (Stage 2b)

원리
----
**건강한 NOR 환자의 자연스러운 replicate 변동성**을 reference로 사용한다.
- NOR(양산부산대) + YNOR(세브란스) 환자 전체를 로드
- 현재 production preprocessing 적용 (calibration ±10 + smooth + baseline + SNV)
- 각 NOR replicate에 대해 corr-to-mean-of-others 계산
- 분포의 5th percentile을 per-spec corr 임계값으로 채택
   → "정상 NOR replicate의 95%가 통과, 5%가 outlier로 분류"
- min_reps_after_qc는 NOR 환자별 잔존 replicate 수의 5th percentile로 결정

이 접근의 장점:
- 임계값이 학습 cohort의 cancer 정보를 보지 않음 (data leakage 없음)
- 임상/규제 설명: "건강한 사람도 가끔 outlier replicate가 나오므로 임계값을
  설정한다" — 외부에 인용 가능
- 필요시 다른 instrument로 NOR 데이터를 추가하면 같은 방식으로 갱신 가능

산출
----
results/qc_validation/threshold_derivation.json
results/qc_validation/threshold_derivation_report.md
results/qc_validation/threshold_corr_distribution.png
"""

from __future__ import annotations

import json
import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR, RESULTS_DIR
from src.sers import read_spectrum, parse_filename
from src.sers.io import find_spectra, make_fixed_grid
from src.sers.preprocessing import preprocess_spectra
from src.sers.qc.qc import apply_stage1_qc

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("threshold_derive")

OUT_DIR = Path(RESULTS_DIR) / "qc_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NOR_GROUPS = ["NOR", "YNOR"]


def load_nor(config) -> dict:
    data_dir = Path(RAW_DATA_DIR)
    raw = {}
    for fp in find_spectra(data_dir, pattern="*.csv"):
        try:
            spec_id = parse_filename(
                fp,
                fallback_group=config.folder_to_group.get(fp.parent.name, "UNK"),
            )
        except Exception:
            continue
        if spec_id.group not in NOR_GROUPS:
            continue
        x, y = read_spectrum(fp)
        raw[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)
    n_subj = len({(g, s) for g, s, _ in raw})
    logger.info(f"NOR subset: {len(raw)} spectra, {n_subj} subjects")
    return raw


def compute_corr_to_mean(processed: dict) -> pd.DataFrame:
    """For every replicate, corr to mean of the OTHER replicates in same subject."""
    groups: dict[tuple[str, str], list[tuple[str, np.ndarray]]] = {}
    for (g, s, r), y in processed.items():
        groups.setdefault((g, s), []).append((r, y))

    rows = []
    for (g, s), reps in groups.items():
        if len(reps) < 2:
            continue
        names = [r for r, _ in reps]
        mat = np.vstack([y for _, y in reps])
        for i, r in enumerate(names):
            mask = np.arange(len(reps)) != i
            mean_other = mat[mask].mean(axis=0)
            with np.errstate(invalid="ignore"):
                c = float(np.corrcoef(mat[i], mean_other)[0, 1])
            rows.append({
                "group": g, "sample_id": s, "replicate": r,
                "corr_to_mean": c if np.isfinite(c) else None,
                "n_reps_in_subject": len(reps),
            })
    return pd.DataFrame(rows)


def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    grid = make_fixed_grid(config)

    logger.info("[1] Load NOR + YNOR ...")
    raw = load_nor(config)
    n_subj_initial = len({(g, s) for g, s, _ in raw})

    logger.info("[2] Stage 1 QC on raw ...")
    s1_pass, _ = apply_stage1_qc(raw)
    raw_s1 = {k: raw[k] for k in s1_pass}

    logger.info("[3] Preprocess (calibration ±10 + smooth + baseline + SNV) ...")
    processed_dict, _, proc_grid = preprocess_spectra(raw_s1, grid, config, qc_passed_keys=None)
    logger.info(f"  preprocessed: {len(processed_dict)}")

    logger.info("[4] Compute per-replicate corr-to-mean ...")
    corr_df = compute_corr_to_mean(processed_dict)
    logger.info(f"  measured {len(corr_df)} replicate values")

    # ── Distribution percentiles ──
    corr_vals = corr_df["corr_to_mean"].dropna().values
    pcts = {f"p{p}": float(np.percentile(corr_vals, p)) for p in [1, 2.5, 5, 10, 25, 50, 75]}
    pcts["mean"] = float(np.mean(corr_vals))
    pcts["std"] = float(np.std(corr_vals))
    pcts["min"] = float(np.min(corr_vals))
    pcts["n"] = int(len(corr_vals))
    logger.info(f"  Corr-to-mean distribution (NOR replicates):")
    for k, v in pcts.items():
        logger.info(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # ── Per-subject n_reps_remaining at candidate corr thresholds ──
    logger.info("\n[5] Survival of NOR subjects under candidate corr thresholds ...")
    surv_rows = []
    for thr in [0.80, 0.85, 0.90, 0.92, 0.95, pcts["p5"], pcts["p10"]]:
        kept_per_subj = (
            corr_df[corr_df["corr_to_mean"] >= thr]
            .groupby(["group", "sample_id"]).size()
        )
        # Add subjects that lost ALL reps (zero count)
        all_subj = corr_df.groupby(["group", "sample_id"]).size()
        kept_per_subj = kept_per_subj.reindex(all_subj.index, fill_value=0)
        for min_n in [2, 3, 4, 5]:
            n_pass_subj = int((kept_per_subj >= min_n).sum())
            surv_rows.append({
                "corr_threshold": round(float(thr), 4),
                "min_reps": min_n,
                "n_subjects_kept": n_pass_subj,
                "n_subjects_total": int(len(all_subj)),
                "retention_pct": round(100 * n_pass_subj / max(len(all_subj), 1), 1),
            })
    surv_df = pd.DataFrame(surv_rows)
    logger.info("\n" + surv_df.to_string(index=False))

    # ── Recommended thresholds ──
    rec_corr = round(pcts["p5"], 3)
    # Recommend min_reps as the largest min_n s.t. NOR retention >= 95% at rec_corr
    sub = surv_df[surv_df["corr_threshold"] == round(float(pcts["p5"]), 4)]
    rec_min_reps = 3
    for _, row in sub.sort_values("min_reps", ascending=False).iterrows():
        if row["retention_pct"] >= 95.0:
            rec_min_reps = int(row["min_reps"])
            break

    logger.info(f"\n[6] Recommended thresholds:")
    logger.info(f"    per_spectrum_corr_threshold = {rec_corr}  (NOR p5)")
    logger.info(f"    min_reps_after_qc           = {rec_min_reps}  (largest with NOR retention ≥95%)")

    # ── Save artifacts ──
    out = {
        "method": "NOR + YNOR replicate corr-to-mean distribution; threshold = p5",
        "n_subjects_initial": n_subj_initial,
        "n_replicates_measured": int(len(corr_vals)),
        "corr_distribution": pcts,
        "recommended": {
            "per_spectrum_corr_threshold": rec_corr,
            "min_reps_after_qc": rec_min_reps,
        },
    }
    (OUT_DIR / "threshold_derivation.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    corr_df.to_csv(OUT_DIR / "threshold_corr_per_replicate.csv", index=False)
    surv_df.to_csv(OUT_DIR / "threshold_survival_grid.csv", index=False)

    # ── Histogram ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(corr_vals, bins=60, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(pcts["p5"], color="red", ls="--", label=f"p5 = {pcts['p5']:.3f} (recommended)")
    ax.axvline(pcts["p1"], color="orange", ls=":", label=f"p1 = {pcts['p1']:.3f}")
    ax.axvline(0.95, color="gray", ls=":", label="v1 strict = 0.95")
    ax.set_xlabel("corr-to-mean (replicate vs other replicates)")
    ax.set_ylabel("count")
    ax.set_title(
        f"NOR replicate corr-to-mean distribution\n"
        f"n={len(corr_vals)} replicates from {n_subj_initial} subjects (NOR + YNOR)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "threshold_corr_distribution.png", dpi=130)

    # ── Markdown report ──
    lines = []
    lines.append("# QC v2 임계값 도출 (Phase 7)\n\n")
    lines.append("## 방법\n")
    lines.append("NOR + YNOR 환자의 replicate corr-to-mean 분포에서 5th percentile을 채택. ")
    lines.append("\"건강한 사람의 95%가 통과하는 값\" 이라는 의미.\n\n")
    lines.append(f"- 입력: {n_subj_initial} subjects, {len(corr_vals)} replicate corr values\n")
    lines.append("- 전처리: production pipeline (calibration ±10 + smooth 11 + baseline 101 + SNV)\n\n")
    lines.append("## Corr-to-mean 분포 (NOR + YNOR)\n\n")
    lines.append("| percentile | value |\n|---|---|\n")
    for k in ["min", "p1", "p2.5", "p5", "p10", "p25", "p50", "p75", "mean"]:
        if k in pcts:
            lines.append(f"| {k} | {pcts[k]:.4f} |\n")
    lines.append("\n## 후보 임계값별 NOR subject 잔존율\n\n")
    lines.append("| corr threshold | min_reps | retention % |\n|---|---|---|\n")
    for _, r in surv_df.iterrows():
        lines.append(f"| {r['corr_threshold']:.4f} | {int(r['min_reps'])} | {r['retention_pct']:.1f} |\n")
    lines.append("\n## 권장값\n\n")
    lines.append(f"- **per_spectrum_corr_threshold = {rec_corr}** (NOR p5)\n")
    lines.append(f"- **min_reps_after_qc = {rec_min_reps}** (해당 corr 임계값에서 NOR 잔존율 ≥95% 보장하는 최대 min_n)\n\n")
    lines.append("## 주의\n")
    lines.append("- v1 strict (0.95)와 비교: 우리 NOR replicate 분포의 50p가 그보다 낮을 수 있음 → strict가 NOR도 떨어뜨림 = 자기모순\n")
    lines.append("- 권장값은 NOR 한 instrument 분포 기반. 다른 instrument에서는 분포가 달라질 수 있어 production 적용 전 cross-instrument 비교 권장.\n")
    (OUT_DIR / "threshold_derivation_report.md").write_text("".join(lines), encoding="utf-8")

    logger.info(f"\n  → {OUT_DIR / 'threshold_derivation_report.md'}")


if __name__ == "__main__":
    main()
