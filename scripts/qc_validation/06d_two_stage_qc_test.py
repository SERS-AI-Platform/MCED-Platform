"""
Phase 6d — Two-Stage QC Test Run

목적
----
src/sers/qc/qc.py 에 새로 추가된 v2 2단 QC 함수가 hard subset에서 어떻게
작동하는지 단계별 drop 카운트와 그룹별 잔존율을 보고한다.

Pipeline tested:
    1. Load raw (hard subset, 12 groups, ~660 subjects)
    2. Stage 1 QC on RAW:
        - intensity_gate
        - cosmic_ray
        - saturation
    3. Preprocess survivors
    4. Stage 2a — per-spectrum corr-to-mean drop
    5. Stage 2b — min_reps_after_qc enforcement
    6. Report group-level retention vs current strict QC

Output
------
results/preprocessing_validation/two_stage_qc_report.md
results/preprocessing_validation/two_stage_qc_drops.csv
results/preprocessing_validation/two_stage_qc_retention.csv
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR, RESULTS_DIR
from src.sers import read_spectrum, parse_filename
from src.sers.io import find_spectra, make_fixed_grid
from src.sers.preprocessing import preprocess_spectra
from src.sers.qc.qc import (
    apply_stage1_qc,
    apply_per_spectrum_corr_qc,
    enforce_min_replicates,
    calculate_replicate_qc,
    identify_qc_failures,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("qc_v2_test")

OUT_DIR = Path(RESULTS_DIR) / "preprocessing_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HARD_GROUPS = {
    "NOR": 60, "DIA": 60, "HBP": 60, "H.D.": 60,
    "PRO": 60, "BRE": 30, "OVA": 60, "LUN": 60, "CRC": 60,
    "CPAN": 60, "YPAN": 30, "BLC": 60,
}

# QC v2 thresholds
PER_SPEC_CORR = 0.90
MIN_REPS = 3


def load_subset(config) -> dict:
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
        if spec_id.group not in HARD_GROUPS:
            continue
        x, y = read_spectrum(fp)
        raw[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)

    keep = set()
    for grp, n_max in HARD_GROUPS.items():
        sids = sorted({k[1] for k in raw if k[0] == grp})[:n_max]
        for k in raw:
            if k[0] == grp and k[1] in sids:
                keep.add(k)
    raw = {k: raw[k] for k in keep}
    n_subj = len({(g, s) for g, s, _ in raw})
    logger.info(f"Subset: {len(raw)} spectra, {n_subj} subjects")
    return raw


def group_retention(initial: dict, final_keys: set) -> pd.DataFrame:
    """Compare initial vs final subject counts by group."""
    init_subj = {}
    for (g, s, _) in initial:
        init_subj.setdefault(g, set()).add(s)
    final_subj = {}
    for (g, s, _) in final_keys:
        final_subj.setdefault(g, set()).add(s)
    rows = []
    for g in sorted(init_subj):
        n0 = len(init_subj[g])
        n1 = len(final_subj.get(g, set()))
        rows.append({
            "group": g, "n_initial": n0, "n_final": n1,
            "n_dropped": n0 - n1,
            "retention_pct": round(100 * n1 / max(n0, 1), 1),
        })
    return pd.DataFrame(rows)


def main():
    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    grid = make_fixed_grid(config)

    logger.info("[1] Load hard subset ...")
    raw = load_subset(config)
    n0_subj = len({(g, s) for g, s, _ in raw})
    n0_spec = len(raw)

    # ── Stage 1 (raw QC) ──
    logger.info("\n[2] Stage 1 — raw QC (intensity gate + cosmic + saturation)")
    s1_pass, s1_drops = apply_stage1_qc(raw)
    logger.info(f"  passed: {len(s1_pass)}/{n0_spec}")
    if len(s1_drops) > 0:
        logger.info("  drops by reason:\n" + s1_drops["reason"].value_counts().to_string())

    raw_s1 = {k: raw[k] for k in s1_pass}

    # ── Preprocess survivors ──
    logger.info("\n[3] Preprocess Stage-1 survivors ...")
    processed_dict, _, proc_grid = preprocess_spectra(
        raw_s1, grid, config, qc_passed_keys=None,
    )
    logger.info(f"  preprocessed: {len(processed_dict)}")

    # ── Stage 2a — per-spectrum corr-to-mean ──
    logger.info(f"\n[4] Stage 2a — per-spectrum corr-to-mean (threshold={PER_SPEC_CORR})")
    s2a_pass, s2a_drops = apply_per_spectrum_corr_qc(
        processed_dict, corr_threshold=PER_SPEC_CORR
    )

    # ── Stage 2b — min replicates per subject ──
    logger.info(f"\n[5] Stage 2b — min_reps_after_qc={MIN_REPS}")
    final_keys, s2b_dropped_subj = enforce_min_replicates(s2a_pass, min_n=MIN_REPS)
    n_final_subj = len({(g, s) for g, s, _ in final_keys})

    # ── Comparison: v1 strict QC for the same subset ──
    logger.info("\n[6] Strict v1 QC (RSD<5, Corr>0.95) for comparison ...")
    proc_for_v1 = {k: (proc_grid, y) for k, y in processed_dict.items()}
    v1_stats = calculate_replicate_qc(proc_for_v1, proc_grid)
    v1_failures = identify_qc_failures(v1_stats, rsd_threshold=5.0, corr_threshold=0.95)
    v1_failed_subj = set(zip(v1_failures["group"], v1_failures["sample_id"])) if len(v1_failures) else set()
    v1_passed_keys = {
        k for k in processed_dict if (k[0], k[1]) not in v1_failed_subj
    }

    # ── Retention tables ──
    ret_v2 = group_retention(raw, final_keys).rename(
        columns={"n_final": "n_final_v2", "n_dropped": "n_dropped_v2",
                 "retention_pct": "retention_v2"}
    )
    ret_v1 = group_retention(raw, v1_passed_keys).rename(
        columns={"n_final": "n_final_v1", "n_dropped": "n_dropped_v1",
                 "retention_pct": "retention_v1"}
    )
    retention = ret_v2.merge(
        ret_v1[["group", "n_final_v1", "n_dropped_v1", "retention_v1"]],
        on="group",
    )
    retention = retention[[
        "group", "n_initial",
        "n_final_v2", "retention_v2",
        "n_final_v1", "retention_v1",
    ]]

    logger.info("\n[Retention] subject-level\n" + retention.to_string(index=False))

    # ── Save outputs ──
    retention.to_csv(OUT_DIR / "two_stage_qc_retention.csv", index=False)
    all_drops = []
    if len(s1_drops) > 0:
        s1_drops["stage"] = "stage1"
        all_drops.append(s1_drops)
    if len(s2a_drops) > 0:
        s2a_drops["stage"] = "stage2a"
        all_drops.append(s2a_drops)
    if len(s2b_dropped_subj) > 0:
        s2b_dropped_subj["stage"] = "stage2b"
        all_drops.append(s2b_dropped_subj)
    drops_df = pd.concat(all_drops, ignore_index=True) if all_drops else pd.DataFrame()
    drops_df.to_csv(OUT_DIR / "two_stage_qc_drops.csv", index=False)

    # ── Report ──
    lines = []
    lines.append("# Two-Stage QC v2 — 검증 결과 (Phase 6d)\n\n")
    lines.append("## 파이프라인\n\n")
    lines.append("```\n")
    lines.append("Stage 1 (raw)\n")
    lines.append("  - intensity_gate (ratio=0.1)\n")
    lines.append("  - cosmic_ray detection (z>10)\n")
    lines.append("  - saturation detection (plateau ≥5)\n")
    lines.append("→ Preprocess (calibration → smooth → baseline → SNV)\n")
    lines.append("Stage 2a (post-preproc): per-replicate corr-to-mean ≥ 0.90 → drop bad reps\n")
    lines.append(f"Stage 2b: min_reps_after_qc ≥ {MIN_REPS} → drop subjects below threshold\n")
    lines.append("```\n\n")
    lines.append(f"## 입력: {n0_spec} spectra, {n0_subj} subjects\n\n")
    lines.append("## 단계별 drop 카운트\n\n")
    lines.append(f"- Stage 1 drops: **{len(s1_drops)}** spectra\n")
    if len(s1_drops):
        lines.append("\n  reason 분포:\n\n")
        for reason, n in s1_drops["reason"].value_counts().items():
            lines.append(f"  - {reason}: {n}\n")
    lines.append(f"\n- Stage 2a drops: **{len(s2a_drops)}** replicates\n")
    lines.append(f"- Stage 2b dropped subjects: **{len(s2b_dropped_subj)}**\n\n")
    lines.append(f"## 최종\n\n- spectra: {len(final_keys)}/{n0_spec}\n- subjects: {n_final_subj}/{n0_subj}\n\n")

    lines.append("## v2 (per-spectrum) vs v1 (strict whole-subject) — 그룹별 잔존율\n\n")
    lines.append("| group | n_initial | v2 keep | v2 % | v1 keep | v1 % |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for _, r in retention.iterrows():
        lines.append(
            f"| {r['group']} | {int(r['n_initial'])} | "
            f"{int(r['n_final_v2'])} | {r['retention_v2']:.1f} | "
            f"{int(r['n_final_v1'])} | {r['retention_v1']:.1f} |\n"
        )

    lines.append("\n## 해석 가이드\n")
    lines.append("- v2가 v1보다 잔존율이 일관되게 높으면 per-spectrum 정책의 가치 입증.\n")
    lines.append("- 특히 BLC, OVA 처럼 v1에서 catastrophic drop이 났던 그룹의 회복 여부 확인.\n")
    lines.append("- v2 잔존율이 100%에 가깝다면 임계값(0.90)이 너무 느슨한 것 — 진짜 outlier도 통과 중.\n")
    lines.append("- Stage 1 cosmic/saturation drops가 0이라면 두 detector가 우리 데이터에 안 맞거나 데이터에 그런 결함이 거의 없는 것 — 별도 검증 필요.\n")

    (OUT_DIR / "two_stage_qc_report.md").write_text("".join(lines), encoding="utf-8")
    logger.info(f"\n  → {OUT_DIR / 'two_stage_qc_report.md'}")


if __name__ == "__main__":
    main()
