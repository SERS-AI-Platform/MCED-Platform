"""
Phase 6 — Preprocessing Pipeline Validation (Ablation + Sensitivity)

목적
----
QC 임계값을 정하기 전에, **현재 전처리 파이프라인 자체가 합당한가**를
정량적으로 검증한다. 각 단계가 실제로 (a) replicate 일관성을 높이고
(b) downstream 분류에 도움이 되는지를 체크한다.

전처리 파이프라인:
    raw → calibration → smooth → baseline → SNV → resample

ablation 시퀀스:
    L0  raw            (smooth=F, baseline=F, snv=F, calibration=F)
    L1  +calibration
    L2  +smooth
    L3  +baseline
    L4  +SNV           (= 현재 production 파이프라인)

각 레벨에서 다음을 측정한다:
    1. Replicate Pearson corr 분포 (mean / median / 5th percentile)
    2. RSD 분포 (%)  — calculate_replicate_qc 재사용
    3. urea peak (1001.4 cm⁻¹) alignment 잔차
    4. NOR vs cancer 단순 LR 5-fold CV AUC (subject medoid spectrum)

추가: smooth window / baseline window 파라미터 sensitivity sweep.

Subset
------
스피드를 위해 NOR 100명 + PRO 100명 subset 사용 (총 ~1000 spectra).
PRO를 고른 이유: 단일 병원·단일 protocol(SMCXD01)이므로 confounder 적음.

Outputs
-------
results/preprocessing_validation/
    preproc_ablation.csv          — 레벨별 metric
    preproc_sensitivity.csv       — 파라미터 sweep
    preproc_validation_report.md  — 권고사항 요약
"""

from __future__ import annotations

import sys
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

# ─── Project imports ───
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.config import load_config, RAW_DATA_DIR, RESULTS_DIR
from src.sers import read_spectrum, parse_filename
from src.sers.io import find_spectra, make_fixed_grid
from src.sers.preprocessing import (
    preprocess_single_spectrum,
    calibrate_spectra_batch,
    trim_to_grid,
    FINGERPRINT_REGION,
)
from src.sers.qc.qc import calculate_replicate_qc

# ─── Logging ───
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("preproc_validation")

OUTPUT_DIR = Path(RESULTS_DIR) / "preprocessing_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Subset config ───
# Easy task: NOR vs PRO (single hospital, single protocol — ceiling effect)
EASY_GROUPS = {"NOR": 100, "PRO": 100}

# Hard task: 7-cancer screening (cancer vs control). Includes BLC (problematic).
# Caps per group keep runtime ≈ minutes. Aliases (CPAN/YPAN→PAN, YNOR→NOR)
# are applied AFTER loading.
HARD_GROUPS = {
    "NOR": 60, "DIA": 60, "HBP": 60, "H.D.": 60,           # controls
    "PRO": 60, "BRE": 30, "OVA": 60, "LUN": 60, "CRC": 60, # cancers
    "CPAN": 60, "YPAN": 30, "BLC": 60,                      # PAN + BLC
}
CANCER_LABELS = {"PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "YPAN", "BLC"}

UREA_REF_WN = 1001.4
UREA_SEARCH_WIN = 30.0  # ± cm⁻¹ around target for residual measurement
LARGE_SHIFT_CM1 = 5.0   # |shift| above this is considered suspicious


# =============================================================================
# Data loading
# =============================================================================
def load_subset(config, group_caps: dict) -> dict:
    """Load raw spectra capped per group (deterministic by sample_id)."""
    data_dir = Path(RAW_DATA_DIR)
    files = list(find_spectra(data_dir, pattern="*.csv"))

    raw = {}
    for fp in files:
        try:
            spec_id = parse_filename(
                fp,
                fallback_group=config.folder_to_group.get(fp.parent.name, "UNK"),
            )
        except Exception:
            continue
        if spec_id.group not in group_caps:
            continue
        x, y = read_spectrum(fp)
        raw[(spec_id.group, spec_id.sample_id, spec_id.replicate)] = (x, y)

    keep = set()
    for grp, n_max in group_caps.items():
        sids = sorted({k[1] for k in raw if k[0] == grp})[:n_max]
        for k in raw:
            if k[0] == grp and k[1] in sids:
                keep.add(k)
    raw = {k: raw[k] for k in keep}

    n_samples = len({(g, s) for g, s, _ in raw})
    logger.info(f"  Subset: {len(raw)} spectra, {n_samples} samples")
    for g in sorted(group_caps):
        n = len({k[1] for k in raw if k[0] == g})
        logger.info(f"    {g:>5s}: {n} samples")
    return raw


# =============================================================================
# Ablation runner
# =============================================================================
def run_pipeline(
    raw_spectra: dict,
    grid: np.ndarray,
    *,
    do_calibration: bool,
    do_smooth: bool,
    do_baseline: bool,
    normalization: str,
    smooth_window: int = 11,
    smooth_poly: int = 3,
    baseline_window: int = 101,
) -> tuple[dict, np.ndarray]:
    """Apply pipeline with the given toggles. Returns processed spectra dict."""
    spectra = raw_spectra
    if do_calibration:
        spectra, _ = calibrate_spectra_batch(
            spectra, target_wn=UREA_REF_WN, window=10.0
        )

    proc_grid = trim_to_grid(grid, region=FINGERPRINT_REGION)
    processed = {}
    for key, (x, y) in spectra.items():
        try:
            y_proc = preprocess_single_spectrum(
                x=x, y=y, grid=proc_grid,
                do_trim=True, trim_region=FINGERPRINT_REGION,
                do_smooth=do_smooth,
                smooth_window=smooth_window,
                smooth_poly=smooth_poly,
                do_baseline=do_baseline,
                baseline_window=baseline_window,
                normalization=normalization,
            )
            processed[key] = y_proc
        except Exception as e:
            logger.warning(f"  skip {key}: {e}")
    return processed, proc_grid


def replicate_metrics(processed: dict, grid: np.ndarray) -> dict:
    """Compute replicate corr / RSD distribution summaries."""
    spectra_for_qc = {k: (grid, y) for k, y in processed.items()}
    qc_df = calculate_replicate_qc(spectra_for_qc, grid)
    if len(qc_df) == 0:
        return {}
    return {
        "n_samples":        int(len(qc_df)),
        "rsd_mean":         float(qc_df["mean_rsd"].mean()),
        "rsd_median":       float(qc_df["mean_rsd"].median()),
        "rsd_p95":          float(qc_df["mean_rsd"].quantile(0.95)),
        "corr_mean":        float(qc_df["mean_corr"].mean()),
        "corr_median":      float(qc_df["mean_corr"].median()),
        "corr_p05":         float(qc_df["mean_corr"].quantile(0.05)),
        "corr_min":         float(qc_df["mean_corr"].min()),
    }


def urea_residual(raw_spectra: dict, do_calibration: bool) -> dict:
    """Measure post-calibration urea peak alignment residual."""
    spectra = raw_spectra
    if do_calibration:
        spectra, _ = calibrate_spectra_batch(
            spectra, target_wn=UREA_REF_WN, window=20.0
        )

    residuals = []
    for key, (x, y) in spectra.items():
        mask = (x >= UREA_REF_WN - UREA_SEARCH_WIN) & (x <= UREA_REF_WN + UREA_SEARCH_WIN)
        if mask.sum() < 5:
            continue
        x_loc, y_loc = x[mask], y[mask]
        peak_x = x_loc[int(np.argmax(y_loc))]
        residuals.append(peak_x - UREA_REF_WN)
    if not residuals:
        return {}
    arr = np.array(residuals)
    return {
        "urea_resid_mean": float(arr.mean()),
        "urea_resid_std":  float(arr.std()),
        "urea_resid_abs_median": float(np.median(np.abs(arr))),
    }


def _label_easy(grp: str) -> int:
    return 1 if grp == "PRO" else 0


def _label_hard(grp: str) -> int:
    return 1 if grp in CANCER_LABELS else 0


def downstream_auc(processed: dict, grid: np.ndarray, label_fn) -> float:
    """Subject-level mean spectrum → 5-fold LR AUC for given labelling."""
    subj = {}
    for (grp, sid, rep), y in processed.items():
        subj.setdefault((grp, sid), []).append(y)
    rows, labels = [], []
    for (grp, sid), reps in subj.items():
        rows.append(np.mean(np.vstack(reps), axis=0))
        labels.append(label_fn(grp))
    X = np.vstack(rows)
    y = np.array(labels)
    if len(np.unique(y)) < 2:
        return float("nan")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(Xte)[:, 1]))
    return float(np.mean(aucs))


# =============================================================================
# Calibration sweep (deep dive on cal window)
# =============================================================================
def run_calibration_sweep(raw_spectra: dict, grid: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare ±10 / ±20 / ±40 cm⁻¹ calibration windows.

    For each window:
        - shift distribution (mean, std, p05, p95, |large_shift| count)
        - replicate corr / RSD after full pipeline
        - downstream AUC (easy task)
    Also returns per-spectrum shifts for the ±20 window so suspicious shifts
    can be inspected.
    """
    rows = []
    detail_per20 = None
    for win in [10.0, 20.0, 40.0]:
        logger.info(f"\n[Calibration sweep] window=±{win} cm⁻¹")
        cal_spectra, shift_df = calibrate_spectra_batch(
            raw_spectra, target_wn=UREA_REF_WN, window=win
        )
        if win == 20.0:
            detail_per20 = shift_df.copy()

        shifts = shift_df["shift_cm1"].dropna().values
        n_large = int(np.sum(np.abs(shifts) > LARGE_SHIFT_CM1))

        # Run full pipeline using these calibrated spectra
        proc_grid = trim_to_grid(grid, region=FINGERPRINT_REGION)
        processed = {}
        for key, (x, y) in cal_spectra.items():
            try:
                processed[key] = preprocess_single_spectrum(
                    x=x, y=y, grid=proc_grid,
                    do_trim=True, trim_region=FINGERPRINT_REGION,
                    do_smooth=True, smooth_window=11, smooth_poly=3,
                    do_baseline=True, baseline_window=101,
                    normalization="snv",
                )
            except Exception:
                continue
        m = replicate_metrics(processed, proc_grid)
        m["window_cm1"] = win
        m["shift_mean"] = float(shifts.mean()) if len(shifts) else float("nan")
        m["shift_std"] = float(shifts.std()) if len(shifts) else float("nan")
        m["shift_abs_p95"] = float(np.percentile(np.abs(shifts), 95)) if len(shifts) else float("nan")
        m["n_large_shift"] = n_large
        m["pct_large_shift"] = 100.0 * n_large / max(len(shifts), 1)
        m["auc_easy"] = downstream_auc(processed, proc_grid, _label_easy)
        rows.append(m)
        logger.info(
            f"   shift μ={m['shift_mean']:+.3f}±{m['shift_std']:.3f}  "
            f"|shift|>{LARGE_SHIFT_CM1}: {n_large}/{len(shifts)} "
            f"({m['pct_large_shift']:.1f}%)  "
            f"corr={m.get('corr_mean',float('nan')):.4f}  "
            f"AUC={m['auc_easy']:.4f}"
        )

    # Also: NO calibration baseline
    logger.info("\n[Calibration sweep] window=NONE (calibration off)")
    proc_grid = trim_to_grid(grid, region=FINGERPRINT_REGION)
    processed = {}
    for key, (x, y) in raw_spectra.items():
        try:
            processed[key] = preprocess_single_spectrum(
                x=x, y=y, grid=proc_grid,
                do_trim=True, trim_region=FINGERPRINT_REGION,
                do_smooth=True, smooth_window=11, smooth_poly=3,
                do_baseline=True, baseline_window=101,
                normalization="snv",
            )
        except Exception:
            continue
    m = replicate_metrics(processed, proc_grid)
    m.update({"window_cm1": 0.0, "shift_mean": 0.0, "shift_std": 0.0,
              "shift_abs_p95": 0.0, "n_large_shift": 0, "pct_large_shift": 0.0})
    m["auc_easy"] = downstream_auc(processed, proc_grid, _label_easy)
    rows.append(m)
    logger.info(
        f"   corr={m.get('corr_mean',float('nan')):.4f}  AUC={m['auc_easy']:.4f}"
    )

    df = pd.DataFrame(rows)
    cols = ["window_cm1", "n_samples", "shift_mean", "shift_std", "shift_abs_p95",
            "n_large_shift", "pct_large_shift",
            "corr_mean", "rsd_mean", "auc_easy"]
    df = df[[c for c in cols if c in df.columns]]
    return df, detail_per20


# =============================================================================
# Main experiments
# =============================================================================
def run_ablation(raw_spectra: dict, grid: np.ndarray, label_fn, task_name: str) -> pd.DataFrame:
    """Sequential ablation: raw → +cal → +smooth → +baseline → +SNV."""
    levels = [
        ("L0_raw",        dict(do_calibration=False, do_smooth=False, do_baseline=False, normalization="none")),
        ("L1_calibrate",  dict(do_calibration=True,  do_smooth=False, do_baseline=False, normalization="none")),
        ("L2_smooth",     dict(do_calibration=True,  do_smooth=True,  do_baseline=False, normalization="none")),
        ("L3_baseline",   dict(do_calibration=True,  do_smooth=True,  do_baseline=True,  normalization="none")),
        ("L4_snv_full",   dict(do_calibration=True,  do_smooth=True,  do_baseline=True,  normalization="snv")),
    ]
    rows = []
    for name, kw in levels:
        logger.info(f"\n[Ablation:{task_name}] {name} ...")
        proc, proc_grid = run_pipeline(raw_spectra, grid, **kw)
        m = replicate_metrics(proc, proc_grid)
        m["urea_resid_abs_median"] = urea_residual(raw_spectra, kw["do_calibration"]).get("urea_resid_abs_median", np.nan)
        m["auc"] = downstream_auc(proc, proc_grid, label_fn)
        m["level"] = name
        m["task"] = task_name
        rows.append(m)
        logger.info(
            f"   corr={m.get('corr_mean', float('nan')):.4f}  "
            f"rsd={m.get('rsd_mean', float('nan')):.2f}%  "
            f"AUC={m['auc']:.4f}"
        )
    df = pd.DataFrame(rows)
    cols = ["task", "level", "n_samples", "corr_mean", "corr_median", "corr_p05",
            "rsd_mean", "rsd_median", "rsd_p95",
            "urea_resid_abs_median", "auc"]
    return df[[c for c in cols if c in df.columns]]


def run_sensitivity(raw_spectra: dict, grid: np.ndarray) -> pd.DataFrame:
    """Parameter sensitivity sweeps (smooth window, baseline window)."""
    rows = []

    # Smooth window sweep (full pipeline otherwise)
    for sw in [5, 11, 21]:
        logger.info(f"\n[Sensitivity] smooth_window={sw}")
        proc, gr = run_pipeline(
            raw_spectra, grid,
            do_calibration=True, do_smooth=True, do_baseline=True,
            normalization="snv", smooth_window=sw, baseline_window=101,
        )
        m = replicate_metrics(proc, gr)
        m["param"], m["value"] = "smooth_window", sw
        m["auc"] = downstream_auc(proc, gr, _label_easy)
        rows.append(m)

    # Baseline window sweep
    for bw in [51, 101, 201]:
        logger.info(f"\n[Sensitivity] baseline_window={bw}")
        proc, gr = run_pipeline(
            raw_spectra, grid,
            do_calibration=True, do_smooth=True, do_baseline=True,
            normalization="snv", smooth_window=11, baseline_window=bw,
        )
        m = replicate_metrics(proc, gr)
        m["param"], m["value"] = "baseline_window", bw
        m["auc"] = downstream_auc(proc, gr, _label_easy)
        rows.append(m)

    df = pd.DataFrame(rows)
    cols = ["param", "value", "n_samples", "corr_mean", "rsd_mean",
            "corr_p05", "rsd_p95", "auc"]
    return df[[c for c in cols if c in df.columns]]


# =============================================================================
# Report
# =============================================================================
def write_report(
    ablation_df: pd.DataFrame,
    sens_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    cal_detail: pd.DataFrame | None,
    out: Path,
) -> None:
    def fmt(v, p=4):
        return f"{v:.{p}f}" if isinstance(v, (int, float)) and not pd.isna(v) else "—"

    lines = []
    lines.append("# 전처리 파이프라인 검증 리포트 (Phase 6)\n")
    lines.append(f"_생성: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")

    # ── Ablation: easy + hard ──
    for task_name in ablation_df["task"].unique():
        sub = ablation_df[ablation_df["task"] == task_name]
        lines.append(f"\n## Ablation — task: **{task_name}**\n")
        lines.append("각 줄은 직전 단계 위에 새 단계를 누적 추가한 결과입니다.\n")
        lines.append("\n| Level | n | corr_mean | rsd_mean(%) | urea_|resid|(cm⁻¹) | AUC |\n")
        lines.append("|---|---|---|---|---|---|\n")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['level']} | {int(r.get('n_samples', 0))} | "
                f"{fmt(r.get('corr_mean'))} | {fmt(r.get('rsd_mean'),2)} | "
                f"{fmt(r.get('urea_resid_abs_median'),3)} | {fmt(r.get('auc'))} |\n"
            )

    lines.append("\n### 해석 가이드\n")
    lines.append("- corr↑ / rsd↓ 가 함께 일어나면 그 단계는 정당화됩니다.\n")
    lines.append("- AUC가 단조 감소하면 over-processing 신호 — 단, easy task는 ceiling으로 신호가 약합니다.\n")
    lines.append("- **hard task의 AUC 변화**가 production 결정에 더 의미 있습니다.\n")

    # ── Calibration sweep ──
    lines.append("\n## Calibration window sweep (easy task)\n")
    lines.append("\n| window(±cm⁻¹) | shift μ | shift σ | |shift| p95 | |shift|>5 (%) | corr_mean | rsd_mean(%) | AUC |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for _, r in cal_df.iterrows():
        win = r["window_cm1"]
        win_label = "OFF" if win == 0 else f"±{win:.0f}"
        lines.append(
            f"| {win_label} | {fmt(r.get('shift_mean'),3)} | {fmt(r.get('shift_std'),3)} | "
            f"{fmt(r.get('shift_abs_p95'),3)} | {fmt(r.get('pct_large_shift'),2)} | "
            f"{fmt(r.get('corr_mean'))} | {fmt(r.get('rsd_mean'),2)} | "
            f"{fmt(r.get('auc_easy'))} |\n"
        )

    if cal_detail is not None and len(cal_detail) > 0:
        large = cal_detail[np.abs(cal_detail["shift_cm1"]) > LARGE_SHIFT_CM1]
        lines.append(f"\n**±20 window 기준 |shift|>{LARGE_SHIFT_CM1} cm⁻¹ 스펙트럼: {len(large)}/{len(cal_detail)}**\n")
        if len(large) > 0:
            by_grp = large.groupby("group").size().sort_values(ascending=False)
            lines.append("\n그룹별 분포:\n\n")
            for grp, n in by_grp.items():
                lines.append(f"- {grp}: {n}\n")

    # ── Parameter sensitivity ──
    lines.append("\n## 파라미터 sensitivity (easy task)\n")
    lines.append("\n| Param | Value | corr_mean | rsd_mean(%) | AUC |\n")
    lines.append("|---|---|---|---|---|\n")
    for _, r in sens_df.iterrows():
        lines.append(
            f"| {r['param']} | {r['value']} | "
            f"{fmt(r.get('corr_mean'))} | {fmt(r.get('rsd_mean'),2)} | "
            f"{fmt(r.get('auc'))} |\n"
        )

    lines.append("\n## 다음 단계\n")
    lines.append("- calibration window 결정 → config 업데이트 필요시 사용자 승인\n")
    lines.append("- 전처리 합리성 확정 후 Phase 7 (threshold derivation) 진행\n")

    out.write_text("".join(lines), encoding="utf-8")


# =============================================================================
# Main
# =============================================================================
def main():
    logger.info("=" * 60)
    logger.info("  Phase 6: Preprocessing Validation")
    logger.info("=" * 60)

    config = load_config(str(PROJECT_ROOT / "config" / "config.yaml"))
    grid = make_fixed_grid(config)
    if grid is None:
        raise RuntimeError("fixed_grid not configured in config.yaml")

    logger.info("\n[1] Loading easy subset (NOR + PRO) ...")
    raw_easy = load_subset(config, EASY_GROUPS)

    logger.info("\n[2] Loading hard subset (7-cancer screening) ...")
    raw_hard = load_subset(config, HARD_GROUPS)

    logger.info("\n[3] Ablation — easy task (NOR vs PRO) ...")
    abl_easy = run_ablation(raw_easy, grid, _label_easy, "easy_NOR_vs_PRO")

    logger.info("\n[4] Ablation — hard task (cancer screening) ...")
    abl_hard = run_ablation(raw_hard, grid, _label_hard, "hard_7cancer_screening")

    ablation_df = pd.concat([abl_easy, abl_hard], ignore_index=True)
    ablation_csv = OUTPUT_DIR / "preproc_ablation.csv"
    ablation_df.to_csv(ablation_csv, index=False)
    logger.info(f"  → {ablation_csv}")

    logger.info("\n[5] Calibration window sweep (on easy subset) ...")
    cal_df, cal_detail = run_calibration_sweep(raw_easy, grid)
    cal_csv = OUTPUT_DIR / "preproc_calibration_sweep.csv"
    cal_df.to_csv(cal_csv, index=False)
    logger.info(f"  → {cal_csv}")
    if cal_detail is not None:
        cal_detail.to_csv(OUTPUT_DIR / "preproc_calibration_shifts_w20.csv", index=False)

    logger.info("\n[6] Parameter sensitivity (easy subset) ...")
    sens_df = run_sensitivity(raw_easy, grid)
    sens_csv = OUTPUT_DIR / "preproc_sensitivity.csv"
    sens_df.to_csv(sens_csv, index=False)
    logger.info(f"  → {sens_csv}")

    logger.info("\n[7] Writing report ...")
    report_path = OUTPUT_DIR / "preproc_validation_report.md"
    write_report(ablation_df, sens_df, cal_df, cal_detail, report_path)
    logger.info(f"  → {report_path}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
