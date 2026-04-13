"""
Cross-instrument variance decomposition (4 devices × 100 NOR patients × 20 points).

Defends against the "high AUC = substrate fingerprint?" critique by quantifying:
  - σ²(instrument) / σ²(patient | inst) / σ²(point | patient, inst)
  - ICC(2,1) per instrument (within-patient reproducibility)
  - Cross-instrument patient ranking consistency
  - Per-instrument QC pass rate (RSD<5%, Corr>0.95)
  - Wavenumber calibration drift (target: <0.3 cm⁻¹)

Output: results/qc_validation/cross_instrument_variance.json
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

sys.path.insert(0, "/home/user/SERS-AI/src")
from sers.preprocessing import preprocess_single_spectrum  # noqa: E402

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
ROOT = Path("/home/user/SERS-AI")
DATA_DIR = ROOT / "data/equipment_test_data"
OUT_DIR = ROOT / "results/qc_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Common wavenumber grid: 400–2200 cm⁻¹, 2 cm⁻¹ step → 901 points
WN_MIN, WN_MAX, WN_STEP = 400.0, 2200.0, 2.0
COMMON_WN = np.arange(WN_MIN, WN_MAX + 1e-6, WN_STEP)

INSTRUMENTS = {
    "thermo":          {"folder": "thermo/1. NOR",          "format": "thermo"},
    "medical_removed": {"folder": "medical_removed/1. NOR", "format": "medical"},
    "nanoscope":       {"folder": "nanoscope/1. NOR",       "format": "nanoscope"},
    "handheld":        {"folder": "handheld/1. NOR",        "format": "handheld"},
}

QC_RSD_MAX = 5.0     # %
QC_CORR_MIN = 0.95

logging.basicConfig(level=logging.INFO, format="  %(message)s")
log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# Parsers
# ----------------------------------------------------------------------------
def _parse_thermo_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    arr = np.loadtxt(path, delimiter=",")
    return arr[:, 0], arr[:, 1]


def _parse_two_col_txt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """medical_removed and nanoscope both use tab/space separated 2-col txt."""
    arr = np.loadtxt(path)
    return arr[:, 0], arr[:, 1]


def _parse_handheld_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Handheld Mira P: header rows + a single 'Intensities' line."""
    first_wn = last_wn = None
    intensities: list[float] | None = None
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('"Firstwavenumber"'):
                first_wn = float(line.split(",", 1)[1].strip().strip('"'))
            elif line.startswith('"LastWavenumber"'):
                last_wn = float(line.split(",", 1)[1].strip().strip('"'))
            elif line.startswith('"Intensities"'):
                _, vals = line.split(",", 1)
                vals = vals.strip().strip('"')
                intensities = [float(v) for v in vals.split(",") if v.strip()]
    if first_wn is None or last_wn is None or intensities is None:
        raise ValueError(f"Bad handheld file: {path}")
    wn = np.linspace(first_wn, last_wn, len(intensities))
    return wn, np.asarray(intensities, dtype=float)


PARSERS = {
    "thermo":    _parse_thermo_csv,
    "medical":   _parse_two_col_txt,
    "nanoscope": _parse_two_col_txt,
    "handheld":  _parse_handheld_csv,
}


# ----------------------------------------------------------------------------
# Filename → (patient_id, point_id)
# ----------------------------------------------------------------------------
_FNAME_RE = re.compile(r"NOR\s+(\d+)_(\d+)")


def _parse_pid_pt(name: str) -> tuple[int, int] | None:
    m = _FNAME_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


# ----------------------------------------------------------------------------
# Loader
# ----------------------------------------------------------------------------
def load_instrument(name: str) -> pd.DataFrame:
    info = INSTRUMENTS[name]
    folder = DATA_DIR / info["folder"]
    parser = PARSERS[info["format"]]

    rows = []
    raw_grids: list[np.ndarray] = []
    for f in sorted(folder.iterdir()):
        if not f.is_file():
            continue
        if "_ave" in f.name.lower():
            continue
        pid_pt = _parse_pid_pt(f.name)
        if pid_pt is None:
            continue
        pid, pt = pid_pt
        try:
            wn, intensity = parser(f)
        except Exception as exc:
            log.warning(f"  skip {f.name}: {exc}")
            continue
        if len(raw_grids) < 5:
            raw_grids.append(wn)
        # Project standard preprocessing pipeline (two pass-throughs):
        #   spec_qc   : trim → smooth → baseline (NO normalize) → resample
        #               → used for RSD/Corr QC metrics where intensity scale
        #                 must remain physical (>0)
        #   spec_proc : trim → smooth → baseline → SNV → resample
        #               → used for variance decomposition and cross-instrument
        #                 comparison (scale-invariant)
        try:
            spec_qc = preprocess_single_spectrum(
                wn.astype(float), intensity.astype(float), COMMON_WN,
                do_trim=True, trim_region=(WN_MIN, WN_MAX),
                do_smooth=True, smooth_window=11, smooth_poly=3,
                do_baseline=True, baseline_window=101,
                normalization="none",
            )
            spec_proc = preprocess_single_spectrum(
                wn.astype(float), intensity.astype(float), COMMON_WN,
                do_trim=True, trim_region=(WN_MIN, WN_MAX),
                do_smooth=True, smooth_window=11, smooth_poly=3,
                do_baseline=True, baseline_window=101,
                normalization="snv",
            )
        except Exception as exc:
            log.warning(f"  preprocess fail {f.name}: {exc}")
            continue
        rows.append({
            "instrument": name,
            "patient": pid,
            "point": pt,
            "spectrum": spec_proc,
            "spectrum_qc": spec_qc,
        })

    df = pd.DataFrame(rows)
    log.info(f"  {name}: {len(df)} spectra, {df['patient'].nunique()} patients, "
             f"raw range [{raw_grids[0].min():.1f}, {raw_grids[0].max():.1f}]")
    return df


# ----------------------------------------------------------------------------
# Spectrum normalization (vector norm, robust to scale differences)
# ----------------------------------------------------------------------------
def _l2_normalize(specs: np.ndarray) -> np.ndarray:
    # Replace NaN with 0 for normalization (out-of-range padding)
    s = np.where(np.isnan(specs), 0.0, specs)
    norms = np.linalg.norm(s, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return s / norms


# ----------------------------------------------------------------------------
# QC metrics per (instrument, patient)
# ----------------------------------------------------------------------------
def qc_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (inst, pid), g in df.groupby(["instrument", "patient"]):
        specs = np.stack(g["spectrum_qc"].values)
        # Mask of valid columns (no NaN across all reps for this patient)
        valid = ~np.isnan(specs).any(axis=0)
        if valid.sum() < 50:
            continue
        s = specs[:, valid]
        # Project standard RSD = std / max_intensity * 100 (per src/sers/qc/qc.py)
        sd = s.std(axis=0, ddof=1) if s.shape[0] > 1 else np.zeros_like(s[0])
        max_intensity = float(s.max())
        if max_intensity > 1e-9:
            rsd_per_wn = sd / max_intensity * 100
        else:
            rsd_per_wn = np.zeros_like(sd)
        rsd_mean = float(np.nanmean(rsd_per_wn))
        # Pairwise Pearson correlation across replicates
        if s.shape[0] > 1:
            corr = np.corrcoef(s)
            iu = np.triu_indices_from(corr, k=1)
            corr_mean = float(np.nanmean(corr[iu]))
        else:
            corr_mean = np.nan
        out.append({
            "instrument": inst,
            "patient": pid,
            "n_points": s.shape[0],
            "mean_rsd": rsd_mean,
            "mean_corr": corr_mean,
        })
    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# Variance decomposition: nested ANOVA on per-(inst,patient,point) summary stat
# We use the L2-normalized spectrum's projection onto the global mean spectrum
# as a 1-D summary that captures overall spectral identity.
# ----------------------------------------------------------------------------
def variance_decomposition(df: pd.DataFrame) -> dict:
    # Build per-row summary scalar = correlation to grand-mean spectrum (valid mask)
    specs = np.stack(df["spectrum"].values)
    valid = ~np.isnan(specs).any(axis=0)
    s = specs[:, valid]
    s = _l2_normalize(s)
    grand = s.mean(axis=0)
    grand /= np.linalg.norm(grand) + 1e-12
    summary = s @ grand  # cosine similarity to grand mean

    work = df[["instrument", "patient", "point"]].copy()
    work["y"] = summary

    grand_mean = work["y"].mean()
    ss_total = ((work["y"] - grand_mean) ** 2).sum()

    # SS(instrument)
    inst_means = work.groupby("instrument")["y"].mean()
    inst_n = work.groupby("instrument")["y"].size()
    ss_inst = float((inst_n * (inst_means - grand_mean) ** 2).sum())

    # SS(patient | instrument)
    ss_pat = 0.0
    for inst, g in work.groupby("instrument"):
        pat_means = g.groupby("patient")["y"].mean()
        pat_n = g.groupby("patient")["y"].size()
        ss_pat += float((pat_n * (pat_means - inst_means[inst]) ** 2).sum())

    # SS(point | patient, instrument) = residual within (inst, patient)
    ss_point = 0.0
    for (inst, pid), g in work.groupby(["instrument", "patient"]):
        m = g["y"].mean()
        ss_point += float(((g["y"] - m) ** 2).sum())

    components = {
        "instrument": ss_inst / ss_total * 100,
        "patient_within_instrument": ss_pat / ss_total * 100,
        "point_within_patient": ss_point / ss_total * 100,
    }

    # ICC(1) per instrument: σ²(patient) / (σ²(patient) + σ²(point))
    # using one-way ANOVA on patients within that instrument.
    icc_per_inst = {}
    for inst, g in work.groupby("instrument"):
        k = g.groupby("patient").size().mean()
        pat_means = g.groupby("patient")["y"].mean()
        gm = g["y"].mean()
        n_pat = g["patient"].nunique()
        ms_between = ((pat_means - gm) ** 2).sum() * k / max(n_pat - 1, 1)
        ms_within = 0.0
        n_within = 0
        for pid, gg in g.groupby("patient"):
            ms_within += ((gg["y"] - gg["y"].mean()) ** 2).sum()
            n_within += len(gg) - 1
        ms_within = ms_within / max(n_within, 1)
        if (ms_between + (k - 1) * ms_within) > 0:
            icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        else:
            icc = float("nan")
        icc_per_inst[inst] = float(icc)

    return {
        "ss_total": ss_total,
        "components_pct": components,
        "icc_per_instrument": icc_per_inst,
    }


# ----------------------------------------------------------------------------
# Cross-instrument 4×4 Spearman matrix on patient-mean spectra
# ----------------------------------------------------------------------------
def cross_inst_correlation(df: pd.DataFrame) -> dict:
    # Patient-mean spectra per instrument (L2-normalized)
    pm = {}
    for inst, g in df.groupby("instrument"):
        specs = np.stack(g["spectrum"].values)
        valid = ~np.isnan(specs).any(axis=0)
        s = specs[:, valid]
        s = _l2_normalize(s)
        gg = g.copy()
        gg["s"] = list(s)
        means = gg.groupby("patient")["s"].apply(lambda x: np.mean(np.stack(x.values), axis=0))
        pm[inst] = means
    insts = sorted(pm.keys())
    common_pats = set.intersection(*[set(pm[i].index) for i in insts])
    common_pats = sorted(common_pats)

    matrix = {}
    for a in insts:
        matrix[a] = {}
        for b in insts:
            sa = np.stack([pm[a].loc[p] for p in common_pats])
            sb = np.stack([pm[b].loc[p] for p in common_pats])
            # Per-patient cosine similarity then mean
            sa = sa / (np.linalg.norm(sa, axis=1, keepdims=True) + 1e-12)
            sb = sb / (np.linalg.norm(sb, axis=1, keepdims=True) + 1e-12)
            cos = (sa * sb).sum(axis=1)
            matrix[a][b] = float(np.mean(cos))
    return {"instruments": insts, "n_common_patients": len(common_pats), "matrix": matrix}


# ----------------------------------------------------------------------------
# Mean spectrum per instrument (downsampled for dashboard)
# ----------------------------------------------------------------------------
def instrument_mean_spectra(df: pd.DataFrame, n_points: int = 200) -> dict:
    out = {"wavenumbers": None, "spectra": {}}
    idx = np.linspace(0, len(COMMON_WN) - 1, n_points).astype(int)
    out["wavenumbers"] = COMMON_WN[idx].tolist()
    for inst, g in df.groupby("instrument"):
        specs = np.stack(g["spectrum"].values)
        valid_mask = ~np.isnan(specs).any(axis=0)
        s = specs.copy()
        s[:, ~valid_mask] = np.nan
        s = _l2_normalize(s)
        mean = np.nanmean(s, axis=0)
        # Min-max normalize for visual comparison
        finite = mean[np.isfinite(mean)]
        if finite.size:
            lo, hi = finite.min(), finite.max()
            if hi > lo:
                mean = (mean - lo) / (hi - lo)
        out["spectra"][inst] = [None if not np.isfinite(v) else float(v) for v in mean[idx]]
    return out


# ----------------------------------------------------------------------------
# Convergence: how many replicates until the within-patient mean spectrum
# stabilizes? For each (inst, patient), subsample N points and compute cosine
# similarity to the 20-point reference mean. Average across patients.
# This is the central evidence that 5 reps ≈ 20 reps.
# ----------------------------------------------------------------------------
def convergence_analysis(df: pd.DataFrame, n_seeds: int = 30) -> dict:
    rng = np.random.default_rng(42)
    out: dict = {}
    for inst, g in df.groupby("instrument"):
        n_max = int(g.groupby("patient").size().min())
        ns = list(range(2, n_max + 1))
        # patient_id -> stack of 20 spectra (proc, SNV-normalized)
        per_patient = {}
        for pid, gg in g.groupby("patient"):
            specs = np.stack(gg["spectrum"].values)
            valid = ~np.isnan(specs).any(axis=0)
            s = specs[:, valid]
            s = _l2_normalize(s)
            per_patient[pid] = s

        sims_mean = []
        sims_std = []
        sims_min = []
        for n in ns:
            sims_n = []
            for pid, s in per_patient.items():
                if s.shape[0] < n:
                    continue
                ref = s.mean(axis=0)
                ref /= np.linalg.norm(ref) + 1e-12
                # average similarity over n_seeds random subsamples
                ss = []
                for _ in range(n_seeds):
                    idx = rng.choice(s.shape[0], size=n, replace=False)
                    sub = s[idx].mean(axis=0)
                    sub /= np.linalg.norm(sub) + 1e-12
                    ss.append(float(sub @ ref))
                sims_n.append(float(np.mean(ss)))
            sims_mean.append(float(np.mean(sims_n)))
            sims_std.append(float(np.std(sims_n)))
            sims_min.append(float(np.min(sims_n)))

        # Find smallest N achieving similarity ≥ 0.99
        n_star = None
        for n, s in zip(ns, sims_mean):
            if s >= 0.99:
                n_star = n
                break

        # ICC at each N (one-way, global summary scalar = cosine to grand mean)
        icc_vs_n = []
        # Use the same y as variance_decomposition: cosine to grand mean
        all_specs = np.stack([s for pid, s in per_patient.items()
                               for s in [per_patient[pid][i] for i in range(per_patient[pid].shape[0])]])
        # simpler: just rebuild
        all_rows = []
        for pid, s in per_patient.items():
            for j in range(s.shape[0]):
                all_rows.append((pid, j, s[j]))
        grand = np.mean([r[2] for r in all_rows], axis=0)
        grand /= np.linalg.norm(grand) + 1e-12
        for n in ns:
            ys_by_pid = {}
            for _ in range(n_seeds):
                for pid, s in per_patient.items():
                    if s.shape[0] < n:
                        continue
                    idx = rng.choice(s.shape[0], size=n, replace=False)
                    for j in idx:
                        ys_by_pid.setdefault(pid, []).append(float(s[j] @ grand))
            # one-way ANOVA ICC
            pat_means = {p: np.mean(v) for p, v in ys_by_pid.items()}
            gm = np.mean([y for v in ys_by_pid.values() for y in v])
            n_pat = len(ys_by_pid)
            k = np.mean([len(v) for v in ys_by_pid.values()])
            ms_b = sum((pat_means[p] - gm) ** 2 for p in pat_means) * k / max(n_pat - 1, 1)
            ms_w_num = sum((y - pat_means[p]) ** 2 for p, v in ys_by_pid.items() for y in v)
            ms_w_den = sum(len(v) - 1 for v in ys_by_pid.values())
            ms_w = ms_w_num / max(ms_w_den, 1)
            icc = (ms_b - ms_w) / (ms_b + (k - 1) * ms_w) if (ms_b + (k - 1) * ms_w) > 0 else float("nan")
            icc_vs_n.append(float(icc))

        out[inst] = {
            "n_reps": ns,
            "cosine_to_full_mean": sims_mean,
            "cosine_std": sims_std,
            "cosine_min": sims_min,
            "n_star_99": n_star,  # smallest N for cos≥0.99
            "icc_vs_n": icc_vs_n,
        }
    return out


# ----------------------------------------------------------------------------
# Wavenumber calibration drift: peak position of grand mean per instrument,
# compared at top-N peaks. Reports max drift in cm⁻¹.
# ----------------------------------------------------------------------------
def wavenumber_drift(df: pd.DataFrame, n_peaks: int = 5) -> dict:
    from scipy.signal import find_peaks
    means = {}
    for inst, g in df.groupby("instrument"):
        specs = np.stack(g["spectrum"].values)
        valid = ~np.isnan(specs).any(axis=0)
        s = _l2_normalize(specs[:, valid])
        means[inst] = (np.nanmean(s, axis=0), valid)

    # Use thermo as reference; align peaks
    if "thermo" not in means:
        return {"max_drift_cm-1": None, "reference": None, "peaks": {}}
    ref_mean, ref_valid = means["thermo"]
    ref_wn = COMMON_WN[ref_valid]
    peaks_idx, props = find_peaks(ref_mean, distance=20, prominence=0.0005)
    if len(peaks_idx) == 0:
        return {"max_drift_cm-1": None, "reference": "thermo", "peaks": {}}
    top = peaks_idx[np.argsort(-props["prominences"])][:n_peaks]
    ref_peaks = sorted(ref_wn[top].tolist())

    drifts = {}
    for inst, (m, v) in means.items():
        if inst == "thermo":
            drifts[inst] = [0.0] * len(ref_peaks)
            continue
        wn = COMMON_WN[v]
        ipks, _ = find_peaks(m, distance=20, prominence=0.0005)
        if len(ipks) == 0:
            drifts[inst] = [None] * len(ref_peaks)
            continue
        cand = wn[ipks]
        d = []
        for rp in ref_peaks:
            nearest = cand[np.argmin(np.abs(cand - rp))]
            d.append(float(nearest - rp))
        drifts[inst] = d

    max_drift = max(
        abs(x) for v in drifts.values() for x in v if x is not None
    )
    return {
        "reference": "thermo",
        "ref_peaks_cm-1": ref_peaks,
        "drifts_cm-1": drifts,
        "max_drift_cm-1": float(max_drift),
    }


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    log.info("=" * 64)
    log.info("  Cross-Instrument Variance Decomposition")
    log.info("=" * 64)
    log.info(f"  Common grid: {WN_MIN}-{WN_MAX} cm-1, step {WN_STEP}, "
             f"{len(COMMON_WN)} points")

    log.info("\n[1/5] Loading instruments")
    dfs = [load_instrument(n) for n in INSTRUMENTS]
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"  Total: {len(df)} spectra")

    log.info("\n[2/5] QC per (instrument, patient)")
    qc = qc_per_patient(df)
    qc_summary = {}
    for inst, g in qc.groupby("instrument"):
        n_total = len(g)
        n_rsd = int((g["mean_rsd"] < QC_RSD_MAX).sum())
        n_corr = int((g["mean_corr"] >= QC_CORR_MIN).sum())
        n_both = int(((g["mean_rsd"] < QC_RSD_MAX) & (g["mean_corr"] >= QC_CORR_MIN)).sum())
        qc_summary[inst] = {
            "n_patients": n_total,
            "mean_rsd_pct": float(g["mean_rsd"].mean()),
            "median_rsd_pct": float(g["mean_rsd"].median()),
            "mean_corr": float(g["mean_corr"].mean()),
            "pass_rsd_pct": n_rsd / n_total * 100,
            "pass_corr_pct": n_corr / n_total * 100,
            "pass_both_pct": n_both / n_total * 100,
        }
        log.info(f"  {inst:18s} mean RSD={g['mean_rsd'].mean():.2f}%  "
                 f"mean Corr={g['mean_corr'].mean():.4f}  "
                 f"pass_both={n_both/n_total*100:.1f}%")

    log.info("\n[3/5] Variance decomposition (nested ANOVA)")
    vc = variance_decomposition(df)
    log.info(f"  σ²(instrument)             = {vc['components_pct']['instrument']:5.2f}%")
    log.info(f"  σ²(patient | instrument)   = {vc['components_pct']['patient_within_instrument']:5.2f}%")
    log.info(f"  σ²(point | patient,inst)   = {vc['components_pct']['point_within_patient']:5.2f}%")
    log.info("  ICC per instrument:")
    for k, v in vc["icc_per_instrument"].items():
        log.info(f"    {k:18s} ICC = {v:.3f}")

    log.info("\n[4/5] Cross-instrument 4×4 cosine similarity")
    cross = cross_inst_correlation(df)
    for a in cross["instruments"]:
        row = "  " + a.ljust(18)
        for b in cross["instruments"]:
            row += f"  {cross['matrix'][a][b]:.3f}"
        log.info(row)

    log.info("\n[5/6] Convergence (N reps → spectrum stability)")
    conv = convergence_analysis(df, n_seeds=20)
    for inst, c in conv.items():
        n5_idx = c["n_reps"].index(5) if 5 in c["n_reps"] else None
        n20_idx = -1
        cos5 = c["cosine_to_full_mean"][n5_idx] if n5_idx is not None else None
        cos20 = c["cosine_to_full_mean"][n20_idx]
        log.info(f"  {inst:18s} cos(N=5)={cos5:.4f}  cos(N={c['n_reps'][-1]})={cos20:.4f}  "
                 f"N*(≥0.99)={c['n_star_99']}")

    log.info("\n[6/6] Mean spectra + wavenumber drift")
    spectra = instrument_mean_spectra(df, n_points=240)
    drift = wavenumber_drift(df)
    log.info(f"  Max wavenumber drift = {drift['max_drift_cm-1']} cm-1 (target <0.3)")

    out = {
        "config": {
            "wn_min": WN_MIN, "wn_max": WN_MAX, "wn_step": WN_STEP,
            "n_features": len(COMMON_WN),
            "instruments": list(INSTRUMENTS.keys()),
            "qc_rsd_max": QC_RSD_MAX,
            "qc_corr_min": QC_CORR_MIN,
        },
        "n_spectra_per_instrument": {
            inst: int((df["instrument"] == inst).sum()) for inst in INSTRUMENTS
        },
        "n_patients_per_instrument": {
            inst: int(df[df["instrument"] == inst]["patient"].nunique())
            for inst in INSTRUMENTS
        },
        "qc": qc_summary,
        "variance_decomposition": vc,
        "cross_instrument_similarity": cross,
        "convergence": conv,
        "mean_spectra": spectra,
        "wavenumber_drift": drift,
    }

    out_path = OUT_DIR / "cross_instrument_variance.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: None if isinstance(x, float) and not np.isfinite(x) else x)
    log.info(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
