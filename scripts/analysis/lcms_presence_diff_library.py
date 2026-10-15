"""Phase LCMS-4: LC-MS detected vs not-detected SERS difference spectra + internal standards library match.

For LC features that split subjects (>= 5 detected and >= 5 not detected inside a lot stratum):
1. Per wavenumber Welch t (detected - not) inside each lot, Stouffer-combined over eligible lots -> z spectrum.
   FWER per feature: max |z| vs N_PERM within-lot permutations of the detection labels.
2. Replication: same z spectra on July liquid (runs 98, 99) and powder (run 110); Pearson r with the Aug
   z spectrum vs N_SHAPE_PERM within-lot permutations of the July labels.
3. Reference: Aug SNV z spectrum vs each internal standard (2025 Thermo metabolite standards, 10 mM in water,
   pre-reagent-change substrate) preprocessed with the same SNV arm; Pearson r, best match p vs the best match
   of permuted z spectra. Candidates only - a spectral resemblance is not an identification.

Run: PYTHONPATH=. python scripts/analysis/lcms_presence_diff_library.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.analysis.lcms_sers_candidate_comparison import GRID, ROOT, bh, load_lcms, load_spectra, preprocess
from scripts.analysis.lcms_sers_july_replication import load_runs, permute_within
from src.sers.config import load_config
from src.sers.visualization import style

LIB_DIR = ROOT / "data/06_supporting_or_previous_outputs/thermo_metabolite_standards_20250828-20251203/Metabolite analysis_Thermo"
OUT = ROOT / "results/lcms_presence_diff_library_20261015"
MIN_EACH = 5
N_PERM = 1000
N_SHAPE_PERM = 200


def z_spectrum(det: pd.Series, X: pd.DataFrame, lot: pd.Series, lots: list[str]) -> np.ndarray:
    """Stouffer-combined Welch t (detected - not detected) per wavenumber over the given lots."""
    ts = []
    for s in lots:
        idx = det.index[lot.loc[det.index] == s]
        a, b = X.loc[idx[det.loc[idx]]].values, X.loc[idx[~det.loc[idx]]].values
        se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
        ts.append((a.mean(axis=0) - b.mean(axis=0)) / se)
    return np.sum(ts, axis=0) / np.sqrt(len(ts))


def eligible_lots(det: pd.Series, lot: pd.Series) -> list[str]:
    return [s for s in sorted(lot.unique())
            if min(det[lot == s].sum(), (~det[lot == s]).sum()) >= MIN_EACH]


def load_library(pp) -> pd.DataFrame:
    """Mean SNV spectrum per standard from replicate files '<name>_<k>.CSV' (averages '_ave' / bare names skipped)."""
    rows = {}
    for f in sorted(LIB_DIR.glob("*_[0-9]*.CSV")):
        name = f.stem.rsplit("_", 1)[0]
        d = np.loadtxt(f, delimiter=",")
        rows.setdefault(name, []).append((d[:, 0], d[:, 1]))
    spec = pd.DataFrame([{"name": n, "wavenumber": x, "intensities": y} for n, reps in rows.items() for x, y in reps])
    _, B = preprocess(spec, pp)
    return pd.DataFrame(B).groupby(spec.name.values).mean().set_axis(GRID, axis=1)


def rowwise_r(Z: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Pearson r between each row of Z (k x W) and each row of L (m x W) -> k x m."""
    zc = (Z - Z.mean(1, keepdims=True)) / Z.std(1, keepdims=True)
    lc = (L - L.mean(1, keepdims=True)) / L.std(1, keepdims=True)
    return zc @ lc.T / Z.shape[1]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pp = load_config().preprocessing
    meta, lc = load_lcms()
    lot = (meta.group == "cancer").map({True: "lot6", False: "lot1"}).loc[lc.index]  # = Aug strip lot split
    detected = lc.notna()

    sets = {"aug_mapping": load_spectra(list(meta.index)),
            "july_liquid": load_runs((98, 99), list(meta.index)),
            "july_powder": load_runs((110,), list(meta.index))}
    arms = {}
    for name, spec in sets.items():
        A, B = preprocess(spec, pp)
        for arm, X in (("raw", A), ("snv", B)):
            arms[(name, arm)] = pd.DataFrame(X).groupby(spec.solum_label.values).median().set_axis(GRID, axis=1).loc[lc.index]
    lib = load_library(pp)
    lib.to_csv(OUT / "library_snv_mean.csv", encoding="utf-8-sig")

    rng = np.random.default_rng(0)
    feats = {f: eligible_lots(detected[f], lot) for f in lc.columns}
    feats = {f: v for f, v in feats.items() if v}
    rows, zcols, lib_rows = [], {}, []
    for f, lots in feats.items():
        det = detected[f]
        for arm in ("snv", "raw"):
            X = arms[("aug_mapping", arm)]
            z = z_spectrum(det, X, lot, lots)
            null = np.array([z_spectrum(det.loc[permute_within(det.index, lot, rng)].set_axis(det.index), X, lot, lots)
                             for _ in range(N_PERM)])
            max_null = np.abs(null).max(axis=1)
            thr = float(np.percentile(max_null, 95))
            k = int(np.argmax(np.abs(z)))
            row = {"lcms_feature": f, "arm": arm, "lots": "+".join(lots),
                   **{f"det_{s}": f"{int(det[lot == s].sum())}/{int((lot == s).sum())}" for s in sorted(lot.unique())},
                   "max_abs_z": float(abs(z[k])), "z_at_max": float(z[k]), "wavenumber_at_max": float(GRID[k]),
                   "null95_max_abs_z": thr, "n_wavenumbers_over_null95": int((np.abs(z) > thr).sum()),
                   "p_fwer": float((1 + (max_null >= abs(z[k])).sum()) / (N_PERM + 1))}
            for jset in ("july_liquid", "july_powder"):
                Xj = arms[(jset, arm)]
                zj = z_spectrum(det, Xj, lot, lots)
                r = float(np.corrcoef(z, zj)[0, 1])
                nr = np.array([np.corrcoef(z, z_spectrum(det.loc[permute_within(det.index, lot, rng)].set_axis(det.index),
                                                          Xj, lot, lots))[0, 1] for _ in range(N_SHAPE_PERM)])
                row[f"{jset}_shape_r"] = r
                row[f"{jset}_p"] = float((1 + (nr >= r).sum()) / (N_SHAPE_PERM + 1))
            zcols[f"{arm}|{f}"] = z
            if arm == "snv":
                r_lib = rowwise_r(z[None, :], lib.values)[0]
                r_null = rowwise_r(null[:200], lib.values).max(axis=1)
                order = np.argsort(-r_lib)
                for rank, j in enumerate(order[:3], 1):
                    lib_rows.append({"lcms_feature": f, "rank": rank, "standard": lib.index[j], "r": float(r_lib[j]),
                                     "p_best_vs_perm": float((1 + (r_null >= r_lib[order[0]]).sum()) / (len(r_null) + 1))
                                     if rank == 1 else np.nan})
            rows.append(row)
    res = pd.DataFrame(rows)
    for p_col, q_col in (("p_fwer", "q_fwer"), ("july_liquid_p", "july_liquid_q"), ("july_powder_p", "july_powder_q")):
        res[q_col] = res.groupby("arm")[p_col].transform(bh)
    res = res.sort_values(["arm", "p_fwer"], ascending=[False, True])
    res.to_csv(OUT / "presence_diff_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(zcols, index=GRID).rename_axis("wavenumber").to_csv(OUT / "presence_z_spectra.csv", encoding="utf-8-sig")
    libdf = pd.DataFrame(lib_rows)
    libdf.to_csv(OUT / "library_match_top3.csv", index=False, encoding="utf-8-sig")

    # figure: SNV z spectra of the 4 strongest features, threshold, best library match (scaled) in grey
    style.apply_style()
    top = res[res.arm == "snv"].head(4)
    fig, axes = plt.subplots(len(top), 1, figsize=(10.0, 2.4 * len(top) + 1.0), sharex=True)
    for ax, r in zip(np.atleast_1d(axes), top.itertuples()):
        z = zcols[f"snv|{r.lcms_feature}"]
        best = libdf[(libdf.lcms_feature == r.lcms_feature) & (libdf["rank"] == 1)].iloc[0]
        ax2 = ax.twinx()
        ax2.plot(GRID, lib.loc[best.standard].values, color=style.COLOR_REFERENCE, lw=0.9)
        ax2.set_yticks([])
        ax2.spines[["top", "right"]].set_visible(False)
        ax.plot(GRID, z, color=style.COLOR_TEXT, lw=1.3)
        for t in (r.null95_max_abs_z, -r.null95_max_abs_z):
            ax.axhline(t, color=style.COLOR_MUTED, ls="--", lw=style.LINEWIDTH_REFERENCE)
        ax.axhline(0, color=style.COLOR_GRID, lw=0.8)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        name = r.lcms_feature.split(" ", 2)[2][:38]
        ax.text(0.01, 0.95, f"{name}  (det lot1 {r.det_lot1}, lot6 {r.det_lot6})  FWER p {r.p_fwer:.3f}  |  "
                f"July powder r {r.july_powder_shape_r:.2f}  |  best standard: {best.standard} (r {best.r:.2f})",
                transform=ax.transAxes, fontsize=style.SIZE_ANNOTATION - 1, va="top")
        style.style_axes(ax, grid="x", ylabel="z")
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm-1)", fontsize=style.SIZE_LABEL)
    style.set_title(np.atleast_1d(axes)[0], "SERS difference: LC-MS detected minus not detected (SNV)",
                    "n=45 Boramae Aug mapping | within-lot Welch t, Stouffer z | dashed = 95% of permutation max |z| "
                    f"(n={N_PERM}) | grey = best-matching 2025 standard (scaled)")
    style.save_png(fig, OUT / "presence_diff_top4.png")

    summary = {
        "phase": "LCMS-4", "n_features_eligible": len(feats), "eligible": {f: v for f, v in feats.items()},
        "n_library_standards": int(len(lib)), "n_perm": N_PERM, "n_shape_perm": N_SHAPE_PERM,
        "snv_fwer_p05": int((res[res.arm == "snv"].p_fwer < 0.05).sum()),
        "raw_fwer_p05": int((res[res.arm == "raw"].p_fwer < 0.05).sum()),
        "snv_powder_shape_p05": int((res[res.arm == "snv"].july_powder_p < 0.05).sum()),
        "snv_liquid_shape_p05": int((res[res.arm == "snv"].july_liquid_p < 0.05).sum()),
        "library_best_p05": int((libdf.p_best_vs_perm < 0.05).sum()),
        "config_sha256_12": hashlib.sha256((ROOT / "config/config.yaml").read_bytes()).hexdigest()[:12],
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
