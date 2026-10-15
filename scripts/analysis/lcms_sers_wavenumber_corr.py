"""Phase LCMS-2: wavenumber-wise correlation spectra between LC-MS/MS candidates and SERS.

Same subjects, preprocessing and lot handling as LCMS-1 (lcms_sers_candidate_comparison.py), but
instead of 17 bands every grid point 600-1800 cm-1 (1 cm-1) is correlated with each LC feature
(statistical heterospectroscopy style).

- Per wavenumber: Spearman inside each strip lot on detected subjects (>= 8), Fisher-z pooled.
- Family-wise error per feature x arm: max |rho_pooled| over wavenumbers vs the same max under
  N_PERM within-lot permutations of the LC values (detected subjects only). BH across feature x arm.

Run: PYTHONPATH=. python scripts/analysis/lcms_sers_wavenumber_corr.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from scripts.analysis.lcms_sers_candidate_comparison import (
    GRID, MIN_DETECTED, ROOT, bh, load_lcms, load_spectra, preprocess,
)
from src.sers.config import load_config
from src.sers.visualization import style

OUT = ROOT / "results/lcms_sers_wavenumber_corr_20261015"
N_PERM = 1000
N_GLOBAL_PERM = 500


def ranks_centered(a: np.ndarray) -> np.ndarray:
    r = stats.rankdata(a, axis=0)
    r = r - r.mean(axis=0)
    return r / np.linalg.norm(r, axis=0)


def corr_spectrum(y: pd.Series, X: pd.DataFrame, lot: pd.Series, rng: np.random.Generator):
    """Observed pooled rho per wavenumber and N_PERM permuted pooled rho rows."""
    zs, zp, ws, ns = [], [], [], {}
    for s in sorted(lot.unique()):
        idx = y.index[(lot.loc[y.index] == s) & y.notna()]
        ns[s] = len(idx)
        if len(idx) < MIN_DETECTED:
            continue
        rx = ranks_centered(X.loc[idx].values)                     # n x W
        ry = ranks_centered(y.loc[idx].values[:, None])[:, 0]      # n
        perm = np.array([rng.permutation(ry) for _ in range(N_PERM)])  # P x n
        clip = lambda r: np.arctanh(np.clip(r, -0.999, 0.999))
        zs.append(clip(ry @ rx))
        zp.append(clip(perm @ rx))
        ws.append(len(idx) - 3)
    if not ws:
        return None, None, ns
    w = np.array(ws, dtype=float)
    obs = np.tanh(np.tensordot(w, np.array(zs), 1) / w.sum())
    null = np.tanh(np.tensordot(w, np.array(zp), 1) / w.sum())
    return obs, null, ns


def pooled_rho(y: pd.Series, X: pd.DataFrame, lot: pd.Series) -> np.ndarray | None:
    """Observed pooled rho per wavenumber only (no permutations)."""
    zs, ws = [], []
    for s in sorted(lot.unique()):
        idx = y.index[(lot.loc[y.index] == s) & y.notna()]
        if len(idx) < MIN_DETECTED:
            continue
        r = ranks_centered(y.loc[idx].values[:, None])[:, 0] @ ranks_centered(X.loc[idx].values)
        zs.append(np.arctanh(np.clip(r, -0.999, 0.999)))
        ws.append(len(idx) - 3)
    if not ws:
        return None
    w = np.array(ws, dtype=float)
    return np.tanh(np.tensordot(w, np.array(zs), 1) / w.sum())


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pp = load_config().preprocessing
    meta, lc = load_lcms()
    spec = load_spectra(list(meta.index))
    lot = spec.groupby("solum_label").strip_lot_id.first().map(lambda v: f"lot{v}")
    A, B = preprocess(spec, pp)
    arms = {arm: pd.DataFrame(X).groupby(spec.solum_label.values).median().set_axis(GRID, axis=1)
            for arm, X in (("raw", A), ("snv", B))}

    rng = np.random.default_rng(0)
    rho_cols, rows, nulls = {}, [], {}
    for arm, X in arms.items():
        for f in lc.columns:
            obs, null, ns = corr_spectrum(lc[f], X, lot, rng)
            if obs is None:
                continue
            k = int(np.argmax(np.abs(obs)))
            max_null = np.abs(null).max(axis=1)
            thr = float(np.percentile(max_null, 95))
            rho_cols[f"{arm}|{f}"] = obs
            nulls[f"{arm}|{f}"] = thr
            rows.append({"arm": arm, "lcms_feature": f, **{f"n_{s}": n for s, n in ns.items()},
                         "max_abs_rho": float(abs(obs[k])), "rho_at_max": float(obs[k]),
                         "wavenumber_at_max": float(GRID[k]), "null95_max_abs_rho": thr,
                         "n_wavenumbers_over_null95": int((np.abs(obs) > thr).sum()),
                         "p_fwer": float((1 + (max_null >= abs(obs[k])).sum()) / (N_PERM + 1))})
    res = pd.DataFrame(rows)
    res["q_bh"] = bh(res.p_fwer)

    # global test: one permutation of whole LC rows within lot shared by all features and arms
    # (keeps POS/NEG and raw/SNV dependence); statistic = number of feature x arm over own null95
    thr = res.set_index(["arm", "lcms_feature"]).null95_max_abs_rho
    obs_count = int((res.max_abs_rho > res.null95_max_abs_rho).sum())
    strata = lot.loc[lc.index]
    null_counts = []
    for _ in range(N_GLOBAL_PERM):
        perm = lc.index.to_numpy().copy()
        for s in strata.unique():
            k = np.flatnonzero(strata.values == s)
            perm[k] = rng.permutation(perm[k])
        lcp = lc.loc[perm].set_axis(lc.index)
        c = 0
        for arm, X in arms.items():
            for f in lc.columns:
                if (arm, f) not in thr.index:
                    continue
                o = pooled_rho(lcp[f], X, lot)
                c += int(o is not None and np.abs(o).max() > thr[(arm, f)])
        null_counts.append(c)
    global_test = {"obs_count_over_null95": obs_count, "null_count_median": float(np.median(null_counts)),
                   "null_count_95pct": float(np.percentile(null_counts, 95)),
                   "p_global": float((1 + sum(n >= obs_count for n in null_counts)) / (N_GLOBAL_PERM + 1))}
    res = res.sort_values("p_fwer")
    res.to_csv(OUT / "feature_max_corr.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rho_cols, index=GRID).rename_axis("wavenumber").to_csv(
        OUT / "corr_spectra.csv", encoding="utf-8-sig")

    # figure: top 4 feature x arm correlation spectra with their permutation threshold
    style.apply_style()
    top = res.head(4)
    fig, axes = plt.subplots(len(top), 1, figsize=(13.0, 2.4 * len(top) + 1.0), sharex=True)
    mean_snv = arms["snv"].mean().values
    for ax, r in zip(np.atleast_1d(axes), top.itertuples()):
        key = f"{r.arm}|{r.lcms_feature}"
        ax2 = ax.twinx()
        ax2.plot(GRID, mean_snv, color=style.COLOR_REFERENCE, lw=0.8, alpha=0.6)
        ax2.set_yticks([])
        ax2.spines[["top", "right"]].set_visible(False)
        ax.plot(GRID, rho_cols[key], color=style.COLOR_TEXT, lw=1.4)
        for t in (nulls[key], -nulls[key]):
            ax.axhline(t, color=style.COLOR_MUTED, ls="--", lw=style.LINEWIDTH_REFERENCE)
        ax.axhline(0, color=style.COLOR_GRID, lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)
        name = r.lcms_feature.split(" ", 2)[2][:40]
        ax.text(0.01, 0.92, f"{name} ({r.arm})  max |rho| {r.max_abs_rho:.2f} @ {r.wavenumber_at_max:.0f}"
                f"  FWER p {r.p_fwer:.2f}", transform=ax.transAxes, fontsize=style.SIZE_ANNOTATION, va="top")
        style.style_axes(ax, grid="x", ylabel="rho")
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm-1)", fontsize=style.SIZE_LABEL)
    style.set_title(np.atleast_1d(axes)[0], "LC-MS/MS candidate vs SERS correlation spectra (top 4)",
                    "n=45 (normal+benign 22 = lot 1, cancer 23 = lot 6) | within-lot Spearman, Fisher-z pooled | "
                    f"dashed = permutation 95% max |rho| (n={N_PERM}) | grey = mean SNV spectrum")
    style.save_png(fig, OUT / "corr_spectra_top4.png")

    summary = {
        "phase": "LCMS-2", "n_subjects": int(len(meta)), "n_spectra": int(len(spec)),
        "n_wavenumbers": int(len(GRID)), "n_feature_arm": int(len(res)), "n_perm": N_PERM, "n_global_perm": N_GLOBAL_PERM,
        "n_p_fwer_05": int((res.p_fwer < 0.05).sum()), "n_q_bh_05": int((res.q_bh < 0.05).sum()),
        "min_p_fwer": float(res.p_fwer.min()), "global_test": global_test, "top": res.head(5)[["arm", "lcms_feature", "max_abs_rho",
                                                                    "wavenumber_at_max", "p_fwer"]].to_dict("records"),
        "config_sha256_12": hashlib.sha256((ROOT / "config/config.yaml").read_bytes()).hexdigest()[:12],
        "git": subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True).stdout.strip(),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
