"""Batch-effect diagnosis: training PRO vs Boramae BPRO.

Compares peak positions, peak distribution, and intensity at two stages:
  (1) RAW (resampled to the model common grid, no normalization) — physical
      intensity scale, baseline, and peak wavenumber positions.
  (2) MODEL space (the predictor's full thermo preprocessing incl. calibration +
      SNV) — what the model actually sees. If groups still diverge here, the
      shift is real for the model (SNV already removes pure scale differences).
"""

import glob
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "deployment"))
import matplotlib  # noqa: E402
from sers_predict import StackingPredictor  # noqa: E402

from src.sers.io import read_spectrum  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

DATA_BPRO = Path(
    os.environ.get(
        "SERS_BORAMAE_BPRO_DIR",
        REPO / "data" / "Thermo" / "20260602_Urine test" / "BPRO",
    )
)
MODEL_DIR = REPO / "artifacts" / "usersnet" / "v1.0.0"
OUT = REPO / "results" / "boramae_20260602_validation"
CAP = 200  # max spectra per group (for a stable mean; full set not needed)

grid = np.load(MODEL_DIR / "common_grid.npy")
pred = StackingPredictor(artifact_dir=MODEL_DIR)
prep_cfg = json.load(open(MODEL_DIR / "preprocessing.json"))
print(
    "preprocessing.json calibration-related keys:",
    {
        k: v
        for k, v in prep_cfg.items()
        if "calib" in k.lower() or "snv" in k.lower() or "baseline" in k.lower()
    },
)


def collect(files):
    files = [f for f in files if "_ave" not in os.path.basename(f)][:CAP]
    raw, model = [], []
    for f in files:
        try:
            x, y = read_spectrum(Path(f))
        except (OSError, ValueError):
            d = np.loadtxt(f, delimiter=",")
            x, y = d[:, 0], d[:, 1]
        raw.append(np.interp(grid, x, y))
        try:
            model.append(np.asarray(pred.preprocess(x, y, instrument="thermo")).ravel())
        except (OSError, ValueError) as exc:
            print(f"Skipping preprocessing for {f}: {exc}")
    return np.array(raw), np.array(model), len(files)


pro_files = glob.glob(str(REPO / "data" / "raw_data" / "**" / "PRO *.CSV"), recursive=True)
bpro_files = glob.glob(os.path.join(DATA_BPRO, "*.CSV"))
pro_raw, pro_mod, n_pro = collect(pro_files)
bpro_raw, bpro_mod, n_bpro = collect(bpro_files)
nor_files = glob.glob(str(REPO / "data" / "raw_data" / "**" / "NOR *.CSV"), recursive=True)
_, nor_mod, n_nor = collect(nor_files)
print(
    f"PRO(train): {n_pro} | BPRO(new): {n_bpro} | NOR(train): {n_nor} spectra | grid {grid.min():.0f}-{grid.max():.0f} cm-1 ({len(grid)} pts)"
)


def peak_table(mean, name, n=10, prom=None):
    prom = prom if prom is not None else (mean.max() - mean.min()) * 0.05
    idx, _ = find_peaks(mean, prominence=prom, distance=8)
    idx = idx[np.argsort(mean[idx])[::-1][:n]]
    idx = idx[np.argsort(grid[idx])]
    print(f"  {name} top peaks (cm-1): " + ", ".join(f"{grid[i]:.0f}" for i in idx))
    return grid[idx]


# ---- intensity stats (raw) ----
print("\n=== RAW intensity (per-spectrum) ===")
for nm, M in [("PRO(train)", pro_raw), ("BPRO(new)", bpro_raw)]:
    print(
        f"  {nm}: max median={np.median(M.max(1)):.0f}  | overall median={np.median(M):.0f}  "
        f"| min median={np.median(M.min(1)):.0f}  (negative => background-subtracted)"
    )
ratio = np.median(bpro_raw.max(1)) / np.median(pro_raw.max(1))
print(f"  intensity scale ratio (BPRO/PRO, by median-max): {ratio:.2f}x")

# ---- peak positions ----
print("\n=== RAW peak positions ===")
pk_pro = peak_table(pro_raw.mean(0), "PRO(train)")
pk_bpro = peak_table(bpro_raw.mean(0), "BPRO(new)")
print("\n=== MODEL-space peak positions (after calibration+SNV) ===")
pk_pro_m = peak_table(pro_mod.mean(0), "PRO(train)") if len(pro_mod) else None
pk_bpro_m = peak_table(bpro_mod.mean(0), "BPRO(new)") if len(bpro_mod) else None


# model-space divergence metric: cosine similarity of mean spectra (with reference baselines)
def _cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


cos = 0.0
if len(pro_mod) and len(bpro_mod):
    cos = _cos(pro_mod.mean(0), bpro_mod.mean(0))
    print("\n=== MODEL-space mean-spectrum cosine similarity (1.0 = identical) ===")
    if len(pro_mod) > 4:
        h = len(pro_mod) // 2
        print(
            f"  PRO vs PRO  (within-class, 2 halves): {_cos(pro_mod[:h].mean(0), pro_mod[h:].mean(0)):.3f}   <- upper bound (same cohort)"
        )
    if len(nor_mod):
        print(
            f"  PRO vs NOR  (two TRAINING classes):   {_cos(pro_mod.mean(0), nor_mod.mean(0)):.3f}   <- how different two real classes are"
        )
    print(f"  PRO vs BPRO (train vs NEW cohort):     {cos:.3f}   <- the comparison")

# ---- figures ----
OUT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(3, 1, figsize=(11, 12))

# (1) raw mean +- std
for nm, M, c in [
    ("PRO train (n=%d)" % n_pro, pro_raw, "#2b6cb0"),
    ("BPRO new (n=%d)" % n_bpro, bpro_raw, "#c53030"),
]:
    m, s = M.mean(0), M.std(0)
    axes[0].plot(grid, m, color=c, label=nm, lw=1.6)
    axes[0].fill_between(grid, m - s, m + s, color=c, alpha=0.15)
axes[0].set_title("(1) RAW mean ± std (resampled, no normalization)")
axes[0].set_xlabel("Wavenumber (cm⁻¹)")
axes[0].set_ylabel("Intensity (a.u.)")
axes[0].legend()

# (2) model-space mean
if len(pro_mod) and len(bpro_mod):
    for nm, M, c in [("PRO train", pro_mod, "#2b6cb0"), ("BPRO new", bpro_mod, "#c53030")]:
        axes[1].plot(grid, M.mean(0), color=c, label=nm, lw=1.6)
    axes[1].set_title(f"(2) MODEL-space mean (calibration+SNV) — cosine sim={cos:.3f}")
    axes[1].set_xlabel("Wavenumber (cm⁻¹)")
    axes[1].set_ylabel("preprocessed")
    axes[1].legend()
    # (3) difference
    axes[2].plot(grid, bpro_mod.mean(0) - pro_mod.mean(0), color="#6b46c1", lw=1.2)
    axes[2].axhline(0, color="gray", lw=0.6)
    axes[2].set_title("(3) MODEL-space difference (BPRO − PRO)")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].set_ylabel("Δ preprocessed")

fig.tight_layout()
fig.savefig(OUT / "batcheffect_pro_vs_bpro.png", dpi=180)
plt.close(fig)
print(f"\nSaved figure: {OUT / 'batcheffect_pro_vs_bpro.png'}")
