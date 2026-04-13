"""
Metabolite Raman spectra analysis: peak detection, SERS urine comparison,
peak assignment, and visualization.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(os.path.dirname(BASE))
METABOLITE_DIR = os.path.join(PROJECT, "data", "Metabolite analysis_Thermo",
                              "Metabolite analysis_Thermo")
SERS_CSV = os.path.join(PROJECT, "results", "processed_spectra.csv")
OUT_DIR = os.path.join(BASE, "standardized")
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Known SERS/Raman vibration assignments ─────────────────────────────────
KNOWN_ASSIGNMENTS = {
    620: "C-S stretch (cysteine, methionine)",
    640: "C-S stretch, tyrosine",
    725: "C-N stretch, adenine ring breathing",
    750: "tryptophan ring breathing",
    830: "tyrosine Fermi resonance",
    855: "tyrosine ring breathing",
    1003: "phenylalanine ring breathing (strongest SERS peak)",
    1030: "phenylalanine C-H in-plane",
    1130: "C-N stretch",
    1175: "tyrosine, phenylalanine",
    1210: "C-C6H5 stretch, phenylalanine, tryptophan",
    1265: "Amide III",
    1340: "CH deformation, tryptophan",
    1400: "COO- symmetric stretch",
    1450: "CH2 deformation",
    1550: "tryptophan, amide II",
    1580: "purine ring, nucleotides",
    1620: "C=C stretch",
    1650: "Amide I (C=O stretch)",
}

# ── 1. Load metabolite spectra and average replicates ──────────────────────
print("=" * 70)
print("Step 1: Loading metabolite Raman spectra")
print("=" * 70)

csv_files = sorted(glob.glob(os.path.join(METABOLITE_DIR, "*.CSV")))
print(f"Found {len(csv_files)} CSV files")

# Group files by metabolite name
metabolite_files = {}
for fp in csv_files:
    fname = os.path.splitext(os.path.basename(fp))[0]
    # Skip _ave files (pre-computed averages)
    if fname.endswith("_ave"):
        continue
    # Remove replicate suffix: name_1, name_2, ...
    base = re.sub(r'_\d+$', '', fname)
    metabolite_files.setdefault(base, []).append(fp)

print(f"Found {len(metabolite_files)} unique metabolites")

# Load and average
metabolite_spectra = {}  # name -> (wavenumber, mean_intensity)
for name, files in sorted(metabolite_files.items()):
    intensities = []
    wn = None
    for fp in files:
        try:
            data = np.loadtxt(fp, delimiter=',')
            if wn is None:
                wn = data[:, 0]
            intensities.append(data[:, 1])
        except Exception as e:
            print(f"  Warning: could not load {fp}: {e}")
    if intensities and wn is not None:
        mean_int = np.mean(intensities, axis=0)
        metabolite_spectra[name] = (wn, mean_int)

print(f"Successfully loaded {len(metabolite_spectra)} metabolites")
for name in sorted(metabolite_spectra.keys()):
    n_files = len(metabolite_files[name])
    print(f"  {name}: {n_files} replicates")

# ── 2. Filter to fingerprint region (400-1800 cm-1) ───────────────────────
print("\n" + "=" * 70)
print("Step 2: Filtering to fingerprint region (400-1800 cm-1)")
print("=" * 70)

WMIN, WMAX = 400, 1800

metabolite_fp = {}  # fingerprint-filtered spectra
for name, (wn, intensity) in metabolite_spectra.items():
    mask = (wn >= WMIN) & (wn <= WMAX)
    if mask.sum() > 10:
        metabolite_fp[name] = (wn[mask], intensity[mask])

print(f"{len(metabolite_fp)} metabolites have data in 400-1800 cm-1 region")

# ── 3. Load SERS urine spectra and compute mean ───────────────────────────
print("\n" + "=" * 70)
print("Step 3: Loading SERS urine spectra")
print("=" * 70)

sers_df = pd.read_csv(SERS_CSV)
# Extract wavenumber columns
wn_cols = [c for c in sers_df.columns if c.startswith('x_')]
sers_wn = np.array([float(c.replace('x_', '')) for c in wn_cols])
sers_mean = sers_df[wn_cols].mean(axis=0).values

# Filter to fingerprint
sers_mask = (sers_wn >= WMIN) & (sers_wn <= WMAX)
sers_wn_fp = sers_wn[sers_mask]
sers_mean_fp = sers_mean[sers_mask]

print(f"SERS data: {len(sers_df)} spectra, {len(wn_cols)} wavenumber points")
print(f"Fingerprint region: {len(sers_wn_fp)} points, {sers_wn_fp[0]:.1f}-{sers_wn_fp[-1]:.1f} cm-1")

# ── 4. Detect peaks ───────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 4: Peak detection")
print("=" * 70)

def detect_peaks(wn, intensity, prominence_factor=0.05, min_distance=5):
    """Detect peaks with smoothing."""
    # Smooth first
    if len(intensity) > 15:
        smooth = savgol_filter(intensity, window_length=11, polyorder=3)
    else:
        smooth = intensity
    prom = prominence_factor * (np.max(smooth) - np.min(smooth))
    peaks, props = find_peaks(smooth, prominence=prom, distance=min_distance)
    return wn[peaks], smooth[peaks], props

# Detect SERS urine peaks
sers_peaks_wn, sers_peaks_int, _ = detect_peaks(sers_wn_fp, sers_mean_fp,
                                                  prominence_factor=0.03)
print(f"SERS urine mean spectrum: {len(sers_peaks_wn)} peaks detected")
print(f"  Major peaks at: {', '.join(f'{w:.0f}' for w in sers_peaks_wn)} cm-1")

# Detect metabolite peaks
metabolite_peaks = {}
for name, (wn, intensity) in metabolite_fp.items():
    pw, pi, _ = detect_peaks(wn, intensity, prominence_factor=0.05)
    metabolite_peaks[name] = pw

print(f"\nPeak detection complete for {len(metabolite_peaks)} metabolites")

# ── 5. Peak matching (±10 cm-1) ──────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 5: Peak matching (SERS peaks vs metabolite peaks, ±10 cm-1)")
print("=" * 70)

TOLERANCE = 10  # cm-1

peak_assignments = []
for sp in sers_peaks_wn:
    matches = []
    for name, mpks in metabolite_peaks.items():
        diffs = np.abs(mpks - sp)
        if np.any(diffs <= TOLERANCE):
            best_idx = np.argmin(diffs)
            matches.append((name, mpks[best_idx], diffs[best_idx]))

    # Find closest known vibration assignment
    known_dists = {k: abs(k - sp) for k in KNOWN_ASSIGNMENTS}
    closest_known = min(known_dists, key=known_dists.get)
    vibration = KNOWN_ASSIGNMENTS[closest_known] if known_dists[closest_known] <= 30 else "unassigned"

    for m_name, m_wn, m_diff in matches:
        peak_assignments.append({
            'sers_peak_cm1': round(sp, 1),
            'metabolite': m_name,
            'metabolite_peak_cm1': round(m_wn, 1),
            'delta_cm1': round(m_diff, 1),
            'vibration_mode': vibration,
        })

    if not matches:
        peak_assignments.append({
            'sers_peak_cm1': round(sp, 1),
            'metabolite': 'none',
            'metabolite_peak_cm1': np.nan,
            'delta_cm1': np.nan,
            'vibration_mode': vibration,
        })

assign_df = pd.DataFrame(peak_assignments)
print(f"Total peak-metabolite matches: {len(assign_df[assign_df['metabolite'] != 'none'])}")
print(f"SERS peaks with no metabolite match: {(assign_df['metabolite'] == 'none').sum()}")

# Summary per SERS peak
for sp in sers_peaks_wn:
    sub = assign_df[(assign_df['sers_peak_cm1'] == round(sp, 1)) &
                     (assign_df['metabolite'] != 'none')]
    if len(sub) > 0:
        mets = ', '.join(sub['metabolite'].tolist()[:5])
        if len(sub) > 5:
            mets += f' ... (+{len(sub)-5} more)'
        print(f"  {sp:.0f} cm-1: {len(sub)} matches [{mets}]")

# ── 6. Spectral correlation (interpolate metabolite to SERS grid) ─────────
print("\n" + "=" * 70)
print("Step 6: Computing spectral correlation")
print("=" * 70)

correlations = {}
for name, (wn, intensity) in metabolite_fp.items():
    # Interpolate metabolite spectrum onto SERS wavenumber grid
    # Only use overlapping range
    overlap_min = max(wn.min(), sers_wn_fp.min())
    overlap_max = min(wn.max(), sers_wn_fp.max())

    if overlap_max - overlap_min < 100:
        continue

    sers_mask_ov = (sers_wn_fp >= overlap_min) & (sers_wn_fp <= overlap_max)
    if sers_mask_ov.sum() < 50:
        continue

    try:
        f_interp = interp1d(wn, intensity, kind='linear', bounds_error=False,
                           fill_value=0)
        met_interp = f_interp(sers_wn_fp[sers_mask_ov])
        sers_ov = sers_mean_fp[sers_mask_ov]

        # Normalize both
        met_norm = (met_interp - met_interp.mean()) / (met_interp.std() + 1e-10)
        sers_norm = (sers_ov - sers_ov.mean()) / (sers_ov.std() + 1e-10)

        corr = np.corrcoef(met_norm, sers_norm)[0, 1]
        correlations[name] = corr
    except Exception:
        pass

corr_df = pd.DataFrame([
    {'metabolite': k, 'pearson_correlation': round(v, 4)}
    for k, v in sorted(correlations.items(), key=lambda x: -x[1])
])
print(f"Computed correlations for {len(corr_df)} metabolites")
print("\nTop 15 metabolites by correlation with mean urine SERS:")
print(corr_df.head(15).to_string(index=False))

# ── 7. Save CSVs ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 7: Saving results")
print("=" * 70)

assign_path = os.path.join(OUT_DIR, "metabolite_peak_assignments.csv")
assign_df.to_csv(assign_path, index=False)
print(f"Saved: {assign_path}")

corr_path = os.path.join(OUT_DIR, "metabolite_sers_correlation.csv")
corr_df.to_csv(corr_path, index=False)
print(f"Saved: {corr_path}")

# ── 8. Figures ─────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 8: Generating figures")
print("=" * 70)

# ── Fig 11: Key metabolite overlay on SERS bands ──────────────────────────
BAND_REGIONS = [
    (600, 650, "600-650"),
    (720, 760, "720-760"),
    (1000, 1050, "1000-1050"),
    (1200, 1300, "1200-1300"),
    (1400, 1500, "1400-1500"),
    (1600, 1700, "1600-1700"),
]

# For each band region, find metabolites that have peaks there
band_metabolites = {}
for lo, hi, label in BAND_REGIONS:
    mets_in_band = set()
    for name, pks in metabolite_peaks.items():
        if np.any((pks >= lo) & (pks <= hi)):
            mets_in_band.add(name)
    band_metabolites[label] = mets_in_band

# Select top metabolites per band (by correlation), take union
key_metabolites = set()
for label, mets in band_metabolites.items():
    # Sort by correlation and take top 3
    ranked = sorted(mets, key=lambda m: correlations.get(m, -1), reverse=True)
    key_metabolites.update(ranked[:3])

key_metabolites = sorted(key_metabolites)
print(f"Fig 11: Overlaying {len(key_metabolites)} key metabolites")

fig, ax = plt.subplots(figsize=(14, 8))

# Plot mean SERS spectrum
sers_norm_plot = (sers_mean_fp - sers_mean_fp.min()) / (sers_mean_fp.max() - sers_mean_fp.min())
ax.plot(sers_wn_fp, sers_norm_plot, 'k-', linewidth=2.5, label='Mean Urine SERS', zorder=10)

# Shade band regions
colors_band = ['#FFE0E0', '#E0FFE0', '#E0E0FF', '#FFFFE0', '#FFE0FF', '#E0FFFF']
for i, (lo, hi, label) in enumerate(BAND_REGIONS):
    ax.axvspan(lo, hi, alpha=0.15, color=colors_band[i % len(colors_band)],
               label=f'{label} cm-1')

# Plot key metabolites (normalize each to 0-1)
cmap = plt.cm.tab20
for idx, name in enumerate(key_metabolites):
    wn, intensity = metabolite_fp[name]
    norm_int = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-10)
    ax.plot(wn, norm_int, linewidth=1.0, alpha=0.7,
            color=cmap(idx / max(len(key_metabolites), 1)),
            label=name)

ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=13)
ax.set_ylabel('Normalized Intensity', fontsize=13)
ax.set_title('Fig 11: Key Metabolite Spectra Overlaid on Major SERS Bands', fontsize=14)
ax.set_xlim(400, 1800)
ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.9)
ax.tick_params(labelsize=11)
plt.tight_layout()
fig11_path = os.path.join(FIG_DIR, "fig11_metabolite_sers_overlay.png")
fig.savefig(fig11_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig11_path}")

# ── Fig 12: Peak assignment heatmap ───────────────────────────────────────
print("Generating Fig 12: Peak assignment heatmap")

# Create binary matrix: metabolites x SERS peak regions
region_labels = [f"{lo}-{hi}" for lo, hi, _ in BAND_REGIONS]

# Get all metabolites that matched any peak
matched_mets = sorted(assign_df[assign_df['metabolite'] != 'none']['metabolite'].unique())

# Build heatmap matrix
heatmap_data = np.zeros((len(matched_mets), len(BAND_REGIONS)))
for i, met in enumerate(matched_mets):
    met_pks = metabolite_peaks.get(met, np.array([]))
    for j, (lo, hi, _) in enumerate(BAND_REGIONS):
        if len(met_pks) > 0 and np.any((met_pks >= lo) & (met_pks <= hi)):
            # Count number of peaks in region
            heatmap_data[i, j] = np.sum((met_pks >= lo) & (met_pks <= hi))

# Filter to metabolites with at least 1 match in any band
row_sums = heatmap_data.sum(axis=1)
keep = row_sums > 0
heatmap_data = heatmap_data[keep]
matched_mets = [m for m, k in zip(matched_mets, keep) if k]

# If too many metabolites, show top 30 by total matches
if len(matched_mets) > 30:
    top_idx = np.argsort(-heatmap_data.sum(axis=1))[:30]
    heatmap_data = heatmap_data[top_idx]
    matched_mets = [matched_mets[i] for i in top_idx]

fig, ax = plt.subplots(figsize=(10, max(8, len(matched_mets) * 0.35)))
cmap_heat = LinearSegmentedColormap.from_list('custom', ['white', '#4CAF50', '#1B5E20'])
im = ax.imshow(heatmap_data, aspect='auto', cmap=cmap_heat, interpolation='nearest')

ax.set_xticks(range(len(region_labels)))
ax.set_xticklabels([f"{lo}-{hi}" for lo, hi, _ in BAND_REGIONS], fontsize=10, rotation=45, ha='right')
ax.set_yticks(range(len(matched_mets)))
ax.set_yticklabels(matched_mets, fontsize=8)
ax.set_xlabel('SERS Peak Region (cm$^{-1}$)', fontsize=12)
ax.set_ylabel('Metabolite', fontsize=12)
ax.set_title('Fig 12: Metabolite Peak Assignment Heatmap\n(color = number of peaks in region)', fontsize=13)

# Add text annotations
for i in range(len(matched_mets)):
    for j in range(len(BAND_REGIONS)):
        val = int(heatmap_data[i, j])
        if val > 0:
            ax.text(j, i, str(val), ha='center', va='center', fontsize=7,
                    color='white' if val >= 2 else 'black')

plt.colorbar(im, ax=ax, shrink=0.5, label='Number of peaks')
plt.tight_layout()
fig12_path = os.path.join(FIG_DIR, "fig12_peak_assignment_heatmap.png")
fig.savefig(fig12_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig12_path}")

# ── Fig 13: Top 10 metabolites by spectral similarity ────────────────────
print("Generating Fig 13: Top 10 metabolites by correlation")

top10 = corr_df.head(10)

fig, axes = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [1, 2.5]})

# Panel A: bar chart
ax = axes[0]
colors_bar = plt.cm.RdYlGn(np.linspace(0.3, 0.95, len(top10)))
bars = ax.barh(range(len(top10)), top10['pearson_correlation'].values, color=colors_bar)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['metabolite'].values, fontsize=9)
ax.set_xlabel('Pearson Correlation with Mean Urine SERS', fontsize=11)
ax.set_title('Fig 13: Top 10 Metabolites with Highest Spectral Similarity to Mean Urine SERS', fontsize=13)
ax.invert_yaxis()
for i, v in enumerate(top10['pearson_correlation'].values):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
ax.set_xlim(0, max(top10['pearson_correlation'].values) * 1.15)

# Panel B: spectral overlay
ax = axes[1]
ax.plot(sers_wn_fp, sers_norm_plot, 'k-', linewidth=2.5, label='Mean Urine SERS', zorder=10)

cmap_lines = plt.cm.tab10
for idx, row in top10.iterrows():
    name = row['metabolite']
    if name in metabolite_fp:
        wn, intensity = metabolite_fp[name]
        norm_int = (intensity - intensity.min()) / (intensity.max() - intensity.min() + 1e-10)
        ax.plot(wn, norm_int, linewidth=1.2, alpha=0.7,
                color=cmap_lines(idx % 10),
                label=f"{name} (r={row['pearson_correlation']:.3f})")

ax.set_xlabel('Raman Shift (cm$^{-1}$)', fontsize=12)
ax.set_ylabel('Normalized Intensity', fontsize=12)
ax.set_xlim(400, 1800)
ax.legend(fontsize=7, loc='upper right', ncol=2, framealpha=0.9)
ax.tick_params(labelsize=10)

plt.tight_layout()
fig13_path = os.path.join(FIG_DIR, "fig13_top10_correlation.png")
fig.savefig(fig13_path, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {fig13_path}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOutputs:")
print(f"  1. {assign_path}")
print(f"  2. {corr_path}")
print(f"  3. {fig11_path}")
print(f"  4. {fig12_path}")
print(f"  5. {fig13_path}")
print(f"\nMetabolites analyzed: {len(metabolite_fp)}")
print(f"SERS peaks detected: {len(sers_peaks_wn)}")
print(f"Peak-metabolite matches: {len(assign_df[assign_df['metabolite'] != 'none'])}")
print(f"Top correlated metabolite: {corr_df.iloc[0]['metabolite']} (r={corr_df.iloc[0]['pearson_correlation']:.4f})")
