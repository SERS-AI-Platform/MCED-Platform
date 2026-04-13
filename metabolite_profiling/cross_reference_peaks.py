"""
Cross-reference cancer-type-specific peaks with Thermo metabolite assignments.
For each cancer type's top peaks, identify candidate metabolites from the Thermo library.
"""

import pandas as pd
import numpy as np
import json
import os

# ── Paths ──
BASE = "/home/user/SERS-AI"
PEAK_FILE = f"{BASE}/results/training/R_BLC_added/logistic_regression/v001/evaluation/stage2/peak_intensity_by_diagnosis.csv"
ASSIGN_FILE = f"{BASE}/metabolite_profiling/data/metabolite_peak_assignments.csv"
CORR_FILE = f"{BASE}/metabolite_profiling/data/metabolite_sers_correlation.csv"
THERMO_FILE = f"{BASE}/thermo_metabolite_data.json"
BAND_FILE = f"{BASE}/metabolite_profiling/data/metabolite_band_by_group.csv"
OUT_DIR = f"{BASE}/metabolite_profiling/data"

TOLERANCE = 15  # cm-1 tolerance for matching

# ── Load data ──
peaks_df = pd.read_csv(PEAK_FILE)
assign_df = pd.read_csv(ASSIGN_FILE)
corr_df = pd.read_csv(CORR_FILE)

with open(THERMO_FILE) as f:
    thermo_data = json.load(f)

# Build correlation lookup
corr_lookup = dict(zip(corr_df["metabolite"], corr_df["pearson_correlation"]))

# Build Thermo metabolite peak database (all peaks, not just top 5)
thermo_peaks = {}
for m in thermo_data["metabolites"]:
    name = m["name"]
    all_peaks = [p["wn"] for p in m["top_peaks"]]
    thermo_peaks[name] = all_peaks

# ── Known band assignments (from report) ──
BAND_ASSIGNMENTS = {
    "~618 (C-S stretch)": {"range": (610, 630), "metabolites": ["Cysteine", "Adenine", "Betaine", "Cholesterol", "Arginine", "Phenylalanine", "Hippuric acid"]},
    "~683 (Creatinine/ring)": {"range": (675, 695), "metabolites": ["Creatinine", "Guanine", "Purine", "Hippuric acid", "Tyrosine"]},
    "~724 (Adenine ring)": {"range": (715, 735), "metabolites": ["Adenine", "Hypoxanthine", "Hippuric acid", "Benzoic acid"]},
    "~795 (Hippuric)": {"range": (785, 805), "metabolites": ["Hippuric acid", "Kynurenine", "Uric acid"]},
    "~849 (Tyr/Trp)": {"range": (840, 860), "metabolites": ["Tyrosine", "Tryptophan", "Creatinine", "Cysteine"]},
    "~895 (C-C stretch)": {"range": (885, 905), "metabolites": ["Hippuric acid", "Trimethylamine-N-oxide", "Uric acid", "Cysteine"]},
    "~934 (C-C protein)": {"range": (925, 945), "metabolites": ["Multiple (>20)", "non-specific"]},
    "~999 (Phe ring)": {"range": (990, 1015), "metabolites": ["Phenylalanine", "2-Phenylacetamide", "Hippuric acid", "Benzoic acid"]},
    "~1148 (C-N/C-O-C)": {"range": (1138, 1158), "metabolites": ["Glycogen", "Glucose", "Xylose", "Cysteine"]},
    "~1231 (Amide III)": {"range": (1220, 1245), "metabolites": ["Tryptophan", "Taurine", "Kynurenine", "Threonine", "Isoleucine"]},
    "~1293 (Amide III/CH2)": {"range": (1283, 1305), "metabolites": ["O-Acetylcarnitine", "Acrylic acid", "Palmitic acid"]},
    "~1352 (CH def/Trp)": {"range": (1340, 1365), "metabolites": ["Adenine", "Guanine", "Tryptophan"]},
    "~1449 (CH2 def)": {"range": (1435, 1465), "metabolites": ["Nearly all (non-specific)", "Lipids", "Fatty acids"]},
    "~1597 (C=C/Purine)": {"range": (1585, 1610), "metabolites": ["Adenine", "Kynurenine", "Tyrosine", "Phenylalanine"]},
    "~1651 (Amide I)": {"range": (1640, 1665), "metabolites": ["Maleic acid", "Glycogen", "Kynurenine", "Stearic acid"]},
    "~1680 (C=O)": {"range": (1670, 1695), "metabolites": ["Guanine", "Uracil", "Maleic acid", "NADH"]},
    "~2098 (S-H / unknown)": {"range": (2085, 2115), "metabolites": ["Unknown / possibly S-H stretch or instrumental"]},
}


def find_metabolite_candidates(wavenumber, tolerance=TOLERANCE):
    """Find metabolite candidates from assignment table within tolerance."""
    mask = assign_df["sers_peak_cm1"].between(wavenumber - tolerance, wavenumber + tolerance)
    matches = assign_df[mask][["metabolite", "metabolite_peak_cm1", "delta_cm1", "vibration_mode"]].copy()
    # Add correlation
    matches["sers_correlation"] = matches["metabolite"].map(corr_lookup)
    matches = matches.sort_values("sers_correlation", ascending=False)
    return matches


def find_band_assignment(wavenumber):
    """Find the known band assignment for a wavenumber."""
    for band_name, info in BAND_ASSIGNMENTS.items():
        lo, hi = info["range"]
        if lo <= wavenumber <= hi:
            return band_name, info["metabolites"]
    return None, []


# ── Main analysis ──
cancer_types = peaks_df["diagnosis"].unique()
results = []

print("=" * 100)
print("SERS-AI: Cancer-Type Peak ↔ Thermo Metabolite Cross-Reference")
print("=" * 100)
print(f"\nTolerance: ±{TOLERANCE} cm⁻¹ | Cancer types: {len(cancer_types)} | Metabolites: {len(thermo_peaks)}")
print(f"Peak assignments: {len(assign_df)} | Metabolite correlations: {len(corr_df)}")

for cancer in cancer_types:
    cancer_df = peaks_df[peaks_df["diagnosis"] == cancer].sort_values("peak_rank")
    n_samples = cancer_df["n_samples"].iloc[0]

    print(f"\n{'─' * 100}")
    print(f"  {cancer} (n={n_samples}) — Top {len(cancer_df)} Peaks")
    print(f"{'─' * 100}")

    for _, row in cancer_df.iterrows():
        wn = row["wavenumber"]
        rank = int(row["peak_rank"])
        intensity = row["peak_intensity"]
        prominence = row["prominence"]

        # Find band assignment
        band_name, band_metabolites = find_band_assignment(wn)

        # Find detailed metabolite candidates
        candidates = find_metabolite_candidates(wn)

        print(f"\n  Peak #{rank}: {wn:.1f} cm⁻¹  (intensity={intensity:.2f}, prominence={prominence:.2f})")
        if band_name:
            print(f"  Band: {band_name}")
            print(f"  Known metabolites: {', '.join(band_metabolites[:5])}")
        else:
            print(f"  Band: (no predefined band)")

        # Top candidates from Thermo data
        if len(candidates) > 0:
            top = candidates.head(8)
            print(f"  Thermo candidates (top {len(top)}, sorted by SERS correlation):")
            for _, c in top.iterrows():
                corr_val = c["sers_correlation"]
                corr_str = f"r={corr_val:.3f}" if pd.notna(corr_val) else "r=N/A"
                vib = c["vibration_mode"] if pd.notna(c["vibration_mode"]) and c["vibration_mode"] != "unassigned" else ""
                print(f"    • {c['metabolite']:<30s} Δ={c['delta_cm1']:.1f} cm⁻¹  {corr_str}  {vib}")
        else:
            print(f"  Thermo candidates: None within ±{TOLERANCE} cm⁻¹")

        # Store result
        for _, c in candidates.iterrows():
            results.append({
                "cancer_type": cancer,
                "n_samples": n_samples,
                "peak_rank": rank,
                "peak_wavenumber": wn,
                "peak_intensity": intensity,
                "peak_prominence": prominence,
                "band_assignment": band_name or "",
                "metabolite": c["metabolite"],
                "metabolite_peak_cm1": c["metabolite_peak_cm1"],
                "delta_cm1": c["delta_cm1"],
                "vibration_mode": c["vibration_mode"],
                "sers_correlation": c["sers_correlation"],
            })

# ── Save results ──
results_df = pd.DataFrame(results)
out_path = f"{OUT_DIR}/cancer_peak_metabolite_crossref.csv"
results_df.to_csv(out_path, index=False)
print(f"\n\n{'=' * 100}")
print(f"Saved: {out_path}")
print(f"Total cross-references: {len(results_df)}")

# ── Summary table: unique metabolites per cancer type ──
print(f"\n\n{'=' * 100}")
print("SUMMARY: Top Metabolite Candidates per Cancer Type (by frequency × correlation)")
print(f"{'=' * 100}")

summary_rows = []
for cancer in cancer_types:
    sub = results_df[results_df["cancer_type"] == cancer].copy()
    if len(sub) == 0:
        continue

    # Score = count of peak appearances × mean correlation
    met_scores = sub.groupby("metabolite").agg(
        n_peaks=("peak_rank", "count"),
        mean_corr=("sers_correlation", "mean"),
        peaks_matched=("peak_wavenumber", lambda x: sorted(x.unique())),
        best_rank=("peak_rank", "min"),
    ).reset_index()
    met_scores["score"] = met_scores["n_peaks"] * met_scores["mean_corr"].fillna(0.3)
    met_scores = met_scores.sort_values("score", ascending=False)

    print(f"\n  {cancer}:")
    for _, m in met_scores.head(10).iterrows():
        peaks_str = ", ".join([f"{p:.0f}" for p in m["peaks_matched"]])
        corr_str = f"r={m['mean_corr']:.3f}" if pd.notna(m['mean_corr']) else "r=N/A"
        print(f"    {m['metabolite']:<30s} peaks: [{peaks_str}]  {corr_str}  score={m['score']:.2f}")

    for _, m in met_scores.iterrows():
        summary_rows.append({
            "cancer_type": cancer,
            "metabolite": m["metabolite"],
            "n_peaks_matched": m["n_peaks"],
            "mean_sers_correlation": m["mean_corr"],
            "peaks_matched": "; ".join([f"{p:.0f}" for p in m["peaks_matched"]]),
            "best_peak_rank": m["best_rank"],
            "composite_score": m["score"],
        })

summary_df = pd.DataFrame(summary_rows)
summary_path = f"{OUT_DIR}/cancer_metabolite_summary.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSaved: {summary_path}")

# ── Cancer-type metabolic signature comparison ──
print(f"\n\n{'=' * 100}")
print("METABOLIC SIGNATURE COMPARISON")
print(f"{'=' * 100}")

# For each cancer, show unique vs shared metabolites
for cancer in cancer_types:
    top_mets = set(
        summary_df[summary_df["cancer_type"] == cancer]
        .nlargest(5, "composite_score")["metabolite"]
    )
    other_top = set()
    for other in cancer_types:
        if other != cancer:
            other_top.update(
                summary_df[summary_df["cancer_type"] == other]
                .nlargest(5, "composite_score")["metabolite"]
            )
    unique = top_mets - other_top
    shared = top_mets & other_top
    print(f"\n  {cancer}:")
    print(f"    Shared (top-5 in other cancers too): {', '.join(sorted(shared)) or 'none'}")
    print(f"    Distinctive: {', '.join(sorted(unique)) or 'none'}")

print(f"\n\nDone.")
