# SERS-AI Metabolite Profiling Report

**Project:** SERS-based Urine Screening for Multi-Cancer Detection
**Date:** 2026-03-18
**Data Source:** 73 urinary metabolite standards measured on Thermo Raman spectrometer
**Analysis Script:** `analyze_metabolites.py`

---

## 1. Overview

This report documents the Raman spectral profiling of 73 urinary metabolites and their mapping to SERS urine spectral features. The goal is to provide molecular-level interpretation of SERS peaks observed in patient urine samples, enabling biologically meaningful feature assignment for cancer detection.

| Item | Value |
|---|---|
| Metabolites measured | **73 compounds** |
| Replicates per compound | 5–6 |
| Total spectra files | 439 CSV |
| Fingerprint region | 400–1800 cm⁻¹ |
| SERS peaks detected | 17 |
| Peak-metabolite matches | 527 (within ±10 cm⁻¹) |
| Metabolite-informed bands | 10 biologically defined bands |

---

## 2. Metabolite Library

### 2.1 Compounds by Category

| Category | Count | Compounds |
|---|---:|---|
| **Amino acids** | 16 | Alanine, Arginine, Aspartic acid, Cysteine, Glutamic acid, Glycine, Histidine, Isoleucine, Leucine, Phenylalanine, Proline, Serine, Threonine, Tryptophan, Tyrosine, Valine |
| **Nucleobases/nucleosides** | 10 | Adenine, Adenosine, Guanine, Hypoxanthine, Inosine, Pseudouridine, Purine, Pyrimidine, Uracil, Xanthine |
| **Organic acids** | 10 | Acrylic acid, Ascorbic acid, Benzoic acid, cis/trans-Aconitic acid, Hippuric acid, Maleic acid, Malic acid, Uric acid, Xylonic acid |
| **Lipids/fatty acids** | 4 | Cholesterol, Palmitic acid, Sphinganine, Stearic acid |
| **Sugars** | 5 | Fucose, Galactosamine, Glucose, Glycogen, Xylose |
| **Gut microbiome metabolites** | 3 | Hippuric acid, Trimethylamine-N-oxide (TMAO), Benzoic acid |
| **Oxidative stress markers** | 2 | 8-Hydroxy-2-deoxyguanosine (8-OHdG), Ascorbic acid |
| **Other** | 12 | Betaine, Choline, Creatine, Creatinine, Kynurenine, NADH, O-Acetylcarnitine, Spermidine, 4-Pyridoxic acid, etc. |

### 2.2 Measurement Protocol

- **Instrument:** Thermo Raman Spectrometer
- **Excitation wavelength:** 785 nm (assumed, standard for biological Raman)
- **Spectral range:** 50–3300 cm⁻¹ (1686 data points per spectrum)
- **Analysis range:** 400–1800 cm⁻¹ (fingerprint region)
- **Sample preparation:** Pure standards dissolved in specified solvents (DI water, DMSO, EtOH, HCl, NaOH as needed)
- **Replicates:** 5–6 per compound, averaged for analysis

---

## 3. Spectral Similarity to Urine SERS

### 3.1 Top 10 Metabolites by Pearson Correlation

| Rank | Metabolite | r | Biological Significance |
|---|---|---:|---|
| 1 | **2-Phenylacetamide** | 0.776 | Phenylalanine catabolite, gut microbiome |
| 2 | **Hippuric acid** | 0.764 | Major urinary organic acid, gut microbiome metabolism |
| 3 | **Stearic acid** | 0.694 | Saturated fatty acid (C18:0) |
| 4 | **Phenylalanine** | 0.661 | Essential amino acid, dominant 1003 cm⁻¹ peak |
| 5 | **Palmitic acid** | 0.639 | Saturated fatty acid (C16:0) |
| 6 | **Adenosine** | 0.627 | Nucleoside, energy metabolism |
| 7 | **TMAO** | 0.620 | Gut microbiome metabolite, cardiovascular risk marker |
| 8 | **Benzoic acid** | 0.600 | Aromatic metabolism, hippuric acid precursor |
| 9 | **Cholesterol** | 0.590 | Lipid metabolism |
| 10 | **Ascorbic acid** | 0.580 | Vitamin C, antioxidant |

![Figure 13: Top 10 Metabolites by Spectral Similarity](figures/fig13_top10_correlation.png)
*Figure 13. Top 10 metabolites ranked by Pearson correlation with mean urine SERS spectrum. (Top) Bar chart showing correlation coefficients. (Bottom) Spectral overlay of top 10 metabolites (colored) with mean urine SERS spectrum (black). 2-Phenylacetamide and Hippuric acid show the highest similarity, indicating they are major contributors to the urine SERS signal.*

**Key Finding:** The top contributors to the urine SERS signal are phenylalanine pathway metabolites (2-Phenylacetamide, Phenylalanine, Benzoic acid, Hippuric acid) and lipids (Stearic/Palmitic acid). This is consistent with SERS preferentially enhancing aromatic and thiol-containing molecules at the gold nanoparticle surface.

---

## 4. Peak Assignment Table

### 4.1 SERS Peak → Metabolite Mapping

17 peaks were detected in the mean urine SERS spectrum. For each peak, metabolites with Raman peaks within ±10 cm⁻¹ were identified.

| SERS Peak (cm⁻¹) | Vibration Mode | Key Contributing Metabolites |
|---|---|---|
| **618** | C-S stretch | Cysteine, Adenine, Betaine, Cholesterol, Arginine |
| **683** | C-S stretch / ring | Creatinine, Guanine, Purine, Hippuric acid, Tyrosine |
| **724** | C-N stretch / Adenine ring breathing | Adenine, Hypoxanthine, Hippuric acid, Benzoic acid |
| **795** | Ring breathing | Hippuric acid, Kynurenine, Uric acid |
| **849** | Tyrosine Fermi resonance | Tyrosine, Tryptophan, Creatinine, Cysteine |
| **895** | C-C stretch | Hippuric acid, TMAO, Uric acid, Cysteine |
| **934** | C-C stretch (protein) | Multiple (>20 metabolites) |
| **999** | **Phenylalanine ring breathing** | **Phenylalanine**, 2-Phenylacetamide, Hippuric acid, Benzoic acid |
| **1148** | C-N stretch / C-O-C | Glycogen, Glucose, Xylose, Cysteine |
| **1231** | Amide III | Tryptophan, Taurine, Kynurenine, Threonine, Isoleucine |
| **1293** | Amide III / CH₂ twist | O-Acetylcarnitine, Acrylic acid, Palmitic acid |
| **1352** | CH deformation / Trp | Adenine, Guanine, Tryptophan |
| **1449** | CH₂ deformation | Nearly all metabolites (non-specific) |
| **1597** | C=C stretch / Purine ring | Adenine, Kynurenine, Tyrosine, Phenylalanine |
| **1651** | Amide I (C=O stretch) | Maleic acid, Glycogen, Kynurenine, Stearic acid |

![Figure 11: Metabolite Spectra Overlaid on Major SERS Bands](figures/fig11_metabolite_sers_overlay.png)
*Figure 11. Key metabolite Raman spectra overlaid on the mean urine SERS spectrum (black line). Colored bands indicate the 6 major SERS peak regions. Metabolites were selected based on highest spectral overlap with each band.*

![Figure 12: Peak Assignment Heatmap](figures/fig12_peak_assignment_heatmap.png)
*Figure 12. Heatmap showing metabolite (rows) × SERS peak region (columns) assignments. Color intensity = number of metabolite peaks in that region. The 1200-1300 cm⁻¹ (Amide III) region has the most diverse metabolite contributions, while 1400-1500 cm⁻¹ (CH₂) is universally present across all metabolites.*

---

## 5. Cancer vs Control — Metabolite-Informed Band Analysis

### 5.1 Band Definitions

Using the metabolite peak assignment data, we defined 10 biologically meaningful spectral bands:

| Band Name | Range (cm⁻¹) | Assigned Metabolites |
|---|---|---|
| Phe ring | 999–1010 | Phenylalanine, 2-Phenylacetamide, Hippuric acid |
| Adenine | 720–730 | Adenine, Hypoxanthine, nucleotide metabolism |
| C-S stretch | 615–625 | Cysteine, thiol compounds |
| Tyr/Trp | 845–855 | Tyrosine, Tryptophan, Cysteine |
| Amide III | 1225–1300 | Tryptophan, Taurine, Kynurenine |
| CH₂ def | 1440–1460 | Lipids, fatty acids (non-specific) |
| Purine/C=C | 1590–1605 | Adenine, Kynurenine, Tyrosine |
| Amide I | 1645–1660 | Maleic acid, Kynurenine, Glycogen |
| Creatinine | 680–690 | Creatinine, Guanine, Purine |
| Hippuric | 790–800 | Hippuric acid, Kynurenine |

### 5.2 Statistical Comparison (Cancer n=870 vs Control n=400)

| Band | Cohen's d | Direction | p-value | Significance |
|---|---:|---|---|---|
| **Creatinine (680-690)** | **-1.26** | Cancer DOWN | 2.9e-83 | *** |
| **Adenine (720-730)** | **-1.13** | Cancer DOWN | 1.4e-72 | *** |
| **Hippuric (790-800)** | **+1.09** | Cancer UP | 4.9e-52 | *** |
| **Purine/C=C (1590-1605)** | **-0.93** | Cancer DOWN | 1.9e-43 | *** |
| **C-S stretch (615-625)** | **+0.76** | Cancer UP | 1.1e-28 | *** |
| **Tyr/Trp (845-855)** | **-0.43** | Cancer DOWN | 4.1e-11 | *** |
| **CH₂ def (1440-1460)** | **-0.38** | Cancer DOWN | 7.6e-09 | *** |
| Amide III (1225-1300) | -0.09 | ns | 0.13 | ns |
| Phe ring (999-1010) | -0.05 | ns | 0.37 | ns |
| Amide I (1645-1660) | +0.01 | ns | 0.86 | ns |

> 7 out of 10 bands show statistically significant differences (p < 0.05). Largest effect size: Creatinine band (d = -1.26).

![Figure 14: Cancer vs Control — Metabolite Band Intensities](figures/fig14_metabolite_band_cancer_vs_control.png)
*Figure 14. Boxplot comparison of 10 metabolite-informed SERS bands between cancer patients (red, n=870) and non-cancer controls (blue, n=400). Cohen's d and p-values shown per band. Creatinine (680-690 cm⁻¹), Adenine (720-730 cm⁻¹), and Hippuric acid (790-800 cm⁻¹) show the largest effect sizes.*

### 5.3 Biological Interpretation

**Bands DECREASED in cancer:**
- **Creatinine (680-690 cm⁻¹), d=-1.26:** Largest effect. Reduced creatinine-related signal may reflect altered renal function or muscle mass loss (cachexia) in cancer patients.
- **Adenine (720-730 cm⁻¹), d=-1.13:** Nucleotide metabolism disruption. Increased purine consumption by rapidly proliferating tumor cells may reduce urinary excretion.
- **Purine/C=C (1590-1605 cm⁻¹), d=-0.93:** Consistent with adenine decrease — reflects overall purine pathway alteration.
- **Tyr/Trp (845-855 cm⁻¹), d=-0.43:** Tryptophan depletion via IDO/TDO enzyme upregulation in tumor microenvironment (immune evasion mechanism).

**Bands INCREASED in cancer:**
- **Hippuric acid (790-800 cm⁻¹), d=+1.09:** Gut microbiome dysbiosis in cancer patients leads to altered hippuric acid production. Reported as a candidate biomarker for colorectal and bladder cancer.
- **C-S stretch (615-625 cm⁻¹), d=+0.76:** Elevated thiol compounds (cysteine, glutathione) reflecting oxidative stress response in cancer.

---

## 6. Disease-Group-Specific Metabolite Profiles

### 6.1 Z-Score Heatmap

![Figure 15: Metabolite Band Heatmap by Disease Group](figures/fig15_metabolite_band_heatmap_by_group.png)
*Figure 15. Z-score normalized band intensities per disease group. Red = elevated relative to mean, Blue = reduced. Bottom annotations show assigned metabolites per band.*

### 6.2 Cancer-Type-Specific Signatures

| Cancer Type | Elevated Bands | Reduced Bands | Interpretation |
|---|---|---|---|
| **CRC (Colorectal)** | Hippuric (+2.32) | Amide III (-2.30), CH₂ (-1.86), Creatinine (-1.66) | Most pronounced gut microbiome alteration |
| **PAN (Pancreatic)** | Amide III (+1.18), Hippuric (+0.85) | Phe ring (-2.63), Adenine (-1.35) | Severe cachexia-related amino acid/nucleotide depletion |
| **LUN (Lung)** | C-S stretch (+1.35), Hippuric (+0.67) | Tyr/Trp (-1.53), Purine (-1.87), Creatinine (-1.33) | Oxidative stress + aromatic amino acid depletion |
| **PRO (Prostate)** | Tyr/Trp (+1.10), Amide I (+1.36) | C-S stretch (-0.79), Hippuric (-0.72) | Unique protein-dominant profile, opposite to other cancers |
| **OVA (Ovarian)** | Amide III (+1.05), Amide I (+1.42) | C-S stretch (-0.66), Hippuric (-0.66) | Protein/amide band signature |
| **BRE (Breast)** | Tyr/Trp (+0.79), Amide I (+1.24) | C-S stretch (-1.02), Hippuric (-0.62) | Similar to PRO/OVA protein pattern |

**Key Observation:** Cancer types cluster into two metabolic signature groups:
1. **Gut-microbiome dominant** (CRC, PAN, LUN): Hippuric acid ↑, Creatinine ↓, Adenine ↓
2. **Protein-metabolism dominant** (PRO, BRE, OVA): Amide I ↑, Tyr/Trp ↑, C-S stretch ↓

---

## 7. Difference Spectra with Metabolite Annotations

### 7.1 Overall Cancer vs Control

![Figure 16: Difference Spectrum with Metabolite Assignments](figures/fig16_difference_spectrum_metabolites.png)
*Figure 16. (A) Mean SERS spectra overlay: cancer (red) vs control (blue). Shaded areas show regions where cancer signal exceeds control (red) or vice versa (blue). (B) Difference spectrum (Cancer − Control) annotated with metabolite peak assignments. Red annotations = cancer-elevated peaks, blue = cancer-reduced peaks.*

### 7.2 Per-Cancer-Type Difference Spectra

![Figure 17: Per-Cancer-Type Difference Spectra vs Normal](figures/fig17_per_cancer_difference.png)
*Figure 17. Difference spectra for each cancer type relative to Normal (NOR). Green bands = metabolite-informed analysis regions. Key patterns: Prostate Ca shows strong positive difference at 1600 cm⁻¹, Colorectal Ca has a dominant peak at 1350 cm⁻¹, Pancreatic Ca shows overall negative differences reflecting cachexia.*

---

## 8. Cancer-Relevant Metabolite Summary

| Metabolite | Cancer Relevance | SERS Band Contribution | Observed Change |
|---|---|---|---|
| **Hippuric acid** | Gut dysbiosis marker; CRC, bladder Ca biomarker candidate | 724, 795, 999, 1597 cm⁻¹ | **↑ in cancer** (d=+1.09) |
| **Creatinine** | Renal function / muscle mass marker | 683 cm⁻¹ | **↓ in cancer** (d=-1.26) |
| **Adenine** | Nucleotide metabolism, cell proliferation | 618, 724, 1352, 1597 cm⁻¹ | **↓ in cancer** (d=-1.13) |
| **Kynurenine** | Trp catabolism via IDO, immune evasion | 1597, 1651 cm⁻¹ | ↓ Purine/C=C band |
| **Cysteine** | Thiol/oxidative stress | 615-625 cm⁻¹ | **↑ in cancer** (d=+0.76) |
| **TMAO** | Gut microbiome, cardiovascular/cancer risk | 750, 1449 cm⁻¹ | Mixed |
| **Spermidine** | Polyamine pathway, tumor proliferation | 1231 cm⁻¹ | ns |
| **8-OHdG** | Oxidative DNA damage marker | 618, 1449 cm⁻¹ | C-S band ↑ overlap |
| **N-Acetylneuraminic acid** | Sialic acid, cancer cell surface glycan changes | 1231, 1449 cm⁻¹ | ns |

---

## 9. Output Files

```
metabolite_profiling/
├── analyze_metabolites.py                        # Analysis script
├── METABOLITE_PROFILING_REPORT.md                # This report
├── data/
│   ├── metabolite_peak_assignments.csv           # 527 peak-metabolite matches
│   ├── metabolite_sers_correlation.csv           # 73 metabolite spectral correlations
│   ├── metabolite_band_by_group.csv              # Per-group band intensity statistics
│   └── metabolite_band_statistics.csv            # Cancer vs control band t-tests
└── figures/
    ├── fig11_metabolite_sers_overlay.png          # Metabolite spectra overlay on SERS
    ├── fig12_peak_assignment_heatmap.png          # Peak assignment heatmap
    ├── fig13_top10_correlation.png                # Top 10 spectral similarity
    ├── fig14_metabolite_band_cancer_vs_control.png  # Cancer/control band comparison
    ├── fig15_metabolite_band_heatmap_by_group.png   # Disease group band heatmap
    ├── fig16_difference_spectrum_metabolites.png     # Annotated difference spectrum
    └── fig17_per_cancer_difference.png              # Per-cancer difference spectra
```

Raw Thermo Raman data: `data/Metabolite analysis_Thermo/`

---

## 10. Implications for Cancer Detection Model

1. **Feature Selection:** The 7 significant metabolite bands (Creatinine, Adenine, Hippuric, Purine, C-S, Tyr/Trp, CH₂) should be prioritized as biologically interpretable features in the classification model.
2. **Cancer Subtype Discrimination:** The two metabolic signature clusters (gut-microbiome vs protein-dominant) suggest that a hierarchical classifier may benefit from cancer-type-specific band weighting.
3. **Confounders:** Creatinine band intensity is strongly affected by renal function — clinical creatinine levels should be included as a covariate in multivariate models.
4. **Validation:** The hippuric acid increase in CRC is consistent with published literature on gut dysbiosis in colorectal cancer patients.

---

*Report generated from `analyze_metabolites.py` output. All spectral comparisons use per-patient averaged SERS spectra (n=1,270 patients: 870 cancer, 400 control).*
