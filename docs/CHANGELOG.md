# Changelog

All notable changes to the SERS Analysis Pipeline will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
-

### Changed
-

### Fixed
-

---

## [0.5.0] - 2026-03-23 (Phase V)

### Added
- Fixed wavenumber grid standardization: 935 spectral features on 402-2198 cm-1 range (previously 933 features with variable grid)
- 6-cancer restructured model: PRO, OVA, LUN, CRC, PAN (merged CPAN+SPAN), BLC
- 7-cancer variant experiment (PRO, OVA, LUN, CRC, CPAN, SPAN, BLC) for comparison
- 8-cancer full-spectrum experiment with BRE included (8,205 spectra)
- Held-out test evaluation (60/20/20 split, 5 repeats) for 6-cancer model
- Grid analysis dashboard (`grid_analysis_dashboard.html`)

### Changed
- Wavenumber grid now enforced at 935 points (was 933), covering 402-2198 cm-1
- CPAN and SPAN merged into PAN for the primary 6-cancer model to reduce class fragmentation
- BRE dropped from primary model due to small cohort size (n=30)
- Production model retrained on fixed grid

### Results (5-fold CV, Logistic Regression)
- 6-cancer (7,990 spectra): Cancer Screening AUC 0.972, Cancer Type ID F1 0.869
- 7-cancer (8,055 spectra): Cancer Screening AUC 0.980, Cancer Type ID F1 0.844
- 8-cancer (8,205 spectra): Cancer Screening AUC 0.980, Cancer Type ID F1 0.813
- Held-out test (6-cancer, single repeat): Det AUC 0.969, Id F1 0.888

---

## [0.4.0] - 2026-03-23 (Phase T)

### Added
- BLC (bladder cancer) cohort: 299 samples from 충북대학교병원, protocol SMCXD06, retrospective
- 8-class model with BLC: PRO, BRE, OVA, LUN, CRC, CPAN, SPAN, BLC
- Clinical metadata for BLC (`data/clinical_data/11. 방광암/SMCXD06_방광암.xlsm`)
- Dashboard v5.0 with 6-cancer support and fixed grid metrics
- Data Explorer with Supabase backend (`data_explorer.html`)
- Equipment QC dashboard (`Equipment_QC_Dashboard.html`)
- Metabolite profiling dashboards: `Thermo_Metabolite_Dashboard.html`
- Pancreatic multi-institute study dashboard (`Pancreatic_MultiInstitute_Study.html`)
- Clinical analysis report (`SMCXD04_Analysis_Report_KR.html`)
- Model test dashboard (`model_test.html`)

### Results (Phase T: 8-class, medoid, 1,641 samples)
- Cancer Screening AUC: 0.969
- Cancer Type ID F1 macro: 0.738
- BLC per-class: Sensitivity 96.7%, BLC AUC 0.998 (best per-class performance)

---

## [0.3.0] - 2026-03-19 (Phases Q-R)

### Added
- Sex-based biological constraint: zero out impossible cancer types based on patient sex (males cannot have OVA, females cannot have PRO)
- `apply_sex_constraint()` function in `models/run_train_val_test.py`
- Sex-based masking in production predictor `scripts/sers_predict.py`
- Baseline correction comparison: Rolling Minimum vs ALS (2 parameter sets)

### Changed
- Production predictor now applies sex constraint automatically when `--sex` is provided
- Sex constraint supersedes ensemble as primary improvement method (simpler, larger gain)

### Results
- **Phase Q (sex constraint, held-out test, 6,200 spectra, 5 cancers):**
  - SERS + sex constraint: Det AUC 0.977, Id F1 macro 0.892 (+4.0pp vs unconstrained 0.852)
  - Fusion + sex constraint: Det AUC 0.986, Id F1 macro 0.884 (+0.7pp)
  - PRO F1: 0.85 -> 0.93 (+8pp), OVA F1: 0.82 -> 0.89 (+7pp)
  - PRO->OVA confusion: 5.2% -> 0%, OVA->PRO confusion: 7.7% -> 0%
- **Phase R (baseline correction validation):**
  - Rolling Minimum: Det AUC 0.977, Id F1 0.892, preprocessing time 8.4s
  - ALS (lam=1e6, p=0.01): Det AUC 0.976, Id F1 0.882, preprocessing time 32.8s
  - ALS (lam=1e7, p=0.001): Det AUC 0.976, Id F1 0.885, preprocessing time 43.8s
  - Conclusion: Rolling Minimum confirmed optimal (best F1, 4x faster)

---

## [0.2.0] - 2026-03-18 (Phases L-P)

### Added
- Normalization ablation experiment (Phase L): tested SNV, None, MinMax, L2
- Confounding analysis (Phase M): demographics-only vs SERS vs combined
- Multimodal fusion pipeline (Phase N): SERS + age/sex/BMI early and late fusion
- Per-cancer sensitivity analysis (Phase O) with LUN stage breakdown
- Held-out test validation (Phase P): 60/20/20 train/val/test, 5 random repeats
- Production inference pipeline: `models/build_production_model.py` and `scripts/sers_predict.py`
- Production model artifacts in `models/production/` (~190 KB total)
- Metabolite profiling: 73 urinary metabolite standards, 527 peak-metabolite matches
- 10 metabolite-informed SERS bands with cancer vs control analysis (7 of 10 significant)
- Clinical reports v2.0 (bilingual Korean/English)
- Dashboard v4.4 with executive/technical toggle

### Changed
- Ensemble experiment (Phase K): 80% LR + 20% ResNet18 blend confirmed as marginal improvement (+0.3pp F1)
- ResNet18 optimization (Phase J): capacity, regularization, loss reweighting, multi-channel input all failed to close gap with LR
- Hyperparameter retuning (Phase H): did not change model ranking
- Learning curve analysis (Phase I): LR outperforms DL at all data fractions

### Results
- **Phase L (normalization ablation, 6,200 spectra, 5 cancers):**
  - LR robust to normalization (Id F1: 0.868-0.879 across all methods)
  - ResNet18 highly sensitive (Id F1: 0.584-0.720)
  - Conclusion: linear separability is intrinsic to data, not caused by SNV
- **Phase M (confounding analysis):**
  - Demographics-only (age+sex+BMI): Det AUC 0.774, Id F1 0.309
  - SERS-only: Det AUC 0.987, Id F1 0.861
  - SERS + age+sex: Det AUC 0.989, Id F1 0.882
  - Age-matched subgroup: SERS maintained (0.987 -> 0.989), demographics dropped (0.766 -> 0.693)
  - Conclusion: SERS detects cancer metabolites, not demographics
- **Phase N (multimodal fusion, 5,050 spectra with matched clinical data):**
  - Early fusion (SERS + age/sex/BMI): Det AUC 0.988, Id F1 0.877
  - Late fusion (90% SERS / 10% clinical): Det AUC 0.989, Id F1 0.864
  - Lab values add minimal value on top of SERS (+0.2% AUC)
- **Phase O (per-cancer sensitivity):**
  - Detection: CRC 98.1% > CPAN 96.9% > LUN 95.7% > PRO 83.8% > OVA 79.7%
  - LUN early-stage (I+II): 99.4% sensitivity (n=136 patients)
  - Top discriminating metabolite bands: Creatinine (d=-1.26), Adenine (d=-1.13), Hippuric acid (d=+1.09)
- **Phase P (held-out test, 6,200 spectra, 5 cancers):**
  - SERS-only: Det AUC 0.977, Id F1 0.852 (CV was 0.981/0.872 -- no inflation)
  - Fusion: Det AUC 0.986, Id F1 0.877 (matches CV exactly)
  - Low variance across 5 splits (std 0.004-0.026)
- **Production model deployed (2026-03-18):**
  - SERS-only + Fusion models in `models/production/`
  - Auto model selection, graceful degradation, JSON output mode
  - No GPU or PyTorch dependency

### Technical Details
- Metabolite profiling: two metabolic clusters identified
  - Gut-microbiome dominant (CRC, CPAN, LUN): Hippuric acid up, Creatinine down
  - Protein-metabolism dominant (PRO, BRE, OVA): Amide I up, Tyr/Trp up
- Operating points defined: screening (95% sens / 90.4% spec), balanced (Youden's J), confirmatory (95% spec / 91.5% sens)

---

## [0.1.0] - 2025-01-13

### Added
- Initial release of SERS spectral analysis pipeline
- Preprocessing module with Savitzky-Golay smoothing, baseline correction, SNV normalization
- Quality control with SNR filtering and replicate correlation analysis
- Medoid-based replicate aggregation
- Configuration management with frozen dataclasses
- YAML-based user configuration
- Command-line interface (`sers run`, `sers validate`)
- Docker support for containerized deployment
- Comprehensive test suite with pytest

### Technical Details
- Python 3.10+ required
- Key dependencies: numpy, pandas, scipy, scikit-learn, pyyaml
- Supports environment variable configuration for deployment paths

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.5.0 | 2026-03-23 | Phase V: Fixed wavenumber grid (935 pts), 6-cancer restructure |
| 0.4.0 | 2026-03-23 | Phase T: BLC added (299 samples), 8-class model, dashboards |
| 0.3.0 | 2026-03-19 | Phases Q-R: Sex constraint (+4pp F1), baseline correction validated |
| 0.2.0 | 2026-03-18 | Phases L-P: Normalization ablation, confounding analysis, fusion, production model |
| 0.1.0 | 2025-01-13 | Initial release |
