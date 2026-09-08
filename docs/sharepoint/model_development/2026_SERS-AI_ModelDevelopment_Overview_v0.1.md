# SERS-AI Model Development Overview

| Field | Value |
| --- | --- |
| SharePoint area | `01_Model_Development` |
| Document role | model structure and approach explanation |
| Version | v0.1 |
| Last updated | 2026-05-15 |
| Rule | concept-level only; no rule-code explanation |

## 1. Purpose

This document explains the SERS-AI model development approach at a review-ready concept level. It covers the diagnostic model structure, feature definitions, preprocessing approach, model evolution, and Git references needed to trace the current implementation.

The model objective is to estimate:

1. cancer vs non-cancer probability
2. cancer type risk among supported cancer classes

The project has evaluated both spectral-only and fusion approaches. The current production-oriented setting supports SERS-only prediction and SERS + clinical feature fusion.

## 2. Model Structure

The diagnostic logic is organized as a two-stage model.

```text
Raw spectrum / patient spectra
        |
        v
QC + preprocessing
        |
        v
Spectral feature vector
        |
        +-----------------------------+
        |                             |
        v                             v
Stage 1: cancer detection       Optional clinical features
P(cancer)                       age, sex, BMI
        |                             |
        +-------------+---------------+
                      |
                      v
Stage 2: cancer type risk
PRO / LUN / CRC / CPAN / OVA / etc.
```

Conceptually, Stage 1 is a binary screening task:

```text
f1(x) -> P(cancer | spectrum)
```

Stage 2 is a conditional type-identification task:

```text
f2(x) -> P(cancer_type | cancer, spectrum)
```

For fusion models, clinical metadata is added as a small structured feature vector:

```text
z = [spectral_features, age, sex_numeric, BMI]
```

The fusion model is not treated as a replacement for the SERS signal. It is used as an auxiliary input to test whether basic clinical metadata improves detection or type identification.

## 3. Feature Definitions

| Feature group | Definition | Conceptual role |
| --- | --- | --- |
| SERS spectral features | intensity values aligned to a common Raman shift grid | primary diagnostic signal |
| Wavenumber grid | standardized Raman shift axis after trimming/resampling | enables sample-to-sample comparison |
| First/second derivative views | slope/curvature views of the spectrum in experimental models | emphasize peak shape and local spectral changes |
| Peak features | selected peak area, height, width, shift, and peak ratios in stacking experiments | interpretable metabolite-band summary |
| Clinical features | age, sex, BMI | optional fusion inputs |
| Binary label | cancer vs non-cancer | Stage 1 target |
| Type label | cancer class label such as PRO, LUN, CRC, CPAN, OVA | Stage 2 target |

Production manifest reference:

| Item | Current production value |
| --- | --- |
| spectral feature count | 933 |
| production cancer types | PRO, LUN, CRC, CPAN, OVA |
| non-cancer groups | NOR, DIA, HBP, H.D. |
| fusion clinical features | age, sex_numeric, BMI |

## 4. Preprocessing Approach

The preprocessing pipeline converts raw SERS spectra into comparable model-ready features.

```text
raw intensity
   -> trim fingerprint region
   -> smooth high-frequency noise
   -> baseline correction
   -> normalize intensity scale
   -> resample to common grid
   -> model feature vector
```

Current production preprocessing parameters:

| Step | Current setting | Concept |
| --- | --- | --- |
| trim | 400-2200 cm^-1 | focus on fingerprint region |
| smoothing | Savitzky-Golay style smoothing, window 11, polynomial 3 | reduce noise while preserving peak shape |
| baseline correction | rolling-minimum style baseline window 101 | remove fluorescence/background trend |
| normalization | SNV | reduce scale/offset variation from measurement conditions |
| common grid | fixed Raman shift grid | make spectra comparable across samples |

The guiding principle is to reduce measurement artifacts while preserving relative metabolite-pattern information.

## 5. Model Development History

| Phase / approach | Conceptual change | Outcome / decision |
| --- | --- | --- |
| Initial medoid baseline | patient-level representative spectrum approach | useful baseline but limited performance |
| All-spectra training | use all available spectra instead of only medoids | improved type identification |
| Classical model benchmark | compare LR, XGBoost, RF, CNN/ResNet-style methods | logistic regression became strongest practical baseline |
| ResNet18-1D | deep spectral encoder with two-stage heads | did not consistently outperform classical models |
| LR + ResNet ensemble | blend classical and deep outputs | marginal improvement only |
| SERS + clinical fusion | add age/sex/BMI to spectral features | improved held-out detection AUC and type F1 |
| Sex constraint | concept-level biological constraint on impossible type confusion | improved type ID F1; transparent and auditable |
| Baseline correction comparison | compare Rolling Minimum vs ALS | Rolling Minimum retained due to best speed/performance balance |
| 7-cancer extension | broaden class coverage and test fusion | fusion achieved strongest 7-cancer type ID F1 in current tracking |

## 6. Current KPI References

| Setting | Detection AUC | Sensitivity | Specificity | Type ID metric |
| --- | ---: | ---: | ---: | ---: |
| 5-cancer SERS-only held-out | 0.977 +/- 0.005 | 0.926 +/- 0.004 | 0.929 +/- 0.019 | F1 0.852 +/- 0.026 |
| 5-cancer fusion held-out | 0.986 +/- 0.004 | 0.939 +/- 0.011 | 0.949 +/- 0.019 | F1 0.877 +/- 0.018 |
| 5-cancer SERS + sex constraint | 0.977 +/- 0.005 | 0.926 +/- 0.004 | 0.929 +/- 0.019 | F1 0.892 +/- 0.026 |
| 7-cancer fusion held-out | 0.979 +/- 0.005 | TBD | TBD | Type ID F1 0.923 +/- 0.022 |

Internal source documents:

- `docs/ml/EXPERIMENT_CONTEXT.md`
- `docs/ml/experiment.md`
- `docs/CHANGELOG.md`

## 7. What This Folder Should Not Contain

Do not store:

- raw spectra
- model binary artifacts
- training logs
- code-level rule descriptions
- temporary notebooks
- duplicate result exports
- implementation-level function explanations

Those belong in Git, local WSL storage, or controlled model artifact storage depending on file type.

## 8. Git Commit References

| Commit | Scope | Relevance |
| --- | --- | --- |
| `45bbc72` | `feat: initial project setup from desktop backup` | imported core model code, preprocessing library, experiment history, production model manifests, and model artifacts |
| `1cd527f` | `Merge OneDrive development materials` | merged later development materials; includes production inference update scope for `scripts/deployment/sers_predict.py` |
| `77b01cc` | `Add development asset governance guide` | added Git/SharePoint/WSL governance rules |
| `56f200b` | `Add SharePoint remap inventory` | added SharePoint remapping inventory used to place this folder |
| `cc3a998` | `Add SharePoint overview and model development docs` | created this SharePoint-facing `01_Model_Development` document set |

Git trace command used for this version:

```bash
git log --oneline --all -- models src scripts/deployment/sers_predict.py docs/ml/experiment.md docs/ml/EXPERIMENT_CONTEXT.md
```

## 9. Version Rule

For every SharePoint-facing revision:

1. update the version number
2. update the KPI/status tables if model results changed
3. add the relevant Git commit hash
4. update `VERSION_LOG.md`
