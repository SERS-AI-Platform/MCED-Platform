# SERS-AI Software Product IFU Linkage

| Field | Value |
| --- | --- |
| SharePoint area | `03_SW_Product` |
| Version | v0.1 |
| Last updated | 2026-07-15 |
| IFU source | `docs/clinical/USER_MANUAL.md` |

## 1. Purpose

This document links the SERS Clinical App user-facing screens and features to the current IFU/User Manual content. It is intended to keep product documentation, usability evaluation, and software behavior aligned.

## 2. Feature-To-IFU Mapping

| App feature / screen | User-facing purpose | IFU section | Required user warning / interpretation |
| --- | --- | --- | --- |
| Login | authenticate authorized user | 5.1 | access limited to authorized users |
| Patient information | enter patient/session context | 5.2 | sex may affect cancer type display constraints |
| Spectrum upload | upload CSV spectra for analysis | 5.3 | CSV must contain wavenumber/intensity columns; replicate files recommended |
| QC review | confirm spectrum suitability | 5.4 | QC failure requires remeasurement or review before interpretation |
| Result review | display valid-replicate count, threshold-exceeding replicate count, SSI score, and type-pattern context | 5.5, 6 | final screening wording follows fixed `n_detected ≥ 3`; result is not a final diagnosis |
| Cancer type scores | show top estimated pattern and relative model classification scores only when additional confirmation is recommended | 5.5, 6 | scores are relative model outputs, not definitive cancer probabilities |
| Report generation | create clinical-style report | 5.6 | report supports clinician review only |
| Audit log | record usage events | 9 | audit log is usage traceability, not clinical interpretation |

## 3. IFU-Controlled Language

Use the following interpretation language consistently across app, report, and SharePoint documents.

| Screen concept | Preferred wording |
| --- | --- |
| at least 3 of 5 uploaded measurements are QC-valid replicates above the internal rule | further evaluation recommended |
| fewer than 3 of 5 uploaded measurements are QC-valid replicates above the internal rule | below decision threshold |
| mean SSI score | patient-level decision metric: <1.0, 1.0–4.0 inclusive, or >4.0 |
| cancer type output | closest cancer type pattern / relative model classification score |
| clinical use | supports clinician review and additional-test decision making |
| limitation | not a standalone diagnostic result |

Avoid:

- strong positive / weak positive
- definitive cancer diagnosis
- cancer-free / normal confirmed
- guaranteed detection

## 4. Required Product Screenshots

| Screenshot | Status | Notes |
| --- | --- | --- |
| Login | TBD | generate from clinical app or use approved mockup |
| Patient registration | TBD | generate from clinical app or use approved mockup |
| Spectrum upload | TBD | generate from clinical app or use approved mockup |
| QC review | TBD | generate from clinical app or use approved mockup |
| Result screen | available as clinical report/dashboard mockup | use only de-identified/demo data |
| Clinical report | available in `solum-dashboard/clinical_reports/screenshots/` | use latest approved version |
| Dashboard clinical section | available in `solum-dashboard/clinical_reports/screenshots/` | use latest approved version |

## 5. Related Usability Documents

| Document | Role |
| --- | --- |
| `docs/clinical/USER_MANUAL.md` | IFU/User Manual source |
| `docs/clinical/USABILITY_EVALUATION_PROTOCOL.md` | usability test protocol |
| `docs/clinical/USABILITY_EVALUATION_FORMS.md` | evaluator forms |
| `docs/clinical/TRACEABILITY_MATRIX.md` | requirement and workflow traceability |

## 6. Version Rule

Update this file whenever:

- app UI flow changes
- report wording changes
- IFU/User Manual changes
- result interpretation language changes
- screenshots are replaced with a new approved version

## 7. 2026-07-15 Alignment Update

- QC validity now requires at least 3 of 5 spectra to pass and no low-intensity, spike-noise, or saturation failure.
- A critical QC failure stops inference and withholds SSI, risk-band, threshold-exceeding replicate count, and Cancer Type ID outputs from screen, PDF, and CSV reports.
- The patient-level screening result uses fixed `n_detected ≥ 3`; the required threshold-exceeding count remains three when `N_valid` is 5, 4, or 3.
- The patient mean SSI determines the screening action: below 1.0, 1.0 through 4.0 inclusive, or above 4.0.
- Replicate-level cancer-signal counts are retained as reference information and do not determine newly generated results.
- Internal mean probability and θ=0.60 are not shown on the user screen or clinical report. Cancer Type ID is shown only for an additional-confirmation recommendation and remains relative classification support.
- Internal performance evidence is the fixed split of 1,628 participants: Train 976, Validation 324, Test 328. Test screening performance is sensitivity 94.61%, specificity 96.55%, and AUROC 0.9925; Test Cancer Type ID (TOO) macro-F1 is 0.9593 on 241 known cancer cases. This evaluation averages patient replicates at the feature level before inference and is not a direct end-to-end evaluation of the deployed per-replicate fixed-three pipeline.
- Cancer Screening metrics may be affected by hospital-source confounding between cancer and non-cancer cohorts and do not establish cross-hospital generalization; independent external clinical validation remains required.
- Updated source documents: `docs/architecture/clinical_webapp_workflow_current.md`, `docs/clinical/USER_MANUAL.md`, `docs/clinical/USABILITY_EVALUATION_PROTOCOL.md`, and `docs/clinical/TRACEABILITY_MATRIX.md`.
