# SERS-AI Software Product IFU Linkage

| Field | Value |
| --- | --- |
| SharePoint area | `03_SW_Product` |
| Version | v0.1 |
| Last updated | 2026-05-15 |
| IFU source | `docs/clinical_use/USER_MANUAL.md` |

## 1. Purpose

This document links the SERS Clinical App user-facing screens and features to the current IFU/User Manual content. It is intended to keep product documentation, usability evaluation, and software behavior aligned.

## 2. Feature-To-IFU Mapping

| App feature / screen | User-facing purpose | IFU section | Required user warning / interpretation |
| --- | --- | --- | --- |
| Login | authenticate authorized user | 5.1 | access limited to authorized users |
| Patient information | enter patient/session context | 5.2 | sex may affect cancer type display constraints |
| Spectrum upload | upload CSV spectra for analysis | 5.3 | CSV must contain wavenumber/intensity columns; replicate files recommended |
| QC review | confirm spectrum suitability | 5.4 | QC failure requires remeasurement or review before interpretation |
| Result review | display SSI score and type-risk context | 5.5, 6 | result is not a final diagnosis |
| Cancer type probabilities | show relative model classification scores | 5.5, 6 | probabilities are model-relative type scores, not definitive cancer probabilities |
| Report generation | create clinical-style report | 5.6 | report supports clinician review only |
| Audit log | record usage events | 9 | audit log is usage traceability, not clinical interpretation |

## 3. IFU-Controlled Language

Use the following interpretation language consistently across app, report, and SharePoint documents.

| Screen concept | Preferred wording |
| --- | --- |
| high screening score | additional confirmation recommended |
| low screening score | below current confirmation threshold |
| cancer type output | relative cancer type risk / model classification score |
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
| `docs/clinical_use/USER_MANUAL.md` | IFU/User Manual source |
| `docs/clinical_use/USABILITY_EVALUATION_PROTOCOL.md` | usability test protocol |
| `docs/clinical_use/USABILITY_EVALUATION_FORMS.md` | evaluator forms |
| `docs/clinical_use/TRACEABILITY_MATRIX.md` | requirement and workflow traceability |

## 6. Version Rule

Update this file whenever:

- app UI flow changes
- report wording changes
- IFU/User Manual changes
- result interpretation language changes
- screenshots are replaced with a new approved version
