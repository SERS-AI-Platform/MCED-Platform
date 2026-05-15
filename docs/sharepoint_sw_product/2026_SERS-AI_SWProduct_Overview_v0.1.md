# SERS-AI Software Product Overview

| Field | Value |
| --- | --- |
| SharePoint area | `03_SW_Product` |
| Product | SERS Clinical App / SERS-AI clinical screening interface |
| Version | v0.1 |
| Last updated | 2026-05-15 |
| Rule | user/product-level description only; no code explanation |

## 1. Purpose

The SERS Clinical App connects the SERS-AI model pipeline to an operator-facing clinical workflow. It lets an authorized user register a patient/session, upload SERS spectrum CSV files, review QC status, view screening results, and generate a clinical-style report.

The app is intended to support clinical review. It does not provide a standalone final diagnosis.

## 2. Intended Users

| User | Main tasks |
| --- | --- |
| Laboratory technician | login, register patient/session, upload spectrum files, review QC pass/fail |
| Clinician | review SSI score, cancer type risk output, QC context, and generated report |
| Administrator | manage users, review audit trail, check application/model status |

## 3. UI Flow

```text
Login
  -> Patient information
  -> Spectrum upload
  -> QC review
  -> Result review
  -> Report generation
  -> Audit trail / administration
```

| UI step | User-facing purpose | Main user input/output |
| --- | --- | --- |
| Login / registration | authenticate user and role | username, password, role |
| Patient information | create analysis session | patient ID, age, sex, optional BMI |
| Spectrum upload | attach SERS measurement files | CSV spectrum files, recommended replicate set |
| QC review | confirm whether spectra are suitable for interpretation | QC pass/fail, signal intensity, replicate correlation |
| Results | show screening output and type-risk context | SSI score, threshold, recommendation, cancer type probabilities |
| Report | produce reviewable clinical-style output | patient/session summary, QC summary, result summary |
| Audit log | support traceability | login, upload, analysis, report-generation events |

## 4. Software Architecture

```text
User browser
   |
   v
FastAPI clinical webapp
   |
   +--> Authentication/session layer
   +--> Patient/session database
   +--> Spectrum upload and QC workflow
   +--> SERS prediction engine
   +--> Clinical report generator
   +--> Audit trail
```

| Component | Product-level role |
| --- | --- |
| Clinical webapp | guides user through the clinical workflow |
| UI templates/static assets | present forms, QC tables, results, and reports |
| Authentication/session layer | limits access to authorized users |
| Clinical database | stores local session metadata, users, and audit records |
| Prediction engine | runs SERS-only or fusion model inference after preprocessing/QC |
| Report generator | converts result data into a clinical-style report view |
| Audit trail | records key user/system events for traceability |

## 5. User-Facing Features

| Feature | User value | Notes |
| --- | --- | --- |
| Role-based login | separates technician/clinician/admin usage | supports usability and traceability |
| Patient/session creation | keeps uploads and results tied to one session | patient identifiers must follow site policy |
| Multi-file spectrum upload | supports replicate SERS measurements | minimum/maximum file limits are shown in the IFU |
| QC review | prevents interpretation of unsuitable spectra | QC failure should prompt remeasurement |
| SSI score display | presents cancer screening signal on a 0-10 index | not a diagnosis |
| Cancer type probability display | shows relative type-risk distribution | only interpreted after screening context |
| Clinical report generation | creates reviewable output for clinician workflow | report is controlled by IFU interpretation rules |
| Audit log | supports accountability and review | not a clinical result |

## 6. Dashboard / Screenshot Requirements

Screenshots stored in this folder should show only synthetic/demo or de-identified data.

Required screenshot set:

| Screenshot type | Required view |
| --- | --- |
| UI flow | login, patient information, upload, QC, results, report |
| Dashboard | clinical dashboard or product dashboard summary |
| Clinical report | representative normal/control and cancer-risk report mockups |
| Error/QC state | QC failure or upload validation example if available |

Initial screenshot candidates are staged from:

- `/home/insu/solum-dashboard/clinical_reports/screenshots/`
- `/home/insu/solum-dashboard/SERS_AI_Clinical_Results.html`
- `/home/insu/solum-dashboard/SERS_AI_Executive_Dashboard.html`

## 7. IFU Connection

The current IFU/User Manual source is:

```text
docs/clinical_use/USER_MANUAL.md
```

Key IFU-linked concepts:

| Product concept | IFU/User Manual section |
| --- | --- |
| Intended use and limitation | sections 1, 6, 7 |
| Target users | section 2 |
| Basic terminology | section 3 |
| Login and patient entry | sections 5.1, 5.2 |
| Spectrum upload | section 5.3 |
| QC review | section 5.4 |
| Result interpretation | sections 5.5, 6 |
| Report generation | section 5.6 |
| Audit trail | section 9 |

## 8. Excluded From This Folder

Do not upload:

- raw or processed spectrum matrices
- uploaded patient files
- `clinical_data.db`
- user credentials
- audit logs with real users/patients
- source code explanations
- temporary screenshots containing identifiers

## 9. Git Commit References

| Commit | Relevance |
| --- | --- |
| `45bbc72` | initial clinical app, templates, prediction pipeline, and IFU/User Manual sources |
| `1cd527f` | merged later development materials and deployment updates |
| `cc3a998` | created SharePoint overview/model development governance |
| `fae245a` | recorded SharePoint document commit references |
| `9e68d15` | latest experiment results governance reference before this SW product package |

Add the final `03_SW_Product` commit hash after commit.
