# SERS-AI Project Goals

| Field | Value |
| --- | --- |
| Project | SERS-AI |
| Document role | Project objective and scope |
| Version | v0.1 |
| Last updated | 2026-05-15 |

## 1. Project Goal

Develop and validate an AI-assisted SERS diagnostic platform that can support multi-cancer screening and cancer type risk estimation from clinically collected spectra, while maintaining a reproducible software pipeline and controlled document governance.

## 2. Technical Objectives

| Objective | Target state |
| --- | --- |
| Reproducible preprocessing | raw spectra can be transformed into model-ready features using versioned scripts/config |
| Cancer detection model | binary cancer/non-cancer detection KPI tracked by AUC, sensitivity, specificity |
| Cancer type identification | type-level risk/F1 tracked for supported cancer classes |
| Clinical/report output | generate review-ready reports and dashboards without exposing raw data unnecessarily |
| Governance | separate code, controlled documents, raw data, logs, and final deliverables |

## 3. Current Scope

In scope:

- SERS spectrum preprocessing and model training/evaluation code.
- Multi-cancer detection and type identification experiments.
- KPI tracking for AUC, sensitivity, specificity, and type ID F1.
- Clinical-use draft documentation and reporting templates.
- SharePoint folder governance and file remapping.

Out of scope for this Project Overview folder:

- raw spectra files
- temporary results and logs
- model binary registry decisions
- personal documents
- duplicate draft decks

## 4. SharePoint Role

The SharePoint `Project_Overview` folder is the management-facing single source. It should answer:

- What is the project?
- What are the current goals?
- What are the latest KPI values?
- What is the current status?
- What are the top risks?
- What changed since the last version?

Detailed experiment history remains in Git under `docs/experiment.md`, `docs/CHANGELOG.md`, and `docs/EXPERIMENT_CONTEXT.md`.
