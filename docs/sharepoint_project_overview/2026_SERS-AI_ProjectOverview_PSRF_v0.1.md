# SERS-AI Project Overview / PSRF

| Field | Value |
| --- | --- |
| Project | SERS-AI multi-cancer screening platform |
| Owner | SOLUM Healthcare |
| Document role | Project single source summary |
| Version | v0.1 |
| Last updated | 2026-05-15 |
| Status | Active development / validation planning |

## 1. One-Page Summary

SERS-AI is a urine SERS spectrum-based AI platform for multi-cancer screening and cancer type risk estimation. The current development focus is to maintain a reproducible analysis pipeline, validate detection and type-identification performance, organize clinical/reporting materials, and prepare governance for SharePoint and GitHub separation.

The project currently treats GitHub as the source of truth for code, configuration, tests, and technical Markdown. SharePoint is used for final or review-ready management, clinical, regulatory, business, and IP-facing documents. Raw spectra, logs, temporary outputs, and local experiment artifacts remain in WSL/local controlled storage unless separately approved.

## 2. Current KPI Snapshot

| KPI | Current value | Context / source |
| --- | ---: | --- |
| 5-cancer SERS-only detection AUC | 0.977 +/- 0.005 | Held-out test, `docs/EXPERIMENT_CONTEXT.md` Phase P |
| 5-cancer fusion detection AUC | 0.986 +/- 0.004 | SERS + age/sex/BMI, held-out test |
| 5-cancer SERS-only sensitivity | 0.926 +/- 0.004 | Held-out test |
| 5-cancer fusion sensitivity | 0.939 +/- 0.011 | Held-out test |
| 5-cancer SERS + sex constraint type ID F1 | 0.892 +/- 0.026 | Phase Q |
| 7-cancer fusion detection AUC | 0.979 +/- 0.005 | Phase W, `docs/experiment.md` |
| 7-cancer fusion type ID F1 | 0.923 +/- 0.022 | Phase W, `docs/experiment.md` |

## 3. Current Status

| Area | Status |
| --- | --- |
| Core ML pipeline | implemented and tracked in Git |
| Held-out validation | completed for key 5-cancer and 7-cancer settings |
| SharePoint governance | folder rules and remapping inventory drafted |
| Conference / monthly meeting assets | removed from Git tracking and staged for SharePoint |
| Dashboard outputs | identified for SharePoint upload; dashboard code should be treated as a separate repo decision |
| Regulatory / clinical documentation | draft materials exist; formal controlled-document owner still TBD |

## 4. Major Risks

| Risk | Current handling |
| --- | --- |
| Confusion between CRC and pancreatic cancer | documented in Phase AB xAI analysis; requires additional data/features |
| Raw data and generated outputs mixed with project files | SharePoint/Git/WSL separation rules created |
| Binary model artifacts in Git | flagged for separate cleanup decision |
| SharePoint documents becoming stale | Project Overview must be updated before management review or KPI/status changes |
| GitHub remote/auth confusion | local commits exist; push requires confirmed repository access |

## 5. Immediate Next Actions

1. Confirm SharePoint `Project_Overview` folder owner and reviewer.
2. Convert these Markdown files to Word/PDF if SharePoint requires controlled document format.
3. Review KPI values before external sharing.
4. Push latest governance commits after GitHub repository access is fixed.
