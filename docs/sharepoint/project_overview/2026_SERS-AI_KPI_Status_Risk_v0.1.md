# SERS-AI KPI, Current Status, and Major Risks

| Field | Value |
| --- | --- |
| Project | SERS-AI |
| Document role | KPI/status/risk tracker |
| Version | v0.1 |
| Last updated | 2026-05-15 |

## 1. KPI Table

| Category | KPI | Current value | Source |
| --- | --- | ---: | --- |
| 5-cancer detection | SERS-only Det AUC | 0.977 +/- 0.005 | `docs/ml/EXPERIMENT_CONTEXT.md`, Phase P |
| 5-cancer detection | Fusion Det AUC | 0.986 +/- 0.004 | `docs/ml/EXPERIMENT_CONTEXT.md`, Phase P |
| 5-cancer detection | SERS-only sensitivity | 0.926 +/- 0.004 | `docs/ml/EXPERIMENT_CONTEXT.md`, Phase P |
| 5-cancer detection | Fusion sensitivity | 0.939 +/- 0.011 | `docs/ml/EXPERIMENT_CONTEXT.md`, Phase P |
| 5-cancer detection | SERS-only specificity | 0.929 +/- 0.019 | `docs/ml/EXPERIMENT_CONTEXT.md`, Phase P |
| 5-cancer detection | Fusion specificity | 0.949 +/- 0.019 | `docs/ml/EXPERIMENT_CONTEXT.md`, Phase P |
| 5-cancer type ID | SERS + sex constraint F1 macro | 0.892 +/- 0.026 | `docs/ml/EXPERIMENT_CONTEXT.md`, Phase Q |
| 7-cancer detection | Fusion Det AUC | 0.979 +/- 0.005 | `docs/ml/experiment.md`, Phase W |
| 7-cancer type ID | Fusion Type ID F1 | 0.923 +/- 0.022 | `docs/ml/experiment.md`, Phase W |

## 2. Status

| Workstream | Current status | Next update trigger |
| --- | --- | --- |
| Experiment tracking | active in Git | new validated phase/result |
| Model performance | strong internal held-out results; external validation still needed | new cohort / new validation protocol |
| SharePoint governance | structure drafted; first upload staging created | SharePoint folder owner approval |
| GitHub governance | non-code AACR/monthly meeting assets removed from tracking | remote repository access fixed |
| Dashboard/report outputs | final outputs identified for SharePoint | decision on separate dashboard repo |
| Clinical/regulatory docs | draft materials exist | formal reviewer and controlled document format confirmed |

## 3. Major Risks And Controls

| Risk | Impact | Control / mitigation | Owner |
| --- | --- | --- | --- |
| KPI values are used externally before review | incorrect claim or regulatory/business confusion | require owner review before external sharing | TBD |
| SharePoint Project Overview becomes stale | team loses single source | update before monthly review and after KPI/status change | TBD |
| Raw data or logs are uploaded to SharePoint | privacy/security/governance issue | upload only staged final/review-ready files | TBD |
| Binary artifacts remain mixed with Git source | repo bloat and reproducibility ambiguity | separate model artifact policy needed | TBD |
| CRC vs pancreatic cancer confusion | type ID performance limitation | more data/features; track Phase AB follow-up | ML owner TBD |
| GitHub remote/auth issue blocks push | governance docs remain local only | confirm repository existence and access | repo owner TBD |

## 4. Update Rule

- Update this file whenever AUC, sensitivity, specificity, type ID F1, status, or top risks change.
- Keep the document to 1-2 pages.
- Record every SharePoint-facing update in `VERSION_LOG.md`.
