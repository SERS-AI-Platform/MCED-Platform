# SERS-AI 03_SW_Product Governance

SharePoint destination:

```text
/01_Projects/SERS-AI/03_SW_Product/
```

Purpose: describe the actual SERS-AI clinical software product from a user and product perspective.

## Required Contents

| Content | Required | Notes |
| --- | --- | --- |
| UI flow | yes | user journey, not implementation |
| Software architecture | yes | system-level components only |
| Feature description | yes | user-facing feature descriptions |
| Dashboard screenshots | yes | final/review-ready screenshots only |
| IFU linkage | yes | connect app screens/features to IFU/User Manual |
| Git reference | yes | commit hash and source paths |

## Strict Rules

- Do not explain code or function-level logic.
- Do not upload raw spectrum data.
- Do not upload credentials, runtime database files, audit logs, or temporary uploads.
- Describe the software from the user perspective.
- Link every feature to intended use, IFU/User Manual section, or usability artifact where possible.
- Screenshots must be review-ready and not contain real patient identifiers.

## Recommended Folder Structure

```text
03_SW_Product/
  2026_SERS-AI_SWProduct_Overview_v0.1.md
  2026_SERS-AI_SWProduct_IFU_Linkage_v0.1.md
  VERSION_LOG.md
  Screenshots/
    UI_Flow/
    Dashboard/
    Clinical_Report/
  IFU/
```

## Primary Git Sources

| Source | Purpose |
| --- | --- |
| `scripts/deployment/sers_clinical_webapp.py` | clinical webapp route/workflow reference |
| `scripts/deployment/templates/` | UI screen reference |
| `scripts/deployment/static/` | UI styling/interaction reference |
| `scripts/deployment/sers_predict.py` | prediction service integration reference |
| `scripts/deployment/clinical_report.py` | clinical report generation reference |
| `scripts/deployment/clinical_db.py` | local clinical session/audit storage reference |
| `docs/clinical/USER_MANUAL.md` | IFU/User Manual source |
| `docs/clinical/USABILITY_EVALUATION_PROTOCOL.md` | usability validation source |
| `docs/clinical/TRACEABILITY_MATRIX.md` | requirement/use-case traceability |
