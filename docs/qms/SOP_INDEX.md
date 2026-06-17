# SOP Index

This index tracks SOP readiness for MCED-Platform. It is not the controlled SOP archive. Approved copies should be stored in SharePoint `/02_Regulatory_QMS/` and linked back here.

## Status Legend

| Status | Meaning |
|---|---|
| Draft | Working SOP exists as a Git/Markdown source or clear repo evidence |
| Partial | Implementation evidence exists, but SOP owner/status/effective date is not controlled |
| Open gate | SOP is needed, but the controlled process is not yet defined |
| Approved | Controlled copy exists in SharePoint and is referenced here |

## Registry

| SOP ID | SOP / Control Area | Pillar | Current repo evidence | Status | Controlled-copy target |
|---|---|---|---|---|---|
| SOP-QMS-001 | Document control and versioning | Document control | [DEVELOPMENT_ASSET_GOVERNANCE.md](../DEVELOPMENT_ASSET_GOVERNANCE.md), [SHAREPOINT_REMAP_INVENTORY_2026-05-13.md](../SHAREPOINT_REMAP_INVENTORY_2026-05-13.md) | Draft | `/02_Regulatory_QMS/Document_Control/` |
| SOP-SW-001 | Software change control and PR workflow | SDLC / change control | [CONTRIBUTING.md](../../CONTRIBUTING.md), `.github/workflows/ci.yml` | Partial | `/02_Regulatory_QMS/Software_Lifecycle/` |
| SOP-SW-002 | CI quality gates and coverage ratchet | V&V / test evidence | [COVERAGE_PROCESS_MATRIX.md](../COVERAGE_PROCESS_MATRIX.md), `scripts/quality/coverage_by_process.py` | Draft | `/02_Regulatory_QMS/Software_Verification/` |
| SOP-SW-003 | Software build, release, and rollback | Configuration / release management | `pyproject.toml`, `config/`, `infra/`, [CHANGELOG.md](../CHANGELOG.md) | Open gate | `/02_Regulatory_QMS/Software_Release/` |
| SOP-SW-004 | Clinical app deployment workflow | Software product control | `scripts/deployment/`, [sharepoint_sw_product](../sharepoint_sw_product/) | Partial | `/02_Regulatory_QMS/Product_Documentation/` |
| SOP-DATA-001 | Data intake, de-identification, and storage | Data governance | [DEVELOPMENT_ASSET_GOVERNANCE.md](../DEVELOPMENT_ASSET_GOVERNANCE.md), `.gitignore`, `config/` | Partial | `/02_Regulatory_QMS/Data_Governance/` |
| SOP-DATA-002 | Spectrum preprocessing and QC workflow | Data governance / V&V | `src/sers/preprocessing.py`, `src/sers/qc/`, `scripts/qc_validation/`, tests | Draft | `/02_Regulatory_QMS/Analytical_Workflow/` |
| SOP-MODEL-001 | Model training and validation | Model development control | [MODEL_WORKFLOW.md](../MODEL_WORKFLOW.md), `scripts/training/`, `tests/` | Partial | `/02_Regulatory_QMS/Model_Development/` |
| SOP-MODEL-002 | Model artifact and configuration management | Model development control / release | `artifacts/`, `config/`, `scripts/training/build_usersnet_production.py` | Open gate | `/02_Regulatory_QMS/Model_Release/` |
| SOP-CLIN-001 | Clinical app IFU and usability traceability | Requirements traceability | [clinical_use](../clinical_use/), [sharepoint_sw_product](../sharepoint_sw_product/) | Draft | `/02_Regulatory_QMS/Product_Documentation/` |
| SOP-SEC-001 | Access control, secrets, and audit trail | Security / access / audit | `scripts/deployment/clinical_auth.py`, `clinical_audit.py`, `.gitignore` | Open gate | `/02_Regulatory_QMS/Security/` |
| SOP-RISK-001 | Software risk management | Risk management | [clinical_use/TRACEABILITY_MATRIX.md](../clinical_use/TRACEABILITY_MATRIX.md), terminology restrictions | Open gate | `/02_Regulatory_QMS/Risk_Management/` |
| SOP-CAPA-001 | CAPA and incident handling | CAPA / incident handling | GitHub issues/PRs can provide trace evidence | Open gate | `/02_Regulatory_QMS/CAPA/` |
| SOP-TRN-001 | Role training and qualification | Training / role responsibility | [ONBOARDING.md](../../ONBOARDING.md) | Open gate | `/02_Regulatory_QMS/Training/` |

## Minimum Fields Needed Before Approval

Each SOP needs these fields before it can move beyond Draft/Partial:

- owner
- reviewer/approver
- effective date
- review cycle
- controlled-copy SharePoint path
- linked evidence
- change history
- training applicability

## Review Cadence

Recommended cadence until the QMS process is formalized:

- review this index before monthly management review
- update after each material PR that changes CI, release, clinical software behavior, model workflow, or data handling
- reconcile Git working docs with SharePoint controlled copies before external review
