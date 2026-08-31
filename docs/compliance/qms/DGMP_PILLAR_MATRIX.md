# DGMP / QMS Pillar Matrix

This matrix is a working readiness tracker for MCED-Platform. It shows where each pillar is evidenced in Git and which gates remain open before the project can describe the software process as externally controlled.

Controlled approval copies belong in SharePoint `/02_Regulatory_QMS/`; this Git document is the technical source and traceability index.

## Pillar Status

| Pillar | Intended control | Current repo evidence | Status | Open gate |
|---|---|---|---|---|
| Document control | Document IDs, versions, owners, approval state, review cycle | [SOP_INDEX.md](SOP_INDEX.md), [DEVELOPMENT_ASSET_GOVERNANCE.md](../DEVELOPMENT_ASSET_GOVERNANCE.md), SharePoint remap docs | Draft index | Assign document owners, effective dates, controlled-copy links |
| SDLC / change control | PR-based change review, branch discipline, commit history, change rationale | `.github/workflows/ci.yml`, [CONTRIBUTING.md](../../../CONTRIBUTING.md), PR #10 quality gates | Active internal gate | Make branch protection and required checks explicit in repo/admin policy |
| V&V / test evidence | Automated tests, coverage floor, process coverage visibility | [COVERAGE_PROCESS_MATRIX.md](../COVERAGE_PROCESS_MATRIX.md), `tests/`, `scripts/quality/coverage_by_process.py` | Package baseline gated | Add separate deployment-software and clinical workflow test lane |
| Requirements traceability | User requirements, risks, evaluation tasks, evidence links | [clinical/TRACEABILITY_MATRIX.md](../../clinical/TRACEABILITY_MATRIX.md), [sharepoint/sw_product](../../sharepoint/sw_product/) | Draft working set | Link requirements to software versions, test evidence, and approved IFU copy |
| Data governance | Raw-data exclusion, de-identification rules, controlled storage, dataset provenance | `.gitignore`, [DEVELOPMENT_ASSET_GOVERNANCE.md](../DEVELOPMENT_ASSET_GOVERNANCE.md), config paths | Partial | Define approved raw-data location, data intake SOP, and dataset release records |
| Model development control | Training workflow, model artifact provenance, validation scope, model-release criteria | [MODEL_WORKFLOW.md](../../architecture/MODEL_WORKFLOW.md), `scripts/training/`, `artifacts/`, coverage/process gates | Partial | Add locked model-release checklist and artifact sign-off record |
| Software product control | Clinical app workflow, IFU linkage, packaging/deployment source | `scripts/deployment/`, [sharepoint/sw_product](../../sharepoint/sw_product/), [clinical](../../clinical/) | Draft working set | Add deployment test lane, build reproducibility evidence, software version policy |
| Configuration / release management | Environment, dependencies, Docker/infra, model config, release notes | `pyproject.toml`, `config/`, `infra/`, [CHANGELOG.md](../../CHANGELOG.md) | Partial | Define release versioning, rollback, and artifact retention SOP |
| Security / access / audit | Authentication, authorization, audit trail, secrets handling | `scripts/deployment/clinical_auth.py`, `clinical_audit.py`, `clinical_db.py`, `.gitignore` | Implementation present | Add security SOP, audit-log retention policy, and secrets review checklist |
| Risk management | Hazard/risk register, mitigations, residual risk, review cadence | clinical-use traceability, terminology restrictions, open quality gates | Open gate | Create controlled risk register and map mitigations to requirements/tests |
| CAPA / incident handling | Issue intake, severity, root cause, corrective action, verification | GitHub issues/PRs can support evidence | Open gate | Define CAPA SOP and issue labels/workflow |
| Training / role responsibility | Role matrix, required training, completion evidence | Team onboarding docs | Open gate | Create role/training matrix and keep completion records outside Git |

## How To Use This In Reviews

Use the matrix as a status slide, not as a compliance claim.

Recommended wording:

> DGMP/QMS pillars are now visible in a single readiness matrix. Engineering gates for code, CI, tests, and package coverage are active; SOP ownership, controlled-copy approval, deployment-software V&V, CAPA, and training records remain open gates.

Avoid:

> DGMP is complete.

That would overstate the current repo state because several pillars are tracked but not yet controlled.

## Update Rule

Update this file when:

- a new SOP is added to [SOP_INDEX.md](SOP_INDEX.md)
- a CI/coverage gate changes
- a SharePoint controlled copy is approved
- a pillar owner or review cadence changes
- software product scope changes under `scripts/deployment/`
