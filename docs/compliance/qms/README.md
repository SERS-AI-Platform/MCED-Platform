# QMS / DGMP Readiness Index

This folder tracks the working DGMP/QMS readiness matrix for MCED-Platform.

## Scope

Git is used for technical working evidence:

- code, tests, CI, coverage and reproducible workflows
- draft traceability matrices
- SOP status indexes and evidence pointers
- implementation references for the clinical software product

Formal controlled copies, approvals, training records, and released SOP PDFs/DOCX should be maintained in SharePoint:

```text
/02_Regulatory_QMS/
```

## Documents

| Document | Purpose |
|---|---|
| [DGMP_PILLAR_MATRIX.md](DGMP_PILLAR_MATRIX.md) | DGMP/QMS pillar status, evidence, open gates, and ownership |
| [SOP_INDEX.md](SOP_INDEX.md) | SOP registry with status, source evidence, and controlled-copy target |

## Rules

- Do not put raw spectra, patient data, credentials, runtime databases, audit logs, or signed controlled copies in Git.
- Use Git for source Markdown and evidence traceability.
- Use SharePoint version history for controlled SOP approval copies.
- When a pillar moves from draft to approved, update both the Git index and the SharePoint controlled copy reference.

## Current Readiness Statement

The repository has active internal engineering gates for code, tests, CI, core import stability, and package coverage. DGMP/QMS readiness is not yet centrally controlled across all SOPs. The open work is to assign owners, fix effective dates, define review cycles, and link approved controlled copies for each SOP.
