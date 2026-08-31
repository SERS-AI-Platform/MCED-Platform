# SERS-AI Development Asset Governance

Version: v0.1  
Date: 2026-05-13  
Owner: R&BD / SERS-AI project owner

## Purpose

This document defines how SERS-AI development assets are separated across WSL/Ubuntu, GitHub, and SharePoint. The goal is to prevent code, raw data, reports, and business documents from being mixed in one location.

This document follows the SharePoint directory governance rules in:

- `SOLUM_Healthcare_SharePoint_Directory_Governance_v1.0.docx`
- `SOLUM_Healthcare_SharePoint_Remapping_Table.docx`

## Source Of Truth

| Asset type | Source of truth | Notes |
| --- | --- | --- |
| Source code | GitHub | Python, JS, Docker, CI, tests, reproducible scripts |
| Runtime environment | GitHub | `pyproject.toml`, `config/`, `infra/`, lock files |
| Raw spectra / raw clinical data | WSL or controlled R&BD storage | Do not commit to Git. Do not place in general SharePoint. |
| Model binaries | WSL by default; Git only for released lightweight baselines | Large `.joblib`, `.npz`, `.npy` need review before commit. |
| Experiment logs | WSL | Summaries may go to docs or Obsidian, not raw logs. |
| Reports / PPT / Word / PDF | SharePoint | Final or review-ready versions only. |
| Regulatory / QMS / SOP | SharePoint | Use `02_Regulatory_QMS`. |
| Clinical study documents | SharePoint | Use `03_Clinical_Study`; do not store identifiable patient raw data. |
| Patent / IP drafts | SharePoint for controlled versions | Working drafts may stay local until ready. |
| Management / weekly / monthly meetings | SharePoint | Use `06_Management_Report`. Do not commit PPT decks. |
| Conference deliverables | SharePoint | Use `01_Projects/SERS-AI/Conference/` or archive. |

## Canonical Local Workspaces

| Local path | Role | Git status |
| --- | --- | --- |
| `/home/insu/SERS-AI` | Main SERS-AI code repository | Git-tracked repository |
| `/home/insu/solum-dashboard` | Dashboard/reporting web assets | Separate workspace, not part of `SERS-AI` Git repo |
| `/home/insu/outputs` | Temporary generated outputs | Not a long-term source of truth |
| `/mnt/c/Users/YoonInsu_HealthCare_/OneDrive - solum` | OneDrive/SharePoint synced source | Do not use as code source of truth |

## Project Boundaries

### 1. SERS-AI Core Platform

Repository: `/home/insu/SERS-AI`

Keep in Git:

- `src/`
- `scripts/`
- `models/*.py`
- `config/`
- `infra/`
- `tests/`
- `.github/`
- reproducible documentation in `docs/`

Keep out of Git:

- raw spectra data
- patient-level raw clinical files
- ad hoc logs
- personal meeting decks
- conference deliverables not needed for reproducibility

### 2. Clinical Web App / Deployment

Location inside repo:

- `scripts/deployment/`
- deployment templates
- demo data that is synthetic and clearly labeled

Allowed in Git:

- app source code
- HTML templates
- synthetic demo data
- deployment scripts

Not allowed:

- real patient uploads
- production credentials
- audit logs

### 3. Metabolite Profiling

Location:

- `metabolite_profiling/`

Allowed in Git:

- analysis scripts
- final small summary tables
- reproducible figure scripts
- report markdown

Review before committing:

- generated dashboards
- large figures
- metabolite reference exports

SharePoint destination for final outputs:

- `/01_Projects/SERS-AI/Metabolite_Profiling/`

### 4. AACR / Conference Deliverables

Local location:

- `/home/insu/SERS-AI/AACR`

Git status:

- ignored by `.gitignore`
- not tracked

SharePoint destination:

- `/01_Projects/SERS-AI/Conference/2026_AACR/`

### 5. Management / Monthly Meeting

Local examples:

- `monthly_meeting_*.pptx`

Git status:

- ignored by `.gitignore`
- not tracked

SharePoint destination:

- `/06_Management_Report/Monthly_Meeting/2026/`

### 6. solum-dashboard

Local location:

- `/home/insu/solum-dashboard`

Recommended handling:

- If this is active code, create a separate Git repository.
- If it is report output only, upload final HTML/PDF/PPT files to SharePoint and keep code local or in a dashboard-specific repo.

Possible SharePoint destinations:

- `/01_Projects/SERS-AI/Dashboard/`
- `/06_Management_Report/IR/`
- `/05_Business_Partnership/Strategy/`

## SharePoint Mapping For Current SERS-AI Assets

| Current asset | Recommended SharePoint path | Git handling |
| --- | --- | --- |
| `publications/aacr/` | `/01_Projects/SERS-AI/Conference/2026_AACR/` | Publication work products remain outside Git |
| `monthly_meeting_*.pptx` | `/06_Management_Report/Monthly_Meeting/2026/` | Ignore in Git |
| `docs/clinical/` | `/03_Clinical_Study/Clinical_Use/` or `/02_Regulatory_QMS/Product_Documentation/` | Keep source docs in Git only if they are technical/reproducible |
| Obsidian `07_Patent/SERS-AI_repository/` | `/04_IP_Patent/2026_SERS-AI/` | Patent working files live in the Obsidian vault, not this repository |
| `metabolite_profiling/*.html` | `/01_Projects/SERS-AI/Metabolite_Profiling/Reports/` | Review before committing generated HTML |
| `figures/` | Depends on purpose: project report, management report, or publication | Keep only reproducibility-critical figures |
| `data/raw_data*` | Controlled R&BD storage only | Never Git |
| `results/` | WSL only; summary reports may go to SharePoint | Never Git by default |
| `logs/`, `*.log`, `*.db` | WSL only | Never Git |
| `artifacts/`, `models/production*` | WSL or model registry; Git only after review | Avoid large binary drift |

## File Naming Rules

Use this format for SharePoint documents:

```text
YYYY_Project_Topic_DocType_Status
```

Examples:

```text
2026_SERS-AI_AACR_Poster_Final.pdf
2026_SERS-AI_MetaboliteProfiling_Report_Final.pdf
2026_SERS-AI_MonthlyMeeting_2026-04_Final.pptx
2026_SERS-AI_IP_Strategy_Draft.docx
```

Avoid:

```text
final_final.pptx
new folder/
backup/
temp/
personal-name-folder/
```

## Development To SharePoint Workflow

1. Develop and analyze in WSL.
2. Commit only source code, configuration, tests, and reproducible technical documentation to Git.
3. Export final report artifacts to a staging folder.
4. Rename files using the standard naming rule.
5. Upload final/review-ready files to the correct SharePoint top-level directory.
6. Keep raw data and logs out of SharePoint unless the storage is explicitly controlled by R&BD policy.
7. Add a link to SharePoint in the relevant Git/Obsidian project note if needed.

## Recommended Staging Folder Layout

Use a local staging folder before upload:

```text
/home/insu/outputs/sharepoint_staging/
  01_Projects/
    SERS-AI/
      Conference/
      Metabolite_Profiling/
      Dashboard/
  02_Regulatory_QMS/
  03_Clinical_Study/
  04_IP_Patent/
  05_Business_Partnership/
  06_Management_Report/
  90_Templates/
  99_Archive/
```

The staging folder is not a source of truth. It exists only to prepare clean uploads.

## Git Hygiene Checklist

Run before every commit:

```bash
git status --short
git diff --stat
git diff --cached --stat
git ls-files AACR monthly_meeting_2026_04.pptx
```

Expected:

- no `AACR/` tracked files
- no `monthly_meeting_*.pptx` tracked files
- no `*.log`, `*.db`, `.env`, raw spectra, or patient raw data

## Current Ignore Policy

The repository should ignore:

- `AACR/`
- `monthly_meeting_*.pptx`
- `*:Zone.Identifier`
- `*.log`
- `*.db`
- `.env`
- raw `data/*` except selected scripts and approved summaries

## Open Cleanup Items

These need separate review before changing tracked files:

1. Decide whether `artifacts/` and binary model files should remain in Git.
2. Decide whether generated HTML dashboards should be tracked or moved to SharePoint only.
3. Decide whether `metabolite_profiling/poster_figures/` belongs in Git or SharePoint.
4. Decide whether `/home/insu/solum-dashboard` should become its own Git repository.
