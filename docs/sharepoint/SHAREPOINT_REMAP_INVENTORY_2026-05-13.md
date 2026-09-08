# SERS-AI SharePoint Remap Inventory

Date: 2026-05-13  
Scope:

- `/home/insu/SERS-AI`
- `/home/insu/solum-dashboard`
- SharePoint governance documents in the Downloads folder

> 2026-07-29 update: repository patent files were moved to the C-drive
> Obsidian vault under `07_Patent/SERS-AI_repository/`. References to
> `docs/patent/` below describe the inventory state captured on 2026-05-13.

Related governance:

- `docs/compliance/DEVELOPMENT_ASSET_GOVERNANCE.md`
- `SOLUM_Healthcare_SharePoint_Directory_Governance_v1.0.docx`
- `SOLUM_Healthcare_SharePoint_Remapping_Table.docx`

## Executive Summary

Current development assets are split across Git, WSL local storage, OneDrive backup folders, generated reports, and dashboard workspaces. The main cleanup rule is:

| Destination | What belongs there |
| --- | --- |
| GitHub | source code, config, tests, reproducible scripts, technical markdown |
| SharePoint | final/review-ready PPT, Word, PDF, HTML reports, management documents |
| WSL local | raw spectra, raw clinical files, logs, model outputs, temporary results |
| Archive/Delete | duplicated reports, obsolete outputs, generated cache, OneDrive metadata |

The recommended local staging root before SharePoint upload is:

```text
/home/insu/outputs/sharepoint_staging/
```

## Immediate Status

| Item | Status |
| --- | --- |
| `AACR/` | local folder exists, ignored by Git, not tracked |
| `monthly_meeting_*.pptx` | local file exists, ignored by Git, not tracked |
| `docs/compliance/DEVELOPMENT_ASSET_GOVERNANCE.md` | added to Git |
| `/home/insu/solum-dashboard` | separate workspace, not part of `SERS-AI` Git repo |
| OneDrive merge logs | local only; not for SharePoint |

## Project-Level Boundaries

| Project / workspace | Local path | Primary purpose | Governance decision |
| --- | --- | --- | --- |
| SERS-AI Core | `/home/insu/SERS-AI` | source code, ML, preprocessing, deployment utilities | GitHub source of truth |
| Clinical Web App | `/home/insu/SERS-AI/scripts/deployment` | clinical upload/reporting demo app | GitHub, only synthetic demo data |
| Metabolite Profiling | `/home/insu/SERS-AI/metabolite_profiling` | metabolite analysis scripts and reports | scripts in Git; final reports to SharePoint |
| AACR | `/home/insu/SERS-AI/AACR` | conference deliverables | SharePoint only, ignored by Git |
| Management Meeting | `/home/insu/SERS-AI/monthly_meeting_*.pptx` | internal meeting deck | SharePoint only, ignored by Git |
| Dashboard Workspace | `/home/insu/solum-dashboard` | dashboards, HTML reports, presentations | split into SharePoint outputs and optional separate Git repo |

## SERS-AI Top-Level Inventory

| Local item | Approx. size | Current nature | Destination | Action |
| --- | ---: | --- | --- | --- |
| `src/` | 732K | package source code | GitHub | keep |
| `scripts/` | 4.2M | analysis, pipeline, deployment scripts | GitHub | keep, review generated/demo data periodically |
| `models/*.py` | part of 59M | model source code | GitHub | keep |
| `models/production*` | part of 59M | model binaries and manifests | Review | decide whether to move binaries to model registry/local WSL |
| `config/` | 40K | runtime config/env locks | GitHub | keep |
| `infra/` | 20K | Docker/deployment infra | GitHub | keep |
| `tests/` | 76K | test code | GitHub | keep |
| `.github/` | 68K | CI, prompts, instructions | GitHub | keep |
| `docs/` | 588K | technical, clinical-use, patent docs | Mixed | split technical Git docs vs SharePoint-controlled Word/PDF/IP docs |
| `data/` | 1.7G | raw/processed/clinical data | WSL controlled storage | keep out of Git except approved scripts/summaries |
| `results/` | 314M | generated outputs | WSL / SharePoint summary only | keep out of Git; upload final summaries only |
| `logs/`, `*.log`, `*.db` | about 2.4M | execution logs/db | WSL only | do not upload or commit |
| `artifacts/` | 58M | trained model binaries | Review | consider moving binary files out of Git |
| `figures/` | 340K | architecture/report figures | Mixed | keep reproducibility-critical figures; final report figures to SharePoint |
| `metabolite_profiling/` | 5.1M | scripts, dashboard, figures | Mixed | keep scripts/report md; final HTML/PDF figures to SharePoint |
| `publications/` | 8.4M | publication/conference assets | SharePoint / Archive | review and move non-code deliverables |
| `AACR/` | 207M | conference deliverables | SharePoint | stage to `/01_Projects/SERS-AI/Conference/2026_AACR/` |
| `monthly_meeting_2026_04.pptx` | 2.6M | management meeting deck | SharePoint | stage to `/06_Management_Report/Monthly_Meeting/2026/` |
| `merge_onedrive_*.log` | about 2M | merge operation logs | WSL local | keep local temporarily; archive/delete after validation |

## GitHub Keep List

These should remain in the `SERS-AI` Git repository:

| Path | Reason |
| --- | --- |
| `src/` | package code |
| `scripts/analysis/`, `scripts/pipeline/`, `scripts/qc_validation/`, `scripts/visualization/` | reproducible analysis code |
| `scripts/deployment/` | clinical demo app/deployment code |
| `models/*.py`, `models/*/model.py`, `models/*/train.py` | model source code |
| `config/` | reproducible configuration |
| `infra/` | deployment infrastructure |
| `tests/` | test suite |
| `.github/workflows/` | CI |
| `.github/instructions/`, `.github/prompts/` | team coding/reporting prompts if intentionally maintained |
| `docs/*.md` | technical documentation |
| `pyproject.toml`, `README.md`, `main.py` | project metadata and entry points |

## SharePoint Upload Candidates

### `/01_Projects/SERS-AI/Project_Overview/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `docs/sharepoint/project_overview/2026_SERS-AI_ProjectOverview_PSRF_v0.1.md` | `2026_SERS-AI_ProjectOverview_PSRF_v0.1.md` | keep as latest 1-page single source |
| `docs/sharepoint/project_overview/2026_SERS-AI_ProjectGoals_v0.1.md` | `2026_SERS-AI_ProjectGoals_v0.1.md` | upload/update when project scope changes |
| `docs/sharepoint/project_overview/2026_SERS-AI_KPI_Status_Risk_v0.1.md` | `2026_SERS-AI_KPI_Status_Risk_v0.1.md` | update before monthly review or KPI/status change |
| `docs/sharepoint/project_overview/VERSION_LOG.md` | `2026_SERS-AI_ProjectOverview_VERSION_LOG.md` | maintain SharePoint-facing version history |

### `/01_Projects/SERS-AI/01_Model_Development/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `docs/sharepoint/model_development/2026_SERS-AI_ModelDevelopment_Overview_v0.1.md` | `2026_SERS-AI_ModelDevelopment_Overview_v0.1.md` | upload as concept-level model structure and approach document |
| `docs/sharepoint/model_development/VERSION_LOG.md` | `2026_SERS-AI_ModelDevelopment_VERSION_LOG.md` | maintain Git commit references and SharePoint version history |
| `docs/sharepoint/model_development/README.md` | optional internal governance note | keep in Git unless SharePoint admins want folder rules uploaded |

### `/01_Projects/SERS-AI/02_Experiment_Results/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `docs/sharepoint/experiment_results/2026_SERS-AI_ExperimentResults_Index_v0.1.md` | `2026_SERS-AI_ExperimentResults_Index_v0.1.md` | upload as result package index and rules |
| `docs/sharepoint/experiment_results/VERSION_LOG.md` | `2026_SERS-AI_ExperimentResults_VERSION_LOG.md` | maintain Git/run references |
| generated result workbook | `2026_SERS-AI_ExperimentResults_PhaseP_Q_5Cancer_v0.1.xlsx` | upload; must include run ID, sample count, dataset definition, commit |
| generated result workbook | `2026_SERS-AI_ExperimentResults_PhaseW_7Cancer_v0.1.xlsx` | upload; must include run ID, sample count, dataset definition, commit |
| `publications/bumbucheo/figures/fig_roc_curves_7c.png` | `Figures/ROC/2026_SERS-AI_PhaseW_7Cancer_ROC.png` | upload final figure only |
| `publications/bumbucheo/figures/fig_confusion_matrices_7c.png` | `Figures/Confusion_Matrix/2026_SERS-AI_PhaseW_7Cancer_ConfusionMatrix.png` | upload final figure only |
| `publications/bumbucheo/figures/fig_sensitivity_bar_7c.png` | `Figures/Performance/2026_SERS-AI_PhaseW_7Cancer_SensitivityBar.png` | upload final figure only |

### `/01_Projects/SERS-AI/03_SW_Product/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `docs/sharepoint/sw_product/2026_SERS-AI_SWProduct_Overview_v0.1.md` | `2026_SERS-AI_SWProduct_Overview_v0.1.md` | upload as product-level software overview |
| `docs/sharepoint/sw_product/2026_SERS-AI_SWProduct_IFU_Linkage_v0.1.md` | `2026_SERS-AI_SWProduct_IFU_Linkage_v0.1.md` | upload as IFU linkage table |
| `docs/sharepoint/sw_product/VERSION_LOG.md` | `2026_SERS-AI_SWProduct_VERSION_LOG.md` | maintain Git and SharePoint version history |
| `docs/clinical/USER_MANUAL.md` | `IFU/2026_SERS-AI_UserManual_Draft.md` | upload as current IFU/User Manual source |
| `docs/clinical/TRACEABILITY_MATRIX.md` | `IFU/2026_SERS-AI_SWProduct_TraceabilityMatrix_Draft.md` | upload if usability traceability is reviewed |
| `/home/insu/solum-dashboard/clinical_reports/screenshots/*.png` | `Screenshots/` | upload only approved de-identified screenshots |

### `/01_Projects/SERS-AI/Conference/2026_AACR/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `AACR/2026 AACR Poster_Final.pdf` | `2026_SERS-AI_AACR_Poster_Final.pdf` | upload |
| `AACR/pipeline.pptx` | `2026_SERS-AI_AACR_Pipeline_Draft.pptx` | review then upload |
| `AACR/figures/` | `Figures/` | upload only final figures |
| `AACR/src/` | not SharePoint by default | keep local unless scripts are needed as supplementary methods |
| `AACR/data/` | not SharePoint by default | keep local/controlled storage |

### `/06_Management_Report/Monthly_Meeting/2026/`

| Local file | Proposed SharePoint name | Action |
| --- | --- | --- |
| `monthly_meeting_2026_04.pptx` | `2026_SERS-AI_MonthlyMeeting_2026-04_Final.pptx` | upload |

### `/01_Projects/SERS-AI/Metabolite_Profiling/Reports/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `metabolite_profiling/METABOLITE_PROFILING_REPORT.md` | `2026_SERS-AI_MetaboliteProfiling_Report.md` | upload/export to PDF if needed |
| `metabolite_profiling/Thermo_Metabolite_Dashboard.html` | `2026_SERS-AI_ThermoMetabolite_Dashboard.html` | upload |
| `metabolite_profiling/figures/` | `Figures/` | upload final figures only |
| `metabolite_profiling/poster_figures/` | `Poster_Figures/` | review; upload final PDF/PNG only |

### `/03_Clinical_Study/Clinical_Use/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `docs/clinical/README.md` | `2026_SERS-AI_ClinicalUse_README.md` | upload if used for clinical-facing review |
| `docs/clinical/TRACEABILITY_MATRIX.md` | `2026_SERS-AI_TraceabilityMatrix_Draft.md` | upload or convert to controlled doc |
| `docs/clinical/USABILITY_EVALUATION_FORMS.md` | `2026_SERS-AI_UsabilityEvaluationForms_Draft.md` | upload |
| `docs/clinical/USABILITY_EVALUATION_PROTOCOL.md` | `2026_SERS-AI_UsabilityEvaluationProtocol_Draft.md` | upload |
| `docs/clinical/USER_MANUAL.md` | `2026_SERS-AI_UserManual_Draft.md` | upload |

### `/04_IP_Patent/2026_SERS-AI/`

| Local file/folder | Proposed SharePoint name | Action |
| --- | --- | --- |
| `docs/patent/*.md` | `2026_SERS-AI_IP_*_Draft.md` | upload if used in IP review |
| `docs/patent/*.docx` | keep as controlled SharePoint docs | consider removing from Git after IP owner review |
| `docs/IP_RD_Patent_Research_2026-03-18.md` | `2026_SERS-AI_IP-RD_Research_Draft.md` | upload |

### `/01_Projects/SERS-AI/Dashboard/`

| Local source | Proposed SharePoint handling | Action |
| --- | --- | --- |
| `/home/insu/solum-dashboard/SERS_AI_CLevel_Presentation.pdf` | `2026_SERS-AI_CLevel_Presentation_Final.pdf` | upload |
| `/home/insu/solum-dashboard/SERS_AI_CLevel_Presentation.pptx` | `2026_SERS-AI_CLevel_Presentation_Final.pptx` | upload |
| `/home/insu/solum-dashboard/SERS_AI_Executive_Dashboard.html` | `2026_SERS-AI_Executive_Dashboard.html` | upload/review |
| `/home/insu/solum-dashboard/SERS_AI_Metabolite_Profiling.html` | `2026_SERS-AI_MetaboliteProfiling_Dashboard.html` | upload/review |
| `/home/insu/solum-dashboard/SERS_AI_Spectral_Explorer.html` | `2026_SERS-AI_SpectralExplorer_Dashboard.html` | upload/review |
| `/home/insu/solum-dashboard/SMCXD04_Analysis_Report_KR.html` | `2026_SERS-AI_SMCXD04_AnalysisReport_KR.html` | upload if approved |

### `/02_Regulatory_QMS/Product_Documentation/`

| Local source | Proposed SharePoint handling | Action |
| --- | --- | --- |
| `/home/insu/solum-dashboard/clinical_reports/*.html` | controlled clinical report mockups | upload selected latest versions only |
| `/home/insu/solum-dashboard/clinical_reports/prostate_v3.0.pdf` | `2026_SERS-AI_ClinicalReport_Prostate_v3.0.pdf` | upload if approved |

### `/05_Business_Partnership/Strategy/`

| Local source | Proposed SharePoint handling | Action |
| --- | --- | --- |
| `/home/insu/solum-dashboard/SERS_AI_Business_Scenarios.html` | `2026_SERS-AI_BusinessScenarios_Draft.html` | upload/review |
| `/home/insu/solum-dashboard/business_scenario/*.pptx` | business scenario decks | upload latest only |
| `/home/insu/solum-dashboard/Consultant_Synthesis_Deck.html` | consultant synthesis | upload/review |

### `/03_Clinical_Study/MFDS_Meeting/` or `/02_Regulatory_QMS/MFDS/`

| Local source | Proposed SharePoint handling | Action |
| --- | --- | --- |
| `/home/insu/solum-dashboard/mfds_consultation_2026/솔루엠헬스케어_식약처대면상담자료_v4.pptx` | `2026_SERS-AI_MFDS_ConsultationDeck_v4.pptx` | upload |
| `/home/insu/solum-dashboard/mfds_consultation_2026/*.md` | MFDS talking points/questions | upload if used for formal review |

## WSL-Only Keep List

These should stay local unless a controlled storage location is explicitly approved:

| Path | Reason |
| --- | --- |
| `data/raw_data/` | raw spectra |
| `data/raw_data_medical/` | raw spectra / medical raw files |
| `data/equipment_test_data/` | equipment raw/test data |
| `data/processed/` | derived data; regenerate where possible |
| `results/` | generated experiment outputs |
| `logs/` | runtime logs |
| `mlflow.db` | local experiment database |
| `pipeline.log`, `test.log`, `texput.log` | logs |
| `/home/insu/solum-dashboard/data/` | dashboard data source; review before sharing |
| `/home/insu/solum-dashboard/server.log`, `tunnel.log`, `tunnel_url.txt` | runtime logs and temporary tunnel info |
| `/home/insu/solum-dashboard/.env` | credentials/config |

## Review Before Git Cleanup

These items are currently sensitive from a repository governance perspective. Do not remove them until the project owner confirms whether they are needed for reproducibility.

| Path | Concern | Recommendation |
| --- | --- | --- |
| `artifacts/` | binary model artifacts in Git | move to model registry/local WSL unless release baselines |
| `models/production/` | binary production artifacts in Git | keep only manifest/config; move binaries if large or sensitive |
| `models/production_stacking/` | binary model artifacts in Git | same as above |
| `data/clinical_data/standardized/figures/` | generated figures in `data` namespace | consider moving final figures to `figures/` or SharePoint |
| `docs/patent/*.docx` | controlled IP documents in Git | consider SharePoint-only after IP owner review |
| `metabolite_profiling/*.html` | generated dashboard HTML | keep in Git only if reproducibility requires it |
| `publications/` | publication deliverables | split source scripts vs final PDFs/PPTs |

## solum-dashboard Inventory

The dashboard workspace is not currently part of the `SERS-AI` Git repo. It should be split into code, final reports, and runtime artifacts.

### Keep As Code If A Separate Dashboard Repo Is Created

| Path | Reason |
| --- | --- |
| `package.json`, `package-lock.json` | JS dependency metadata |
| `spa/` | frontend source, after excluding `.parcel-cache` and build artifacts |
| `serve.py`, `sync_dashboard.py`, `sync_data.py`, `upload_data.py`, `verify_dashboard.py` | dashboard utilities |
| `migration.sql`, `migration_v2.sql` | database migration code |
| `*.mjs`, `*.js` screenshot scripts | report automation |

### SharePoint Final Output Candidates

| Path | Destination |
| --- | --- |
| `SERS_AI_CLevel_Presentation.pdf` | `/06_Management_Report/IR/` or `/01_Projects/SERS-AI/Dashboard/` |
| `SERS_AI_CLevel_Presentation.pptx` | `/06_Management_Report/IR/` |
| `SERS_AI_Executive_Dashboard.html` | `/06_Management_Report/IR/` |
| `SERS_AI_Metabolite_Profiling.html` | `/01_Projects/SERS-AI/Metabolite_Profiling/Reports/` |
| `SERS_AI_Spectral_Explorer.html` | `/01_Projects/SERS-AI/Dashboard/` |
| `SERS_Medical_Data_Evaluation.html` | `/03_Clinical_Study/Reports/` |
| `SMCXD04_Analysis_Report_KR.html` | `/03_Clinical_Study/Reports/` |
| `clinical_reports/*.html`, `clinical_reports/*.pdf` | `/02_Regulatory_QMS/Product_Documentation/` or `/03_Clinical_Study/Clinical_Report_Mockups/` |
| `mfds_consultation_2026/*.pptx` | `/03_Clinical_Study/MFDS_Meeting/` |

### Delete / Ignore / Local Only

| Path | Reason |
| --- | --- |
| `.env` | credentials/config |
| `server.log`, `tunnel.log`, `tunnel_url.txt` | runtime artifacts |
| `__pycache__/` | generated cache |
| `spa/.parcel-cache/` | frontend build cache |
| `_bundle_tmp/` | generated bundle cache |
| `*:Zone.Identifier` | Windows metadata |
| `img/정부24 - 주민등록표 등본(초본) 발급 _ 문서출력.pdf` | likely personal/sensitive; remove from project area after review |

## Priority Plan

### Step 1. Protect Git

Status: mostly done.

- `AACR/` ignored
- `monthly_meeting_*.pptx` ignored
- `*:Zone.Identifier` ignored
- governance guide added

Next Git review:

- review `artifacts/` and binary model files
- review `docs/patent/*.docx`
- decide whether generated dashboards belong in Git

### Step 2. Prepare SharePoint Upload Staging

Use:

```text
/home/insu/outputs/sharepoint_staging/
```

Recommended first staging set:

```text
01_Projects/SERS-AI/Conference/2026_AACR/
06_Management_Report/Monthly_Meeting/2026/
01_Projects/SERS-AI/Metabolite_Profiling/Reports/
01_Projects/SERS-AI/Dashboard/
03_Clinical_Study/MFDS_Meeting/
04_IP_Patent/2026_SERS-AI/
```

### Step 3. Upload Only Final/Review-Ready Files

Do not upload:

- raw spectra
- experiment logs
- cache folders
- `.env`
- draft duplicates
- personal documents

### Step 4. Decide Separate Repo For Dashboard

Decision needed:

```text
Should /home/insu/solum-dashboard become a separate GitHub repo?
```

Suggested repo name:

```text
sers-ai-dashboard
```

### Step 5. Quarterly Cleanup

Run this checklist once per quarter:

```bash
git status --short
git ls-files AACR monthly_meeting_2026_04.pptx
find . -name '*.log' -o -name '*.db' -o -name '*:Zone.Identifier'
du -sh data results artifacts models publications metabolite_profiling
```

## Proposed File Naming For First Upload Batch

| Local asset | SharePoint filename |
| --- | --- |
| `AACR/2026 AACR Poster_Final.pdf` | `2026_SERS-AI_AACR_Poster_Final.pdf` |
| `AACR/pipeline.pptx` | `2026_SERS-AI_AACR_Pipeline_Draft.pptx` |
| `monthly_meeting_2026_04.pptx` | `2026_SERS-AI_MonthlyMeeting_2026-04_Final.pptx` |
| `metabolite_profiling/Thermo_Metabolite_Dashboard.html` | `2026_SERS-AI_ThermoMetabolite_Dashboard.html` |
| `metabolite_profiling/METABOLITE_PROFILING_REPORT.md` | `2026_SERS-AI_MetaboliteProfiling_Report.md` |
| `solum-dashboard/SERS_AI_CLevel_Presentation.pdf` | `2026_SERS-AI_CLevel_Presentation_Final.pdf` |
| `solum-dashboard/SERS_AI_CLevel_Presentation.pptx` | `2026_SERS-AI_CLevel_Presentation_Final.pptx` |
| `solum-dashboard/SMCXD04_Analysis_Report_KR.html` | `2026_SERS-AI_SMCXD04_AnalysisReport_KR.html` |
| `solum-dashboard/mfds_consultation_2026/솔루엠헬스케어_식약처대면상담자료_v4.pptx` | `2026_SERS-AI_MFDS_ConsultationDeck_v4.pptx` |

## Notes

- This inventory is a governance plan, not a destructive cleanup script.
- No local files should be deleted until the project owner confirms.
- SharePoint upload should be performed after staging names are reviewed.
- Git cleanup of binary models should be handled in a separate commit because it may affect reproducibility.
