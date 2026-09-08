# SERS-AI Project Overview Governance

Purpose: maintain a short, current, SharePoint-ready single source for the SERS-AI project.

SharePoint destination:

```text
/01_Projects/SERS-AI/Project_Overview/
```

## File Set

| File | Role | Length rule | Update rule |
| --- | --- | --- | --- |
| `2026_SERS-AI_ProjectOverview_PSRF_v0.1.md` | 1-page project summary / PSRF | 1 page preferred, 2 pages maximum | update whenever project status or KPI changes |
| `2026_SERS-AI_ProjectGoals_v0.1.md` | project objective and scope | 1-2 pages | update only when objective/scope changes |
| `2026_SERS-AI_KPI_Status_Risk_v0.1.md` | KPI, current status, major risks | 1-2 pages | update at least before monthly management review |
| `VERSION_LOG.md` | lightweight version history | concise table | update for every SharePoint-facing revision |

## Governance Rules

- Keep Project Overview as the latest project single source.
- Keep each file short: 1-2 pages maximum.
- Do not store raw data, logs, temporary analysis, or duplicate decks in this folder.
- Use SharePoint version history for formal version control.
- Use Git for source Markdown history and rationale.
- Update the version log whenever a document is revised for SharePoint.

## Source References

Primary internal references:

- `docs/ml/experiment.md`
- `docs/CHANGELOG.md`
- `docs/ml/EXPERIMENT_CONTEXT.md`
- `docs/compliance/DEVELOPMENT_ASSET_GOVERNANCE.md`
- `docs/SHAREPOINT_REMAP_INVENTORY_2026-05-13.md`
