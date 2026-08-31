# SERS-AI 01_Model_Development Governance

SharePoint destination:

```text
/01_Projects/SERS-AI/01_Model_Development/
```

Purpose: explain the model structure and development approach at concept level.

## File Set

| File | Role | Rule |
| --- | --- | --- |
| `2026_SERS-AI_ModelDevelopment_Overview_v0.1.md` | model structure, features, preprocessing, change history | concept-level only |
| `VERSION_LOG.md` | SharePoint-facing version log | update for every revision |

## Content Rules

- Explain model structure and approach, not implementation details.
- Include feature definitions and preprocessing concepts.
- Include model change history.
- Do not explain rule code or function-level logic.
- Equations and diagrams are allowed.
- Every version must include Git commit references.
- Keep the document concise enough for review, ideally 2-4 pages.

## Primary Git Sources

| Source | Purpose |
| --- | --- |
| `models/model.py` | conceptual reference for 2-stage ResNet-style architecture |
| `models/production/manifest.json` | production model scope, classes, clinical features, thresholds |
| `models/production/preprocessing.json` | production preprocessing parameters |
| `models/run_train_val_test.py` | held-out validation workflow reference |
| `models/run_multimodal.py` | SERS + clinical fusion reference |
| `models/train_ensemble.py` | LR + ResNet ensemble reference |
| `scripts/deployment/sers_predict.py` | production inference behavior reference |
| `src/sers/preprocessing.py` | preprocessing concept reference |
| `docs/ml/experiment.md` | phase-level model history |
| `docs/ml/EXPERIMENT_CONTEXT.md` | detailed experiment context |
