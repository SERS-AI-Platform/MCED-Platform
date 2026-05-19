# Active Analysis Scripts

Last cleaned: 2026-05-19

This directory now only keeps analysis scripts tied to the active
uSERS-Net/STK-V2 workflow or its calibration validation.

Active model source of truth:

- Public model name: `uSERS-Net`
- Internal architecture: STK-V2, 10 base models + ElasticNet meta learner
- Trainer: `scripts/training/train_usersnet.py`
- Production builder: `scripts/training/build_usersnet_production.py`
- Artifact pointer: `artifacts/usersnet/current`
- Main run output: `results/training/stacking_v2`
- Main figure output: `results/figures/training/stacking_v2`

Kept here:

| Path | Role |
|---|---|
| `stk_v2/full_eval.py` | STK-V2 re-evaluation and peak interpretation |
| `stk_v2/meta_shap.py` | STK-V2 meta-learner SHAP export |
| `stk_v2/figures/` | STK-V2 publication/diagnostic figure helpers; includes fixed-split and nested-CV rendering CLIs |
| `calibration/validate_pds_stacking.py` | PDS calibration validation against uSERS-Net artifacts |
| `calibration/cross_instrument_calibration.py` | PDS calibration transfer experiment |
| `calibration/cross_instrument_sweep.py` | PDS calibration sweep |

Moved out:

- Historical phase, weekend, alpha-blend, ablation, poster, dashboard, and one-off
  scripts are under `scripts/legacy/analysis/`.
- Historical R figure/export scripts are under `scripts/legacy/visualization_r/`.
- Historical evaluation helpers are under `scripts/legacy/evaluation/`.

Do not put new one-off experiments back in this directory. Put exploratory work
under `scripts/legacy/` or a dated scratch area unless it is part of the active
STK-V2 workflow.
