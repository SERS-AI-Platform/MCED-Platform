"""Legacy models retained as comparison baselines.

Submodules:
    resnet_v1         — ResNet18-1D two-stage (pre-stacking baseline).
    lr_fusion_v1      — LR on SERS + clinical (former production).
    sersnet_ensemble  — 0.8·LR_fusion + 0.2·ResNet (clinical-aware ensemble).

These are kept importable so `sers train --model resnet18` etc. continue to
work and so that `sers compare` can run fair head-to-head evaluations.
"""
