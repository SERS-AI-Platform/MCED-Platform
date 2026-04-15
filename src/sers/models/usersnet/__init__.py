"""uSERS-Net — current production model (Stacking V2).

Public API:
    - load_production: load the artifact at `artifacts/usersnet/current/`.
    - stacking:        Level-0 base + Level-1 meta-learner implementations.
    - clinical_fusion: clinical variable loader + scaler (v1.1.0 target).

Internal architecture: Stacking V2 (10 base models + ElasticNet meta).
Clinical fusion integration is a skeleton; full late-fusion at meta-learner
is planned for v1.1.0.
"""

from sers.models.usersnet import stacking, clinical_fusion  # noqa: F401

__all__ = ["stacking", "clinical_fusion"]
