"""Model registry for the active SERS-AI model set."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    name: str
    public_name: str
    module: str
    train_script: str
    artifact: str
    status: str


MODEL_REGISTRY = {
    "usersnet": ModelSpec(
        name="usersnet",
        public_name="uSERS-Net",
        module="sers.models.usersnet.stacking",
        train_script="scripts/training/train_usersnet.py",
        artifact="artifacts/usersnet/current",
        status="active",
    ),
}


__all__ = ["MODEL_REGISTRY", "ModelSpec"]
