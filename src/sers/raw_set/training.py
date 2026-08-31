from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from torch.utils.data import DataLoader

from sers.raw_set.data import PatientRecord, RawPatientDataset, RawPatientItem
from sers.raw_set.model import (
    RawSetConfig,
    RawSetLossWeights,
    RawSetModel,
    RawSetTargets,
    raw_set_loss,
)

FloatArray = NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class FoldData:
    train: tuple[PatientRecord, ...]
    validation: tuple[PatientRecord, ...]


@dataclass(frozen=True, slots=True)
class RawStandardizer:
    center: FloatArray
    scale: FloatArray


@dataclass(frozen=True, slots=True)
class RawTrainingConfig:
    model: RawSetConfig
    epochs: int = 30
    batch_size: int = 32
    learning_rate: float = 1e-3
    reconstruction_weight: float = 0.2
    consistency_weight: float = 0.1
    seed: int = 42
    device: str = "cuda"


@dataclass(frozen=True, slots=True)
class EvaluationContext:
    batch_size: int
    device: torch.device


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    loss: float
    validation_auc: float


@dataclass(frozen=True, slots=True)
class FittedRawSet:
    model: RawSetModel
    standardizer: RawStandardizer
    history: tuple[EpochMetrics, ...]
    device: str


@dataclass(frozen=True, slots=True)
class RawPredictions:
    patient_keys: tuple[str, ...]
    groups: tuple[str, ...]
    labels: np.ndarray
    probabilities: np.ndarray


@dataclass(frozen=True, slots=True)
class BinaryMetrics:
    auc: float
    brier: float
    accuracy: float


def fit_standardizer(records: tuple[PatientRecord, ...]) -> RawStandardizer:
    replicate_rows = np.concatenate([record.replicates for record in records], axis=0)
    center: FloatArray = np.asarray(
        np.mean(replicate_rows, axis=0, dtype=np.float64),
        dtype=np.float32,
    )
    scale: FloatArray = np.asarray(
        np.std(replicate_rows, axis=0, dtype=np.float64),
        dtype=np.float32,
    )
    scale = np.asarray(np.maximum(scale, np.finfo(np.float32).eps), dtype=np.float32)
    return RawStandardizer(center=center, scale=scale)


def standardize_records(
    records: tuple[PatientRecord, ...],
    standardizer: RawStandardizer,
) -> tuple[PatientRecord, ...]:
    transformed = []
    for record in records:
        transformed.append(
            PatientRecord(
                patient_key=record.patient_key,
                group=record.group,
                cancer=record.cancer,
                replicates=(record.replicates - standardizer.center) / standardizer.scale,
                average=(record.average - standardizer.center) / standardizer.scale,
                average_usable=record.average_usable,
            )
        )
    return tuple(transformed)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _loader(
    records: tuple[PatientRecord, ...],
    batch_size: int,
    shuffle: bool,
) -> DataLoader[RawPatientItem]:
    return DataLoader(
        RawPatientDataset(records),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def _targets(batch: RawPatientItem, device: torch.device) -> RawSetTargets:
    return RawSetTargets(
        labels=batch.label.to(device),
        average=batch.average.to(device),
        average_mask=batch.average_mask.to(device),
        replicate_mask=batch.replicate_mask.to(device),
    )


def _validation_auc(
    model: RawSetModel,
    records: tuple[PatientRecord, ...],
    context: EvaluationContext,
) -> float:
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in _loader(records, context.batch_size, False):
            output = model(
                batch.spectra.to(context.device),
                batch.replicate_mask.to(context.device),
            )
            labels.append(batch.label.numpy())
            probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
    return float(roc_auc_score(np.concatenate(labels), np.concatenate(probabilities)))


def fit_raw_set(data: FoldData, config: RawTrainingConfig) -> FittedRawSet:
    _seed_everything(config.seed)
    device = torch.device(config.device)
    standardizer = fit_standardizer(data.train)
    train_records = standardize_records(data.train, standardizer)
    validation_records = standardize_records(data.validation, standardizer)
    model = RawSetModel(config.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    positives = sum(record.cancer for record in data.train)
    negatives = len(data.train) - positives
    positive_class = negatives / max(positives, 1)
    weights = RawSetLossWeights(
        reconstruction=config.reconstruction_weight,
        consistency=config.consistency_weight,
        positive_class=positive_class,
    )
    history = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in _loader(train_records, config.batch_size, True):
            targets = _targets(batch, device)
            output = model(batch.spectra.to(device), targets.replicate_mask)
            losses = raw_set_loss(output, targets, weights)
            optimizer.zero_grad()
            losses.total.backward()
            optimizer.step()
            epoch_losses.append(float(losses.total.detach().cpu()))
        validation_auc = _validation_auc(
            model,
            validation_records,
            EvaluationContext(batch_size=config.batch_size, device=device),
        )
        history.append(
            EpochMetrics(
                epoch=epoch,
                loss=float(np.mean(epoch_losses)),
                validation_auc=validation_auc,
            )
        )
    return FittedRawSet(
        model=model,
        standardizer=standardizer,
        history=tuple(history),
        device=config.device,
    )


def predict_raw_set(
    fitted: FittedRawSet,
    records: tuple[PatientRecord, ...],
) -> RawPredictions:
    transformed = standardize_records(records, fitted.standardizer)
    device = torch.device(fitted.device)
    probabilities: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    patient_keys: list[str] = []
    groups: list[str] = []
    fitted.model.eval()
    with torch.no_grad():
        for batch in _loader(transformed, 64, False):
            output = fitted.model(
                batch.spectra.to(device),
                batch.replicate_mask.to(device),
            )
            probabilities.append(torch.sigmoid(output.logits).cpu().numpy())
            labels.append(batch.label.numpy())
            patient_keys.extend(batch.patient_key)
            groups.extend(batch.group)
    return RawPredictions(
        patient_keys=tuple(patient_keys),
        groups=tuple(groups),
        labels=np.concatenate(labels),
        probabilities=np.concatenate(probabilities),
    )


def binary_metrics(predictions: RawPredictions) -> BinaryMetrics:
    labels = predictions.labels.astype(int)
    probabilities = predictions.probabilities
    return BinaryMetrics(
        auc=float(roc_auc_score(labels, probabilities)),
        brier=float(brier_score_loss(labels, probabilities)),
        accuracy=float(accuracy_score(labels, probabilities >= 0.5)),
    )


def gradient_wavenumber_importance(
    fitted: FittedRawSet,
    records: tuple[PatientRecord, ...],
) -> np.ndarray:
    transformed = standardize_records(records, fitted.standardizer)
    device = torch.device(fitted.device)
    importance = np.zeros(fitted.standardizer.center.shape, dtype=np.float64)
    valid_replicates = 0
    fitted.model.eval()
    for batch in _loader(transformed, 32, False):
        spectra = batch.spectra.to(device).requires_grad_(True)
        replicate_mask = batch.replicate_mask.to(device)
        output = fitted.model(spectra, replicate_mask)
        fitted.model.zero_grad()
        output.logits.sum().backward()
        contribution = torch.abs(spectra.grad * spectra)
        contribution = contribution * replicate_mask.unsqueeze(-1)
        importance += contribution.sum(dim=(0, 1)).detach().cpu().numpy()
        valid_replicates += int(replicate_mask.sum().cpu())
    return (importance / max(valid_replicates, 1)).astype(np.float32)
