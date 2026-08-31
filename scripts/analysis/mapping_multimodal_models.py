from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
from run_mapping_multiscale_resnet import BLOCK_CHOICES, MultiScaleResidualBlock, _auc


@dataclass(frozen=True, slots=True)
class ClinicalScaler:
    center: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True, slots=True)
class MultimodalFit:
    model: torch.nn.Module
    history: tuple[dict[str, float | int | str], ...]
    best_epoch: int
    device: str


def fit_clinical_scaler(values: np.ndarray, patient_ids: np.ndarray) -> ClinicalScaler:
    selected = values[np.asarray(patient_ids, dtype=int) - 1]
    center = np.nanmedian(selected, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    filled = np.where(np.isfinite(selected), selected, center)
    scale = np.nanstd(filled, axis=0, ddof=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return ClinicalScaler(center.astype(np.float32), scale.astype(np.float32))


def transform_clinical(values: np.ndarray, scaler: ClinicalScaler) -> np.ndarray:
    filled = np.where(np.isfinite(values), values, scaler.center)
    return ((filled - scaler.center) / scaler.scale).astype(np.float32)


class ClinicalInputResNet(torch.nn.Module):
    def __init__(self, n_classes: int, clinical_dim: int, n_blocks: int = 3) -> None:
        super().__init__()
        if n_blocks not in BLOCK_CHOICES:
            raise ValueError(f"n_blocks must be one of {BLOCK_CHOICES}")
        self.clinical_encoder = torch.nn.Sequential(
            torch.nn.Linear(clinical_dim, 16),
            torch.nn.LayerNorm(16),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(16, 8),
            torch.nn.GELU(),
        )
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(9, 32, kernel_size=3, padding=1),
            torch.nn.GroupNorm(8, 32),
            torch.nn.GELU(),
        )
        blocks: list[torch.nn.Module] = []
        in_channels = 32
        blocks_per_stage = n_blocks // 3
        for stage_index, out_channels in enumerate((32, 64, 128)):
            for _ in range(blocks_per_stage):
                blocks.append(MultiScaleResidualBlock(in_channels, out_channels))
                in_channels = out_channels
            if stage_index < 2:
                blocks.append(torch.nn.MaxPool1d(2))
        self.blocks = torch.nn.Sequential(*blocks)
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Dropout(0.20),
            torch.nn.Linear(128, 64),
            torch.nn.GELU(),
            torch.nn.Dropout(0.10),
            torch.nn.Linear(64, n_classes),
        )

    def forward(self, spectrum: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        clinical_embedding = self.clinical_encoder(clinical)
        clinical_channel = clinical_embedding.unsqueeze(-1).expand(-1, -1, spectrum.shape[-1])
        fused = torch.cat((spectrum.unsqueeze(1), clinical_channel), dim=1)
        return self.head(self.blocks(self.stem(fused)))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _predict(model: torch.nn.Module, spectrum: np.ndarray, clinical: np.ndarray, device: str) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(spectrum), 512):
            spectrum_batch = torch.from_numpy(spectrum[start : start + 512]).to(device=device, dtype=torch.float32)
            clinical_batch = torch.from_numpy(clinical[start : start + 512]).to(device=device, dtype=torch.float32)
            output.append(torch.softmax(model(spectrum_batch, clinical_batch), dim=1).cpu().numpy())
    return np.vstack(output)


def fit_multimodal(
    spectrum_train: np.ndarray,
    clinical_train: np.ndarray,
    y_train: np.ndarray,
    spectrum_val: np.ndarray,
    clinical_val: np.ndarray,
    y_val: np.ndarray,
    n_classes: int,
    clinical_dim: int,
    n_blocks: int,
    seed: int,
    epochs: int,
    device: str,
) -> MultimodalFit:
    _set_seed(seed)
    model = ClinicalInputResNet(n_classes, clinical_dim, n_blocks).to(device)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = len(y_train) / np.maximum(n_classes * counts, 1.0)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(spectrum_train.astype(np.float32)),
        torch.from_numpy(clinical_train.astype(np.float32)),
        torch.from_numpy(y_train.astype(np.int64)),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = 1
    stale = 0
    patience = 8 if epochs > 12 else max(2, epochs)
    history: list[dict[str, float | int | str]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch_spectrum, batch_clinical, batch_y in loader:
            batch_spectrum = batch_spectrum.to(device=device, dtype=torch.float32)
            batch_clinical = batch_clinical.to(device=device, dtype=torch.float32)
            batch_y = batch_y.to(device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_spectrum, batch_clinical), batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch_y)
            count += len(batch_y)
        model.eval()
        with torch.inference_mode():
            val_spectrum = torch.from_numpy(spectrum_val.astype(np.float32)).to(device=device)
            val_clinical = torch.from_numpy(clinical_val.astype(np.float32)).to(device=device)
            val_y_tensor = torch.from_numpy(y_val.astype(np.int64)).to(device=device)
            val_logits = model(val_spectrum, val_clinical)
            val_loss = float(loss_fn(val_logits, val_y_tensor).cpu())
            val_probability = torch.softmax(val_logits, dim=1).cpu().numpy()
        train_probability = _predict(model, spectrum_train, clinical_train, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "validation_loss": val_loss,
                "train_auc": _auc(y_train, train_probability),
                "validation_auc": _auc(y_val, val_probability),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "best_epoch": best_epoch,
            }
        )
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    for row in history:
        row["best_epoch"] = best_epoch
    return MultimodalFit(model, tuple(history), best_epoch, device)


def predict_multimodal(model: torch.nn.Module, spectrum: np.ndarray, clinical: np.ndarray, device: str) -> np.ndarray:
    return _predict(model, spectrum, clinical, device)
