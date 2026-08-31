from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class RawSetConfig:
    n_wavenumbers: int
    channels: int = 32
    latent_dim: int = 64
    dropout: float = 0.2


@dataclass(frozen=True, slots=True)
class RawSetOutput:
    logits: torch.Tensor
    reconstruction: torch.Tensor
    patient_embedding: torch.Tensor
    replicate_embeddings: torch.Tensor
    attention: torch.Tensor


@dataclass(frozen=True, slots=True)
class RawSetTargets:
    labels: torch.Tensor
    average: torch.Tensor
    average_mask: torch.Tensor
    replicate_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RawSetLossWeights:
    reconstruction: float = 0.2
    consistency: float = 0.1
    positive_class: float = 1.0


@dataclass(frozen=True, slots=True)
class RawSetLoss:
    total: torch.Tensor
    classification: torch.Tensor
    reconstruction: torch.Tensor
    consistency: torch.Tensor


class ReplicateEncoder(nn.Module):
    def __init__(self, config: RawSetConfig) -> None:
        super().__init__()
        channels = config.channels
        self.network = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=9, stride=2, padding=4),
            nn.GroupNorm(4, channels),
            nn.GELU(),
            nn.Conv1d(channels, channels * 2, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, channels * 2),
            nn.GELU(),
            nn.Conv1d(channels * 2, config.latent_dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.network(spectra.unsqueeze(1))
        return encoded


class RawSetModel(nn.Module):
    def __init__(self, config: RawSetConfig) -> None:
        super().__init__()
        self.encoder = ReplicateEncoder(config)
        self.attention = nn.Sequential(
            nn.Linear(config.latent_dim, config.latent_dim // 2),
            nn.Tanh(),
            nn.Linear(config.latent_dim // 2, 1),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, 1),
        )
        self.decoder = nn.Sequential(
            nn.LayerNorm(config.latent_dim),
            nn.Linear(config.latent_dim, config.n_wavenumbers),
        )

    def forward(self, spectra: torch.Tensor, replicate_mask: torch.Tensor) -> RawSetOutput:
        batch_size, replicate_count, n_wavenumbers = spectra.shape
        masked_spectra = spectra.masked_fill(~replicate_mask.unsqueeze(-1), 0.0)
        embeddings = self.encoder(masked_spectra.reshape(-1, n_wavenumbers))
        embeddings = embeddings.reshape(batch_size, replicate_count, -1)
        scores = self.attention(embeddings).squeeze(-1)
        scores = scores.masked_fill(~replicate_mask, -torch.inf)
        attention = torch.softmax(scores, dim=1)
        patient_embedding = torch.sum(attention.unsqueeze(-1) * embeddings, dim=1)
        return RawSetOutput(
            logits=self.classifier(patient_embedding).squeeze(-1),
            reconstruction=self.decoder(patient_embedding),
            patient_embedding=patient_embedding,
            replicate_embeddings=embeddings,
            attention=attention,
        )


DEFAULT_LOSS_WEIGHTS = RawSetLossWeights()


def raw_set_loss(
    output: RawSetOutput,
    targets: RawSetTargets,
    weights: RawSetLossWeights = DEFAULT_LOSS_WEIGHTS,
) -> RawSetLoss:
    """Combine diagnosis, trusted-average reconstruction, and replicate consistency."""

    positive_weight = torch.tensor(
        weights.positive_class,
        device=output.logits.device,
        dtype=output.logits.dtype,
    )
    classification = F.binary_cross_entropy_with_logits(
        output.logits,
        targets.labels,
        pos_weight=positive_weight,
    )
    if targets.average_mask.any():
        reconstruction = F.mse_loss(
            output.reconstruction[targets.average_mask],
            targets.average[targets.average_mask],
        )
    else:
        reconstruction = output.reconstruction.sum() * 0.0

    centered = output.replicate_embeddings - output.patient_embedding.unsqueeze(1)
    squared_distance = torch.mean(torch.square(centered), dim=-1)
    valid_distance = squared_distance * targets.replicate_mask
    consistency = valid_distance.sum() / targets.replicate_mask.sum().clamp_min(1)
    total = (
        classification
        + weights.reconstruction * reconstruction
        + weights.consistency * consistency
    )
    return RawSetLoss(
        total=total,
        classification=classification,
        reconstruction=reconstruction,
        consistency=consistency,
    )
