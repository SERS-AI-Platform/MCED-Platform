"""
FiLM (Feature-wise Linear Modulation) — Clinical-conditioned ResNet18-1D

Clinical variables (age, sex, BMI) generate per-layer gamma/beta parameters
that modulate ResNet feature maps:   x_out = x * gamma + beta

"If patient is male, amplify prostate-cancer features."

Architecture:
    Clinical (B, 3) ─┐
                      ├─→ FiLMGenerator → (γ, β) per layer
    SERS (B, 1800) ──→ ResNet18-1D with FiLM modulation → (B, 256)
                                                            ├─→ BinaryHead  → P(cancer)
                                                            └─→ TypeHead    → P(type|cancer)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import (
    ModelConfig, BasicBlock1D, BinaryHead, CancerTypeHead, BaseCancerDetector,
)


@dataclass
class FiLMConfig:
    """Extra config for FiLM fusion."""
    n_clinical: int = 3
    film_hidden: int = 64
    n_input_channels: int = 3  # raw + 1st_deriv + 2nd_deriv


class FiLMGenerator(nn.Module):
    """Generate (gamma, beta) from clinical features for one ResNet layer.

    clinical (B, n_clinical) → MLP → (gamma, beta) each (B, n_channels)

    Initialized so gamma=1, beta=0 (identity modulation at start).
    """

    def __init__(self, n_clinical: int, n_channels: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_clinical, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * n_channels),
        )
        # Identity init: gamma=1, beta=0
        with torch.no_grad():
            self.mlp[-1].weight.zero_()
            bias = torch.zeros(2 * n_channels)
            bias[:n_channels] = 1.0  # gamma = 1
            self.mlp[-1].bias.copy_(bias)

    def forward(self, clinical: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (gamma, beta) each of shape (B, C)."""
        out = self.mlp(clinical)
        gamma, beta = out.chunk(2, dim=-1)
        return gamma, beta


class FiLMResNet1DEncoder(nn.Module):
    """ResNet18-1D with FiLM conditioning after each layer.

    Same architecture as ResNet1DEncoder but with clinical modulation.
    """

    def __init__(self, config: ModelConfig, film_config: FiLMConfig):
        super().__init__()
        channels = config.resnet_channels  # (32, 64, 128, 256)
        blocks = config.resnet_blocks      # (2, 2, 2, 2)

        # Stem (multi-channel: raw + derivatives)
        self.conv1 = nn.Conv1d(film_config.n_input_channels, channels[0],
                               kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(channels[0], channels[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels[-1]

        # FiLM generators — one per layer
        self.film1 = FiLMGenerator(film_config.n_clinical, channels[0], film_config.film_hidden)
        self.film2 = FiLMGenerator(film_config.n_clinical, channels[1], film_config.film_hidden)
        self.film3 = FiLMGenerator(film_config.n_clinical, channels[2], film_config.film_hidden)
        self.film4 = FiLMGenerator(film_config.n_clinical, channels[3], film_config.film_hidden)

        self._init_weights()

    def _make_layer(self, in_ch: int, out_ch: int, n_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample = None
        if stride != 1 or in_ch != out_ch:
            downsample = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        layers = [BasicBlock1D(in_ch, out_ch, stride, downsample)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock1D(out_ch, out_ch))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    @staticmethod
    def _apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: x * gamma + beta. gamma/beta are (B, C), x is (B, C, L)."""
        return x * gamma.unsqueeze(-1) + beta.unsqueeze(-1)

    def forward(self, spectra: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        """(B, C, L) or (B, L) spectra + (B, n_clinical) clinical → (B, 256) embedding."""
        x = spectra
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, L) → (B, 1, L)

        # Stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # Layer 1 + FiLM
        x = self.layer1(x)
        g1, b1 = self.film1(clinical)
        x = self._apply_film(x, g1, b1)

        # Layer 2 + FiLM
        x = self.layer2(x)
        g2, b2 = self.film2(clinical)
        x = self._apply_film(x, g2, b2)

        # Layer 3 + FiLM
        x = self.layer3(x)
        g3, b3 = self.film3(clinical)
        x = self._apply_film(x, g3, b3)

        # Layer 4 + FiLM
        x = self.layer4(x)
        g4, b4 = self.film4(clinical)
        x = self._apply_film(x, g4, b4)

        x = self.global_pool(x)
        return x.squeeze(-1)


class FiLMCancerDetector(nn.Module):
    """Two-stage cancer detector with FiLM clinical conditioning.

    Same heads as BaseCancerDetector but encoder accepts (spectra, clinical).
    """

    def __init__(self, config: ModelConfig, film_config: FiLMConfig):
        super().__init__()
        self.config = config
        self.film_config = film_config

        self.encoder = FiLMResNet1DEncoder(config, film_config)
        enc_dim = self.encoder.output_dim

        self.binary_head = BinaryHead(enc_dim, config.head_hidden_dim, config.dropout_rate)
        self.cancer_type_head = CancerTypeHead(
            enc_dim, config.n_cancer_types, config.head_hidden_dim, config.dropout_rate,
        )
        self.register_buffer(
            "cancer_thresholds",
            torch.full((config.n_cancer_types,), config.default_threshold),
        )

    def forward(
        self, spectra: torch.Tensor, clinical: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        emb = self.encoder(spectra, clinical)
        binary_logit = self.binary_head(emb)
        cancer_logits = self.cancer_type_head(emb)

        return {
            "binary_logit": binary_logit,
            "binary_prob": torch.sigmoid(binary_logit),
            "cancer_logits": cancer_logits,
            "cancer_probs": F.softmax(cancer_logits, dim=-1),
            "embedding": emb,
        }


def build_film_model(config: ModelConfig, film_config: FiLMConfig) -> FiLMCancerDetector:
    return FiLMCancerDetector(config, film_config)


def film_model_summary(model: FiLMCancerDetector) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_p = sum(p.numel() for p in model.encoder.parameters())
    film_p = sum(
        sum(p.numel() for p in fg.parameters())
        for fg in [model.encoder.film1, model.encoder.film2,
                   model.encoder.film3, model.encoder.film4]
    )
    heads_p = (
        sum(p.numel() for p in model.binary_head.parameters())
        + sum(p.numel() for p in model.cancer_type_head.parameters())
    )
    lines = [
        "FiLM-ResNet18-1D Two-Stage Cancer Detector",
        f"  Encoder (ResNet + FiLM): {encoder_p:,} params",
        f"    - ResNet base:         {encoder_p - film_p:,} params",
        f"    - FiLM generators:     {film_p:,} params",
        f"  Classification heads:    {heads_p:,} params",
        f"  Total: {total:,} ({trainable:,} trainable)",
        f"  Clinical features: {model.film_config.n_clinical}",
        f"  FiLM hidden dim: {model.film_config.film_hidden}",
    ]
    return "\n".join(lines)
