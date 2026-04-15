"""
Cross-Attention Fusion — SERS + Clinical via Transformer Cross-Attention

Each clinical variable becomes a separate token. The SERS embedding (from ResNet)
attends to clinical tokens, learning which spectral patterns interact with which
clinical variables.

Architecture:
    SERS (B, 1800) ──→ ResNet18-1D ──→ (B, 256) ──→ project ──→ query (B, 1, d)
    Clinical (B, 3) ──→ per-var Linear ──→ key/value tokens (B, 3, d)
                                            │
                                    Cross-Attention × n_layers
                                            │
                                        (B, d_model) ──→ BinaryHead + TypeHead

Attention weights are returned for interpretability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import (
    ModelConfig, ResNet1DEncoder, BasicBlock1D, BinaryHead, CancerTypeHead,
)


@dataclass
class CrossAttentionConfig:
    """Extra config for Cross-Attention fusion."""
    n_clinical: int = 3
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 2
    attn_dropout: float = 0.1
    ffn_expansion: int = 4
    n_input_channels: int = 3  # raw + 1st_deriv + 2nd_deriv


class ClinicalTokenizer(nn.Module):
    """Project each clinical variable (scalar) into a d_model-dim token.

    Preserves per-variable identity for interpretable attention weights.
    """

    def __init__(self, n_clinical: int, d_model: int):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, d_model),
                nn.LayerNorm(d_model),
            )
            for _ in range(n_clinical)
        ])

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        """(B, n_clinical) → (B, n_clinical, d_model)."""
        tokens = []
        for i, proj in enumerate(self.projections):
            tokens.append(proj(clinical[:, i:i+1]))  # (B, d_model)
        return torch.stack(tokens, dim=1)  # (B, n_clinical, d_model)


class CrossAttentionBlock(nn.Module):
    """Pre-norm cross-attention + FFN block.

    Query: SERS token(s), Key/Value: clinical tokens.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 ffn_expansion: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self, query: torch.Tensor, kv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, 1, d_model) — SERS token
            kv: (B, n_clinical, d_model) — clinical tokens

        Returns:
            (updated_query, attn_weights) where attn_weights is (B, 1, n_clinical)
        """
        # Cross-attention with residual
        q_norm = self.norm1(query)
        attn_out, attn_weights = self.cross_attn(q_norm, kv, kv, need_weights=True)
        query = query + attn_out

        # FFN with residual
        query = query + self.ffn(self.norm2(query))

        return query, attn_weights


class MultiChannelResNet1DEncoder(nn.Module):
    """ResNet18-1D encoder with configurable input channels (for raw + derivatives)."""

    def __init__(self, config: ModelConfig, n_input_channels: int = 3):
        super().__init__()
        channels = config.resnet_channels
        blocks = config.resnet_blocks

        self.conv1 = nn.Conv1d(n_input_channels, channels[0], kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(channels[0], channels[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(channels[0], channels[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(channels[1], channels[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(channels[2], channels[3], blocks[3], stride=2)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_dim = channels[-1]
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, n_blocks, stride=1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        return x.squeeze(-1)


class CrossAttentionFusionEncoder(nn.Module):
    """Multi-channel ResNet1D encoder + cross-attention fusion with clinical tokens."""

    def __init__(self, config: ModelConfig, xattn_config: CrossAttentionConfig):
        super().__init__()
        self.xattn_config = xattn_config

        # Spectral encoder (multi-channel: raw + derivatives)
        self.spectral_encoder = MultiChannelResNet1DEncoder(
            config, n_input_channels=xattn_config.n_input_channels
        )
        enc_dim = self.spectral_encoder.output_dim

        # Project encoder output to d_model if different
        self.d_model = xattn_config.d_model
        if enc_dim != self.d_model:
            self.sers_proj = nn.Sequential(
                nn.Linear(enc_dim, self.d_model),
                nn.LayerNorm(self.d_model),
            )
        else:
            self.sers_proj = nn.LayerNorm(self.d_model)

        # Clinical tokenizer
        self.clinical_tokenizer = ClinicalTokenizer(
            xattn_config.n_clinical, self.d_model
        )

        # Cross-attention layers
        self.attn_layers = nn.ModuleList([
            CrossAttentionBlock(
                self.d_model, xattn_config.n_heads,
                xattn_config.attn_dropout, xattn_config.ffn_expansion,
            )
            for _ in range(xattn_config.n_layers)
        ])

        self.final_norm = nn.LayerNorm(self.d_model)
        self.output_dim = self.d_model

    def forward(
        self, spectra: torch.Tensor, clinical: torch.Tensor
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            spectra: (B, 1800)
            clinical: (B, n_clinical)

        Returns:
            (embedding, attn_weights_list)
            embedding: (B, d_model)
            attn_weights_list: list of (B, 1, n_clinical) per layer
        """
        # Spectral encoding
        sers_emb = self.spectral_encoder(spectra)  # (B, 256)

        # Project to d_model and reshape as single query token
        sers_token = self.sers_proj(sers_emb).unsqueeze(1)  # (B, 1, d_model)

        # Clinical tokens
        clin_tokens = self.clinical_tokenizer(clinical)  # (B, n_clinical, d_model)

        # Cross-attention layers
        attn_weights_list = []
        q = sers_token
        for layer in self.attn_layers:
            q, aw = layer(q, clin_tokens)
            attn_weights_list.append(aw)

        # Final embedding
        emb = self.final_norm(q.squeeze(1))  # (B, d_model)
        return emb, attn_weights_list


class CrossAttentionCancerDetector(nn.Module):
    """Two-stage cancer detector with cross-attention clinical fusion."""

    def __init__(self, config: ModelConfig, xattn_config: CrossAttentionConfig):
        super().__init__()
        self.config = config
        self.xattn_config = xattn_config

        self.encoder = CrossAttentionFusionEncoder(config, xattn_config)
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
        emb, attn_weights = self.encoder(spectra, clinical)
        binary_logit = self.binary_head(emb)
        cancer_logits = self.cancer_type_head(emb)

        return {
            "binary_logit": binary_logit,
            "binary_prob": torch.sigmoid(binary_logit),
            "cancer_logits": cancer_logits,
            "cancer_probs": F.softmax(cancer_logits, dim=-1),
            "embedding": emb,
            "attn_weights": attn_weights,
        }


def build_cross_attention_model(
    config: ModelConfig, xattn_config: CrossAttentionConfig
) -> CrossAttentionCancerDetector:
    return CrossAttentionCancerDetector(config, xattn_config)


def cross_attention_model_summary(model: CrossAttentionCancerDetector) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    spectral_p = sum(p.numel() for p in model.encoder.spectral_encoder.parameters())
    attn_p = sum(p.numel() for p in model.encoder.attn_layers.parameters())
    tokenizer_p = sum(p.numel() for p in model.encoder.clinical_tokenizer.parameters())
    heads_p = (
        sum(p.numel() for p in model.binary_head.parameters())
        + sum(p.numel() for p in model.cancer_type_head.parameters())
    )
    lines = [
        "CrossAttention-ResNet18-1D Two-Stage Cancer Detector",
        f"  Spectral encoder (ResNet): {spectral_p:,} params",
        f"  Clinical tokenizer:        {tokenizer_p:,} params",
        f"  Cross-attention layers:    {attn_p:,} params",
        f"  Classification heads:      {heads_p:,} params",
        f"  Total: {total:,} ({trainable:,} trainable)",
        f"  d_model={model.xattn_config.d_model}, "
        f"n_heads={model.xattn_config.n_heads}, "
        f"n_layers={model.xattn_config.n_layers}",
        f"  Clinical features: {model.xattn_config.n_clinical}",
    ]
    return "\n".join(lines)
