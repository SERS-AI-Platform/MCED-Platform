"""
Spectral Patch Transformer for SERS Cancer Classification
==========================================================

Patches of 21 wavenumber points → self-attention → [CLS] → classification
No CNN local bias — global receptive field from layer 1.
~426K params (half of ResNet18-1D).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SpectralPatchEmbedding(nn.Module):
    """Non-overlapping 1D patch embedding via Conv1d."""

    def __init__(self, in_channels: int, d_model: int, patch_size: int = 21):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv1d(in_channels, d_model, kernel_size=patch_size,
                              stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) → (B, N_patches, d_model)"""
        x = self.proj(x)  # (B, d_model, N_patches)
        return x.transpose(1, 2)  # (B, N_patches, d_model)


class TransformerEncoderBlock(nn.Module):
    """Pre-norm Transformer encoder block with stochastic depth."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int,
                 dropout: float = 0.2, drop_path: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop_path_rate = drop_path

    def _drop_path(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_path_rate == 0.0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) > self.drop_path_rate
        return x * keep / (1.0 - self.drop_path_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self._drop_path(attn_out)
        # FFN with pre-norm
        x = x + self._drop_path(self.ffn(self.norm2(x)))
        return x


class SpectralPatchTransformer(nn.Module):
    """
    Spectral Patch Transformer Encoder.

    Input:  (B, C, L) — C channels, L wavenumber features
    Output: (B, d_model) — [CLS] token representation

    Architecture:
        Patch Embed → [CLS] + Pos Enc → N × TransformerBlock → [CLS] out
    """

    def __init__(self, input_length: int = 933, in_channels: int = 3,
                 patch_size: int = 21, d_model: int = 128, n_heads: int = 4,
                 n_layers: int = 3, ffn_dim: int = 256, dropout: float = 0.2,
                 drop_path_rate: float = 0.1):
        super().__init__()
        self.d_model = d_model

        # Pad input to be divisible by patch_size
        self.pad_len = (patch_size - input_length % patch_size) % patch_size
        padded_len = input_length + self.pad_len
        n_patches = padded_len // patch_size

        self.patch_embed = SpectralPatchEmbedding(in_channels, d_model, patch_size)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Positional encoding (learnable, initialized with sinusoidal)
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))
        self._init_pos_embed(n_patches, d_model)

        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks with linearly increasing drop path
        dpr = [drop_path_rate * i / max(n_layers - 1, 1) for i in range(n_layers)]
        self.blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, ffn_dim, dropout, dpr[i])
            for i in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_dim = d_model

        self._init_weights()

    def _init_pos_embed(self, n_patches, d_model):
        """Initialize positional encoding with sinusoidal values."""
        pos = torch.arange(n_patches + 1).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe = torch.zeros(1, n_patches + 1, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div[:d_model // 2])
        self.pos_embed.data.copy_(pe)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, L) or (B, L) → (B, d_model)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # Pad
        if self.pad_len > 0:
            x = F.pad(x, (0, self.pad_len))

        # Patch embedding
        x = self.patch_embed(x)  # (B, N_patches, d_model)
        B = x.shape[0]

        # Prepend [CLS]
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N_patches+1, d_model)

        # Add positional encoding
        x = self.pos_drop(x + self.pos_embed)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # [CLS] token output
        x = self.norm(x[:, 0])  # (B, d_model)
        return x


# =============================================================================
# Full classification model
# =============================================================================
class SpectralTransformerDetector(nn.Module):
    """Spectral Transformer + Two-stage classification heads."""

    def __init__(self, config, transformer_config=None):
        super().__init__()
        self.config = config

        # Default transformer config
        tc = transformer_config or {}
        self.encoder = SpectralPatchTransformer(
            input_length=config.n_spectral_features,
            in_channels=config.input_channels,
            patch_size=tc.get("patch_size", 21),
            d_model=tc.get("d_model", 128),
            n_heads=tc.get("n_heads", 4),
            n_layers=tc.get("n_layers", 3),
            ffn_dim=tc.get("ffn_dim", 256),
            dropout=tc.get("dropout", 0.2),
            drop_path_rate=tc.get("drop_path_rate", 0.1),
        )

        enc_dim = self.encoder.output_dim

        # Same heads as BaseCancerDetector
        self.binary_head = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(enc_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.head_hidden_dim, 1),
        )

        self.cancer_type_head = nn.Sequential(
            nn.Dropout(config.dropout_rate),
            nn.Linear(enc_dim, config.head_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate * 0.5),
            nn.Linear(config.head_hidden_dim, config.n_cancer_types),
        )

    def forward(self, spectra: torch.Tensor):
        emb = self.encoder(spectra)
        binary_logit = self.binary_head(emb)
        cancer_logits = self.cancer_type_head(emb)

        return {
            "binary_logit": binary_logit,
            "binary_prob": torch.sigmoid(binary_logit),
            "cancer_logits": cancer_logits,
            "cancer_probs": F.softmax(cancer_logits, dim=-1),
            "embedding": emb,
        }


if __name__ == "__main__":
    from sers.models._legacy.resnet_v1.model import ModelConfig

    config = ModelConfig(n_spectral_features=933, input_channels=3,
                         head_hidden_dim=128, dropout_rate=0.5)
    model = SpectralTransformerDetector(config)
    x = torch.randn(4, 3, 933)
    out = model(x)
    print(f"Binary: {out['binary_logit'].shape}")
    print(f"Cancer: {out['cancer_logits'].shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total:,}")
