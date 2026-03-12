"""
SERS Cancer Detection — ResNet18-1D Two-Stage Model

Architecture (v2 — regularized, smaller encoder):
    ┌─────────────────────────────────────────────────────────┐
    │ ResNet18-1D Encoder (slim)                              │
    │   Conv1d(1→32, K=7, S=2) + BN + ReLU + MaxPool(3,S=2) │
    │   Layer1: 2× BasicBlock(32→32)                         │
    │   Layer2: 2× BasicBlock(32→64,   S=2)                  │
    │   Layer3: 2× BasicBlock(64→128,  S=2)                  │
    │   Layer4: 2× BasicBlock(128→256, S=2)                  │
    │   GlobalAvgPool1D → 256-dim                             │
    └──────────────┬──────────────────────────────────────────┘
                   │
    ┌──────────────▼──────────────────────────────────────────┐
    │ Stage 1: Binary (Cancer vs Non-cancer)                  │
    │   Dropout(0.5) → Linear(256→64) + ReLU → Linear(64→1)  │
    └──────────────┬──────────────────────────────────────────┘
                   │ if P(cancer) > threshold
    ┌──────────────▼──────────────────────────────────────────┐
    │ Stage 2: Cancer Type (7 classes)                        │
    │   Dropout(0.5) → Linear(256→64) + ReLU → Linear(64→7)  │
    │   PRO, BRE, OVA, LUN, CRC, CPAN, SPAN                 │
    └─────────────────────────────────────────────────────────┘

v2 changes (anti-overfitting):
    - Encoder: (64,128,256,512) → (32,64,128,256), ~4x fewer params
    - Dropout: 0.3 → 0.5, weight_decay: 1e-4 → 1e-3
    - Spectral augmentation: noise, scale, shift (training only)
    - Label smoothing (0.1) for Stage 2 CE loss
    - WeightedRandomSampler for class-balanced batches

Author: SOLUM Healthcare / Insu Kim
Date: 2026-02
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# =============================================================================
# Model Configuration
# =============================================================================
@dataclass
class ModelConfig:
    """Configuration for ResNet18-1D two-stage model.

    Cancer types (7): PRO, BRE, OVA, LUN, CRC, CPAN, SPAN
    Non-cancer (4):   NOR, DIA, HBP, H.D.  (binary label=0, no subtype)
    """

    # --- Input ---
    n_spectral_features: int = 1800

    # --- Label scheme (CPAN/SPAN separate, no PAN alias) ---
    cancer_types: Tuple[str, ...] = ("PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "SPAN")
    cancer_groups_raw: Tuple[str, ...] = (
        "PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "SPAN",
    )
    non_cancer_groups: Tuple[str, ...] = ("NOR", "DIA", "HBP", "H.D.")
    group_aliases: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def n_cancer_types(self) -> int:
        return len(self.cancer_types)

    # --- ResNet18-1D Encoder ---
    resnet_channels: Tuple[int, ...] = (32, 64, 128, 256)
    resnet_blocks: Tuple[int, ...] = (2, 2, 2, 2)  # BasicBlocks per layer
    encoder_output_dim: int = 256  # after GlobalAvgPool

    # --- Classification Heads ---
    head_hidden_dim: int = 64  # proportional to 256-dim encoder

    # --- Regularization ---
    dropout_rate: float = 0.5

    # --- Training ---
    learning_rate: float = 5e-4      # lower LR for deeper model
    weight_decay: float = 1e-3
    batch_size: int = 32
    n_epochs: int = 150              # more epochs for deeper model
    early_stopping_patience: int = 20
    scheduler_patience: int = 10
    stage1_loss_weight: float = 1.0
    stage2_loss_weight: float = 1.0

    # --- Reproducibility ---
    random_state: int = 42

    # --- Thresholds ---
    default_threshold: float = 0.5

    # ------------------------------------------------------------------
    # Label helpers
    # ------------------------------------------------------------------
    def resolve_group(self, raw_group: str) -> str:
        for alias, members in self.group_aliases.items():
            if raw_group in members:
                return alias
        return raw_group

    def is_cancer(self, raw_group: str) -> bool:
        return raw_group in self.cancer_groups_raw

    def cancer_type_index(self, raw_group: str) -> int:
        resolved = self.resolve_group(raw_group)
        if resolved in self.cancer_types:
            return self.cancer_types.index(resolved)
        return -1

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_pipeline_config(
        cls,
        config_dict: Dict[str, Any],
        n_spectral_features: int = 1800,
        **overrides,
    ) -> "ModelConfig":
        """Build from config.yaml — CPAN/SPAN stay separate."""
        ds = config_dict.get("dataset", {})

        categories = ds.get("categories", {})
        cancer_raw = tuple(categories.get("cancer", []))
        non_cancer = tuple(
            categories.get("control", []) + categories.get("non_cancer", [])
        )

        # No alias merging: cancer_types = cancer_raw directly
        cancer_types = list(cancer_raw)

        # Order by display.group_order
        display_order = config_dict.get("display", {}).get("group_order", [])
        if display_order:
            ordered = [g for g in display_order if g in cancer_types]
            remaining = [g for g in cancer_types if g not in ordered]
            cancer_types = ordered + remaining
        cancer_types = tuple(cancer_types)

        modeling = config_dict.get("modeling", {})
        random_state = modeling.get("random_state", 42)

        kwargs = dict(
            n_spectral_features=n_spectral_features,
            cancer_types=cancer_types,
            cancer_groups_raw=cancer_raw,
            non_cancer_groups=non_cancer,
            group_aliases={},
            random_state=random_state,
        )
        kwargs.update(overrides)
        return cls(**kwargs)


# =============================================================================
# ResNet18-1D Building Blocks
# =============================================================================
class BasicBlock1D(nn.Module):
    """ResNet BasicBlock adapted for 1D signals.

    Two 3×1 convolutions with skip connection.
    If stride > 1 or channels change, downsample shortcut is applied.
    """
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1,
                 downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1DEncoder(nn.Module):
    """ResNet18-style 1D encoder for SERS spectra.

    Input:  (B, 1, L) or (B, L)  — L spectral points
    Output: (B, 512)              — global feature vector

    Architecture trace (L=1800):
        conv1 k=7,s=2,p=3 → (B, 64, 900)
        maxpool k=3,s=2,p=1 → (B, 64, 450)
        layer1: 2× Block(64,64) → (B, 64, 450)
        layer2: 2× Block(64,128,s=2) → (B, 128, 225)
        layer3: 2× Block(128,256,s=2) → (B, 256, 113)
        layer4: 2× Block(256,512,s=2) → (B, 512, 57)
        GAP → (B, 512)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        channels = config.resnet_channels  # (64, 128, 256, 512)
        blocks = config.resnet_blocks      # (2, 2, 2, 2)

        # Stem
        self.conv1 = nn.Conv1d(1, channels[0], kernel_size=7, stride=2,
                               padding=3, bias=False)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L) or (B, 1, L) → (B, 512)"""
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


class ShallowCNN1DEncoder(nn.Module):
    """Compact 1D CNN encoder for baseline comparison."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.projection = nn.Linear(128, config.encoder_output_dim)
        self.output_dim = config.encoder_output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x).squeeze(-1)
        return self.projection(x)


# =============================================================================
# Classification Heads (wider for 512-dim input)
# =============================================================================
class BinaryHead(nn.Module):
    """Stage 1: Cancer vs Non-cancer. Output: logit (B, 1)."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class CancerTypeHead(nn.Module):
    """Stage 2: Cancer type (7 classes). Output: logits (B, 7)."""
    def __init__(self, input_dim: int, n_classes: int, hidden_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# =============================================================================
# Full Model
# =============================================================================
class SERSCancerDetector(nn.Module):
    """Two-stage ResNet18-1D cancer detection model.

    Stage 1: Binary — Cancer or not?
    Stage 2: Cancer type — Which of 7 types? (gated by Stage 1)

    Cancer types: PRO, BRE, OVA, LUN, CRC, CPAN, SPAN
    Non-cancer:   NOR, DIA, HBP, H.D. (binary label=0)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.encoder = ResNet1DEncoder(config)
        enc_dim = self.encoder.output_dim  # 512

        self.binary_head = BinaryHead(
            enc_dim, config.head_hidden_dim, config.dropout_rate
        )
        self.cancer_type_head = CancerTypeHead(
            enc_dim, config.n_cancer_types, config.head_hidden_dim, config.dropout_rate
        )

        self.register_buffer(
            "cancer_thresholds",
            torch.full((config.n_cancer_types,), config.default_threshold),
        )

    def forward(self, spectra: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Returns
        -------
        dict:
            binary_logit  (B, 1)
            binary_prob   (B, 1)    P(cancer)
            cancer_logits (B, 7)    raw logits
            cancer_probs  (B, 7)    softmax probabilities
            embedding     (B, 512)  encoder output
        """
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

    @torch.no_grad()
    def predict(
        self,
        spectra: torch.Tensor,
        binary_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """Two-stage gated inference."""
        self.eval()
        out = self.forward(spectra)

        cancer_prob = out["binary_prob"].squeeze(-1)
        is_cancer = cancer_prob > binary_threshold

        type_confidence, predicted_type = out["cancer_probs"].max(dim=-1)
        type_thresholds = self.cancer_thresholds[predicted_type]
        type_passes = type_confidence > type_thresholds

        predicted_type = torch.where(
            is_cancer & type_passes,
            predicted_type,
            torch.full_like(predicted_type, -1),
        )

        return {
            "is_cancer": is_cancer,
            "cancer_prob": cancer_prob,
            "predicted_type": predicted_type,
            "type_confidence": type_confidence,
            "cancer_probs": out["cancer_probs"],
        }

    def decode_predictions(
        self, pred: Dict[str, torch.Tensor]
    ) -> List[Dict[str, Any]]:
        results = []
        for i in range(len(pred["is_cancer"])):
            r = {
                "is_cancer": bool(pred["is_cancer"][i]),
                "cancer_prob": float(pred["cancer_prob"][i]),
            }
            idx = int(pred["predicted_type"][i])
            if idx >= 0:
                r["cancer_type"] = self.config.cancer_types[idx]
                r["confidence"] = float(pred["type_confidence"][i])
            else:
                r["cancer_type"] = None
                r["confidence"] = None
            r["type_probs"] = {
                name: float(pred["cancer_probs"][i, j])
                for j, name in enumerate(self.config.cancer_types)
            }
            results.append(r)
        return results

    def set_cancer_thresholds(self, thresholds: Dict[str, float]):
        for i, name in enumerate(self.config.cancer_types):
            if name in thresholds:
                self.cancer_thresholds[i] = thresholds[name]
        logger.info(
            f"Thresholds: "
            f"{dict(zip(self.config.cancer_types, self.cancer_thresholds.tolist()))}"
        )


class CNN1DCancerDetector(SERSCancerDetector):
    """Two-stage shallow 1D CNN model for baseline comparison."""

    def __init__(self, config: ModelConfig):
        nn.Module.__init__(self)
        self.config = config

        self.encoder = ShallowCNN1DEncoder(config)
        enc_dim = self.encoder.output_dim

        self.binary_head = BinaryHead(
            enc_dim, config.head_hidden_dim, config.dropout_rate
        )
        self.cancer_type_head = CancerTypeHead(
            enc_dim, config.n_cancer_types, config.head_hidden_dim, config.dropout_rate
        )

        self.register_buffer(
            "cancer_thresholds",
            torch.full((config.n_cancer_types,), config.default_threshold),
        )


# =============================================================================
# Loss Function
# =============================================================================
class TwoStageLoss(nn.Module):
    """BCE(binary) + CE(cancer_type | cancer samples only)."""

    def __init__(
        self,
        stage1_weight: float = 1.0,
        stage2_weight: float = 1.0,
        pos_weight_stage1: Optional[torch.Tensor] = None,
        class_weights_stage2: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.w1 = stage1_weight
        self.w2 = stage2_weight
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_stage1)
        self.ce = nn.CrossEntropyLoss(
            weight=class_weights_stage2, label_smoothing=label_smoothing,
        )

    def forward(
        self,
        output: Dict[str, torch.Tensor],
        binary_target: torch.Tensor,
        cancer_type_target: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss1 = self.bce(output["binary_logit"], binary_target.float())

        loss2 = torch.tensor(0.0, device=loss1.device)
        cancer_mask = binary_target.squeeze(-1) > 0.5
        if cancer_mask.any():
            loss2 = self.ce(
                output["cancer_logits"][cancer_mask],
                cancer_type_target[cancer_mask].long(),
            )

        total = self.w1 * loss1 + self.w2 * loss2
        return {"total": total, "stage1": loss1, "stage2": loss2}


# =============================================================================
# Dataset
# =============================================================================
class SERSDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for two-stage model with optional spectral augmentation."""

    def __init__(
        self,
        spectra: np.ndarray,
        binary_labels: np.ndarray,
        cancer_type_labels: np.ndarray,
        groups: Optional[List[str]] = None,
        sample_ids: Optional[List[str]] = None,
        augment: bool = False,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        shift_max: int = 3,
    ):
        self.spectra = torch.FloatTensor(spectra)
        self.binary_labels = torch.FloatTensor(binary_labels).unsqueeze(-1)
        self.cancer_type_labels = torch.LongTensor(cancer_type_labels)
        self.groups = groups or ["UNK"] * len(spectra)
        self.sample_ids = sample_ids or [str(i) for i in range(len(spectra))]
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_max = shift_max

    def __len__(self):
        return len(self.spectra)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        # Gaussian noise
        x = x + torch.randn_like(x) * self.noise_std
        # Random intensity scaling
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        x = x * scale
        # Random spectral shift (circular)
        shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=0)
        return x

    def __getitem__(self, idx):
        spec = self.spectra[idx]
        if self.augment:
            spec = self._augment(spec)
        return {
            "spectra": spec,
            "binary_label": self.binary_labels[idx],
            "cancer_type_label": self.cancer_type_labels[idx],
        }


# =============================================================================
# Utilities
# =============================================================================
def build_model(model_name: str, config: ModelConfig) -> nn.Module:
    """Factory for torch-based SERS models."""
    model_name = str(model_name).lower()
    if model_name in {"resnet18", "resnet", "sers_resnet18"}:
        return SERSCancerDetector(config)
    if model_name in {"cnn1d", "cnn", "shallow_cnn"}:
        return CNN1DCancerDetector(config)
    raise ValueError(f"Unsupported torch model: {model_name}")


def model_summary(model: nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = model.config
    lines = [
        "=" * 64,
        "SERS Cancer Detector — ResNet18-1D (Two-Stage)",
        "=" * 64,
        f"Input features:       {cfg.n_spectral_features}",
        f"ResNet channels:      {cfg.resnet_channels}",
        f"ResNet blocks:        {cfg.resnet_blocks}",
        f"Encoder output dim:   {cfg.encoder_output_dim}",
        f"Head hidden dim:      {cfg.head_hidden_dim}",
        f"Cancer types ({cfg.n_cancer_types}):   {cfg.cancer_types}",
        f"Non-cancer groups:    {cfg.non_cancer_groups}",
        f"Aliases:              {cfg.group_aliases}",
        f"Total parameters:     {total:>12,}",
        f"Trainable parameters: {trainable:>12,}",
        "-" * 64,
    ]
    for name, mod in model.named_children():
        n = sum(p.numel() for p in mod.parameters())
        lines.append(f"  {name:25s} {n:>12,} params")
    lines.append("=" * 64)
    return "\n".join(lines)


# =============================================================================
# Smoke Test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    config = ModelConfig()
    model = SERSCancerDetector(config)
    print(model_summary(model))

    B = 8
    x = torch.randn(B, config.n_spectral_features)
    out = model(x)
    print(f"\nForward (batch={B}):")
    for k, v in out.items():
        print(f"  {k:18s}  {str(v.shape):>15s}")

    # Loss
    binary_y = torch.randint(0, 2, (B, 1)).float()
    cancer_y = torch.where(
        binary_y.squeeze() > 0.5,
        torch.randint(0, config.n_cancer_types, (B,)),
        torch.full((B,), -1, dtype=torch.long),
    )
    criterion = TwoStageLoss()
    loss = criterion(out, binary_y, cancer_y)
    print(f"\nLoss: total={loss['total']:.4f}, "
          f"s1={loss['stage1']:.4f}, s2={loss['stage2']:.4f}")

    # Predict
    pred = model.predict(x)
    decoded = model.decode_predictions(pred)
    print(f"\nPredictions:")
    for d in decoded[:3]:
        print(f"  cancer={d['is_cancer']}, type={d['cancer_type']}, "
              f"prob={d['cancer_prob']:.3f}")

    # Config.yaml test
    try:
        import yaml
        with open("config.yaml") as f:
            raw = yaml.safe_load(f)
        mc = ModelConfig.from_pipeline_config(raw, n_spectral_features=1800)
        print(f"\nFrom config.yaml:")
        print(f"  cancer_types ({mc.n_cancer_types}): {mc.cancer_types}")
        print(f"  aliases: {mc.group_aliases}")
        for g in ["CRC", "CPAN", "SPAN", "NOR"]:
            print(f"  {g:5s} → idx={mc.cancer_type_index(g)}, cancer={mc.is_cancer(g)}")
    except FileNotFoundError:
        print("\nconfig.yaml not found, skipping")
