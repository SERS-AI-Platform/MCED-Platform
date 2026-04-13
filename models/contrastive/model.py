"""
Contrastive Pretraining for SERS — Model & Loss
=================================================

Uses same-patient replicates as natural positive pairs.
NT-Xent loss pushes encoder to ignore measurement noise.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, Sampler

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.model import ResNet1DEncoder, ModelConfig


# =============================================================================
# NT-Xent Loss (SimCLR-style)
# =============================================================================
class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss."""

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        z_i, z_j: (B, D) — L2-normalized embeddings of positive pairs
        """
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Mask out self-similarity
        mask = ~torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(~mask, -1e9)

        # Positive pair indices: (i, i+B) and (i+B, i)
        labels = torch.cat([torch.arange(B, 2 * B), torch.arange(0, B)], dim=0).to(z.device)

        return F.cross_entropy(sim, labels)


# =============================================================================
# Contrastive Encoder (ResNet1D + Projection Head)
# =============================================================================
class ContrastiveEncoder(nn.Module):
    """ResNet1D encoder + MLP projection head for contrastive pretraining."""

    def __init__(self, config: ModelConfig, proj_dim: int = 128):
        super().__init__()
        self.encoder = ResNet1DEncoder(config)
        enc_dim = self.encoder.output_dim

        self.projection = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.BatchNorm1d(enc_dim),
            nn.ReLU(inplace=True),
            nn.Linear(enc_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        """Returns both representation (for downstream) and projection (for loss)."""
        h = self.encoder(x)  # (B, enc_dim)
        z = self.projection(h)  # (B, proj_dim)
        z = F.normalize(z, dim=1)
        return h, z

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Representation only (no projection head)."""
        return self.encoder(x)


# =============================================================================
# Dataset: Patient-indexed for replicate pairs
# =============================================================================
class ContrastiveReplicateDataset(Dataset):
    """Indexes by patient, returns two random replicates as positive pair."""

    def __init__(self, X: np.ndarray, groups: np.ndarray, sample_ids: np.ndarray,
                 augment: bool = True, noise_std: float = 0.02,
                 scale_range: tuple = (0.9, 1.1), shift_max: int = 3,
                 mask_prob: float = 0.1, mask_len_range: tuple = (10, 50)):
        """
        X: (N_spectra, C, L) multi-channel spectra
        groups: (N,) group labels
        sample_ids: (N,) sample IDs
        """
        # Build patient → spectrum indices mapping
        self.X = torch.FloatTensor(X)
        self.patient_to_indices = {}
        for i, (g, s) in enumerate(zip(groups, sample_ids)):
            key = f"{g}_{s}"
            self.patient_to_indices.setdefault(key, []).append(i)

        # Only keep patients with >= 2 replicates
        self.patients = [k for k, v in self.patient_to_indices.items() if len(v) >= 2]
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.mask_prob = mask_prob
        self.mask_len_range = mask_len_range

    def __len__(self):
        return len(self.patients)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Augment spectrum for contrastive learning."""
        # Noise (independent per channel)
        x = x + torch.randn_like(x) * self.noise_std
        # Scale (shared across channels)
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        x = x * scale
        # Shift (shared across channels)
        shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
        # Random mask
        if torch.rand(1).item() < self.mask_prob:
            L = x.shape[-1]
            mask_len = torch.randint(*self.mask_len_range, (1,)).item()
            start = torch.randint(0, max(1, L - mask_len), (1,)).item()
            x[..., start:start + mask_len] = 0
        return x

    def __getitem__(self, idx):
        patient = self.patients[idx]
        indices = self.patient_to_indices[patient]
        # Random sample 2 replicates
        i, j = np.random.choice(len(indices), size=2, replace=False)
        x_i = self.X[indices[i]].clone()
        x_j = self.X[indices[j]].clone()
        if self.augment:
            x_i = self._augment(x_i)
            x_j = self._augment(x_j)
        return {"anchor": x_i, "positive": x_j}


class PatientBatchSampler(Sampler):
    """Samples batches of patient indices (no duplicate patients per batch)."""

    def __init__(self, n_patients: int, batch_size: int, drop_last: bool = True):
        self.n_patients = n_patients
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        indices = np.random.permutation(self.n_patients)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start:start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch.tolist()

    def __len__(self):
        n = self.n_patients // self.batch_size
        if not self.drop_last and self.n_patients % self.batch_size:
            n += 1
        return n
