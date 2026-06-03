"""
Multi-View Attention Network Experiment

Three-input architecture:
  1. Full spectrum (935) → 1D-CNN encoder → 64d
  2. 1st derivative (935) → 1D-CNN encoder → 64d
  3. Peak features (75)  → FC encoder → 64d

  → Cross-Attention fusion → Classification heads

Compares against LR baselines on same data.

Usage:
    python scripts/analysis/multiview_attention_experiment.py
    python scripts/analysis/multiview_attention_experiment.py --epochs 200
"""

from __future__ import annotations

import sys
import logging
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks, peak_widths
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.integrate import trapezoid

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, f1_score

import warnings
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from src.sers.config import RESULTS_DIR

logger = logging.getLogger(__name__)

CANCER_TYPES = ["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"]
NON_CANCER = ["NOR", "DIA", "HBP", "H.D."]
# Aliases: PAN ← CPAN+YPAN, NOR ← YNOR, SPAN excluded (post-op)
GROUP_ALIASES = {"PAN": ["CPAN", "YPAN"], "NOR": ["YNOR"]}

# Known SERS peaks from metabolite profiling
KNOWN_PEAKS = [
    (448.1, "ring_deform", 15), (538.7, "SS_stretch", 15),
    (617.7, "CS_stretch", 15), (683.3, "creatinine", 15),
    (723.8, "adenine", 15), (795.1, "hippuric", 15),
    (849.1, "tyrosine", 15), (895.4, "uric_acid", 15),
    (933.9, "creatinine2", 15), (999.5, "phe_urea", 15),
    (1147.9, "uric_CN", 15), (1230.8, "amide_III", 20),
    (1292.5, "CH2_twist", 15), (1352.3, "trp_fermi", 15),
    (1448.7, "CH2_deform", 20), (1597.1, "purine_CC", 20),
    (1651.1, "amide_I", 20),
]


# =============================================================================
# 1. Feature Extraction
# =============================================================================

def extract_1st_derivative(X):
    return np.apply_along_axis(
        lambda y: savgol_filter(y, window_length=11, polyorder=3, deriv=1),
        axis=1, arr=X,
    )


def voigt_func(x, amplitude, center, sigma, gamma):
    return amplitude * voigt_profile(x - center, sigma, gamma)


def extract_peak_features(X, wavenumbers):
    """Extract Voigt-fitted peak features at 17 known positions."""
    n_samples = len(X)
    n_peaks = len(KNOWN_PEAKS)

    # 4 features per peak + 7 ratios
    peak_areas = np.zeros((n_samples, n_peaks))
    peak_heights = np.zeros((n_samples, n_peaks))
    peak_fwhms = np.zeros((n_samples, n_peaks))
    peak_shifts = np.zeros((n_samples, n_peaks))

    for pi, (center, name, half_w) in enumerate(KNOWN_PEAKS):
        lo_wn, hi_wn = center - half_w * 1.5, center + half_w * 1.5
        mask = (wavenumbers >= lo_wn) & (wavenumbers <= hi_wn)
        if mask.sum() < 5:
            continue

        x_region = wavenumbers[mask]
        for si in range(n_samples):
            y_region = X[si, mask]
            y_shifted = y_region - np.min(y_region)

            try:
                amp_guess = max(np.max(y_shifted), 1e-6)
                p0 = [amp_guess, center, half_w / 3, half_w / 3]
                bounds_lo = [0, center - half_w, 0.1, 0.1]
                bounds_hi = [amp_guess * 10, center + half_w, half_w * 2, half_w * 2]
                popt, _ = curve_fit(voigt_func, x_region, y_shifted,
                                   p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=2000)
                amp, ctr, sigma, gamma = popt
                y_fit = voigt_func(x_region, *popt)
                area = trapezoid(y_fit, x_region)
                fL = 2 * gamma
                fG = 2 * sigma * np.sqrt(2 * np.log(2))
                fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL**2 + fG**2)
            except (RuntimeError, ValueError):
                amp, ctr, area, fwhm = 0.0, center, 0.0, 0.0

            peak_areas[si, pi] = area
            peak_heights[si, pi] = amp
            peak_fwhms[si, pi] = fwhm
            peak_shifts[si, pi] = ctr - center

    # Stack: [area_1..area_17, height_1..height_17, fwhm_1..fwhm_17, shift_1..shift_17]
    features = [peak_areas, peak_heights, peak_fwhms, peak_shifts]

    # Add 7 biologically meaningful ratios
    peak_name_to_idx = {p[1]: i for i, p in enumerate(KNOWN_PEAKS)}
    ratio_pairs = [
        ("phe_urea", "adenine"), ("phe_urea", "creatinine"),
        ("hippuric", "creatinine"), ("CS_stretch", "creatinine"),
        ("amide_I", "CH2_deform"), ("adenine", "purine_CC"),
        ("tyrosine", "phe_urea"),
    ]
    for na, nb in ratio_pairs:
        ia, ib = peak_name_to_idx[na], peak_name_to_idx[nb]
        features.append(peak_areas[:, ia:ia+1] / (peak_areas[:, ib:ib+1] + 1e-10))

    return np.hstack(features)  # (N, 75)


# =============================================================================
# 2. Multi-View Attention Network
# =============================================================================

class SpectralCNNEncoder(nn.Module):
    """Lightweight 1D-CNN for spectrum/derivative encoding.

    Kept small to prevent overfitting on 1,700 samples.
    """
    def __init__(self, input_dim=935, embed_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),  # 935 → 233

            nn.Conv1d(16, 32, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),  # 233 → 58

            nn.Conv1d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),  # 58 → 1
        )
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x):
        """(B, L) → (B, embed_dim)"""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        h = self.conv(x).squeeze(-1)  # (B, 64)
        return self.proj(h)


class PeakFCEncoder(nn.Module):
    """FC encoder for peak features (75-dim)."""
    def __init__(self, input_dim=75, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class MultiViewCrossAttention(nn.Module):
    """Cross-attention between 3 views.

    Each view attends to all other views to learn which
    combinations are most informative for classification.

    peak_features (query) × spectrum (key/value)
    → "이 peak 위치에서 스펙트럼의 어떤 특징이 중요한가"
    """
    def __init__(self, embed_dim=64, n_heads=4, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, tokens):
        """
        tokens: (B, 3, embed_dim) — 3 view embeddings as token sequence
        returns: (B, embed_dim) — fused representation
        """
        # Self-attention across 3 views
        attn_out, attn_weights = self.self_attn(tokens, tokens, tokens)
        tokens = self.norm1(tokens + attn_out)
        tokens = self.norm2(tokens + self.ffn(tokens))

        # Pool: weighted average using learned importance
        fused = tokens.mean(dim=1)  # (B, embed_dim)
        return fused, attn_weights


class MultiViewAttentionNet(nn.Module):
    """
    Three-input architecture:
        Full spectrum (935) → CNN → 64d ─┐
        1st derivative (935) → CNN → 64d ─┼→ Cross-Attention → 64d → Heads
        Peak features (75) → FC → 64d  ─┘

    Two-stage output:
        Stage 1: Binary (cancer vs non-cancer)
        Stage 2: Cancer type (7 classes)
    """
    def __init__(self, n_spectral=935, n_peak_feat=75, embed_dim=64,
                 n_heads=4, n_cancer_types=7, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim

        # Three branch encoders
        self.spectrum_encoder = SpectralCNNEncoder(n_spectral, embed_dim)
        self.deriv_encoder = SpectralCNNEncoder(n_spectral, embed_dim)
        self.peak_encoder = PeakFCEncoder(n_peak_feat, embed_dim)

        # View-type embeddings (learnable, like positional encoding)
        self.view_embedding = nn.Parameter(torch.randn(3, embed_dim) * 0.02)

        # Cross-attention fusion
        self.attention = MultiViewCrossAttention(embed_dim, n_heads, dropout=0.1)

        # Classification heads
        self.binary_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.cancer_type_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, n_cancer_types),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, spectrum, derivative, peak_features):
        """
        spectrum:       (B, 935)
        derivative:     (B, 935)
        peak_features:  (B, 75)
        """
        # Encode each view
        e_spec = self.spectrum_encoder(spectrum)     # (B, 64)
        e_deriv = self.deriv_encoder(derivative)     # (B, 64)
        e_peak = self.peak_encoder(peak_features)    # (B, 64)

        # Stack as token sequence + add view-type embedding
        tokens = torch.stack([e_spec, e_deriv, e_peak], dim=1)  # (B, 3, 64)
        tokens = tokens + self.view_embedding.unsqueeze(0)       # broadcast

        # Cross-attention fusion
        fused, attn_weights = self.attention(tokens)  # (B, 64)

        # Classification
        binary_logit = self.binary_head(fused)         # (B, 1)
        cancer_logits = self.cancer_type_head(fused)   # (B, 7)

        return {
            "binary_logit": binary_logit,
            "binary_prob": torch.sigmoid(binary_logit),
            "cancer_logits": cancer_logits,
            "cancer_probs": F.softmax(cancer_logits, dim=-1),
            "embedding": fused,
            "attn_weights": attn_weights,
        }

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# =============================================================================
# 3. Dataset
# =============================================================================

class MultiViewDataset(Dataset):
    def __init__(self, X_spec, X_deriv, X_peak, binary_labels, cancer_type_labels,
                 augment=False, noise_std=0.02):
        self.X_spec = torch.FloatTensor(X_spec)
        self.X_deriv = torch.FloatTensor(X_deriv)
        self.X_peak = torch.FloatTensor(X_peak)
        self.binary = torch.FloatTensor(binary_labels).unsqueeze(-1)
        self.cancer_type = torch.LongTensor(cancer_type_labels)
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.X_spec)

    def __getitem__(self, idx):
        spec = self.X_spec[idx]
        deriv = self.X_deriv[idx]
        peak = self.X_peak[idx]

        if self.augment:
            # Same noise/scale for spectrum and derivative (physically consistent)
            noise = torch.randn_like(spec) * self.noise_std
            scale = torch.empty(1).uniform_(0.95, 1.05).item()
            spec = spec * scale + noise
            deriv = deriv * scale + noise * 0.5  # derivative gets less noise

            # Peak features: small multiplicative noise
            peak = peak * (1 + torch.randn_like(peak) * 0.02)

        return {
            "spectrum": spec,
            "derivative": deriv,
            "peak_features": peak,
            "binary_label": self.binary[idx],
            "cancer_type_label": self.cancer_type[idx],
        }


# =============================================================================
# 4. Training Loop
# =============================================================================

class TwoStageLoss(nn.Module):
    def __init__(self, stage2_weight=1.0, label_smoothing=0.1):
        super().__init__()
        self.w2 = stage2_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, output, binary_target, cancer_type_target):
        loss1 = self.bce(output["binary_logit"], binary_target)

        cancer_mask = binary_target.squeeze(-1) > 0.5
        loss2 = torch.tensor(0.0, device=loss1.device)
        if cancer_mask.any():
            loss2 = self.ce(
                output["cancer_logits"][cancer_mask],
                cancer_type_target[cancer_mask],
            )

        total = loss1 + self.w2 * loss2
        return {"total": total, "stage1": loss1, "stage2": loss2}


def make_balanced_sampler(labels):
    counts = np.bincount(labels)
    weights = 1.0 / counts[labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        spec = batch["spectrum"].to(device)
        deriv = batch["derivative"].to(device)
        peak = batch["peak_features"].to(device)
        bl = batch["binary_label"].to(device)
        cl = batch["cancer_type_label"].to(device)

        output = model(spec, deriv, peak)
        losses = criterion(output, bl, cl)

        optimizer.zero_grad()
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += losses["total"].item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_bp, all_bt, all_cl, all_ct = [], [], [], []

    for batch in loader:
        spec = batch["spectrum"].to(device)
        deriv = batch["derivative"].to(device)
        peak = batch["peak_features"].to(device)
        bl = batch["binary_label"].to(device)
        cl = batch["cancer_type_label"].to(device)

        output = model(spec, deriv, peak)
        losses = criterion(output, bl, cl)
        total_loss += losses["total"].item()

        all_bp.append(output["binary_prob"].cpu().numpy())
        all_bt.append(bl.cpu().numpy())
        all_cl.append(output["cancer_logits"].cpu().numpy())
        all_ct.append(cl.cpu().numpy())

    bp = np.concatenate(all_bp).squeeze()
    bt = np.concatenate(all_bt).squeeze()
    cl = np.concatenate(all_cl)
    ct = np.concatenate(all_ct)

    try:
        det_auc = roc_auc_score(bt, bp)
    except ValueError:
        det_auc = np.nan

    cancer_mask = ct >= 0
    if cancer_mask.sum() > 0:
        ct_pred = cl[cancer_mask].argmax(axis=1)
        type_f1 = f1_score(ct[cancer_mask], ct_pred, average="macro", zero_division=0)
    else:
        type_f1 = np.nan

    return total_loss / len(loader), det_auc, type_f1


def train_fold(model, train_ds, val_ds, config, device, fold_idx):
    sampler = make_balanced_sampler(train_ds.binary.squeeze().numpy().astype(int))
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                             sampler=sampler, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                           shuffle=False, num_workers=0, pin_memory=True)

    criterion = TwoStageLoss(stage2_weight=1.0, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=10, factor=0.5
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 20 == 0:
            logger.info(f"    Fold {fold_idx+1} Epoch {epoch+1}: "
                       f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                       f"AUC={val_auc:.4f} F1={val_f1:.4f}")

        if patience_counter >= config["patience"]:
            logger.info(f"    Early stopping at epoch {epoch+1}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    _, det_auc, type_f1 = evaluate(model, val_loader, criterion, device)
    return det_auc, type_f1


# =============================================================================
# 5. LR Baselines (for fair comparison)
# =============================================================================

def run_lr_baseline(X, groups, sample_ids, name, n_splits=5, random_state=42):
    """Groups should already be alias-resolved and filtered before calling."""
    cancer_set = set(CANCER_TYPES)
    bl = np.array([1 if g in cancer_set else 0 for g in groups])
    ctl = np.array([CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups])

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    aucs, f1s = [], []

    for ti, vi in skf.split(X, bl, groups=sample_ids):
        model = make_pipeline(StandardScaler(),
                             LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                              class_weight="balanced", random_state=random_state))
        model.fit(X[ti], bl[ti])
        prob = model.predict_proba(X[vi])[:, 1]
        try:
            aucs.append(roc_auc_score(bl[vi], prob))
        except ValueError:
            aucs.append(np.nan)

        ct_train, ct_test = ctl[ti], ctl[vi]
        cm_tr, cm_te = ct_train >= 0, ct_test >= 0
        present = sorted(set(ct_train[cm_tr]))
        if len(present) >= 2 and cm_te.sum() > 0:
            lm = {c: i for i, c in enumerate(present)}
            im = {i: c for c, i in lm.items()}
            yt = np.array([lm[c] for c in ct_train[cm_tr]])
            yte = np.array([lm.get(c, -1) for c in ct_test[cm_te]])
            seen = yte >= 0
            if seen.sum() > 0:
                m2 = make_pipeline(StandardScaler(),
                                  LogisticRegression(C=1.0, max_iter=1000, solver="saga",
                                                    class_weight="balanced", random_state=random_state))
                m2.fit(X[ti][cm_tr], yt)
                pred = m2.predict(X[vi][cm_te][seen])
                yp = np.array([im[p] for p in pred])
                ytrue = np.array([im[t] for t in yte[seen]])
                f1s.append(f1_score(ytrue, yp, average="macro", zero_division=0))
            else:
                f1s.append(np.nan)
        else:
            f1s.append(np.nan)

    return {
        "name": name,
        "det_auc": f"{np.nanmean(aucs):.4f}±{np.nanstd(aucs):.4f}",
        "type_f1": f"{np.nanmean(f1s):.4f}±{np.nanstd(f1s):.4f}",
        "det_auc_mean": np.nanmean(aucs),
        "type_f1_mean": np.nanmean(f1s),
    }


# =============================================================================
# 6. Main
# =============================================================================

def load_data(csv_path=None):
    if csv_path is None:
        csv_path = RESULTS_DIR / "processed_spectra.csv"
    df = pd.read_csv(csv_path)
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    wavenumbers = np.array([float(c.replace("x_", "")) for c in feat_cols])
    return df, feat_cols, wavenumbers


def aggregate_mean(df, feat_cols):
    return df.groupby(["group", "sample_id"], as_index=False).agg(
        {**{c: "mean" for c in feat_cols}, "replicate": "count"}
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()

    t0 = datetime.now()
    out_dir = RESULTS_DIR / "multiview_attention"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(out_dir / "experiment.log", mode="w", encoding="utf-8"),
        ],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 64)
    logger.info("  Multi-View Attention Network Experiment")
    logger.info("=" * 64)
    logger.info(f"  Device: {device}")
    logger.info(f"  Config: embed_dim={args.embed_dim}, heads={args.n_heads}, "
                f"dropout={args.dropout}, epochs={args.epochs}, lr={args.lr}")

    # --- Data Loading ---
    logger.info("\n[1] Loading & preparing data...")
    df, feat_cols, wavenumbers = load_data()
    df_agg = aggregate_mean(df, feat_cols)

    X_spec = df_agg[feat_cols].values
    groups = df_agg["group"].values
    sample_ids = df_agg["sample_id"].values

    logger.info(f"  Samples: {len(X_spec)}")

    # --- Feature Extraction ---
    logger.info("\n[2] Extracting 3 views...")
    logger.info("  1st derivative...")
    X_deriv = extract_1st_derivative(X_spec)
    logger.info("  Peak features (Voigt fitting, 17 peaks)...")
    X_peak = extract_peak_features(X_spec, wavenumbers)
    X_peak = np.nan_to_num(X_peak, nan=0.0, posinf=0.0, neginf=0.0)
    n_peak_feat = X_peak.shape[1]

    logger.info(f"  Shapes: spectrum={X_spec.shape}, deriv={X_deriv.shape}, peak={X_peak.shape}")

    # --- Normalize peak features globally (for NN stability) ---
    peak_scaler = StandardScaler()
    X_peak_scaled = peak_scaler.fit_transform(X_peak)

    # --- Resolve aliases & Labels ---
    def resolve_group(g):
        for alias, members in GROUP_ALIASES.items():
            if g in members:
                return alias
        return g

    groups = np.array([resolve_group(g) for g in groups])

    cancer_set = set(CANCER_TYPES)
    valid_groups = set(CANCER_TYPES) | set(NON_CANCER)
    valid_mask = np.isin(groups, list(valid_groups))
    X_spec = X_spec[valid_mask]
    X_deriv = X_deriv[valid_mask]
    X_peak_scaled = X_peak_scaled[valid_mask]
    X_peak_raw = X_peak[valid_mask]
    groups = groups[valid_mask]
    sample_ids = sample_ids[valid_mask]

    binary_labels = np.array([1 if g in cancer_set else 0 for g in groups])
    cancer_type_labels = np.array([CANCER_TYPES.index(g) if g in cancer_set else -1 for g in groups])

    logger.info(f"  Valid samples: {len(X_spec)} "
                f"(cancer={binary_labels.sum()}, non-cancer={(1-binary_labels).sum()})")

    # --- Model info ---
    model_tmp = MultiViewAttentionNet(
        n_spectral=X_spec.shape[1], n_peak_feat=n_peak_feat,
        embed_dim=args.embed_dim, n_heads=args.n_heads,
        n_cancer_types=len(CANCER_TYPES), dropout=args.dropout,
    )
    total_params, trainable_params = model_tmp.count_params()
    logger.info(f"\n  Model params: {total_params:,} total, {trainable_params:,} trainable")
    del model_tmp

    # --- CV ---
    logger.info(f"\n[3] {args.n_splits}-fold CV...")
    config = {
        "batch_size": args.batch_size, "lr": args.lr,
        "weight_decay": args.weight_decay, "epochs": args.epochs,
        "patience": args.patience,
    }

    skf = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
    fold_aucs, fold_f1s = [], []

    for fold_idx, (ti, vi) in enumerate(skf.split(X_spec, binary_labels, groups=sample_ids)):
        logger.info(f"\n  ── Fold {fold_idx+1}/{args.n_splits} (train={len(ti)}, val={len(vi)}) ──")

        # Normalize spectrum & derivative per fold (prevent leakage)
        spec_scaler = StandardScaler()
        deriv_scaler = StandardScaler()
        pk_scaler = StandardScaler()

        Xs_tr = spec_scaler.fit_transform(X_spec[ti])
        Xs_va = spec_scaler.transform(X_spec[vi])
        Xd_tr = deriv_scaler.fit_transform(X_deriv[ti])
        Xd_va = deriv_scaler.transform(X_deriv[vi])
        Xp_tr = pk_scaler.fit_transform(X_peak_raw[ti])
        Xp_va = pk_scaler.transform(X_peak_raw[vi])

        train_ds = MultiViewDataset(Xs_tr, Xd_tr, Xp_tr,
                                    binary_labels[ti], cancer_type_labels[ti], augment=True)
        val_ds = MultiViewDataset(Xs_va, Xd_va, Xp_va,
                                  binary_labels[vi], cancer_type_labels[vi], augment=False)

        model = MultiViewAttentionNet(
            n_spectral=X_spec.shape[1], n_peak_feat=n_peak_feat,
            embed_dim=args.embed_dim, n_heads=args.n_heads,
            n_cancer_types=len(CANCER_TYPES), dropout=args.dropout,
        ).to(device)

        det_auc, type_f1 = train_fold(model, train_ds, val_ds, config, device, fold_idx)
        fold_aucs.append(det_auc)
        fold_f1s.append(type_f1)
        logger.info(f"  Fold {fold_idx+1}: Det AUC={det_auc:.4f}, Type F1={type_f1:.4f}")

    mv_auc_mean, mv_auc_std = np.nanmean(fold_aucs), np.nanstd(fold_aucs)
    mv_f1_mean, mv_f1_std = np.nanmean(fold_f1s), np.nanstd(fold_f1s)

    logger.info(f"\n  Multi-View Attention: "
                f"Det AUC={mv_auc_mean:.4f}±{mv_auc_std:.4f}, "
                f"Type F1={mv_f1_mean:.4f}±{mv_f1_std:.4f}")

    # --- LR Baselines ---
    logger.info("\n[4] LR baselines for comparison...")

    baselines = []
    # Full spectrum
    b = run_lr_baseline(X_spec, groups, sample_ids, "LR (full_spectrum)")
    baselines.append(b)
    logger.info(f"  {b['name']}: AUC={b['det_auc']}, F1={b['type_f1']}")

    # 1st derivative
    b = run_lr_baseline(X_deriv, groups, sample_ids, "LR (1st_deriv)")
    baselines.append(b)
    logger.info(f"  {b['name']}: AUC={b['det_auc']}, F1={b['type_f1']}")

    # Combined (concat all 3)
    X_concat = np.hstack([X_spec, X_deriv, X_peak_raw])
    b = run_lr_baseline(X_concat, groups, sample_ids, "LR (concat_3view)")
    baselines.append(b)
    logger.info(f"  {b['name']}: AUC={b['det_auc']}, F1={b['type_f1']}")

    # --- Summary ---
    logger.info("\n" + "=" * 75)
    logger.info("  FINAL COMPARISON")
    logger.info("=" * 75)

    results = [
        {"Model": "Multi-View Attention", "Params": f"{trainable_params:,}",
         "Det_AUC": f"{mv_auc_mean:.4f}±{mv_auc_std:.4f}",
         "Type_F1": f"{mv_f1_mean:.4f}±{mv_f1_std:.4f}"},
    ]
    for b in baselines:
        results.append({
            "Model": b["name"], "Params": "~935",
            "Det_AUC": b["det_auc"], "Type_F1": b["type_f1"],
        })

    summary_df = pd.DataFrame(results)
    logger.info(f"\n{summary_df.to_string(index=False)}")

    # Save
    summary_df.to_csv(out_dir / "results.csv", index=False)
    detail = {
        "multiview_attention": {
            "fold_aucs": [float(x) for x in fold_aucs],
            "fold_f1s": [float(x) for x in fold_f1s],
            "params": trainable_params,
            "config": {k: v for k, v in vars(args).items()},
        },
        "baselines": baselines,
    }
    with open(out_dir / "detail.json", "w") as f:
        json.dump(detail, f, indent=2, default=str)

    elapsed = datetime.now() - t0
    logger.info(f"\n  Completed in {elapsed.total_seconds():.0f}s")
    logger.info(f"  Results saved to: {out_dir}")


if __name__ == "__main__":
    main()
