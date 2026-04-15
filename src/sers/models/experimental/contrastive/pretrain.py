#!/usr/bin/env python3
"""
Contrastive Pretraining + Fine-tuning for SERS
================================================

Phase 1: Pretrain encoder with NT-Xent on replicate pairs
Phase 2: Fine-tune or linear probe for cancer classification
Phase 3: Compare with baseline (random init)

Usage:
    python models/contrastive/pretrain.py --data thermo
"""

from __future__ import annotations
import sys, json, argparse, logging, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sers.models._legacy.resnet_v1.model import ModelConfig, SERSCancerDetector, TwoStageLoss, SERSDataset
from sers.models.experimental.contrastive.model import ContrastiveEncoder, NTXentLoss, ContrastiveReplicateDataset, PatientBatchSampler
from src.sers.config import RESULTS_DIR

# Reuse data loading
from sers.models.usersnet.stacking import preprocess_channel, load_raw_multichannel

logging.basicConfig(level=logging.INFO, format="%(asctime)s │ %(levelname)-7s │ %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUT_DIR = RESULTS_DIR / "contrastive"
CANCER_TYPES = ("PRO", "LUN", "CRC", "CPAN", "OVA", "BRE", "BLC")
NON_CANCER_GROUPS = ("NOR", "DIA", "HBP", "H.D.")


# =============================================================================
# Pretraining
# =============================================================================
def pretrain_encoder(X_3ch, df_meta, n_epochs=300, batch_size=128,
                     lr=1e-3, temperature=0.1, device="cuda"):
    """Pretrain encoder with contrastive loss on all data."""
    groups = df_meta["group"].values
    sample_ids = df_meta["sample_id"].values

    dataset = ContrastiveReplicateDataset(X_3ch, groups, sample_ids, augment=True)
    sampler = PatientBatchSampler(len(dataset), batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, pin_memory=True)

    n_ch = X_3ch.shape[1]
    config = ModelConfig(n_spectral_features=X_3ch.shape[2], input_channels=n_ch,
                         resnet_channels=(32, 64, 128, 256), encoder_output_dim=256)
    model = ContrastiveEncoder(config, proj_dim=128).to(device)
    criterion = NTXentLoss(temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    logger.info(f"  Pretraining: {len(dataset)} patients, {n_epochs} epochs, "
                f"batch={batch_size}, temp={temperature}")

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for batch in loader:
            anchor = batch["anchor"].to(device)
            positive = batch["positive"].to(device)

            _, z_i = model(anchor)
            _, z_j = model(positive)
            loss = criterion(z_i, z_j)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if (epoch + 1) % 50 == 0:
            logger.info(f"    Epoch {epoch+1}/{n_epochs}: loss={avg_loss:.4f}")

    return model, losses


# =============================================================================
# Evaluation: Linear Probe + Fine-tune
# =============================================================================
def extract_embeddings(model, X, device, batch_size=256):
    """Extract encoder representations for all spectra."""
    model.eval()
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            x = torch.FloatTensor(X[i:i+batch_size]).to(device)
            h = model.encode(x)
            embeddings.append(h.cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def evaluate_linear_probe(embeddings, binary_labels, cancer_type_labels, sample_ids, n_splits=5):
    """Linear probe: LR on frozen encoder embeddings."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    s1_probs = np.full(len(embeddings), np.nan)
    s2_probs = np.full((len(embeddings), len(CANCER_TYPES)), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(embeddings, binary_labels, sample_ids)):
        # S1
        s1 = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, solver="saga", class_weight="balanced"))
        s1.fit(embeddings[tr_idx], binary_labels[tr_idx])
        s1_probs[val_idx] = s1.predict_proba(embeddings[val_idx])[:, 1]

        # S2
        cancer_mask = binary_labels[tr_idx] == 1
        if cancer_mask.sum() > 10:
            s2 = make_pipeline(StandardScaler(), LogisticRegression(
                C=1.0, max_iter=2000, solver="saga", class_weight="balanced", multi_class="multinomial"))
            s2.fit(embeddings[tr_idx][cancer_mask], cancer_type_labels[tr_idx][cancer_mask])
            raw = s2.predict_proba(embeddings[val_idx])
            for i, cls in enumerate(s2.classes_):
                if cls < len(CANCER_TYPES):
                    s2_probs[val_idx, cls] = raw[:, i]

    auc = roc_auc_score(binary_labels, s1_probs)
    f1b = f1_score(binary_labels, (s1_probs > 0.5).astype(int), average="macro")
    bm = binary_labels == 1
    f1t = f1_score(cancer_type_labels[bm], s2_probs[bm].argmax(axis=1), average="macro", zero_division=0)

    return {"auc": auc, "f1_bin": f1b, "f1_type": f1t}


def evaluate_finetune(pretrained_model, X_3ch, binary_labels, cancer_type_labels,
                      sample_ids, device, n_splits=5, n_epochs=100):
    """Fine-tune pretrained encoder with classification heads."""
    n_ch = X_3ch.shape[1]
    n_feat = X_3ch.shape[2]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    s1_probs = np.full(len(X_3ch), np.nan)
    s2_probs = np.full((len(X_3ch), len(CANCER_TYPES)), np.nan)

    for fold, (tr_idx, val_idx) in enumerate(sgkf.split(X_3ch, binary_labels, sample_ids)):
        config = ModelConfig(n_spectral_features=n_feat, input_channels=n_ch,
                             cancer_types=CANCER_TYPES, non_cancer_groups=NON_CANCER_GROUPS,
                             resnet_channels=(32, 64, 128, 256), encoder_output_dim=256,
                             head_hidden_dim=128, dropout_rate=0.5,
                             learning_rate=1e-4, weight_decay=1e-3)
        model = SERSCancerDetector(config).to(device)

        # Load pretrained encoder weights
        pretrained_dict = {k.replace("encoder.", ""): v
                          for k, v in pretrained_model.encoder.state_dict().items()}
        model.encoder.load_state_dict(pretrained_dict)

        criterion = TwoStageLoss(use_focal_loss=True, focal_gamma=2.0)

        # Differential LR: encoder slower, heads faster
        encoder_params = list(model.encoder.parameters())
        head_params = list(model.binary_head.parameters()) + list(model.cancer_type_head.parameters())
        optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": 1e-4},
            {"params": head_params, "lr": 5e-4},
        ], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

        from torch.utils.data import WeightedRandomSampler
        train_ds = SERSDataset(X_3ch[tr_idx], binary_labels[tr_idx], cancer_type_labels[tr_idx], augment=True)
        val_ds = SERSDataset(X_3ch[val_idx], binary_labels[val_idx], cancer_type_labels[val_idx], augment=False)

        class_counts = np.bincount(binary_labels[tr_idx].astype(int), minlength=2).clip(1)
        weights = 1.0 / class_counts[binary_labels[tr_idx].astype(int)]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

        train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=0, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)

        best_loss, best_state, patience = float("inf"), None, 0
        for epoch in range(n_epochs):
            model.train()
            for batch in train_dl:
                out = model(batch["spectra"].to(device))
                losses = criterion(out, batch["binary_label"].to(device), batch["cancer_type_label"].to(device))
                optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % 5 == 0:
                model.eval()
                vl = []
                with torch.no_grad():
                    for batch in val_dl:
                        out = model(batch["spectra"].to(device))
                        l = criterion(out, batch["binary_label"].to(device), batch["cancer_type_label"].to(device))
                        vl.append(l["total"].item())
                mean_vl = np.mean(vl)
                if mean_vl < best_loss - 1e-4:
                    best_loss = mean_vl
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience >= 8:
                        break

        if best_state:
            model.load_state_dict(best_state)
        model.eval()

        with torch.no_grad():
            all_s1, all_s2 = [], []
            for batch in val_dl:
                out = model(batch["spectra"].to(device))
                all_s1.append(out["binary_prob"].cpu().numpy().squeeze(-1))
                all_s2.append(out["cancer_probs"].cpu().numpy())
            s1_probs[val_idx] = np.concatenate(all_s1)
            s2_probs[val_idx] = np.concatenate(all_s2)

        auc = roc_auc_score(binary_labels[val_idx], s1_probs[val_idx])
        bm = binary_labels[val_idx] == 1
        f1t = f1_score(cancer_type_labels[val_idx][bm], s2_probs[val_idx][bm].argmax(axis=1),
                       average="macro", zero_division=0)
        logger.info(f"  Fold {fold}: AUC={auc:.4f} F1-type={f1t:.4f}")

    auc = roc_auc_score(binary_labels, s1_probs)
    f1b = f1_score(binary_labels, (s1_probs > 0.5).astype(int), average="macro")
    bm = binary_labels == 1
    f1t = f1_score(cancer_type_labels[bm], s2_probs[bm].argmax(axis=1), average="macro", zero_division=0)
    return {"auc": auc, "f1_bin": f1b, "f1_type": f1t}


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", "-d", default="thermo", choices=["thermo", "medical"])
    parser.add_argument("--pretrain-epochs", type=int, default=300)
    parser.add_argument("--finetune-epochs", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.1)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = datetime.now()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 64)
    logger.info("  Contrastive Pretraining — SERS Cancer Classification")
    logger.info("=" * 64)

    THERMO_MAP = {
        "1. Prostate cancer (100개)": "PRO", "2. Breast cancer (30개)": "BRE",
        "3. Ovarian cancer (70개)": "OVA", "4. Lung cancer (300개)": "LUN",
        "5. Normal (100개)": "NOR", "6. Diabetes (100개)": "DIA",
        "7. High blood pressure (100개)": "HBP",
        "8. High blood pressure + Diabetes (100개)": "H.D.",
        "9. Colorectal cancer (300개)": "CRC",
        "10-1. C-Pancreatic cancer (70개)": "CPAN",
        "10-3. Y-Pancreatic cancer (YPAN)": "YPAN",
        "11 BLC (299개)": "BLC", "12. Y-Normal (YNOR)": "YNOR",
    }
    MEDICAL_MAP = {
        "1. Prostate cancer (100개)": "PRO", "2. Breast cancer (30개)": "BRE",
        "3. Ovarian cancer (70개)": "OVA", "4. Lung cancer (300개)": "LUN",
        "5. Normal (100개)": "NOR", "6. Diabetes (100개)": "DIA",
        "7. High blood pressure (100개)": "HBP",
        "8. High blood pressure + Diabetes (100개)": "H.D.",
        "9. Colorectal cancer (300개)": "CRC",
        "10-1. C-Pancreatic cancer (70개)": "CPAN",
        "10-3. Y-Pancreatic cancer (30개)": "YPAN",
        "11. Bladdder Cancer (299개)": "BLC",
        "12. Y-Normal (29개)": "YNOR",
    }

    grid = np.linspace(402.0, 2198.0, 933)
    data_dir = PROJECT_ROOT / "data" / ("raw_data" if args.data == "thermo" else "raw_data_medical")
    folder_map = THERMO_MAP if args.data == "thermo" else MEDICAL_MAP
    pattern = "*.CSV" if args.data == "thermo" else "*.txt"
    max_rep = None if args.data == "thermo" else {"NOR": 6}

    logger.info(f"\n  Loading RAW spectra ({args.data})...")
    X_3ch, df_meta = load_raw_multichannel(data_dir, folder_map, grid, pattern, max_rep)
    df_meta["group"] = df_meta["group"].replace({"YPAN": "CPAN", "YNOR": "NOR"})

    valid = set(CANCER_TYPES) | set(NON_CANCER_GROUPS)
    mask = df_meta["group"].isin(valid).values
    X_3ch = X_3ch[mask]
    df_meta = df_meta[mask].reset_index(drop=True)

    groups_arr = df_meta["group"].values
    sample_ids = (df_meta["group"] + "_" + df_meta["sample_id"].astype(str)).values
    binary_labels = np.array([1.0 if g in CANCER_TYPES else 0.0 for g in groups_arr])
    ct_map = {ct: i for i, ct in enumerate(CANCER_TYPES)}
    cancer_type_labels = np.array([ct_map.get(g, -1) for g in groups_arr])

    logger.info(f"  {len(X_3ch)} spectra, {int(binary_labels.sum())} cancer")

    # Phase 1: Pretrain
    logger.info(f"\n  ── Phase 1: Contrastive Pretraining ──")
    pretrained_model, losses = pretrain_encoder(
        X_3ch, df_meta, n_epochs=args.pretrain_epochs,
        temperature=args.temperature, device=str(device))

    # Save loss curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color="#E91E63")
    ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent Loss")
    ax.set_title("Contrastive Pretraining Loss"); ax.grid(True, alpha=0.3)
    fig.savefig(OUTPUT_DIR / "pretrain_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save encoder
    torch.save(pretrained_model.encoder.state_dict(), OUTPUT_DIR / "pretrained_encoder.pt")

    # Phase 2: Evaluate
    results = {}

    # 2a: Linear probe on pretrained embeddings
    logger.info(f"\n  ── Phase 2a: Linear Probe (frozen encoder + LR) ──")
    embeddings = extract_embeddings(pretrained_model, X_3ch, device)
    results["contrastive_probe"] = evaluate_linear_probe(
        embeddings, binary_labels, cancer_type_labels, sample_ids)
    logger.info(f"  Linear probe: AUC={results['contrastive_probe']['auc']:.4f}, "
                f"F1-type={results['contrastive_probe']['f1_type']:.4f}")

    # 2b: Fine-tune pretrained encoder
    logger.info(f"\n  ── Phase 2b: Fine-tune (pretrained encoder + heads) ──")
    results["contrastive_finetune"] = evaluate_finetune(
        pretrained_model, X_3ch, binary_labels, cancer_type_labels,
        sample_ids, device, n_epochs=args.finetune_epochs)
    logger.info(f"  Fine-tune: AUC={results['contrastive_finetune']['auc']:.4f}, "
                f"F1-type={results['contrastive_finetune']['f1_type']:.4f}")

    # 2c: Baseline — random init encoder + LR (for comparison)
    logger.info(f"\n  ── Phase 2c: Baseline (random encoder + LR) ──")
    config = ModelConfig(n_spectral_features=X_3ch.shape[2], input_channels=X_3ch.shape[1],
                         resnet_channels=(32, 64, 128, 256), encoder_output_dim=256)
    random_model = ContrastiveEncoder(config).to(device)
    random_embeddings = extract_embeddings(random_model, X_3ch, device)
    results["random_probe"] = evaluate_linear_probe(
        random_embeddings, binary_labels, cancer_type_labels, sample_ids)
    logger.info(f"  Random probe: AUC={results['random_probe']['auc']:.4f}, "
                f"F1-type={results['random_probe']['f1_type']:.4f}")

    # 2d: Direct LR on flattened features (ultimate baseline)
    logger.info(f"\n  ── Phase 2d: Direct LR on raw+d1+d2 flatten ──")
    X_flat = X_3ch.reshape(len(X_3ch), -1)
    results["lr_flatten"] = evaluate_linear_probe(
        X_flat, binary_labels, cancer_type_labels, sample_ids)
    logger.info(f"  LR flatten: AUC={results['lr_flatten']['auc']:.4f}, "
                f"F1-type={results['lr_flatten']['f1_type']:.4f}")

    # Summary
    report = {"timestamp": datetime.now().isoformat(), "data": args.data,
              "duration_sec": (datetime.now() - t0).total_seconds(),
              "pretrain_epochs": args.pretrain_epochs, "temperature": args.temperature,
              "results": results}
    with open(OUTPUT_DIR / f"contrastive_report_{args.data}.json", "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n{'='*64}")
    logger.info(f"  CONTRASTIVE RESULTS ({args.data.upper()})")
    logger.info(f"{'='*64}")
    logger.info(f"  {'Method':<25} {'AUC':>8} {'F1-bin':>8} {'F1-type':>8}")
    logger.info(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8}")
    for name, r in results.items():
        logger.info(f"  {name:<25} {r['auc']:>8.4f} {r['f1_bin']:>8.4f} {r['f1_type']:>8.4f}")
    logger.info(f"{'='*64}")
    logger.info(f"  Duration: {(datetime.now() - t0).total_seconds():.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
