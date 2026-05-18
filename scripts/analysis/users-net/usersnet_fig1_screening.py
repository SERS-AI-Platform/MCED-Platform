"""Figure 1 — Cancer screening: 6-model ROC overlay + uSERS-Net binary CM."""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.special import softmax
from sklearn.metrics import auc, confusion_matrix, roc_curve

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "AACR" / "src"))
from nature_style import MODEL_COLORS

RUN_DIR = PROJECT_ROOT / "results" / "runs" / "2026-05-11_usersnet_alpha0.8_ovr_roc"
EXTRA = RUN_DIR / "extra_models"

ALPHA = 0.8
MODELS_ORDER = ["Logistic Regression", "Random Forest", "XGBoost", "CNN1D", "ResNet18", "uSERS-Net"]
N_BOOT = 1000


def boot_ci(y, s, n=N_BOOT, seed=42):
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yt = y[i]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        fpr, tpr, _ = roc_curve(yt, s[i])
        aucs.append(auc(fpr, tpr))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def load_predictions():
    inputs = np.load(RUN_DIR / "resnet18_ckpts" / "fold_inputs.npz", allow_pickle=True)
    bl = inputs["bl"].astype(int)
    ctl = inputs["ctl"]; groups_arr = inputs["groups_arr"]
    sex = np.load(EXTRA / "sex_for_constraint.npy")

    # ResNet18 OOF binary prob: reconstruct from saved fold ckpts via re-inference would take time;
    # instead, use OOF derived during training (saved in fold_summary not full vector) — we need to
    # re-run val predictions. Faster: use IG-fold mapping to load fold_predictions from training stage.
    # The ResNet IG run did NOT save val_binary_prob OOF. Re-run inference now.
    import torch
    import yaml
    sys.path.insert(0, str(PROJECT_ROOT))
    from sers.models._legacy.resnet_v1.model import ModelConfig, SERSCancerDetector
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "analysis"))
    import importlib.util as _iu
    _spec = _iu.spec_from_file_location("legacy_train_resnet",
                                        PROJECT_ROOT / "scripts" / "training" / "_legacy" / "train_resnet.py")
    _legacy = _iu.module_from_spec(_spec); sys.modules["legacy_train_resnet"] = _legacy
    _spec.loader.exec_module(_legacy)
    apply_class_selection = _legacy.apply_class_selection

    with open(PROJECT_ROOT / "config" / "config.yaml", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    L = inputs["X"].shape[1]
    mc = ModelConfig.from_pipeline_config(raw_cfg, n_spectral_features=L)
    mc = apply_class_selection(mc,
                               cancer_types=["PRO", "BRE", "OVA", "LUN", "CRC", "PAN", "BLC"],
                               non_cancer_groups=["NOR", "DIA", "HBP", "H.D."])
    mc_run = ModelConfig(**{**mc.__dict__, "n_spectral_features": L,
                            "learning_rate": 3e-4, "dropout_rate": 0.3, "weight_decay": 1e-3,
                            "batch_size": 32, "n_epochs": 150,
                            "resnet_channels": (64, 128, 256, 512), "encoder_output_dim": 512,
                            "random_state": 42})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = inputs["X"].shape[0]
    rn_bp = np.full(n, np.nan)
    rn_cl = np.full((n, 7), np.nan)
    for fi in range(5):
        ck = torch.load(RUN_DIR / "resnet18_ckpts" / f"fold_{fi}.pt",
                        map_location=device, weights_only=False)
        m = SERSCancerDetector(mc_run).to(device)
        m.load_state_dict(ck["model_state_dict"]); m.eval()
        va = ck["val_idx"]
        with torch.no_grad():
            xb = torch.from_numpy(inputs["X"][va].astype(np.float32)).to(device)
            out = m(xb)
            rn_bp[va] = out["binary_prob"].squeeze(-1).cpu().numpy()
            rn_cl[va] = out["cancer_probs"].cpu().numpy()
        del m; torch.cuda.empty_cache()
    # Apply sex constraint to rn_cl
    pro_idx, ova_idx = 0, 2  # CANCER_TYPES index
    rn_cl[sex == 0, pro_idx] = 0.0
    rn_cl[sex == 1, ova_idx] = 0.0
    row_s = rn_cl.sum(axis=1, keepdims=True)
    nz = row_s.squeeze() > 0
    rn_cl[nz] = rn_cl[nz] / row_s[nz]

    # Load base model OOF
    lr = np.load(EXTRA / "logistic_regression_oof.npz")
    rf = np.load(EXTRA / "random_forest_oof.npz")
    xg = np.load(EXTRA / "xgboost_oof.npz")
    cn = np.load(EXTRA / "cnn1d_oof.npz")

    # uSERS-Net binary = α·LR + (1-α)·ResNet18 binary
    usersnet_bp = ALPHA * lr["val_binary_prob"] + (1 - ALPHA) * rn_bp
    usersnet_cl = ALPHA * lr["val_cancer_logits"] + (1 - ALPHA) * rn_cl

    preds = {
        "Logistic Regression": (lr["val_binary_prob"], lr["val_cancer_logits"]),
        "Random Forest":       (rf["val_binary_prob"], rf["val_cancer_logits"]),
        "XGBoost":             (xg["val_binary_prob"], xg["val_cancer_logits"]),
        "CNN1D":               (cn["val_binary_prob"], cn["val_cancer_logits"]),
        "ResNet18":            (rn_bp, rn_cl),
        "uSERS-Net":           (usersnet_bp, usersnet_cl),
    }
    np.savez_compressed(RUN_DIR / "all_models_oof.npz",
                        **{f"{k}_bp": v[0] for k, v in preds.items()},
                        **{f"{k}_cl": v[1] for k, v in preds.items()},
                        bl=bl, ctl=ctl, groups_arr=groups_arr, sex=sex)
    return preds, bl, ctl, groups_arr


def main():
    preds, bl, ctl, groups_arr = load_predictions()

    fig = plt.figure(figsize=(13, 5.5))
    gs = GridSpec(1, 2, width_ratios=[1, 0.8], wspace=0.3)
    ax_roc, ax_cm = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    rows = []
    for name in MODELS_ORDER:
        bp, _ = preds[name]
        valid = ~np.isnan(bp)
        fpr, tpr, _ = roc_curve(bl[valid], bp[valid])
        a = auc(fpr, tpr)
        lo, hi = boot_ci(bl[valid], bp[valid])
        color = MODEL_COLORS[name]
        lw = 2.6 if name == "uSERS-Net" else 1.6
        ax_roc.plot(fpr, tpr, color=color, lw=lw,
                    label=f"{name}  AUC={a:.3f} [{lo:.3f}, {hi:.3f}]")
        rows.append({"model": name, "auc": a, "ci_lo": lo, "ci_hi": hi})
    ax_roc.plot([0, 1], [0, 1], "k--", lw=0.6, alpha=0.4)
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.02)
    ax_roc.set_xlabel("1 − Specificity"); ax_roc.set_ylabel("Sensitivity")
    leg = ax_roc.legend(loc="lower right", fontsize=8.5)
    for text in leg.get_texts():
        if "uSERS-Net" in text.get_text():
            text.set_fontweight("bold")
    ax_roc.grid(alpha=0.3)

    # uSERS-Net CM at Youden threshold
    bp_us, _ = preds["uSERS-Net"]
    fpr, tpr, thr = roc_curve(bl, bp_us)
    youden = thr[np.argmax(tpr - fpr)]
    pred_bin = (bp_us >= youden).astype(int)
    cm = confusion_matrix(bl, pred_bin)
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn); spec = tn / (tn + fp); acc = (tp + tn) / len(bl)
    im = ax_cm.imshow(cm, cmap="Reds", aspect="auto")
    for (i, j), v in np.ndenumerate(cm):
        ax_cm.text(j, i, f"{v}", ha="center", va="center",
                   color="white" if v > cm.max() * 0.5 else "black", fontsize=14, fontweight="bold")
    ax_cm.set_xticks([0, 1]); ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["Non-Cancer", "Cancer"]); ax_cm.set_yticklabels(["Non-Cancer", "Cancer"])
    ax_cm.set_xlabel("Predicted"); ax_cm.set_ylabel("True")
    plt.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)
    plt.tight_layout()
    out_png = RUN_DIR / "fig1_screening_roc_cm.png"
    plt.savefig(out_png, dpi=250, bbox_inches="tight"); plt.close()
    pd.DataFrame(rows).to_csv(RUN_DIR / "fig1_binary_auc.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame({"true": bl, "pred": pred_bin, "score": bp_us}).to_csv(
        RUN_DIR / "fig1_usersnet_binary_predictions.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
