"""
boramae_stkv2_plots.py
======================
STK-V2 stacking (Cancer Screening) 결과 figure.

기존 boramae_plots.py 스타일에 맞춤:
  - save_plot() 헬퍼(PNG+PDF, 300 dpi) 재사용
  - COLORS / FIG_DIR / LABELS 를 boramae_data 에서 import
  - suptitle "FigNN. ..." 규칙 유지

노트북/스크립트에서 사용 예
--------------------------
    from boramae_stkv2_plots import plot_stkv2_screening_results
    plot_stkv2_screening_results(y_screen, oof, fold_metrics)

의존:
    y_true        : (n,) 0/1 라벨
    oof_proba     : (n,) nested-CV OOF cancer probability
    fold_metrics  : run_nested_cv 가 반환하는 [(auc, bacc), ...] (선택)
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

# ── boramae 공통 자원 재사용 (없으면 fallback) ───────────────────────────────
try:
    from boramae_data import COLORS, FIG_DIR
except Exception:  # 독립 실행/테스트용 fallback
    from pathlib import Path
    FIG_DIR = Path("./figures")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    COLORS = {
        "Prostate cancer": "#B2182B",
        "Control": "#4D4D4D",
        "Biopsy-negative": "#1B9E77",
    }

from sklearn.metrics import (
    roc_curve, roc_auc_score, confusion_matrix,
    balanced_accuracy_score, precision_recall_curve, average_precision_score,
)

DECISION_THRESHOLD = 0.4  # STK-V2 artifact cancer threshold

_CANCER_COLOR = COLORS.get("Prostate cancer", "#B2182B")
_NONCANCER_COLOR = COLORS.get("Control", "#4D4D4D")


def save_plot(name: str) -> None:
    """boramae_plots.save_plot 와 동일: PNG + PDF, 300 dpi."""
    for ext in ("png", "pdf"):
        plt.savefig(FIG_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close()


def plot_stkv2_screening_results(
    y_true: np.ndarray,
    oof_proba: np.ndarray,
    fold_metrics: list[tuple[float, float]] | None = None,
    threshold: float = DECISION_THRESHOLD,
    fig_name: str = "fig06_stkv2_screening_results",
    title: str = "Fig06. STK-V2 stacking — Cancer Screening (nested subject-level CV)",
) -> dict:
    """
    STK-V2 스크리닝 결과를 4-panel figure로 저장한다.
      (1) ROC curve            (2) Confusion matrix (thr)
      (3) Probability 분포      (4) Fold별 AUC / balanced-acc bar
    반환: 주요 지표 dict
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(oof_proba, dtype=float)
    pred = (proba >= threshold).astype(int)

    auc = roc_auc_score(y_true, proba)
    bacc = balanced_accuracy_score(y_true, pred)
    ap = average_precision_score(y_true, proba)
    cm = confusion_matrix(y_true, pred)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── (1) ROC ──────────────────────────────────────────────
    fpr, tpr, _ = roc_curve(y_true, proba)
    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=_CANCER_COLOR, lw=2.0, label=f"ROC (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1.0, ls="--")
    ax.set_xlabel("1 - Specificity (FPR)")
    ax.set_ylabel("Sensitivity (TPR)")
    ax.set_title("ROC — Cancer vs Non-cancer")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2)

    # ── (2) Confusion matrix ─────────────────────────────────
    ax = axes[0, 1]
    im = ax.imshow(cm, cmap="Reds")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-cancer", "Cancer"])
    ax.set_yticklabels(["Non-cancer", "Cancer"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (thr={threshold:.2f})")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # ── (3) Probability 분포 ─────────────────────────────────
    ax = axes[1, 0]
    ax.hist(proba[y_true == 0], bins=20, alpha=0.55,
            color=_NONCANCER_COLOR, label="Non-cancer")
    ax.hist(proba[y_true == 1], bins=20, alpha=0.55,
            color=_CANCER_COLOR, label="Cancer")
    ax.axvline(threshold, color="#333333", ls="--", lw=1.2,
               label=f"threshold={threshold:.2f}")
    ax.set_xlabel("OOF cancer probability")
    ax.set_ylabel("Subjects")
    ax.set_title("Predicted probability distribution")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)

    # ── (4) Fold별 metric ────────────────────────────────────
    ax = axes[1, 1]
    if fold_metrics:
        fm = np.asarray(fold_metrics, dtype=float)
        folds = np.arange(1, len(fm) + 1)
        width = 0.38
        ax.bar(folds - width / 2, fm[:, 0], width,
               color=_CANCER_COLOR, label="AUC")
        ax.bar(folds + width / 2, fm[:, 1], width,
               color=_NONCANCER_COLOR, label="Balanced acc")
        ax.axhline(0.5, color="#999999", ls="--", lw=1.0)
        ax.set_xticks(folds)
        ax.set_xlabel("Outer fold")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1)
        ax.set_title(f"Per-fold (AUC {fm[:,0].mean():.3f}±{fm[:,0].std():.3f})")
        ax.legend(frameon=False)
    else:
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ax.plot(rec, prec, color=_CANCER_COLOR, lw=2.0,
                label=f"PR (AP={ap:.3f})")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall")
        ax.legend(frameon=False, loc="lower left")
    ax.grid(alpha=0.2)

    fig.suptitle(title, fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_plot(fig_name)

    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": float(auc),
        "average_precision": float(ap),
        "balanced_accuracy": float(bacc),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else float("nan"),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else float("nan"),
        "ppv": float(tp / (tp + fp)) if (tp + fp) else float("nan"),
        "npv": float(tn / (tn + fn)) if (tn + fn) else float("nan"),
        "confusion_matrix": cm.tolist(),
        "figure_png": str(FIG_DIR / f"{fig_name}.png"),
    }


if __name__ == "__main__":
    # 합성 데이터 렌더링 자가 점검
    rng = np.random.default_rng(0)
    n = 113
    y = (rng.random(n) < 0.38).astype(int)   # cancer ~43
    proba = np.clip(0.35 + 0.25 * y + rng.normal(0, 0.2, n), 0, 1)
    folds = [(0.60, 0.61), (0.58, 0.59), (0.63, 0.60), (0.55, 0.57), (0.59, 0.62)]
    result = plot_stkv2_screening_results(y, proba, folds)
    print("saved:", result["figure_png"])
    print("AUC=%.3f  bacc=%.3f  sens=%.3f  spec=%.3f"
          % (result["auc"], result["balanced_accuracy"],
             result["sensitivity"], result["specificity"]))
