"""External validation of STK-V2 (usersnet v1.0.0) on the Boramae 20260602 urine cohort.

Screening (cancer vs non-cancer): BPRO = cancer (positive), BNOR = normal (negative).
Cancer Type ID: among cancer-flagged samples, which type is predicted (expect PRO).

Aggregation: sample-level via predict_patient (mean of per-replicate probabilities +
majority vote). The instrument-native ``_ave`` files are excluded; raw _1.._5 reps used.
No clinical features are supplied (pure-spectra / sers_only; no sex masking).

NOTE: this is EXTERNAL validation (BPRO/BNOR are not in the training set). Both classes
come from the same hospital (Boramae), so the test set itself has no cross-hospital
confound. Sample sizes are small (esp. normals) — read counts/CIs, not just rates.
"""

import argparse
import glob
import json
import math
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # silence sklearn version-mismatch noise (loads fine)

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "deployment"))

DATA = Path(
    os.environ.get(
        "SERS_BORAMAE_DATA_DIR",
        REPO / "data" / "Thermo" / "20260602_Urine test",
    )
)
MODEL_DIR = REPO / "artifacts" / "usersnet" / "v1.0.0"
OUT = REPO / "results" / "boramae_20260602_validation"
GROUPS = {"BNOR": 0, "BPRO": 1}  # BPRO = cancer (positive)
# Sex biological constraint (type-ID only): BPRO = prostate -> male (masks OVA etc.).
# BNOR normal-control sex unknown -> left unconstrained (only affects spurious FP type labels).
SEX = {"BNOR": None, "BPRO": "M"}
MODE = "balanced"


def group_samples(grp: str) -> dict[str, dict[str, str]]:
    """Return {sample_id: {replicate: filepath}} for a group folder."""
    files = sorted(glob.glob(os.path.join(DATA, grp, "*.CSV")))
    samples: dict[str, dict[str, str]] = defaultdict(dict)
    for f in files:
        stem = os.path.basename(f)[:-4]  # strip .CSV
        sample, rep = stem.rsplit("_", 1)
        samples[sample][rep] = f
    return samples


def print_grouping() -> dict[str, dict[str, dict[str, str]]]:
    """Print and return the parsed grouping; flag any non-standard replicate sets."""
    parsed = {}
    for grp in GROUPS:
        samples = group_samples(grp)
        parsed[grp] = samples
        reps_seen = sorted({r for s in samples.values() for r in s})
        n_files = sum(len(s) for s in samples.values())
        print(
            f"=== {grp}: {n_files} files, {len(samples)} samples; rep suffixes seen = {reps_seen}"
        )
        anomalies = 0
        for s in sorted(samples):
            non_ave = sorted(r for r in samples[s] if r != "ave")
            if non_ave != ["1", "2", "3", "4", "5"]:
                print(f"   anomaly  {s}: reps={sorted(samples[s])}")
                anomalies += 1
        usable = sum(
            1
            for s in samples
            if sorted(r for r in samples[s] if r != "ave") == ["1", "2", "3", "4", "5"]
        )
        print(f"   usable samples (exactly _1.._5): {usable} | anomalies: {anomalies}")
        print(f"   first 5 sample ids: {sorted(samples)[:5]}")
    return parsed


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-only", action="store_true", help="Only print grouping, then exit.")
    args = ap.parse_args()

    parsed = print_grouping()
    if args.check_only:
        return

    from sers_predict import StackingPredictor  # noqa: E402

    print(f"\nLoading model: {MODEL_DIR}")
    predictor = StackingPredictor(artifact_dir=MODEL_DIR)

    rows = []
    for grp, y_true in GROUPS.items():
        samples = parsed[grp]
        for s in sorted(samples):
            reps = sorted(r for r in samples[s] if r != "ave")
            if reps != ["1", "2", "3", "4", "5"]:
                continue  # skip non-standard sets (reported as anomalies above)
            files = [samples[s][r] for r in reps]
            res = predictor.predict_patient(files, sex=SEX[grp], mode=MODE, instrument="thermo")
            status = res.get("status")
            if status != "ok":
                rows.append(
                    {
                        "group": grp,
                        "sample": s,
                        "y_true": y_true,
                        "status": status,
                        "prob": None,
                        "y_pred": None,
                        "type_pred": None,
                        "qc_pass": 0,
                    }
                )
                print(f"  [{grp}] {s}: status={status} ({res.get('message', '')})")
                continue
            pd_ = res["patient_decision"]
            prob = pd_["model_probability_mean"]
            y_pred = int(bool(pd_["cancer_detected"]))
            type_pred = res.get("cancer_type_prediction")
            qc_pass = res["qc_summary"]["passed"]
            rows.append(
                {
                    "group": grp,
                    "sample": s,
                    "y_true": y_true,
                    "status": "ok",
                    "prob": prob,
                    "y_pred": y_pred,
                    "type_pred": type_pred,
                    "qc_pass": qc_pass,
                }
            )
            print(
                f"  [{grp}] {s}: prob={prob:.4f} pred={'CANCER' if y_pred else 'normal'} "
                f"type={type_pred} qc={qc_pass}/5 vote={pd_['majority_vote']}"
            )

    # ---- assemble ----
    OUT.mkdir(parents=True, exist_ok=True)
    ok = [r for r in rows if r["status"] == "ok"]
    errors = [r for r in rows if r["status"] != "ok"]

    import csv

    with open(OUT / "per_sample_results.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "group",
                "sample",
                "y_true",
                "y_pred",
                "prob",
                "type_pred",
                "qc_pass",
                "status",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    y_true = np.array([r["y_true"] for r in ok])
    y_pred = np.array([r["y_pred"] for r in ok])
    probs = np.array([r["prob"] for r in ok])

    from sklearn.metrics import confusion_matrix, roc_auc_score

    # Screening CM: labels [0=normal, 1=cancer] -> [[TN,FP],[FN,TP]]
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    n_pos, n_neg = int((y_true == 1).sum()), int((y_true == 0).sum())

    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    acc = (tp + tn) / len(ok) if ok else float("nan")
    bal_acc = (sens + spec) / 2
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    try:
        auc = roc_auc_score(y_true, probs)
    except Exception as e:  # noqa: BLE001
        auc = float("nan")
        print(f"AUC could not be computed: {e}")

    sens_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)

    # ---- type-ID (one-row breakdown; not a square CM) ----
    bpro_detected = [r for r in ok if r["group"] == "BPRO" and r["y_pred"] == 1]
    bpro_type_dist = Counter(r["type_pred"] for r in bpro_detected)
    pro_correct = bpro_type_dist.get("PRO", 0)
    bnor_fp_types = Counter(r["type_pred"] for r in ok if r["group"] == "BNOR" and r["y_pred"] == 1)

    # ---- report ----
    lines = []
    lines.append("=" * 64)
    lines.append("STK-V2 (usersnet v1.0.0) — Boramae 20260602 external validation")
    lines.append(
        f"mode = {MODE} (threshold {predictor.operating_modes[MODE]['threshold']}); "
        f"aggregation = mean-prob + majority-vote; sex constraint BPRO=M (type-ID only)"
    )
    lines.append("=" * 64)
    lines.append(
        f"Samples scored OK: {len(ok)}  (cancer/BPRO={n_pos}, normal/BNOR={n_neg}); "
        f"errors/excluded: {len(errors)}"
    )
    lines.append("")
    lines.append("SCREENING confusion matrix (counts):")
    lines.append("                 pred normal   pred cancer")
    lines.append(f"  true normal       TN={tn:<8}   FP={fp}")
    lines.append(f"  true cancer       FN={fn:<8}   TP={tp}")
    lines.append("")
    lines.append("SCREENING metrics (with denominators):")
    lines.append(
        f"  Sensitivity (recall) = {sens:.3f}  = {tp}/{tp + fn}   95% CI {sens_ci[0]:.2f}-{sens_ci[1]:.2f}"
    )
    lines.append(
        f"  Specificity          = {spec:.3f}  = {tn}/{tn + fp}   95% CI {spec_ci[0]:.2f}-{spec_ci[1]:.2f}  "
        f"** underpowered: n_normal={n_neg} **"
    )
    lines.append(f"  Accuracy             = {acc:.3f}  = {tp + tn}/{len(ok)}")
    lines.append(f"  Balanced accuracy    = {bal_acc:.3f}")
    lines.append(f"  F1 (cancer)          = {f1:.3f}")
    lines.append(f"  ROC-AUC (mean prob)  = {auc:.3f}  (threshold-free; n={len(ok)})")
    lines.append(
        f"  PPV = {ppv:.3f} = {tp}/{tp + fp} | NPV = {npv:.3f} = {tn}/{tn + fn}   "
        f"** prevalence-distorted: this set is {n_pos / len(ok) * 100:.0f}% cancer, NOT screening prevalence **"
    )
    lines.append("")
    lines.append("CANCER TYPE ID (one-row breakdown — only PRO present in this cohort):")
    lines.append(f"  BPRO flagged cancer (eligible for typing): {len(bpro_detected)}/{n_pos}")
    lines.append(f"    predicted-type distribution: {dict(bpro_type_dist)}")
    lines.append(
        f"    typed as PRO (correct): {pro_correct}/{len(bpro_detected)}"
        + (f" = {pro_correct / len(bpro_detected):.3f}" if bpro_detected else "")
    )
    lines.append(f"  BNOR false-positives' assigned type (spurious): {dict(bnor_fp_types)}")
    if errors:
        lines.append("")
        lines.append(
            f"Excluded ({len(errors)}): "
            + ", ".join(f"{r['group']}/{r['sample']}({r['status']})" for r in errors)
        )
    report = "\n".join(lines)
    print("\n" + report)
    (OUT / "metrics_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "model": "usersnet/v1.0.0 (STK-V2)",
        "mode": MODE,
        "n_ok": len(ok),
        "n_cancer_BPRO": n_pos,
        "n_normal_BNOR": n_neg,
        "n_excluded": len(errors),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
        "sensitivity": sens,
        "sensitivity_ci95": sens_ci,
        "specificity": spec,
        "specificity_ci95": spec_ci,
        "specificity_note": "underpowered",
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "f1": f1,
        "auc": auc,
        "ppv": ppv,
        "npv": npv,
        "prevalence_note": "PPV/NPV distorted; set is not screening prevalence",
        "type_id": {
            "bpro_detected": len(bpro_detected),
            "type_dist": dict(bpro_type_dist),
            "pro_correct": pro_correct,
            "bnor_fp_types": dict(bnor_fp_types),
        },
    }
    (OUT / "metrics_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ---- figures ----
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred normal", "pred cancer"])
    ax.set_yticks([0, 1], ["true normal", "true cancer"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=18,
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    ax.set_title(f"Screening CM — STK-V2 @ {MODE}\nBoramae 20260602 (n={len(ok)})")
    fig.tight_layout()
    fig.savefig(OUT / "screening_confusion_matrix.png", dpi=200)
    plt.close(fig)

    print(f"\nSaved to: {OUT}")
    print("  - per_sample_results.csv")
    print("  - metrics_report.txt / metrics_summary.json")
    print("  - screening_confusion_matrix.png")


if __name__ == "__main__":
    main()
