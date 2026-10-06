"""Exploratory probe: 1D-conv denoising autoencoder (AE) on replicate spectra + LR, KAO 1,630 cohort.

Question: does denoising individual replicate spectra with an AE (trained without labels) change the
downstream LR classification performance compared with the current subject-mean pipeline?

AE training pairs (no labels): input = one replicate spectrum (933 pts), target = the mean of the OTHER
4 replicates of the same subject (leave-one-out mean; the replicate itself is never in its target).

Leakage control: for every (seed, fold) the AE is trained ONLY on replicates of the training-fold
subjects (10% of those subjects held out for early stopping). Test-fold subjects never enter AE
training. The trained AE is then applied to replicates of both training and test fold subjects.
5 seeds x 5 folds = 25 AE trainings.

Conditions (same seed, same fold, same LR protocol as scripts/analysis/dscf_encoder_probe_kao1630.py):
  A  raw replicates -> subject mean -> LR            (current pipeline, reference)
  B  AE(replicate) for each replicate -> subject mean -> LR   (main comparison)
  C  subject mean (as in A) -> AE once -> LR         (denoise after averaging, reference only)

Tasks:
  task1_cancer_vs_control : 1,200 cancer vs 430 control -> AUC, balanced accuracy
  task2_cancer_type       : cancer subjects only, 7 types -> macro-F1 (balanced accuracy also stored)

CV: StratifiedGroupKFold(5, shuffle=True, random_state=seed), groups = subject id, seeds 42, 7, 123,
2024, 31337. The split is stratified on the 11-way source model_group (7 cancer + 4 control) so that
ONE set of folds (and one AE per fold) serves both tasks; task2 uses the same folds restricted to
cancer subjects. Classifier Pipeline(StandardScaler, LogisticRegression), C picked by inner
StratifiedKFold(3) GridSearchCV on the training fold only (C in {0.01, 0.1, 1}).

Noise-estimation diagnostics on test-fold subjects (per seed x fold):
  within-subject RMSE (replicate vs LOO mean) before / after AE, and reduction ratio
  between-subject variance (mean over wavenumbers of the across-subject variance of subject means)
  before (A) / after (B, C), and retention ratio
  residual RMS per wavenumber (before / after), pooled over folds and averaged over seeds
  training curves (epoch, train/val loss) per seed x fold

Stop conditions: non-finite loss, GPU OOM, or (first seed, first fold) within-subject RMSE after AE
larger than before.

Run from repo root (conda env sers-analysis):

    python scripts/analysis/denoising_ae_probe_kao1630.py

Everything is computed locally; nothing is sent to external services.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
KAO_CSV = ROOT / "results/kao_20260610_updated_cohort/processed_spectra.csv"
OUT_DIR = ROOT / "results/denoising_ae_probe_kao1630_20260917"

CANCER_GROUPS = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
CONTROL_GROUPS = ["NOR", "DIA", "HBP", "H.D."]
# same merge as scripts/analysis/learning_curve_sample_size.py
SOURCE_TO_MODEL = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}

SEEDS = [42, 7, 123, 2024, 31337]
C_GRID = [0.01, 0.1, 1.0]
TASKS = ["task1_cancer_vs_control", "task2_cancer_type"]
CONDS = ["A", "B", "C"]

# ---- AE hyper-parameters: fixed, not tuned (exploratory) ----
AE_CFG = dict(
    channels=[16, 32, 32, 32],  # encoder conv channels (decoder mirrors them)
    kernel=7, stride=2, latent_dim=48,
    activation="ReLU",
    lr=1e-3, weight_decay=0.0, batch_size=128, max_epochs=100,
    val_subject_fraction=0.10, early_stopping_patience=10,
    loss="MSE (in globally standardized units)",
    input_scaling="global scalar standardization: subtract the mean and divide by the SD of ALL values of the "
                  "training-fold replicate matrix (one scalar each, no per-wavenumber scaling); AE outputs are "
                  "mapped back to the original scale before averaging / LR / diagnostics",
)

log = logging.getLogger("dae_probe")


class StopRun(RuntimeError):
    pass


# ----------------------------------------------------------------------------- data
def load_kao():
    """Return replicate tensor R (n_subj, 5, n_wn), subject means X, wavenumbers, labels, ids, info."""
    df = pd.read_csv(KAO_CSV)
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    wn = np.array([float(c[2:]) for c in feat_cols])
    df["model_group"] = df["group"].map(lambda g: SOURCE_TO_MODEL.get(g, g))
    subj = df.groupby(["group", "sample_id"], sort=False)
    n_rep = subj.size().to_numpy()
    if not (n_rep == n_rep[0]).all():
        raise StopRun(f"unequal replicate counts per subject: {pd.Series(n_rep).value_counts().to_dict()}")
    n_r = int(n_rep[0])
    R = df[feat_cols].to_numpy(dtype=np.float64).reshape(-1, n_r, len(feat_cols))  # rows are grouped by subject
    # sanity: the reshape must follow the same subject order as the groupby
    first_idx = subj.head(1).index.to_numpy()
    assert (first_idx == np.arange(0, len(df), n_r)).all(), "rows are not contiguous per subject"
    X = R.mean(axis=1)
    g = subj["model_group"].first().to_numpy()
    src = subj["group"].first().to_numpy()
    ids = np.array([f"{a}_{b}" for a, b in subj.groups.keys()])
    info = {
        "csv": str(KAO_CSV.relative_to(ROOT)),
        "n_rows": int(len(df)),
        "n_wavenumber_points": len(feat_cols),
        "wavenumber_range": [float(wn.min()), float(wn.max())],
        "n_subjects": int(len(X)),
        "replicates_per_subject": {str(k): int(v) for k, v in pd.Series(n_rep).value_counts().items()},
        "rows_by_source_group": {k: int(v) for k, v in df["group"].value_counts().items()},
        "subjects_by_source_group": {k: int(v) for k, v in pd.Series(src).value_counts().items()},
        "subjects_by_model_group": {k: int(v) for k, v in pd.Series(g).value_counts().items()},
        "n_cancer": int(np.isin(g, CANCER_GROUPS).sum()),
        "n_control": int(np.isin(g, CONTROL_GROUPS).sum()),
        "value_range_replicates": [float(R.min()), float(R.max())],
        "value_range_subject_mean": [float(X.min()), float(X.max())],
        "per_row_mean_sd_first3": {"mean": R.reshape(-1, R.shape[2]).mean(1)[:3].round(6).tolist(),
                                   "sd": R.reshape(-1, R.shape[2]).std(1)[:3].round(6).tolist()},
    }
    return R, X, wn, g, ids, info


def loo_targets(R):
    """Leave-one-out mean over replicates: target[i, r] = mean of R[i, s] for s != r."""
    n_r = R.shape[1]
    return (R.sum(axis=1, keepdims=True) - R) / (n_r - 1)


# ----------------------------------------------------------------------------- AE
def build_ae(n_wn, cfg, torch):
    nn = torch.nn
    act = getattr(nn, cfg["activation"])
    k, s, p = cfg["kernel"], cfg["stride"], cfg["kernel"] // 2
    chans = [1] + list(cfg["channels"])
    lengths = [n_wn]
    enc = []
    for cin, cout in zip(chans[:-1], chans[1:]):
        enc += [nn.Conv1d(cin, cout, k, stride=s, padding=p), act()]
        lengths.append((lengths[-1] + 2 * p - k) // s + 1)
    flat = chans[-1] * lengths[-1]
    dec = []
    rch = chans[::-1]
    for i, (cin, cout) in enumerate(zip(rch[:-1], rch[1:])):
        lin, lout = lengths[::-1][i], lengths[::-1][i + 1]
        op = lout - ((lin - 1) * s - 2 * p + k)
        assert 0 <= op < s, (lin, lout, op)
        dec.append(nn.ConvTranspose1d(cin, cout, k, stride=s, padding=p, output_padding=op))
        if i < len(rch) - 2:
            dec.append(act())

    class ConvDAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(*enc)
            self.to_latent = nn.Linear(flat, cfg["latent_dim"])
            self.from_latent = nn.Linear(cfg["latent_dim"], flat)
            self.dec = nn.Sequential(*dec)
            self.c_last, self.l_last = chans[-1], lengths[-1]

        def forward(self, x):  # x: (B, L)
            h = self.enc(x[:, None, :]).flatten(1)
            z = self.to_latent(h)
            h = self.from_latent(z).view(-1, self.c_last, self.l_last)
            return self.dec(h)[:, 0, :]

    model = ConvDAE()
    desc = {"encoder_lengths": lengths, "flatten_dim": flat,
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
            "architecture": str(model)}
    return model, desc


def train_ae(R_tr, cfg, seed, device, torch):
    """Train one AE on replicates of R_tr (n_subj, n_rep, n_wn). Returns model, scaler, curve, info."""
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    n = len(R_tr)
    perm = rng.permutation(n)
    n_val = max(1, int(round(cfg["val_subject_fraction"] * n)))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    mu, sd = float(R_tr.mean()), float(R_tr.std())
    T_tr = loo_targets(R_tr)

    def flat(idx, arr):
        return torch.as_tensor(((arr[idx] - mu) / sd).reshape(-1, arr.shape[2]), dtype=torch.float32, device=device)

    x_fit, t_fit = flat(fit_idx, R_tr), flat(fit_idx, T_tr)
    x_val, t_val = flat(val_idx, R_tr), flat(val_idx, T_tr)
    model, desc = build_ae(R_tr.shape[2], cfg, torch)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    gen = torch.Generator(device="cpu").manual_seed(seed)
    best, best_ep, best_state, bad = np.inf, -1, None, 0
    curve = []
    bs = cfg["batch_size"]
    t0 = time.time()
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train()
        order = torch.randperm(len(x_fit), generator=gen).to(device)
        tot = 0.0
        for s in range(0, len(order), bs):
            b = order[s:s + bs]
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(x_fit[b]), t_fit[b])
            if not torch.isfinite(loss):
                raise StopRun(f"non-finite training loss at epoch {ep}")
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        tr_loss = tot / len(x_fit)
        model.eval()
        with torch.no_grad():
            vl = 0.0
            for s in range(0, len(x_val), 1024):
                vl += float(torch.nn.functional.mse_loss(model(x_val[s:s + 1024]), t_val[s:s + 1024], reduction="sum"))
            val_loss = vl / x_val.numel()
        if not np.isfinite(val_loss):
            raise StopRun(f"non-finite validation loss at epoch {ep}")
        curve.append({"epoch": ep, "train_loss": tr_loss, "val_loss": val_loss})
        if val_loss < best - 1e-7:
            best, best_ep, bad = val_loss, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["early_stopping_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    info = {"n_fit_subjects": int(len(fit_idx)), "n_val_subjects": int(len(val_idx)),
            "n_fit_spectra": int(len(x_fit)), "n_val_spectra": int(len(x_val)),
            "scale_mean": mu, "scale_sd": sd, "epochs_run": len(curve), "best_epoch": best_ep,
            "best_val_loss": float(best), "final_train_loss": curve[-1]["train_loss"],
            "train_seconds": round(time.time() - t0, 1), **desc}
    return model, (mu, sd), curve, info


def apply_ae(model, M, scaler, device, torch, batch=2048):
    """Denoise rows of M (n, n_wn) and return in original scale."""
    mu, sd = scaler
    out = []
    with torch.no_grad():
        for s in range(0, len(M), batch):
            x = torch.as_tensor((M[s:s + batch] - mu) / sd, dtype=torch.float32, device=device)
            out.append(model(x).double().cpu().numpy() * sd + mu)
    Y = np.concatenate(out)
    if not np.isfinite(Y).all():
        raise StopRun("non-finite AE output")
    return Y


# ----------------------------------------------------------------------------- CV / LR
def make_splits(strat, groups, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(strat)), strat, groups))


def fit_predict(Xtr, ytr, Xte, task, seed):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=10000))
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    scoring = "roc_auc" if task == "task1_cancer_vs_control" else "f1_macro"
    gs = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, cv=inner, scoring=scoring,
                      n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)
    return gs.predict_proba(Xte), gs.predict(Xte), gs.best_params_["logisticregression__C"], gs.classes_


def metrics(task, y, proba, pred, classes):
    if task == "task1_cancer_vs_control":
        return {"auc": roc_auc_score(y, proba[:, list(classes).index(1)]),
                "balanced_accuracy": balanced_accuracy_score(y, pred)}
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "balanced_accuracy": balanced_accuracy_score(y, pred)}


def diagnostics(R_te, Rd_te, X_te, Xb_te, Xc_te):
    """Noise-estimation diagnostics on test-fold subjects (original scale)."""
    T = loo_targets(R_te)
    res_before = R_te - T
    res_after = Rd_te - T
    rmse_before = np.sqrt((res_before ** 2).mean(axis=2))  # (n_subj, n_rep)
    rmse_after = np.sqrt((res_after ** 2).mean(axis=2))
    var_a = X_te.var(axis=0, ddof=1).mean()
    var_b = Xb_te.var(axis=0, ddof=1).mean()
    var_c = Xc_te.var(axis=0, ddof=1).mean()
    d = {"within_rmse_before": float(rmse_before.mean()),
         "within_rmse_after": float(rmse_after.mean()),
         "within_rmse_reduction": float(1 - rmse_after.mean() / rmse_before.mean()),
         "within_rmse_after_lt_before_frac_replicates": float((rmse_after < rmse_before).mean()),
         "between_var_A": float(var_a), "between_var_B": float(var_b), "between_var_C": float(var_c),
         "between_var_retention_B": float(var_b / var_a), "between_var_retention_C": float(var_c / var_a),
         "mean_abs_shift_B_minus_A": float(np.abs(Xb_te - X_te).mean()),
         "mean_abs_shift_C_minus_A": float(np.abs(Xc_te - X_te).mean()),
         "mean_abs_diff_B_minus_C": float(np.abs(Xb_te - Xc_te).mean())}
    sq_before = (res_before ** 2).reshape(-1, R_te.shape[2])  # per wavenumber, pooled later
    sq_after = (res_after ** 2).reshape(-1, R_te.shape[2])
    return d, sq_before, sq_after


def summarize(runs: pd.DataFrame, diag: pd.DataFrame):
    out = {"level_used": "seed_pooled_oof (per seed: out-of-fold predictions over 5 folds)"}
    seed_df = runs[runs["level"] == "seed_pooled_oof"]
    for task, tdf in seed_df.groupby("task"):
        mcols = ["auc", "balanced_accuracy"] if task == "task1_cancer_vs_control" else ["macro_f1", "balanced_accuracy"]
        conds = {}
        for cond, cdf in tdf.groupby("condition"):
            conds[cond] = {m: {"mean": float(cdf[m].mean()), "sd": float(cdf[m].std(ddof=1)),
                               "per_seed": {int(s): float(v) for s, v in zip(cdf["seed"], cdf[m])}}
                           for m in mcols}
        fold_df = runs[(runs["level"] == "fold") & (runs["task"] == task)]
        fold_mean = {}
        for cond, cdf in fold_df.groupby("condition"):
            per_seed = cdf.groupby("seed")[mcols].mean()
            fold_mean[cond] = {m: {"mean": float(per_seed[m].mean()), "sd": float(per_seed[m].std(ddof=1))}
                               for m in mcols}
        paired = {}
        for first, second in [("B", "A"), ("C", "A"), ("B", "C")]:
            c = tdf[tdf["condition"] == first].set_index("seed")
            b = tdf[tdf["condition"] == second].set_index("seed")
            key = f"{first}-{second}"
            paired[key] = {}
            for m in mcols:
                d = (c[m] - b.loc[c.index, m])
                paired[key][m] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)),
                                  "n_seeds_first_better": int((d > 0).sum()), "n_seeds": int(len(d)),
                                  "per_seed": {int(s): float(v) for s, v in d.items()}}
        out[task] = {"conditions_seed_pooled_oof": conds,
                     "conditions_mean_of_fold_metrics": fold_mean,
                     "paired_differences_seed_pooled_oof": paired}
    dcols = [c for c in diag.columns if c not in ("seed", "fold")]
    per_seed = diag.groupby("seed")[dcols].mean()
    out["ae_diagnostics_test_fold"] = {
        "level": "per (seed, fold) on test-fold subjects; seed value = mean over 5 folds; mean/sd over seeds",
        **{c: {"mean": float(per_seed[c].mean()), "sd": float(per_seed[c].std(ddof=1)),
               "per_seed": {int(s): float(v) for s, v in per_seed[c].items()}} for c in dcols},
    }
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--max-epochs", type=int, default=AE_CFG["max_epochs"])
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="subset of seeds (smoke tests only)")
    args = ap.parse_args()
    AE_CFG["max_epochs"] = args.max_epochs
    SEEDS[:] = args.seeds

    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(args.out / "run.log", mode="a", encoding="utf-8"),
                                  logging.StreamHandler()])
    import torch

    meta = {
        "script": str(Path(__file__).relative_to(ROOT)),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": platform.python_version(), "torch": torch.__version__,
        "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__,
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "seeds": SEEDS,
        "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed); groups=subject id (group_sampleid); "
              "stratified on the 11-way model_group (7 cancer + 4 control) so that one fold set / one AE per fold "
              "serves both tasks; task2 = same folds restricted to cancer subjects. (Deviation from the DSCF probe, "
              "which stratified each task on its own label.)",
        "classifier": "Pipeline(StandardScaler, LogisticRegression(L2, lbfgs, max_iter=10000, class_weight=None))",
        "C_selection": f"GridSearchCV over C={C_GRID}, inner StratifiedKFold(3, shuffle=True, random_state=seed) on the training fold only; scoring roc_auc (task1) / f1_macro (task2); refit on full training fold",
        "aggregation": "subject mean of replicates; subject key = (source group, sample_id); CPAN/YPAN->PAN, YNOR->NOR",
        "reported_level": "seed-level metrics from pooled out-of-fold predictions; fold-level rows also in runs.csv",
        "task1_threshold": "balanced accuracy from predict() (argmax, i.e. p>=0.5)",
        "ae": {**AE_CFG,
               "training_pairs": "input = one replicate (933 pts); target = mean of the other 4 replicates of the same "
                                 "subject (leave-one-out mean; the replicate itself is never in its own target)",
               "leakage_control": "AE trained per (seed, fold) on training-fold subjects only (10% of them held out "
                                  "as AE validation for early stopping, split by subject with RandomState(seed*100+fold)); "
                                  "test-fold subjects never used in AE training; torch.manual_seed(seed*100+fold)",
               "early_stopping": "patience 10 epochs on validation MSE; best-epoch weights restored",
               "application": "B: AE applied to every replicate (train and test subjects) then subject mean; "
                              "C: AE applied once to the raw subject-mean spectrum",
               "denoised_outputs_written_to_disk": False},
        "conditions": {"A": "raw replicates -> subject mean -> LR (reference)",
                       "B": "AE(replicate) -> subject mean -> LR",
                       "C": "subject mean -> AE -> LR"},
        "diagnostics": {
            "within_rmse": "test-fold subjects; RMSE over 933 points between a replicate (before) or its AE reconstruction "
                           "(after) and the leave-one-out mean of the other 4 ORIGINAL replicates; averaged over replicates; "
                           "reduction = 1 - after/before",
            "between_var": "test-fold subjects; per-wavenumber variance (ddof=1) of subject-mean spectra across subjects, "
                           "averaged over wavenumbers; A raw means, B denoised-replicate means, C AE(raw mean); retention = B/A, C/A",
            "residual_by_wavenumber": "RMS over all test-fold replicates of a seed (5 folds pooled = every subject once) "
                                      "of (replicate - LOO mean) before and (AE(replicate) - LOO mean) after; then mean over seeds",
            "ae_training_curves": "epoch-wise train MSE (mean over minibatches) and validation MSE in standardized units",
        },
    }
    try:
        R, X, wn, g, ids, info = load_kao()
    except StopRun as e:
        log.error("STOP: %s", e)
        return 2
    meta["data"] = info
    log.info("data: %s", json.dumps(info, ensure_ascii=False))
    assert info["n_subjects"] == 1630 and info["n_cancer"] == 1200 and info["n_control"] == 430, info
    _, ae_desc = build_ae(R.shape[2], AE_CFG, torch)
    meta["ae"].update({k: ae_desc[k] for k in ("encoder_lengths", "flatten_dim", "n_parameters", "architecture")})
    log.info("AE: %d parameters, encoder lengths %s", ae_desc["n_parameters"], ae_desc["encoder_lengths"])

    meta["limitations"] = [
        "Retrospective cohort: each cancer type was collected at different sites/periods, so site/batch confounding with class labels is possible (task2 especially).",
        "Input is already preprocessed (processed_spectra.csv: SNV-scaled per row, values include negatives), not raw intensity spectra; the AE learns residual replicate-to-replicate variation after that preprocessing.",
        "AE architecture and hyper-parameters fixed a priori (one configuration, no tuning).",
        "Leave-one-out replicate mean as target: the AE target itself still contains replicate noise (4-replicate mean), so the achievable 'after' RMSE floor is not zero.",
        "Task2: BRE has 30 subjects (~6 per test fold); macro-F1 is noisy.",
        "Exploratory analysis; no external test set.",
    ]

    def dump_meta():
        with open(args.out / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    dump_meta()

    y1 = np.isin(g, CANCER_GROUPS).astype(int)
    cancer_mask = np.isin(g, CANCER_GROUPS)
    y2 = g.astype(str)
    strat = g.astype(str)  # 11-way stratification

    # ---- resume support: seed-level checkpoints ----
    p_runs = args.out / "runs_partial.csv"
    p_diag = args.out / "ae_diagnostics_partial.csv"
    p_curve = args.out / "ae_training_curves.csv"
    p_res = args.out / "residual_partial.npz"
    p_fit = args.out / "ae_fit_info_partial.csv"
    rows, diag_rows, curve_rows, fit_rows = [], [], [], []
    res_seed = {}  # seed -> (sum_sq_before, sum_sq_after, n) pooled over folds
    done_seeds = set()
    if p_runs.exists() and p_diag.exists() and p_res.exists():
        prev = pd.read_csv(p_runs, encoding="utf-8-sig")
        done = prev[prev["level"] == "seed_pooled_oof"].groupby("seed").size()
        done_seeds = {int(s) for s, n in done.items() if n == len(TASKS) * len(CONDS)}
        rows = prev[prev["seed"].isin(done_seeds)].to_dict("records")
        dprev = pd.read_csv(p_diag, encoding="utf-8-sig")
        diag_rows = dprev[dprev["seed"].isin(done_seeds)].to_dict("records")
        if p_curve.exists():
            cprev = pd.read_csv(p_curve, encoding="utf-8-sig")
            curve_rows = cprev[cprev["seed"].isin(done_seeds)].to_dict("records")
        if p_fit.exists():
            fprev = pd.read_csv(p_fit, encoding="utf-8-sig")
            fit_rows = fprev[fprev["seed"].isin(done_seeds)].to_dict("records")
        z = np.load(p_res)
        for s in done_seeds:
            if f"before_{s}" in z:
                res_seed[s] = (z[f"before_{s}"], z[f"after_{s}"], int(z[f"n_{s}"]))
        done_seeds = {s for s in done_seeds if s in res_seed}
        meta["resumed_seeds"] = sorted(done_seeds)
        log.info("resuming: seeds already complete %s", sorted(done_seeds))

    def save_ckpt():
        pd.DataFrame(rows).to_csv(p_runs, index=False, encoding="utf-8-sig")
        pd.DataFrame(diag_rows).to_csv(p_diag, index=False, encoding="utf-8-sig")
        pd.DataFrame(curve_rows).to_csv(p_curve, index=False, encoding="utf-8-sig")
        pd.DataFrame(fit_rows).to_csv(p_fit, index=False, encoding="utf-8-sig")
        np.savez(p_res, **{f"before_{s}": v[0] for s, v in res_seed.items()},
                 **{f"after_{s}": v[1] for s, v in res_seed.items()},
                 **{f"n_{s}": v[2] for s, v in res_seed.items()})

    t_all = time.time()
    first_fold_checked = bool(done_seeds)
    try:
        for seed in SEEDS:
            if seed in done_seeds:
                continue
            t_seed = time.time()
            splits = make_splits(strat, ids, seed)
            oof = {(t, c): None for t in TASKS for c in CONDS}
            oof_pred = {}
            classes_seen = {}
            sq_b = np.zeros(R.shape[2])
            sq_a = np.zeros(R.shape[2])
            n_res = 0
            seed_rows, seed_diag = [], []
            for k, (tr, te) in enumerate(splits):
                fseed = seed * 100 + k
                torch.cuda.reset_peak_memory_stats()
                model, scaler, curve, finfo = train_ae(R[tr], AE_CFG, fseed, args.device, torch)
                finfo.pop("architecture")
                finfo["peak_gpu_mem_MB"] = round(torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                curve_rows += [{"seed": seed, "fold": k, **c} for c in curve]
                fit_rows.append({"seed": seed, "fold": k, "n_train_subjects": len(tr), "n_test_subjects": len(te), **finfo})
                # apply to all replicates (train + test subjects) and to raw subject means
                Rd = apply_ae(model, R.reshape(-1, R.shape[2]), scaler, args.device, torch).reshape(R.shape)
                Xb = Rd.mean(axis=1)
                Xc = apply_ae(model, X, scaler, args.device, torch)
                del model
                torch.cuda.empty_cache()
                d, sqb, sqa = diagnostics(R[te], Rd[te], X[te], Xb[te], Xc[te])
                seed_diag.append({"seed": seed, "fold": k, **d})
                sq_b += sqb.sum(axis=0)
                sq_a += sqa.sum(axis=0)
                n_res += len(sqb)
                log.info("seed %5d fold %d | AE %d ep (best %d) %.1fs val %.5f | within RMSE %.4f -> %.4f (%.1f%%) | "
                         "between var retention B %.3f C %.3f", seed, k, finfo["epochs_run"], finfo["best_epoch"],
                         finfo["train_seconds"], finfo["best_val_loss"], d["within_rmse_before"], d["within_rmse_after"],
                         100 * d["within_rmse_reduction"], d["between_var_retention_B"], d["between_var_retention_C"])
                if not first_fold_checked:
                    first_fold_checked = True
                    if d["within_rmse_after"] >= d["within_rmse_before"]:
                        diag_rows += seed_diag
                        save_ckpt()
                        raise StopRun(f"first fold: within-subject RMSE after AE ({d['within_rmse_after']:.5f}) >= before "
                                      f"({d['within_rmse_before']:.5f}) -> AE does not reduce replicate noise")
                feats = {"A": X, "B": Xb, "C": Xc}
                for task in TASKS:
                    if task == "task1_cancer_vs_control":
                        trk, tek, y = tr, te, y1
                    else:
                        trk, tek, y = tr[cancer_mask[tr]], te[cancer_mask[te]], y2
                    for cond in CONDS:
                        Xc_ = feats[cond]
                        t0 = time.time()
                        proba, pred, best_c, classes = fit_predict(Xc_[trk], y[trk], Xc_[tek], task, seed)
                        if oof[(task, cond)] is None:
                            oof[(task, cond)] = np.full((len(y), len(classes)), np.nan)
                            oof_pred[(task, cond)] = np.empty(len(y), dtype=object)
                        oof[(task, cond)][tek] = proba
                        oof_pred[(task, cond)][tek] = pred
                        classes_seen[(task, cond)] = classes
                        m = metrics(task, y[tek], proba, pred, classes)
                        seed_rows.append({"task": task, "condition": cond, "seed": seed, "fold": k, "level": "fold",
                                          "n_train": len(trk), "n_test": len(tek), "n_features": Xc_.shape[1],
                                          "best_C": best_c, **m})
                        log.info("  %s | %s | fold %d | %s | C=%g | %.1fs", task, cond, k,
                                 " ".join(f"{a}={b:.4f}" for a, b in m.items()), best_c, time.time() - t0)
            # pooled OOF per seed
            for task in TASKS:
                y = y1 if task == "task1_cancer_vs_control" else y2
                sel = np.ones(len(y), bool) if task == "task1_cancer_vs_control" else cancer_mask
                for cond in CONDS:
                    pr = oof[(task, cond)][sel]
                    pd_ = oof_pred[(task, cond)][sel].astype(y.dtype)
                    assert np.isfinite(pr).all()
                    m = metrics(task, y[sel], pr, pd_, classes_seen[(task, cond)])
                    seed_rows.append({"task": task, "condition": cond, "seed": seed, "fold": "pooled_oof",
                                      "level": "seed_pooled_oof", "n_train": np.nan, "n_test": int(sel.sum()),
                                      "n_features": X.shape[1], "best_C": np.nan, **m})
                    log.info("%s | %s | seed %5d | pooled OOF %s", task, cond, seed,
                             " ".join(f"{a}={b:.4f}" for a, b in m.items()))
            rows += seed_rows
            diag_rows += seed_diag
            res_seed[seed] = (sq_b, sq_a, n_res)
            save_ckpt()  # 중단 대비: 시드마다 중간 저장
            log.info("seed %d complete in %.1f min -> checkpoint saved", seed, (time.time() - t_seed) / 60)
    except StopRun as e:
        meta["stopped"] = f"stop condition: {e}"
        log.error("STOP: %s", e)
        dump_meta()
        return 2
    except torch.cuda.OutOfMemoryError as e:
        meta["stopped"] = f"stop condition: GPU OOM ({e})"
        log.error("STOP: OOM")
        dump_meta()
        return 2

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out / "runs.csv", index=False, encoding="utf-8-sig")
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(args.out / "ae_diagnostics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(fit_rows).to_csv(args.out / "ae_fit_info.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(curve_rows).to_csv(p_curve, index=False, encoding="utf-8-sig")
    # residual RMS per wavenumber: per seed (pooled over folds) then mean over seeds
    res = pd.DataFrame({"wavenumber": wn})
    for s in SEEDS:
        b, a, n = res_seed[s]
        res[f"rms_before_seed{s}"] = np.sqrt(b / n)
        res[f"rms_after_seed{s}"] = np.sqrt(a / n)
    res["rms_before_mean_over_seeds"] = res[[f"rms_before_seed{s}" for s in SEEDS]].mean(axis=1)
    res["rms_after_mean_over_seeds"] = res[[f"rms_after_seed{s}" for s in SEEDS]].mean(axis=1)
    res["rms_ratio_after_over_before"] = res["rms_after_mean_over_seeds"] / res["rms_before_mean_over_seeds"]
    res.to_csv(args.out / "residual_by_wavenumber.csv", index=False, encoding="utf-8-sig")
    summary = summarize(runs, diag)
    fit = pd.DataFrame(fit_rows)
    summary["ae_fit"] = {"n_trainings": int(len(fit)),
                         "epochs_run": {"mean": float(fit["epochs_run"].mean()), "min": int(fit["epochs_run"].min()),
                                        "max": int(fit["epochs_run"].max())},
                         "best_epoch": {"mean": float(fit["best_epoch"].mean()), "min": int(fit["best_epoch"].min()),
                                        "max": int(fit["best_epoch"].max())},
                         "n_hit_max_epochs": int((fit["epochs_run"] >= AE_CFG["max_epochs"]).sum()),
                         "best_val_loss": {"mean": float(fit["best_val_loss"].mean()), "sd": float(fit["best_val_loss"].std(ddof=1))},
                         "train_seconds": {"mean": float(fit["train_seconds"].mean()), "total": float(fit["train_seconds"].sum())},
                         "peak_gpu_mem_MB_max": float(fit["peak_gpu_mem_MB"].max()),
                         "n_parameters": ae_desc["n_parameters"]}
    with open(args.out / "summary.json", "w", encoding="utf-8-sig") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["total_minutes"] = round((time.time() - t_all) / 60, 1)
    dump_meta()
    log.info("done -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
