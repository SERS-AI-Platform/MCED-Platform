"""Exploratory probe: DSCF pretrained encoder features + LR on the KAO 1,630 cohort.

DSCF = Deep Spectral Component Filtering (Xue et al., Nat Mach Intell 2025).
Code: github.com/streamflowmaster/Deep-Spectral-Component-Filtering-DSCF- (GPL-3.0)
Weights: figshare 28130648 ``SiT_PPS_tiny.pt`` (CC BY 4.0)

Conditions (all share the same folds per task x seed):
  A   LR on the original 933-point grid (processed_spectra.csv values)
  B   LR on 512-point linear interpolation + per-sample min-max  (main reference for C)
  C_* LR on DSCF encoder features: the B matrix fed through patch_embed -> enc1..enc4
      (eval mode, no masking), each stage mean-pooled over the length axis.
      Every stage is reported as its own condition (C_enc1..C_enc4).
      Model is built from the checkpoint's stored model_args (outplanes=100) and loaded strictly.
  D_* control: identical architecture, random initialization (torch.manual_seed = CV seed),
      same input / stages / pooling / folds / C selection as C (D_enc1..D_enc4).

Tasks:
  task1_cancer_vs_control : 1,200 cancer vs 430 control (NOR, DIA, HBP, H.D.) -> AUC, balanced acc
  task2_cancer_type       : cancer subjects only, 7 types -> macro-F1 (balanced acc also stored)

CV: StratifiedGroupKFold(5, shuffle=True, random_state=seed), groups = subject id,
seeds 42, 7, 123, 2024, 31337. Pipeline StandardScaler + LogisticRegression; C picked by
inner StratifiedKFold(3) GridSearchCV on the training fold only (same grid for all conditions).

Run from repo root (conda env sers-analysis):

    python scripts/analysis/dscf_encoder_probe_kao1630.py

Everything is computed locally; nothing is sent to external services.
"""
from __future__ import annotations

import argparse
import hashlib
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
OUT_DIR = ROOT / "results/dscf_encoder_probe_kao1630_20260915"
SCRATCH = Path("/home/user/workspace/_scratch/2026-09-15-dscf-kao1630")

CANCER_GROUPS = ["PRO", "LUN", "CRC", "PAN", "OVA", "BRE", "BLC"]
CONTROL_GROUPS = ["NOR", "DIA", "HBP", "H.D."]
# same merge as scripts/analysis/learning_curve_sample_size.py
SOURCE_TO_MODEL = {"CPAN": "PAN", "YPAN": "PAN", "YNOR": "NOR"}

SEEDS = [42, 7, 123, 2024, 31337]
C_GRID = [0.01, 0.1, 1.0]
SIG_LEN = 512
WEIGHT_MD5 = "aee403eb46487f12d48f0b1d11ec3e8e"
# config of the public code (customized_task/forward_protocol.py, scale='tiny'); reference only —
# the model is built from the checkpoint's stored model_args (differences recorded in metadata)
DSCF_CFG = dict(
    inplanes=1, outplanes=1, encoder_name="SiT", decoder_name="PPS", embed_dim=64,
    layers=[3, 6, 8, 3], d_layers=[3, 6, 8, 3], sig_len=SIG_LEN, patch_size=4,
    mask=False,  # inference: no random masking (mask has no parameters)
)

log = logging.getLogger("dscf_probe")


class StopRun(RuntimeError):
    pass


# ----------------------------------------------------------------------------- data
def load_kao():
    df = pd.read_csv(KAO_CSV)
    feat_cols = [c for c in df.columns if c.startswith("x_")]
    wn = np.array([float(c[2:]) for c in feat_cols])
    df["model_group"] = df["group"].map(lambda g: SOURCE_TO_MODEL.get(g, g))
    subj = df.groupby(["group", "sample_id"], sort=False)
    X = subj[feat_cols].mean().to_numpy(dtype=np.float64)
    g = subj["model_group"].first().to_numpy()
    src = subj["group"].first().to_numpy()
    n_rep = subj.size().to_numpy()
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
        "value_range_subject_mean": [float(X.min()), float(X.max())],
    }
    return X, wn, g, ids, info


def to_512_minmax(X, wn):
    grid = np.linspace(wn.min(), wn.max(), SIG_LEN)
    Xi = np.stack([np.interp(grid, wn, row) for row in X])
    mn = Xi.min(axis=1, keepdims=True)
    mx = Xi.max(axis=1, keepdims=True)
    return (Xi - mn) / (mx - mn + 1e-8), grid


# ----------------------------------------------------------------------------- DSCF
def md5sum(path, chunk=1 << 24):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def load_dscf(weights, repo, pydeps, device):
    import torch
    from types import SimpleNamespace

    sys.path.insert(0, str(pydeps))
    sys.path.insert(0, str(repo / "pretrain"))
    import DSCF_models_pe_ as dscf  # noqa: E402  (scratch clone, unmodified)

    report = {"weights": str(weights)}
    if not weights.exists():
        raise StopRun(f"weights missing: {weights}")
    md5 = md5sum(weights)
    report["md5"] = md5
    report["md5_expected"] = WEIGHT_MD5
    if md5 != WEIGHT_MD5:
        raise StopRun(f"md5 mismatch {md5} != {WEIGHT_MD5} (download incomplete/corrupt)")

    try:
        ckpt = torch.load(weights, map_location="cpu", weights_only=True)
        report["torch_load"] = "weights_only=True"
    except Exception as e:  # trusted file from the paper authors
        log.info("weights_only load failed (%s); retrying weights_only=False", type(e).__name__)
        ckpt = torch.load(weights, map_location="cpu", weights_only=False)
        report["torch_load"] = "weights_only=False"

    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        report["checkpoint_top_keys"] = list(ckpt.keys())
        sd = ckpt["model"]
    else:
        sd = ckpt
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    prefixes = sorted({k.split(".")[0] for k in sd})
    report["state_dict_n_keys"] = len(sd)
    report["state_dict_top_prefixes"] = prefixes
    if not any(p.startswith("Dec_") for p in prefixes):
        raise StopRun(f"unexpected checkpoint layout, prefixes={prefixes}")
    if not (isinstance(ckpt, dict) and "model_args" in ckpt):
        raise StopRun("checkpoint has no stored model_args")
    cls_name = "MultiDec_1d_model"
    ckpt_args = dict(vars(ckpt["model_args"]))
    report["checkpoint_model_args"] = _jsonable(ckpt_args)
    report["checkpoint_iter_num"] = _jsonable(ckpt.get("iter_num"))
    report["checkpoint_best_val_loss"] = _jsonable(ckpt.get("best_val_loss"))
    report["checkpoint_args_vs_public_tiny_cfg"] = {
        k: {"checkpoint": _jsonable(ckpt_args.get(k)), "public_customized_task": v}
        for k, v in DSCF_CFG.items() if k != "mask" and ckpt_args.get(k) != v
    }
    # inference: device supplied here, random masking disabled (mask has no parameters)
    build_cfg = {**ckpt_args, "device": "cpu", "mask": False}
    report["build_config"] = _jsonable(build_cfg)
    report["mask_handling"] = (f"checkpoint mask={ckpt_args.get('mask')!r}; built with mask=False and the "
                               "encoder is called directly (patch_embed->enc1..enc4), so forward()'s random "
                               "masking is never applied; model.eval() + torch.no_grad()")
    model = dscf.MultiDec_1d_model(SimpleNamespace(**build_cfg))
    report["model_class"] = cls_name

    msd = model.state_dict()
    missing = sorted(set(msd) - set(sd))
    unexpected = sorted(set(sd) - set(msd))
    shape_mismatch = sorted(
        f"{k}: ckpt {tuple(sd[k].shape)} vs model {tuple(msd[k].shape)}"
        for k in set(sd) & set(msd) if tuple(sd[k].shape) != tuple(msd[k].shape)
    )
    report.update(missing_keys=missing, unexpected_keys=unexpected, shape_mismatch=shape_mismatch)
    try:
        model.load_state_dict(sd, strict=True)
        report["strict_load"] = "ok"
    except RuntimeError as e:
        report["strict_load"] = "failed"
        report["strict_load_error"] = str(e)[:4000]
        return None, report, None
    model.eval().to(device)
    return model, report, (dscf, build_cfg)


def _jsonable(v):
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, (int, float, str, bool, type(None))):
        return v
    if hasattr(v, "item"):
        return v.item()
    return str(v)


def build_random_model(dscf, build_cfg, seed, device):
    """Same architecture as the loaded model, random init with torch.manual_seed(seed)."""
    import torch
    from types import SimpleNamespace

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = dscf.MultiDec_1d_model(SimpleNamespace(**build_cfg))
    return model.eval().to(device)


def extract_features(model, X512, device, batch=64):
    import torch

    feats = {f"enc{i}": [] for i in range(1, 5)}
    shapes = {}
    with torch.no_grad():
        for s in range(0, len(X512), batch):
            x = torch.as_tensor(X512[s:s + batch], dtype=torch.float32, device=device)[:, None, :]
            h = model.patch_embed(x)  # B, E, L/patch
            shapes["patch_embed"] = list(h.shape[1:])
            for i, enc in enumerate([model.enc1, model.enc2, model.enc3, model.enc4], start=1):
                h = enc(h)  # B, C, L
                shapes[f"enc{i}"] = list(h.shape[1:])
                feats[f"enc{i}"].append(h.mean(dim=2).double().cpu().numpy())
    out = {k: np.concatenate(v) for k, v in feats.items()}
    for k, v in out.items():
        if not np.isfinite(v).all():
            raise StopRun(f"non-finite encoder features at {k}")
    return out, shapes


# ----------------------------------------------------------------------------- CV
def make_splits(y, groups, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


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


def run_cv(conditions, y, groups, task, ckpt=None):
    rows = []
    for seed in SEEDS:
        splits = make_splits(y, groups, seed)
        for cond, Xsrc in conditions.items():
            Xc = Xsrc[seed] if isinstance(Xsrc, dict) else Xsrc  # D_*: per-seed random init
            n_cls = len(np.unique(y))
            oof_proba = np.full((len(y), n_cls), np.nan)
            oof_pred = np.empty(len(y), dtype=object)
            t0 = time.time()
            for k, (tr, te) in enumerate(splits):
                proba, pred, best_c, classes = fit_predict(Xc[tr], y[tr], Xc[te], task, seed)
                oof_proba[te] = proba
                oof_pred[te] = pred
                m = metrics(task, y[te], proba, pred, classes)
                rows.append({"task": task, "condition": cond, "seed": seed, "fold": k,
                             "level": "fold", "n_train": len(tr), "n_test": len(te),
                             "n_features": Xc.shape[1], "best_C": best_c, **m})
            oof_pred = oof_pred.astype(y.dtype)
            m = metrics(task, y, oof_proba, oof_pred, classes)
            rows.append({"task": task, "condition": cond, "seed": seed, "fold": "pooled_oof",
                         "level": "seed_pooled_oof", "n_train": np.nan, "n_test": len(y),
                         "n_features": Xc.shape[1], "best_C": np.nan, **m})
            log.info("%s | %-7s | seed %5d | %s | %.1fs", task, cond, seed,
                     " ".join(f"{a}={b:.4f}" for a, b in m.items()), time.time() - t0)
        if ckpt is not None:  # 중단 대비: 시드마다 중간 저장
            pd.DataFrame(rows).to_csv(ckpt, index=False, encoding="utf-8-sig")
    return rows


def summarize(runs: pd.DataFrame):
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
        present = set(tdf["condition"].unique())
        pairs = []
        for cond in sorted(c for c in present if c.startswith("C_")):
            pairs.append((cond, "B"))
            dcond = "D_" + cond[2:]
            if dcond in present:
                pairs.append((cond, dcond))
        for first, second in pairs:
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
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=SCRATCH / "weights/SiT_PPS_tiny.pt")
    ap.add_argument("--repo", type=Path, default=SCRATCH / "dscf_repo")
    ap.add_argument("--pydeps", type=Path, default=SCRATCH / "pydeps")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-task1", action="store_true",
                    help="reuse runs_partial_task1.csv from an earlier run of this script instead of recomputing task1")
    args = ap.parse_args()

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
        "dscf_repo": str(args.repo), "weights": str(args.weights),
        "dscf_config": DSCF_CFG,
        "seeds": SEEDS, "cv": "StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed); groups=subject id (group_sampleid); same splits for all conditions within task x seed",
        "classifier": "Pipeline(StandardScaler, LogisticRegression(L2, lbfgs, max_iter=10000, class_weight=None))",
        "C_selection": f"GridSearchCV over C={C_GRID}, inner StratifiedKFold(3, shuffle=True, random_state=seed) on the training fold only; scoring roc_auc (task1) / f1_macro (task2); refit on full training fold",
        "aggregation": "subject mean of replicates; subject key = (source group, sample_id); CPAN/YPAN->PAN, YNOR->NOR",
        "reported_level": "seed-level metrics from pooled out-of-fold predictions; fold-level rows also in runs.csv",
        "task1_threshold": "balanced accuracy from predict() (argmax, i.e. p>=0.5)",
    }
    X, wn, g, ids, info = load_kao()
    meta["data"] = info
    log.info("data: %s", json.dumps(info, ensure_ascii=False))
    assert info["n_subjects"] == 1630 and info["n_cancer"] == 1200 and info["n_control"] == 430, info

    X512, grid = to_512_minmax(X, wn)
    meta["input_B_C"] = {"interp": "np.interp linear onto 512 evenly spaced points over the original wavenumber range",
                         "grid_range": [float(grid[0]), float(grid[-1])],
                         "grid_step_cm-1": float(grid[1] - grid[0]),
                         "normalization": "per-sample min-max to [0,1] after interpolation (same matrix used for B and fed to encoder for C)"}

    def dump_meta():
        with open(args.out / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

    meta["limitations"] = [
        "Retrospective cohort: each cancer type was collected at different sites/periods, so site/batch confounding with class labels is possible (task2 especially).",
        "DSCF pretraining domain (per paper: serum SERS, 638 nm, Ag/Au nanostars plus synthetic IR/Raman/UV mixtures) differs from ours (urine SERS, 785 nm).",
        "Input is already preprocessed (processed_spectra.csv; values include negatives, subject-mean range " + f"{info['value_range_subject_mean'][0]:.3f}..{info['value_range_subject_mean'][1]:.3f}" + "), not raw-type non-negative intensity spectra; pretraining code normalizes by dividing by the max, here min-max was applied instead.",
        "Encoder-feature linear probe (mean-pooled stage outputs) is not a pathway validated in the DSCF paper.",
        "Task2: BRE has 30 subjects (~6 per test fold); macro-F1 is noisy.",
        "Exploratory analysis; no external test set.",
        "Checkpoint records iter_num=159 and best_val_loss=1e10, so whether pretraining ran to completion is unclear.",
        "Public finetuning setting (customized_task, output_channels=1) differs from the pretrained checkpoint's stored setting (outplanes=100); the model was built with the checkpoint's model_args. First attempt with the public setting stopped at strict load (see run_metadata_stop1.json).",
    ]

    try:
        model, load_report, builder = load_dscf(args.weights, args.repo, args.pydeps, args.device)
    except StopRun as e:
        meta["stopped"] = f"stop condition: {e}"
        log.error("STOP: %s", e)
        dump_meta()
        return 2
    meta["weight_load"] = load_report
    log.info("load report: model=%s strict=%s missing=%d unexpected=%d shape_mismatch=%d",
             load_report.get("model_class"), load_report.get("strict_load"),
             len(load_report["missing_keys"]), len(load_report["unexpected_keys"]),
             len(load_report["shape_mismatch"]))
    if model is None:
        meta["stopped"] = "stop condition 2: strict state_dict load failed"
        log.error("STOP: strict load failed")
        dump_meta()
        return 2

    try:
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        feats, shapes = extract_features(model, X512, args.device)
        meta["encoder_features"] = {
            "path": "x(B,1,512) -> patch_embed(Conv1d k=4,s=4) -> enc1 -> enc2 -> enc3 -> enc4; model.eval(), torch.no_grad(), no masking, decoders unused",
            "pooling": "mean over the length axis of each stage output (B,C,L) -> (B,C)",
            "stage_output_shapes_C_L": shapes,
            "feature_dims": {k: int(v.shape[1]) for k, v in feats.items()},
            "all_stages_reported": True,
            "extraction_seconds": round(time.time() - t0, 1),
            "peak_gpu_mem_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1) if "cuda" in args.device else None,
        }
    except torch.cuda.OutOfMemoryError as e:
        meta["stopped"] = f"stop condition 3: GPU OOM ({e})"
        log.error("STOP: OOM")
        dump_meta()
        return 2
    log.info("encoder features: %s", json.dumps(meta["encoder_features"]))
    dscf_mod, build_cfg = builder
    pre_first = next(model.parameters()).detach().flatten()[:8].double().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    dump_meta()

    # ---- D: random-init control, one model per CV seed ----
    feats_D = {f"enc{i}": {} for i in range(1, 5)}
    d_info = {}
    try:
        for seed in SEEDS:
            t0 = time.time()
            rmodel = build_random_model(dscf_mod, build_cfg, seed, args.device)
            rf, rshapes = extract_features(rmodel, X512, args.device)
            rand_first = next(rmodel.parameters()).detach().flatten()[:8].double().cpu().numpy()
            for k, v in rf.items():
                feats_D[k][seed] = v
            d_info[seed] = {"stage_output_shapes_C_L": rshapes,
                            "extraction_seconds": round(time.time() - t0, 1),
                            "first_param_differs_from_pretrained": bool(not np.allclose(rand_first, pre_first))}
            del rmodel
            torch.cuda.empty_cache()
            log.info("D random-init seed %d features extracted (%s)", seed, d_info[seed])
    except torch.cuda.OutOfMemoryError as e:
        meta["stopped"] = f"stop condition 3: GPU OOM in D extraction ({e})"
        log.error("STOP: OOM")
        dump_meta()
        return 2
    meta["random_init_control_D"] = {
        "construction": "MultiDec_1d_model(build_config) after torch.manual_seed(seed) and np.random.seed(seed); seed = CV seed of that run; library default init + model._init_weights (Linear/Embedding normal std 0.02)",
        "path_pooling": "identical to C (patch_embed->enc1..enc4, eval, no_grad, no masking, mean over length)",
        "per_seed": d_info,
    }
    dump_meta()

    conds_all = {"A": X, "B": X512,
                 **{f"C_{k}": v for k, v in feats.items()},
                 **{f"D_{k}": v for k, v in feats_D.items()}}
    meta["conditions"] = {"A": f"original grid ({X.shape[1]} pts)", "B": "512 interp + min-max",
                          **{f"C_{k}": f"DSCF pretrained {k} mean-pooled ({v.shape[1]}-d)" for k, v in feats.items()},
                          **{f"D_{k}": f"random-init (seed=CV seed) {k} mean-pooled ({v.shape[1]}-d)" for k, v in feats.items()}}

    def subset(v, m):
        return {s: a[m] for s, a in v.items()} if isinstance(v, dict) else v[m]

    rows = []
    # task 1: cancer vs control
    y1 = np.isin(g, CANCER_GROUPS).astype(int)
    p1 = args.out / "runs_partial_task1.csv"
    if args.skip_task1 and p1.exists():
        prev = pd.read_csv(p1, encoding="utf-8-sig")
        rows += prev.to_dict("records")
        meta["task1_source"] = f"reused {p1.name} (earlier run of this script, same seeds/config)"
        log.info("task1 reused from %s (%d rows)", p1, len(prev))
    else:
        rows += run_cv(conds_all, y1, ids, "task1_cancer_vs_control", ckpt=p1)
        pd.DataFrame(rows).to_csv(p1, index=False, encoding="utf-8-sig")
    # task 2: cancer type, cancer subjects only
    mask = np.isin(g, CANCER_GROUPS)
    y2 = g[mask].astype(str)
    rows += run_cv({k: subset(v, mask) for k, v in conds_all.items()}, y2, ids[mask], "task2_cancer_type",
                   ckpt=args.out / "runs_partial_task2.csv")

    runs = pd.DataFrame(rows)
    runs.to_csv(args.out / "runs.csv", index=False, encoding="utf-8-sig")
    summary = summarize(runs)
    with open(args.out / "summary.json", "w", encoding="utf-8-sig") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    dump_meta()
    log.info("done -> %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
