#!/usr/bin/env python3
"""검체당 몇 개의 측정점이면 충분한가 -- 매핑의 원래 목적에 대한 답.

현재 보라매 매핑은 검체당 121점을 찍는다. 측정 기간을 줄이는 방법은 두 가지다.
사람을 늘리거나(측정자 변수가 새로 생긴다), 검체당 점 수를 줄이거나(새 변수가 없다).
이 스크립트는 후자가 어디까지 가능한지 잰다.

기준: k개 점으로 만든 평균 스펙트럼이 121점 전체 평균과 얼마나 다른가. 그 차이를
'같은 검체를 다른 날 찍었을 때의 차이'와 비교한다. 점을 줄여서 생기는 오차가 이미
존재하는 날짜 간 변동보다 작아지는 지점이 실무적 손익분기다 -- 그보다 더 찍는 것은
더 큰 다른 오차원에 묻힌다.

실행:
    python scripts/analysis/mapping_point_count_convergence.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sers.config import load_config  # noqa: E402
from sers.signal import baseline_correction, resample, smooth, snv  # noqa: E402
from sers.visualization import use_korean_font  # noqa: E402

CACHE = Path("/tmp/claude-1000/-home-user-SERS-AI/spectra_cache.npz")
OUT_DIR = ROOT / "results" / "operator_variability"
POINT_COUNTS = (3, 5, 10, 15, 20, 30, 40, 60, 80, 100, 121)
N_DRAWS = 40
SEED = 20260904


def load(cfg):
    blob = np.load(CACHE, allow_pickle=True)
    inventory = pd.DataFrame(blob["inventory"], columns=list(blob["columns"]))
    wavenumbers, intensities = blob["wavenumbers"], blob["intensities"]

    low, high = cfg.trim_region
    step = float(np.median(np.diff(wavenumbers[0])))
    grid = np.arange(low, high + step, step)

    processed = np.empty((len(inventory), grid.size))
    for i in range(len(inventory)):
        y = resample(wavenumbers[i], intensities[i], grid)
        y = smooth(y, window=cfg.smooth_window, poly=cfg.smooth_poly)
        processed[i] = snv(baseline_correction(y, window=cfg.baseline_window))
    return inventory, grid, processed


def convergence(inventory, processed):
    """점 수 k별로, k점 평균과 전체 평균의 상관·RMSE를 검체마다 계산한다."""
    rng = np.random.default_rng(SEED)
    groups = inventory.groupby("solum_label").indices
    rows = []
    for label, idx in groups.items():
        block = processed[np.asarray(idx)]
        full = block.mean(axis=0)
        full_centered = full - full.mean()
        full_norm = np.linalg.norm(full_centered)
        for k in POINT_COUNTS:
            if k > len(block):
                continue
            corrs, rmses = [], []
            draws = 1 if k == len(block) else N_DRAWS
            for _ in range(draws):
                pick = rng.choice(len(block), size=k, replace=False)
                partial = block[pick].mean(axis=0)
                centered = partial - partial.mean()
                corrs.append(float(centered @ full_centered
                                   / (np.linalg.norm(centered) * full_norm)))
                rmses.append(float(np.sqrt(np.mean((partial - full) ** 2))))
            rows.append({"solum_label": label, "n_points": k,
                         "corr_to_full": float(np.mean(corrs)),
                         "rmse_to_full": float(np.mean(rmses))})
    return pd.DataFrame(rows)


def reference_scales(inventory, processed, rng_seed=SEED):
    """점 축소 오차를 재는 두 개의 자를 만든다. 단위를 맞추는 게 핵심이다.

    - between_sample: 같은 run 안에서 서로 다른 두 검체의 평균 스펙트럼 간 RMSE.
      분류 모델이 실제로 쓰는 신호의 크기다. 점을 줄여 생긴 오차가 이것에 비해
      작아야 성능에 영향이 없다.
    - between_day: 서로 다른 run의 '검체 평균' 스펙트럼 간 RMSE를 검체 단위로 맞춘 값.
      run 평균은 20여 검체를 평균해 노이즈가 줄어 있으므로 sqrt(n)을 되곱해 보정한다.

    BNOR만 쓴다 -- BPRO를 섞으면 질환 차이가 들어간다.
    """
    rng = np.random.default_rng(rng_seed)
    bnor = inventory[inventory.label_prefix == "BNOR"]
    sample_means, sample_runs = [], []
    for label, idx in bnor.groupby("solum_label").indices.items():
        sample_means.append(processed[np.asarray(bnor.index[idx])].mean(axis=0))
        sample_runs.append(bnor.iloc[idx[0]]["run_id"])
    sample_means = np.vstack(sample_means)
    sample_runs = np.asarray(sample_runs)

    within_run_pairs = []
    for run in np.unique(sample_runs):
        rows = np.where(sample_runs == run)[0]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                within_run_pairs.append(
                    float(np.sqrt(np.mean((sample_means[rows[i]] - sample_means[rows[j]]) ** 2)))
                )
    between_sample = float(np.median(within_run_pairs))

    run_means, run_sizes = [], []
    for run in np.unique(sample_runs):
        rows = sample_runs == run
        run_means.append(sample_means[rows].mean(axis=0))
        run_sizes.append(rows.sum())
    run_means = np.vstack(run_means)
    raw_day = [
        float(np.sqrt(np.mean((run_means[i] - run_means[j]) ** 2)))
        for i in range(len(run_means)) for j in range(i + 1, len(run_means))
    ]
    scale = np.sqrt(np.mean(run_sizes))
    return between_sample, float(np.median(raw_day)) * scale


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "config" / "config.yaml").preprocessing

    inventory, _, processed = load(cfg)
    inventory = inventory.reset_index(drop=True)
    inventory["run_id"] = inventory["run_id"].astype(int)

    curve = convergence(inventory, processed)
    summary = (
        curve.groupby("n_points")
        .agg(corr_median=("corr_to_full", "median"),
             corr_p05=("corr_to_full", lambda s: s.quantile(0.05)),
             rmse_median=("rmse_to_full", "median"),
             rmse_p95=("rmse_to_full", lambda s: s.quantile(0.95)))
        .reset_index()
    )
    between_sample, between_day = reference_scales(inventory, processed)

    print("=== 검체당 측정점 수별 수렴 (n=113 검체, SNV 후) ===")
    print("corr/rmse는 121점 전체 평균 스펙트럼 대비")
    show = summary.copy()
    show["오차/검체간차이"] = show.rmse_median / between_sample
    show["오차/날짜간차이"] = show.rmse_median / between_day
    print(show.round(4).to_string(index=False))
    print(f"\n자 1) 검체 간 차이 (같은 run 내 두 검체) RMSE 중앙값 = {between_sample:.4f}")
    print(f"자 2) 날짜 간 차이 (검체 단위로 보정)         RMSE 중앙값 = {between_day:.4f}")

    for name, ref in (("검체 간 차이의 10%", 0.10 * between_sample),
                      ("날짜 간 차이", between_day)):
        ok = summary[summary.rmse_median < ref].n_points
        answer = int(ok.min()) if len(ok) else None
        print(f"점 축소 오차 < {name}: {answer if answer else '121점으로도 미달'}점")

    korean = use_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(summary.n_points, summary.corr_median, "o-", label="중앙값" if korean else "median")
    axes[0].plot(summary.n_points, summary.corr_p05, "s--", alpha=0.6,
                 label="하위 5%" if korean else "5th pct")
    axes[0].set_ylabel("121점 평균과의 상관" if korean else "corr to 121-point mean")
    axes[1].plot(summary.n_points, summary.rmse_median, "o-", label="중앙값" if korean else "median")
    axes[1].plot(summary.n_points, summary.rmse_p95, "s--", alpha=0.6,
                 label="상위 95%" if korean else "95th pct")
    axes[1].axhline(between_day, color="crimson", ls=":",
                    label=("날짜 간 차이" if korean else "between-day"))
    axes[1].axhline(0.10 * between_sample, color="darkgreen", ls="-.",
                    label=("검체 간 차이의 10%" if korean else "10% of between-sample"))
    axes[1].set_ylabel("121점 평균과의 RMSE" if korean else "RMSE to 121-point mean")
    for ax in axes:
        ax.set_xlabel("검체당 측정점 수" if korean else "points per sample")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(
        "검체당 측정점 수 수렴 곡선 (n=113 검체, SG 평활 → rolling-min baseline → SNV)"
        if korean else
        "Points-per-sample convergence (n=113 samples)", fontsize=10)
    path = OUT_DIR / "point_count_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    curve.to_csv(OUT_DIR / "point_count_convergence_per_sample.csv",
                 index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_DIR / "point_count_convergence.csv",
                   index=False, encoding="utf-8-sig")
    print(f"출력: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
