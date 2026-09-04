#!/usr/bin/env python3
"""측정자(operator)에 따른 SERS 스펙트럼 변동성 분석.

설계상의 한계를 먼저 밝혀둔다. aecd_platform의 6개 run에서 측정자는 날짜가 바뀔 때만
바뀐다 -- 엄찬호 = 8/10, 8/11, 8/12 / 고은혜 = 8/13, 8/13, 8/14. 두 측정자가 같은 날
측정한 적도, 같은 검체를 다시 측정한 적도 없다. 따라서 "측정자 효과"는 날짜 효과,
strip unit 효과, 그날의 장비 보정 상태와 완전히 aliasing 되어 있고 주효과로 분리
추정할 수 없다.

그래서 이 스크립트가 내놓는 것은 다음 세 가지다.

1. 분산 성분 분해 (run / 검체 / 측정점). run 수준 분산이 검체 간 분산에 비해 얼마나
   큰지가 실제로 답할 수 있는 질문이다.
2. 엄찬호의 3일치 BNOR run으로 측정자 내부의 날짜 간 변동 폭을 구하고, 고은혜의
   BNOR run(8/13) 하나가 그 폭 안에 들어오는지를 표준화 편차로 본다. 한쪽 측정자의
   날짜가 1개뿐이라 p-value는 의미가 없어 계산하지 않는다.
3. 전처리(SNV) 전후 비교. SNV가 흡수해 버리는 차이인지, 모델까지 살아남는 차이인지가
   실무적으로 중요한 구분이다.

측정자 비교는 BNOR 검체로만 한다. BPRO(전립선암)는 고은혜만 측정해서, BPRO를 넣으면
질환 효과와 측정자 효과가 섞인다.

실행:
    python scripts/analysis/operator_variability_analysis.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from sers.config import load_config  # noqa: E402
from sers.signal import baseline_correction, resample, smooth, snv  # noqa: E402
from sers.visualization import use_korean_font  # noqa: E402

DEFAULT_CACHE = Path("/tmp/claude-1000/-home-user-SERS-AI/spectra_cache.npz")
DEFAULT_OUT = ROOT / "results" / "operator_variability"

INVENTORY_SQL = """
SELECT
    rs.measurement_id,
    r.measurement_run_id      AS run_id,
    r.measurement_date,
    r.operator_name,
    r.strip_lot_id,
    r.temperature_c,
    r.humidity_percent,
    m.point_no,
    smp.solum_label,
    u.cancer_group,
    u.label_prefix
FROM measurement.raw_spectra AS rs
JOIN measurement.measurements AS m USING (measurement_id)
JOIN measurement.runs AS r        USING (measurement_run_id)
JOIN master.samples AS smp        ON smp.sample_id = m.sample_id
JOIN clinical.uti_indicators AS u ON u.sample_id = m.sample_id
ORDER BY rs.measurement_id
"""


def connect():
    import psycopg2

    return psycopg2.connect(
        dbname=os.environ.get("PGDATABASE", "aecd_platform"),
        host=os.environ["PGHOST"],
        user=os.environ.get("PGUSER"),
        password=os.environ.get("PGPASSWORD"),
    )


def load_spectra(cache: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """DB에서 스펙트럼을 읽어 (inventory, wavenumbers, intensities)를 돌려준다.

    파장축은 날짜마다 0.1~0.2 cm-1씩 다르다(일일 보정 결과). 그래서 파장축도 함께
    싣고 뒤에서 공통 격자로 재샘플링한다.
    """
    if cache.exists():
        blob = np.load(cache, allow_pickle=True)
        inventory = pd.DataFrame(blob["inventory"], columns=list(blob["columns"]))
        print(f"cache hit: {cache} ({len(inventory)} spectra)")
        return inventory, blob["wavenumbers"], blob["intensities"]

    conn = connect()
    inventory = pd.read_sql(INVENTORY_SQL, conn)
    cur = conn.cursor(name="spectra")
    cur.itersize = 500
    cur.execute(
        "SELECT measurement_id, wavenumber, intensities FROM measurement.raw_spectra "
        "ORDER BY measurement_id"
    )
    ids, wns, ints = [], [], []
    for measurement_id, wavenumber, intensity in cur:
        ids.append(measurement_id)
        wns.append(wavenumber)
        ints.append(intensity)
    cur.close()
    conn.close()

    order = pd.Series(range(len(ids)), index=ids)
    inventory = inventory.set_index("measurement_id").loc[ids].reset_index()
    wavenumbers = np.asarray(wns, dtype=float)
    intensities = np.asarray(ints, dtype=float)
    assert len(order) == len(inventory)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        inventory=inventory.to_numpy(dtype=object),
        columns=np.array(inventory.columns),
        wavenumbers=wavenumbers,
        intensities=intensities,
    )
    print(f"fetched {len(inventory)} spectra -> {cache}")
    return inventory, wavenumbers, intensities


def spectrum_metrics(grid: np.ndarray, raw: np.ndarray, cfg) -> dict[str, float]:
    """스펙트럼 1개의 획득 품질 지표. 전처리 전 원본에서 잰다."""
    smoothed = smooth(raw, window=cfg.smooth_window, poly=cfg.smooth_poly)
    noise = float(np.std(raw - smoothed))
    corrected = baseline_correction(smoothed, window=cfg.baseline_window)
    baseline = float(np.median(smoothed - corrected))
    peak = float(np.max(corrected))
    return {
        "auc": float(np.trapezoid(raw, grid)),
        "median_intensity": float(np.median(raw)),
        "max_intensity": float(np.max(raw)),
        "baseline_level": baseline,
        "noise_sd": noise,
        # SNR: 베이스라인 제거 후 최대 피크를 평활화 잔차(노이즈 바닥)로 나눈 값
        "snr": peak / noise if noise > 0 else np.nan,
    }


def build_matrix(inventory, wavenumbers, intensities, cfg):
    """공통 격자로 재샘플링한 원본 행렬, SNV 행렬, 스펙트럼별 지표를 만든다."""
    low, high = cfg.trim_region
    step = float(np.median(np.diff(wavenumbers[0])))
    grid = np.arange(low, high + step, step)

    n = len(inventory)
    raw_matrix = np.empty((n, grid.size))
    snv_matrix = np.empty((n, grid.size))
    rows = []
    for i in range(n):
        y = resample(wavenumbers[i], intensities[i], grid)
        raw_matrix[i] = y
        rows.append(spectrum_metrics(grid, y, cfg))
        processed = smooth(y, window=cfg.smooth_window, poly=cfg.smooth_poly)
        processed = baseline_correction(processed, window=cfg.baseline_window)
        snv_matrix[i] = snv(processed)
        if (i + 1) % 2000 == 0:
            print(f"  processed {i + 1}/{n}")

    metrics = pd.concat([inventory.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return grid, raw_matrix, snv_matrix, metrics


def add_shape_metrics(metrics, snv_matrix):
    """SNV 후에도 남는 형태 차이. SNV는 배수적 차이를 지워버리므로 강도 지표로는
    측정자 효과가 사라진 것처럼 보인다 -- 형태로 다시 재야 한다."""
    global_mean = snv_matrix.mean(axis=0)
    centered = snv_matrix - snv_matrix.mean(axis=1, keepdims=True)
    gm = global_mean - global_mean.mean()
    denom = np.linalg.norm(centered, axis=1) * np.linalg.norm(gm)
    metrics = metrics.copy()
    metrics["corr_to_global_mean"] = (centered @ gm) / np.where(denom > 0, denom, np.nan)
    return metrics


def replicate_consistency(metrics, snv_matrix):
    """검체 내 121개 측정점이 서로 얼마나 일치하는지 (SNV 후 평균 상관)."""
    out = []
    for label, idx in metrics.groupby("solum_label").groups.items():
        block = snv_matrix[np.asarray(idx)]
        centered = block - block.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(centered, axis=1)
        corr = (centered @ centered.T) / np.outer(norms, norms)
        upper = corr[np.triu_indices(len(block), k=1)]
        out.append(
            {
                "solum_label": label,
                "n_points": len(block),
                "mean_point_corr": float(np.mean(upper)),
                "min_point_corr": float(np.min(upper)),
            }
        )
    return pd.DataFrame(out)


def variance_components(sample_level, metric):
    """run / 검체 수준 분산 비율. 검체가 run 안에 완전히 nested 되어 있으므로
    일원 분산분석의 성분 추정으로 충분하다."""
    groups = [g[metric].to_numpy() for _, g in sample_level.groupby("run_id")]
    counts = np.array([len(g) for g in groups])
    means = np.array([g.mean() for g in groups])
    grand = np.concatenate(groups).mean()
    k, total = len(groups), counts.sum()
    ss_between = float((counts * (means - grand) ** 2).sum())
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    ms_between = ss_between / (k - 1)
    ms_within = ss_within / (total - k)
    n0 = (total - (counts**2).sum() / total) / (k - 1)
    var_run = max((ms_between - ms_within) / n0, 0.0)
    var_sample = ms_within
    share = var_run / (var_run + var_sample) if var_run + var_sample > 0 else 0.0
    f_stat = ms_between / ms_within if ms_within > 0 else np.nan
    # run 효과의 유의성. 정규성 가정을 피하려고 Kruskal-Wallis도 같이 본다.
    p_anova = float(stats.f.sf(f_stat, k - 1, total - k)) if ms_within > 0 else np.nan
    p_kruskal = float(stats.kruskal(*groups).pvalue)
    return {
        "metric": metric,
        "var_run": var_run,
        "var_sample": var_sample,
        "run_share_pct": 100 * share,
        "f_stat": f_stat,
        "p_anova": p_anova,
        "p_kruskal": p_kruskal,
    }


def operator_contrast(sample_level, metric):
    """엄찬호의 날짜 간 변동 폭을 기준으로 고은혜의 BNOR run 위치를 잰다.

    고은혜의 BNOR 날짜가 1개뿐이라 측정자 효과와 날짜 효과가 분리되지 않는다.
    그래서 p-value 대신 '엄찬호 날짜 간 SD의 몇 배인가'만 보고한다.
    """
    run_means = sample_level.groupby(["run_id", "operator_name"])[metric].mean()
    ref = run_means.xs("엄찬호", level="operator_name")
    other = run_means.xs("고은혜", level="operator_name")
    between_day_sd = float(ref.std(ddof=1))
    deviation = float(other.mean() - ref.mean())
    return {
        "metric": metric,
        "ref_run_means": ", ".join(f"{v:.4g}" for v in ref),
        "ref_between_day_sd": between_day_sd,
        "other_run_mean": float(other.mean()),
        "difference": deviation,
        "difference_in_ref_sd": deviation / between_day_sd if between_day_sd > 0 else np.nan,
    }


def pca_figure(metrics, snv_matrix, out_path):
    """검체 평균 SNV 스펙트럼의 PCA. run이 블록을 이루는지 눈으로 확인한다."""
    order = metrics.groupby("solum_label").indices
    labels = sorted(order)
    means = np.vstack([snv_matrix[order[label]].mean(axis=0) for label in labels])
    meta = (
        metrics.groupby("solum_label")[["run_id", "operator_name", "label_prefix"]]
        .first()
        .loc[labels]
    )
    scores = PCA(n_components=2).fit(means)
    projected = scores.transform(means)

    # 한글 폰트가 있으면 한글 라벨, 없으면 Operator A/B 영문 라벨로 내려간다.
    korean = use_korean_font()
    operators = sorted(meta["operator_name"].unique(),
                       key=lambda name: meta.loc[meta.operator_name == name, "run_id"].min())
    alias = {name: (name if korean else f"Operator {chr(65 + i)}")
             for i, name in enumerate(operators)}
    meta = meta.assign(operator_alias=meta["operator_name"].map(alias))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    labels = (("run(날짜)별", "측정자별", "검체 평균 SNV 스펙트럼 PCA", "n=113 검체 × 121 point; SG 평활 → rolling-min baseline → SNV")
              if korean else
              ("by run (day)", "by operator", "PCA of per-sample mean SNV spectra",
               "n=113 samples, 121 points each; SNV after SG smoothing + rolling-min baseline"))
    for ax, key, title in (
        (axes[0], "run_id", labels[0]),
        (axes[1], "operator_alias", labels[1]),
    ):
        for value, group in meta.groupby(key, sort=True):
            rows = meta.index.get_indexer(group.index)
            for prefix, marker in (("BNOR", "o"), ("BPRO", "^")):
                mask = (meta.iloc[rows]["label_prefix"] == prefix).to_numpy()
                if not mask.any():
                    continue
                ax.scatter(projected[rows][mask, 0], projected[rows][mask, 1],
                           label=f"{value} ({prefix})", s=28, alpha=0.75, marker=marker)
        ax.set_xlabel(f"PC1 ({scores.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({scores.explained_variance_ratio_[1]:.1%})")
        ax.set_title(f"{labels[2]} — {title}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(alpha=0.25)
    fig.suptitle(labels[3], fontsize=9, y=-0.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return scores.explained_variance_ratio_[:2], alias


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT / "config" / "config.yaml").preprocessing
    inventory, wavenumbers, intensities = load_spectra(args.cache)
    for column in ("run_id", "point_no", "strip_lot_id"):
        inventory[column] = inventory[column].astype(int)

    print("preprocessing...")
    _, _, snv_matrix, metrics = build_matrix(inventory, wavenumbers, intensities, cfg)
    metrics = add_shape_metrics(metrics, snv_matrix)

    METRICS = ["auc", "median_intensity", "baseline_level", "noise_sd", "snr",
               "corr_to_global_mean"]
    sample_level = (
        metrics.groupby(
            ["solum_label", "run_id", "operator_name", "measurement_date",
             "cancer_group", "label_prefix", "strip_lot_id"],
            as_index=False,
        )[METRICS].median()
    )
    sample_level = sample_level.merge(
        replicate_consistency(metrics, snv_matrix), on="solum_label", how="left"
    )
    METRICS_SAMPLE = METRICS + ["mean_point_corr"]

    metrics.to_csv(args.out_dir / "spectrum_metrics.csv", index=False, encoding="utf-8-sig")
    sample_level.to_csv(args.out_dir / "sample_metrics.csv", index=False, encoding="utf-8-sig")

    bnor = sample_level[sample_level.label_prefix == "BNOR"]

    print("\n=== 설계 ===")
    print(
        sample_level.groupby(["operator_name", "measurement_date", "run_id",
                              "label_prefix"]).size().rename("samples").to_string()
    )

    print("\n=== 1. 분산 성분: run vs 검체 (BNOR only, n=%d) ===" % len(bnor))
    vc = pd.DataFrame([variance_components(bnor, m) for m in METRICS_SAMPLE])
    print(vc.to_string(index=False))
    vc.to_csv(args.out_dir / "variance_components.csv", index=False, encoding="utf-8-sig")

    print("\n=== 2. 측정자 대비 (BNOR only; 날짜·strip unit과 aliasing) ===")
    oc = pd.DataFrame([operator_contrast(bnor, m) for m in METRICS_SAMPLE])
    print(oc.to_string(index=False))
    oc.to_csv(args.out_dir / "operator_contrast.csv", index=False, encoding="utf-8-sig")

    print("\n=== 3. run별 검체 수준 요약 (BNOR only) ===")
    print(
        bnor.groupby(["run_id", "operator_name", "measurement_date"])[METRICS_SAMPLE]
        .agg(["mean", "std"])
        .round(4)
        .to_string()
    )

    figure_path = args.out_dir / "pca_by_run_and_operator.png"
    explained, alias = pca_figure(metrics, snv_matrix, figure_path)
    print("\n=== 4. PCA (검체 평균 SNV 스펙트럼) ===")
    print(f"PC1 {explained[0]:.1%}, PC2 {explained[1]:.1%} -> {figure_path}")
    print("그림 범례 대응: " + ", ".join(f"{v} = {k}" for k, v in alias.items()))

    print(f"\n출력: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
