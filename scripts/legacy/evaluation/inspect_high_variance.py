"""
High-variance sample 스펙트럼 시각화
- 각 sample의 replicate별 스펙트럼 오버레이
- 같은 그룹 평균 스펙트럼 비교
- replicate 간 차이(residual) 시각화
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ── 설정 ──
PROC = Path("results/preprocessing_dacr/processed_spectra.csv")
OUT  = Path("results/preprocessing_dacr/figures")
OUT.mkdir(parents=True, exist_ok=True)

# 확인할 샘플 (critical outlier + 상위 high-variance)
TARGETS = [
    ("LUN", 104),   # median_cv=53.4%, corr=0.938
    ("NOR", 46),     # median_cv=50.8%, corr=0.913
    ("BLC", 204),    # min_corr=0.877
    ("PRO", 36),     # min_corr=0.879
    ("CRC", 220),    # mean_cv=2899 (극단적)
    ("DIA", 55),     # mean_cv=1224
    ("PRO", 100),    # mean_cv=1392
]

# ── 데이터 로드 ──
print(f"Loading {PROC} ...")
df = pd.read_csv(PROC)

# wavenumber 컬럼 추출 (숫자 컬럼만)
meta_cols = ['group', 'sample_id', 'replicate']
wn_cols = [c for c in df.columns if c.startswith('x_')]
wavenumbers = np.array([float(c.replace('x_', '')) for c in wn_cols])

# ── 그룹 컬럼 이름 자동 감지 ──
group_col = [c for c in meta_cols if c.lower() in ('group', 'spectral_group', 'label')][0]
sid_col = [c for c in meta_cols if 'sample' in c.lower() or 'sid' in c.lower() or c == 'sample_id'][0]

print(f"  Group column: {group_col}")
print(f"  Sample ID column: {sid_col}")

# ── Figure 생성 ──
n_targets = len(TARGETS)
fig, axes = plt.subplots(n_targets, 3, figsize=(24, 5 * n_targets))
if n_targets == 1:
    axes = axes.reshape(1, -1)

fig.suptitle("High-Variance Sample Inspection\n(Replicate overlay / Group mean comparison / Residual)",
             fontsize=16, fontweight='bold', y=1.01)

for row, (grp, sid) in enumerate(TARGETS):
    # 해당 sample의 replicates
    mask = (df[group_col] == grp) & (df[sid_col].astype(str) == str(sid))
    sample_df = df[mask]
    
    # 같은 그룹 전체 (평균 계산용)
    group_df = df[df[group_col] == grp]
    group_mean = group_df[wn_cols].mean().values
    group_std  = group_df[wn_cols].std().values
    
    if len(sample_df) == 0:
        print(f"  ⚠ {grp}_{sid}: not found in processed_spectra (QC에서 이미 제거됨)")
        for col in range(3):
            axes[row, col].text(0.5, 0.5, f"{grp}_{sid}\nNOT FOUND\n(removed by QC)",
                               ha='center', va='center', fontsize=14, color='red',
                               transform=axes[row, col].transAxes)
            axes[row, col].set_facecolor('#fff5f5')
        continue
    
    spectra = sample_df[wn_cols].values  # (n_reps, n_wn)
    sample_mean = spectra.mean(axis=0)
    n_reps = len(spectra)
    
    # pairwise correlation 계산
    corrs = []
    for i in range(n_reps):
        for j in range(i+1, n_reps):
            corrs.append(np.corrcoef(spectra[i], spectra[j])[0, 1])
    mean_corr = np.mean(corrs) if corrs else 0
    min_corr = np.min(corrs) if corrs else 0
    
    # ─── Panel 1: Replicate overlay ───
    ax = axes[row, 0]
    colors = plt.cm.tab10(np.linspace(0, 1, n_reps))
    for i in range(n_reps):
        ax.plot(wavenumbers, spectra[i], alpha=0.7, linewidth=0.8, color=colors[i],
                label=f"rep {i+1}")
    ax.plot(wavenumbers, sample_mean, 'k-', linewidth=2, alpha=0.9, label='mean')
    ax.set_title(f"{grp}_{sid}  |  {n_reps} reps  |  corr={mean_corr:.3f}",
                fontsize=12, fontweight='bold')
    ax.set_ylabel("Intensity (SNV)")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.2)
    
    # ─── Panel 2: vs Group mean ───
    ax = axes[row, 1]
    ax.fill_between(wavenumbers, group_mean - group_std, group_mean + group_std,
                    alpha=0.15, color='blue', label='group mean ± SD')
    ax.plot(wavenumbers, group_mean, 'b-', linewidth=1.5, alpha=0.6, label='group mean')
    ax.plot(wavenumbers, sample_mean, 'r-', linewidth=1.5, alpha=0.9, label=f'{grp}_{sid} mean')
    
    # 차이가 큰 영역 하이라이트
    diff = np.abs(sample_mean - group_mean)
    threshold = 2 * group_std
    outlier_mask = diff > threshold
    if outlier_mask.any():
        ax.fill_between(wavenumbers, ax.get_ylim()[0], ax.get_ylim()[1],
                        where=outlier_mask, alpha=0.1, color='red', label='> 2σ deviation')
    
    ax.set_title(f"vs {grp} group mean (n={len(group_df)})", fontsize=11)
    ax.set_ylabel("Intensity (SNV)")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.2)
    
    # ─── Panel 3: Residual (replicate 간 차이) ───
    ax = axes[row, 2]
    for i in range(n_reps):
        residual = spectra[i] - sample_mean
        ax.plot(wavenumbers, residual, alpha=0.6, linewidth=0.7, color=colors[i],
                label=f"rep {i+1}")
    ax.axhline(y=0, color='k', linewidth=0.5, linestyle='--')
    ax.set_title(f"Residual (rep − mean)  |  min_corr={min_corr:.3f}", fontsize=11)
    ax.set_ylabel("Residual")
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.2)
    
    # 마지막 행에만 x-label
    if row == n_targets - 1:
        for col in range(3):
            axes[row, col].set_xlabel("Wavenumber (cm⁻¹)")

plt.tight_layout()
out_path = OUT / "high_variance_inspection.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n✅ Saved: {out_path}")
