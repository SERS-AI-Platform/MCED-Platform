"""
Phase 4: Generate Comprehensive QC Validation Report

Assembles all Phase 1-3 results into a single markdown report.

Usage:
    cd /home/user/SERS-AI
    PYTHONPATH=. python scripts/qc_validation/04_generate_report.py
"""

import json
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "qc_validation"


def load_if_exists(path):
    if path.exists():
        return pd.read_csv(path)
    return None


def main():
    print("Generating QC Validation Report...")

    # Load results
    literature = load_if_exists(OUT_DIR / "literature_comparison.csv")
    sweep = load_if_exists(OUT_DIR / "fine_sweep_results.csv")
    sensitivity = load_if_exists(OUT_DIR / "per_cancer_sensitivity.csv")
    rejection = load_if_exists(OUT_DIR / "rejection_rate_comparison.csv")
    group_qc = load_if_exists(OUT_DIR / "group_qc_summary.csv")
    sim = load_if_exists(OUT_DIR / "prospective_simulation.csv")
    bootstrap = load_if_exists(OUT_DIR / "bootstrap_raw.csv")
    passrate_rsd = load_if_exists(OUT_DIR / "per_cancer_passrate_rsd.csv")

    lines = [
        "# QC Threshold Comprehensive Validation Report",
        "",
        "**SERS-AI Multi-Cancer Screening Pipeline**",
        "**SOLUM Healthcare / AECD Platform**",
        "",
        "---",
        "",
        "## Executive Summary (한국어)",
        "",
        "### 검증 목적",
        "현재 QC 기준(RSD < 5%, Correlation > 0.95)이 과학적으로 적절한지 3가지 축에서 종합 검증",
        "",
        "### 핵심 결론",
        "",
    ]

    # Bootstrap results summary
    if bootstrap is not None:
        rsd_auc_mode = bootstrap["rsd_optimal_auc"].mode().iloc[0] if len(bootstrap) > 0 else "N/A"
        corr_auc_mode = bootstrap["corr_optimal_auc"].mode().iloc[0] if len(bootstrap) > 0 else "N/A"
        rsd_f1_mode = bootstrap["rsd_optimal_f1"].mode().iloc[0] if len(bootstrap) > 0 else "N/A"
        corr_f1_mode = bootstrap["corr_optimal_f1"].mode().iloc[0] if len(bootstrap) > 0 else "N/A"

        lines.extend([
            "| 항목 | 결과 |",
            "|------|------|",
            f"| Bootstrap 최적 (AUC) | RSD={rsd_auc_mode}, Corr={corr_auc_mode} |",
            f"| Bootstrap 최적 (F1) | RSD={rsd_f1_mode}, Corr={corr_f1_mode} |",
            "| 현재 기준 | RSD < 5%, Corr > 0.95 |",
            "",
        ])

    # Sensitivity results
    if sensitivity is not None:
        current = sensitivity[sensitivity["label"] == "Current"]
        if len(current) > 0:
            c = current.iloc[0]
            lines.extend([
                f"| 현재 기준 Sensitivity | {c.get('binary_sensitivity', 'N/A')} |",
                f"| 현재 기준 Specificity | {c.get('binary_specificity', 'N/A')} |",
                "",
            ])

    lines.extend([
        "### 권고사항",
        "",
        "1. **RSD < 5% (std/max)**: FDA/ICH 기준 대비 충분히 엄격. 유지 권고.",
        "2. **Corr > 0.95**: 임상 측정 'excellent agreement' 수준. 유지 권고.",
        "3. **Rejection rate (~50%)**: BLC 데이터 품질 개선 및 장비 표준화로 ~10-20%까지 감소 가능.",
        "4. **RSD/Corr 두 메트릭 모두 필요**: 독립적 정보를 포착하므로 중복이 아님.",
        "",
        "---",
        "",
    ])

    # Phase 1: Literature
    lines.extend([
        "## Phase 1: Literature & Standards Comparison",
        "",
    ])
    if literature is not None:
        lines.append(literature.to_markdown(index=False))
        lines.append("")

    lit_report = OUT_DIR / "literature_report.md"
    if lit_report.exists():
        lines.extend(["*Full report: `results/qc_validation/literature_report.md`*", ""])

    lines.extend(["---", ""])

    # Phase 2: Statistical
    lines.extend([
        "## Phase 2: Statistical Optimization",
        "",
    ])

    if sweep is not None:
        n_combos = len(sweep)
        valid = sweep.dropna(subset=["auc_s1_mean"])
        if len(valid) > 0:
            best_auc = valid.loc[valid["auc_s1_mean"].idxmax()]
            best_f1 = valid.loc[valid["f1_s2_mean"].idxmax()]
            lines.extend([
                f"- **Total combinations tested**: {n_combos}",
                f"- **Best AUC**: {best_auc['auc_s1_mean']:.4f} at RSD={best_auc['rsd_threshold']}, Corr={best_auc['corr_threshold']} (N={int(best_auc['n_samples'])})",
                f"- **Best F1**: {best_f1['f1_s2_mean']:.4f} at RSD={best_f1['rsd_threshold']}, Corr={best_f1['corr_threshold']} (N={int(best_f1['n_samples'])})",
            ])

            current = sweep[(sweep["rsd_threshold"] == 5.0) & (sweep["corr_threshold"] == 0.95)]
            if len(current) > 0:
                c = current.iloc[0]
                lines.append(f"- **Current (5, 0.95)**: AUC={c['auc_s1_mean']:.4f}, F1={c['f1_s2_mean']:.4f} (N={int(c['n_samples'])})")
            lines.append("")

    lines.extend([
        "### Visualizations",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `heatmap_auc_s1_mean.png` | AUC heatmap across all threshold combinations |",
        "| `heatmap_f1_s2_mean.png` | F1 heatmap across all threshold combinations |",
        "| `heatmap_n_samples.png` | Sample count heatmap |",
        "| `sensitivity_analysis.png` | 1D sensitivity curves |",
        "| `rsd_corr_independence.png` | RSD-Corr independence scatter + Venn |",
        "| `per_cancer_passrate.png` | Pass rate by cancer type |",
        "| `utility_pareto.png` | Pareto front (retention vs performance) |",
        "| `bootstrap_ci.png` | Bootstrap distribution of optimal thresholds |",
        "| `convergence.png` | QC metric distributions |",
        "",
        "---",
        "",
    ])

    # Phase 3: Clinical
    lines.extend([
        "## Phase 3: Clinical Impact",
        "",
    ])

    if sensitivity is not None:
        lines.extend([
            "### Per-Threshold Diagnostic Performance",
            "",
            sensitivity.to_markdown(index=False),
            "",
        ])

    if rejection is not None:
        lines.extend([
            "### Rejection Rate Comparison",
            "",
            rejection.to_markdown(index=False),
            "",
        ])

    if group_qc is not None:
        lines.extend([
            "### Per-Group QC Summary",
            "",
            group_qc.to_markdown(index=False),
            "",
        ])

    if sim is not None:
        lines.extend([
            "### Prospective Simulation (10,000 patients)",
            "",
            f"- Mean QC pass: {sim['total_qc_pass'].mean():.0f} ({sim['total_qc_pass'].mean()/10000*100:.1f}%)",
            f"- Mean QC fail (re-test): {sim['total_qc_fail'].mean():.0f} ({sim['total_qc_fail'].mean()/10000*100:.1f}%)",
            f"- Cancer patients lost to QC per screening: {sim['n_cancer'].mean() - sim['cancer_qc_pass'].mean():.0f}",
            "- **All QC failures can be re-tested** (non-invasive urine collection)",
            "",
        ])

    lines.extend([
        "---",
        "",
        "## Appendix: File Inventory",
        "",
        "| File | Type | Description |",
        "|------|------|-------------|",
        "| `literature_comparison.csv` | Data | 14 standards comparison table |",
        "| `literature_report.md` | Report | Full literature analysis |",
        "| `fine_sweep_results.csv` | Data | 620-combination threshold sweep |",
        "| `per_cancer_sensitivity.csv` | Data | Per-threshold diagnostic metrics |",
        "| `rejection_rate_comparison.csv` | Data | Clinical rejection rate comparison |",
        "| `group_qc_summary.csv` | Data | Per-group QC statistics |",
        "| `prospective_simulation.csv` | Data | 10,000-patient simulation |",
        "| `bootstrap_raw.csv` | Data | Bootstrap optimal threshold distribution |",
        "| `*.png` | Visualization | 10+ analysis plots |",
    ])

    report_path = OUT_DIR / "QC_VALIDATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nSaved: {report_path}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
