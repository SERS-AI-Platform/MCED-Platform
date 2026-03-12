"""
Visualization functions for SERS spectroscopy data.
Each function saves ONE plot to ONE file (no subplots unless noted).
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)

DEFAULT_CANCER_GROUPS = ("PRO", "BRE", "OVA", "LUN", "CRC", "CPAN", "SPAN")


def _ensure_output_dir(output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)


def _extract_feature_columns(df: pd.DataFrame, prefix: str = "x_") -> List[str]:
    return [c for c in df.columns if str(c).startswith(prefix)]


def _feature_axis_from_names(feature_names: Sequence[str]) -> np.ndarray:
    axis = []
    for i, name in enumerate(feature_names):
        try:
            axis.append(float(str(name).split("_", 1)[1]))
        except (IndexError, ValueError):
            axis.append(float(i))
    return np.asarray(axis, dtype=float)


def _normalize_shap_values(shap_values: Any) -> List[np.ndarray]:
    if isinstance(shap_values, list):
        return [np.asarray(v) for v in shap_values]

    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 2:
        return [shap_values]
    if shap_values.ndim == 3:
        return [shap_values[:, :, c] for c in range(shap_values.shape[-1])]
    raise ValueError(f"Unsupported SHAP value shape: {shap_values.shape}")


def visualize_raw_spectra(
    raw_spectra: Dict,
    output_dir: Path,
    group: str = None,
    sample_id: str = None,
    n_examples: int = 5
) -> None:
    """
    Visualize raw spectra - ONE plot per file.
    
    Saves: raw_spectra_{group}_{sample_id}.png
    """
    # Get one example
    replicate_data = {}
    for key in raw_spectra.keys():
        g, sid, replicate = key
        if group and g != group:
            continue
        
        rep_key = (g, replicate) if group else replicate

        if rep_key not in replicate_data:
            replicate_data[rep_key] = []
        
        replicate_data[rep_key].append((key, raw_spectra[key]))

    for rep_key, spectra_list in sorted(replicate_data.items()):
        fig, ax = plt.subplots(figsize=(12, 7))

        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_list)))

        for j, (key, y_raw) in enumerate(sorted(spectra_list)):
            g, sid, rep = key
            ax.plot(y_raw, color = colors[j], linewidth=1, alpha=0.6, label=f"S{sid} Rep {rep}")
        
        if isinstance(rep_key, tuple):
            g, rep = rep_key
            title = f'{g} - Replicate {rep} - Raw Spectra'
            output_path = output_dir / f"raw_spectra_{g}_rep{rep}.png"
        else:
            rep = rep_key
            title = f'All Groups - Replicate {rep} - Raw Spectra'
            output_path = output_dir / f"raw_spectra_allgroups_rep{rep}.png"

        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=14)
        ax.set_ylabel("Intensity (a.u.)", fontsize=14)
        ncol = 5 if len(spectra_list) ==5 else 1
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved: {output_path.name}")


def visualize_preprocessed_spectra_by_replicate(
    processed_spectra: Dict,
    grid: np.ndarray,
    output_dir: Path,
    group: str = None
) -> None:
    """
    Visualize preprocessed spectra grouped by replicate number.
    
    Saves: preprocessed_spectra_{group}_replicate_{rep}.png
    """
    # Group by replicate number
    replicates_data = {}
    
    for key in processed_spectra.keys():
        g, sid, replicate = key
        
        if group and g != group:
            continue
        
        rep_key = (g, replicate) if group else replicate
        
        if rep_key not in replicates_data:
            replicates_data[rep_key] = []
        
        replicates_data[rep_key].append((key, processed_spectra[key]))
    
    # Plot each replicate group
    for rep_key, spectra_list in sorted(replicates_data.items()):
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(spectra_list)))
        
        for j, (key, y_proc) in enumerate(spectra_list):
            g, sid, rep = key
            ax.plot(grid, y_proc, color=colors[j], linewidth=1.5, alpha=0.6,
                   label=f"{g} S{sid}")
        
        # Title and labels
        if isinstance(rep_key, tuple):
            g, rep = rep_key
            title = f'{g} - All Samples (Replicate {rep})'
            filename = f"preprocessed_spectra_{g}_replicate_{rep}.png"
        else:
            rep = rep_key
            title = f'All Groups - Replicate {rep}'
            filename = f"preprocessed_spectra_all_replicate_{rep}.png"
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=14)
        ax.set_ylabel("Intensity (a.u.)", fontsize=14)
        
        ncol = 3 if len(spectra_list) > 10 else 1
        ax.legend(loc='upper right', fontsize=10, ncol=ncol)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path.name}")


def visualize_raw_spectra_by_sample(
    raw_spectra: Dict,
    output_dir: Path,
    group: str = None,
    n_examples: int = 5
) -> None:
    """
    Visualize raw spectra - one sample per file, all replicates together.
    
    Saves: raw_spectra_{group}_{sample_id}.png
    """
    samples = {}
    for key in raw_spectra.keys():
        g, sid, replicate = key
        if group and g != group:
            continue
        sample_key = (g, sid)
        if sample_key not in samples:
            samples[sample_key] = []
        samples[sample_key].append(key)
    
    # Plot each sample separately
    for (g, sid), replicate_keys in list(samples.items())[:n_examples]:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(replicate_keys)))
        
        for j, key in enumerate(sorted(replicate_keys)):
            x_raw, y_raw = raw_spectra[key]
            rep = key[2]
            ax.plot(x_raw, y_raw, color=colors[j], linewidth=2, alpha=0.7, 
                   label=f"Replicate {rep}")
        
        ax.set_title(f'{g} Sample {sid} - Raw Spectra', fontsize=16, fontweight='bold')
        ax.set_xlabel("Raman Shift (cm⁻¹)", fontsize=14)
        ax.set_ylabel("Intensity (a.u.)", fontsize=14)
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = output_dir / f"raw_spectra_{g}_{sid}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved: {output_path.name}")


def plot_sample_distribution_pie(
    group_stats_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Plot sample distribution pie chart - ONE file.
    
    Saves: sample_distribution_pie.png
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    groups = group_stats_df['group'].values
    n_samples = group_stats_df['n_samples'].values
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    ax.pie(
        n_samples,
        labels=groups,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 14, 'fontweight': 'bold'}
    )
    ax.set_title('Sample Distribution by Group', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "sample_distribution_pie.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path.name}")


def plot_spectra_count_bar(
    group_stats_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Plot total spectra count bar chart - ONE file.
    
    Saves: spectra_count_bar.png
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    groups = group_stats_df['group'].values
    n_spectra = group_stats_df['n_spectra'].values
    x_pos = np.arange(len(groups))
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    ax.bar(x_pos, n_spectra, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups, fontsize=14)
    ax.set_ylabel('Number of Spectra', fontsize=14)
    ax.set_title('Total Spectra by Group', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, v in enumerate(n_spectra):
        ax.text(i, v + max(n_spectra)*0.02, str(int(v)), 
               ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_path = output_dir / "spectra_count_bar.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path.name}")


# group-wise replicate variance plots 그룹 레벨에서의 변동성 확인
def plot_replicate_variance_by_group(
        variance_df: pd.DataFrame,
        output_dir: Path,
        cv_threshold: float = 15.0,
        groups: Optional[List[str]] = None
    ) -> None:
    
    # Group filtering
    if groups is not None:
        variance_df = variance_df[variance_df['group'].isin(groups)].copy()

    groups = sorted(variance_df['group'].unique())
    for g in groups:
        group_data = variance_df[variance_df['group'] == g].copy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot 1: CV distribution
        axes[0].hist(
            group_data['mean_cv'],
            bins=20,
            color='#56B4E9',
            alpha = 0.7,
            edgecolor='black'
        )
        axes[0].axvline(cv_threshold, color='red', linestyle='--', linewidth=2, label=f'CV threshold ({cv_threshold}%)')
        axes[0].set_xlabel('Mean CV (%)', fontsize=14)
        axes[0].set_ylabel('Count', fontsize=14)
        axes[0].set_title(f'{g} - CV Distribution', fontsize=16, fontweight='bold')
        axes[0].legend(fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Correlation distribution
        axes[1].hist(
            group_data['mean_pairwise_correlation'],
            bins=20,
            color='#56B4E9',
            alpha=0.7,
            edgecolor='black'
        )
        axes[1].set_xlabel('Mean Pairwise Correlation', fontsize=14)
        axes[1].set_ylabel('Count', fontsize=14)
        axes[1].set_title(f'{g} - Correlation Distribution', fontsize=16, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Quality scatter
        colors_scatter = ['red' if cv > cv_threshold else 'green' for cv in group_data['mean_cv']]
        axes[2].scatter(
            group_data['mean_pairwise_correlation'],
            group_data['mean_cv'],
            c=colors_scatter,
            alpha=0.7,
            s=50
        )
        axes[2].axhline(cv_threshold, color='red', linestyle='--', linewidth=2, label=f'CV threshold ({cv_threshold}%)')
        axes[2].set_xlabel('Mean Pairwise Correlation', fontsize=14)
        axes[2].set_ylabel('Mean CV (%)', fontsize=14)
        axes[2].set_title(f'{g} - Quality Scatter', fontsize=16, fontweight='bold')
        axes[2].legend(fontsize=12)
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / f"replicate_variance_by_group_{g}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved: {output_path.name}")


def plot_variance_heatmap(
    variance_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Plot CV heatmap (samples × groups) - ONE file.
    
    Saves: variance_heatmap.png
    """
    pivot_data = variance_df.pivot_table(
        index='sample_id',
        columns='group',
        values='mean_cv',
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot_data) * 0.3)))
    
    sns.heatmap(
        pivot_data,
        annot=True,
        fmt='.1f',
        cmap='RdYlGn_r',
        vmin=0,
        vmax=20,
        cbar_kws={'label': 'Mean CV (%)'},
        linewidths=0.5,
        ax=ax
    )
    
    ax.set_title('Replicate Variability Heatmap (CV%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Group', fontsize=14)
    ax.set_ylabel('Sample ID', fontsize=14)
    
    plt.tight_layout()
    output_path = output_dir / "variance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved: {output_path.name}")


def plot_cancer_peak_difference(
    spectra_df: pd.DataFrame,
    output_path: Path,
    group_col: str = "group",
    feature_prefix: str = "x_",
    cancer_groups: Optional[Sequence[str]] = None,
    top_k: int = 15,
    title: str = "Cancer vs Non-cancer Peak Difference",
) -> pd.DataFrame:
    """
    Plot mean spectra for cancer vs non-cancer and highlight the largest differences.

    Parameters
    ----------
    spectra_df : pd.DataFrame
        DataFrame containing a group column and spectral feature columns.
    output_path : Path
        Output image path.
    group_col : str
        Column containing disease group labels.
    feature_prefix : str
        Prefix used by spectral columns, e.g. x_401.81.
    cancer_groups : sequence of str, optional
        Groups treated as cancer. Defaults to common 7-class cancer labels.
    top_k : int
        Number of most-different peaks to annotate.

    Returns
    -------
    pd.DataFrame
        Summary table of the top differing peaks.
    """
    if group_col not in spectra_df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    feature_cols = _extract_feature_columns(spectra_df, prefix=feature_prefix)
    if not feature_cols:
        raise ValueError(f"No spectral columns found with prefix '{feature_prefix}'")

    cancer_groups = tuple(cancer_groups or DEFAULT_CANCER_GROUPS)
    is_cancer = spectra_df[group_col].isin(cancer_groups).to_numpy()
    if is_cancer.sum() == 0 or (~is_cancer).sum() == 0:
        raise ValueError("Both cancer and non-cancer samples are required")

    X = spectra_df[feature_cols].to_numpy(dtype=float)
    axis = _feature_axis_from_names(feature_cols)

    cancer_mean = X[is_cancer].mean(axis=0)
    non_cancer_mean = X[~is_cancer].mean(axis=0)
    cancer_std = X[is_cancer].std(axis=0)
    non_cancer_std = X[~is_cancer].std(axis=0)
    diff = cancer_mean - non_cancer_mean

    top_k = min(top_k, len(feature_cols))
    top_idx = np.argsort(np.abs(diff))[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(axis, non_cancer_mean, color="#2b6cb0", linewidth=2, label="Non-cancer mean")
    ax.fill_between(
        axis,
        non_cancer_mean - non_cancer_std,
        non_cancer_mean + non_cancer_std,
        color="#2b6cb0",
        alpha=0.12,
    )
    ax.plot(axis, cancer_mean, color="#c53030", linewidth=2, label="Cancer mean")
    ax.fill_between(
        axis,
        cancer_mean - cancer_std,
        cancer_mean + cancer_std,
        color="#c53030",
        alpha=0.12,
    )
    ax.set_xlabel("Raman Shift (cm^-1)", fontsize=12)
    ax.set_ylabel("Mean Intensity (a.u.)", fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(axis, diff, color="#222222", linewidth=1.5, alpha=0.8, label="Cancer - Non-cancer")
    ax2.axhline(0.0, color="#444444", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Difference (a.u.)", fontsize=12)

    for idx in top_idx:
        wn = axis[idx]
        ax.axvline(wn, color="#ffb703", linestyle=":", linewidth=1, alpha=0.8)
        ax2.text(
            wn,
            diff[idx],
            f"{wn:.0f}",
            fontsize=8,
            rotation=90,
            va="bottom" if diff[idx] >= 0 else "top",
            ha="center",
        )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")

    top_peaks = pd.DataFrame(
        {
            "feature": [feature_cols[i] for i in top_idx],
            "wavenumber": axis[top_idx],
            "cancer_mean": cancer_mean[top_idx],
            "non_cancer_mean": non_cancer_mean[top_idx],
            "difference": diff[top_idx],
            "abs_difference": np.abs(diff[top_idx]),
        }
    ).sort_values("abs_difference", ascending=False, ignore_index=True)
    return top_peaks


def plot_group_peak_difference(
    spectra_df: pd.DataFrame,
    output_path: Path,
    target_group: str,
    reference_groups: Optional[Sequence[str]] = None,
    group_col: str = "group",
    feature_prefix: str = "x_",
    top_k: int = 15,
    title: Optional[str] = None,
    target_color: str = "#c53030",
    reference_color: str = "#2b6cb0",
) -> pd.DataFrame:
    """
    Plot mean spectra for one target group versus a reference set and highlight top peak differences.
    """
    if group_col not in spectra_df.columns:
        raise KeyError(f"Missing group column: {group_col}")

    feature_cols = _extract_feature_columns(spectra_df, prefix=feature_prefix)
    if not feature_cols:
        raise ValueError(f"No spectral columns found with prefix '{feature_prefix}'")

    df = spectra_df.copy()
    target_mask = df[group_col].astype(str) == str(target_group)
    if reference_groups is None:
        reference_mask = ~target_mask
    else:
        reference_mask = df[group_col].isin(reference_groups).to_numpy()

    if target_mask.sum() == 0:
        raise ValueError(f"No samples found for target group '{target_group}'")
    if reference_mask.sum() == 0:
        raise ValueError("Reference group set is empty")

    X = df[feature_cols].to_numpy(dtype=float)
    axis = _feature_axis_from_names(feature_cols)

    target_mean = X[target_mask].mean(axis=0)
    reference_mean = X[reference_mask].mean(axis=0)
    target_std = X[target_mask].std(axis=0)
    reference_std = X[reference_mask].std(axis=0)
    diff = target_mean - reference_mean

    top_k = min(top_k, len(feature_cols))
    top_idx = np.argsort(np.abs(diff))[-top_k:][::-1]
    reference_label = "Reference mean" if reference_groups is None else "Other diagnoses mean"

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(axis, reference_mean, color=reference_color, linewidth=2, label=reference_label)
    ax.fill_between(
        axis,
        reference_mean - reference_std,
        reference_mean + reference_std,
        color=reference_color,
        alpha=0.12,
    )
    ax.plot(axis, target_mean, color=target_color, linewidth=2, label=f"{target_group} mean")
    ax.fill_between(
        axis,
        target_mean - target_std,
        target_mean + target_std,
        color=target_color,
        alpha=0.12,
    )
    ax.set_xlabel("Raman Shift (cm^-1)", fontsize=12)
    ax.set_ylabel("Mean Intensity (a.u.)", fontsize=12)
    ax.set_title(title or f"{target_group} vs Reference Peak Difference", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(axis, diff, color="#222222", linewidth=1.5, alpha=0.85, label=f"{target_group} - Reference")
    ax2.axhline(0.0, color="#444444", linestyle="--", linewidth=1, alpha=0.6)
    ax2.set_ylabel("Difference (a.u.)", fontsize=12)

    for idx in top_idx:
        wn = axis[idx]
        ax.axvline(wn, color="#ffb703", linestyle=":", linewidth=1, alpha=0.8)
        ax2.text(
            wn,
            diff[idx],
            f"{wn:.0f}",
            fontsize=8,
            rotation=90,
            va="bottom" if diff[idx] >= 0 else "top",
            ha="center",
        )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")

    top_peaks = pd.DataFrame(
        {
            "feature": [feature_cols[i] for i in top_idx],
            "wavenumber": axis[top_idx],
            "target_group": target_group,
            "target_mean": target_mean[top_idx],
            "reference_mean": reference_mean[top_idx],
            "difference": diff[top_idx],
            "abs_difference": np.abs(diff[top_idx]),
        }
    ).sort_values("abs_difference", ascending=False, ignore_index=True)
    return top_peaks


def compute_gradient_shap_values(
    model: Any,
    X_background: np.ndarray,
    X_explain: np.ndarray,
    device: Any,
):
    """
    Compute SHAP values with GradientExplainer for torch models.

    The model should return:
    - binary: shape (N,) or (N, 1)
    - multiclass: shape (N, C)
    """
    try:
        import shap
        import torch
    except ImportError as exc:
        raise ImportError("shap and torch are required for SHAP visualization") from exc

    X_background = np.asarray(X_background, dtype=np.float32)
    X_explain = np.asarray(X_explain, dtype=np.float32)

    explainer = shap.GradientExplainer(
        model,
        torch.as_tensor(X_background, dtype=torch.float32, device=device),
    )
    shap_values = explainer.shap_values(
        torch.as_tensor(X_explain, dtype=torch.float32, device=device)
    )
    return shap_values


def plot_binary_shap_summary(
    shap_values: Any,
    X_explain: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 30,
    title: str = "Binary SHAP Summary",
) -> np.ndarray:
    """Create a SHAP beeswarm summary plot for binary output."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP visualization") from exc

    shap_list = _normalize_shap_values(shap_values)
    if len(shap_list) != 1:
        raise ValueError("Binary SHAP plot expects a single SHAP array")

    sv = np.asarray(shap_list[0])
    X_explain = np.asarray(X_explain)
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(X_explain.shape[1])]

    mean_abs = np.abs(sv).mean(axis=0)
    top_k = min(top_k, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]

    _ensure_output_dir(Path(output_path))
    shap.summary_plot(
        sv[:, top_idx],
        X_explain[:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        show=False,
        max_display=top_k,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")
    return sv


def plot_multiclass_shap_summary(
    shap_values: Any,
    X_explain: np.ndarray,
    class_names: Sequence[str],
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
    max_cols: int = 4,
    title: str = "Multiclass SHAP Summary",
) -> List[np.ndarray]:
    """
    Create one beeswarm summary subplot per class for multiclass SHAP values.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP visualization") from exc

    shap_list = _normalize_shap_values(shap_values)
    X_explain = np.asarray(X_explain)
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(X_explain.shape[1])]

    n_classes = min(len(shap_list), len(class_names))
    cols = min(max_cols, max(1, n_classes))
    rows = (n_classes + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.atleast_1d(axes).flatten()

    for class_idx in range(n_classes):
        plt.sca(axes[class_idx])
        sv = np.asarray(shap_list[class_idx])
        mean_abs = np.abs(sv).mean(axis=0)
        top_k_class = min(top_k, len(mean_abs))
        top_idx = np.argsort(mean_abs)[-top_k_class:][::-1]
        shap.summary_plot(
            sv[:, top_idx],
            X_explain[:, top_idx],
            feature_names=[feature_names[i] for i in top_idx],
            show=False,
            max_display=top_k_class,
            plot_size=None,
        )
        axes[class_idx].set_title(str(class_names[class_idx]))

    for idx in range(n_classes, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {Path(output_path).name}")
    return [np.asarray(v) for v in shap_list[:n_classes]]


def summarize_shap_feature_importance(
    shap_values: Any,
    feature_names: Optional[Sequence[str]] = None,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    """Summarize SHAP feature importance as mean absolute contribution."""
    importance = build_shap_spectrum_profile(shap_values, feature_names)
    importance = importance.sort_values("mean_abs_shap", ascending=False, ignore_index=True)

    if top_k is not None:
        importance = importance.head(int(top_k)).reset_index(drop=True)
    return importance


def build_shap_spectrum_profile(
    shap_values: Any,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a full-spectrum SHAP profile ordered along the spectral axis."""
    shap_list = _normalize_shap_values(shap_values)
    if len(shap_list) != 1:
        raise ValueError("SHAP spectrum profile expects a single SHAP array")

    sv = np.asarray(shap_list[0])
    n_features = sv.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    axis = _feature_axis_from_names(feature_names)
    profile = pd.DataFrame(
        {
            "feature": list(feature_names),
            "wavenumber": axis,
            "mean_abs_shap": np.abs(sv).mean(axis=0),
            "mean_shap": sv.mean(axis=0),
        }
    ).sort_values("wavenumber", ascending=True, ignore_index=True)
    return profile


def build_mean_spectrum_profile(
    spectra: np.ndarray,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Build a full-spectrum summary with mean and std along the spectral axis."""
    spectra = np.asarray(spectra, dtype=float)
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        raise ValueError("Mean spectrum profile expects a 2D array with at least one sample")

    n_features = spectra.shape[1]
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(n_features)]

    axis = _feature_axis_from_names(feature_names)
    profile = pd.DataFrame(
        {
            "feature": list(feature_names),
            "wavenumber": axis,
            "mean_intensity": spectra.mean(axis=0),
            "std_intensity": spectra.std(axis=0),
        }
    ).sort_values("wavenumber", ascending=True, ignore_index=True)
    return profile


def plot_mean_spectrum(
    spectra: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Mean Spectrum",
    color: str = "#c0392b",
) -> pd.DataFrame:
    """Plot mean spectrum with +-1 std band."""
    profile = build_mean_spectrum_profile(spectra, feature_names)

    x = profile["wavenumber"].to_numpy()
    mean_y = profile["mean_intensity"].to_numpy()
    std_y = profile["std_intensity"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, mean_y, color=color, linewidth=1.8)
    ax.fill_between(x, mean_y - std_y, mean_y + std_y, color=color, alpha=0.16)
    ax.set_xlabel("Raman Shift (cm^-1)")
    ax.set_ylabel("Mean Intensity (a.u.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")
    return profile


def plot_mean_spectra_overlay(
    spectra_by_class: Sequence[np.ndarray],
    class_names: Sequence[str],
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    colors: Optional[Sequence[str]] = None,
    title: str = "Mean Spectra by Diagnosis",
) -> pd.DataFrame:
    """Plot one mean spectrum per class on a shared axis."""
    rows = []
    fig, ax = plt.subplots(figsize=(13, 5))

    for idx, (spectra, class_name) in enumerate(zip(spectra_by_class, class_names)):
        spectra = np.asarray(spectra, dtype=float)
        if spectra.ndim != 2 or spectra.shape[0] == 0:
            continue
        profile = build_mean_spectrum_profile(spectra, feature_names)
        color = colors[idx] if colors is not None and idx < len(colors) else None
        ax.plot(
            profile["wavenumber"],
            profile["mean_intensity"],
            linewidth=1.7,
            color=color,
            label=f"{class_name} (n={spectra.shape[0]})",
        )
        profile.insert(0, "diagnosis", class_name)
        profile.insert(1, "n_samples", int(spectra.shape[0]))
        rows.append(profile)

    if not rows:
        raise ValueError("No class spectra available for overlay plot")

    ax.set_xlabel("Raman Shift (cm^-1)")
    ax.set_ylabel("Mean Intensity (a.u.)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)

    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")
    return pd.concat(rows, ignore_index=True)


def plot_shap_feature_importance_bar(
    importance_df: pd.DataFrame,
    output_path: Path,
    title: str = "SHAP Feature Importance",
    color: str = "#c0392b",
) -> None:
    """Plot mean absolute SHAP feature importance as a horizontal bar chart."""
    if importance_df.empty:
        raise ValueError("Importance dataframe is empty")

    data = importance_df.iloc[::-1].copy()
    labels = [
        f"{row.feature} ({row.wavenumber:.1f})"
        for row in data.itertuples(index=False)
    ]

    fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(data))))
    ax.barh(labels, data["mean_abs_shap"], color=color, alpha=0.85)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)

    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")


def plot_shap_mean_magnitude_spectrum(
    shap_values: Any,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Mean |SHAP| Spectrum",
    color: str = "#c0392b",
) -> pd.DataFrame:
    """Plot mean absolute SHAP across the spectral axis."""
    profile = build_shap_spectrum_profile(shap_values, feature_names)

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(profile["wavenumber"], profile["mean_abs_shap"], color=color, linewidth=1.8)
    ax.fill_between(
        profile["wavenumber"],
        0,
        profile["mean_abs_shap"],
        color=color,
        alpha=0.15,
    )
    ax.set_xlabel("Raman Shift (cm^-1)")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")
    return profile


def plot_shap_mean_signed_spectrum(
    shap_values: Any,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    title: str = "Mean SHAP Spectrum",
    color_positive: str = "#c0392b",
    color_negative: str = "#2980b9",
) -> pd.DataFrame:
    """Plot signed mean SHAP across the spectral axis."""
    profile = build_shap_spectrum_profile(shap_values, feature_names)

    x = profile["wavenumber"].to_numpy()
    y = profile["mean_shap"].to_numpy()

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(x, y, color="#222222", linewidth=1.4)
    ax.fill_between(x, 0, y, where=y >= 0, color=color_positive, alpha=0.2)
    ax.fill_between(x, 0, y, where=y < 0, color=color_negative, alpha=0.2)
    ax.axhline(0, color="#444444", linestyle="--", linewidth=1)
    ax.set_xlabel("Raman Shift (cm^-1)")
    ax.set_ylabel("Mean SHAP value")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    _ensure_output_dir(Path(output_path))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")
    return profile


def plot_class_shap_summary(
    shap_values: Any,
    X_explain: np.ndarray,
    output_path: Path,
    feature_names: Optional[Sequence[str]] = None,
    top_k: int = 20,
    title: str = "Class SHAP Summary",
) -> np.ndarray:
    """Create a SHAP beeswarm summary plot for a single diagnosis/class."""
    try:
        import shap
    except ImportError as exc:
        raise ImportError("shap is required for SHAP visualization") from exc

    shap_list = _normalize_shap_values(shap_values)
    if len(shap_list) != 1:
        raise ValueError("Class SHAP plot expects a single SHAP array")

    sv = np.asarray(shap_list[0])
    X_explain = np.asarray(X_explain)
    if X_explain.shape[0] == 0:
        raise ValueError("Class SHAP plot requires at least one sample")
    if feature_names is None:
        feature_names = [f"x_{i}" for i in range(X_explain.shape[1])]

    mean_abs = np.abs(sv).mean(axis=0)
    top_k = min(top_k, len(mean_abs))
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]

    _ensure_output_dir(Path(output_path))
    shap.summary_plot(
        sv[:, top_idx],
        X_explain[:, top_idx],
        feature_names=[feature_names[i] for i in top_idx],
        show=False,
        max_display=top_k,
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved: {Path(output_path).name}")
    return sv

__all__ = [
    "visualize_raw_spectra",
    "visualize_preprocessed_spectra_by_replicate",
    "visualize_raw_spectra_by_sample",
    "plot_sample_distribution_pie",
    "plot_spectra_count_bar",
    "plot_replicate_variance_by_group",
    "plot_variance_heatmap",
    "plot_cancer_peak_difference",
    "plot_group_peak_difference",
    "build_shap_spectrum_profile",
    "build_mean_spectrum_profile",
    "compute_gradient_shap_values",
    "plot_binary_shap_summary",
    "plot_class_shap_summary",
    "plot_mean_spectrum",
    "plot_mean_spectra_overlay",
    "plot_multiclass_shap_summary",
    "plot_shap_mean_magnitude_spectrum",
    "plot_shap_mean_signed_spectrum",
    "plot_shap_feature_importance_bar",
    "summarize_shap_feature_importance",
]
