"""
Visualization functions for SERS spectroscopy data.
Each function saves ONE plot to ONE file (no subplots).
"""

from pathlib import Path
from typing import Dict,List,Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


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