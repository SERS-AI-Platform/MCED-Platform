"""
Data analysis and exploration for SERS datasets.

This module provides functions to:
- Analyze dataset structure (groups, samples, replicates)
- Check data completeness and identify missing replicates
- Generate comprehensive dataset reports
- Calculate group-level statistics
"""

from typing import Dict, List, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np

import pandas as pd

import logging

logger = logging.getLogger(__name__)


def analyze_dataset_structure(
    spectra_dict: Dict
) -> pd.DataFrame:
    """
    Analyze the structure of the dataset.
    
    Returns a DataFrame with:
    - Number of samples per group
    - Number of replicates per sample
    - Distribution of replicates
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> data
        
    Returns
    -------
    pd.DataFrame
        Dataset structure analysis
    """
    # Group data by (group, sample_id)
    samples = defaultdict(list)
    
    for key in spectra_dict.keys():
        group, sample_id, replicate = key
        sample_key = (group, sample_id)
        samples[sample_key].append(replicate)
    
    # Analyze structure
    data = []
    for (group, sample_id), replicates in samples.items():
        data.append({
            'group': group,
            'sample_id': sample_id,
            'n_replicates': len(replicates),
            'replicate_ids': ','.join(map(str, sorted(replicates)))
        })
    
    df = pd.DataFrame(data)
    
    return df


def get_group_statistics(
    spectra_dict: Dict
) -> pd.DataFrame:
    """
    Calculate summary statistics for each group.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> data
        
    Returns
    -------
    pd.DataFrame
        Group statistics with columns:
        - group
        - n_samples: number of unique samples
        - n_spectra: total number of spectra
        - mean_replicates: average replicates per sample
        - min_replicates: minimum replicates
        - max_replicates: maximum replicates
    """
    # Group by sample
    samples_by_group = defaultdict(lambda: defaultdict(list))
    
    for key in spectra_dict.keys():
        group, sample_id, replicate = key
        samples_by_group[group][sample_id].append(replicate)
    
    # Calculate statistics
    stats = []
    for group, samples in samples_by_group.items():
        n_samples = len(samples)
        n_spectra = sum(len(reps) for reps in samples.values())
        replicate_counts = [len(reps) for reps in samples.values()]
        
        stats.append({
            'group': group,
            'n_samples': n_samples,
            'n_spectra': n_spectra,
            'mean_replicates': np.mean(replicate_counts),
            'median_replicates': np.median(replicate_counts),
            'min_replicates': np.min(replicate_counts),
            'max_replicates': np.max(replicate_counts)
        })
    
    df = pd.DataFrame(stats).sort_values('group')
    
    return df


def check_data_completeness(
    spectra_dict: Dict,
    expected_replicates: int = 3
) -> pd.DataFrame:
    """
    Check for missing or incomplete replicate sets.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> data
    expected_replicates : int
        Expected number of replicates per sample
        
    Returns
    -------
    pd.DataFrame
        Samples with incomplete replicate sets
    """
    # Group by sample
    samples = defaultdict(list)
    
    for key in spectra_dict.keys():
        group, sample_id, replicate = key
        sample_key = (group, sample_id)
        samples[sample_key].append(replicate)
    
    # Find incomplete samples
    incomplete = []
    for (group, sample_id), replicates in samples.items():
        n_reps = len(replicates)
        
        if n_reps != expected_replicates:
            missing = set(range(1, expected_replicates + 1)) - set(replicates)
            
            incomplete.append({
                'group': group,
                'sample_id': sample_id,
                'n_replicates': n_reps,
                'expected_replicates': expected_replicates,
                'has_replicates': ','.join(map(str, sorted(replicates))),
                'missing_replicates': ','.join(map(str, sorted(missing))) if missing else 'extra',
                'status': 'incomplete' if n_reps < expected_replicates else 'extra'
            })
    
    if incomplete:
        df = pd.DataFrame(incomplete)
        logger.warning(f"Found {len(incomplete)} samples with incomplete replicate sets")
    else:
        df = pd.DataFrame()
        logger.info("All samples have complete replicate sets")
    
    return df


def get_sample_id_distribution(
    spectra_dict: Dict
) -> pd.DataFrame:
    """
    Get distribution of sample IDs across groups.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> data
        
    Returns
    -------
    pd.DataFrame
        Sample ID distribution by group
    """
    # Extract unique (group, sample_id) pairs
    samples = set()
    for key in spectra_dict.keys():
        group, sample_id, _ = key
        samples.add((group, sample_id))
    
    # Group by group
    sample_ids_by_group = defaultdict(set)
    for group, sample_id in samples:
        sample_ids_by_group[group].add(sample_id)
    
    # Create distribution
    data = []
    for group, sample_ids in sorted(sample_ids_by_group.items()):
        sorted_ids = sorted(sample_ids, key=lambda x: int(x) if x.isdigit() else x)
        
        data.append({
            'group': group,
            'n_samples': len(sample_ids),
            'sample_ids': ','.join(sorted_ids),
            'min_sample_id': min(sorted_ids, key=lambda x: int(x) if x.isdigit() else 0),
            'max_sample_id': max(sorted_ids, key=lambda x: int(x) if x.isdigit() else 0)
        })
    
    return pd.DataFrame(data)


def generate_dataset_report(
    spectra_dict: Dict,
    metadata_df: pd.DataFrame = None,
    output_path: Path = None
) -> str:
    """
    Generate comprehensive dataset report.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> data
    metadata_df : pd.DataFrame, optional
        Additional metadata (e.g., from file loading)
    output_path : Path, optional
        If provided, save report to this file
        
    Returns
    -------
    str
        Report text
    """
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("SERS DATASET REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Overall statistics
    report_lines.append("OVERALL STATISTICS")
    report_lines.append("-" * 80)
    report_lines.append(f"Total spectra: {len(spectra_dict)}")
    
    # Count unique samples
    unique_samples = set()
    for key in spectra_dict.keys():
        group, sample_id, _ = key
        unique_samples.add((group, sample_id))
    report_lines.append(f"Unique samples: {len(unique_samples)}")
    
    # Count groups
    groups = set(key[0] for key in spectra_dict.keys())
    report_lines.append(f"Groups: {len(groups)} ({', '.join(sorted(groups))})")
    report_lines.append("")
    
    # Group statistics
    report_lines.append("GROUP STATISTICS")
    report_lines.append("-" * 80)
    group_stats = get_group_statistics(spectra_dict)
    report_lines.append(group_stats.to_string(index=False))
    report_lines.append("")
    
    # Sample ID distribution
    report_lines.append("SAMPLE ID DISTRIBUTION")
    report_lines.append("-" * 80)
    sample_dist = get_sample_id_distribution(spectra_dict)
    report_lines.append(sample_dist.to_string(index=False))
    report_lines.append("")
    
    # Data completeness check
    report_lines.append("DATA COMPLETENESS CHECK")
    report_lines.append("-" * 80)
    
    # Determine expected replicates from mode
    samples = defaultdict(list)
    for key in spectra_dict.keys():
        group, sample_id, replicate = key
        sample_key = (group, sample_id)
        samples[sample_key].append(replicate)
    
    replicate_counts = [len(reps) for reps in samples.values()]
    from collections import Counter
    mode_replicates = Counter(replicate_counts).most_common(1)[0][0]
    
    incomplete = check_data_completeness(spectra_dict, expected_replicates=mode_replicates)
    if len(incomplete) > 0:
        report_lines.append(f"Expected replicates per sample: {mode_replicates}")
        report_lines.append(f"Samples with incomplete data: {len(incomplete)}")
        report_lines.append("")
        report_lines.append(incomplete.to_string(index=False))
    else:
        report_lines.append(f"✓ All samples have {mode_replicates} replicates (complete)")
    report_lines.append("")
    
    # Metadata summary (if provided)
    if metadata_df is not None:
        report_lines.append("SPECTRAL DATA CHARACTERISTICS")
        report_lines.append("-" * 80)
        report_lines.append(f"Wavenumber range: {metadata_df['x_min'].min():.2f} - {metadata_df['x_max'].max():.2f} cm⁻¹")
        report_lines.append(f"Average data points per spectrum: {metadata_df['n_points'].mean():.0f}")
        report_lines.append("")
    
    # Footer
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    # Save to file if requested
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_text)
        logger.info(f"Dataset report saved to {output_path}")
    
    return report_text


def identify_duplicate_samples(
    spectra_dict: Dict
) -> pd.DataFrame:
    """
    Identify samples that appear in multiple groups.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> data
        
    Returns
    -------
    pd.DataFrame
        Samples appearing in multiple groups
    """
    # Track which groups each sample_id appears in
    sample_groups = defaultdict(set)
    
    for key in spectra_dict.keys():
        group, sample_id, _ = key
        sample_groups[sample_id].add(group)
    
    # Find duplicates
    duplicates = []
    for sample_id, groups in sample_groups.items():
        if len(groups) > 1:
            duplicates.append({
                'sample_id': sample_id,
                'n_groups': len(groups),
                'groups': ','.join(sorted(groups))
            })
    
    if duplicates:
        df = pd.DataFrame(duplicates)
        logger.warning(f"Found {len(duplicates)} sample IDs appearing in multiple groups")
    else:
        df = pd.DataFrame()
        logger.info("No duplicate sample IDs across groups")
    
    return df


def get_replicate_distribution(
    spectra_dict: Dict
) -> pd.DataFrame:
    """
    Get distribution of replicate counts.
    
    Parameters
    ----------
    spectra_dict : dict
        Dictionary with keys (group, sample_id, replicate) -> data
        
    Returns
    -------
    pd.DataFrame
        Distribution of replicate counts
    """
    # Count replicates per sample
    samples = defaultdict(list)
    for key in spectra_dict.keys():
        group, sample_id, replicate = key
        sample_key = (group, sample_id)
        samples[sample_key].append(replicate)
    
    replicate_counts = [len(reps) for reps in samples.values()]
    
    # Create distribution
    from collections import Counter
    distribution = Counter(replicate_counts)
    
    data = []
    for n_replicates, count in sorted(distribution.items()):
        percentage = (count / len(samples)) * 100
        data.append({
            'n_replicates': n_replicates,
            'n_samples': count,
            'percentage': percentage
        })
    
    return pd.DataFrame(data)
