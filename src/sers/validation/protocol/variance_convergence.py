"""
SERS Measurement Variance Convergence Analysis

Purpose: Determine optimal number of measurements for gold standard protocol
Target: FDA submission documentation

CV Thresholds (based on literature):
- Target: ≤10% (excellent, biomedical standard)
- Acceptable: ≤15% (practical for SERS)
- Maximum: ≤20% (SERS characteristic tolerance)

References:
- Muehlethaler et al. (2016) Forensic Sci Int: SERS validation, RSD <10%
- RSC Chem Sci (2020): Commercial substrate ~20% acceptable
- Aronhime et al. (2014): Biomedical CV <5% good, 5-10% acceptable
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================
@dataclass(frozen=True)
class ConvergenceConfig:
    """Configuration for convergence analysis."""
    cv_target: float = 10.0       # Target CV (%)
    cv_acceptable: float = 15.0   # Acceptable CV (%)
    cv_maximum: float = 20.0      # Maximum tolerable CV (%)

    # Convergence detection
    cv_stability_window: int = 3  # Window for stability check
    cv_stability_threshold: float = 1.0  # % change threshold for stability

    # Bootstrap settings
    n_bootstrap: int = 1000
    confidence_level: float = 0.95

    # Marginal reduction threshold
    marginal_reduction_threshold: float = 0.5  # % CV reduction


# ============================================================
# Core Analysis Functions
# ============================================================
def calculate_cv(values: np.ndarray) -> float:
    """Calculate Coefficient of Variation (%) from array."""
    mean_val = np.mean(values)
    if mean_val == 0:
        return np.inf
    return (np.std(values, ddof=1) / mean_val) * 100


def cumulative_cv_analysis(
    measurements: np.ndarray,
    n_permutations: int = 100
) -> pd.DataFrame:
    """
    Calculate CV as function of number of measurements.

    Parameters
    ----------
    measurements : np.ndarray
        Shape (n_samples, n_measurements) or (n_measurements,) for single sample
    n_permutations : int
        Number of random orderings to average over

    Returns
    -------
    pd.DataFrame
        Columns: n_measurements, mean_cv, std_cv, ci_lower, ci_upper
    """
    if measurements.ndim == 1:
        measurements = measurements.reshape(1, -1)

    n_samples, max_measurements = measurements.shape

    results = []

    for n_meas in range(1, max_measurements + 1):
        cvs_across_permutations = []

        for _ in range(n_permutations):
            sample_cvs = []
            for sample_idx in range(n_samples):
                # Random subset of measurements
                indices = np.random.choice(max_measurements, size=n_meas, replace=False)
                subset = measurements[sample_idx, indices]

                if n_meas >= 2:
                    cv = calculate_cv(subset)
                    if np.isfinite(cv):
                        sample_cvs.append(cv)

            if sample_cvs:
                cvs_across_permutations.append(np.mean(sample_cvs))

        if cvs_across_permutations:
            mean_cv = np.mean(cvs_across_permutations)
            std_cv = np.std(cvs_across_permutations)
            ci_lower, ci_upper = np.percentile(cvs_across_permutations, [2.5, 97.5])
        else:
            mean_cv, std_cv, ci_lower, ci_upper = np.nan, np.nan, np.nan, np.nan

        results.append({
            'n_measurements': n_meas,
            'mean_cv': mean_cv,
            'std_cv': std_cv,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        })

    return pd.DataFrame(results)


def find_convergence_point(
    cv_df: pd.DataFrame,
    config: ConvergenceConfig = ConvergenceConfig()
) -> Dict:
    """
    Find the point where CV stabilizes.

    Methods:
    1. First point where CV < target threshold
    2. First point where marginal reduction < threshold
    3. First point where CV is stable for window size

    Returns
    -------
    dict
        Contains convergence points by different criteria
    """
    df = cv_df.dropna()

    result = {
        'cv_target_reached': None,
        'cv_acceptable_reached': None,
        'stability_reached': None,
        'marginal_plateau': None,
        'recommended': None
    }

    # 1. Target CV reached
    target_mask = df['mean_cv'] <= config.cv_target
    if target_mask.any():
        result['cv_target_reached'] = df[target_mask]['n_measurements'].iloc[0]

    # 2. Acceptable CV reached
    acceptable_mask = df['mean_cv'] <= config.cv_acceptable
    if acceptable_mask.any():
        result['cv_acceptable_reached'] = df[acceptable_mask]['n_measurements'].iloc[0]

    # 3. Stability (CV change < threshold over window)
    if len(df) >= config.cv_stability_window + 1:
        cv_values = df['mean_cv'].values
        for i in range(len(cv_values) - config.cv_stability_window):
            window = cv_values[i:i + config.cv_stability_window + 1]
            max_change = np.max(np.abs(np.diff(window)))
            if max_change < config.cv_stability_threshold:
                result['stability_reached'] = df.iloc[i]['n_measurements']
                break

    # 4. Marginal reduction plateau
    if len(df) >= 3:
        cv_values = df['mean_cv'].values
        reductions = -np.diff(cv_values)  # Positive = CV decreased

        for i in range(1, len(reductions)):
            if reductions[i] < config.marginal_reduction_threshold:
                result['marginal_plateau'] = df.iloc[i + 1]['n_measurements']
                break

    # Recommended: Take the earliest point that meets acceptable AND shows stability
    candidates = [v for v in [
        result['cv_acceptable_reached'],
        result['stability_reached'],
        result['marginal_plateau']
    ] if v is not None]

    if candidates:
        result['recommended'] = max(candidates)  # Take more conservative

    return result


def bootstrap_variance_stability(
    measurements: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> pd.DataFrame:
    """
    Assess variance stability using bootstrap resampling.

    At each n, bootstrap to get distribution of estimated variance.
    Stability = narrow CI of variance estimates.
    """
    if measurements.ndim == 1:
        measurements = measurements.reshape(1, -1)

    n_samples, max_meas = measurements.shape
    results = []

    for n_meas in range(2, max_meas + 1):
        bootstrap_variances = []

        for _ in range(n_bootstrap):
            # For each sample, randomly select n_meas measurements
            total_variance_estimates = []

            for sample_idx in range(n_samples):
                indices = np.random.choice(max_meas, size=n_meas, replace=True)
                subset = measurements[sample_idx, indices]
                var_estimate = np.var(subset, ddof=1)
                total_variance_estimates.append(var_estimate)

            bootstrap_variances.append(np.mean(total_variance_estimates))

        alpha = 1 - confidence_level
        ci_lower, ci_upper = np.percentile(bootstrap_variances, [alpha/2*100, (1-alpha/2)*100])
        ci_width = ci_upper - ci_lower

        results.append({
            'n_measurements': n_meas,
            'mean_variance': np.mean(bootstrap_variances),
            'variance_ci_lower': ci_lower,
            'variance_ci_upper': ci_upper,
            'variance_ci_width': ci_width,
            'relative_ci_width': ci_width / np.mean(bootstrap_variances) * 100 if np.mean(bootstrap_variances) > 0 else np.inf
        })

    return pd.DataFrame(results)


def marginal_variance_reduction(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate marginal CV reduction for each additional measurement.

    This helps answer: "What do I gain by adding one more measurement?"
    """
    df = cv_df.copy()
    df['cv_reduction'] = -df['mean_cv'].diff()  # Positive = improvement
    df['cv_reduction_pct'] = df['cv_reduction'] / df['mean_cv'].shift(1) * 100
    df['cumulative_reduction'] = df['cv_reduction'].cumsum()

    return df


# ============================================================
# Visualization Functions
# ============================================================
def plot_convergence_analysis(
    cv_df: pd.DataFrame,
    convergence_points: Dict,
    config: ConvergenceConfig = ConvergenceConfig(),
    output_path: Optional[str] = None
):
    """
    Comprehensive visualization of convergence analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. CV vs Number of Measurements
    ax1 = axes[0, 0]
    ax1.plot(cv_df['n_measurements'], cv_df['mean_cv'], 'b-o', linewidth=2, markersize=6)
    ax1.fill_between(
        cv_df['n_measurements'],
        cv_df['ci_lower'],
        cv_df['ci_upper'],
        alpha=0.3, color='blue', label='95% CI'
    )

    # Threshold lines
    ax1.axhline(y=config.cv_target, color='green', linestyle='--', linewidth=1.5, label=f'Target ({config.cv_target}%)')
    ax1.axhline(y=config.cv_acceptable, color='orange', linestyle='--', linewidth=1.5, label=f'Acceptable ({config.cv_acceptable}%)')
    ax1.axhline(y=config.cv_maximum, color='red', linestyle='--', linewidth=1.5, label=f'Maximum ({config.cv_maximum}%)')

    # Mark convergence points
    if convergence_points['recommended']:
        ax1.axvline(x=convergence_points['recommended'], color='purple', linestyle=':', linewidth=2)
        ax1.annotate(
            f"Recommended: n={convergence_points['recommended']}",
            xy=(convergence_points['recommended'], cv_df['mean_cv'].min()),
            xytext=(convergence_points['recommended'] + 1, cv_df['mean_cv'].min() + 2),
            fontsize=10, color='purple',
            arrowprops=dict(arrowstyle='->', color='purple')
        )

    ax1.set_xlabel('Number of Measurements', fontsize=12)
    ax1.set_ylabel('Coefficient of Variation (%)', fontsize=12)
    ax1.set_title('CV Convergence Analysis', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(1, cv_df['n_measurements'].max())

    # 2. Marginal CV Reduction
    ax2 = axes[0, 1]
    marginal_df = marginal_variance_reduction(cv_df)
    valid_marginal = marginal_df.dropna()

    colors = ['green' if r > config.marginal_reduction_threshold else 'gray'
              for r in valid_marginal['cv_reduction']]
    ax2.bar(valid_marginal['n_measurements'], valid_marginal['cv_reduction'], color=colors)
    ax2.axhline(y=config.marginal_reduction_threshold, color='red', linestyle='--',
                label=f'Threshold ({config.marginal_reduction_threshold}%)')

    ax2.set_xlabel('Number of Measurements', fontsize=12)
    ax2.set_ylabel('CV Reduction (%)', fontsize=12)
    ax2.set_title('Marginal CV Reduction per Additional Measurement', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Relative CI Width (Variance Stability)
    ax3 = axes[1, 0]
    ax3.plot(cv_df['n_measurements'], cv_df['std_cv'], 'g-o', linewidth=2, markersize=6)
    ax3.set_xlabel('Number of Measurements', fontsize=12)
    ax3.set_ylabel('Standard Deviation of CV (%)', fontsize=12)
    ax3.set_title('CV Estimate Stability', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. Summary Table
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════╗
    ║           CONVERGENCE ANALYSIS SUMMARY                   ║
    ╠══════════════════════════════════════════════════════════╣
    ║                                                          ║
    ║  CV Thresholds (Literature-based):                       ║
    ║  ─────────────────────────────────                       ║
    ║  • Target (Excellent):    ≤{config.cv_target:5.1f}%                        ║
    ║  • Acceptable:            ≤{config.cv_acceptable:5.1f}%                        ║
    ║  • Maximum (SERS limit):  ≤{config.cv_maximum:5.1f}%                        ║
    ║                                                          ║
    ║  Convergence Points:                                     ║
    ║  ───────────────────                                     ║
    ║  • Target CV reached at:      n = {str(convergence_points['cv_target_reached'] or 'N/A'):>4}               ║
    ║  • Acceptable CV reached at:  n = {str(convergence_points['cv_acceptable_reached'] or 'N/A'):>4}               ║
    ║  • Stability reached at:      n = {str(convergence_points['stability_reached'] or 'N/A'):>4}               ║
    ║  • Marginal plateau at:       n = {str(convergence_points['marginal_plateau'] or 'N/A'):>4}               ║
    ║                                                          ║
    ║  ═══════════════════════════════════════════════════     ║
    ║  ★ RECOMMENDED: n = {str(convergence_points['recommended'] or 'N/A'):>4} measurements              ║
    ║  ═══════════════════════════════════════════════════     ║
    ║                                                          ║
    ║  Final CV at max measurements:                           ║
    ║  {cv_df['mean_cv'].iloc[-1]:5.2f}% ± {cv_df['std_cv'].iloc[-1]:5.2f}% (95% CI: {cv_df['ci_lower'].iloc[-1]:5.2f}-{cv_df['ci_upper'].iloc[-1]:5.2f}%)       ║
    ╚══════════════════════════════════════════════════════════╝
    """

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {output_path}")

    plt.show()
    return fig


def plot_peak_specific_convergence(
    measurements: np.ndarray,
    peak_names: List[str],
    config: ConvergenceConfig = ConvergenceConfig(),
    output_path: Optional[str] = None
):
    """
    Analyze convergence for multiple SERS peaks.

    Parameters
    ----------
    measurements : np.ndarray
        Shape (n_peaks, n_samples, n_measurements)
    peak_names : list
        Names of peaks (e.g., wavenumber positions)
    """
    n_peaks = len(peak_names)
    fig, axes = plt.subplots(1, n_peaks, figsize=(5 * n_peaks, 4), sharey=True)

    if n_peaks == 1:
        axes = [axes]

    summary_results = []

    for idx, (ax, peak_name) in enumerate(zip(axes, peak_names)):
        peak_data = measurements[idx]
        cv_df = cumulative_cv_analysis(peak_data)
        convergence = find_convergence_point(cv_df, config)

        ax.plot(cv_df['n_measurements'], cv_df['mean_cv'], '-o', linewidth=2, markersize=5)
        ax.fill_between(cv_df['n_measurements'], cv_df['ci_lower'], cv_df['ci_upper'], alpha=0.3)

        ax.axhline(y=config.cv_target, color='green', linestyle='--', alpha=0.7)
        ax.axhline(y=config.cv_acceptable, color='orange', linestyle='--', alpha=0.7)

        if convergence['recommended']:
            ax.axvline(x=convergence['recommended'], color='red', linestyle=':', linewidth=2)

        ax.set_xlabel('n Measurements')
        ax.set_title(f'Peak {peak_name}')
        ax.grid(True, alpha=0.3)

        summary_results.append({
            'peak': peak_name,
            'recommended_n': convergence['recommended'],
            'final_cv': cv_df['mean_cv'].iloc[-1],
            'target_reached': convergence['cv_target_reached']
        })

    axes[0].set_ylabel('CV (%)')

    plt.suptitle('Peak-Specific CV Convergence Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()

    return pd.DataFrame(summary_results)


# ============================================================
# Main Analysis Pipeline
# ============================================================
def run_convergence_analysis(
    measurements: np.ndarray,
    sample_ids: Optional[List[str]] = None,
    config: ConvergenceConfig = ConvergenceConfig(),
    output_dir: Optional[str] = None
) -> Dict:
    """
    Complete convergence analysis pipeline.

    Parameters
    ----------
    measurements : np.ndarray
        Shape (n_samples, n_measurements) for single metric
        or (n_samples, n_measurements, n_features) for spectral data
    sample_ids : list, optional
        Sample identifiers
    config : ConvergenceConfig
        Analysis configuration
    output_dir : str, optional
        Directory to save results

    Returns
    -------
    dict
        Complete analysis results
    """
    print("=" * 60)
    print("SERS Measurement Variance Convergence Analysis")
    print("=" * 60)

    # Handle spectral data: use mean intensity or specific peak
    if measurements.ndim == 3:
        print(f"\nInput: {measurements.shape[0]} samples × {measurements.shape[1]} measurements × {measurements.shape[2]} features")
        print("Using mean spectral intensity for convergence analysis...")
        measurements_1d = measurements.mean(axis=2)
    else:
        measurements_1d = measurements
        print(f"\nInput: {measurements_1d.shape[0]} samples × {measurements_1d.shape[1]} measurements")

    # 1. Cumulative CV Analysis
    print("\n[1/4] Calculating cumulative CV...")
    cv_df = cumulative_cv_analysis(measurements_1d, n_permutations=100)

    # 2. Find Convergence Points
    print("[2/4] Finding convergence points...")
    convergence_points = find_convergence_point(cv_df, config)

    # 3. Bootstrap Variance Stability
    print("[3/4] Assessing variance stability (bootstrap)...")
    bootstrap_df = bootstrap_variance_stability(
        measurements_1d,
        n_bootstrap=config.n_bootstrap,
        confidence_level=config.confidence_level
    )

    # 4. Marginal Reduction
    print("[4/4] Calculating marginal reduction...")
    marginal_df = marginal_variance_reduction(cv_df)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print("\n• CV Thresholds:")
    print(f"  - Target (Excellent): ≤{config.cv_target}%")
    print(f"  - Acceptable: ≤{config.cv_acceptable}%")
    print(f"  - Maximum: ≤{config.cv_maximum}%")

    print("\n• Convergence Points:")
    print(f"  - Target CV ({config.cv_target}%) reached at: n = {convergence_points['cv_target_reached'] or 'Not reached'}")
    print(f"  - Acceptable CV ({config.cv_acceptable}%) reached at: n = {convergence_points['cv_acceptable_reached'] or 'Not reached'}")
    print(f"  - Stability reached at: n = {convergence_points['stability_reached'] or 'Not reached'}")
    print(f"  - Marginal plateau at: n = {convergence_points['marginal_plateau'] or 'Not reached'}")

    print(f"\n★ RECOMMENDED MEASUREMENTS: n = {convergence_points['recommended'] or 'Unable to determine'}")

    print(f"\n• Final Performance (n={measurements_1d.shape[1]}):")
    print(f"  - Mean CV: {cv_df['mean_cv'].iloc[-1]:.2f}%")
    print(f"  - 95% CI: [{cv_df['ci_lower'].iloc[-1]:.2f}%, {cv_df['ci_upper'].iloc[-1]:.2f}%]")

    # Visualization
    print("\n" + "=" * 60)
    print("Generating visualization...")

    output_path = None
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'convergence_analysis.png')

    fig = plot_convergence_analysis(cv_df, convergence_points, config, output_path)

    # Save results
    if output_dir:
        cv_df.to_csv(os.path.join(output_dir, 'cv_convergence.csv'), index=False)
        bootstrap_df.to_csv(os.path.join(output_dir, 'bootstrap_stability.csv'), index=False)
        marginal_df.to_csv(os.path.join(output_dir, 'marginal_reduction.csv'), index=False)

        # Summary report
        with open(os.path.join(output_dir, 'convergence_report.txt'), 'w') as f:
            f.write("SERS Measurement Variance Convergence Analysis Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Date: {pd.Timestamp.now()}\n")
            f.write(f"Samples: {measurements_1d.shape[0]}\n")
            f.write(f"Max Measurements: {measurements_1d.shape[1]}\n\n")
            f.write("CV Thresholds (Literature-based):\n")
            f.write(f"  Target: ≤{config.cv_target}%\n")
            f.write(f"  Acceptable: ≤{config.cv_acceptable}%\n")
            f.write(f"  Maximum: ≤{config.cv_maximum}%\n\n")
            f.write("Convergence Points:\n")
            for key, value in convergence_points.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"\nRECOMMENDED: n = {convergence_points['recommended']} measurements\n")

        print(f"\nResults saved to: {output_dir}")

    return {
        'cv_df': cv_df,
        'convergence_points': convergence_points,
        'bootstrap_df': bootstrap_df,
        'marginal_df': marginal_df,
        'config': config,
        'figure': fig
    }


# ============================================================
# Demo with Synthetic Data
# ============================================================
def generate_demo_data(
    n_samples: int = 50,
    n_measurements: int = 20,
    true_cv: float = 8.0,
    seed: int = 42
) -> np.ndarray:
    """
    Generate synthetic SERS-like measurement data for testing.

    Simulates realistic SERS variance characteristics:
    - Sample-to-sample variation (biological)
    - Measurement-to-measurement variation (technical)
    """
    np.random.seed(seed)

    # Sample means (biological variation)
    sample_means = np.random.lognormal(mean=5.0, sigma=0.5, size=n_samples)

    # Technical variation per measurement
    data = np.zeros((n_samples, n_measurements))

    for i, mean in enumerate(sample_means):
        # CV = std/mean * 100 -> std = mean * CV / 100
        std = mean * true_cv / 100
        data[i, :] = np.random.normal(loc=mean, scale=std, size=n_measurements)
        data[i, :] = np.maximum(data[i, :], 0.1)  # SERS intensities are positive

    return data


if __name__ == "__main__":
    # Demo run
    print("Running demo with synthetic data...\n")

    # Generate synthetic data mimicking 50 samples × 20 measurements
    demo_data = generate_demo_data(n_samples=50, n_measurements=20, true_cv=12.0)

    # Run analysis
    config = ConvergenceConfig(
        cv_target=10.0,
        cv_acceptable=15.0,
        cv_maximum=20.0
    )

    results = run_convergence_analysis(
        demo_data,
        config=config,
        output_dir='/home/claude/convergence_results'
    )
