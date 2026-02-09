"""
SERS Preprocessing Pipeline - Main Orchestration Script

This script coordinates the entire preprocessing workflow:
1. Load configuration and validate data
2. Read and parse raw spectra
3. Preprocess spectra (smoothing, baseline, normalization)
4. Calculate variance and QC metrics
5. Generate visualizations
6. Save results

Usage:
    python main.py
"""

import sys
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.sers.config import RAW_DATA_DIR, RESULTS_DIR, FIG_DIR
from src.sers import load_config, read_spectrum, parse_filename
from src.sers.io import find_spectra, make_common_grid

# Import new modularized functions
from src.sers.preprocessing import (
    preprocess_spectra,
    calculate_replicate_variance,
    identify_problematic_samples,
    save_processed_spectra
)
from src.sers.visualization import (
    visualize_preprocessing,
    plot_replicate_variance_by_group,
    plot_variance_heatmap,
    plot_cv_boxplot
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)


# ==================== Exception Classes ====================
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    pass


class DataNotFoundError(PipelineError):
    """Raised when data directory or files are not found."""
    pass


class ConfigurationError(PipelineError):
    """Raised when configuration loading fails."""
    pass


# ==================== Helper Functions ====================
def validate_data_directory(data_dir: Path) -> None:
    """Validate that data directory exists and contains spectrum files."""
    if not data_dir.exists():
        msg = f"Data directory {data_dir} does not exist."
        logger.error(msg)
        raise DataNotFoundError(msg)

    files = list(find_spectra(data_dir))
    if not files:
        msg = f"No spectrum files found in {data_dir}."
        logger.error(msg)
        raise DataNotFoundError(msg)


def load_raw_spectra(files: list[Path], folder_mapping: dict) -> tuple:
    """
    Load raw spectra from files and create metadata DataFrame.
    
    Returns
    -------
    raw_spectra : dict
        Dictionary with (group, sample_id, replicate) -> (x, y)
    metadata : pd.DataFrame
        Metadata for each spectrum
    """
    raw_spectra = {}
    metadata = []

    for file_path in files:
        try:
            spec_id = parse_filename(
                file_path,
                fallback_group=folder_mapping.get(file_path.parent.name, "UNK")
            )
            
            x, y = read_spectrum(file_path)

            # Store spectrum
            key = (spec_id.group, spec_id.sample_id, spec_id.replicate)
            raw_spectra[key] = (x, y)

            # Collect metadata
            metadata.append({
                "file": file_path.name,
                "group": spec_id.group,
                "sample_id": spec_id.sample_id,
                "replicate": spec_id.replicate,
                "n_points": len(x),
                "x_min": x.min(),
                "x_max": x.max()
            })
        
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            continue

    if not raw_spectra:
        raise DataNotFoundError("All files failed to load.")
    
    return raw_spectra, pd.DataFrame(metadata)


def display_data_statistics(meta_df: pd.DataFrame) -> None:
    """Display summary statistics of loaded data."""
    logger.info("\n=== Data Statistics ===")
    
    # Group summary
    group_summary = meta_df.groupby("group").agg({
        "sample_id": "nunique",
        "replicate": "count"
    }).rename(columns={"sample_id": "n_samples", "replicate": "n_spectra"})
    
    logger.info("\nGroup Summary:")
    logger.info(f"\n{group_summary}")

    # Raman shift range
    x_mins = meta_df["x_min"].min()
    x_maxs = meta_df["x_max"].max()
    logger.info(f"\nRaman shift range: {x_mins:.2f} to {x_maxs:.2f} cm⁻¹")


# ==================== Main Pipeline ====================
def main():
    """Main pipeline orchestration."""
    
    try:
        # ========== Step 1: Load Configuration ==========
        logger.info("\n" + "="*60)
        logger.info("SERS Preprocessing Pipeline Started")
        logger.info("="*60)
        
        logger.info("\n[Step 1] Loading configuration...")
        config = load_config("config.yaml")
        logger.info("Configuration loaded successfully.")
        
        logger.info(f"Folder mapping: {len(config.folder_to_group)} entries")
        for folder, group in list(config.folder_to_group.items())[:3]:
            logger.info(f"  {folder} -> {group}")
        if len(config.folder_to_group) > 3:
            logger.info(f"  ... and {len(config.folder_to_group) - 3} more")

        # ========== Step 2: Validate and Load Data ==========
        logger.info("\n[Step 2] Validating data directory...")
        data_dir = Path(RAW_DATA_DIR)
        validate_data_directory(data_dir)
        
        files = list(find_spectra(data_dir, pattern="*.csv"))
        logger.info(f"Found {len(files)} spectrum files")
        for f in files[:3]:
            logger.info(f"  {f.name}")
        if len(files) > 3:
            logger.info(f"  ... and {len(files) - 3} more")
        
        logger.info(f"\n[Step 3] Loading raw spectra...")
        raw_spectra, meta_df = load_raw_spectra(files, config.folder_to_group)
        logger.info(f"Loaded {len(raw_spectra)} spectra successfully")
        
        # Display statistics
        display_data_statistics(meta_df)
        
        # Save metadata
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(FIG_DIR, exist_ok=True)
        
        meta_csv_path = RESULTS_DIR / "metadata_summary.csv"
        meta_df.to_csv(meta_csv_path, index=False)
        logger.info(f"\nMetadata saved to {meta_csv_path}")

        # ========== Step 3: Dataset Exploration ==========
        from src.sers.analysis import (
            get_group_statistics,
            check_data_completeness,
            generate_dataset_report,
            get_replicate_distribution,
            get_sample_id_distribution,
            identify_duplicate_samples
        )
        from src.sers.visualization import (
            plot_dataset_overview,
            plot_replicate_distribution,
            plot_sample_id_ranges
        )
        
        logger.info("\n[Step 3] Analyzing dataset structure...")
        
        # Get group statistics
        group_stats = get_group_statistics(raw_spectra)
        logger.info("\nGroup Statistics:")
        logger.info(f"\n{group_stats.to_string(index=False)}")
        
        group_stats_csv = RESULTS_DIR / "group_statistics.csv"
        group_stats.to_csv(group_stats_csv, index=False)
        
        # Check data completeness
        completeness = check_data_completeness(raw_spectra, expected_replicates=3)
        if len(completeness) > 0:
            logger.warning(f"\n⚠️  Found {len(completeness)} samples with incomplete data:")
            logger.warning(f"\n{completeness.to_string(index=False)}")
            completeness.to_csv(RESULTS_DIR / "incomplete_samples.csv", index=False)
        
        # Check for duplicate sample IDs
        duplicates = identify_duplicate_samples(raw_spectra)
        if len(duplicates) > 0:
            logger.warning(f"\n⚠️  Found {len(duplicates)} sample IDs in multiple groups:")
            logger.warning(f"\n{duplicates.to_string(index=False)}")
            duplicates.to_csv(RESULTS_DIR / "duplicate_sample_ids.csv", index=False)
        
        # Generate comprehensive report
        report = generate_dataset_report(
            raw_spectra,
            metadata_df=meta_df,
            output_path=RESULTS_DIR / "dataset_report.txt"
        )
        logger.info("\n" + report)
        
        # Generate dataset overview visualizations
        logger.info("\n[Step 3.1] Generating dataset overview visualizations...")
        plot_dataset_overview(group_stats, output_dir=FIG_DIR)
        
        replicate_dist = get_replicate_distribution(raw_spectra)
        plot_replicate_distribution(replicate_dist, output_dir=FIG_DIR)
        
        sample_dist = get_sample_id_distribution(raw_spectra)
        plot_sample_id_ranges(sample_dist, output_dir=FIG_DIR)
        
        logger.info("Dataset exploration visualizations complete")

        # ========== Step 4: Preprocessing ==========
        logger.info("\n[Step 4] Preprocessing spectra...")
        x_arrays = [x for x, y in raw_spectra.values()]
        common_grid = make_common_grid(x_arrays)
        logger.info(f"Common grid: {len(common_grid)} points from {common_grid[0]:.2f} to {common_grid[-1]:.2f} cm⁻¹")
        
        processed_spectra, prep_stats_df = preprocess_spectra(
            raw_spectra,
            common_grid,
            config
        )
        logger.info(f"Preprocessed {len(processed_spectra)} spectra")
        
        prep_stats_csv = RESULTS_DIR / "preprocessing_stats.csv"
        prep_stats_df.to_csv(prep_stats_csv, index=False)
        logger.info(f"Preprocessing stats saved to {prep_stats_csv}")

        # ========== Step 5: Variance Analysis ==========
        logger.info("\n[Step 5] Calculating replicate variance...")
        variance_df = calculate_replicate_variance(
            processed_spectra,
            common_grid,
            peak_region=(600, 1800)
        )
        
        logger.info("\nVariance Summary:")
        logger.info(f"  Mean CV: {variance_df['mean_cv'].mean():.2f} ± {variance_df['mean_cv'].std():.2f}%")
        logger.info(f"  Mean correlation: {variance_df['mean_pairwise_correlation'].mean():.3f}")
        
        # Identify problematic samples
        problematic = identify_problematic_samples(
            variance_df,
            cv_threshold=15.0,
            correlation_threshold=0.90
        )
        
        if len(problematic) > 0:
            logger.warning(f"\n⚠️  Found {len(problematic)} problematic samples:")
            logger.warning(f"\n{problematic[['group', 'sample_id', 'mean_cv', 'mean_pairwise_correlation', 'failure_reason']]}")
        else:
            logger.info("\n✓ All samples passed QC thresholds")
        
        # Save variance statistics
        variance_csv = RESULTS_DIR / "replicate_variance_stats.csv"
        variance_df.to_csv(variance_csv, index=False)
        logger.info(f"\nVariance stats saved to {variance_csv}")
        
        if len(problematic) > 0:
            problematic_csv = RESULTS_DIR / "problematic_samples.csv"
            problematic.to_csv(problematic_csv, index=False)
            logger.info(f"Problematic samples saved to {problematic_csv}")

        # ========== Step 6: Visualization ==========
        logger.info("\n[Step 6] Generating visualizations...")
        
        # Raw vs preprocessed comparison
        visualize_preprocessing(
            raw_spectra,
            processed_spectra,
            common_grid,
            output_dir=FIG_DIR,
            n_examples=3
        )
        
        # Variance plots by group
        plot_replicate_variance_by_group(
            variance_df,
            output_dir=FIG_DIR,
            cv_threshold=15.0
        )
        
        # Variance heatmap
        plot_variance_heatmap(variance_df, output_dir=FIG_DIR)
        
        # CV boxplots
        plot_cv_boxplot(variance_df, output_dir=FIG_DIR)
        
        logger.info("All visualizations generated")

        # ========== Step 7: Save Processed Data ==========
        logger.info("\n[Step 7] Saving processed spectra...")
        save_processed_spectra(
            processed_spectra,
            common_grid,
            output_path=RESULTS_DIR / "processed_spectra.csv"
        )

        # ========== Pipeline Complete ==========
        logger.info("\n" + "="*60)
        logger.info("Pipeline completed successfully!")
        logger.info("="*60)
        logger.info(f"\nResults saved to: {RESULTS_DIR}")
        logger.info(f"Figures saved to: {FIG_DIR}")
        
        return 0  # Success

    except ConfigurationError as e:
        logger.error(f"Configuration error: {e}")
        return 1
    except DataNotFoundError as e:
        logger.error(f"Data error: {e}")
        return 2
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
