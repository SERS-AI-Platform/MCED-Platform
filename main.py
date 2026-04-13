"""
SERS Preprocessing Pipeline - Main Orchestration Script

This script coordinates the entire preprocessing workflow:
1. Load configuration and validate data
2. Read and parse raw spectra
3. Analyze dataset structure
4. Preprocess spectra (smoothing, baseline, normalization)
5. Calculate variance and QC metrics
6. Generate visualizations
7. Save results

Usage:
    python main.py                    # Thermo (raw_data) — default
    python main.py --data-source medical  # Medical (raw_data_medical)
"""

import argparse
import sys
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.sers.config import load_config, RAW_DATA_DIR, RAW_DATA_MEDICAL_DIR, RESULTS_DIR, FIG_DIR
from src.sers.io import (
    read_spectrum, parse_filename, find_spectra, make_common_grid,
    load_dataset, filter_spectra,
)

# Preprocessing
from src.sers.preprocessing import (
    preprocess_spectra,
    calculate_replicate_variance,
    identify_problematic_samples,
    save_processed_spectra,
)

# Analysis
from src.sers.analysis import (
    get_group_statistics,
    check_data_completeness,
    generate_dataset_report,
    get_replicate_distribution,
    get_sample_id_distribution,
    identify_duplicate_samples,
)

# Visualization (only functions that exist)
from src.sers.visualization import (
    plot_sample_distribution_pie,
    plot_spectra_count_bar,
    plot_replicate_variance_by_group,
    plot_variance_heatmap,
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
def validate_data_directory(data_dir: Path, pattern: str = "*.csv") -> None:
    """Validate that data directory exists and contains spectrum files."""
    if not data_dir.exists():
        msg = f"Data directory {data_dir} does not exist."
        logger.error(msg)
        raise DataNotFoundError(msg)

    files = list(find_spectra(data_dir, pattern=pattern))
    if not files:
        msg = f"No spectrum files ({pattern}) found in {data_dir}."
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
    logger.info(f"\nRaman shift range: {x_mins:.2f} to {x_maxs:.2f} cm-1")


# ==================== Main Pipeline ====================
def parse_args():
    parser = argparse.ArgumentParser(description="SERS Preprocessing Pipeline")
    parser.add_argument(
        "--data-source",
        choices=["thermo", "medical"],
        default="thermo",
        help="Data source: 'thermo' (raw_data, default) or 'medical' (raw_data_medical)",
    )
    return parser.parse_args()


def main():
    """Main pipeline orchestration."""
    args = parse_args()
    is_medical = args.data_source == "medical"

    try:
        # ========== Step 1: Load Configuration ==========
        logger.info("\n" + "="*60)
        logger.info("SERS Preprocessing Pipeline Started")
        logger.info(f"Data source: {args.data_source}")
        logger.info("="*60)

        logger.info("\n[Step 1] Loading configuration...")
        config = load_config("config/config.yaml")
        logger.info("Configuration loaded successfully.")

        # Select data source
        if is_medical:
            data_dir = Path(RAW_DATA_MEDICAL_DIR)
            folder_mapping = config.folder_to_group_medical
            file_pattern = config.medical.file_pattern
            exclude_patterns = config.medical.exclude_patterns
            read_subdir = config.medical.read_from_subdir
            logger.info(f"Medical mode: read_from={read_subdir}/, "
                        f"SG smooth={config.medical.do_smooth}, calibration={config.medical.do_calibration}")
        else:
            data_dir = Path(RAW_DATA_DIR)
            folder_mapping = config.folder_to_group
            file_pattern = "*.csv"
            exclude_patterns = None
            read_subdir = None

        logger.info(f"Folder mapping: {len(folder_mapping)} entries")
        for folder, group in list(folder_mapping.items())[:3]:
            logger.info(f"  {folder} -> {group}")
        if len(folder_mapping) > 3:
            logger.info(f"  ... and {len(folder_mapping) - 3} more")

        # ========== Step 2: Validate and Load Data ==========
        logger.info("\n[Step 2] Validating data directory...")
        validate_data_directory(data_dir, pattern=file_pattern)

        # Use load_dataset for medical (supports BG subtraction + filtering)
        if is_medical:
            wn_shift = config.medical.wavenumber_shift
            logger.info(f"\nLoading medical spectra (pattern={file_pattern}, subdir={read_subdir}, shift={wn_shift:+.1f} cm⁻¹)...")
            dataset = load_dataset(
                data_dir,
                folder_to_group=folder_mapping,
                pattern=file_pattern,
                exclude_patterns=exclude_patterns,
                read_from_subdir=read_subdir,
                wavenumber_shift=wn_shift,
            )
            raw_spectra = dataset.spectra
            meta_df = dataset.metadata
            logger.info(f"Loaded {len(raw_spectra)} spectra, {len(dataset.failed_files)} failed")
        else:
            files = list(find_spectra(data_dir, pattern=file_pattern))
            logger.info(f"Found {len(files)} spectrum files")
            for f in files[:3]:
                logger.info(f"  {f.name}")
            if len(files) > 3:
                logger.info(f"  ... and {len(files) - 3} more")

            logger.info(f"\nLoading raw spectra...")
            raw_spectra, meta_df = load_raw_spectra(files, folder_mapping)
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
        logger.info("\n[Step 3] Analyzing dataset structure...")

        # Get group statistics
        group_stats = get_group_statistics(raw_spectra)
        logger.info("\nGroup Statistics:")
        logger.info(f"\n{group_stats.to_string(index=False)}")

        group_stats_csv = RESULTS_DIR / "group_statistics.csv"
        group_stats.to_csv(group_stats_csv, index=False)

        # Check data completeness
        expected_reps = config.medical.expected_reps if is_medical else config.qc.expected_reps
        completeness = check_data_completeness(
            raw_spectra, expected_replicates=expected_reps
        )
        if len(completeness) > 0:
            logger.warning(f"\nFound {len(completeness)} samples with incomplete data:")
            logger.warning(f"\n{completeness.to_string(index=False)}")
            completeness.to_csv(RESULTS_DIR / "incomplete_samples.csv", index=False)

        # Check for duplicate sample IDs
        duplicates = identify_duplicate_samples(raw_spectra)
        if len(duplicates) > 0:
            logger.warning(f"\nFound {len(duplicates)} sample IDs in multiple groups:")
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
        plot_sample_distribution_pie(group_stats, output_dir=FIG_DIR)
        plot_spectra_count_bar(group_stats, output_dir=FIG_DIR)
        logger.info("Dataset exploration visualizations complete")

        # ========== Step 4: Preprocessing ==========
        logger.info("\n[Step 4] Preprocessing spectra...")
        x_arrays = [x for x, y in raw_spectra.values()]
        common_grid = make_common_grid(x_arrays)
        logger.info(f"Common grid: {len(common_grid)} points from {common_grid[0]:.2f} to {common_grid[-1]:.2f} cm-1")

        # Medical mode: override preprocessing params (no smoothing, no calibration)
        preprocess_config = config
        if is_medical:
            from dataclasses import replace
            from src.sers.config import PreprocessingConfig
            medical_prep = replace(
                config.preprocessing,
                do_smooth=config.medical.do_smooth,
                do_calibration=config.medical.do_calibration,
            )
            preprocess_config = replace(config, preprocessing=medical_prep)
            logger.info("  Medical overrides: do_smooth=False, do_calibration=False")

        processed_spectra, prep_stats_df, proc_grid = preprocess_spectra(
            raw_spectra,
            common_grid,
            preprocess_config
        )
        logger.info(f"Preprocessed {len(processed_spectra)} spectra")

        prep_stats_csv = RESULTS_DIR / "preprocessing_stats.csv"
        prep_stats_df.to_csv(prep_stats_csv, index=False)
        logger.info(f"Preprocessing stats saved to {prep_stats_csv}")

        # ========== Step 5: Variance Analysis ==========
        logger.info("\n[Step 5] Calculating replicate variance...")
        variance_df = calculate_replicate_variance(
            processed_spectra,
            proc_grid,
            peak_region=(600, 1800)
        )

        logger.info("\nVariance Summary:")
        logger.info(f"  Mean CV: {variance_df['mean_cv'].mean():.2f} +/- {variance_df['mean_cv'].std():.2f}%")
        logger.info(f"  Mean correlation: {variance_df['mean_pairwise_correlation'].mean():.3f}")

        # Identify problematic samples
        problematic = identify_problematic_samples(
            variance_df,
            cv_threshold=15.0,
            correlation_threshold=0.90
        )

        if len(problematic) > 0:
            logger.warning(f"\nFound {len(problematic)} problematic samples:")
            logger.warning(f"\n{problematic[['group', 'sample_id', 'mean_cv', 'mean_pairwise_correlation', 'failure_reason']]}")
        else:
            logger.info("\nAll samples passed QC thresholds")

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

        # Variance plots by group
        plot_replicate_variance_by_group(
            variance_df,
            output_dir=FIG_DIR,
            cv_threshold=15.0
        )

        # Variance heatmap
        plot_variance_heatmap(variance_df, output_dir=FIG_DIR)

        logger.info("All visualizations generated")

        # ========== Step 7: Save Processed Data ==========
        logger.info("\n[Step 7] Saving processed spectra...")
        save_processed_spectra(
            processed_spectra,
            proc_grid,
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
