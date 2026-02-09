"""
Quick Data Exploration Script

데이터 전처리 없이 전체 데이터셋 구조만 빠르게 파악하는 스크립트

Usage:
    python explore_data.py
"""

import logging
from pathlib import Path

from src.sers.config import RAW_DATA_DIR, RESULTS_DIR, FIG_DIR
from src.sers import load_config, parse_filename, find_spectra, read_spectrum
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """빠른 데이터 탐색."""
    
    logger.info("\n" + "="*60)
    logger.info("SERS Dataset Quick Exploration")
    logger.info("="*60)
    
    # Load config
    config = load_config("config.yaml")
    
    # Find files
    data_dir = Path(RAW_DATA_DIR)
    files = list(find_spectra(data_dir, pattern="*.csv"))
    logger.info(f"\nFound {len(files)} spectrum files")
    
    # Parse filenames only (don't read full data)
    raw_spectra = {}
    for file_path in files:
        try:
            spec_id = parse_filename(
                file_path,
                fallback_group=config.folder_to_group.get(file_path.parent.name, "UNK")
            )
            
            # Just create a placeholder
            key = (spec_id.group, spec_id.sample_id, spec_id.replicate)
            raw_spectra[key] = None  # Don't load actual data
            
        except Exception as e:
            logger.warning(f"Failed to parse {file_path.name}: {e}")
    
    logger.info(f"Parsed {len(raw_spectra)} spectra")
    
    # Analyze structure
    logger.info("\n" + "="*60)
    logger.info("Dataset Structure Analysis")
    logger.info("="*60)
    
    # Group statistics
    group_stats = get_group_statistics(raw_spectra)
    logger.info("\nGroup Statistics:")
    logger.info(f"\n{group_stats.to_string(index=False)}")
    
    # Data completeness
    completeness = check_data_completeness(raw_spectra, expected_replicates=3)
    if len(completeness) > 0:
        logger.warning(f"\n⚠️  Found {len(completeness)} samples with incomplete data:")
        logger.warning(f"\n{completeness.to_string(index=False)}")
    else:
        logger.info("\n✓ All samples have complete replicate sets")
    
    # Duplicate check
    duplicates = identify_duplicate_samples(raw_spectra)
    if len(duplicates) > 0:
        logger.warning(f"\n⚠️  Found {len(duplicates)} duplicate sample IDs:")
        logger.warning(f"\n{duplicates.to_string(index=False)}")
    
    # Generate report
    logger.info("\nGenerating comprehensive report...")
    report = generate_dataset_report(
        raw_spectra,
        output_path=RESULTS_DIR / "dataset_report_quick.txt"
    )
    
    # Save CSVs
    group_stats.to_csv(RESULTS_DIR / "group_statistics.csv", index=False)
    if len(completeness) > 0:
        completeness.to_csv(RESULTS_DIR / "incomplete_samples.csv", index=False)
    if len(duplicates) > 0:
        duplicates.to_csv(RESULTS_DIR / "duplicate_sample_ids.csv", index=False)
    
    # Generate visualizations
    logger.info("\nGenerating visualizations...")
    
    plot_dataset_overview(group_stats, output_dir=FIG_DIR)
    
    replicate_dist = get_replicate_distribution(raw_spectra)
    plot_replicate_distribution(replicate_dist, output_dir=FIG_DIR)
    
    sample_dist = get_sample_id_distribution(raw_spectra)
    plot_sample_id_ranges(sample_dist, output_dir=FIG_DIR)
    
    logger.info("\n" + "="*60)
    logger.info("Exploration Complete!")
    logger.info("="*60)
    logger.info(f"\nResults saved to: {RESULTS_DIR}")
    logger.info(f"Figures saved to: {FIG_DIR}")
    logger.info(f"\nSee dataset_report_quick.txt for full report")


if __name__ == "__main__":
    main()
