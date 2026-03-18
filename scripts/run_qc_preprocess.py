"""
SERS Pipeline: QC → Preprocessing

QC를 통과한 스펙트럼만 전처리하는 파이프라인.

Usage:
    python run_qc_preprocess.py
    python run_qc_preprocess.py --config config.yaml --normalization snv
    python run_qc_preprocess.py --normalization minmax --no-trim

Pipeline:
    ┌─────────────────────────────────┐
    │  1. Load raw spectra            │
    │  2. Create common grid          │
    │  3. Run QC pipeline             │
    │     ├─ Intensity gate           │
    │     └─ Replicate QC (RSD/Corr)  │
    │  4. Preprocess QC-passed only   │
    │     ├─ ① Trim (400-2200 cm⁻¹)  │
    │     ├─ ② Smooth (Savitzky-Golay)│
    │     ├─ ③ Baseline correction    │
    │     └─ ④ Normalize (SNV/etc)    │
    │  5. Save results                │
    └─────────────────────────────────┘
"""

import sys
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# ─── Package imports ───
from src.sers.config import load_config, RAW_DATA_DIR, RESULTS_DIR, FIG_DIR
from src.sers import read_spectrum, parse_filename
from src.sers.io import find_spectra, make_common_grid

# QC pipeline
from src.sers.qc.qc import run_qc_pipeline

# Preprocessing (updated module with trim + multi-normalization)
from src.sers.preprocessing import (
    preprocess_spectra,
    calculate_replicate_variance,
    identify_problematic_samples,
    save_processed_spectra,
)


# ─── Logging ───
def setup_logging(log_file: str = "pipeline_qc_preprocess.log"):
    """Configure logging to both console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        ],
    )
    return logging.getLogger(__name__)


# ─── Exceptions ───
class PipelineError(Exception):
    pass

class DataNotFoundError(PipelineError):
    pass


# ─── Step 1: Load Data ───
def load_raw_spectra(
    data_dir: Path,
    folder_mapping: dict,
    pattern: str = "*.csv",
) -> tuple[dict, pd.DataFrame]:
    """
    Load all raw spectra and build metadata DataFrame.

    Returns
    -------
    raw_spectra : dict
        (group, sample_id, replicate) -> (x_array, y_array)
    meta_df : pd.DataFrame
        File-level metadata
    """
    files = list(find_spectra(data_dir, pattern=pattern))
    if not files:
        raise DataNotFoundError(f"No {pattern} files in {data_dir}")

    raw_spectra = {}
    metadata = []

    for fp in files:
        try:
            spec_id = parse_filename(
                fp,
                fallback_group=folder_mapping.get(fp.parent.name, "UNK"),
            )
            x, y = read_spectrum(fp)
            key = (spec_id.group, spec_id.sample_id, spec_id.replicate)
            raw_spectra[key] = (x, y)
            metadata.append({
                "file": fp.name,
                "group": spec_id.group,
                "sample_id": spec_id.sample_id,
                "replicate": spec_id.replicate,
                "n_points": len(x),
                "x_min": float(x.min()),
                "x_max": float(x.max()),
            })
        except Exception as e:
            logging.warning(f"Skip {fp.name}: {e}")

    if not raw_spectra:
        raise DataNotFoundError("All files failed to load.")

    return raw_spectra, pd.DataFrame(metadata)


# ─── Step 2: Extract QC-passed keys ───
def get_qc_passed_keys(
    raw_spectra: dict,
    failures: pd.DataFrame,
) -> set:
    """
    Determine which spectrum keys passed QC.

    QC failures are at the SAMPLE level (group, sample_id).
    If a sample fails, ALL its replicates are excluded.

    Parameters
    ----------
    raw_spectra : dict
        All raw spectra
    failures : pd.DataFrame
        Failed samples from run_qc_pipeline()

    Returns
    -------
    set
        Keys (group, sample_id, replicate) that passed QC
    """
    if len(failures) == 0:
        return set(raw_spectra.keys())

    # Failed sample identifiers
    failed_samples = set(
        zip(failures["group"], failures["sample_id"])
    )

    passed = set()
    for key in raw_spectra:
        group, sid, rep = key
        if (group, sid) not in failed_samples:
            passed.add(key)

    return passed


# ─── CLI ───
def parse_args():
    parser = argparse.ArgumentParser(
        description="SERS QC → Preprocessing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_qc_preprocess.py
  python run_qc_preprocess.py --normalization minmax
  python run_qc_preprocess.py --normalization l2 --no-trim
  python run_qc_preprocess.py --config my_config.yaml
        """,
    )
    parser.add_argument(
        "--config", "-c",
        default="config/config.yaml",
        help="Path to config.yaml (default: config/config.yaml)",
    )
    parser.add_argument(
        "--normalization", "-n",
        choices=["snv", "minmax", "l2", "area", "none"],
        default=None,
        help="Override normalization method (default: from config)",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Skip trimming to fingerprint region",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Override output directory (default: from config/env)",
    )
    parser.add_argument(
        "--skip-qc",
        action="store_true",
        help="Skip QC and preprocess all spectra (not recommended)",
    )
    return parser.parse_args()


# ─── Main Pipeline ───
def main():
    args = parse_args()
    logger = setup_logging()

    start_time = datetime.now()

    logger.info("=" * 64)
    logger.info("  SERS Pipeline: QC → Preprocessing")
    logger.info(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 64)

    try:
        # ════════════════════════════════════════════════════
        # Step 1: Configuration
        # ════════════════════════════════════════════════════
        logger.info("\n[Step 1] Loading configuration...")
        config = load_config(args.config)

        # CLI overrides (PreprocessingConfig is frozen, so use object.__setattr__)
        if args.normalization:
            object.__setattr__(config.preprocessing, 'normalization', args.normalization)
            logger.info(f"  → Normalization overridden to: {args.normalization}")

        if args.no_trim:
            object.__setattr__(config.preprocessing, 'do_trim', False)
            logger.info("  → Trimming disabled via --no-trim")

        # Output directory
        output_dir = Path(args.output_dir) if args.output_dir else Path(RESULTS_DIR)
        fig_dir = Path(FIG_DIR)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fig_dir, exist_ok=True)

        norm_method = getattr(config.preprocessing, 'normalization',
                              'snv' if config.preprocessing.use_snv else 'none')
        logger.info(f"  Config: {args.config}")
        logger.info(f"  Normalization: {norm_method}")
        logger.info(f"  Output: {output_dir}")

        # ════════════════════════════════════════════════════
        # Step 2: Load Raw Data
        # ════════════════════════════════════════════════════
        logger.info("\n[Step 2] Loading raw spectra...")
        data_dir = Path(RAW_DATA_DIR)

        if not data_dir.exists():
            raise DataNotFoundError(f"Data directory not found: {data_dir}")

        raw_spectra, meta_df = load_raw_spectra(
            data_dir, config.folder_to_group
        )

        n_spectra = len(raw_spectra)
        n_samples = meta_df.groupby(["group", "sample_id"]).ngroups
        n_groups = meta_df["group"].nunique()

        logger.info(f"  Loaded: {n_spectra} spectra, {n_samples} samples, {n_groups} groups")

        # Group summary
        group_summary = meta_df.groupby("group").agg(
            n_samples=("sample_id", "nunique"),
            n_spectra=("file", "count"),
        )
        for grp, row in group_summary.iterrows():
            logger.info(f"    {grp:>5s}: {row.n_samples:4d} samples, {row.n_spectra:5d} spectra")

        # Save metadata
        meta_df.to_csv(output_dir / "metadata_raw.csv", index=False)

        # Common grid
        x_arrays = [x for x, y in raw_spectra.values()]
        common_grid = make_common_grid(x_arrays)
        logger.info(f"  Common grid: {len(common_grid)} points "
                     f"({common_grid[0]:.1f} – {common_grid[-1]:.1f} cm⁻¹)")

        # ════════════════════════════════════════════════════
        # Step 3: Quality Control
        # ════════════════════════════════════════════════════
        if args.skip_qc:
            logger.warning("\n[Step 3] QC SKIPPED (--skip-qc flag)")
            logger.warning("  ⚠ All spectra will be preprocessed without quality filtering")
            qc_passed_keys = None
            failures = pd.DataFrame()
            qc_stats = pd.DataFrame()
        else:
            logger.info("\n[Step 3] Running QC pipeline...")

            qc_config = config.qc
            logger.info(f"  Thresholds: RSD < {qc_config.rsd_threshold}%, "
                         f"Corr > {qc_config.corr_threshold}")

            # Run QC
            qc_result = run_qc_pipeline(
                raw_spectra,
                common_grid,
                rsd_threshold=qc_config.rsd_threshold,
                corr_threshold=qc_config.corr_threshold,
            )

            # Unpack results (handle different return signatures)
            if len(qc_result) == 4:
                gate_df, qc_stats, failures, group_qc_summary = qc_result
            elif len(qc_result) == 3:
                qc_stats, failures, group_qc_summary = qc_result
                gate_df = None
            else:
                raise PipelineError(f"Unexpected QC result format: {len(qc_result)} items")

            # QC-passed keys
            qc_passed_keys = get_qc_passed_keys(raw_spectra, failures)

            n_passed_samples = len(qc_stats) - len(failures)
            n_total_samples = len(qc_stats)
            pass_rate = 100 * n_passed_samples / n_total_samples if n_total_samples > 0 else 0

            logger.info(f"\n  QC Results:")
            logger.info(f"    Total samples:  {n_total_samples}")
            logger.info(f"    Passed:         {n_passed_samples} ({pass_rate:.1f}%)")
            logger.info(f"    Failed:         {len(failures)}")
            logger.info(f"    Spectra to preprocess: {len(qc_passed_keys)}")

            if len(failures) > 0:
                logger.info(f"\n  Failed samples by group:")
                fail_by_group = failures.groupby("group").size()
                for grp, count in fail_by_group.items():
                    logger.info(f"    {grp}: {count}")

            # Save QC results
            qc_stats.to_csv(output_dir / "qc_stats.csv", index=False)
            if len(failures) > 0:
                failures.to_csv(output_dir / "qc_failures.csv", index=False)
            if gate_df is not None:
                gate_df.to_csv(output_dir / "qc_intensity_gate.csv", index=False)

            logger.info(f"  QC results saved to {output_dir}/")

        # ════════════════════════════════════════════════════
        # Step 4: Preprocessing
        # ════════════════════════════════════════════════════
        logger.info("\n[Step 4] Preprocessing QC-passed spectra...")

        processed, prep_stats_df, proc_grid = preprocess_spectra(
            raw_spectra,
            common_grid,
            config,
            qc_passed_keys=qc_passed_keys,
        )

        logger.info(f"\n  Preprocessing complete:")
        logger.info(f"    Processed spectra: {len(processed)}")
        logger.info(f"    Grid points: {len(proc_grid)} "
                     f"({proc_grid[0]:.1f} – {proc_grid[-1]:.1f} cm⁻¹)")

        # Preprocessing stats summary
        if len(prep_stats_df) > 0:
            logger.info(f"    Raw intensity range:  "
                         f"{prep_stats_df['raw_min'].min():.1f} – "
                         f"{prep_stats_df['raw_max'].max():.1f}")
            logger.info(f"    Proc intensity range: "
                         f"{prep_stats_df['proc_min'].min():.3f} – "
                         f"{prep_stats_df['proc_max'].max():.3f}")

        # Save preprocessing stats
        prep_stats_df.to_csv(output_dir / "preprocessing_stats.csv", index=False)

        # ════════════════════════════════════════════════════
        # Step 5: Post-preprocessing Variance Check
        # ════════════════════════════════════════════════════
        logger.info("\n[Step 5] Post-preprocessing replicate variance...")

        variance_df = calculate_replicate_variance(
            processed, proc_grid,
            peak_region=(600, 1800),
        )

        if len(variance_df) > 0:
            logger.info(f"    Mean CV:   {variance_df['mean_cv'].mean():.2f}%")
            logger.info(f"    Mean Corr: {variance_df['mean_pairwise_correlation'].mean():.4f}")

            # Check for remaining problematic samples
            problematic = identify_problematic_samples(
                variance_df,
                cv_threshold=15.0,
                correlation_threshold=0.90,
            )
            if len(problematic) > 0:
                logger.warning(f"    ⚠ {len(problematic)} samples still show high variance after preprocessing")
            else:
                logger.info(f"    ✓ All preprocessed samples have acceptable variance")

            variance_df.to_csv(output_dir / "post_preprocessing_variance.csv", index=False)

        # ════════════════════════════════════════════════════
        # Step 6: Save Processed Data
        # ════════════════════════════════════════════════════
        logger.info("\n[Step 6] Saving processed spectra...")

        save_processed_spectra(
            processed,
            proc_grid,
            output_path=output_dir / "processed_spectra.csv",
        )

        # Also save the grid for downstream use
        np.savetxt(
            output_dir / "common_grid.csv",
            proc_grid,
            header="wavenumber_cm-1",
            comments="",
        )

        # ════════════════════════════════════════════════════
        # Summary
        # ════════════════════════════════════════════════════
        elapsed = datetime.now() - start_time

        logger.info("\n" + "=" * 64)
        logger.info("  Pipeline Complete!")
        logger.info("=" * 64)
        logger.info(f"  Elapsed:      {elapsed}")
        logger.info(f"  Input:        {n_spectra} raw spectra")
        if not args.skip_qc:
            logger.info(f"  QC passed:    {len(qc_passed_keys)} spectra")
        logger.info(f"  Preprocessed: {len(processed)} spectra")
        logger.info(f"  Grid:         {len(proc_grid)} points "
                     f"({proc_grid[0]:.1f} – {proc_grid[-1]:.1f} cm⁻¹)")
        logger.info(f"  Normalization: {norm_method}")
        logger.info(f"  Output dir:   {output_dir}")
        logger.info("")
        logger.info("  Output files:")
        logger.info(f"    ├─ metadata_raw.csv")
        if not args.skip_qc:
            logger.info(f"    ├─ qc_stats.csv")
            logger.info(f"    ├─ qc_failures.csv")
        logger.info(f"    ├─ preprocessing_stats.csv")
        logger.info(f"    ├─ post_preprocessing_variance.csv")
        logger.info(f"    ├─ processed_spectra.csv          ← ML 입력 데이터")
        logger.info(f"    └─ common_grid.csv")
        logger.info("=" * 64)

        return 0

    except DataNotFoundError as e:
        logger.error(f"\n✗ Data error: {e}")
        return 1
    except PipelineError as e:
        logger.error(f"\n✗ Pipeline error: {e}")
        return 2
    except KeyboardInterrupt:
        logger.warning("\n✗ Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"\n✗ Unexpected error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
