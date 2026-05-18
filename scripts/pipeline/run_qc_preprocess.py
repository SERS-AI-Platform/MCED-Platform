"""
SERS Pipeline: Preprocessing → QC

모든 스펙트럼을 전처리한 후, 전처리된 스펙트럼에 대해 QC를 수행.
QC를 통과한 전처리 스펙트럼만 최종 출력.

Usage:
    python run_qc_preprocess.py
    python run_qc_preprocess.py --config config.yaml --normalization snv
    python run_qc_preprocess.py --normalization minmax --no-trim

Pipeline:
    ┌──────────────────────────────────────────┐
    │  1. Load raw spectra                     │
    │  2. Create common grid                   │
    │  2.5. Wavenumber calibration (opt.)      │
    │       └─ Urea 1001.4 cm⁻¹ alignment     │
    │  3. Preprocess ALL spectra               │
    │     ├─ ① Trim (400-2200 cm⁻¹)           │
    │     ├─ ② Smooth (Savitzky-Golay)         │
    │     ├─ ③ Baseline correction             │
    │     └─ ④ Normalize (SNV/etc)             │
    │  4. QC on preprocessed spectra           │
    │     ├─ Intensity gate (on raw)           │
    │     └─ Replicate QC (RSD/Corr on proc)   │
    │  5. Filter to QC-passed only & save      │
    └──────────────────────────────────────────┘
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
from src.sers.io import find_spectra, make_common_grid, make_fixed_grid

# QC pipeline
from src.sers.qc.qc import (
    calculate_intensity_gate,
    calculate_replicate_qc,
    identify_qc_failures,
    summarize_qc_by_group,
    # v2 two-stage
    apply_stage1_qc,
    apply_per_spectrum_corr_qc,
    enforce_min_replicates,
)

# Preprocessing (updated module with trim + multi-normalization)
from src.sers.preprocessing import (
    preprocess_spectra,
    calculate_replicate_variance,
    identify_problematic_samples,
    save_processed_spectra,
    calibrate_spectra_batch,
)

COHORT_GROUPS = {
    'cohort1628': ['CRC', 'LUN', 'CPAN', "YPAN", "OVA","BRE", "BLC", "PRO",
                    "NOR","DIA","HBP","H.D",'YNOR'],
    'cohort1598': ['CRC', 'LUN', 'CPAN', "YPAN", "OVA", "BLC", "PRO",
                    "NOR","DIA","HBP","H.D",'YNOR'],
    'cohort1539': ['CRC', 'LUN', 'CPAN', "OVA", "BLC", "PRO",
                    "NOR","DIA","HBP","H.D"]
}

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


# ─── Helper: Extract QC-passed keys ───
def get_qc_passed_keys(
    all_keys,
    failures: pd.DataFrame,
) -> set:
    """
    Determine which spectrum keys passed QC.

    QC failures are at the SAMPLE level (group, sample_id).
    If a sample fails, ALL its replicates are excluded.
    """
    if len(failures) == 0:
        return set(all_keys)

    failed_samples = set(
        zip(failures["group"], failures["sample_id"])
    )

    passed = set()
    for key in all_keys:
        group, sid, rep = key
        if (group, sid) not in failed_samples:
            passed.add(key)

    return passed


# ─── CLI ───
def parse_args():
    parser = argparse.ArgumentParser(
        description="SERS Preprocessing → QC Pipeline",
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
        "--cohort",
        choices=['cohort1628','cohort1598','cohort1539'],
        default="cohort1628",
        help="Cohort name for config overrides (default: cohort1628)",
    )

    parser.add_argument(
        "--groups", "-g",
        default=None,
        help="Comma-separated group codes to include, e.g CRC,LUN,CPAN,NOR"
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
        help="Skip QC and use all preprocessed spectra (equivalent to --qc-policy none)",
    )
    parser.add_argument(
        "--qc-policy",
        choices=["v2", "strict", "none"],
        default="v2",
        help="QC policy: v2 (two-stage per-spectrum, default), "
             "strict (v1 whole-subject RSD<5/Corr>0.95), "
             "none (no filtering).",
    )
    return parser.parse_args()


# ─── Main Pipeline ───
def main():
    args = parse_args()
    logger = setup_logging()

    start_time = datetime.now()

    logger.info("=" * 64)
    logger.info("  SERS Pipeline: Preprocessing → QC")
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
        
        selected_groups = None
        if args.groups:
            selected_groups = [g.strip().upper() for g in args.groups.split(",") if g.strip()]
        elif args.cohort != "cohort1628":
            selected_groups = COHORT_GROUPS.get(args.cohort)
        
        if selected_groups:
            before = len(raw_spectra)

            keep_meta = meta_df['group'].str.upper().isin(selected_groups)
            meta_df = meta_df.loc[keep_meta].copy()
            
            keep_keys = set(
                zip(meta_df['group'], meta_df['sample_id'], meta_df['replicate'])
            )
            raw_spectra = {
                key: value for key, value in raw_spectra.items()
                if key in keep_keys
            }

            logger.info(
                f"Cohort: {args.cohort}, groups = {selected_groups} "
                f" Kept: {len(raw_spectra)}/{before}"   
            )
            
            if not raw_spectra:
                raise DataNotFoundError(f"No spectra left after filtering to groups: {selected_groups}")

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

        # Common grid (use fixed grid from config if available)
        fixed = make_fixed_grid(config)
        if fixed is not None:
            common_grid = fixed
            logger.info(f"  Fixed grid: {len(common_grid)} points "
                         f"({common_grid[0]:.1f} – {common_grid[-1]:.1f} cm⁻¹)")
        else:
            x_arrays = [x for x, y in raw_spectra.values()]
            common_grid = make_common_grid(x_arrays)
            logger.info(f"  Dynamic grid: {len(common_grid)} points "
                         f"({common_grid[0]:.1f} – {common_grid[-1]:.1f} cm⁻¹)")

        # ════════════════════════════════════════════════════
        # Step 2.5: Wavenumber Calibration
        # ════════════════════════════════════════════════════
        prep = config.preprocessing
        do_cal = getattr(prep, 'do_calibration', False)
        if do_cal:
            cal_ref = getattr(prep, 'calibration_reference_wn', 1001.4)
            cal_win = getattr(prep, 'calibration_window', 20.0)
            logger.info(f"\n[Step 2.5] Wavenumber calibration...")
            logger.info(f"  Reference peak: {cal_ref} cm⁻¹ (urea C-N stretch)")
            logger.info(f"  Search window: ±{cal_win} cm⁻¹")

            raw_spectra, shift_df = calibrate_spectra_batch(
                raw_spectra, target_wn=cal_ref, window=cal_win,
            )

            # Log per-group calibration statistics
            for grp, grp_df in shift_df.groupby("group"):
                shifts = grp_df["shift_cm1"].values
                logger.info(f"    {grp:>5s}: mean shift = {shifts.mean():+.4f} cm⁻¹, "
                             f"std = {shifts.std():.4f}, "
                             f"range = [{shifts.min():+.4f}, {shifts.max():+.4f}]")

            # Save calibration shifts
            shift_df.to_csv(output_dir / "calibration_shifts.csv", index=False)
            logger.info(f"  Calibration shifts saved to {output_dir}/calibration_shifts.csv")
        else:
            logger.info("\n[Step 2.5] Wavenumber calibration: SKIPPED (do_calibration=false)")

        # ════════════════════════════════════════════════════
        # Resolve QC policy (skip_qc flag → policy=none shorthand)
        # ════════════════════════════════════════════════════
        qc_policy = "none" if args.skip_qc else args.qc_policy
        logger.info(f"\n  QC policy: {qc_policy}")

        # ════════════════════════════════════════════════════
        # Step 2.9: Stage 1 QC on RAW (v2 policy only)
        # ════════════════════════════════════════════════════
        stage1_drops = pd.DataFrame()
        if qc_policy == "v2":
            logger.info("\n[Step 2.9] Stage 1 QC on raw spectra (v2)...")
            qc_config = config.qc
            s1_pass, stage1_drops = apply_stage1_qc(
                raw_spectra,
                fingerprint_region=tuple(qc_config.fingerprint_region),
                intensity_gate_ratio=qc_config.intensity_gate_ratio,
                cosmic_isolation=getattr(qc_config, "cosmic_isolation", 2.0),
                cosmic_height=getattr(qc_config, "cosmic_height", 0.3),
                sat_plateau=getattr(qc_config, "saturation_plateau", 5),
            )
            n_raw_before = len(raw_spectra)
            raw_spectra = {k: raw_spectra[k] for k in s1_pass}
            logger.info(
                f"  Stage 1: {len(raw_spectra)}/{n_raw_before} raw spectra kept "
                f"({len(stage1_drops)} dropped)"
            )
            if len(stage1_drops) > 0:
                logger.info("  Drops by reason:")
                for reason, cnt in stage1_drops["reason"].value_counts().items():
                    logger.info(f"    {reason}: {cnt}")
                stage1_drops.to_csv(output_dir / "qc_stage1_drops.csv", index=False)

        # ════════════════════════════════════════════════════
        # Step 3: Preprocess surviving spectra
        # ════════════════════════════════════════════════════
        logger.info("\n[Step 3] Preprocessing spectra...")

        processed, prep_stats_df, proc_grid = preprocess_spectra(
            raw_spectra,
            common_grid,
            config,
            qc_passed_keys=None,  # Stage 1 already filtered raw_spectra if v2
        )

        logger.info(f"\n  Preprocessing complete:")
        logger.info(f"    Processed spectra: {len(processed)}")
        logger.info(f"    Grid points: {len(proc_grid)} "
                     f"({proc_grid[0]:.1f} – {proc_grid[-1]:.1f} cm⁻¹)")

        if len(prep_stats_df) > 0:
            logger.info(f"    Raw intensity range:  "
                         f"{prep_stats_df['raw_min'].min():.1f} – "
                         f"{prep_stats_df['raw_max'].max():.1f}")
            logger.info(f"    Proc intensity range: "
                         f"{prep_stats_df['proc_min'].min():.3f} – "
                         f"{prep_stats_df['proc_max'].max():.3f}")

        prep_stats_df.to_csv(output_dir / "preprocessing_stats.csv", index=False)

        # ════════════════════════════════════════════════════
        # Step 4: QC on preprocessed spectra
        # ════════════════════════════════════════════════════
        if qc_policy == "none":
            logger.warning("\n[Step 4] QC SKIPPED (policy=none)")
            logger.warning("  ⚠ All spectra will be saved without quality filtering")
            qc_passed_keys = set(processed.keys())
            failures = pd.DataFrame()
            qc_stats = pd.DataFrame()
        elif qc_policy == "v2":
            logger.info("\n[Step 4] Stage 2 QC on preprocessed spectra (v2)...")
            qc_config = config.qc
            corr_thr = getattr(qc_config, "per_spectrum_corr_threshold", 0.925)
            min_reps = getattr(qc_config, "min_reps_after_qc", 4)
            logger.info(f"  Stage 2a: per-replicate corr ≥ {corr_thr}")
            logger.info(f"  Stage 2b: min_reps_after_qc = {min_reps}")

            s2a_pass, s2a_drops = apply_per_spectrum_corr_qc(
                processed, corr_threshold=corr_thr
            )
            qc_passed_keys, s2b_subj_drops = enforce_min_replicates(
                s2a_pass, min_n=min_reps
            )

            n_s2a_dropped = len(s2a_drops)
            n_subj_dropped = len(s2b_subj_drops)
            logger.info(
                f"  Stage 2a: {len(s2a_pass)}/{len(processed)} replicates kept "
                f"({n_s2a_dropped} dropped)"
            )
            logger.info(
                f"  Stage 2b: {len(qc_passed_keys)}/{len(s2a_pass)} replicates kept "
                f"({n_subj_dropped} subjects dropped)"
            )

            if len(s2a_drops) > 0:
                s2a_drops.to_csv(output_dir / "qc_stage2a_drops.csv", index=False)
            if len(s2b_subj_drops) > 0:
                s2b_subj_drops.to_csv(output_dir / "qc_stage2b_dropped_subjects.csv", index=False)

            # No v1-style stats in v2 path
            qc_stats = pd.DataFrame()
            failures = pd.DataFrame()
        else:
            logger.info("\n[Step 4] Running strict v1 QC on preprocessed spectra...")

            qc_config = config.qc
            logger.info(f"  Thresholds: RSD < {qc_config.rsd_threshold}%, "
                         f"Corr > {qc_config.corr_threshold}")

            # Level 0: Intensity gate on RAW spectra (detects enhancement failure)
            gate_df = calculate_intensity_gate(
                raw_spectra,
                fingerprint_region=qc_config.fingerprint_region,
                intensity_gate_ratio=qc_config.intensity_gate_ratio,
            )
            n_gate_fail = int((~gate_df["gate_pass"]).sum())

            # Level 1: Replicate QC on PREPROCESSED spectra
            # Wrap processed {key: y_array} as {key: (grid, y_array)} for QC functions
            processed_as_raw = {
                key: (proc_grid, y_proc)
                for key, y_proc in processed.items()
            }
            qc_stats = calculate_replicate_qc(
                processed_as_raw,
                proc_grid,
                fingerprint_region=qc_config.fingerprint_region,
            )

            # Identify failures
            failures = identify_qc_failures(
                qc_stats,
                rsd_threshold=qc_config.rsd_threshold,
                corr_threshold=qc_config.corr_threshold,
            )
            group_qc_summary = summarize_qc_by_group(qc_stats)

            # QC-passed keys
            qc_passed_keys = get_qc_passed_keys(processed.keys(), failures)

            n_passed_samples = len(qc_stats) - len(failures)
            n_total_samples = len(qc_stats)
            pass_rate = 100 * n_passed_samples / n_total_samples if n_total_samples > 0 else 0

            logger.info(f"\n  QC Results:")
            logger.info(f"    Intensity gate failures: {n_gate_fail}/{len(gate_df)} spectra")
            logger.info(f"    Total samples:  {n_total_samples}")
            logger.info(f"    Passed:         {n_passed_samples} ({pass_rate:.1f}%)")
            logger.info(f"    Failed:         {len(failures)}")
            logger.info(f"    Spectra to save: {len(qc_passed_keys)}")

            if len(failures) > 0:
                logger.info(f"\n  Failed samples by group:")
                fail_by_group = failures.groupby("group").size()
                for grp, count in fail_by_group.items():
                    logger.info(f"    {grp}: {count}")

            # Save QC results
            qc_stats.to_csv(output_dir / "qc_stats.csv", index=False)
            if len(failures) > 0:
                failures.to_csv(output_dir / "qc_failures.csv", index=False)
            gate_df.to_csv(output_dir / "qc_intensity_gate.csv", index=False)

            logger.info(f"  QC results saved to {output_dir}/")

        # ════════════════════════════════════════════════════
        # Step 5: Filter & Save
        # ════════════════════════════════════════════════════
        logger.info("\n[Step 5] Filtering to QC-passed spectra & saving...")

        # Filter processed spectra to QC-passed only
        processed_passed = {
            key: y_proc for key, y_proc in processed.items()
            if key in qc_passed_keys
        }

        logger.info(f"  Filtered: {len(processed_passed)}/{len(processed)} spectra passed QC")

        # Post-QC replicate variance
        variance_df = calculate_replicate_variance(
            processed_passed, proc_grid,
            peak_region=(600, 1800),
        )

        if len(variance_df) > 0:
            logger.info(f"    Mean CV:   {variance_df['mean_cv'].mean():.2f}%")
            logger.info(f"    Mean Corr: {variance_df['mean_pairwise_correlation'].mean():.4f}")

            problematic = identify_problematic_samples(
                variance_df,
                cv_threshold=15.0,
                correlation_threshold=0.90,
            )
            if len(problematic) > 0:
                logger.warning(f"    ⚠ {len(problematic)} samples still show high variance")
            else:
                logger.info(f"    ✓ All samples have acceptable variance")

            variance_df.to_csv(output_dir / "post_preprocessing_variance.csv", index=False)

        # Save
        save_processed_spectra(
            processed_passed,
            proc_grid,
            output_path=output_dir / "processed_spectra.csv",
        )

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
        logger.info(f"  Preprocessed: {len(processed)} spectra (all)")
        if not args.skip_qc:
            logger.info(f"  QC passed:    {len(processed_passed)} spectra")
        logger.info(f"  Grid:         {len(proc_grid)} points "
                     f"({proc_grid[0]:.1f} – {proc_grid[-1]:.1f} cm⁻¹)")
        logger.info(f"  Calibration:  {'ON' if do_cal else 'OFF'}")
        logger.info(f"  Normalization: {norm_method}")
        logger.info(f"  Pipeline order: Calibration → Preprocess → QC")
        logger.info(f"  Output dir:   {output_dir}")
        logger.info("")
        logger.info("  Output files:")
        logger.info(f"    ├─ metadata_raw.csv")
        if do_cal:
            logger.info(f"    ├─ calibration_shifts.csv")
        logger.info(f"    ├─ preprocessing_stats.csv")
        if not args.skip_qc:
            logger.info(f"    ├─ qc_stats.csv")
            logger.info(f"    ├─ qc_failures.csv")
            logger.info(f"    ├─ qc_intensity_gate.csv")
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
