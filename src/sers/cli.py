"""Command-line interface for SERS analysis pipeline."""

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import load_config, DATA_DIR, RESULTS_DIR
from .analysis import run_pipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="sers",
        description="SERS Spectral Analysis Pipeline",
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"%(prog)s {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run analysis pipeline")
    run_parser.add_argument(
        "-i", "--input",
        type=Path,
        default=DATA_DIR,
        help="Input data directory",
    )
    run_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Output directory",
    )
    run_parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Config YAML file",
    )
    run_parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate data files")
    validate_parser.add_argument(
        "-i", "--input",
        type=Path,
        default=DATA_DIR,
        help="Input data directory",
    )
    
    args = parser.parse_args()
    
    if args.command == "run":
        cmd_run(args)
    elif args.command == "validate":
        cmd_validate(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_run(args):
    """Execute the analysis pipeline."""
    config = load_config(args.config) if args.config else load_config()
    
    result = run_pipeline(
        data_dir=args.input,
        config=config,
        verbose=not args.quiet,
    )
    
    # Save outputs
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Save QC report
    qc_path = args.output / "qc_report.csv"
    result.qc_report.to_csv(qc_path, index=False)
    print(f"Saved QC report: {qc_path}")
    
    # Save processed spectra as NPZ
    X, y, sample_ids = result.to_matrix()
    npz_path = args.output / "processed_spectra.npz"
    import numpy as np
    np.savez(
        npz_path,
        X=X,
        y=y,
        sample_ids=sample_ids,
        grid=result.grid,
    )
    print(f"Saved spectra: {npz_path}")


def cmd_validate(args):
    """Validate input data files."""
    from .io import find_spectra, parse_filename, read_spectrum
    
    files = find_spectra(args.input)
    print(f"Found {len(files)} spectrum files")
    
    errors = []
    for path in files:
        try:
            parse_filename(path)
            read_spectrum(path)
        except Exception as e:
            errors.append((path.name, str(e)))
    
    if errors:
        print(f"\n{len(errors)} files with issues:")
        for name, err in errors[:10]:
            print(f"  {name}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
        sys.exit(1)
    else:
        print("All files valid!")


if __name__ == "__main__":
    main()
