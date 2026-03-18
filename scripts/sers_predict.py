#!/usr/bin/env python3
"""
SERS Cancer Screening — Inference CLI

Predicts cancer probability and type from raw SERS spectrum files.

Usage:
    python scripts/sers_predict.py spectrum.csv
    python scripts/sers_predict.py spectrum.csv --age 55 --sex M --bmi 24.3
    python scripts/sers_predict.py *.CSV --output results.json --quiet

Prerequisites:
    python models/build_production_model.py   # Run once to build model artifacts
"""

from __future__ import annotations

import sys
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime

import numpy as np
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.io import read_spectrum
from src.sers.preprocessing import preprocess_single_spectrum

logger = logging.getLogger("sers_predict")


class ProductionPredictor:
    """Loads production model artifacts and runs inference."""

    def __init__(self, artifact_dir: Path | str = None):
        if artifact_dir is None:
            artifact_dir = PROJECT_ROOT / "models" / "production"
        self.artifact_dir = Path(artifact_dir)

        if not self.artifact_dir.exists():
            raise FileNotFoundError(
                f"Model artifacts not found at {self.artifact_dir}\n"
                f"Run: python models/build_production_model.py"
            )

        # Load manifest
        with open(self.artifact_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        # Load preprocessing config
        with open(self.artifact_dir / "preprocessing.json") as f:
            self.prep = json.load(f)

        # Load grid
        self.grid = np.load(self.artifact_dir / "common_grid.npy")

        # Load models
        mf = self.manifest["model_files"]
        self.s1_sers = joblib.load(self.artifact_dir / mf["stage1_sers"])
        self.s2_sers = joblib.load(self.artifact_dir / mf["stage2_sers"])
        self.s1_fusion = joblib.load(self.artifact_dir / mf["stage1_fusion"])
        self.s2_fusion = joblib.load(self.artifact_dir / mf["stage2_fusion"])

        self.cancer_types = self.manifest["cancer_types"]
        self.bmi_median = self.manifest["bmi_fill_median"]
        self.s2_classes = self.manifest.get("stage2_classes", list(range(len(self.cancer_types))))

    def preprocess(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Preprocess a raw spectrum to feature vector."""
        return preprocess_single_spectrum(
            x=x, y=y, grid=self.grid,
            do_trim=self.prep["do_trim"],
            trim_region=tuple(self.prep["trim_region"]),
            do_smooth=self.prep["do_smooth"],
            smooth_window=self.prep["smooth_window"],
            smooth_poly=self.prep["smooth_poly"],
            do_baseline=self.prep["do_baseline"],
            baseline_window=self.prep["baseline_window"],
            normalization=self.prep["normalization"],
        )

    def _sanity_check(self, x, y, filepath):
        """Basic sanity checks on raw spectrum."""
        warnings_list = []

        if len(x) < 100:
            warnings_list.append(f"Very few data points ({len(x)})")

        if x.max() < 2200 or x.min() > 400:
            warnings_list.append(
                f"Wavenumber range {x.min():.0f}-{x.max():.0f} may not cover "
                f"fingerprint region (400-2200 cm-1)"
            )

        if np.all(y == 0) or np.all(np.isnan(y)):
            warnings_list.append("All-zero or all-NaN intensities")

        return warnings_list

    def predict_single(self, filepath: str | Path,
                       age: float = None, sex: str = None,
                       bmi: float = None) -> dict:
        """Predict cancer from a single spectrum file."""
        filepath = Path(filepath)
        result = {
            "file": filepath.name,
            "status": "ok",
            "warnings": [],
        }

        try:
            # Read raw spectrum
            x, y = read_spectrum(filepath)
            result["n_points"] = len(x)
            result["wavenumber_range"] = f"{x.min():.1f}-{x.max():.1f}"

            # Sanity check
            result["warnings"] = self._sanity_check(x, y, filepath)

            # Preprocess
            features = self.preprocess(x, y).reshape(1, -1)

            # Decide model variant
            use_fusion = (age is not None and sex is not None)
            if use_fusion:
                sex_numeric = 1.0 if str(sex).upper() in ("M", "MALE") else 0.0
                bmi_val = bmi if bmi is not None else self.bmi_median
                if bmi is None:
                    result["warnings"].append(f"BMI missing, using training median ({self.bmi_median:.1f})")
                clinical = np.array([[age, sex_numeric, bmi_val]])
                X = np.hstack([features, clinical])
                s1_model = self.s1_fusion
                s2_model = self.s2_fusion
                result["model_variant"] = "fusion"
                result["clinical_features"] = {"age": age, "sex": sex, "bmi": bmi_val}
            else:
                X = features
                s1_model = self.s1_sers
                s2_model = self.s2_sers
                result["model_variant"] = "sers_only"
                if age is None and sex is None:
                    pass  # Expected
                else:
                    result["warnings"].append(
                        "Both age and sex required for fusion mode. Falling back to SERS-only."
                    )

            # Stage 1: cancer probability
            cancer_prob = float(s1_model.predict_proba(X)[0, 1])
            result["cancer_probability"] = round(cancer_prob, 4)
            result["cancer_detected"] = cancer_prob > 0.5

            # Stage 2: cancer type (always compute, even if non-cancer predicted)
            type_probs = s2_model.predict_proba(X)[0]
            s2_classes = s2_model.classes_
            type_prob_dict = {}
            for i, cls_idx in enumerate(s2_classes):
                name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else f"class_{cls_idx}"
                type_prob_dict[name] = round(float(type_probs[i]), 4)

            best_type_idx = type_probs.argmax()
            best_cls = s2_classes[best_type_idx]
            best_name = self.cancer_types[best_cls] if best_cls < len(self.cancer_types) else f"class_{best_cls}"

            result["cancer_type_prediction"] = best_name
            result["cancer_type_confidence"] = round(float(type_probs[best_type_idx]), 4)
            result["cancer_type_probabilities"] = type_prob_dict

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)

        return result

    def predict_batch(self, filepaths: list[str | Path],
                      age: float = None, sex: str = None,
                      bmi: float = None) -> list[dict]:
        """Predict cancer for multiple spectrum files."""
        return [self.predict_single(fp, age, sex, bmi) for fp in filepaths]


def format_result_text(result: dict) -> str:
    """Format a single result as human-readable text."""
    lines = []
    lines.append(f"  File: {result['file']}")

    if result["status"] == "error":
        lines.append(f"  ERROR: {result.get('message', 'Unknown error')}")
        return "\n".join(lines)

    variant = result.get("model_variant", "sers_only")
    lines.append(f"  Model: {variant}")

    # Stage 1
    prob = result["cancer_probability"]
    detected = result["cancer_detected"]
    bar = "#" * int(prob * 30) + "-" * (30 - int(prob * 30))
    status = "CANCER DETECTED" if detected else "Non-cancer"
    lines.append(f"  Cancer probability: {prob:.1%}  [{bar}]  {status}")

    # Stage 2
    if detected:
        best = result["cancer_type_prediction"]
        conf = result["cancer_type_confidence"]
        lines.append(f"  Predicted type: {best} (confidence: {conf:.1%})")

        probs = result["cancer_type_probabilities"]
        sorted_probs = sorted(probs.items(), key=lambda x: -x[1])
        prob_str = "  Type probabilities: " + " | ".join(
            f"{name}: {p:.1%}" for name, p in sorted_probs
        )
        lines.append(prob_str)

    if result.get("warnings"):
        for w in result["warnings"]:
            lines.append(f"  WARNING: {w}")

    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(
        description="SERS Cancer Screening — Predict from spectrum files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python sers_predict.py sample.CSV
  python sers_predict.py sample.CSV --age 55 --sex M --bmi 24.3
  python sers_predict.py *.CSV --output results.json --quiet
        """,
    )
    p.add_argument("spectra", nargs="+", help="Spectrum CSV file(s)")
    p.add_argument("--age", type=float, default=None, help="Patient age")
    p.add_argument("--sex", type=str, default=None, help="Patient sex (M/F)")
    p.add_argument("--bmi", type=float, default=None, help="Patient BMI")
    p.add_argument("--model-dir", default=None, help="Model artifacts directory")
    p.add_argument("--output", "-o", default=None, help="Output JSON file path")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress text output, JSON only")
    args = p.parse_args()

    if not args.quiet:
        logging.basicConfig(level=logging.WARNING, format="%(message)s")

    # Load predictor
    try:
        predictor = ProductionPredictor(args.model_dir)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    if not args.quiet:
        print("=" * 60)
        print("  SERS Cancer Screening")
        print("=" * 60)
        variant = "SERS + age/sex/BMI (fusion)" if (args.age and args.sex) else "SERS only"
        print(f"  Model: {variant}")
        print(f"  Files: {len(args.spectra)}")
        if args.age:
            print(f"  Patient: age={args.age}, sex={args.sex}, bmi={args.bmi or 'auto'}")
        print("-" * 60)

    # Run predictions
    results = predictor.predict_batch(args.spectra, args.age, args.sex, args.bmi)

    # Output
    output = {
        "version": "1.0",
        "model_variant": results[0].get("model_variant", "sers_only") if results else "unknown",
        "timestamp": datetime.now().isoformat(),
        "n_files": len(results),
        "results": results,
    }

    # Summary stats
    ok_results = [r for r in results if r["status"] == "ok"]
    if ok_results:
        n_detected = sum(1 for r in ok_results if r["cancer_detected"])
        avg_prob = np.mean([r["cancer_probability"] for r in ok_results])
        output["summary"] = {
            "total": len(ok_results),
            "cancer_detected": n_detected,
            "non_cancer": len(ok_results) - n_detected,
            "mean_cancer_probability": round(float(avg_prob), 4),
        }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        if not args.quiet:
            print(f"\n  Results saved to: {args.output}")

    if not args.quiet:
        for result in results:
            print()
            print(format_result_text(result))

        if ok_results:
            print()
            print("-" * 60)
            print(f"  Summary: {output['summary']['cancer_detected']}/{output['summary']['total']} "
                  f"cancer detected (mean prob: {output['summary']['mean_cancer_probability']:.1%})")
        print("=" * 60)
    else:
        # Quiet mode: output JSON to stdout
        if not args.output:
            print(json.dumps(output, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
