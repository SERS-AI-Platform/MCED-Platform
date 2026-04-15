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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.sers.io import read_spectrum
from src.sers.preprocessing import preprocess_single_spectrum

logger = logging.getLogger("sers_predict")

# QC thresholds (matching training pipeline)
INTENSITY_GATE_RATIO = 0.1      # flag if fp_mean < median * 0.1
MIN_REPLICATE_CORRELATION = 0.90  # flag if corr with median spectrum < 0.90


class ProductionPredictor:
    """Loads production model artifacts and runs inference."""

    def __init__(self, artifact_dir: Path | str = None):
        if artifact_dir is None:
            artifact_dir = PROJECT_ROOT / "artifacts" / "baselines" / "lr-fusion" / "v1.0.0"
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

        # Operating modes with pre-computed thresholds
        self.operating_modes = self.manifest.get("operating_modes", {
            "screening": {"threshold": 0.35, "description": "High sensitivity for screening (Sens ~95%)"},
            "balanced": {"threshold": 0.50, "description": "Balanced sensitivity/specificity"},
            "confirmatory": {"threshold": 0.70, "description": "High specificity for confirmation (Spec ~95%)"},
        })
        self.default_mode = self.manifest.get("default_mode", "screening")

        # ── Cross-instrument calibration (optional) ──
        # The model was trained on Thermo data. To run inference on spectra from
        # the Medical Raman instrument, a Piecewise Direct Standardization (PDS)
        # transform is applied after standard preprocessing to map the feature
        # vector into the Thermo feature space the model expects.
        self.pds = None
        pds_path = self.artifact_dir / "calibration" / "pds.npz"
        if pds_path.exists():
            try:
                npz = np.load(pds_path)
                # Sanity: calibration grid must match model grid
                if npz["grid"].shape == self.grid.shape and np.allclose(npz["grid"], self.grid):
                    self.pds = {
                        "medical_to_thermo": {
                            "F": npz["F_medical_to_thermo"],
                            "mu_s": npz["mu_s_medical_to_thermo"],
                            "mu_t": npz["mu_t_medical_to_thermo"],
                        },
                        "thermo_to_medical": {
                            "F": npz["F_thermo_to_medical"],
                            "mu_s": npz["mu_s_thermo_to_medical"],
                            "mu_t": npz["mu_t_thermo_to_medical"],
                        },
                    }
                    logger.info(f"Loaded PDS calibration from {pds_path}")
                else:
                    logger.warning(
                        f"PDS grid mismatch (pds={npz['grid'].shape} vs "
                        f"model={self.grid.shape}); ignoring calibration file"
                    )
            except Exception as e:
                logger.warning(f"Failed to load PDS calibration: {e}")

        # Medical urea-peak wavenumber shift applied BEFORE preprocessing when
        # running Medical-instrument inference.
        self.medical_wavenumber_shift = -28.0

    def _apply_pds(self, features: np.ndarray, instrument: str) -> np.ndarray:
        """Map a preprocessed feature vector into the model's training domain.

        Parameters
        ----------
        features : (n_features,) or (1, n_features) array
            SNV-normalized feature vector in the *source* instrument domain.
        instrument : str
            Source instrument: 'thermo' (no-op), 'medical' (apply M→T PDS).

        Returns
        -------
        np.ndarray
            Feature vector in the Thermo (training) domain. Shape preserved.
        """
        inst = (instrument or "thermo").lower()
        if inst in ("thermo", "thermo_scientific", "production"):
            return features
        if inst not in ("medical", "medical_raman"):
            logger.warning(f"Unknown instrument '{instrument}' — treating as Thermo")
            return features
        if self.pds is None:
            logger.warning(
                "Medical-instrument input received but no PDS calibration file "
                f"found at {self.artifact_dir / 'calibration' / 'pds.npz'}. "
                "Predictions on Medical data WILL be unreliable."
            )
            return features
        p = self.pds["medical_to_thermo"]
        vec = np.atleast_2d(features)
        mapped = (vec - p["mu_s"]) @ p["F"] + p["mu_t"]
        return mapped.reshape(features.shape)

    def preprocess(self, x: np.ndarray, y: np.ndarray,
                   instrument: str = "thermo") -> np.ndarray:
        """Preprocess a raw spectrum to feature vector.

        When instrument='medical', applies the Medical-specific preprocessing
        (wavenumber shift, no smoothing) and then the PDS calibration transfer
        to the Thermo feature space.
        """
        inst = (instrument or "thermo").lower()
        if inst in ("medical", "medical_raman"):
            # Urea-peak alignment (Medical → Thermo reference frame)
            x = x + self.medical_wavenumber_shift
            features = preprocess_single_spectrum(
                x=x, y=y, grid=self.grid,
                do_trim=self.prep["do_trim"],
                trim_region=tuple(self.prep["trim_region"]),
                do_smooth=False,  # Medical BG-removed data is already smoothed
                smooth_window=self.prep["smooth_window"],
                smooth_poly=self.prep["smooth_poly"],
                do_baseline=self.prep["do_baseline"],
                baseline_window=self.prep["baseline_window"],
                normalization=self.prep["normalization"],
            )
            return self._apply_pds(features, "medical")

        # Default: Thermo (original behaviour)
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
                       bmi: float = None, mode: str = None,
                       instrument: str = "thermo") -> dict:
        """Predict cancer from a single spectrum file.

        Parameters
        ----------
        instrument : str
            Source instrument. 'thermo' (default, no calibration) or 'medical'
            (applies Medical→Thermo PDS calibration transfer before inference).
        """
        filepath = Path(filepath)
        result = {
            "file": filepath.name,
            "status": "ok",
            "warnings": [],
            "instrument": instrument,
        }

        try:
            # Read raw spectrum
            x, y = read_spectrum(filepath)
            result["n_points"] = len(x)
            result["wavenumber_range"] = f"{x.min():.1f}-{x.max():.1f}"

            # Sanity check
            result["warnings"] = self._sanity_check(x, y, filepath)

            if (instrument or "thermo").lower() in ("medical", "medical_raman"):
                if self.pds is None:
                    result["warnings"].append(
                        "Medical instrument specified but no PDS calibration "
                        "file loaded — predictions will be unreliable."
                    )
                else:
                    result["calibration"] = "pds_medical_to_thermo"

            # Preprocess (+ PDS if instrument=medical)
            features = self.preprocess(x, y, instrument=instrument).reshape(1, -1)

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

            # Apply threshold based on operating mode
            active_mode = mode or self.default_mode
            if active_mode not in self.operating_modes:
                active_mode = self.default_mode
            threshold = self.operating_modes[active_mode]["threshold"]
            result["cancer_detected"] = cancer_prob > threshold
            result["operating_mode"] = active_mode
            result["threshold"] = threshold

            # Stage 2: cancer type (always compute, even if non-cancer predicted)
            type_probs = s2_model.predict_proba(X)[0].copy()
            s2_classes = s2_model.classes_

            # Sex-based biological constraint: mask impossible cancer types
            # Males cannot have ovarian cancer; Females cannot have prostate cancer
            if sex is not None:
                sex_upper = str(sex).upper()
                constrained_types = []
                for i, cls_idx in enumerate(s2_classes):
                    name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else ""
                    if sex_upper in ("M", "MALE") and name == "OVA":
                        type_probs[i] = 0.0
                        constrained_types.append("OVA")
                    elif sex_upper in ("F", "FEMALE") and name == "PRO":
                        type_probs[i] = 0.0
                        constrained_types.append("PRO")
                # Renormalize after masking
                prob_sum = type_probs.sum()
                if prob_sum > 0:
                    type_probs = type_probs / prob_sum
                if constrained_types:
                    result["sex_constrained_types"] = constrained_types

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

    def _qc_replicate_set(self, raw_spectra: list[dict]) -> list[dict]:
        """Run QC on a set of replicate spectra (same patient).

        QC steps (matching training pipeline):
          1. Intensity gate: flag spectra with fp_mean < median * 0.1
          2. Replicate correlation: flag spectra with corr < 0.90 vs median

        Parameters
        ----------
        raw_spectra : list of dict
            Each dict has 'x', 'y', 'filepath', 'features' (preprocessed)

        Returns
        -------
        list of dict with added 'qc_pass', 'qc_flags' fields
        """
        if not raw_spectra:
            return raw_spectra

        fp_region = tuple(self.prep["trim_region"])

        # Step 1: Intensity gate
        fp_means = []
        for s in raw_spectra:
            mask = (s['x'] >= fp_region[0]) & (s['x'] <= fp_region[1])
            fp_mean = float(np.mean(np.abs(s['y'][mask]))) if mask.any() else float(np.mean(np.abs(s['y'])))
            s['fp_mean'] = fp_mean
            fp_means.append(fp_mean)

        median_fp = float(np.median(fp_means))
        gate_threshold = median_fp * INTENSITY_GATE_RATIO

        for s in raw_spectra:
            s['qc_flags'] = []
            s['qc_pass'] = True
            if s['fp_mean'] < gate_threshold:
                s['qc_flags'].append(f"intensity_gate_fail (fp_mean={s['fp_mean']:.1f} < {gate_threshold:.1f})")
                s['qc_pass'] = False

        # Step 2: Replicate correlation (only if multiple spectra)
        if len(raw_spectra) >= 2:
            feature_matrix = np.vstack([s['features'] for s in raw_spectra])
            median_spectrum = np.median(feature_matrix, axis=0)

            for i, s in enumerate(raw_spectra):
                corr = float(np.corrcoef(s['features'], median_spectrum)[0, 1])
                s['replicate_correlation'] = round(corr, 4)
                if corr < MIN_REPLICATE_CORRELATION:
                    s['qc_flags'].append(f"low_correlation ({corr:.3f} < {MIN_REPLICATE_CORRELATION})")
                    s['qc_pass'] = False

        return raw_spectra

    def predict_patient(self, filepaths: list[str | Path],
                        age: float = None, sex: str = None,
                        bmi: float = None, mode: str = None,
                        instrument: str = "thermo") -> dict:
        """Predict cancer for a single patient from multiple replicate spectra.

        Full pipeline matching training:
          1. Read & preprocess each spectrum (+ PDS calibration if Medical)
          2. QC: intensity gate + replicate correlation
          3. Predict each QC-passing spectrum individually
          4. Patient-level decision: mean probability + majority vote
        """
        # Step 1: Read & preprocess all replicates
        raw_spectra = []
        read_errors = []
        for fp in filepaths:
            fp = Path(fp)
            try:
                x, y = read_spectrum(fp)
                features = self.preprocess(x, y, instrument=instrument)
                raw_spectra.append({
                    'filepath': fp,
                    'x': x, 'y': y,
                    'features': features,
                })
            except Exception as e:
                read_errors.append({'file': fp.name, 'error': str(e)})

        if not raw_spectra:
            return {
                'status': 'error',
                'message': 'All spectra failed to read',
                'read_errors': read_errors,
            }

        # Step 2: QC
        raw_spectra = self._qc_replicate_set(raw_spectra)
        qc_passed = [s for s in raw_spectra if s['qc_pass']]
        qc_failed = [s for s in raw_spectra if not s['qc_pass']]

        if not qc_passed:
            return {
                'status': 'error',
                'message': f'All {len(raw_spectra)} spectra failed QC',
                'qc_summary': {
                    'total': len(raw_spectra),
                    'passed': 0,
                    'failed': len(raw_spectra),
                    'failures': [{'file': s['filepath'].name, 'flags': s['qc_flags']} for s in qc_failed],
                },
            }

        # Step 3: Predict each QC-passing spectrum
        active_mode = mode or self.default_mode
        if active_mode not in self.operating_modes:
            active_mode = self.default_mode
        threshold = self.operating_modes[active_mode]["threshold"]

        use_fusion = (age is not None and sex is not None)
        if use_fusion:
            sex_numeric = 1.0 if str(sex).upper() in ("M", "MALE") else 0.0
            bmi_val = bmi if bmi is not None else self.bmi_median
            s1_model = self.s1_fusion
            s2_model = self.s2_fusion
        else:
            s1_model = self.s1_sers
            s2_model = self.s2_sers
            sex_numeric = None
            bmi_val = None

        per_replicate = []
        cancer_probs = []
        type_prob_arrays = []

        for s in qc_passed:
            X = s['features'].reshape(1, -1)
            if use_fusion:
                clinical = np.array([[age, sex_numeric, bmi_val]])
                X = np.hstack([X, clinical])

            cp = float(s1_model.predict_proba(X)[0, 1])
            tp = s2_model.predict_proba(X)[0].copy()
            cancer_probs.append(cp)
            type_prob_arrays.append(tp)

            per_replicate.append({
                'file': s['filepath'].name,
                'cancer_probability': round(cp, 4),
                'cancer_detected': cp > threshold,
                'replicate_correlation': s.get('replicate_correlation'),
            })

        # Step 4: Patient-level aggregation
        mean_prob = float(np.mean(cancer_probs))
        n_detected = sum(1 for cp in cancer_probs if cp > threshold)
        majority_detected = n_detected > len(cancer_probs) / 2

        # Mean type probabilities across replicates
        mean_type_probs = np.mean(type_prob_arrays, axis=0)
        s2_classes = s2_model.classes_

        # Sex constraint
        constrained_types = []
        if sex is not None:
            sex_upper = str(sex).upper()
            for i, cls_idx in enumerate(s2_classes):
                name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else ""
                if sex_upper in ("M", "MALE") and name == "OVA":
                    mean_type_probs[i] = 0.0
                    constrained_types.append("OVA")
                elif sex_upper in ("F", "FEMALE") and name == "PRO":
                    mean_type_probs[i] = 0.0
                    constrained_types.append("PRO")
            prob_sum = mean_type_probs.sum()
            if prob_sum > 0:
                mean_type_probs = mean_type_probs / prob_sum

        type_prob_dict = {}
        for i, cls_idx in enumerate(s2_classes):
            name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else f"class_{cls_idx}"
            type_prob_dict[name] = round(float(mean_type_probs[i]), 4)

        best_idx = mean_type_probs.argmax()
        best_cls = s2_classes[best_idx]
        best_name = self.cancer_types[best_cls] if best_cls < len(self.cancer_types) else f"class_{best_cls}"

        result = {
            'status': 'ok',
            'pipeline': 'full (QC + preprocess + predict + patient aggregation)',
            'model_variant': 'fusion' if use_fusion else 'sers_only',
            'instrument': instrument,
            'calibration': ('pds_medical_to_thermo'
                            if (instrument or '').lower() in ('medical', 'medical_raman')
                            and self.pds is not None else 'none'),
            'operating_mode': active_mode,
            'threshold': threshold,

            # QC summary
            'qc_summary': {
                'total': len(raw_spectra),
                'passed': len(qc_passed),
                'failed': len(qc_failed),
                'failures': [{'file': s['filepath'].name, 'flags': s['qc_flags']} for s in qc_failed],
            },

            # Patient-level decision
            'patient_decision': {
                'screening_index': round(mean_prob, 4),
                'cancer_detected': majority_detected,
                'majority_vote': f'{n_detected}/{len(qc_passed)}',
                'method': 'mean_probability + majority_vote',
            },

            # Cancer type (only meaningful if detected)
            'cancer_type_prediction': best_name if majority_detected else None,
            'cancer_type_confidence': round(float(mean_type_probs[best_idx]), 4) if majority_detected else None,
            'cancer_type_probabilities': type_prob_dict if majority_detected else None,

            # Per-replicate details
            'per_replicate': per_replicate,

            # Clinical
            'clinical_features': {'age': age, 'sex': sex, 'bmi': bmi_val} if use_fusion else None,
        }

        if constrained_types:
            result['sex_constrained_types'] = constrained_types

        if read_errors:
            result['read_errors'] = read_errors

        return result

    def predict_batch(self, filepaths: list[str | Path],
                      age: float = None, sex: str = None,
                      bmi: float = None, mode: str = None,
                      instrument: str = "thermo") -> list[dict]:
        """Predict cancer for multiple spectrum files (individual, no QC aggregation).
        For patient-level prediction with QC, use predict_patient() instead."""
        return [self.predict_single(fp, age, sex, bmi, mode, instrument=instrument)
                for fp in filepaths]


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


class StackingPredictor(ProductionPredictor):
    """Stacking V2 ensemble predictor (10 base + ElasticNet meta).

    Inherits QC logic and predict_patient from ProductionPredictor,
    overrides model loading and single-spectrum prediction.
    """

    def __init__(self, artifact_dir: Path | str = None):
        if artifact_dir is None:
            artifact_dir = PROJECT_ROOT / "models" / "production_stacking"
        self.artifact_dir = Path(artifact_dir)

        if not self.artifact_dir.exists():
            raise FileNotFoundError(
                f"Stacking model artifacts not found at {self.artifact_dir}\n"
                f"Run: python models/build_production_stacking.py"
            )

        with open(self.artifact_dir / "manifest.json") as f:
            self.manifest = json.load(f)

        with open(self.artifact_dir / "preprocessing.json") as f:
            self.prep = json.load(f)

        self.grid = np.load(self.artifact_dir / "common_grid.npy")

        # Load base models
        with open(self.artifact_dir / "base_models_config.json") as f:
            self.base_config = json.load(f)

        self.base_models = {}
        for name, cfg in self.base_config.items():
            self.base_models[name] = {
                "s1": joblib.load(self.artifact_dir / f"base_{name}_s1.joblib"),
                "s2": joblib.load(self.artifact_dir / f"base_{name}_s2.joblib"),
                "channels": cfg["channels"],
                "model": cfg["model"],
            }

        # Load meta-learners
        self.meta_s1 = joblib.load(self.artifact_dir / "meta_s1.joblib")
        self.meta_s2 = joblib.load(self.artifact_dir / "meta_s2.joblib")

        # Load peak config
        with open(self.artifact_dir / "peak_config.json") as f:
            self.peak_config = json.load(f)
        self.known_peaks = self.peak_config["known_peaks"]

        self.cancer_types = self.manifest["cancer_types"]
        self.n_classes = len(self.cancer_types)
        self.bmi_median = 24.0  # default

        self.operating_modes = self.manifest.get("operating_modes", {})
        self.default_mode = self.manifest.get("default_mode", "screening")

        # Compatibility: set s1/s2 models for parent class methods that check them
        self.s1_sers = self.meta_s1
        self.s2_sers = self.meta_s2
        self.s1_fusion = self.meta_s1
        self.s2_fusion = self.meta_s2

        # Load PDS calibration artifact (shared logic with ProductionPredictor)
        self.pds = None
        pds_path = self.artifact_dir / "calibration" / "pds.npz"
        if pds_path.exists():
            try:
                npz = np.load(pds_path)
                if npz["grid"].shape == self.grid.shape and np.allclose(npz["grid"], self.grid):
                    self.pds = {
                        "medical_to_thermo": {
                            "F": npz["F_medical_to_thermo"],
                            "mu_s": npz["mu_s_medical_to_thermo"],
                            "mu_t": npz["mu_t_medical_to_thermo"],
                        },
                        "thermo_to_medical": {
                            "F": npz["F_thermo_to_medical"],
                            "mu_s": npz["mu_s_thermo_to_medical"],
                            "mu_t": npz["mu_t_thermo_to_medical"],
                        },
                    }
                    logger.info(f"Loaded PDS calibration from {pds_path}")
            except Exception as e:
                logger.warning(f"Failed to load PDS calibration: {e}")
        self.medical_wavenumber_shift = -28.0

        logger.info(f"StackingPredictor loaded: {len(self.base_models)} base models, "
                     f"cancer_types={self.cancer_types}")

    def _preprocess_multichannel(self, x: np.ndarray, y: np.ndarray,
                                 instrument: str = "thermo") -> np.ndarray:
        """Preprocess raw spectrum to 3-channel array (raw, d1, d2).

        For instrument='medical', applies the -28 cm⁻¹ urea shift and skips
        SG smoothing on the d0 channel (Medical BG-removed data is already
        smoothed). After trim+norm, applies the Medical→Thermo PDS transform
        to each channel independently.

        NOTE: Applying PDS to d1/d2 channels is an approximation — the PDS
        transform was fit on SNV-normalized raw spectra, but stacking uses
        SG-derivative channels. Since PDS is effectively a banded linear map,
        per-channel application is a reasonable first-order correction; a
        cleaner alternative would refit PDS per derivative order.
        """
        from scipy.signal import savgol_filter as sg
        sw = self.prep.get("smooth_window", 11)
        sp = self.prep.get("smooth_poly", 3)
        bw = self.prep.get("baseline_window", 101)
        region = tuple(self.prep.get("trim_region", [400, 2200]))

        from src.sers.preprocessing import trim_spectrum, baseline_correction, normalize_spectrum, resample as rs

        inst = (instrument or "thermo").lower()
        if inst in ("medical", "medical_raman"):
            x = x + self.medical_wavenumber_shift

        channels = []
        for deriv in [0, 1, 2]:
            # Medical BG-removed data is pre-smoothed on d0, but derivatives
            # still need an SG kernel to produce the derivative itself.
            if inst in ("medical", "medical_raman") and deriv == 0:
                y_proc = y.copy()
            else:
                y_proc = sg(y, sw, sp, deriv=deriv)
            x_tr, y_tr = trim_spectrum(x.copy(), y_proc, region=region)
            if deriv == 0:
                y_tr = baseline_correction(y_tr, window=bw)
            y_tr = normalize_spectrum(y_tr, method="snv")
            channels.append(rs(x_tr, y_tr, self.grid))

        multichannel = np.stack(channels, axis=0)  # (3, n_features)

        # Apply PDS (Medical → Thermo) per channel if applicable
        if inst in ("medical", "medical_raman") and self.pds is not None:
            p = self.pds["medical_to_thermo"]
            for ci in range(multichannel.shape[0]):
                v = multichannel[ci:ci + 1, :]
                multichannel[ci, :] = ((v - p["mu_s"]) @ p["F"] + p["mu_t"]).ravel()

        return multichannel

    def _extract_peak_features_single(self, spectrum_raw: np.ndarray) -> np.ndarray:
        """Extract peak features from a single raw-channel spectrum."""
        from scipy.optimize import curve_fit
        from scipy.special import voigt_profile
        from scipy.integrate import trapezoid

        def voigt_func(x, amplitude, center, sigma, gamma):
            return amplitude * voigt_profile(x - center, sigma, gamma)

        wn = self.grid
        n_peaks = len(self.known_peaks)
        areas = np.zeros(n_peaks)
        heights = np.zeros(n_peaks)
        fwhms = np.zeros(n_peaks)
        shifts = np.zeros(n_peaks)

        for pi, (center, name, half_w) in enumerate(self.known_peaks):
            mask = (wn >= center - half_w * 1.5) & (wn <= center + half_w * 1.5)
            if mask.sum() < 5:
                continue
            x_region = wn[mask]
            y_shifted = spectrum_raw[mask] - np.min(spectrum_raw[mask])
            try:
                amp_guess = max(np.max(y_shifted), 1e-6)
                popt, _ = curve_fit(
                    voigt_func, x_region, y_shifted,
                    p0=[amp_guess, center, half_w / 3, half_w / 3],
                    bounds=([0, center - half_w, 0.1, 0.1],
                            [amp_guess * 10, center + half_w, half_w * 2, half_w * 2]),
                    maxfev=2000,
                )
                amp, ctr, sigma, gamma = popt
                y_fit = voigt_func(x_region, *popt)
                area = trapezoid(y_fit, x_region)
                fL = 2 * gamma
                fG = 2 * sigma * np.sqrt(2 * np.log(2))
                fwhm = 0.5346 * fL + np.sqrt(0.2166 * fL ** 2 + fG ** 2)
            except (RuntimeError, ValueError):
                amp, ctr, area, fwhm = 0.0, center, 0.0, 0.0
            areas[pi] = area
            heights[pi] = amp
            fwhms[pi] = fwhm
            shifts[pi] = ctr - center

        features = [areas, heights, fwhms, shifts]
        # Peak ratios
        pn2i = {p[1]: i for i, p in enumerate(self.known_peaks)}
        for na, nb in [("phe_urea", "adenine"), ("phe_urea", "creatinine"),
                       ("hippuric", "creatinine"), ("CS_stretch", "creatinine"),
                       ("amide_I", "CH2_deform"), ("adenine", "purine_CC"),
                       ("tyrosine", "phe_urea")]:
            if na in pn2i and nb in pn2i:
                features.append(np.array([areas[pn2i[na]] / (areas[pn2i[nb]] + 1e-10)]))

        return np.nan_to_num(np.hstack(features), nan=0.0, posinf=0.0, neginf=0.0)

    def _get_base_predictions(self, multichannel: np.ndarray, peak_features: np.ndarray) -> tuple:
        """Run all base models and return (s1_probs, s2_probs).

        Returns:
            s1_probs: (n_base,) array of cancer probabilities
            s2_probs: (n_base, n_classes) array of type probabilities
        """
        n_base = len(self.base_models)
        s1_probs = np.zeros(n_base)
        s2_probs = np.zeros((n_base, self.n_classes))

        for bi, (name, bm) in enumerate(self.base_models.items()):
            ch = bm["channels"]
            if ch == "peak":
                X = peak_features.reshape(1, -1)
            else:
                X = multichannel[ch, :].reshape(1, -1)

            s1_probs[bi] = bm["s1"].predict_proba(X)[0, 1]

            raw_s2 = bm["s2"].predict_proba(X)[0]
            classes = bm["s2"].classes_ if hasattr(bm["s2"], "classes_") else range(self.n_classes)
            for i, cls in enumerate(classes):
                if cls < self.n_classes:
                    s2_probs[bi, cls] = raw_s2[i]

        return s1_probs, s2_probs

    def predict_single(self, filepath: str | Path,
                       age: float = None, sex: str = None,
                       bmi: float = None, mode: str = None,
                       instrument: str = "thermo") -> dict:
        """Predict using stacking ensemble."""
        filepath = Path(filepath)
        result = {"file": filepath.name, "status": "ok", "warnings": [],
                  "instrument": instrument}

        try:
            x, y = read_spectrum(filepath)
            result["n_points"] = len(x)
            result["wavenumber_range"] = f"{x.min():.1f}-{x.max():.1f}"
            result["warnings"] = self._sanity_check(x, y, filepath)

            inst = (instrument or "thermo").lower()
            if inst in ("medical", "medical_raman"):
                if self.pds is None:
                    result["warnings"].append(
                        "Medical instrument specified but no PDS calibration "
                        "loaded — stacking predictions will be unreliable."
                    )
                else:
                    result["calibration"] = "pds_medical_to_thermo (per-channel)"

            # 3-channel preprocessing (applies PDS per channel if Medical)
            multichannel = self._preprocess_multichannel(x, y, instrument=instrument)

            # Peak features from raw channel
            peak_feat = self._extract_peak_features_single(multichannel[0])

            # Base model predictions
            s1_probs, s2_probs = self._get_base_predictions(multichannel, peak_feat)

            # Meta-learner
            meta_s1_features = s1_probs.reshape(1, -1)
            meta_s2_features = s2_probs.reshape(1, -1)
            meta_all = np.hstack([meta_s1_features, meta_s2_features])

            cancer_prob = float(self.meta_s1.predict_proba(meta_s1_features)[0, 1])

            # Operating mode
            active_mode = mode or self.default_mode
            if active_mode not in self.operating_modes:
                active_mode = self.default_mode
            threshold = self.operating_modes[active_mode]["threshold"]

            result["cancer_probability"] = round(cancer_prob, 4)
            result["cancer_detected"] = cancer_prob > threshold
            result["model_variant"] = "stacking_v2"
            result["operating_mode"] = active_mode
            result["threshold"] = threshold

            # Stage 2: cancer type
            type_probs = self.meta_s2.predict_proba(meta_all)[0].copy()
            s2_classes = self.meta_s2.classes_ if hasattr(self.meta_s2, "classes_") else \
                self.meta_s2.named_steps.get("logisticregression", self.meta_s2[-1]).classes_

            # Sex constraint
            constrained_types = []
            if sex:
                sex_upper = str(sex).upper()
                for i, cls_idx in enumerate(s2_classes):
                    ct_name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else ""
                    if sex_upper in ("M", "MALE") and ct_name == "OVA":
                        type_probs[i] = 0.0
                        constrained_types.append("OVA")
                    elif sex_upper in ("F", "FEMALE") and ct_name == "PRO":
                        type_probs[i] = 0.0
                        constrained_types.append("PRO")
                prob_sum = type_probs.sum()
                if prob_sum > 0:
                    type_probs = type_probs / prob_sum

            type_prob_dict = {}
            for i, cls_idx in enumerate(s2_classes):
                ct_name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else f"class_{cls_idx}"
                type_prob_dict[ct_name] = round(float(type_probs[i]), 4)

            best_idx = type_probs.argmax()
            best_cls = s2_classes[best_idx]
            best_name = self.cancer_types[best_cls] if best_cls < len(self.cancer_types) else f"class_{best_cls}"

            result["cancer_type_prediction"] = best_name
            result["cancer_type_confidence"] = round(float(type_probs[best_idx]), 4)
            result["cancer_type_probabilities"] = type_prob_dict

            if constrained_types:
                result["sex_constrained_types"] = constrained_types

        except Exception as e:
            result["status"] = "error"
            result["message"] = str(e)
            logger.error(f"Prediction failed for {filepath}: {e}")

        return result

    def predict_patient(self, filepaths: list[str | Path],
                        age: float = None, sex: str = None,
                        bmi: float = None, mode: str = None,
                        instrument: str = "thermo") -> dict:
        """Patient-level prediction with QC + replicate aggregation.

        Overrides parent to use stacking predict_single internally.
        """
        # Step 1: Read & preprocess all replicates
        raw_spectra = []
        read_errors = []
        for fp in filepaths:
            fp = Path(fp)
            try:
                x, y = read_spectrum(fp)
                features = self.preprocess(x, y, instrument=instrument)
                multichannel = self._preprocess_multichannel(x, y, instrument=instrument)
                peak_feat = self._extract_peak_features_single(multichannel[0])
                raw_spectra.append({
                    'filepath': fp,
                    'x': x, 'y': y,
                    'features': features,
                    'multichannel': multichannel,
                    'peak_features': peak_feat,
                })
            except Exception as e:
                read_errors.append({'file': fp.name, 'error': str(e)})

        if not raw_spectra:
            return {'status': 'error', 'message': 'All spectra failed to read', 'read_errors': read_errors}

        # Step 2: QC (reuse parent)
        raw_spectra = self._qc_replicate_set(raw_spectra)
        qc_passed = [s for s in raw_spectra if s['qc_pass']]
        qc_failed = [s for s in raw_spectra if not s['qc_pass']]

        if not qc_passed:
            return {
                'status': 'error',
                'message': f'All {len(raw_spectra)} spectra failed QC',
                'qc_summary': {
                    'total': len(raw_spectra), 'passed': 0, 'failed': len(raw_spectra),
                    'failures': [{'file': s['filepath'].name, 'flags': s['qc_flags']} for s in qc_failed],
                },
            }

        # Step 3: Predict each QC-passing spectrum via stacking
        active_mode = mode or self.default_mode
        if active_mode not in self.operating_modes:
            active_mode = self.default_mode
        threshold = self.operating_modes[active_mode]["threshold"]

        per_replicate = []
        cancer_probs = []
        type_prob_arrays = []

        for s in qc_passed:
            s1_probs, s2_probs = self._get_base_predictions(s['multichannel'], s['peak_features'])

            meta_s1_feat = s1_probs.reshape(1, -1)
            meta_s2_feat = s2_probs.reshape(1, -1)
            meta_all = np.hstack([meta_s1_feat, meta_s2_feat])

            cp = float(self.meta_s1.predict_proba(meta_s1_feat)[0, 1])
            tp = self.meta_s2.predict_proba(meta_all)[0].copy()

            cancer_probs.append(cp)
            type_prob_arrays.append(tp)

            per_replicate.append({
                'file': s['filepath'].name,
                'cancer_probability': round(cp, 4),
                'cancer_detected': cp > threshold,
                'replicate_correlation': s.get('replicate_correlation'),
            })

        # Step 4: Patient-level aggregation
        mean_prob = float(np.mean(cancer_probs))
        n_detected = sum(1 for cp in cancer_probs if cp > threshold)
        majority_detected = n_detected > len(cancer_probs) / 2

        mean_type_probs = np.mean(type_prob_arrays, axis=0)
        s2_classes = self.meta_s2.classes_ if hasattr(self.meta_s2, "classes_") else \
            self.meta_s2.named_steps.get("logisticregression", self.meta_s2[-1]).classes_

        # Sex constraint
        constrained_types = []
        if sex:
            sex_upper = str(sex).upper()
            for i, cls_idx in enumerate(s2_classes):
                ct_name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else ""
                if sex_upper in ("M", "MALE") and ct_name == "OVA":
                    mean_type_probs[i] = 0.0
                    constrained_types.append("OVA")
                elif sex_upper in ("F", "FEMALE") and ct_name == "PRO":
                    mean_type_probs[i] = 0.0
                    constrained_types.append("PRO")
            prob_sum = mean_type_probs.sum()
            if prob_sum > 0:
                mean_type_probs = mean_type_probs / prob_sum

        type_prob_dict = {}
        for i, cls_idx in enumerate(s2_classes):
            ct_name = self.cancer_types[cls_idx] if cls_idx < len(self.cancer_types) else f"class_{cls_idx}"
            type_prob_dict[ct_name] = round(float(mean_type_probs[i]), 4)

        best_idx = mean_type_probs.argmax()
        best_cls = s2_classes[best_idx]
        best_name = self.cancer_types[best_cls] if best_cls < len(self.cancer_types) else f"class_{best_cls}"

        result = {
            'status': 'ok',
            'pipeline': 'stacking_v2 (10 base + ElasticNet meta)',
            'model_variant': 'stacking_v2',
            'operating_mode': active_mode,
            'threshold': threshold,
            'qc_summary': {
                'total': len(raw_spectra), 'passed': len(qc_passed), 'failed': len(qc_failed),
                'failures': [{'file': s['filepath'].name, 'flags': s['qc_flags']} for s in qc_failed],
            },
            'patient_decision': {
                'screening_index': round(mean_prob, 4),
                'cancer_detected': majority_detected,
                'majority_vote': f'{n_detected}/{len(qc_passed)}',
                'method': 'mean_probability + majority_vote',
            },
            'cancer_type_prediction': best_name if majority_detected else None,
            'cancer_type_confidence': round(float(mean_type_probs[best_idx]), 4) if majority_detected else None,
            'cancer_type_probabilities': type_prob_dict if majority_detected else None,
            'per_replicate': per_replicate,
        }

        if constrained_types:
            result['sex_constrained_types'] = constrained_types
        if read_errors:
            result['read_errors'] = read_errors

        return result


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
    p.add_argument("--mode", "-m", default=None,
                   choices=["screening", "balanced", "confirmatory"],
                   help="Operating mode: screening (high sensitivity), balanced, confirmatory (high specificity)")
    p.add_argument("--instrument", "-i", default="thermo",
                   choices=["thermo", "medical"],
                   help="Source instrument. 'medical' triggers PDS calibration "
                        "(Medical→Thermo) before model inference. Default: thermo.")
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
        mode = args.mode or predictor.default_mode
        mode_info = predictor.operating_modes.get(mode, {})
        print(f"  Model: {variant}")
        print(f"  Mode: {mode} — {mode_info.get('description', '')}")
        print(f"  Threshold: {mode_info.get('threshold', 0.5)}")
        print(f"  Files: {len(args.spectra)}")
        if args.age:
            print(f"  Patient: age={args.age}, sex={args.sex}, bmi={args.bmi or 'auto'}")
        print("-" * 60)

    # Run predictions
    results = predictor.predict_batch(args.spectra, args.age, args.sex, args.bmi,
                                       args.mode, instrument=args.instrument)

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
