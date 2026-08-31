from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "notebooks/aecd_api_model_mean_spectrum_outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
PUBLICATION_FIGURE_DIR = (
    REPO_ROOT / "publications/전향검체/보라매병원/figures/mean_spectrum"
)
SOURCE_SCRIPT = (
    REPO_ROOT / "scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py"
)


def load_source_module():
    spec = importlib.util.spec_from_file_location("mean_spectrum_pipeline", SOURCE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the mean-spectrum pipeline module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_saved_arrays() -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    pd.DataFrame,
    np.ndarray,
    np.ndarray,
]:
    mean_frame = pd.read_csv(
        OUTPUT_DIR / "mean_representative_spectra.csv",
        encoding="utf-8-sig",
    )
    band_frame = pd.read_csv(
        OUTPUT_DIR / "signal_noise_band_summary.csv",
        encoding="utf-8-sig",
    )
    shift_values = pd.read_csv(
        OUTPUT_DIR / "calibration_shift_values.csv",
        encoding="utf-8-sig",
    )["calibration_shift_cm1"].to_numpy(dtype=np.float64)
    wn_columns = [column for column in mean_frame.columns if column.startswith("wn_")]
    common_grid = np.asarray(
        [float(column.removeprefix("wn_")) for column in wn_columns],
        dtype=np.float64,
    )
    means = mean_frame[wn_columns].to_numpy(dtype=np.float64)
    labels = mean_frame["label"].astype(str).tolist()
    noise_by_label: dict[str, np.ndarray] = {}
    for label, subset in band_frame.groupby("label"):
        ordered = subset.sort_values("wavenumber_cm1")
        if len(ordered) != len(common_grid):
            raise RuntimeError(f"Band summary grid length mismatch for {label}.")
        if not np.allclose(
            ordered["wavenumber_cm1"].to_numpy(dtype=np.float64),
            common_grid,
            rtol=0,
            atol=1e-3,
        ):
            raise RuntimeError(f"Band summary grid values mismatch for {label}.")
        noise_by_label[str(label)] = ordered["noise_sigma_rep"].to_numpy(
            dtype=np.float64
        )
    noise_vectors = np.vstack([noise_by_label[label] for label in labels])
    return common_grid, means, labels, band_frame, noise_vectors, shift_values


def main() -> None:
    pipeline = load_source_module()
    common_grid, means, labels, band_frame, noise_vectors, shift_values = load_saved_arrays()
    peak_registry = pipeline.build_resolution_aware_peak_registry(
        common_grid,
        band_frame,
    )
    if peak_registry.empty:
        raise RuntimeError("No resolution-aware common peak candidates were found.")
    output_registry = OUTPUT_DIR / "resolution_aware_peak_registry.csv"
    peak_registry.to_csv(output_registry, index=False, encoding="utf-8-sig")
    output_figure = FIGURE_DIR / "resolution_aware_peak_zoom.png"
    pipeline.plot_resolution_aware_peak_zoom(
        common_grid,
        means,
        labels,
        noise_vectors,
        band_frame,
        peak_registry,
        output_figure,
    )
    publication_figure = PUBLICATION_FIGURE_DIR / "fig08_resolution_aware_peak_zoom.png"
    shutil.copy2(output_figure, publication_figure)
    overview_figure = FIGURE_DIR / "signal_noise_band_validation.png"
    pipeline.plot_signal_noise_band_validation(
        common_grid,
        means,
        labels,
        noise_vectors,
        [],
        band_frame,
        peak_registry,
        shift_values,
        overview_figure,
    )
    publication_overview_figure = PUBLICATION_FIGURE_DIR / "fig07_signal_noise_band_validation.png"
    shutil.copy2(overview_figure, publication_overview_figure)
    snr_figure = FIGURE_DIR / "resolution_aware_snr_zoom.png"
    pipeline.plot_resolution_aware_snr_zoom(
        common_grid,
        band_frame,
        peak_registry,
        snr_figure,
    )
    publication_snr_figure = PUBLICATION_FIGURE_DIR / "fig09_resolution_aware_snr_zoom.png"
    shutil.copy2(snr_figure, publication_snr_figure)
    common = peak_registry[peak_registry["common_candidate"]]
    validation_metadata = {
        "source": str(OUTPUT_DIR / "mean_representative_spectra.csv"),
        "effective_resolution_fwhm_cm1": pipeline.INSTRUMENT_RESOLUTION_FWHM_CM1,
        "same_peak_match_tolerance_cm1": pipeline.INSTRUMENT_RESOLUTION_FWHM_CM1,
        "minimum_separation_points": pipeline.resolution_min_distance_points(
            common_grid
        ),
        "candidate_rule": "group-median local contrast peak with mean-spectrum SNR >= 3",
        "common_candidate_count": int(len(common)),
        "common_candidate_centers_cm1": [
            float(value)
            for value in common["representative_wavenumber_cm1"]
        ],
        "signal_band_definition": "candidate peak with mean-spectrum SNR >= 3, shaded over the peak half-prominence width",
        "repeat_level_signal_band_definition": "repeat SNR >= 3 from local contrast divided by repeat residual SD",
        "repeat_level_signal_band_fraction": {
            str(label): float(frame["signal_band_snr3"].mean())
            for label, frame in band_frame.groupby("label")
        },
    }
    (OUTPUT_DIR / "resolution_aware_peak_validation.json").write_text(
        json.dumps(validation_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    metadata_path = OUTPUT_DIR / "run_metadata.json"
    if metadata_path.exists():
        run_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        run_metadata.setdefault("signal_noise_band", {})[
            "resolution_aware_peak_selection"
        ] = validation_metadata
        run_metadata["resolution_aware_peak_validation"] = validation_metadata
        run_metadata.setdefault("publication_figures", {}).update(
            {
                "fig07_signal_noise_band_validation.png": str(
                    publication_overview_figure
                ),
                "fig08_resolution_aware_peak_zoom.png": str(publication_figure),
                "fig09_resolution_aware_snr_zoom.png": str(publication_snr_figure),
            }
        )
        metadata_path.write_text(
            json.dumps(run_metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(f"common_candidates={len(common)}")
    print(", ".join(f"{value:.2f}" for value in common["representative_wavenumber_cm1"]))
    print(f"figure={output_figure}")
    print(f"publication_figure={publication_figure}")
    print(f"overview_figure={overview_figure}")
    print(f"publication_overview_figure={publication_overview_figure}")
    print(f"snr_figure={snr_figure}")
    print(f"publication_snr_figure={publication_snr_figure}")


if __name__ == "__main__":
    main()
