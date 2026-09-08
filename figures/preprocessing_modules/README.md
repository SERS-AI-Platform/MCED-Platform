# Preprocessing Module Figures

Current repo preprocessing defaults represented here:

- Wavenumber calibration: urea 1001.4 cm-1, +/-10 cm-1 search
- Fingerprint trim: 400-2200 cm-1
- Smoothing: Savitzky-Golay window 11, polynomial 3
- Baseline correction: rolling minimum window 101
  - Baseline-corrected intensity is nonnegative before SNV.
- Normalization: SNV
  - SNV mean-centers the baseline-corrected spectrum, so the final model input can contain negative values.
- Fixed grid resampling: 402-2198 cm-1, 935 points

Additional workflow diagram:

- `07_aecd_repeat_spectrum_pipeline.html`: Editable 16:9 HTML/CSS source for AECD repeat-spectrum calibration, robust within-subject QC, subject mean aggregation, and four-model elastic-net stacking.
- `07_aecd_repeat_spectrum_pipeline.png`: Rendered 1600x900 figure. Mini spectra are synthetic and illustrate processing only.
- `08_aecd_pipeline_modules.html`: Editable 3-by-2 module-chart source for the AECD repeat-spectrum workflow.
- `08_aecd_pipeline_modules.png`: Rendered 1600x900 module overview matching the six-chart pipeline format.

Real AECD subject preprocessing figures:

- `aecd_subject_real/01_raw_repeat_spectra.png`: One de-identified subject's raw clinical repeat spectra from the AECD API.
- `aecd_subject_real/02_ps_wavenumber_calibration.png`: Measurement-date and instrument-matched PS Raman-shift calibration.
- `aecd_subject_real/03_common_grid_interpolation.png`: Linear interpolation onto the 400-2200 cm-1 common grid.
- `aecd_subject_real/04_within_subject_robust_qc.png`: Median/MAD robust spectral-distance QC for the repeats.
- `aecd_subject_real/05_qc_retained_spectra.png`: Retained and excluded repeats after QC.
- `aecd_subject_real/06_subject_mean_spectrum.png`: Arithmetic mean of QC-passed repeats, with no later spectral preprocessing.
- `aecd_subject_real/manifest.json`: De-identified source, point-count, and QC provenance for the rendered figures.

Generate the real-data figures with:

```bash
uv run scripts/visualization/aecd_subject_preprocessing_figures.py
```

Example spectrum: `/Users/ian/Desktop/임상데이터/20260508_Urine test/1. NOR/NOR 68_1.CSV`
