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

Example spectrum: `/Users/ian/Desktop/임상데이터/20260508_Urine test/1. NOR/NOR 68_1.CSV`
