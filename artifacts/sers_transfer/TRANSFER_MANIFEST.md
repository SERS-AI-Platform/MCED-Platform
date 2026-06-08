# SERS Clinical Transfer Package

Created: 2026-06-08

This folder mirrors the analysis outputs copied from `/Users/ian/Downloads/data/organized_data/06_analysis_results`.

## Data Copy

Organized SERS and clinical data were copied to:

- `/Users/ian/MCED-Platform/data/sers_clinical_organized`

This copy includes:

- `01_clinical_metadata`
- `02_sers_primary_pooled_acquisition`
- `03_sers_date_lot_balanced_acquisition`
- `04_machine_repeatability_tests`
- `05_not_yet_analyzed`
- `06_analysis_results`
- `99_manifests`

## Analysis Scripts

Reproduction scripts were copied to:

- `/Users/ian/MCED-Platform/scripts/analysis/sers_transfer`

Scripts:

- `organize_sers_data.py`
- `organize_clinical_metadata.py`
- `run_primary_to_lot_balanced_same_id_test.py`
- `generate_primary_lot_balanced_difference_figures.py`
- `generate_group_sample_match_figures_by_device.py`

## Main Outputs

Use this result first for device-wise group/sample matching:

- `/Users/ian/MCED-Platform/artifacts/sers_transfer/group_sample_match_by_device/FIGURE_GUIDE_KO.md`
- `/Users/ian/MCED-Platform/artifacts/sers_transfer/group_sample_match_by_device/figures`

Supporting outputs:

- `primary_to_lot_balanced_same_id_test_axis_aligned`
- `primary_lot_balanced_difference_diagnostics`
- `group_sample_match_by_device`

## Current Matching Summary

The device-wise group/sample figures use:

- feature: `vector_derivative_snv`
- preprocessing: 400-1800 cm-1 interpolation, SNV, Savitzky-Golay first derivative + SNV
- x-axis correction: lot-balanced spectra corrected by date-level PS/Si median reference shift
- exclusions: averaged files, NF, Po./post-op, BPRO/BNOR, blank/reference/QC

Summary:

| Dataset | n samples | group accuracy | exact sample accuracy | median true rank |
| --- | ---: | ---: | ---: | ---: |
| Primary Thermo CV | 1700 | 0.9282 | 0.7488 | 1.0 |
| Thermo | 1611 | 0.2315 | 0.0317 | 408.0 |
| Handheld | 1609 | 0.2436 | 0.0273 | 374.0 |
| Medical | 1609 | 0.0777 | 0.0099 | 516.5 |
