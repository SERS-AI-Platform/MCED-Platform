# Corrected Primary-to-Lot-Balanced Same-ID Test

## Correction

`BPRO` and `BNOR` are treated as distinct groups, not as `PRO`/`NOR`.
The previous `BPRO -> PRO` assumption was invalid and the old output folder was archived as invalid.

## Setup

- Train/source acquisition: primary pooled Thermo raw data.
- Test acquisitions: lot-balanced Thermo, Handheld, and Medical folders.
- Candidate labels: exact overlapping sample IDs only per test dataset.
- Excluded: averaged files, NF files, Po./post-op files, BPRO, BNOR, blank/reference/QC files.
- Preprocessing: interpolate 400-1800 cm-1, SNV, and Savitzky-Golay first derivative + SNV.
- Axis alignment: Lot-balanced spectra were x-axis corrected by subtracting the date-level median PS/Si reference peak shift. Primary spectra were left unchanged because no primary PS/Si reference folders were found in the organized copy.

## Data Used

- Primary replicate spectra: 8500
- Primary sample IDs: 1700
- lot_balanced_thermo: 8142 replicate spectra, 1628 sample IDs, 1611 exact overlaps
- lot_balanced_handheld: 8133 replicate spectra, 1626 sample IDs, 1609 exact overlaps
- lot_balanced_medical: 9753 replicate spectra, 1626 sample IDs, 1609 exact overlaps

## Main Metrics

| level | feature | candidate_mode | test_dataset | n | identity_top1_accuracy | identity_top3_accuracy | identity_top5_accuracy | group_accuracy | median_true_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replicate | vector_derivative_snv | dataset_overlap_only | lot_balanced_handheld | 8048 | 0.0275 | 0.0503 | 0.0651 | 0.2398 | 366.0000 |
| replicate | vector_derivative_snv | dataset_overlap_only | lot_balanced_medical | 9651 | 0.0103 | 0.0240 | 0.0349 | 0.0867 | 514.0000 |
| replicate | vector_derivative_snv | dataset_overlap_only | lot_balanced_thermo | 8057 | 0.0284 | 0.0503 | 0.0604 | 0.2340 | 410.0000 |
| sample_majority | vector_derivative_snv | dataset_overlap_only | lot_balanced_handheld | 1609 | 0.0273 |  |  | 0.2436 | 374.0000 |
| sample_majority | vector_derivative_snv | dataset_overlap_only | lot_balanced_medical | 1609 | 0.0099 |  |  | 0.0777 | 516.5000 |
| sample_majority | vector_derivative_snv | dataset_overlap_only | lot_balanced_thermo | 1611 | 0.0317 |  |  | 0.2315 | 408.0000 |

## Figures

- `figures/01_accuracy_summary.png`
- `figures/02_true_rank_distribution.png`
- `figures/03_group_same_id_heatmap.png`
- `figures/04_group_confusion_matrices.png`
- `figures/05_before_after_axis_alignment_accuracy.png` when a baseline comparison is available.
- `figures/06_before_after_axis_alignment_rank.png` when a baseline comparison is available.

## Interpretation Guide

Compare the primary leave-one-replicate control with lot-balanced test rows.
If primary CV is high but lot-balanced same-ID accuracy is low, the same sample
identity is not stable across acquisition/date/lot/instrument under this preprocessing.
If disease/group accuracy remains high, disease-level signal may still transfer even
when subject identity does not.
