# Corrected Primary-to-Lot-Balanced Same-ID Test

## Correction

`BPRO` and `BNOR` are treated as distinct groups, not as `PRO`/`NOR`.
The previous `BPRO -> PRO` assumption was invalid and the old output folder was archived as invalid.

## Setup

- Train/source acquisition: primary pooled Thermo raw data.
- Test acquisitions: lot-balanced Thermo and Handheld folders.
- Candidate labels: exact overlapping sample IDs only per test dataset.
- Excluded: averaged files, NF files, Po./post-op files, BPRO, BNOR, blank/reference/QC files.
- Preprocessing: interpolate 400-1800 cm-1, SNV, and Savitzky-Golay first derivative + SNV.

## Data Used

- Primary replicate spectra: 8500
- Primary sample IDs: 1700
- lot_balanced_thermo: 8142 replicate spectra, 1628 sample IDs, 1611 exact overlaps
- lot_balanced_handheld: 8133 replicate spectra, 1626 sample IDs, 1609 exact overlaps
- lot_balanced_medical: 9753 replicate spectra, 1626 sample IDs, 1609 exact overlaps

## Main Metrics

| level | feature | candidate_mode | test_dataset | n | identity_top1_accuracy | identity_top3_accuracy | identity_top5_accuracy | group_accuracy | median_true_rank |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| replicate | vector_derivative_snv | dataset_overlap_only | lot_balanced_handheld | 8048 | 0.0272 | 0.0504 | 0.0657 | 0.2408 | 367.0000 |
| replicate | vector_derivative_snv | dataset_overlap_only | lot_balanced_medical | 9651 | 0.0097 | 0.0248 | 0.0362 | 0.0876 | 511.0000 |
| replicate | vector_derivative_snv | dataset_overlap_only | lot_balanced_thermo | 8057 | 0.0281 | 0.0489 | 0.0601 | 0.2299 | 410.0000 |
| sample_majority | vector_derivative_snv | dataset_overlap_only | lot_balanced_handheld | 1609 | 0.0280 |  |  | 0.2455 | 373.0000 |
| sample_majority | vector_derivative_snv | dataset_overlap_only | lot_balanced_medical | 1609 | 0.0093 |  |  | 0.0827 | 515.0000 |
| sample_majority | vector_derivative_snv | dataset_overlap_only | lot_balanced_thermo | 1611 | 0.0304 |  |  | 0.2278 | 407.0000 |

## Figures

- `figures/01_accuracy_summary.png`
- `figures/02_true_rank_distribution.png`
- `figures/03_group_same_id_heatmap.png`
- `figures/04_group_confusion_matrices.png`

## Interpretation Guide

Compare the primary leave-one-replicate control with lot-balanced test rows.
If primary CV is high but lot-balanced same-ID accuracy is low, the same sample
identity is not stable across acquisition/date/lot/instrument under this preprocessing.
If disease/group accuracy remains high, disease-level signal may still transfer even
when subject identity does not.
