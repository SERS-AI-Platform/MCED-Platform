# Primary vs Lot-Balanced Difference Diagnostics

## Scope

These figures diagnose the residual mismatch after PS/Si reference x-axis alignment.
Primary spectra are used as the reference acquisition; lot-balanced Thermo, Handheld, and Medical sample centroids are compared to matched primary sample centroids.

## Main Finding

The same lot-balanced sample is usually not closest to its true primary centroid. A wrong primary sample often has higher cosine similarity than the true primary sample.
This points to residual acquisition/date/lot/instrument effects and spectral-shape differences, not just a simple wavenumber-axis offset.

## Similarity Summary

| Dataset | n | same-ID acc. | group acc. | median true rank | median true sim. | median best wrong sim. | median true-best margin |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Thermo | 1611 | 0.0310 | 0.2377 | 405.0 | 0.565 | 0.893 | -0.301 |
| Handheld | 1609 | 0.0280 | 0.2449 | 369.0 | 0.493 | 0.781 | -0.269 |
| Medical | 1609 | 0.0118 | 0.0845 | 501.0 | 0.134 | 0.436 | -0.280 |

## Figures

- `figures/01_pca_centroids_acquisition_vs_group.png`: whether acquisition separates the feature space.
- `figures/02_pca_same_sample_shift_vectors.png`: same sample movement from primary to each lot-balanced acquisition.
- `figures/03_same_id_vs_wrong_primary_similarity.png`: true primary similarity versus wrong-primary similarity distributions.
- `figures/04_true_vs_best_wrong_similarity_scatter.png`: pointwise true-vs-best-wrong comparison.
- `figures/05_wrong_minus_true_similarity_margin.png`: how often the wrong primary centroid wins.
- `figures/06_mean_snv_spectra_and_delta_vs_primary.png`: average residual spectrum shape difference.
- `figures/07_abs_delta_by_wavenumber_vs_primary.png`: wavenumber regions with largest residual difference.
- `figures/08_group_delta_heatmap_vs_primary.png`: group-level residual spectral differences.
- `figures/09_same_id_accuracy_by_date_heatmap.png`: date-level same-ID performance.
- `figures/10_reference_shift_vs_same_id_accuracy.png`: shows reference x-axis shift size is not enough to explain mismatch.
- `figures/11_failure_examples_true_vs_wrong_primary_overlay.png`: concrete failed examples.
- `figures/12_pc_variance_explained_by_metadata.png`: metadata factor effect size over first 20 PCs.

## Metadata Effect Size

- Acquisition/instrument: weighted PC R2 = 0.503
- Clinical group: weighted PC R2 = 0.053
- Date folder: weighted PC R2 = 0.222

## Date Summary

- Date-level rows: 30
- Sample centroid rows: 6441
