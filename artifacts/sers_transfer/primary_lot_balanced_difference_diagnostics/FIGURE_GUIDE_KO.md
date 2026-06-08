# Primary vs Lot-Balanced Difference Figure Guide

## 핵심 결론

PS/Si reference로 x-axis를 맞춘 뒤에도, lot-balanced sample은 자기 자신의 primary centroid보다 다른 primary sample centroid에 더 가까운 경우가 대부분입니다. 따라서 지금 불일치의 주된 원인은 단순 Raman shift 보정 문제가 아니라 acquisition/instrument/date/lot에 따른 residual spectral-shape 차이입니다.

요약 수치:

| Dataset | same-ID acc. | group acc. | median true rank | median true sim. | median best wrong sim. |
| --- | ---: | ---: | ---: | ---: | ---: |
| Thermo | 0.0310 | 0.2377 | 405.0 | 0.565 | 0.893 |
| Handheld | 0.0280 | 0.2449 | 369.0 | 0.493 | 0.781 |
| Medical | 0.0118 | 0.0845 | 501.0 | 0.134 | 0.436 |

PC metadata effect size:

| Factor | weighted PC R2 |
| --- | ---: |
| Acquisition/instrument | 0.503 |
| Date folder | 0.222 |
| Clinical group | 0.053 |

즉 현재 feature에서는 clinical group보다 acquisition/instrument/date 효과가 훨씬 큽니다.

## Figure별 의미

### 01_pca_centroids_acquisition_vs_group.png

sample centroid를 PCA로 펼친 그림입니다. 왼쪽은 acquisition/instrument 색상, 오른쪽은 clinical group 색상입니다. acquisition 색이 더 뚜렷하게 갈라지면 batch/instrument 효과가 group signal보다 큰 것입니다.

### 02_pca_same_sample_shift_vectors.png

같은 sample ID의 primary 위치에서 lot-balanced 위치로 이동한 방향을 선으로 표시합니다. 선들이 무작위가 아니라 특정 방향으로 밀리면, 같은 sample이라도 acquisition 조건이 feature 위치를 체계적으로 바꾸고 있다는 뜻입니다.

### 03_same_id_vs_wrong_primary_similarity.png

lot-balanced sample이 true primary sample과 얼마나 비슷한지, 그리고 잘못된 primary sample과 얼마나 비슷한지를 boxplot으로 비교합니다. true same sample보다 best wrong primary가 높게 나오면 same-ID가 실패하는 직접적인 이유입니다.

### 04_true_vs_best_wrong_similarity_scatter.png

x축은 true primary similarity, y축은 best wrong primary similarity입니다. 점이 대각선 위에 있으면 wrong primary가 true primary보다 더 가까운 것입니다. 대부분 위에 있으면 classifier가 같은 sample을 맞출 수 없습니다.

### 05_wrong_minus_true_similarity_margin.png

`best wrong similarity - true similarity` 분포입니다. 0보다 크면 같은 sample이 wrong sample에게 진 것입니다. 분포가 양수 쪽에 있으면 sample identity가 acquisition을 넘어 안정적이지 않다는 뜻입니다.

### 06_mean_snv_spectra_and_delta_vs_primary.png

위쪽은 primary와 lot-balanced 평균 SNV spectrum, 아래쪽은 matched sample 기준 `lot-balanced - primary` 평균 차이입니다. x-axis 보정 후에도 baseline/peak shape 차이가 남는지 봅니다.

### 07_abs_delta_by_wavenumber_vs_primary.png

어느 Raman shift 구간에서 primary와 lot-balanced 차이가 큰지 보여줍니다. peak 위치 문제가 아니라 특정 spectral region intensity/shape 차이가 크면 여기서 드러납니다.

### 08_group_delta_heatmap_vs_primary.png

group별로 `lot-balanced - matched primary` 차이를 heatmap으로 본 그림입니다. 특정 group에서만 문제가 큰지, 아니면 acquisition 전체에서 공통적으로 차이가 나는지 확인합니다.

### 09_same_id_accuracy_by_date_heatmap.png

date folder별 same-ID accuracy입니다. 특정 날짜 하나만 문제가 아니라 대부분 날짜에서 낮으면, 단일 bad batch보다 전체 acquisition protocol/lot/instrument 차이 문제에 가깝습니다.

### 10_reference_shift_vs_same_id_accuracy.png

PS/Si reference x-axis shift 크기와 same-ID accuracy의 관계입니다. shift가 큰 날짜만 accuracy가 낮은 패턴이 아니면, wavenumber drift만으로는 불일치를 설명하기 어렵습니다.

### 11_failure_examples_true_vs_wrong_primary_overlay.png

실제로 실패한 sample 몇 개를 골라 lot-balanced spectrum, true primary spectrum, best wrong primary spectrum을 같이 그린 것입니다. wrong primary가 왜 더 가깝게 잡히는지 spectrum shape로 확인하는 예시입니다.

### 12_pc_variance_explained_by_metadata.png

첫 20개 PC에서 acquisition/instrument, date folder, clinical group이 feature variation을 얼마나 설명하는지 비교합니다. acquisition/instrument와 date가 group보다 훨씬 크면 current preprocessing으로는 biological signal보다 batch signal이 더 강하다는 의미입니다.
