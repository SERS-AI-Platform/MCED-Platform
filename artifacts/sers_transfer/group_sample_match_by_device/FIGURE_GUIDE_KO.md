# Group and Sample Matching by Device

이 figure set은 각 장비에서 먼저 clinical group이 맞는지, 그리고 그 안에서 exact sample ID까지 맞는지를 보기 위한 것입니다.

상태 정의:

- Exact sample match: predicted sample ID가 true sample ID와 완전히 같음.
- Group only: clinical group은 맞지만 sample ID는 다름.
- Wrong group: clinical group부터 틀림.

## Device Summary

| Dataset | n | group acc. | exact sample acc. | median true rank |
| --- | ---: | ---: | ---: | ---: |
| Primary Thermo CV | 1700 | 0.9282 | 0.7488 | 1.0 |
| Thermo | 1611 | 0.2315 | 0.0317 | 408.0 |
| Handheld | 1609 | 0.2436 | 0.0273 | 374.0 |
| Medical | 1609 | 0.0777 | 0.0099 | 516.5 |

## Figures

- `01_device_group_then_sample_status_stacked.png`: group/sample matching 상태를 장비별 stacked bar로 표시.
- `02_device_group_vs_exact_sample_accuracy.png`: group accuracy와 exact sample accuracy를 직접 비교.
- `03_group_confusion_by_device.png`: true group vs predicted group confusion matrix.
- `04_match_status_by_true_group_and_device.png`: 각 group에서 exact/group-only/wrong-group 비율.
- `05_sample_level_match_status_grid_by_device.png`: 모든 sample을 점으로 표시. 초록은 exact sample, 노랑은 group-only, 빨강은 wrong group.
- `06_true_sample_rank_by_group_and_device.png`: true sample이 후보 중 몇 등인지. 1이면 exact match.
- `07_group_accuracy_vs_sample_accuracy_by_group.png`: group별 group accuracy와 sample accuracy 관계.

## Interpretation

현재 결과는 대부분의 장비에서 exact sample match가 거의 없고, group-only match도 제한적입니다. 즉 문제가 sample ID 수준에서만 생긴 것이 아니라, group-level transfer도 충분히 안정적이지 않습니다.
