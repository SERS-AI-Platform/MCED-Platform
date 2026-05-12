---
applyTo: "models/**/*.py,scripts/analysis/**/*.py,scripts/pipeline/**/*.py"
---

# Clinical merge — `models/clinical_utils.py` 사용 필수

Clinical 데이터와 spectral 데이터를 merge할 때 반드시 `models/clinical_utils.py`를 사용할 것.

## Why
직접 merge하면 BLC(→BLA), BRE(hospital ID), OVA(hospital ID), H.D.(공백 차이), LUN 201-300(hospital ID) 등 ID 불일치로 **400명 이상 누락**. clinical_utils.py에 이 매핑이 모두 구현됨.

## Usage

```python
from models.clinical_utils import load_clinical, merge_clinical_features

clin = load_clinical(PROJECT_ROOT)
clinical_cols = ["age", "sex_numeric", "bmi"]
clinical_data = merge_clinical_features(df_spec, clin, clinical_cols, PROJECT_ROOT)
```

## Key functions
- `load_clinical()`: `all_clinical_standardized.csv` 로드 + `sex_numeric` 컬럼 추가
- `build_id_mappings()`: BRE/OVA(Excel SoluM Label), BLC(stage→순번), H.D.(공백) 매핑
- `merge_clinical_features()`: 3단계 매칭 (직접 → ID 매핑 → alias fallback) + BMI imputation

## Path
`/home/user/SERS-AI/models/clinical_utils.py`

## 안티패턴
- ❌ `pd.merge(spec_df, clin_df, on='id')` 직접 merge
- ❌ 별도 ID 매핑 함수 새로 작성
- ✅ `merge_clinical_features()` 호출
