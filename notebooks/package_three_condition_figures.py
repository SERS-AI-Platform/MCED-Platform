from pathlib import Path
import json, hashlib, platform, zipfile
import pandas as pd
import numpy as np
import sklearn, scipy, matplotlib, xgboost
from PIL import Image

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'three_condition_figures_20260828'
historical={
 'direct':{'binary':[.6701,.6523], 'three_class':[.5459,.3979,.3627]},
 'raw':{'binary':[.7163,.6711], 'three_class':[.6595,.4406,.3980]},
 'legacy':{'binary':[.7405,.6621], 'three_class':[.6053,.4429,.4165]},
}
metrics=pd.read_csv(OUT/'verified_metrics.csv')
audit=[]
for r in metrics.to_dict('records'):
    for i,k in enumerate(['auc','bacc']+(['f1'] if r['task']=='three_class' else [])):
        old=historical[r['condition']][r['task']][i]
        audit.append(dict(condition=r['condition'],task=r['task'],metric=k,
                          user_table=old,recomputed=r[k],difference=r[k]-old,
                          matches_4dp=round(r[k],4)==old))
pd.DataFrame(audit).to_csv(OUT/'historical_vs_recomputed.csv',index=False)
sources=[ROOT/'regenerate_three_condition_figures.py',ROOT/'plot_three_condition_figures.py',
 ROOT/'train_stkv2_stacking.py',ROOT/'stk_v2_preprocess.py',ROOT/'build_stkv2_dataset.py',
 ROOT/'aecd_api_model_3class_engine.py',ROOT/'aecd_api_model_patent_dwt_clinical_performance.py',
 ROOT.parent/'scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py',
 ROOT/'aecd_api_model_mean_spectrum_outputs/mean_representative_spectra.csv']
provenance={'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__,
 'scipy':scipy.__version__,'matplotlib':matplotlib.__version__,'xgboost':xgboost.__version__,
 'sources':[{ 'file':str(p.relative_to(ROOT.parent)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources]}
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
readme='''# 세 조건 × 네 종류 그림

## 무엇을 만들었는가

총 12개 figure이며 각각 PNG(3200×1800), SVG, PDF로 제공됩니다.
각 조건 폴더에는 다음 네 figure가 있습니다.

1. `01_three_class_performance`: 3군 혼동행렬 / OvR ROC / Macro AUC·BAcc·Macro-F1
2. `02_binary_performance`: 2군 혼동행렬 / ROC / AUC·BAcc·민감도·특이도
3. `03_binary_peaks`: Non-cancer / Cancer 평균 스펙트럼과 탐색적 peak
4. `04_three_class_peaks`: Control / PDC / Cancer 평균 스펙트럼과 탐색적 peak

## 명칭과 해석

그림의 표시명은 요청에 따라 다음과 같이 변경했습니다. 내부 condition 키와 데이터는 유지했습니다.
- `direct` → **최소 전처리 후 진행**
- `raw` → **Peak 형태 보존 및 Denoising, baseline correction 방식**
- `legacy` → **기존 방식**

두 번째 표시명은 STK-V2의 smoothing·rolling-minimum baseline 보정·SNV 처리를 설명하는 이름이며, peak 보존 효과가 별도로 입증되었거나 CDAE를 구현했다는 뜻은 아닙니다.

- `direct`: Current direct mean. 저장된 PS 정렬·반복측정 QC 후 피험자 평균 CSV를 재평가. 추가 스펙트럼 전처리는 없지만 정렬·QC·평균은 수행된 조건입니다.
- `raw`: DWT raw 비교군. reference-peak 정렬 후 전체 반복측정 평균에 STK-V2를 적용. Han CDAE/Lorentzian 방법의 구현 결과가 아닙니다.
- `legacy`: 933-point 보간·피험자 평균 후 현재 저장된 STK-V2 코드로 재구성. Liu/Lin 방법의 구현 결과로 표기하지 않습니다.
- Xue의 성분별 필터링·saliency 실험은 이번 세 조건에 포함되지 않습니다.
- STK-V2의 raw/ch0는 완전한 무전처리 스펙트럼이 아닙니다. smoothing, rolling-minimum baseline, SNV를 거친 채널입니다. Raw/Legacy peak 그림은 실제 모델의 ch0를 표시합니다.
- Threshold만 바꾸면 AUC는 변하지 않습니다. 과거 AUC와 재계산 AUC가 다르면 단순 threshold 차이로 설명할 수 없습니다.

## 과거 표와 이번 그림의 관계

이 그림은 과거 요약값에서 ROC를 추정한 것이 아닙니다. 실제 피험자 단위 OOF 예측을 새로 생성하고 그 값에서 모든 성능을 계산했습니다. 과거 표와 불일치하는 값은 수정·강제하지 않았습니다.
`historical_vs_recomputed.csv`는 사용자 표와 이번 그림의 차이를 보여줍니다. `verified_metrics.csv`가 그림의 수치 기준입니다.
과거 실행 환경·입력·코드·분할이 완전히 보존된 재현이라고 주장하지 않습니다. 현재 저장 코드와 현재 API/저장 CSV로 재평가한 결과입니다. 기존 결과 파일은 덮어쓰지 않았습니다.

## 평가 및 peak 정의

- 113명: Control 21, PDC 49, Cancer 43. 첨부 예시의 109명과 다른 코호트입니다.
- 5×5 nested GroupKFold. 한 피험자는 한 행이며 반복측정은 fold를 넘지 않습니다.
- Raw/Legacy 2군 threshold=0.4. Direct 2군 threshold는 outer training data 내에서 선택. 3군 판정은 argmax입니다.
- 3군 AUC는 pooled OOF macro one-vs-rest입니다. fold AUC 평균과 구별합니다.
- Legacy 2군은 원래 subject-key 정렬과 동일한 그룹 순서를 사용하고 3군은 기존 engine의 행 인덱스 그룹을 사용합니다. 따라서 모든 조건이 동일한 fold라는 주장은 하지 않습니다.
- 평균 스펙트럼은 피험자별 스펙트럼을 동일 가중치로 평균했습니다. 음영은 피험자 간 ±1 SEM이며 측정 noise 또는 신뢰구간이 아닙니다.
- Peak 후보는 pooled mean의 국소 최대값에서 최소 간격 20 cm⁻¹, prominence=전체 진폭 범위의 3%로 검출했습니다.
- 강조는 후보 중 집단 평균 간 분산 / 평균 집단 내 분산이 큰 최대 5개 위치(±5 cm⁻¹)입니다. 전체 코호트에서 탐색적으로 선택했으며 검증된 판별 peak, 모델 중요도, saliency, 분자 동정 또는 통계적 유의성을 의미하지 않습니다.
- Hospital confounding 가능성이 있습니다. 외부 병원 일반화·임상 검증 결과가 아닙니다.

## 파일 및 재현

공유 ZIP에는 figure, 집계 스펙트럼/peak CSV, 혼동행렬, 검증 지표와 provenance만 포함합니다. 피험자별 OOF와 입력 배열은 ZIP에서 제외했습니다.
재평가 스크립트: `../regenerate_three_condition_figures.py`
그림 생성 스크립트: `../plot_three_condition_figures.py`
각 폴더의 `predictions.npz`와 `oof_predictions.csv`는 작업 폴더 내부 재현용입니다.
'''
(OUT/'README.md').write_text(readme,encoding='utf-8')
images=[]
for key in ['direct','raw','legacy']:
    files=sorted((OUT/key).glob('0*.png'))
    assert len(files)==4,(key,len(files))
    for p in files:
        im=Image.open(p);assert im.size==(3200,1800),p
        images.append((key,p,im.copy()))
# Compact overview: 3 rows x 4 columns, each 16:9 thumbnail.
contact=Image.new('RGB',(1600,675),'white')
for i,(_,p,im) in enumerate(images):
    im.thumbnail((400,225));contact.paste(im,((i%4)*400,(i//4)*225))
contact.save(OUT/'overview.png')
with zipfile.ZipFile(OUT/'three_conditions_12_figures.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob('*'):
        if not p.is_file():continue
        if p.suffix not in ['.png','.svg','.pdf','.csv','.json','.md']:continue
        if p.name in ['oof_predictions.csv']:continue
        if any(x.startswith('_') for x in p.relative_to(OUT).parts):continue
        z.write(p,p.relative_to(OUT))
print('Packaged 12 figures in PNG/SVG/PDF; each PNG is 3200x1800.')
