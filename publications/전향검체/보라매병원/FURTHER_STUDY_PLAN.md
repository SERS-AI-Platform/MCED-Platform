# 보라매병원 전향검체 Further Study 정리

## 현재 위치

| 항목 | 현재 분석 기준 | 원래 목표/확장 기준 | 해석 |
|---|---:|---:|---|
| Non-cancer | 68명: Control 21명 + Biopsy-negative 47명 | Control 약 150명 | 현재 non-cancer 축은 생검 음성군이 많이 포함되어 있어 단순 healthy control과 다르다. |
| Cancer | Prostate cancer 41명 | Cancer 약 300명 | 현재 성능과 peak는 exploratory이며, full cohort에서 재검증해야 한다. |
| Screening 성능 | ROC-AUC 0.714, balanced accuracy 0.704 | Locked model + prospective validation | Cancer Screening AUC는 현재 표본 수가 작고 hospital/batch confounding 가능성이 있어 외부 일반화 근거로 쓰면 안 된다. |
| 주요 differential peak regions | 1000, 1447, 1650, 935, 744 cm^-1 등 | Full cohort에서 재현성 확인 | 1000 cm^-1 부근은 urea-like band 후보지만, 현재는 분자 동정이 아니라 spectral association으로 표기해야 한다. |
| Grade 방향성 | 1000 cm^-1 urea-like band는 GG1-2보다 GG3-5에서 낮은 경향 | Grade/PSA/임상정보 보정 모델 | 단조 감소로 단정하면 안 되며, GG5 소표본 재상승 가능성을 포함해 검증해야 한다. |

## 1. Raman Calibration 및 Peak Alignment 계획

### 1.1 용어 정리

| 용어 | 쉬운 설명 | 왜 문제가 되는가 | Ramcheck A1로 확인할 것 |
|---|---|---|---|
| CCD pixel spacing | 검출기 CCD의 픽셀 간 간격이다. Raman spectrum은 빛의 위치를 CCD 픽셀에서 읽고, 그 픽셀 번호를 cm^-1 값으로 바꾼다. | 픽셀-파수 변환식이 조금만 틀어져도 실제 같은 peak가 998, 1001 cm^-1처럼 다르게 보일 수 있다. | 같은 reference peak가 항상 같은 pixel/cm^-1 위치에 찍히는지, run마다 일정하게 밀리는지 확인한다. |
| Grating dispersion | 회절격자 grating이 파장별 빛을 얼마나 벌려서 CCD에 보내는지에 대한 값이다. | 한쪽 구간은 거의 맞는데 다른 구간은 더 밀리는 비선형 shift가 생길 수 있다. | Ramcheck A1의 여러 reference peak를 써서 전 구간에서 shift가 일정한지, 구간별로 다른지 본다. |
| Slit width | spectrometer에 들어가는 빛의 입구 폭이다. 좁으면 peak가 날카롭고, 넓으면 signal은 커질 수 있지만 peak가 넓어진다. | peak가 넓어지면 옆 peak와 겹쳐 peak center가 실제보다 이동한 것처럼 보일 수 있다. | reference peak의 FWHM이 run/batch마다 얼마나 달라지는지 본다. |
| Instrumental line shape | 장비가 원래 날카로운 선을 측정했을 때 어떤 모양의 peak로 기록하는지 나타내는 장비 고유의 peak spread 함수다. | 실제 분자 peak가 아니라 장비 함수 때문에 peak 폭/비대칭/shoulder가 생길 수 있다. | reference peak 모양을 fitting해서 폭, 비대칭, tailing이 일정한지 확인한다. |
| Peak overlap | 가까운 두 peak가 서로 겹쳐 하나의 넓은 peak처럼 보이는 현상이다. | 두 peak의 상대 intensity가 바뀌면 peak 위치가 이동한 것처럼 보인다. 실제 chemical shift가 아닐 수 있다. | reference peak 주변 shoulder와 deconvolution 결과가 일정한지 확인한다. |
| Shift | 같은 peak가 cm^-1 축에서 좌우로 이동해 보이는 현상이다. | 질병 차이인지 장비 calibration 문제인지 구분이 필요하다. | 같은 reference peak의 measured position과 expected position 차이를 계산한다. |
| Broadening | peak 폭이 넓어지는 현상이다. | peak height가 낮아지고 area/center가 변하며 이웃 peak와 겹칠 수 있다. | reference peak의 FWHM과 peak area/height ratio를 추적한다. |
| Standard-based calibration | known peak 위치를 가진 표준물질을 먼저 측정하고, measured peak가 expected peak에 맞도록 cm^-1 축을 보정하는 방법이다. | sample 자체의 biological peak를 anchor로 쓰지 않아도 되어 class signal을 덜 훼손한다. | Ramcheck A1 reference peak로 pixel-to-wavenumber 변환식과 residual error를 만든다. |
| Constrained alignment | 아무렇게나 spectrum을 맞추는 것이 아니라, 허용 가능한 shift 범위와 anchor 조건을 제한해서 정렬하는 방법이다. | 과보정하면 cancer signal까지 지워질 수 있다. | Ramcheck A1에서 관찰된 shift/broadening 범위 안에서만 sample alignment를 허용한다. |

정리하면, Ramcheck A1은 sample의 biology를 보기 위한 것이 아니라 장비가 같은 reference peak를 매번 같은 위치와 폭으로 측정하는지 확인하기 위한 기준이다. 여기서 관찰된 shift/broadening 범위를 먼저 정량화한 뒤, sample spectrum에는 standard-based calibration을 우선 적용하고, 남는 작은 잔차에 대해서만 constrained alignment를 적용한다.

| 문제/변수 | 발생 가능한 현상 | 수집/기록해야 할 정보 | 알고리즘 보정 | 학습 영향 차단 방법 | 산출물 |
|---|---|---|---|---|---|
| CCD pixel spacing | wavenumber grid가 run마다 미세하게 달라짐 | 장비 ID, CCD map, acquisition date, calibration file | pixel-to-wavenumber polynomial calibration, common grid resampling | raw peak position 대신 calibrated grid 사용, acquisition batch group split | calibration residual plot, per-run grid shift table |
| Grating dispersion | spectrum 전 구간에 비선형 shift 발생 | grating setting, spectrograph config | multi-point standard peak fitting 후 piecewise/polynomial warping | calibration standard residual이 tolerance 초과하면 QC fail | wavelength calibration curve |
| Slit width / instrumental line shape | 날카로운 peak가 넓어짐, 이웃 peak overlap 시 apparent shift | slit width, resolution spec, FWHM estimate | instrument line shape 추정, FWHM-aware peak fitting, broad-band feature 사용 | peak center 단독 feature보다 band area/shape feature 병행 | peak FWHM distribution, overlap-risk peak list |
| Laser wavelength drift / temperature | 전체 spectrum의 global shift | laser wavelength log, room/instrument temperature | global shift correction + local warping | shift augmentation으로 모델 robust training | pre/post alignment mean spectra |
| Focus / sample height / substrate hotspot | intensity scale과 local enhancement 변동 | replicate-level intensity, focus metric, substrate lot | replicate QC, robust subject aggregation, SNV/area normalization | replicate-level outlier down-weighting, subject mean/median ensemble | replicate consistency report |
| Fluorescence/background | baseline 차이와 false broad peak | raw fluorescence flag, baseline shape metric | baseline correction parameter lock, fluorescence score as QC covariate | fluorescence-heavy spectrum 제외 또는 covariate adjustment | preprocessing audit figure |
| Peak overlap | 이웃 peak가 겹쳐 위치 이동처럼 보임 | candidate peak FWHM, local shoulder peaks | multi-peak deconvolution, derivative-assisted local maxima, band area feature | single-point peak feature 의존도 제한 | deconvolved peak table |

## 2. Alignment 알고리즘 후보

| 접근 | 적용 위치 | 장점 | 위험 | 권장 사용 |
|---|---|---|---|---|
| Calibration standard 기반 polynomial alignment | raw spectrum → calibrated grid | 물리적으로 가장 방어 가능 | standard run 누락 시 적용 어려움 | 최우선. 매 acquisition batch마다 standard spectrum 확보 |
| Internal anchor peak alignment | preprocessing 후 | 별도 standard 없이 가능 | disease-related peak를 anchor로 쓰면 biological signal을 지울 수 있음 | urea-like 1000 cm^-1 하나만 쓰지 말고 stable multi-anchor 후보로 제한 |
| Constrained DTW / COW | local shift correction | 비선형 shift 보정 가능 | 과보정 시 class signal을 왜곡 | standard 기반 보정 후 잔차 보정으로만 사용 |
| Peak-cluster consensus alignment | group/blind 전체 sample 기준 | peak cluster 단위 해석과 연결 쉬움 | class imbalance에 민감 | blinded full cohort에서 consensus peak library 생성 |
| Shift augmentation | model training | 장비 shift에 robust | 실제 calibration 오류를 숨길 수 있음 | alignment 후 residual shift 범위 안에서만 사용 |
| Batch/domain adversarial learning | model training | batch/instrument 정보를 덜 쓰게 함 | 표본 수가 작으면 불안정 | full 450명 이상에서 batch label이 충분할 때 사용 |

## 3. SERS Enhancement와 Urea-like Band 처리

### 3.1 Peak family의 의미

여기서 `peak family`는 특정 peak를 하나의 대사체로 바로 동정한다는 뜻이 아니다. **같은 검체 조건 변화 또는 SERS 측정 조건 변화에 대해 비슷한 반응을 보이는 peak들의 분석 단위**를 뜻한다. SERS intensity는 검체 내 분자량뿐 아니라 금속 표면과의 결합 친화도, hotspot 접근성, 기판 lot, pH와 염 농도 등에 의해 달라지므로, 서로 다른 band의 intensity를 동일한 농도 척도로 직접 비교하면 안 된다.

| Peak family 후보 | 운영상 정의 | 확인할 질문 | 분류 기준 |
|---|---|---|---|
| Urea-responsive family | 988-1012 cm^-1 중심 band와 urea spike-in/희석에 함께 반응하는 peak | 1000 cm^-1 변화가 실제 urea 농도 또는 소변 농축도의 영향인가? | urea spike 농도에 대한 dose-response, urine creatinine/specific gravity 보정 후 변화, 반복 측정 안정성 |
| Metabolite-associated family | urea spike에는 거의 반응하지 않지만 cancer/Grade 또는 다른 urine chemistry와 연관되는 peak | urea-like band와 독립적인 생물학적 정보가 있는가? | urea 보정 후 association, peak 간 공변동, 표준물질 spike-in 또는 LC-MS 연계 |
| Substrate-sensitive family | 동일 검체에서도 substrate lot, spot 또는 측정일에 따라 크게 변하는 peak | 생물학이 아니라 SERS enhancement 변동인가? | 동일 aliquot의 lot/day/spot별 CV와 mixed-effect 분산 성분 |
| Stable reference candidate | 동일 검체와 측정 조건 변화에서 상대적으로 안정적인 peak | band ratio나 normalization 기준으로 사용할 수 있는가? | 낮은 technical CV, 임상군과 무관함, batch 간 재현성 |

따라서 peak family는 최초에는 **분자 이름이 아니라 반응 패턴으로 정의**해야 한다. 이후 표준물질 spike-in, peak 위치 일치, 농도별 반응, 필요 시 LC-MS 결과가 함께 맞을 때만 urea 또는 특정 metabolite assignment의 신뢰도를 높인다. 현재 1000 cm^-1는 `urea-like` 또는 `urea-responsive candidate`로 표기하는 것이 적절하다.

| 이슈 | 현재 해석 | 분석 전략 | 모델 전략 | 검증 |
|---|---|---|---|---|
| 1000 cm^-1 urea-like band가 강함 | Cancer/PSA/grade와 관련 가능성이 있으나 농도 직접 측정은 아님 | single point, 988-1012 cm^-1 band mean/max, peak area를 모두 산출 | urea-like band 단독 모델과 전체 spectrum 모델을 분리 비교 | grade, PSA, urine specific gravity/creatinine 보정 후 유지되는지 확인 |
| 다른 band는 enhancement가 다를 수 있음 | SERS hotspot/분자 affinity 차이 가능 | peak별 replicate CV, substrate lot effect, band ratio 분석 | peak family별 normalization 또는 uncertainty weighting | substrate lot leave-out validation |
| intensity scale 변동 | SNV 후에도 상대 intensity 해석 | total area, SNV, vector norm, internal ratio 병행 | normalization ensemble로 안정성 확인 | normalization별 peak rank concordance |
| urea/creatinine/hydration 영향 | urine dilution confound 가능 | urine creatinine, specific gravity, osmolality 수집 | clinical covariate residualization 또는 multimodal input | covariate-adjusted peak effect table |

### 3.2 Enhancement 차이를 확인하는 실험 설계

| 실험 | 비교 단위 | 통제할 변수 | 알 수 있는 결론 | 권장 분석 |
|---|---|---|---|---|
| 동일 검체 기술 반복 | 한 검체를 균질화한 뒤 동일 aliquot를 여러 spot/chip에서 반복 | 검체 조성, 전처리, 주입량 | spot 간 hotspot 변동과 측정 반복성 | peak별 CV, ICC, within-sample variance |
| 동일 검체의 substrate lot/day 교차 | 같은 aliquot를 여러 substrate lot와 측정일에 배치 | 검체 생물학 | lot/day에 따라 특정 band만 더 증강되는지 | mixed-effect model: sample random effect, lot/day fixed or random effect |
| Urea spike-in 및 dilution series | 같은 urine pool에 알려진 농도의 urea를 단계적으로 첨가 | 나머지 검체 matrix | 1000 cm^-1와 함께 dose-response를 보이는 urea-responsive family | peak별 slope, 비선형성, recovery, limit of detection |
| Matrix-matched 표준물질 실험 | pooled urine 또는 synthetic urine에 후보 metabolite를 개별/혼합 첨가 | pH, 염 농도, 총 단백, 희석도 | 후보 대사체가 어느 band를 만들고 다른 분자와 경쟁 증강하는지 | spike 농도-peak area curve, 혼합물 interaction term |
| 임상 매칭 비교 | Cancer와 Biopsy-negative 또는 Control을 임상변수 기준으로 매칭 | age, PSA, urine creatinine/specific gravity, eGFR, pH, 혈뇨/염증, 보관 조건, batch | 임상적 혼란변수를 줄인 뒤에도 cancer/Grade 차이가 남는지 | propensity score 또는 exact/caliper matching 후 paired/conditional analysis |
| Cancer 내부 Grade 비교 | Grade가 다른 환자를 PSA, age, urine chemistry, tumor burden 가능 범위에서 매칭 | PSA와 소변 농축도 등 | peak가 단순 PSA/희석도가 아니라 Grade와 연관되는지 | covariate-adjusted ordinal model, matched sensitivity analysis |

### 3.3 같은 검체와 매칭 검체가 각각 필요한 이유

1. **같은 검체 실험이 먼저 필요하다.** 하나의 균질화된 urine sample을 나누어 substrate lot, spot, 측정일만 바꾸면 biological difference가 고정된다. 이때 1000 cm^-1는 안정적인데 다른 band만 크게 변한다면, 그 차이는 대사체 농도 차이보다 band별 SERS enhancement 또는 기판 선택성의 영향으로 해석할 수 있다.
2. **임상 매칭 검체는 그 다음 단계다.** 같은 검체만으로는 cancer 또는 Grade association을 판단할 수 없다. Cancer와 Biopsy-negative군을 PSA, 연령, urine creatinine/specific gravity, 신기능, 염증·혈뇨, 채뇨·보관 조건, 측정 batch가 최대한 비슷하도록 구성한 뒤 peak family 차이가 유지되는지 봐야 한다.
3. **Grade 결론에는 Cancer 내부 비교가 필요하다.** GG1-2와 GG3-5를 비교할 때 PSA와 소변 농축도가 한쪽에 치우치면 urea-like band 감소를 Grade 효과로 오인할 수 있다. 매칭 또는 회귀 보정 후에도 방향과 effect size가 유지되어야 Grade-associated signal이라고 표현할 수 있다.
4. **완전한 매칭만 고집하면 표본이 크게 줄 수 있다.** 우선 핵심 변수에 대한 exact/caliper matching 분석을 하고, 전체 Cancer cohort에서는 같은 변수를 포함한 회귀 또는 mixed-effect model을 병행한다. 두 분석의 방향이 일치할 때 결론의 신뢰도가 높아진다.

실험의 판정 순서는 `동일 검체 반복성 확인 -> urea/후보 metabolite spike-in으로 peak family 정의 -> 임상 매칭군에서 cancer/Grade association 확인 -> 전체 cohort 보정 모델에서 재현`으로 고정한다. 동일 검체 실험만으로는 생물학적 연관성을, 임상 매칭 비교만으로는 band별 SERS enhancement 원인을 확정할 수 없으므로 두 종류의 실험이 모두 필요하다.

## 4. 건강검진 소변검사로 들어갈 때 필요한 검체/임상 파라미터

| 파라미터 그룹 | 필수 항목 | 목적 | 모델 사용 방식 |
|---|---|---|---|
| 채뇨 조건 | 채뇨 시간, first void/midstream, 금식 여부, 채뇨-측정 시간, 보관 온도, freeze-thaw 횟수 | pre-analytical variation 통제 | QC covariate, batch effect model |
| 기본 urine chemistry | specific gravity, pH, urine creatinine, osmolality | dilution/acid-base 보정 | spectrum feature residualization, normalization stratification |
| 일반 소변검사 | protein/albumin, glucose, ketone, bilirubin/urobilinogen, blood/RBC, WBC/leukocyte esterase, nitrite | hematuria/UTI/renal-metabolic confound 확인 | exclusion flag 또는 covariate adjustment |
| 혈액/신장 기능 | serum creatinine, eGFR, diabetes status | urea/creatinine 관련 confound | urea-like band 해석 보정 |
| 전립선 임상정보 | PSA, free PSA, PSA density, prostate volume, PI-RADS, biopsy result | Biopsy-negative군과 Cancer의 혼동 해소 | multimodal model, stratified evaluation |
| 암 병리정보 | Gleason score, ISUP Grade Group, tumor burden, stage, treatment history | aggressiveness signal 검증 | multi-task learning: cancer + grade + risk group |
| 염증/비암성 전립선 질환 | BPH, prostatitis, UTI, medication | PSA 상승 false positive 분리 | separate biopsy-negative phenotype model |

## 5. PSA와 Grade Confounding을 AI적으로 보정하는 분석 계획

임상정보와 OOF 분류 결과를 연결하는 필드 정의 및 공개·내부 식별자 경계는 [Clinical Classification Analysis Schema](CLINICAL_CLASSIFICATION_SCHEMA.md)에 고정한다. 이 schema는 먼저 spectrum-only 모델의 오분류 원인을 평가하기 위한 것이며, 임상정보를 모델 입력에 포함하는 multimodal 모델은 별도 분석으로 비교한다.

| 질문 | 분석 설계 | AI/통계 방법 | 성공 기준 |
|---|---|---|---|
| PSA 때문에 cancer/control이 섞이나? | Control, Biopsy-negative, Cancer 3그룹을 기본 task로 유지 | 3-class classifier + one-vs-rest AUC + confusion matrix | Cancer가 Biopsy-negative군과 구분되는 spectrum signal 확인 |
| peak가 PSA 자체를 반영하나? | 각 peak intensity를 PSA 연속값과 비교 | Spearman/partial correlation, spline regression | PSA 보정 후에도 cancer association 유지 |
| peak가 Grade를 반영하나? | Cancer 내부에서 GG1-2 vs GG3-5, GG별 trend 분석 | ordinal regression, multi-task model, monotonic trend test | grade-associated peak가 screening peak와 겹치는지 확인 |
| 임상정보를 넣으면 spectrum signal이 사라지나? | spectrum-only vs clinical-only vs spectrum+clinical 비교 | nested OOF, likelihood/decision-curve comparison | spectrum 추가 시 calibration, AUC, sensitivity 개선 |
| 장비/batch가 성능을 만든 것인가? | acquisition date/device/substrate lot leave-out | mixed effect, ComBat/RUV, domain adversarial, leave-batch-out CV | batch-out 성능이 random CV와 크게 무너지지 않음 |
| 특정 peak가 모델을 과도하게 지배하나? | peak ablation, band masking, shift perturbation | permutation importance, occlusion, stability selection | top peak 제거 후 성능 변화와 대체 band 확인 |

## 6. Further Study 단계별 실행안

| 단계 | 목표 | 실행 | Go/No-go 기준 |
|---|---|---|---|
| Phase 0: 현재 109명 재분석 고정 | 분석 파이프라인 lock | Fig01-06, peak table, grade table, calibration QC template 확정 | 같은 입력에서 같은 산출물 재현 |
| Phase 1: full Boramae cohort 확장 | 300 cancer / 150 control 목표 반영 | 동일 preprocessing + QC rule로 재생성, excluded reason 기록 | group별 n, Biopsy-negative, grade 분포 확보 |
| Phase 2: calibration study | Raman 기기 변동 정량화 | standard spectrum batch마다 측정, FWHM/shift/warping residual 산출 | shift residual이 predefined tolerance 이내 |
| Phase 3: urine parameter study | dilution/UTI/renal confound 분리 | urine chemistry + clinical metadata 결합 | urea-like band의 adjusted effect 확인 |
| Phase 4: model lock | 실사용 후보 모델 확정 | spectrum-only, clinical-only, multimodal, batch-robust model 비교 | nested OOF + leave-batch-out 모두 통과 |
| Phase 5: external/prospective validation | 건강검진 소변검사 적용 가능성 평가 | locked model로 신규 검체 blind prediction | sensitivity/specificity, calibration, decision curve 사전 기준 충족 |

## 7. 문장화 가능한 핵심 메시지

| 주제 | 논문/계획서 표현 |
|---|---|
| Calibration | Raman peak shift는 단순 biological shift가 아니라 CCD pixel mapping, grating dispersion, slit width, instrumental line shape, local peak overlap의 영향을 받을 수 있으므로 standard-based calibration과 constrained alignment를 병행한다. |
| SERS enhancement | SERS intensity는 molecule abundance와 enhancement efficiency가 섞인 signal이므로, urea-like band와 다른 metabolite-associated bands를 동일한 농도 척도로 해석하지 않고 peak family별 안정성과 covariate-adjusted association을 검증한다. |
| PSA/Grade | Biopsy-negative군을 별도 축으로 유지하고, PSA/Grade/urine chemistry를 포함한 multimodal 및 covariate-adjusted 분석으로 cancer-associated signal과 PSA/grade-associated signal을 분리한다. |
| 현재 결과의 위치 | 현재 109명 결과는 exploratory signal discovery이며, full Boramae cohort와 calibration-controlled prospective validation에서 재현성을 확인해야 한다. |
