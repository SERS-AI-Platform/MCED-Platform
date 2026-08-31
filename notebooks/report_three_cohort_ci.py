from pathlib import Path
import json,hashlib,zipfile,shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/three_cohort_spectra_ci_20260828_v1'
D=dict(np.load(OUT/'subject_spectra.npz'));x=D['grid']
F=pd.read_csv(OUT/'regional_difference_tests.csv');T=pd.read_csv(OUT/'pointwise_difference_tests.csv');S=pd.read_csv(OUT/'mean_spectra_95ci.csv');P=pd.read_csv(OUT/'primary_raw_mean_tests.csv');G=pd.read_csv(OUT/'global_spectrum_tests.csv')
keys=['retrospective','liquid','powder'];labels=['Retrospective specimens (n=91)','Boramae liquid reducing agent (n=41)','Boramae powder reducing agent (n=43)'];colors=['#4D4D4D','#D95F02','#2C7FB8'];styles=['-','--','-.']
audit=[]
for r in P.itertuples():
 akey,bkey=('liquid','retrospective') if r.comparison.startswith('liquid') else ('powder','liquid' if r.paired else 'retrospective')
 a=D[akey+'_raw'];b=D[bkey+'_raw']
 if r.paired:a=a[[np.where(D['powder_codes']==k)[0][0] for k in D['liquid_codes']]]
 a=np.trapezoid(a,x=x,axis=1)/(x[-1]-x[0]);b=np.trapezoid(b,x=x,axis=1)/(x[-1]-x[0])
 if r.paired:
  d=a-b;tv=d.mean()/(d.std(ddof=1)/np.sqrt(len(d)));df=len(d)-1
 else:
  va=a.var(ddof=1)/len(a);vb=b.var(ddof=1)/len(b);tv=(a.mean()-b.mean())/np.sqrt(va+vb);df=(va+vb)**2/(va**2/(len(a)-1)+vb**2/(len(b)-1))
 pv=2*stats.t.sf(abs(tv),df)
 np.testing.assert_allclose([tv,pv],[r.t,r.p],rtol=1e-10)
 audit.append(dict(comparison=r.comparison,manual_t=tv,manual_df=df,manual_p=pv,passed=True))
pd.DataFrame(audit).to_csv(OUT/'independent_primary_validation.csv',index=False)
for mode in ['raw','processed']:
 fig,ax=plt.subplots(figsize=(16,6));fig.subplots_adjust(left=.075,right=.985,top=.87,bottom=.20)
 for k,label,c,ls in zip(keys,labels,colors,styles):
  s=S[(S.cohort==k)&(S['mode']==mode)];ax.fill_between(x,s.ci_low,s.ci_high,color=c,alpha=.25,linewidth=0);ax.plot(x,s['mean'],color=c,ls=ls,lw=2,label=label)
 ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(frameon=False,fontsize=11,loc='upper right');ax.set_xlabel('Raman shift (cm$^{-1}$)',fontsize=12);ax.set_ylabel('Raw intensity (a.u.)' if mode=='raw' else 'SNV intensity',fontsize=12)
 ax.set_title(('Raw' if mode=='raw' else 'Processed')+' prostate cancer spectra: mean and pointwise 95% bootstrap CI',loc='left',fontsize=17)
 fig.text(.075,.045,'10,000 subject-bootstrap resamples; repeated measurements averaged within subject.\nNot simultaneous confidence bands; acquisition/batch and hospital differences remain confounded.',fontsize=10)
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{mode}_mean_95ci.{ext}',dpi=180)
 plt.close(fig)
names={'liquid_minus_retrospective':'액상 − 후향','powder_minus_retrospective':'분말 − 후향','powder_minus_liquid_paired':'분말 − 액상 (공통 코드 41개)'}
table='| 비교 | 평균 차이 (a.u.) | 95% CI | Holm 보정 p |\n|---|---:|---:|---:|\n'
for r in P.itertuples():table+=f'| {names[r.comparison]} | {r.difference:.2f} | [{r.ci_low:.2f}, {r.ci_high:.2f}] | {r.holm_p_3:.3g} |\n'
bands='| 구간 중심 (±10 cm⁻¹) | 분말 − 액상 (SNV) | 95% CI | BH-FDR q |\n|---|---:|---:|---:|\n'
for r in F[(F.comparison=='powder_minus_liquid_paired')&(F['mode']=='processed')&F.region.str.startswith('band_')].itertuples():bands+=f'| {r.region.split("_")[1]} | {r.difference:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.fdr_q_all_regions:.3g} |\n'
text='''# 전립선암 3개 코호트 스펙트럼: 95% CI 및 그룹 차이 분석

## 요약
- 원본 그림과 같은 피험자별 배열을 재구성했다: 후향 91명, 액상 41명, 분말 43명. 모두 전립선암 군이다.
- 원본 집계 코드는 세 그룹 모두 `mean ± 1.96 SEM` CI를 그렸다. 낮은 강도의 후향/액상 CI가 공통 축에서 좁게 보이는 문제를 확인했다.
- 이번에는 피험자 단위 bootstrap 10,000회로 95% percentile CI를 다시 산출하고, 후향/액상 확대 패널과 전처리 후 CI를 함께 제공했다.
- Raw intensity 차이는 통계적으로 크며, 전처리 후에도 여러 고정 파수 구간에서 차이가 남는다. Raw 강도 증가를 피크 품질, SNR 또는 진단성능 개선으로 해석하지 않는다.

## 원본과 집계
원본 코드: `publications/전향검체/보라매병원/src/three_cohort_spectra.py`.

- 후향: clean_cohort_20260605 manifest의 PRO 검체 91개, 각 5회.
- 액상: 보라매 임상표의 제외 기준을 적용한 prostate 41개, 20260709 원천의 각 5회.
- 분말: data/mapping/clinical_df.xlsx의 prostate 43개, 각 121회.
- `_ave` 파일은 제외. 공통 grid는 402–2198 cm⁻¹, 935 point.
- Raw는 trim 및 공통-grid 보간 후 검체 내 반복 평균이다. 새 baseline/QC/PS alignment는 추가하지 않았다.
- Processed는 원본 정의대로 각 반복 spectrum에 SG(window 11, polyorder 3) → baseline correction(window 101) → SNV → 보간을 적용한 뒤 검체 내 평균이다. 평균 후 전처리와 혼동하지 않는다.

## 1차 검정: 전체 파수 범위 평균 raw intensity
각 검체에서 `적분 intensity / 파수 폭`을 계산했다. 아래 CI는 평균 차이의 bootstrap CI이며, p는 양측 t 검정에 3개 비교 Holm 보정을 적용했다.

'''+table+'''
그룹별 평균값: 후향 111.744 a.u., 액상 48.998 a.u., 분말 전체 43개 1796.893 a.u. 대응 분석의 분말 41개 평균은 1812.539 a.u.로 전체 43개 평균과 구분한다.

## 전처리 후 고정 구간 차이: 분말 − 액상
각 구간의 적분을 구간 폭으로 나눈 평균 SNV intensity이다. 값은 개별 피크 높이나 local-baseline 보정 피크 면적이 아니다.

'''+bands+'''
분말–액상 1800–2000 cm⁻¹ 구간은 raw에서 크게 다르지만, 전처리 후 평균 차이 0.013 SNV, 95% CI [-0.002, 0.030], q≈0.144로 유의하지 않았다. 반면 1000/1445 구간은 낮고 1650 구간은 높은 차이가 남았다. 전처리로 전체 offset이 줄어도 상대적인 spectral profile 차이가 남는다는 해석과 부합한다. 원인 또는 분자 성분을 확정하는 분석은 아니다.

## 통계 설계
- 평균 CI: 그룹별 검체 bootstrap 10,000회, seed 20260828; 독립 단위는 spectrum row가 아니라 검체 평균이다.
- 차이 CI: 후향 비교는 두 그룹 독립 bootstrap; 액상–분말은 공통 검체코드 41개의 차이를 pair 단위로 bootstrap. 추가 분말 2개는 대응 비교에서 제외되지만 43개 평균 그림과 후향 비교에는 포함된다.
- 후향 비교는 Welch t, 액상–분말은 paired t. 1차 전체 raw 평균 3개 비교는 Holm 보정.
- 파수별 raw/processed × 3비교 × 935점 = 5,610개 검정에 전체 BH-FDR 적용. 인접 파수의 의존성이 있으므로 유의점 수는 독립적인 발견 수가 아니다. BH 결과는 탐색적으로 제시한다.
- 전체 곡선의 동일 평균 귀무가설은 각 비교에서 935개 p의 최소값×935 Bonferroni로 보수적으로 검정하고, 6개 비교에 Holm 보정했다. 이는 적어도 한 파수의 차이를 검정하며 모든 파수가 다르다는 뜻이 아니다. global_spectrum_tests.csv 참조.
- 구간 분석은 raw 8개 / processed 7개 × 3비교 = 45개 검정에 BH-FDR 적용. 구간은 앞서 논의한 745/935/1000/1445/1650/2100 ±10 cm⁻¹ 및 1800–2000, raw 전체 범위다. 데이터와 앞선 그림을 보고 정한 탐색 구간이며 사전등록된 확증 분석이 아니다.
- 보조 순위검정: 후향 비교 Mann–Whitney, 대응 비교 Wilcoxon. 45개 보조 검정에도 별도 BH 보정. primary/regional CSV에서 확인할 수 있다.
- 그림의 95% CI는 pointwise이다. 곡선 전체를 동시에 덮는 95% band나 다중비교 보정 CI가 아니다. 검정과 CI의 보정 기준이 다를 수 있다.

## 대응 관계와 해석의 제한
1. 액상–분말은 동일 solum_label 41개를 연결했고 성별과 cancer_type이 41/41 일치했다. 수집일은 비교 가능한 비결측 쌍이 없었다. 따라서 대응 결과는 공유 검체코드가 같은 생물학적 검체를 뜻한다는 전제하에 해석한다. 독립적인 환자/채취시점 확인을 대신하지 않는다.
2. 후향과 보라매는 독립 코호트로 취급했다. 병원·보관·취득조건·측정일·장비·환원제 lot 등 공변량을 조정하지 않았다. 환원제의 단독 인과효과는 추정하지 않았다.
3. 반복 수가 5/5/121로 다르다. 피험자별 동일 가중치로 평균했지만 측정오차의 차이는 남는다. 이번 CI는 새 측정/새 lot까지 포함하는 계층적 불확실성이 아니다.
4. 고정 파수에서의 차이는 peak 위치 이동, 폭, baseline 및 상대 강도 변화를 함께 반영한다. 특히 745 구간의 감소가 인접한 powder peak 자체의 높이 감소를 의미하지 않는다. Peak tracking/정렬 기반 분석은 별도다.
5. 입력의 Raw라는 명칭은 원본 코드의 보간 전 신호를 뜻한다. 음수 값이 존재하므로 장비 내부 처리나 취득 전 background 처리 여부를 추정하지 않는다.
6. 임상 진단성능 비교가 아니라 **암 검체 내 코호트별 스펙트럼 분포 비교**다. AUC 저하 원인을 직접 입증하지 않는다.

## 검증 및 산출물
- independent_primary_validation.csv: 직접 계산한 t 통계량/자유도/t 분포 p와 원 분석의 일치 확인.
- linkage_validation.json: 식별자를 공개하지 않은 코드 연결/임상필드 일치 개수.
- validation.json, source_hashes.json: 배열/CI/검정 수 검사와 소스 해시.
- three_cohort_mean_95ci: 전체 raw / 낮은 강도 확대 / processed 3패널.
- raw_mean_95ci, processed_mean_95ci: 개별 그림.
- cohort_differences_95ci: 평균 차이와 pointwise CI, FDR 유의점.
- 모든 그림 PNG/SVG/PDF. CSV는 실제 계산값이며 이미지에서 읽은 값이 아니다.
- 공유 ZIP은 aggregate 파일만 포함. subject_spectra.npz는 연구용 개별자료로 제외했다.
'''
(OUT/'README.md').write_text(text,encoding='utf-8')
files=[ROOT/'notebooks/audit_three_cohort_ci.py',ROOT/'notebooks/analyze_three_cohort_ci.py',Path(__file__),ROOT/'publications/전향검체/보라매병원/src/three_cohort_spectra.py',ROOT/'artifacts/usersnet/v1.0.0/common_grid.npy',ROOT/'data/mapping/clinical_df.xlsx',ROOT/'data/clinical_data/보라매 병원 임상정보.xlsx',ROOT/'results/clean_cohort_20260605/clean_cohort_manifest.csv']
(OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},ensure_ascii=False,indent=2))
for p in files[:3]:shutil.copy2(p,OUT/p.name)
with zipfile.ZipFile(OUT/'three_cohort_ci_statistics.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():
  if p.suffix in ['.csv','.json','.md','.png','.svg','.pdf','.py']:z.write(p,p.name)
print('Independent manual tests verified; report and aggregate ZIP saved')
