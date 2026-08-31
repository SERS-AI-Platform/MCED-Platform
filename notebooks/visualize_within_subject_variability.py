from pathlib import Path
import json,hashlib,zipfile,shutil
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/within_subject_variability_20260828_v1'
R=pd.read_csv(OUT/'subject_variability.csv');S=pd.read_csv(OUT/'group_variability_summary.csv');T=pd.read_csv(OUT/'primary_comparisons.csv');C=pd.read_csv(OUT/'subject_band_cv.csv');BT=pd.read_csv(OUT/'band_cv_comparisons.csv');W=pd.read_csv(OUT/'within_sd_spectrum.csv');SS=pd.read_csv(OUT/'single_subset_paired_sensitivity.csv')
P=json.loads((OUT/'protocol.json').read_text());D=dict(np.load(OUT/'_replicate_inputs.npz'))
keys=['retrospective','liquid','powder'];names=['후향 검체','액상 환원제','분말 환원제'];colors=['#4D4D4D','#D95F02','#2C7FB8'];metrics=P['primary_metrics'];centers=P['centers']
titles=['전체 보정 면적의 CV','상대 피크 면적의 CV','스펙트럼 형태의 반복 SD']
units=['CV (%)','CV (%)','정규화 RMS SD']
for p in Path('/home/user/.fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'NanumSquare','font.size':13,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})

def save(fig,name):
 for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180)
 plt.close(fig)

# Independent checks of the subject-level SD formula, CV, and paired test.
check=[]
for key in keys[:2]:
 shape=D[key+'_shape'];area=D[key+'_area'];f=D[key+'_fraction']
 cv=100*np.sqrt(np.sum((area-area.mean(1,keepdims=True))**2,axis=1)/4)/area.mean(1)
 rms=np.sqrt(np.sum((shape-shape.mean(1,keepdims=True))**2,axis=(1,2))/(4*shape.shape[2]))
 bcv=np.median(100*np.sqrt(np.sum((f-f.mean(1,keepdims=True))**2,axis=1)/4)/f.mean(1),axis=1)
 for metric,v in zip(metrics,[cv,bcv,rms]):
  ref=R[(R.cohort==key)&(R.metric==metric)].value.to_numpy();np.testing.assert_allclose(v,ref,rtol=1e-11,atol=1e-12);check.append(dict(check='manual_formula',cohort=key,metric=metric,passed=True))
ids=np.array([np.where(D['powder_codes']==c)[0][0] for c in D['liquid_codes']])
for metric in metrics:
 a=R[(R.cohort=='powder')&(R.metric==metric)].value.to_numpy()[ids];b=R[(R.cohort=='liquid')&(R.metric==metric)].value.to_numpy();delta=a-b
 ref=T[(T.comparison=='powder_minus_liquid_paired')&(T.metric==metric)].iloc[0]
 ranks=stats.rankdata(abs(delta));stat=min(ranks[delta>0].sum(),ranks[delta<0].sum());np.testing.assert_allclose(stat,ref.statistic)
 check.append(dict(check='manual_signed_rank_statistic',cohort='paired',metric=metric,passed=True))
 draws=np.load(OUT/'_powder_subset_metrics.npz')[metric]
 np.testing.assert_allclose(draws.mean(1),R[(R.cohort=='powder')&(R.metric==metric)].value,rtol=1e-12)
assert len(R)==175*4 and C.eligible_fraction.min()==1
idx=np.random.default_rng(np.random.SeedSequence([P['seed'],0])).choice(121,5,replace=False)
area=D['powder_area'][0,idx];f=D['powder_fraction'][0,idx];shape=D['powder_shape'][0,idx]
manual=[100*np.std(area,ddof=1)/np.mean(area),np.median(100*np.std(f,axis=0,ddof=1)/np.mean(f,axis=0)),np.sqrt(np.sum((shape-shape.mean(0))**2)/(4*shape.shape[1]))]
for metric,v in zip(metrics,manual):
 np.testing.assert_allclose(v,np.load(OUT/'_powder_subset_metrics.npz')[metric][0,0],rtol=1e-11)
 check.append(dict(check='reconstructed_first_powder_subset',cohort='powder',metric=metric,passed=True))
pd.DataFrame(check).to_csv(OUT/'independent_validation.csv',index=False)

fig,axes=plt.subplots(1,3,figsize=(16,9));fig.subplots_adjust(left=.065,right=.985,top=.75,bottom=.24,wspace=.29)
rng=np.random.default_rng(34)
for ax,metric,title,unit in zip(axes,metrics,titles,units):
 vals=[R[(R.cohort==k)&(R.metric==metric)].value.to_numpy() for k in keys]
 boxes=ax.boxplot(vals,positions=[0,1,2],widths=.48,patch_artist=True,showfliers=False,medianprops={'color':'black','linewidth':2})
 for i,(v,c) in enumerate(zip(vals,colors)):
  boxes['boxes'][i].set(facecolor=c,alpha=.22);ax.scatter(i+rng.uniform(-.19,.19,len(v)),v,s=16,color=c,alpha=.65,edgecolor='none')
  s=S[(S.cohort==keys[i])&(S.metric==metric)].iloc[0];fmt='.3f' if metric=='shape_rms_sd' else '.2f'
  ax.text(i,1.05,f"{s['median']:{fmt}}\n[{s.median_ci_low:{fmt}}, {s.median_ci_high:{fmt}}]",ha='center',va='bottom',transform=ax.get_xaxis_transform(),fontsize=12,color=c,weight='bold')
 ax.set_xticks([0,1,2],[f'{n}\n(n={len(v)})' for n,v in zip(names,vals)]);ax.set_title(title,fontsize=16,pad=74);ax.set_ylabel(unit);ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18)
fig.suptitle('동일 검체 내 반복측정 변동성 비교',x=.065,ha='left',y=.97,fontsize=25,weight='bold')
fig.text(.065,.915,'모든 조건 5회 기준 · 낮을수록 안정적 · 숫자: 검체별 지표의 중앙값 [95% bootstrap CI]',fontsize=15)
fig.text(.065,.145,'점 1개 = 검체 1개. 상자 = IQR, 가로선 = 중앙값, 수염 = 1.5×IQR 범위.',fontsize=13)
fig.text(.065,.11,'분말: 121개 중 5개 비복원 추출 × 500회 → 검체별 지표 평균. 추출 500회를 독립 표본으로 세지 않음.',fontsize=13)
fig.text(.065,.07,'측정 위치·배치·기관 차이를 포함한 관측 재현성이다. 환원제만의 효과나 진단 AUC를 검정한 결과가 아님.',fontsize=12,color='#555555')
save(fig,'01_variability_distributions')

fig,axes=plt.subplots(1,3,figsize=(16,9));fig.subplots_adjust(left=.07,right=.98,top=.79,bottom=.22,wspace=.30)
for ax,metric,title,unit in zip(axes,metrics,titles,units):
 a=R[(R.cohort=='liquid')&(R.metric==metric)].value.to_numpy();b=R[(R.cohort=='powder')&(R.metric==metric)].value.to_numpy()[ids]
 for av,bv in zip(a,b):ax.plot([0,1],[av,bv],color='#9CA3AF',alpha=.5,lw=.8)
 ax.scatter(np.zeros(len(a)),a,color=colors[1],s=23);ax.scatter(np.ones(len(b)),b,color=colors[2],s=23)
 t=T[(T.metric==metric)&(T.comparison=='powder_minus_liquid_paired')].iloc[0];fmt='.3f' if metric=='shape_rms_sd' else '.2f';suffix='' if metric=='shape_rms_sd' else ' %p'
 ax.set_title(title,fontsize=15);ax.text(.03,.96,f"대응 차이 중앙값 +{t.difference:{fmt}}{suffix}\n95% CI [{t.ci_low:{fmt}}, {t.ci_high:{fmt}}]\nHolm p={t.holm_p_9:.2g}",transform=ax.transAxes,va='top',fontsize=12,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
 ax.set_xticks([0,1],['액상','분말 (5개 추출)']);ax.set_xlim(-.2,1.2);ax.set_ylim(0,max(a.max(),b.max())*1.38);ax.set_ylabel(unit);ax.grid(axis='y',alpha=.18)
fig.suptitle('공통 검체코드 41개: 액상–분말 대응 비교',x=.07,ha='left',y=.95,fontsize=24,weight='bold')
fig.text(.07,.13,'선 1개 = 공통 검체코드 1개. 양수 차이는 분말에서 변동성이 더 큼을 의미. 주요 9개 검정에 Holm 보정.',fontsize=13)
fig.text(.07,.085,'코드·성별·암종 일치 기준 연결; 동일 생물학적 검체라는 전제. 수집일 대응은 확인되지 않음.',fontsize=12,color='#555555')
save(fig,'02_paired_variability')

band_summary=C.groupby(['cohort','center']).cv_pct.agg(median='median',q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75)).reset_index();band_summary.to_csv(OUT/'band_cv_summary.csv',index=False)
matrix=np.array([[band_summary[(band_summary.cohort==k)&(band_summary.center==c)]['median'].iloc[0] for c in centers] for k in keys])
fig,(ax,ax2)=plt.subplots(2,1,figsize=(16,9),gridspec_kw={'height_ratios':[1,1.5]});fig.subplots_adjust(left=.10,right=.96,top=.86,bottom=.16,hspace=.48)
im=ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0);ax.set_xticks(range(6),[f'{c} ±10' for c in centers]);ax.set_yticks(range(3),names);ax.set_xlabel('고정 파수 구간 (cm^-1)');ax.set_title('구간별 상대 면적 CV 중앙값 (%) — 낮을수록 안정적',loc='left',fontsize=15)
for i in range(3):
 for j in range(6):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',color='white' if matrix[i,j]>matrix.max()*.62 else '#111827',fontsize=14)
fig.colorbar(im,ax=ax,fraction=.02,pad=.02,label='CV (%)')
for key,name,color,ls in zip(keys,names,colors,['-','--','-.']):
 w=W[W.cohort==key];ax2.plot(w.wavenumber,w.mean_within_sd,color=color,ls=ls,lw=1.8,label=name);ax2.fill_between(w.wavenumber,w.ci_low,w.ci_high,color=color,alpha=.16)
ax2.legend(frameon=False,ncol=3,loc='upper right');ax2.set_xlabel('Raman shift (cm^-1)');ax2.set_ylabel('평균 검체 내 SD (정규화)');ax2.set_title('파수별 반복측정 변동성 — 평균 intensity가 아닌 검체 내 SD',loc='left',fontsize=15);ax2.grid(axis='y',alpha=.18);ax2.set_ylim(bottom=0)
fig.suptitle('상대 피크 및 스펙트럼 형태의 변동 위치',x=.10,ha='left',y=.96,fontsize=24,weight='bold')
fig.text(.10,.055,'음영: 검체 내 SD의 그룹 평균에 대한 pointwise 95% bootstrap CI. 고정 구간 변동에는 피크 위치 이동도 포함됨.\n상대 면적 = 구간 보정 면적 / 전체 보정 면적. 모든 검체·추출에서 6개 구간 모두 분모 기준을 통과함.',fontsize=12)
save(fig,'03_band_and_spectral_variability')

sens=SS.groupby('metric').agg(median_delta=('median_paired_difference','median'),min_delta=('median_paired_difference','min'),max_delta=('median_paired_difference','max'),positive_fraction=('median_paired_difference',lambda a:np.mean(a>0)),fraction_unadjusted_p_below_05=('wilcoxon_p',lambda a:np.mean(a<.05))).reset_index();sens.to_csv(OUT/'subset_sensitivity_summary.csv',index=False)
full=pd.read_csv(OUT/'powder_full121_sensitivity.csv').groupby('metric').value.median().reset_index(name='full121_median');full.to_csv(OUT/'powder_full121_summary.csv',index=False)
contract={'question':'Which cohort has higher within-specimen repeat variability at five repeats?','figures':{'01':'3 boxplots with all 175 specimen dots and median CIs; magnitude-free metrics','02':'41 paired-code trajectories per primary metric; multiplicity-adjusted tests','03':'3x6 median relative-band CV heatmap and within-SD-versus-wavenumber curves'},'primary_unit':'specimen','lower_is_better':True,'surface':'standalone Matplotlib PNG/SVG/PDF','privacy':'only aggregate plots shared; local subject-index tables excluded from zip'}
(OUT/'chart_contract.json').write_text(json.dumps(contract,indent=2))
def summary_table():
 out='| 지표 | 후향 91개 | 액상 41개 | 분말 43개 |\n|---|---:|---:|---:|\n'
 for m,title in zip(metrics,titles):
  cells=[]
  for k in keys:
   s=S[(S.cohort==k)&(S.metric==m)].iloc[0];fmt='.3f' if m=='shape_rms_sd' else '.2f';cells.append(f"{s['median']:{fmt}} [{s.median_ci_low:{fmt}}, {s.median_ci_high:{fmt}}]")
  out+='| '+title+' | '+' | '.join(cells)+' |\n'
 return out
paired='| 지표 | 대응 차이 중앙값 (분말−액상) | 95% CI | Holm p |\n|---|---:|---:|---:|\n'
for m,title in zip(metrics,titles):
 t=T[(T.metric==m)&(T.comparison=='powder_minus_liquid_paired')].iloc[0];paired+=f'| {title} | {t.difference:.4f} | [{t.ci_low:.4f}, {t.ci_high:.4f}] | {t.holm_p_9:.3g} |\n'
report='''# 동일 검체 내 반복측정 변동성: 3개 코호트 비교

## 목적과 결과
절대 a.u.의 그룹 평균 차이를 비교하지 않는다. 각 검체를 반복 측정할 때 전체 신호 크기, 상대 피크 비중, 정규화 스펙트럼 형태가 얼마나 변하는지 비교한다. 세 주요 지표 모두 낮을수록 안정적이다.

검체별 지표의 중앙값 [중앙값의 95% bootstrap CI]:

'''+summary_table()+'''
분말군은 세 주요 지표 모두에서 액상과 후향보다 높은 변동성을 보였다. 액상–후향은 전체 면적 CV와 형태 SD의 차이가 유의하지 않았고 상대 피크 CV는 액상에서 낮았다. 이 결과는 관측된 반복/측정 위치 재현성이며 환원제의 단독 인과효과나 진단 AUC를 뜻하지 않는다.

## 지표 정의
각 반복 spectrum의 원래 축에서 trim → Savitzky–Golay(window 11, polynomial 3) → rolling-min baseline(window 101) → 공통 935-point grid 보간을 적용한다. 원본 함수의 검체별 평균과 일치하는지 확인했다.

1. **전체 보정 면적 CV (%)**: 각 반복의 baseline-corrected 전체 면적을 A_r라 하면 `100 × sample_SD(A_r) / mean(A_r)`. 한 검체에서 전체 intensity gain이 흔들리는 정도다. 단위와 공통 배율에 무관하지만 background correction과 획득 조건의 영향을 받는다. 원래 raw mean의 그룹 간 차이가 아니다.
2. **상대 피크 면적 CV (%)**: 고정 구간 면적을 전체 보정 면적으로 나눈 비율 f_r,b에 대해 `100 × sample_SD(f_r,b) / mean(f_r,b)`. 6구간 CV의 중앙값을 검체 점수로 사용한다. 이는 전체 intensity gain 변화와 구분되는 상대 pattern 변동이다.
3. **스펙트럼 형태 RMS SD**: 각 반복을 공통 grid에서 zero-mean/unit-SD로 정규화하고, 파수별 반복 간 sample SD의 RMS를 계산한다. `sqrt(mean_wavelength(sample_variance_across_repeats))`. 공통 offset/gain의 영향을 제거한 전체 형태 변동이다. 변동계수가 아니며 % 단위가 아니다.
4. 보조 지표 **상대 구간 비율 RMS SD(%p)**도 subject/group CSV에 포함했다. 평균 분모가 작은 구간에서 CV가 커지는 현상과 구분해 볼 수 있다.

고정 구간: 745/935/1000/1445/1650/2100 cm^-1 중심 ±10. Peak height나 피팅한 단일 peak area가 아니라 고정 구간의 적분값이다. Peak 위치 이동과 폭 변화도 반영된다.

## 5회 조건 맞추기
- 후향 91개, 액상 41개: 각 5개 반복 모두 사용.
- 분말 43개: 각 121개 중 5개를 비복원 무작위 추출, 검체마다 500회. 각 회차의 변동성 지표를 계산하고 그 평균을 검체 점수 하나로 사용.
- 검체별 500회는 독립 환자 표본이 아니다. 검정 n은 91/41/43 또는 공통 코드 41쌍.
- 이 추정량은 분말에서 가능한 5개 부분집합에 대한 기대 지표다. 분말 점수는 500회 평균으로 선택 잡음이 줄지만 액상/후향은 한 세트만 있으므로 지표 추정 정밀도가 같지는 않다.
- 이 차이를 보기 위해 분말에서 검체당 단 한 세트만 사용한 대응 비교를 500회 별도로 산출했다. single_subset_paired_sensitivity.csv와 subset_sensitivity_summary.csv의 p 비율은 민감도 요약이며 합쳐진 유의확률이나 새 검정이 아니다.
- 분말 121개 전체의 지표는 powder_full121_summary.csv에 별도 제공한다. 이는 5개 주 비교와 섞지 않는다.

## 검정과 효과
검체별 지표 분포가 치우칠 수 있어 후향 비교는 양측 Mann–Whitney U, 공통 코드 액상–분말 비교는 양측 Wilcoxon signed-rank를 사용했다. 주요 3지표×3비교=9개 검정에 Holm 보정했다.

'''+paired+'''
독립 비교 효과는 두 그룹 중앙값의 차이, 대응 비교 효과는 검체별 차이의 중앙값이다. 이 둘은 서로 다른 추정량이다. Rank 검정은 그 CI와 완전히 같은 귀무가설을 검정하지 않는다. Mann–Whitney는 분포/순위 차이, Wilcoxon은 대칭적인 차이 분포의 중심 이동을 검정한다. 대응 비교의 분포 가정에 덜 민감한 sign test도 primary_comparisons.csv에 보조 제공했다.

CI는 검체 bootstrap 10,000회 percentile 구간이다. 대응 비교는 쌍 단위 재표집, 독립 비교는 그룹별 재표집. 관측된 반복과 분말 부분집합 평균에 조건부이며, 새 lot나 새 측정의 계층적 불확실성을 포함하지 않는다. CI 자체는 다중비교 보정하지 않았다.

6개 구간×3비교=18개 상대 구간 CV 검정에는 별도 Holm 보정했다. 이들은 앞선 그림을 기준으로 정한 탐색 구간이며 사전등록 확증 분석이 아니다.

## 분모와 결측 검증
상대 구간의 평균 비율이 전체 면적의 0.1% 이상이고 모든 반복에서 양수일 때만 해당 CV를 허용했다. 검체별 요약은 최소 4개 유효 구간을 요구했다. 실제로 모든 검체의 모든 부분집합에서 6개 구간이 전부 기준을 통과했다. SNV의 0 근처/음수 평균에 CV를 적용하지 않았다.

## 원본 및 제한
- 입력은 앞선 three_cohort_spectra_ci_20260828_v1과 동일한 원천. 기존 검체 평균과 수치적으로 일치함을 검증했다.
- 액상–분말 41개 공유 solum_label 및 성별/암종 일치를 이용했다. 같은 생물학적 검체라는 전제가 필요하며 수집일 연결은 확인되지 않았다.
- 반복 5개와 mapping 121개의 공간적 배치/취득 방식이 같다고 확인한 것은 아니다. 5개로 수를 맞춰도 측정 위치 이질성, 획득시간, 장비, 날짜, lot, 보관 차이는 남는다.
- 개별 검체의 biological heterogeneity와 spot-to-spot 측정 변동을 완전히 분리하지 않는다. 같은 위치 반복의 순수 instrument repeatability만을 뜻하지 않는다.
- 각 군은 모두 전립선암 검체다. 이번 분석으로 변동성이 분류 AUC 저하를 일으켰다고 결론내리지 않는다.
- 노이즈 없는 평평한 스펙트럼이 좋은 측정이라는 뜻은 아니다. 정규화 형태 SD와 면적 CV는 재현성 지표이며 신호의 분자 특이성, 정보량 또는 정확도를 직접 평가하지 않는다.

## 시각화 해석
- 01: 점 1개=검체 1개; 상자 IQR, 선 중앙값, 수염 1.5IQR. 숫자는 중앙값 [95% CI].
- 02: 공통 코드 41쌍의 변동성 이동. 숫자는 대응 차이 중앙값과 CI 및 보정 p.
- 03: 위는 구간별 CV 중앙값, 아래는 평균 intensity가 아니라 **검체 내 반복 SD의 그룹 평균**. 음영은 pointwise CI.

## 검증 및 재현
independent_validation.csv에 원시 반복에서 직접 계산한 CV/SD 공식과 signed-rank 통계량의 재검산 결과를 저장했다. 코드, protocol, source hashes를 보존했다. 입력 배열과 검체 인덱스별 상세 파일은 로컬 연구자료로 유지하고 공유 ZIP에서는 제외했다.
'''
(OUT/'README.md').write_text(report,encoding='utf-8')
sources=[ROOT/'notebooks/within_subject_variability.py',Path(__file__),ROOT/'publications/전향검체/보라매병원/src/three_cohort_spectra.py',ROOT/'src/sers/signal.py']
(OUT/'source_hashes.json').write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},ensure_ascii=False,indent=2))
for p in sources[:2]:shutil.copy2(p,OUT/p.name)
excluded={'subject_variability.csv','subject_band_cv.csv','band_eligibility.csv','powder_full121_sensitivity.csv'}
with zipfile.ZipFile(OUT/'within_subject_variability_results.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in OUT.iterdir():
  if p.suffix in ['.png','.svg','.pdf','.csv','.md','.json','.py'] and p.name not in excluded:z.write(p,p.name)
print(summary_table());print(paired);print(sens.to_string(index=False));print(full.to_string(index=False));print('VALIDATED AND EXPORTED')
