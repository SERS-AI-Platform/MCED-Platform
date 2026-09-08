"""Verify saved predictions independently, then export aggregate-only figures/report."""
from pathlib import Path
import hashlib,json,zipfile
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'results/three_method_repeat_auc_20260828_v1'
NAMES={'direct':'최소 전처리 후 진행','raw':'Peak 형태 보존 및 Denoising, baseline correction 방식','legacy':'기존 방식'}
SUBTITLES={'direct':'PS alignment + QC → mean → direct stacking','raw':'Reference-peak alignment → mean → STK-V2 (raw control)','legacy':'933-point interpolation → mean → STK-V2'}
COLORS={'direct':'#6B7280','raw':'#391D8D','legacy':'#0A306D'}
MARKERS={'direct':'o','raw':'s','legacy':'^'}
LINES={'direct':':','raw':'--','legacy':'-'}

def main():
    df=pd.read_csv(OUT/'repeat_auc_summary.csv')
    iterations=pd.read_csv(OUT/'repeat_auc_iterations.csv')
    full=pd.read_csv(OUT/'full_repeat_validation.csv').set_index('method')
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert len(df)==27 and len(iterations)==540
    checks=[]
    for key in NAMES:
        coverage=np.zeros(113,dtype=int)
        for i in range(5):
            fold=joblib.load(OUT/f'{key}_fold{i}.joblib')
            assert not set(fold['tr'])&set(fold['te'])
            assert set(fold['tr'])|set(fold['te'])==set(range(113))
            coverage[fold['te']]+=1
        assert np.all(coverage==1)
    for row in df.itertuples():
        p=np.load(OUT/f'{row.method}_n{row.n_repeats:03d}_predictions.npz')
        a=np.array([roc_auc_score(p['y'],x) for x in p['probability']])
        assert p['probability'].shape==(20,113)
        np.testing.assert_allclose([a.mean(),a.std(ddof=1)],[row.mean,row.sd],rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(a,iterations.query('method==@row.method and n_repeats==@row.n_repeats').auc,rtol=1e-12)
        checks.append(dict(method=row.method,n_repeats=row.n_repeats,verified=True))
    assert (full.max_probability_difference<1e-5).all()
    pd.DataFrame(checks).to_csv(OUT/'independent_validation.csv',index=False)
    for f in Path('/home/user/.fonts').glob('*.ttf'):font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({'font.family':'NanumSquare','font.size':13,'axes.labelcolor':'#111827','text.color':'#111827','axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','axes.unicode_minus':False})
    counts=protocol['counts']
    for selected,filename in [(['direct','raw','legacy'],'repeat_auc_comparison')]+[([k],f'{k}_repeat_auc') for k in NAMES]:
        fig,ax=plt.subplots(figsize=(16,9));fig.subplots_adjust(left=.085,right=.96,bottom=.25,top=.80)
        fig.text(.055,.94,'반복측정 개수에 따른 Cancer Screening AUC',fontsize=26,weight='bold')
        fig.text(.055,.885,'113명 · 피험자 분리 5×5 nested CV · n개 평균을 재계산 · 무작위 추출 20회',fontsize=17)
        for key in selected:
            s=df[df.method==key].sort_values('n_repeats');x=s.n_repeats.to_numpy();a=s['mean'].to_numpy();sd=s.sd.to_numpy()
            label=NAMES[key] if key!='raw' else 'Peak 형태 보존 및 Denoising, baseline correction 방식 [raw 대조군]'
            ax.plot(x,a,color=COLORS[key],marker=MARKERS[key],linestyle=LINES[key],lw=2.5,label=label)
            ax.fill_between(x,a-sd,a+sd,color=COLORS[key],alpha=.10)
            ax.scatter([125],[full.loc[key,'auc']],s=110,marker=MARKERS[key],color=COLORS[key])
            ax.annotate(f"{full.loc[key,'auc']:.4f}",(125,full.loc[key,'auc']),xytext=(-7,9),textcoords='offset points',ha='right',color=COLORS[key])
        ax.axhline(.5,color='#333333',linestyle=':',lw=1)
        ax.axvline(113,color='#D6E0EF',lw=1)
        ax.set(ylim=(.40,.85),xlim=(-2,132),ylabel='Subject-level pooled OOF ROC-AUC',xlabel='평균에 사용한 반복측정 수 n (개)')
        ax.set_xticks([1,9,16,25,49,81,100,125],['1','9','16','25','49','81','100','전체 사용*'])
        ax.grid(axis='y',color='#D6E0EF',lw=.6)
        ax.legend(loc='lower right',fontsize=12,frameon=False)
        fig.text(.055,.145,'음영: Monte Carlo 평균 ± 1 SD (환자 표본의 신뢰구간 아님). 전체 사용*: QC 후 105–120개 / 다른 두 방법 121개.',fontsize=13)
        fig.text(.055,.11,'전체 평균으로 학습한 모델 고정. QC는 전체 측정 기준으로 먼저 적용. 기존 방식은 원래의 다른 fold 배치를 유지.',fontsize=13)
        fig.text(.055,.075,'문헌 방법의 재현 실험이 아님. 병원 confound 및 외부 일반화 미검증. Source: three_method_repeat_auc_20260828_v1',fontsize=12,color='#6B7280')
        for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{filename}.{ext}',dpi=180)
        plt.close(fig)
    display=df.copy();display['method_name']=display.method.map(NAMES)
    display.to_csv(OUT/'repeat_auc_summary_named.csv',index=False,encoding='utf-8-sig')
    table='| n | 최소 전처리 후 진행 | Peak 형태 보존… 방식 (raw 대조군) | 기존 방식 |\n|---:|---:|---:|---:|\n'
    for n in counts:
        cells=[]
        for k in NAMES:
            r=df[(df.method==k)&(df.n_repeats==n)].iloc[0];cells.append(f"{r['mean']:.4f} ± {r.sd:.4f}")
        table+=f"| {n} | "+' | '.join(cells)+' |\n'
    table+='| 전체 사용 | '+' | '.join(f"{full.loc[k,'auc']:.4f}" for k in NAMES)+' |\n'
    report='''# 세 방법의 반복측정 개수별 AUC 별도 산출

## 산출 결과
값은 Cancer Screening 2군 subject-level pooled OOF ROC-AUC의 무작위 추출 20회 평균 ± 표본 SD이다. 전체 사용은 반복 추출이 없는 기준점이다.

'''+table+'''
## 평가 방법
- 대상: 113명. Control + PDC를 Non-cancer로, Cancer를 양성으로 정의. 클래스별 수는 protocol.json 참조.
- 원본 API 스펙트럼에서 각 전처리의 반복측정 배열을 복원하고, 전체 평균이 기존 입력과 일치하는지 검증했다.
- 각 방법의 기존 5-fold outer / 5-fold inner GroupKFold 및 기존 모델 설정을 유지했다. Outer test 피험자는 해당 모델 학습에서 제외된다.
- 전체 반복측정 평균으로 nested stacking을 한 번 학습하고 고정했다. n마다 재학습한 결과가 아니다.
- 각 피험자에서 n개의 측정을 비복원 추출 → 평균 spectrum 계산 → 해당 방법의 입력 변환 → 그 피험자를 보지 않은 outer 모델로 예측한다.
- 매 추출 회차의 113명 예측을 모아 AUC를 계산한다. Fold AUC 평균이나 반복 spectrum 단위 AUC가 아니다. 예측확률을 평균한 뒤 AUC를 한 번 구한 값도 아니다.
- 각 n에서 20회; seed=20260828. 같은 회차·피험자의 순열 앞 n개를 사용해 n 사이에 nested subset을 구성했다. Raw/legacy는 동일한 측정 인덱스를 사용한다. Direct는 QC 후 풀이 다르다.
- 분류 threshold는 ROC-AUC 계산에 사용되지 않는다.

## 실제 입력 처리와 표시명
1. 최소 전처리 후 진행: 저장된 측정별 PS 축 보정값 적용 → 공통 934-point grid 보간 → 피험자 내 MAD 기반 QC → n개 평균 → direct stacking. 추가 baseline/denoising 없음.
2. Peak 형태 보존 및 Denoising, baseline correction 방식: reference-peak alignment → n개 평균 → STK-V2 채널 및 peak features. 실제 실행은 DWT 실험의 raw 대조군이며 Han CDAE를 실행한 것이 아니다. STK-V2에는 smoothing, baseline 처리, SNV, derivative 및 peak feature 변환이 있다.
3. 기존 방식: 원래 축의 933-point 보간 → n개 평균 → 기존 STK-V2. Liu/Lin 논문 구현으로 간주하지 않는다.

## 해석 제한
- 이번 결과는 측정 개수에 대한 **고정 모델의 사후 입력 민감도 분석**이다. n개만 수집해 새 모델을 학습하는 실험이나 전향적 최소 필요 측정 수의 검증이 아니다.
- Direct QC는 전체 반복측정을 보고 결정한 뒤 QC 통과 측정에서 추출했다. 따라서 n개만 취득했을 때 실행 가능한 QC를 평가한 결과가 아니다.
- Direct QC 통과 수는 105–120개로 다르다. 공통 비교는 n≤100으로 제한했고 전체 사용은 별도 기준점이다. 다른 두 방법은 121개다.
- 기존 방식은 원래 subject-key 순서에 따른 fold를 유지했다. 세 방법 간 차이는 전처리만의 인과 효과나 유의한 우월성으로 해석하면 안 된다.
- 음영/±는 반복 선택에 따른 SD이며 모집단·환자 bootstrap 신뢰구간이 아니다. 20회 Monte Carlo 탐색 산출이며 작은 차이의 순위는 단정하지 않는다.
- ResNet repeat 분석은 개별 spectrum 예측 후 확률 집계 방식이고 이번 분석은 spectrum 평균 후 예측 방식이다. 서로 같은 추론 절차가 아니다.
- Hospital-confound 가능성 및 외부 일반화 미검증.

## 검증과 파일
- input_validation.json: 원본으로 재구성한 전체 평균과 기존 입력의 최대 오차.
- full_repeat_validation.csv: 전체 평균을 사용했을 때 기존 예측과 새 모델 예측 비교.
- independent_validation.csv: 저장된 n별 예측에서 AUC/SD를 독립 재계산한 검증.
- repeat_auc_summary_named.csv: 한글 표시명 포함 요약. repeat_auc_iterations.csv: 회차별 AUC.
- repeat_auc_comparison 및 방법별 repeat_auc: PNG/SVG/PDF.
- _replicates.joblib, fold 모델, predictions.npz는 로컬 재현용 연구자료이며 공유 ZIP에서 제외했다. 식별자를 제거한 인덱스도 외부 공유에 주의한다.
'''
    (OUT/'README.md').write_text(report,encoding='utf-8')
    sources=[ROOT/'calculate_three_method_repeat_auc.py',Path(__file__),ROOT/'train_stkv2_stacking.py',ROOT/'stk_v2_preprocess.py',ROOT/'build_stkv2_dataset.py',ROOT/'aecd_api_model_patent_dwt_clinical_performance.py',ROOT.parent/'scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py']
    (OUT/'source_hashes.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},indent=2))
    import shutil
    for f in sources[:2]:shutil.copy2(f,OUT/f.name)
    with zipfile.ZipFile(OUT/'repeat_auc_aggregate_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in OUT.iterdir():
            if f.suffix in ['.png','.svg','.pdf','.md','.json','.py','.csv'] and f.name!='repeat_counts.csv':z.write(f,f.name)
    print(table)
    print('VALIDATED 27 conditions / 540 AUC calculations; figures and report exported')

if __name__=='__main__':main()
