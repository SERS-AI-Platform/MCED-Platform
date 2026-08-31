"""Repeatability, not between-cohort mean intensity. Independent unit: specimen."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'publications/전향검체/보라매병원/src')]
import three_cohort_spectra as source
OUT=ROOT/'results/within_subject_variability_20260828_v1';OUT.mkdir(exist_ok=True)
PREV=ROOT/'results/three_cohort_spectra_ci_20260828_v1'
OLD=dict(np.load(PREV/'subject_spectra.npz'));GRID=OLD['grid']
KEYS=['retrospective','liquid','powder'];CENTERS=[745,935,1000,1445,1650,2100]
MC=500;BOOT=10000;SEED=20260828;MIN_FRACTION=.001
PRIMARY=['total_area_cv_pct','relative_band_cv_pct','shape_rms_sd']

def prepare():
 path=OUT/'_replicate_inputs.npz'
 if path.exists():return dict(np.load(path))
 captured=[]
 def capture(grouped,grid):
  raw=[];processed=[];areas=[];fractions=[]
  for key in sorted(grouped):
   rr=[];ss=[];aa=[];ff=[]
   for p in grouped[key]:
    x,y=source.read_spectrum(p);mask=(x>=grid.min())&(x<=grid.max());x=x[mask];y=y[mask]
    corrected=source.baseline_correction(savgol_filter(y,11,3,mode='interp'),window=101)
    z=np.interp(grid,x,corrected);assert z.min()>-1e-8
    area=np.trapezoid(z,x=grid);assert area>0
    rr.append(np.interp(grid,x,y));ss.append(np.interp(grid,x,source.snv(corrected)))
    aa.append(area);ff.append([np.trapezoid(z[(grid>=c-10)&(grid<=c+10)],x=grid[(grid>=c-10)&(grid<=c+10)])/area for c in CENTERS])
   raw.append(rr);processed.append(ss);areas.append(aa);fractions.append(ff)
  captured.append((np.array(raw),np.array(processed),np.array(areas),np.array(fractions)))
  print('captured cohort',len(raw),'repeat count',len(raw[0]),flush=True)
  return np.mean(raw,axis=1),np.mean(processed,axis=1)
 source._subject_means=capture
 source.load_three_cohort_series(ROOT,GRID)
 data={'grid':GRID,'liquid_codes':OLD['liquid_codes'],'powder_codes':OLD['powder_codes']};errors={}
 for k,(raw,processed,area,fraction) in zip(KEYS,captured):
  np.testing.assert_allclose(raw.mean(1),OLD[k+'_raw'],rtol=1e-10,atol=1e-8)
  np.testing.assert_allclose(processed.mean(1),OLD[k+'_processed'],rtol=1e-10,atol=1e-8)
  errors[k]=float(np.abs(processed.mean(1)-OLD[k+'_processed']).max())
  # Re-standardize after interpolation so shape metric is strictly scale-free on the common grid.
  shape=(processed-processed.mean(2,keepdims=True))/processed.std(2,keepdims=True)
  data[k+'_shape']=shape;data[k+'_area']=area;data[k+'_fraction']=fraction
 np.savez_compressed(path,**data)
 (OUT/'input_validation.json').write_text(json.dumps(errors,indent=2))
 return data

def metrics(shape,area,fraction):
 # batch x repeats x points / bands
 sd=shape.std(axis=1,ddof=1);band_sd=fraction.std(axis=1,ddof=1)
 avg=fraction.mean(1);valid=(avg>=MIN_FRACTION)&(fraction.min(1)>0)
 cv=np.divide(100*band_sd,avg,out=np.full_like(avg,np.nan),where=valid)
 score=np.nanmedian(cv,axis=1);score[valid.sum(1)<4]=np.nan
 return dict(total_area_cv_pct=100*area.std(1,ddof=1)/area.mean(1),relative_band_cv_pct=score,
             shape_rms_sd=np.sqrt((sd**2).mean(1)),relative_band_rms_sd_pp=100*np.sqrt((band_sd**2).mean(1))),sd,cv,valid

def holm(p):
 p=np.asarray(p);order=np.argsort(p);q=np.empty_like(p);q[order]=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))));return q

def main():
 data=prepare();rows=[];bandrows=[];fullrows=[];maps={};mcmetrics={};cv_arrays={};validity=[]
 for key in KEYS:
  shape=data[key+'_shape'];area=data[key+'_area'];fraction=data[key+'_fraction'];n=len(shape)
  maplist=[];cvs=[];metrics_list=[]
  for s in range(n):
   if key=='powder':
    rng=np.random.default_rng(np.random.SeedSequence([SEED,s]));idx=np.array([rng.choice(shape.shape[1],5,replace=False) for _ in range(MC)])
   else:idx=np.arange(5)[None,:]
   m,sd,cv,valid=metrics(shape[s][idx],area[s][idx],fraction[s][idx]);metrics_list.append(m)
   maplist.append(sd.mean(0));cvs.append(np.nanmean(cv,axis=0))
   for metric,vals in m.items():
    assert np.isfinite(vals).all(),(key,s,metric)
    rows.append(dict(cohort=key,subject_index=s,metric=metric,value=vals.mean(),selection_sd=vals.std(ddof=1) if len(vals)>1 else 0,subset_count=len(vals),repeats_used=5))
   for j,c in enumerate(CENTERS):
    bandrows.append(dict(cohort=key,subject_index=s,center=c,cv_pct=np.nanmean(cv[:,j]),eligible_fraction=valid[:,j].mean()))
   validity.append(dict(cohort=key,subject_index=s,minimum_eligible_bands=int(valid.sum(1).min()),maximum_eligible_bands=int(valid.sum(1).max())))
  maps[key]=np.array(maplist);cv_arrays[key]=np.array(cvs)
  if key=='powder':
   mcmetrics={metric:np.stack([m[metric] for m in metrics_list]) for metric in PRIMARY}
   full,_,_,_=metrics(shape,area,fraction)
   for metric,vals in full.items():
    for s,v in enumerate(vals):fullrows.append(dict(cohort=key,subject_index=s,metric=metric,value=v,repeats_used=121))
  print(key,'metrics complete',flush=True)
 R=pd.DataFrame(rows);R.to_csv(OUT/'subject_variability.csv',index=False)
 pd.DataFrame(bandrows).to_csv(OUT/'subject_band_cv.csv',index=False);pd.DataFrame(fullrows).to_csv(OUT/'powder_full121_sensitivity.csv',index=False);pd.DataFrame(validity).to_csv(OUT/'band_eligibility.csv',index=False)
 np.savez_compressed(OUT/'_powder_subset_metrics.npz',**mcmetrics)
 rng=np.random.default_rng(SEED+1);summary=[];maprows=[]
 for key in KEYS:
  for metric in R.metric.unique():
   v=R[(R.cohort==key)&(R.metric==metric)].value.to_numpy();med=np.median(v);idx=rng.integers(0,len(v),(BOOT,len(v)));lo,hi=np.quantile(np.median(v[idx],axis=1),[.025,.975])
   summary.append(dict(cohort=key,metric=metric,n=len(v),median=med,q25=np.quantile(v,.25),q75=np.quantile(v,.75),median_ci_low=lo,median_ci_high=hi,mean=v.mean()))
  a=maps[key];weights=rng.multinomial(len(a),np.ones(len(a))/len(a),size=BOOT)/len(a);lo,hi=np.quantile(weights@a,[.025,.975],axis=0)
  for j,x in enumerate(GRID):maprows.append(dict(cohort=key,wavenumber=x,mean_within_sd=a[:,j].mean(),ci_low=lo[j],ci_high=hi[j]))
 S=pd.DataFrame(summary);S.to_csv(OUT/'group_variability_summary.csv',index=False);pd.DataFrame(maprows).to_csv(OUT/'within_sd_spectrum.csv',index=False)
 pairedidx=np.array([np.where(data['powder_codes']==k)[0][0] for k in data['liquid_codes']])
 comparisons=[('liquid_minus_retrospective','liquid','retrospective',False),('powder_minus_retrospective','powder','retrospective',False),('powder_minus_liquid_paired','powder','liquid',True)]
 tests=[];bandtests=[];sensitivity=[]
 def compare(a,b,paired):
  if paired:
   d=a-b;idx=rng.integers(0,len(d),(BOOT,len(d)));boot=np.median(d[idx],axis=1);test=stats.wilcoxon(d,alternative='two-sided');effect=np.median(d);estimand='median_paired_difference'
   nonzero=d[d!=0];signp=stats.binomtest(int((nonzero>0).sum()),len(nonzero),.5).pvalue if len(nonzero) else 1.
  else:
   boot=np.median(a[rng.integers(0,len(a),(BOOT,len(a)))],axis=1)-np.median(b[rng.integers(0,len(b),(BOOT,len(b)))],axis=1)
   test=stats.mannwhitneyu(a,b,alternative='two-sided',method='auto');effect=np.median(a)-np.median(b);estimand='difference_of_medians';signp=np.nan
  lo,hi=np.quantile(boot,[.025,.975]);return dict(n_a=len(a),n_b=len(b),difference=effect,ci_low=lo,ci_high=hi,p=test.pvalue,statistic=test.statistic,estimand=estimand,sign_test_p=signp)
 for metric in PRIMARY:
  for name,ka,kb,paired in comparisons:
   a=R[(R.cohort==ka)&(R.metric==metric)].value.to_numpy();b=R[(R.cohort==kb)&(R.metric==metric)].value.to_numpy()
   if paired:a=a[pairedidx]
   tests.append(dict(metric=metric,comparison=name,paired=paired,**compare(a,b,paired)))
  # One five-repeat draw per powder specimen, repeated 500 times. This is a sensitivity distribution, not 500 independent samples.
  b=R[(R.cohort=='liquid')&(R.metric==metric)].value.to_numpy()
  for iteration in range(MC):
   d=mcmetrics[metric][pairedidx,iteration]-b
   sensitivity.append(dict(metric=metric,iteration=iteration,median_paired_difference=np.median(d),wilcoxon_p=stats.wilcoxon(d).pvalue))
 for j,c in enumerate(CENTERS):
  for name,ka,kb,paired in comparisons:
   a=cv_arrays[ka][:,j];b=cv_arrays[kb][:,j]
   if paired:
    a=a[pairedidx];valid=np.isfinite(a)&np.isfinite(b);a=a[valid];b=b[valid]
   else:a=a[np.isfinite(a)];b=b[np.isfinite(b)]
   bandtests.append(dict(center=c,comparison=name,paired=paired,**compare(a,b,paired)))
 T=pd.DataFrame(tests);T['holm_p_9']=holm(T.p);T.to_csv(OUT/'primary_comparisons.csv',index=False)
 BT=pd.DataFrame(bandtests);BT['holm_p_18']=holm(BT.p);BT.to_csv(OUT/'band_cv_comparisons.csv',index=False)
 pd.DataFrame(sensitivity).to_csv(OUT/'single_subset_paired_sensitivity.csv',index=False)
 protocol=dict(seed=SEED,powder_subsets=MC,bootstrap=BOOT,repeats_per_subset=5,centers=CENTERS,halfwidth_cm1=10,minimum_mean_band_fraction=MIN_FRACTION,minimum_eligible_bands=4,primary_metrics=PRIMARY,independent_unit='specimen',paired_linkage='41 shared solum_label codes; same specimen identity assumption; collection dates unavailable',primary_adjustment='Holm across 9 tests',band_adjustment='Holm across 18 tests',summary_ci='subject bootstrap percentile CI of median, conditional on available repeats and averaged powder subsets',powder_estimand='per-specimen mean metric over 500 random five-repeat subsets; raw/processed means are NOT the evaluated outcome')
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));print(S.to_string(index=False));print(T.to_string(index=False));print('COMPLETE',flush=True)

if __name__=='__main__':main()
