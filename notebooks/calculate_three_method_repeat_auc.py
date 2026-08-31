"""Retrospective held-out repeat-count sensitivity; never retrain on test subjects."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parent
OUT=ROOT.parent/'results/three_method_repeat_auc_20260828_v1'
OUT.mkdir(parents=True,exist_ok=True)
os.environ['AECD_PATENT_DWT_OUTDIR']=str(OUT/'_dwt_import')
os.environ['AECD_MEAN_SPECTRUM_OUTDIR']=str(OUT/'_mean_import')
import sys, importlib.util, json, warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore',category=ConvergenceWarning)
import aecd_api_model_patent_dwt_clinical_performance as dwt
import train_stkv2_stacking as stk
sys.path.insert(0,str(ROOT.parent/'scripts/analysis'))
import aecd_api_model_mean_spectrum_clinical_performance as mean
COUNTS=[1,3,5,9,16,25,49,81,100]
MC=20
SEED=20260828

def features_one(x,grid):
    c=dwt.stk_v2_channels(grid,x,target_grid=dwt.STK_TARGET_GRID)
    p=dwt.extract_peak_features(dwt.STK_TARGET_GRID,c[0])
    return c,p

def views_for(X,grid,key):
    if key=='direct': return {'raw':X}
    pairs=joblib.Parallel(n_jobs=16)(joblib.delayed(features_one)(x,grid) for x in X)
    c=np.stack([p[0] for p in pairs]);p=np.stack([p[1] for p in pairs])
    return stk.build_views(dict(raw=c[:,0],d1=c[:,1],d2=c[:,2],peak=p))

def prepare():
    path=OUT/'_replicates.joblib'
    if path.exists(): return joblib.load(path)
    old=dict(np.load(ROOT/'three_condition_figures_20260828/_inputs.npz'))
    items,_,_=dwt.load_api_spectra()
    keys,labels,raw,_=dwt.group_and_align_subjects(items,old['grid'])
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in items: grouped[str(r['subject_key'])].append(r)
    shifts=pd.read_csv(ROOT/'aecd_api_model_mean_spectrum_outputs/calibration_shift_values.csv').set_index('measurement_id')
    direct=[];legacy=[];counts=[]
    for i,k in enumerate(keys):
        rows=sorted(grouped[k],key=lambda r:(int(r.get('replicate_number',0)),int(r.get('measurement_id',0))))
        legacy.append(np.asarray([np.interp(old['legacy_grid'],r['wavenumber'],r['intensities']) for r in rows]))
        ps=np.asarray([np.interp(old['grid'],np.asarray(r['wavenumber'])+float(shifts.loc[int(r['measurement_id']),'applied_axis_correction_cm1']),r['intensities']) for r in rows])
        keep,_,_=mean.qc_repeats(ps);direct.append(ps[keep])
        counts.append(dict(subject_index=i,all_repeats=len(rows),qc_passed=int(keep.sum())))
    assert np.array_equal(old['y3'],old['direct_y3'])
    verification={}
    for k,reps in [('raw',raw),('legacy',legacy),('direct',direct)]:
        X=np.stack([r.mean(0) for r in reps])
        verification[k]=float(np.max(np.abs(X-old[k])))
        np.testing.assert_allclose(X,old[k],rtol=1e-9,atol=1e-8)
    data=dict(old=old,raw=raw,legacy=legacy,direct=direct)
    joblib.dump(data,path,compress=3)
    pd.DataFrame(counts).to_csv(OUT/'repeat_counts.csv',index=False)
    (OUT/'input_validation.json').write_text(json.dumps(verification,indent=2))
    print('INPUT VERIFIED',verification,flush=True)
    return data

def fit_fold(key,views,y,groups,fold,tr,te):
    path=OUT/f'{key}_fold{fold}.joblib'
    if path.exists(): return joblib.load(path)
    specs=[(n,'raw',f) for n,f in mean.make_direct_base_models()] if key=='direct' else stk.make_base_models()
    meta_X=np.zeros((len(tr),len(specs)));models=[]
    for j,(name,v,factory) in enumerate(specs):
        def make():
            m=factory()
            if 'n_jobs' in m.get_params(deep=False):m.set_params(n_jobs=2)
            return m
        for itr,ite in GroupKFold(5).split(views[v][tr],y[tr],groups[tr]):
            m=make();m.fit(views[v][tr][itr],y[tr][itr]);meta_X[ite,j]=m.predict_proba(views[v][tr][ite])[:,1]
        m=make();m.fit(views[v][tr],y[tr]);models.append((v,m))
    meta=mean._make_meta_model() if key=='direct' else stk._make_meta_model()
    meta.fit(meta_X,y[tr])
    result=dict(te=te,tr=tr,models=models,meta=meta)
    assert not set(tr)&set(te)
    joblib.dump(result,path,compress=3)
    print(key,'fold',fold,'trained',flush=True)
    return result

def predict(folds,views,n_subjects):
    total=len(views['raw']);assert total%n_subjects==0
    p=np.full(total,np.nan)
    for f in folds:
        ids=np.concatenate([f['te']+i*n_subjects for i in range(total//n_subjects)])
        z=np.column_stack([m.predict_proba(views[v][ids])[:,1] for v,m in f['models']])
        p[ids]=f['meta'].predict_proba(z)[:,1]
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    return p.reshape(-1,n_subjects)

def main():
    data=prepare();old=data['old'];y=(old['y3']==2).astype(int);N=len(y)
    rows=[];baselines=[]
    for key in ['direct','raw','legacy']:
        print('START',key,flush=True)
        grid=old['legacy_grid'] if key=='legacy' else old['grid']
        cache=OUT/f'{key}_full_views.joblib'
        if cache.exists():views=joblib.load(cache)
        else:
            views=views_for(old[key],grid,key);joblib.dump(views,cache)
        groups=old['legacy_groups'] if key=='legacy' else np.arange(N)
        folds=joblib.Parallel(n_jobs=5)(joblib.delayed(fit_fold)(key,views,y,groups,i,tr,te) for i,(tr,te) in enumerate(GroupKFold(5).split(views['raw'],y,groups)))
        p=predict(folds,views,N)[0]
        reference=np.load(ROOT/f'three_condition_figures_20260828/{key}/predictions.npz')['binary']
        delta=float(np.max(np.abs(p-reference)))
        auc=roc_auc_score(y,p)
        baselines.append(dict(method=key,auc=auc,reference_auc=roc_auc_score(y,reference),max_probability_difference=delta,min_repeats=min(map(len,data[key])),max_repeats=max(map(len,data[key]))))
        pd.DataFrame(baselines).to_csv(OUT/'full_repeat_validation.csv',index=False)
        # Numerical tolerance allows parallel BLAS round-off, not materially different predictions.
        assert delta<1e-5,(key,delta)
        print(key,'FULL VERIFIED',auc,'max probability delta',delta,flush=True)
        for n in COUNTS:
            dest=OUT/f'{key}_n{n:03d}_predictions.npz'
            if dest.exists():probs=np.load(dest)['probability']
            else:
                X=[]
                for iteration in range(MC):
                    for subject,r in enumerate(data[key]):
                        rng=np.random.default_rng(np.random.SeedSequence([SEED,iteration,subject]))
                        idx=rng.permutation(len(r))[:n]
                        assert len(idx)==n and len(np.unique(idx))==n
                        X.append(r[idx].mean(0))
                v=views_for(np.asarray(X),grid,key)
                probs=predict(folds,v,N)
                np.savez_compressed(dest,probability=probs,y=y,n=n,seed=SEED)
            for i,pr in enumerate(probs):rows.append(dict(method=key,n_repeats=n,iteration=i,auc=roc_auc_score(y,pr)))
            pd.DataFrame(rows).to_csv(OUT/'repeat_auc_iterations.csv',index=False)
            print(key,'n',n,'AUC',np.mean([roc_auc_score(y,pr) for pr in probs]),flush=True)
    frame=pd.DataFrame(rows)
    summary=frame.groupby(['method','n_repeats']).auc.agg(mean='mean',sd='std',minimum='min',maximum='max',mc_iterations='count').reset_index()
    summary.to_csv(OUT/'repeat_auc_summary.csv',index=False)
    (OUT/'protocol.json').write_text(json.dumps(dict(mc_iterations=MC,seed=SEED,counts=COUNTS,subjects=N,class_counts=np.bincount(y).tolist(),evaluation='binary subject-level pooled OOF AUC; fixed full-mean trained nested 5x5 GroupKFold models; sampled test means; no replacement',qc='Retrospective selection using full subject repeat set before sampling',folds='Original pipeline splits retained; legacy group ordering differs',uncertainty='Monte Carlo SD, not patient-bootstrap confidence interval'),indent=2))
    print('COMPLETE',flush=True)

if __name__=='__main__':main()
