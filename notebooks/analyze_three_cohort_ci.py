"""Source-based subject-level CIs and spectrum contrasts; no image digitization."""
from pathlib import Path
import sys,json,hashlib,shutil,zipfile
import numpy as np
import pandas as pd
import openpyxl
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'results/three_cohort_spectra_ci_20260828_v1'
D=dict(np.load(OUT/'subject_spectra.npz'));X=D['grid'];B=10000
KEYS=['retrospective','liquid','powder']
LABELS=['Retrospective specimens (n=91)','Boramae liquid reducing agent (n=41)','Boramae powder reducing agent (n=43)']
COLORS=['#4D4D4D','#D95F02','#2C7FB8'];STYLES=['-','--','-.']
def read_rows(path):
 w=openpyxl.load_workbook(path,read_only=True,data_only=True);it=w.active.iter_rows(values_only=True);h=next(it);rows=[dict(zip(h,r)) for r in it];w.close();return rows
def norm(v):return str(v).strip().replace('_',' ')
liquid={norm(r['solum_label']):r for r in read_rows(ROOT/'data/clinical_data/보라매 병원 임상정보.xlsx')}
powder={norm(r['solum_label']):r for r in read_rows(ROOT/'data/mapping/clinical_df.xlsx') if r['cohort_group']=='prostate'}
link={}
common=[k for k in powder if k in liquid and int(k.split()[-1]) in D['liquid_codes']]
assert len(common)==41
for field in ['sex','collection_date','cancer_type']:
 pairs=[(liquid[k].get(field),powder[k].get(field)) for k in common]
 valid=[(a,b) for a,b in pairs if a is not None and b is not None]
 link[field]={'nonmissing_pairs':len(valid),'exact_agreement':sum(norm(a)==norm(b) for a,b in valid)}
(OUT/'linkage_validation.json').write_text(json.dumps({'shared_specimen_labels':len(common),'clinical_field_agreement':link},indent=2))
print('LINKAGE',link,flush=True)
P=np.array([np.where(D['powder_codes']==k)[0][0] for k in D['liquid_codes']])
rng=np.random.default_rng(20260828)
boot={};summ=[];counts=[]
for k,label in zip(KEYS,LABELS):
 n=len(D[k+'_raw']);w=rng.multinomial(n,np.ones(n)/n,size=B)/n
 assert np.isfinite(D[k+'_raw']).all() and np.isfinite(D[k+'_processed']).all()
 counts.append(dict(cohort=k,subjects=n,minimum_repeats=int(D[k+'_repeat_counts'].min()),maximum_repeats=int(D[k+'_repeat_counts'].max())))
 for mode in ['raw','processed']:
  a=D[k+'_'+mode];bm=w@a;boot[k+'_'+mode]=bm
  low,high=np.quantile(bm,[.025,.975],axis=0)
  for j in range(len(X)):summ.append(dict(cohort=k,mode=mode,wavenumber=X[j],mean=a[:,j].mean(),ci_low=low[j],ci_high=high[j],n=n))
pd.DataFrame(counts).to_csv(OUT/'cohort_counts.csv',index=False)
S=pd.DataFrame(summ);S.to_csv(OUT/'mean_spectra_95ci.csv',index=False)
comparisons=[('liquid_minus_retrospective','liquid','retrospective',False),('powder_minus_retrospective','powder','retrospective',False),('powder_minus_liquid_paired','powder','liquid',True)]
regions=[('full_range',X.min(),X.max()),('background_1800_2000',1800,2000)]+[(f'band_{c}',c-10,c+10) for c in [745,935,1000,1445,1650,2100]]
point=[];features=[];globals_=[]
def integrate(a,mask):return np.trapezoid(a[:,mask],x=X[mask],axis=1)/(X[mask][-1]-X[mask][0])
for mode in ['raw','processed']:
 for name,ka,kb,paired in comparisons:
  a=D[ka+'_'+mode];b=D[kb+'_'+mode]
  if paired:
   a=a[P];delta=a-b;n=len(delta);w=rng.multinomial(n,np.ones(n)/n,size=B)/n;bd=w@delta
   test=stats.ttest_rel(a,b,axis=0)
  else:
   bd=boot[ka+'_'+mode]-boot[kb+'_'+mode];test=stats.ttest_ind(a,b,equal_var=False,axis=0)
  diff=a.mean(0)-b.mean(0);lo,hi=np.quantile(bd,[.025,.975],axis=0)
  for j in range(len(X)):point.append(dict(comparison=name,mode=mode,wavenumber=X[j],mean_difference=diff[j],ci_low=lo[j],ci_high=hi[j],t=test.statistic[j],p=test.pvalue[j],n_a=len(a),n_b=len(b),paired=paired))
  globals_.append(dict(comparison=name,mode=mode,global_bonferroni_p=min(1.,float(np.min(test.pvalue))*len(X))))
  for region,left,right in regions:
   # Overall SNV level is constrained by normalization; do not test it as a signal endpoint.
   if mode=='processed' and region=='full_range':continue
   mask=(X>=left)&(X<=right);va=integrate(a,mask);vb=integrate(b,mask)
   ft=stats.ttest_rel(va,vb) if paired else stats.ttest_ind(va,vb,equal_var=False)
   fb=integrate(bd,mask);fci=np.quantile(fb,[.025,.975])
   if paired: effect=(va-vb).mean()/(va-vb).std(ddof=1);etype='paired_dz'
   else:
    sd=np.sqrt(((len(va)-1)*va.var(ddof=1)+(len(vb)-1)*vb.var(ddof=1))/(len(va)+len(vb)-2));effect=(va.mean()-vb.mean())/sd*(1-3/(4*(len(va)+len(vb))-9));etype='Hedges_g'
   sensitivity=stats.wilcoxon(va-vb).pvalue if paired else stats.mannwhitneyu(va,vb,alternative='two-sided').pvalue
   features.append(dict(comparison=name,mode=mode,region=region,actual_low=X[mask][0],actual_high=X[mask][-1],mean_a=va.mean(),mean_b=vb.mean(),difference=va.mean()-vb.mean(),ci_low=fci[0],ci_high=fci[1],t=ft.statistic,p=ft.pvalue,effect_size=effect,effect_type=etype,sensitivity_rank_p=sensitivity,paired=paired,n_a=len(a),n_b=len(b)))
  print(mode,name,'computed',flush=True)
def holm(p):
 p=np.array(p);o=np.argsort(p);q=np.empty_like(p);q[o]=np.minimum(1,np.maximum.accumulate(p[o]*(len(p)-np.arange(len(p)))));return q
T=pd.DataFrame(point);T['fdr_q_all_5610']=stats.false_discovery_control(T.p.to_numpy());T.to_csv(OUT/'pointwise_difference_tests.csv',index=False)
F=pd.DataFrame(features);F['fdr_q_all_regions']=stats.false_discovery_control(F.p.to_numpy());F['rank_fdr_q']=stats.false_discovery_control(F.sensitivity_rank_p.to_numpy());F.to_csv(OUT/'regional_difference_tests.csv',index=False)
primary=F[(F['mode']=='raw')&(F.region=='full_range')].copy();primary['holm_p_3']=holm(primary.p);primary.to_csv(OUT/'primary_raw_mean_tests.csv',index=False)
G=pd.DataFrame(globals_);G['holm_p_6']=holm(G.global_bonferroni_p)
G['fdr_significant_grid_points']=[int(((T.comparison==r.comparison)&(T['mode']==r.mode)&(T.fdr_q_all_5610<.05)).sum()) for r in G.itertuples()]
G.to_csv(OUT/'global_spectrum_tests.csv',index=False)
plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(3,1,figsize=(16,12),sharex=True);fig.subplots_adjust(left=.075,right=.985,top=.93,bottom=.10,hspace=.26)
for ax,mode,ks,title in [(axes[0],'raw',KEYS,'A  Raw spectra: all cohorts'),(axes[1],'raw',KEYS[:2],'B  Raw spectra: expanded view of the lower-intensity cohorts'),(axes[2],'processed',KEYS,'C  Processed spectra: smoothing + baseline correction + SNV')]:
 for k in ks:
  i=KEYS.index(k);s=S[(S.cohort==k)&(S['mode']==mode)]
  ax.fill_between(X,s.ci_low,s.ci_high,color=COLORS[i],alpha=.24,linewidth=0)
  ax.plot(X,s['mean'],color=COLORS[i],ls=STYLES[i],lw=1.7,label=LABELS[i])
 ax.set_title(title,loc='left',fontsize=14);ax.set_ylabel('Raw intensity (a.u.)' if mode=='raw' else 'SNV intensity');ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right',fontsize=10,frameon=False)
axes[-1].set_xlabel('Raman shift (cm$^{-1}$)')
fig.suptitle('Prostate cancer spectra: subject means and pointwise 95% bootstrap CI',fontsize=19,x=.075,ha='left')
fig.text(.075,.033,'10,000 subject-bootstrap resamples; 91 / 41 / 43 specimens. Repeats averaged within subject (5 / 5 / 121).\nPointwise intervals are not simultaneous bands. Acquisition/batch and hospital differences are not isolated reagent effects.',fontsize=10)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'three_cohort_mean_95ci.{ext}',dpi=180)
plt.close(fig)
fig,axes=plt.subplots(3,2,figsize=(16,11),sharex=True);fig.subplots_adjust(left=.075,right=.98,top=.91,bottom=.11,hspace=.35,wspace=.25)
for i,(name,_,_,paired) in enumerate(comparisons):
 for j,mode in enumerate(['raw','processed']):
  ax=axes[i,j];v=T[(T.comparison==name)&(T['mode']==mode)];ax.fill_between(X,v.ci_low,v.ci_high,color='#0A306D',alpha=.22);ax.plot(X,v.mean_difference,color='#0A306D',lw=1.3);ax.axhline(0,color='#333333',ls='--',lw=.9)
  sig=v.fdr_q_all_5610.to_numpy()<.05;ax.scatter(X[sig],np.full(sig.sum(),.025),s=2,color='#B7791F',transform=ax.get_xaxis_transform())
  ax.set_title(name.replace('_',' ')+' | '+mode,loc='left',fontsize=11);ax.set_ylabel('Difference (a.u.)' if mode=='raw' else 'Difference (SNV)');ax.grid(axis='y',alpha=.2)
for ax in axes[-1]:ax.set_xlabel('Raman shift (cm$^{-1}$)')
fig.suptitle('Between-cohort differences with pointwise 95% bootstrap CI',fontsize=19)
fig.text(.075,.035,'Liquid/powder: 41 shared specimen labels, paired analysis. Retrospective comparisons: independent cohorts.\nGold ticks: BH-FDR q<0.05 across all 5,610 pointwise tests. Statistical differences do not establish reagent causality.',fontsize=10)
for ext in ['png','svg','pdf']:fig.savefig(OUT/f'cohort_differences_95ci.{ext}',dpi=180)
plt.close(fig)
checks={'finite_spectra':True,'counts':[len(D[k+'_raw']) for k in KEYS],'bootstrap_resamples':B,'seed':20260828,'paired_codes':len(P),'pointwise_tests':len(T),'regional_tests':len(F),'all_cis_ordered':bool((T.ci_low<=T.ci_high).all()),'p_valid':bool(T.p.between(0,1).all()),'original_ci':'all three groups had mean +/- 1.96 SEM; lower curves were visually compressed','source':'publications/전향검체/보라매병원/src/three_cohort_spectra.py'}
(OUT/'validation.json').write_text(json.dumps(checks,indent=2))
print('PRIMARY\n',primary[['comparison','mean_a','mean_b','difference','ci_low','ci_high','p','holm_p_3']].to_string(index=False));print('GLOBAL\n',G.to_string(index=False));print('DONE',flush=True)
