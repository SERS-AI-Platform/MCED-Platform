"""Publication figures from actual saved OOF arrays, with source audit."""
from pathlib import Path
import json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, balanced_accuracy_score, f1_score
from scipy.signal import find_peaks

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'three_condition_figures_20260828'
NAMES={'direct':'최소 전처리 후 진행', 'raw':'Peak 형태 보존 및 Denoising, baseline correction 방식', 'legacy':'기존 방식'}
LABEL_FONT=FontProperties(fname='/home/user/.fonts/NanumSquareR.ttf')
METHODS={
 'direct':'PS alignment + repeat QC + subject mean; no additional spectral preprocessing',
 'raw':'STK-V2: smoothing + rolling-minimum baseline correction + SNV; no CDAE or DWT',
 'legacy':'933-point interpolated subject mean + STK-V2; no Liu-specific implementation',
}
COLORS=['#2674AD','#7030A0','#D97520']
NAVY='#001F3C'; GRAY='#6B7280'
CMAP=LinearSegmentedColormap.from_list('clinical_blues',['#F4F6FA','#90BFD7','#0A306D'])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':15,
 'axes.labelsize':12,'axes.spines.top':False,'axes.spines.right':False,
 'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})
records=[]; manifest=[]

def frame(key,title,subtitle):
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.05,.944,title,fontsize=23,fontweight='bold',color=NAVY)
    fig.text(.05,.899,NAMES[key]+' | Re-evaluated 28 Aug 2026',fontproperties=LABEL_FONT,fontsize=14,color=GRAY)
    fig.text(.05,.860,subtitle,fontsize=12,color=GRAY)
    fig.add_artist(plt.Line2D([.05,.95],[.829,.829],transform=fig.transFigure,color=NAVY,lw=2))
    return fig

def finish(fig,key,stem,notes):
    fig.text(.05,.100,notes,fontsize=11,color=GRAY,linespacing=1.6)
    fig.text(.05,.043,'Source: '+key+'/predictions.npz | Research use only; hospital confounding may affect classification.',fontsize=10,color=GRAY)
    dest=OUT/key
    for ext in ['png','svg','pdf']:
        fig.savefig(dest/f'{stem}.{ext}',dpi=200)
    manifest.append({'condition':key,'display_name':NAMES[key],'figure':stem,'source':f'{key}/predictions.npz'})
    plt.close(fig)

def cmplot(ax,cm,labels):
    ax.imshow(cm,cmap=CMAP,vmin=0,vmax=max(1,cm.max()))
    for i in range(len(cm)):
        for j in range(len(cm)):
            ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=19,
                    color='white' if cm[i,j]>cm.max()*.58 else NAVY)
    ax.set_xticks(range(len(labels)),labels,rotation=25,ha='right')
    ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True'); ax.set_title('Confusion matrix',pad=14)

def bars(ax,names,values,color):
    ax.bar(range(len(names)),values,color=color,width=.6)
    for i,v in enumerate(values): ax.text(i,v+.025,f'{v:.4f}',ha='center',fontsize=12)
    ax.set_xticks(range(len(names)),names,rotation=25,ha='right')
    ax.set_ylim(0,1.1);ax.set_yticks(np.linspace(0,1,6));ax.set_ylabel('Score')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_title('Pooled OOF metrics',pad=14)

def performance(key,d,multi):
    y=d['y3'] if multi else d['y2']
    p=d['three'] if multi else d['binary']
    pred=p.argmax(axis=1) if multi else d['pred']
    labels=['Control','PDC','Cancer'] if multi else ['Non-cancer','Cancer']
    cm=confusion_matrix(y,pred,labels=np.arange(len(labels)))
    bacc=balanced_accuracy_score(y,pred)
    aucs=[roc_auc_score(y==i,p[:,i]) for i in range(3)] if multi else [roc_auc_score(y,p)]
    auc=float(np.mean(aucs))
    f1=f1_score(y,pred,average='macro' if multi else 'binary')
    counts=', '.join(f'{name}: {int((y==i).sum())}' for i,name in enumerate(labels))
    fig=frame(key,'Three-class performance' if multi else 'Cancer Screening performance',f'Subject-level 5 x 5 nested GroupKFold | n={len(y)} | {counts}')
    gs=fig.add_gridspec(1,3,left=.07,right=.955,bottom=.28,top=.735,wspace=.36)
    axes=[fig.add_subplot(gs[0,i]) for i in range(3)]
    cmplot(axes[0],cm,labels)
    ax=axes[1]
    if multi:
        for i,(name,c,style) in enumerate(zip(labels,COLORS,['-','--','-.'])):
            x,z,_=roc_curve(y==i,p[:,i]);ax.plot(x,z,color=c,ls=style,lw=2,label=f'{name}: {aucs[i]:.4f}')
        ax.set_title(f'One-vs-rest ROC | Macro AUC {auc:.4f}',fontsize=13,pad=14)
    else:
        x,z,_=roc_curve(y,p);ax.plot(x,z,color='#391D8D',lw=2,label=f'AUC = {auc:.4f}')
        ax.set_title('Screening ROC',pad=14)
    ax.plot([0,1],[0,1],ls='--',color=GRAY,lw=1)
    ax.set(xlim=(0,1),ylim=(0,1.02),xlabel='False positive rate',ylabel='True positive rate')
    ax.legend(loc='lower right',fontsize=10,frameon=False)
    if multi:
        bars(axes[2],['Macro AUC','BAcc','Macro-F1'],[auc,bacc,f1],NAVY)
        policy='Three-class decision: argmax probability. AUC is pooled OOF macro OvR, not mean fold AUC.'
    else:
        tn,fp,fn,tp=cm.ravel()
        bars(axes[2],['AUC','BAcc','Sensitivity','Specificity'],[auc,bacc,tp/(tp+fn),tn/(tn+fp)],'#391D8D')
        t=d['thresholds']
        policy=f'Decision threshold: {t[0]:.2f} (fixed).' if np.ptp(t)==0 else f'Decision thresholds selected within outer training data: {t.min():.3f}-{t.max():.3f}.'
    stem='01_three_class_performance' if multi else '02_binary_performance'
    finish(fig,key,stem,METHODS[key]+'\n'+policy)
    rec=dict(condition=key,task='three_class' if multi else 'binary',n=len(y),auc=auc,bacc=bacc,f1=f1)
    if multi: rec.update(control_auc=aucs[0],pdc_auc=aucs[1],cancer_auc=aucs[2])
    records.append(rec)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(OUT/key/(stem+'_confusion.csv'))

def spectra(key,d,multi):
    y=d['y3'] if multi else d['y2']; X=d['plot_X']; grid=d['plot_grid']
    labels=['Control','PDC','Cancer'] if multi else ['Non-cancer','Cancer']
    colors=COLORS if multi else [COLORS[0],COLORS[2]]
    means=np.stack([X[y==i].mean(axis=0) for i in range(len(labels))])
    sem=np.stack([X[y==i].std(axis=0,ddof=1)/np.sqrt((y==i).sum()) for i in range(len(labels))])
    pooled=X.mean(axis=0)
    distance=max(1,int(round(20/np.median(np.diff(grid)))))
    peaks,_=find_peaks(pooled,distance=distance,prominence=max(1e-10,.03*np.ptp(pooled)))
    within=np.mean([X[y==i].var(axis=0,ddof=1) for i in range(len(labels))],axis=0)
    effect=means.var(axis=0)/(within+1e-12)
    selected=sorted(peaks[np.argsort(effect[peaks])[-5:]]) if len(peaks) else []
    fig=frame(key,'Three-class mean spectra and exploratory peaks' if multi else 'Screening mean spectra and exploratory peaks',METHODS[key])
    ax=fig.add_axes([.09,.255,.86,.50])
    for i,(name,c,style) in enumerate(zip(labels,colors,['-','--','-.'])):
        ax.plot(grid,means[i],color=c,ls=style,lw=1.8,label=f'{name} (n={int((y==i).sum())})')
        ax.fill_between(grid,means[i]-sem[i],means[i]+sem[i],color=c,alpha=.11,lw=0)
        ax.scatter(grid[selected],means[i,selected],s=32,color=c,edgecolor=NAVY,lw=.6,zorder=4)
    for k,j in enumerate(selected):
        ax.axvspan(grid[j]-5,grid[j]+5,color='#B7791F',alpha=.12,lw=0)
        ax.text(grid[j],1.025+(k%2)*.038,f'{grid[j]:.0f}',transform=ax.get_xaxis_transform(),ha='center',fontsize=10,color=GRAY)
    ax.set_xlabel('Raman shift (cm$^{-1}$)')
    ax.set_ylabel('Raw intensity (a.u.)' if key=='direct' else 'STK-V2 ch0 intensity (SNV units)')
    ax.grid(alpha=.16);ax.legend(loc='upper right',fontsize=11,frameon=False)
    ax.set_xlim(grid[0],grid[-1])
    stem='04_three_class_peaks' if multi else '03_binary_peaks'
    notes='Lines: unweighted mean across subjects; shading: +/-1 SEM across subjects, not measurement noise.\nMarked peaks: up to five pooled local maxima ranked by between-group / within-group variance; descriptive, not validated biomarkers.'
    finish(fig,key,stem,notes)
    table={'wavenumber_cm1':grid}
    for i,name in enumerate(labels):table[name+'_mean']=means[i];table[name+'_sem']=sem[i]
    pd.DataFrame(table).to_csv(OUT/key/(stem+'_source.csv'),index=False)
    pd.DataFrame({'wavenumber_cm1':grid[selected],'descriptive_variance_ratio':effect[selected]}).to_csv(OUT/key/(stem+'_selected_peaks.csv'),index=False)

for key in NAMES:
    source=OUT/key/'predictions.npz'
    if not source.exists():continue
    d=dict(np.load(source,allow_pickle=False))
    assert np.isfinite(d['binary']).all() and np.isfinite(d['three']).all()
    assert np.allclose(d['three'].sum(axis=1),1)
    assert len(d['y3'])==len(d['plot_X'])==113
    performance(key,d,True);performance(key,d,False)
    spectra(key,d,False);spectra(key,d,True)
pd.DataFrame(records).to_csv(OUT/'verified_metrics.csv',index=False)
(OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(pd.DataFrame(records).to_string(index=False))
