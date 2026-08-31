from pathlib import Path
import sys,json
import numpy as np
import openpyxl
ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'publications/전향검체/보라매병원/src')]
import three_cohort_spectra as t
OUT=ROOT/'results/three_cohort_spectra_ci_20260828_v1'
OUT.mkdir(exist_ok=True)
liq=t._boramae_cancer_ids(ROOT)
w=openpyxl.load_workbook(ROOT/'data/mapping/clinical_df.xlsx',read_only=True,data_only=True)
rows=w.active.iter_rows(values_only=True);h=next(rows);ix={str(v):i for i,v in enumerate(h) if v is not None}
cancer=sorted(str(row[ix['solum_label']]).replace('_',' ') for row in rows if row[ix['cohort_group']]=='prostate')
w.close()
print('mapping headers',list(ix),flush=True)
print('counts liquid, powder, overlapping numeric codes',len(liq),len(cancer),len(liq&{int(x.split()[-1]) for x in cancer}),flush=True)
grid=np.load(ROOT/'artifacts/usersnet/v1.0.0/common_grid.npy')
print('grid',grid[[0,-1]],len(grid),flush=True)
counts=[]
original=t._subject_means
def tracked(grouped,grid):
    counts.append([len(grouped[i]) for i in sorted(grouped)])
    return original(grouped,grid)
t._subject_means=tracked
if not (OUT/'subject_spectra.npz').exists():
    cohorts=t.load_three_cohort_series(ROOT,grid)
    d={'grid':grid,'liquid_codes':np.array(sorted(liq)), 'powder_codes':np.array([int(x.split()[-1]) for x in cancer])}
    for i,(name,c) in enumerate(zip(['retrospective','liquid','powder'],cohorts)):
        d[name+'_raw']=c.raw_spectra;d[name+'_processed']=c.processed_spectra;d[name+'_repeat_counts']=np.array(counts[i])
        print(name,c.raw_spectra.shape,'repeats',min(counts[i]),max(counts[i]),'raw mean',c.raw_spectra.mean(),flush=True)
    np.savez_compressed(OUT/'subject_spectra.npz',**d)
print('COMPLETE',flush=True)
