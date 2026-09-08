"""Re-evaluate three real pipelines; never reconstruct ROC from summary AUC.

Run with the existing SERS analysis environment. All outputs are isolated.
"""
from pathlib import Path
import os
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'three_condition_figures_20260828'
OUT.mkdir(exist_ok=True)
os.environ['AECD_PATENT_DWT_OUTDIR'] = str(OUT / '_dwt_import')
os.environ['AECD_MEAN_SPECTRUM_OUTDIR'] = str(OUT / '_mean_import')
import sys, importlib.util, json, warnings
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)
import aecd_api_model_patent_dwt_clinical_performance as dwt
import aecd_api_model_3class_engine as multi
import train_stkv2_stacking as stk
spec = importlib.util.spec_from_file_location('mean_pipeline', ROOT.parent / 'scripts/analysis/aecd_api_model_mean_spectrum_clinical_performance.py')
mean = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mean
spec.loader.exec_module(mean)

def limited(factory):
    def make():
        model = factory()
        if 'n_jobs' in model.get_params(deep=False):
            model.set_params(n_jobs=4)
        return model
    return make
original_stk = stk.make_base_models
stk.make_base_models = lambda: [(n,v,limited(f)) for n,v,f in original_stk()]
original_direct = mean.make_direct_base_models
mean.make_direct_base_models = lambda: [(n,limited(f)) for n,f in original_direct()]

def prepare():
    cache = OUT / '_inputs.npz'
    if cache.exists():
        return dict(np.load(cache, allow_pickle=False))
    print('Loading API spectra', flush=True)
    items, _, _ = dwt.load_api_spectra()
    grid, _, _ = dwt.construct_common_grid(items)
    keys, labels, reps, _ = dwt.group_and_align_subjects(items, grid)
    raw = np.vstack([r.mean(axis=0) for r in reps])
    from collections import defaultdict
    grouped = defaultdict(list)
    for item in items:
        grouped[str(item['subject_key'])].append(item)
    legacy_grid = np.linspace(400.,2200.,933)
    legacy = np.vstack([np.mean([np.interp(legacy_grid, r['wavenumber'], r['intensities']) for r in grouped[k]],axis=0) for k in keys])
    # Use the exported PS-aligned/QC mean dataset, avoiding a new DB lookup.
    direct_frame = pd.read_csv(ROOT/'aecd_api_model_mean_spectrum_outputs/mean_representative_spectra.csv')
    direct = direct_frame.filter(regex='^wn_').to_numpy(dtype=float)
    direct_y3 = np.asarray([multi.CLASS_INDEX[x] for x in direct_frame['label']])
    y3 = np.asarray([multi.CLASS_INDEX[x] for x in labels])
    # Keep original group sort ordering without exporting real subject keys.
    legacy_groups = np.unique(np.asarray(keys), return_inverse=True)[1]
    data = dict(grid=grid, legacy_grid=legacy_grid, raw=raw, legacy=legacy,
                direct=direct, direct_y3=direct_y3, y3=y3, legacy_groups=legacy_groups)
    np.savez_compressed(cache, **data)
    print('Input cohort:', len(y3), np.bincount(y3).tolist(), flush=True)
    return data

data = prepare()
y3 = data['y3']; y2 = (y3==2).astype(int); groups=np.arange(len(y3))
for key in ['direct', 'raw', 'legacy']:
    y3=data['direct_y3'] if key=='direct' else data['y3']
    y2=(y3==2).astype(int)
    dest=OUT/key; dest.mkdir(exist_ok=True)
    if (dest/'predictions.npz').exists():
        print(key, 'cached', flush=True); continue
    print('Preparing',key,flush=True)
    grid=data['legacy_grid'] if key=='legacy' else data['grid']
    X=data[key]
    if key=='direct':
        views={'raw':X}; plot_X=X; plot_grid=grid
        specs=[(n,'raw',f) for n,f in mean.make_direct_base_models()]
        binary,pred,thresholds,folds=mean.run_nested_direct_stacking(X,y2,groups)
    else:
        views, channels=dwt.build_stkv2_views(X,grid)
        plot_X=channels[:,0,:]; plot_grid=dwt.STK_TARGET_GRID
        specs=stk.make_base_models()
        binary, folds=stk.run_nested_cv(views,y2,data['legacy_groups'] if key=='legacy' else groups,seed=0)
        thresholds=np.full(len(y2),0.4); pred=(binary>=thresholds).astype(int)
    print(key, 'binary complete; starting 3-class',flush=True)
    three, three_folds=multi.run_nested_multiclass(views,y3,groups,specs)
    np.savez_compressed(dest/'predictions.npz', y2=y2,y3=y3,binary=binary,pred=pred,
                        thresholds=thresholds,three=three,plot_X=plot_X,plot_grid=plot_grid)
    pd.DataFrame(dict(subject_index=np.arange(len(y2)),y_binary=y2,p_binary=binary,
                      predicted_binary=pred,threshold=thresholds,y_three=y3,
                      p_control=three[:,0],p_pdc=three[:,1],p_cancer=three[:,2])).to_csv(dest/'oof_predictions.csv',index=False)
    pd.DataFrame(three_folds).to_csv(dest/'three_class_folds.csv',index=False)
    print(key,'SAVED',flush=True)
print('All three conditions complete',flush=True)
