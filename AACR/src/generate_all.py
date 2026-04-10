"""Generate all AACR figures with unified styling."""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Step 0: Export JSON config for R scripts
print("Exporting figure_config.json...")
sys.path.insert(0, SCRIPT_DIR)
from nature_style import export_config_json
export_config_json()

# Step 1: Generate Python figures (order matters for dependencies)
scripts = [
    "table1_demographics_v7.py",   # Table 1: Demographics (latest version)
    "fig1_pipeline.py",             # Fig 1: Study pipeline schematic
    "fig_spectra_shap.py",          # Fig 2: SERS spectra + SHAP peaks
    "fig_shap_bars.py",             # Fig 3: Feature importance bars
    "fig_confusion_matrices.py",    # Fig 4: Confusion matrices (Stage 1+2)
    # "gen_fusion_predictions.py",  # Data: early fusion fold predictions (run separately if needed)
    "fig9_auc_curves.py",           # Fig 5: ROC curves (S1 benchmark + S2 per-cancer)
    "fig8_subgroup_benchmark.py",   # Fig 6: Per-cancer subgroup benchmark heatmap
    "fig_stage2_radar.py",          # Fig 7: Stage 2 radar chart (6 models × 5 cancer types)
    "fig_stage_sensitivity.py",     # Fig 8: Stage-stratified detection sensitivity
]

for script in scripts:
    path = os.path.join(SCRIPT_DIR, script)
    if not os.path.exists(path):
        print(f"SKIP: {script} (not found)")
        continue
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"ERROR: {script} failed with code {result.returncode}")
    else:
        print(f"OK: {script}")

print(f"\n{'='*60}")
print(f"All figures generated. Check: {os.path.join(os.path.dirname(SCRIPT_DIR), 'figures')}/")
print(f"{'='*60}")
