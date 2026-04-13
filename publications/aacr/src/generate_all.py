"""Generate all Nature-style paper figures."""

import subprocess
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

scripts = [
    "fig1_pipeline.py",
    "fig2_spectra_peaks.py",
    "fig3_roc_curves.py",
    "fig4_stage_performance.py",
    "fig5_model_comparison.py",
]

for script in scripts:
    path = os.path.join(SCRIPT_DIR, script)
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print(f"{'='*60}")
    result = subprocess.run([sys.executable, path], cwd=SCRIPT_DIR)
    if result.returncode != 0:
        print(f"ERROR: {script} failed with code {result.returncode}")
    else:
        print(f"OK: {script}")

print(f"\n{'='*60}")
print("All figures generated. Check: figures/paper/output/")
print(f"{'='*60}")
