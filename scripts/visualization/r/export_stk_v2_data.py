#!/usr/bin/env python3
"""Flatten STK-V2 outputs into tidy CSVs for R consumption."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "results" / "training" / "stacking_optimization_v2"
OOF_DIR = ROOT / "results" / "weekend_experiments" / "stacking_optimization"
OUT = SRC / "r_data"
OUT.mkdir(exist_ok=True)

# 1. Per-base top features (long format)
peak = json.loads((SRC / "peak_interpretation.json").read_text())
rows = []
for name, info in peak["per_base_model"].items():
    for rank, feat in enumerate(info["top_features"], 1):
        rows.append({
            "base_model": name,
            "feature_space": info["feature_space"],
            "rank": rank,
            "feature_label": (str(feat.get("wavenumber", feat.get("feature")))
                              + (f" ({feat['view']})" if "view" in feat else "")),
            "wavenumber": feat.get("wavenumber"),
            "view": feat.get("view"),
            "feature_name": feat.get("feature"),
            "importance": feat.get("importance", feat.get("score")),
        })
pd.DataFrame(rows).to_csv(OUT / "per_base_top_features.csv", index=False)

# 2. Stacking aggregate top peaks
pd.DataFrame(peak["stacking_aggregate_top_peaks"]).to_csv(
    OUT / "stacking_aggregate_top_peaks.csv", index=False)

# 3. Meta weights × contributions
mw = peak["meta_weights_s1"]
co = peak["base_model_contributions"]
pd.DataFrame([{"base_model": k, "meta_weight": mw[k], "contribution": co[k]}
              for k in mw]).to_csv(OUT / "meta_weights_contributions.csv", index=False)

# 4. Meta-learner CV summary (for boxplot proxy with mean/std)
nested = json.loads((OOF_DIR / "best_ensemble_config.json").read_text())
pd.DataFrame(nested["meta_learner_summary"]).to_csv(
    OUT / "meta_learner_summary.csv", index=False)

# 5. Per-fold meta results (for actual boxplot)
ml_csv = OOF_DIR / "nested_cv_results.csv"
if ml_csv.exists():
    pd.read_csv(ml_csv).to_csv(OUT / "nested_cv_results.csv", index=False)

# 6. Single vs ensemble
sv = OOF_DIR / "single_vs_ensemble.csv"
if sv.exists():
    pd.read_csv(sv).to_csv(OUT / "single_vs_ensemble.csv", index=False)

print(f"wrote tidy CSVs to {OUT}")
for p in sorted(OUT.glob("*.csv")):
    print(f"  {p.name}")
