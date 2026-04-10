#!/usr/bin/env python3
"""Backfill experiment_registry.json from manifest.json (Phase A~V)."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MANIFEST_PATH = PROJECT_ROOT / "models" / "manifest.json"
REGISTRY_PATH = PROJECT_ROOT / "logs" / "experiment_registry.json"


def main():
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        manifest = json.load(f)

    experiments = []
    for m in manifest["models"]:
        phase = m.get("phase", "")
        name = m.get("name", "")
        cancer_types = m.get("cancer_types", [])
        nc = len([c for c in cancer_types if c not in ("NOR", "DIA", "HBP", "H.D.")])

        # Infer variable from notes/name
        name_lower = name.lower()
        if "medoid" in name_lower or "all-spectra" in name_lower or "aggregation" in name_lower:
            variable = "aggregation"
        elif "normalization" in name_lower or "norm" in name_lower:
            variable = "normalization"
        elif "sex" in name_lower or "constraint" in name_lower:
            variable = "constraint"
        elif "blc" in name_lower or "bre" in name_lower or "added" in name_lower:
            variable = "cancer_set"
        elif "confound" in name_lower:
            variable = "confounding"
        elif "fusion" in name_lower or "multimodal" in name_lower:
            variable = "fusion"
        elif "resnet" in name_lower or "deep" in name_lower or "ensemble" in name_lower:
            variable = "model_type"
        elif "fixed_grid" in name_lower or "grid" in name_lower:
            variable = "preprocessing"
        elif "held-out" in name_lower or "test" in name_lower:
            variable = "evaluation"
        else:
            variable = "baseline"

        metrics = m.get("metrics", {})
        det_auc = metrics.get("det_auc", metrics.get("s1_auc", "N/A"))
        id_f1 = metrics.get("id_f1_macro", metrics.get("s2_f1_macro", "N/A"))
        result_summary = f"Det AUC {det_auc}, Id F1 {id_f1}"

        entry = {
            "name": f"{nc}c_{phase}_{name.replace(' ', '_')[:40]}",
            "phase": phase,
            "date": m.get("date", "unknown"),
            "hypothesis": m.get("notes", name),
            "variable": variable,
            "baseline": None,
            "cancer_types": cancer_types,
            "non_cancer_groups": m.get("non_cancer_groups", []),
            "aggregation": m.get("aggregation", "unknown"),
            "models": [m.get("model_type", "unknown")],
            "n_samples": m.get("n_samples", 0),
            "tags": ["backfill", f"phase-{phase}"],
            "result_summary": result_summary,
            "artifacts_dir": m.get("artifacts", [""])[0] if m.get("artifacts") else "",
        }
        experiments.append(entry)

    registry = {"experiments": experiments}
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False, default=str)

    print(f"✓ {len(experiments)}개 실험 → {REGISTRY_PATH}")
    for e in experiments:
        print(f"  Phase {e['phase']:5s} | {e['variable']:15s} | {e['result_summary']}")


if __name__ == "__main__":
    main()
