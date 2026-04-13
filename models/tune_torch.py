from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.train import save_benchmark_visualizations


DEFAULT_SEARCH_SPACES = {
    "xgboost": [
        {
            "name": "baseline_hist",
            "args": ["--xgb-n-estimators", "300", "--xgb-max-depth", "5", "--xgb-learning-rate", "0.05"],
        },
        {
            "name": "deeper_slower",
            "args": ["--xgb-n-estimators", "500", "--xgb-max-depth", "6", "--xgb-learning-rate", "0.03", "--xgb-subsample", "0.9", "--xgb-colsample-bytree", "0.9"],
        },
        {
            "name": "shallower_faster",
            "args": ["--xgb-n-estimators", "250", "--xgb-max-depth", "4", "--xgb-learning-rate", "0.07", "--xgb-min-child-weight", "2"],
        },
    ],
    "logistic_regression": [
        {
            "name": "baseline_balanced",
            "args": ["--logreg-c", "1.0", "--logreg-max-iter", "1000"],
        },
        {
            "name": "stronger_regularization",
            "args": ["--logreg-c", "0.5", "--logreg-max-iter", "1500"],
        },
        {
            "name": "weaker_regularization",
            "args": ["--logreg-c", "2.0", "--logreg-max-iter", "1500"],
        },
    ],
    "random_forest": [
        {
            "name": "baseline_large_forest",
            "args": ["--rf-n-estimators", "500"],
        },
        {
            "name": "deeper_forest",
            "args": ["--rf-n-estimators", "700", "--rf-max-depth", "30"],
        },
        {
            "name": "regularized_forest",
            "args": ["--rf-n-estimators", "400", "--rf-max-depth", "20", "--rf-min-samples-leaf", "2"],
        },
    ],
    "resnet18": [
        {
            "name": "baseline_like_v002",
            "args": ["--lr", "5e-4", "--batch-size", "32", "--weight-decay", "1e-3", "--dropout-rate", "0.5"],
        },
        {
            "name": "larger_batch_milder_reg",
            "args": ["--lr", "4e-4", "--batch-size", "64", "--weight-decay", "1.2e-3", "--dropout-rate", "0.45"],
        },
        {
            "name": "balanced_stage2",
            "args": ["--lr", "3e-4", "--batch-size", "64", "--weight-decay", "1.5e-3", "--dropout-rate", "0.5", "--stage2-loss-weight", "1.1"],
        },
    ],
    "cnn1d": [
        {
            "name": "higher_capacity_head",
            "args": ["--lr", "8e-4", "--batch-size", "64", "--dropout-rate", "0.3", "--head-hidden-dim", "128"],
        },
        {
            "name": "balanced_default",
            "args": ["--lr", "5e-4", "--batch-size", "32", "--dropout-rate", "0.4", "--head-hidden-dim", "128"],
        },
        {
            "name": "slower_bigger_batch",
            "args": ["--lr", "3e-4", "--batch-size", "64", "--dropout-rate", "0.3", "--head-hidden-dim", "96"],
        },
    ],
}


def parse_args():
    p = argparse.ArgumentParser(description="Model fine-tuning runner")
    p.add_argument("--input", "-i", default="results/processed_spectra.csv")
    p.add_argument("--output", "-o", default="models/results/02_tuning")
    p.add_argument(
        "--models",
        nargs="+",
        choices=["logistic_regression", "random_forest", "xgboost", "resnet18", "cnn1d"],
        default=["xgboost", "resnet18", "cnn1d"],
    )
    p.add_argument("--cancer-types", nargs="+", default=["PRO", "BRE", "OVA", "LUN", "CRC", "CPAN"])
    p.add_argument("--non-cancer-groups", nargs="+", default=["NOR", "DIA", "HBP", "H.D."])
    p.add_argument("--aggregate", default="none", choices=["medoid", "mean", "none"])
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--experiment-prefix", default="tuning")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--prefetch-factor", type=int, default=2)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args()


def load_trial_summary(output_root: Path, experiment_name: str, model_name: str, version_name: str):
    summary_path = output_root / experiment_name / model_name / version_name / "training_summary.json"
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    return summary


def build_trial_command(args, model_name: str, trial_name: str, experiment_name: str, extra_args: list[str]):
    cmd = [
        sys.executable,
        "models/train.py",
        "--model",
        model_name,
        "--aggregate",
        args.aggregate,
        "--input",
        args.input,
        "--output",
        args.output,
        "--experiment",
        experiment_name,
        "--version",
        trial_name,
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
        "--prefetch-factor",
        str(args.prefetch_factor),
        "--no-mlflow",
        "--cancer-types",
        *args.cancer_types,
        "--non-cancer-groups",
        *args.non_cancer_groups,
    ]
    if args.epochs is not None:
        cmd.extend(["--epochs", str(args.epochs)])
    if args.no_amp:
        cmd.append("--no-amp")
    cmd.extend(extra_args)
    return cmd


def main():
    args = parse_args()
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    plan_rows = []
    result_rows = []
    for model_name in args.models:
        trials = DEFAULT_SEARCH_SPACES[model_name]
        for idx, trial in enumerate(trials, start=1):
            trial_name = f"tune_{idx:02d}_{trial['name']}"
            experiment_name = f"{args.experiment_prefix}_{model_name}_{trial_name}"
            cmd = build_trial_command(args, model_name, trial_name, experiment_name, trial["args"])
            plan_rows.append({"model_name": model_name, "trial": trial_name, "command": " ".join(cmd)})
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
            summary = load_trial_summary(output_root, experiment_name, model_name, trial_name)
            row = {
                "experiment": experiment_name,
                "model_name": model_name,
                "model": summary["model"],
                "version": trial_name,
                "trial": trial_name,
                "output": summary["output_dir"],
                "aggregate": summary["aggregate"],
                "n_samples": summary["n_samples"],
                "n_features": summary["n_features"],
            }
            row.update(summary["metrics"])
            row["model_params_json"] = json.dumps(summary.get("model_params", {}), ensure_ascii=False)
            result_rows.append(row)

    plan_df = pd.DataFrame(plan_rows)
    result_df = pd.DataFrame(result_rows)
    plan_df.to_csv(output_root / "tuning_plan.csv", index=False)
    result_df.to_csv(output_root / "tuning_summary.csv", index=False)
    save_benchmark_visualizations(result_df, output_root)
    print(output_root / "tuning_summary.csv")


if __name__ == "__main__":
    raise SystemExit(main())
