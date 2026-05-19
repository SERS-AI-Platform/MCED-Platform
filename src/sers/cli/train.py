"""sers train — Model training commands."""

import click

from ._run import run_script


# ---------- sers train <model> ----------

@click.group(invoke_without_command=True)
@click.pass_context
def train(ctx):
    """Train a model.

    \b
    Subcommands:
        sers train resnet18      ResNet18-1D (default)
        sers train lr            Logistic Regression (SAGA/OvR)
        sers train rf            Random Forest
        sers train xgboost       XGBoost
        sers train cnn1d         1D CNN
        sers train stacking      Stacking ensemble
        sers train multichannel  Multi-channel model
        sers train tvt           Train/Val/Test split experiment
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── Shared options for single-model training ──

_common_options = [
    click.option("--input", "-i", default=None,
                 help="Input spectra CSV [results/processed_spectra.csv]."),
    click.option("--output", "-o", default=None,
                 help="Output directory [results/training]."),
    click.option("--config", "-c", default=None, help="Config YAML."),
    click.option("--experiment", default=None, help="Experiment name."),
    click.option("--version", "version_tag", default=None, help="Version tag."),
    click.option("--cancer-types", multiple=True,
                 help="Cancer types (repeat for multiple)."),
    click.option("--non-cancer-groups", multiple=True,
                 help="Non-cancer groups (repeat for multiple)."),
    click.option("--aggregate", "-a",
                 type=click.Choice(["medoid", "mean", "none"]),
                 default=None, help="Aggregation method."),
    click.option("--n-splits", type=int, default=None, help="CV splits."),
    click.option("--epochs", type=int, default=None, help="Training epochs."),
    click.option("--batch-size", type=int, default=None, help="Batch size."),
    click.option("--lr", type=float, default=None, help="Learning rate."),
    click.option("--weight-decay", type=float, default=None, help="L2 reg."),
    click.option("--dropout-rate", type=float, default=None, help="Dropout."),
    click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]),
                 default=None, help="Device [auto]."),
    click.option("--no-amp", is_flag=True, default=False,
                 help="Disable mixed precision."),
    click.option("--no-mlflow", is_flag=True, default=False,
                 help="Disable MLflow logging."),
    click.option("--hypothesis", default=None, help="Hypothesis label."),
    click.option("--variable", default=None, help="Variable name."),
    click.option("--baseline", default=None, help="Baseline model."),
    click.option("--tags", multiple=True, help="Tags (repeat for multiple)."),
    click.option("--phase", default=None, help="Phase label."),
    click.option("--exclude-patients", default=None,
                 help="CSV with (group, sample_id) exclusions."),
]


def _add_common_options(func):
    for option in reversed(_common_options):
        func = option(func)
    return func


def _build_train_args(model_name: str, **kwargs) -> list[str]:
    """Build CLI args list for the legacy single-model trainer."""
    args = ["--model", model_name]

    simple_map = {
        "input": "--input",
        "output": "--output",
        "config": "--config",
        "experiment": "--experiment",
        "version_tag": "--version",
        "aggregate": "--aggregate",
        "n_splits": "--n-splits",
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "lr": "--lr",
        "weight_decay": "--weight-decay",
        "dropout_rate": "--dropout-rate",
        "device": "--device",
        "hypothesis": "--hypothesis",
        "variable": "--variable",
        "baseline": "--baseline",
        "phase": "--phase",
        "exclude_patients": "--exclude-patients",
    }
    for kwarg_key, flag in simple_map.items():
        val = kwargs.get(kwarg_key)
        if val is not None:
            args += [flag, str(val)]

    # Flags
    if kwargs.get("no_amp"):
        args.append("--no-amp")
    if kwargs.get("no_mlflow"):
        args.append("--no-mlflow")

    # Multi-value
    cancer_types = kwargs.get("cancer_types", ())
    if cancer_types:
        args += ["--cancer-types"] + list(cancer_types)
    non_cancer = kwargs.get("non_cancer_groups", ())
    if non_cancer:
        args += ["--non-cancer-groups"] + list(non_cancer)
    tags = kwargs.get("tags", ())
    if tags:
        args += ["--tags"] + list(tags)

    return args


# ── ResNet18 ──

@train.command("resnet18")
@_add_common_options
@click.option("--stage2-loss-weight", type=float, default=None)
@click.option("--head-hidden-dim", type=int, default=None)
@click.option("--resnet-channels", multiple=True, type=int,
              help="Channel dims (repeat for multiple).")
@click.option("--use-focal-loss", is_flag=True, default=False)
@click.option("--focal-gamma", type=float, default=None)
@click.option("--class-balance-beta", type=float, default=None)
def train_resnet18(stage2_loss_weight, head_hidden_dim, resnet_channels,
                   use_focal_loss, focal_gamma, class_balance_beta, **kwargs):
    """Train ResNet18-1D model.

    \b
    Examples:
        sers train resnet18
        sers train resnet18 --epochs 200 --lr 3e-4
        sers train resnet18 --use-focal-loss --focal-gamma 2.0
    """
    args = _build_train_args("resnet18", **kwargs)
    if stage2_loss_weight is not None:
        args += ["--stage2-loss-weight", str(stage2_loss_weight)]
    if head_hidden_dim is not None:
        args += ["--head-hidden-dim", str(head_hidden_dim)]
    if resnet_channels:
        args += ["--resnet-channels"] + [str(c) for c in resnet_channels]
    if use_focal_loss:
        args.append("--use-focal-loss")
    if focal_gamma is not None:
        args += ["--focal-gamma", str(focal_gamma)]
    if class_balance_beta is not None:
        args += ["--class-balance-beta", str(class_balance_beta)]
    run_script("models/legacy/scripts/train.py", args)


# ── Logistic Regression ──

@train.command("lr")
@_add_common_options
@click.option("--logreg-c", type=float, default=None, help="Regularization C.")
@click.option("--logreg-max-iter", type=int, default=None, help="Max iterations.")
def train_lr(logreg_c, logreg_max_iter, **kwargs):
    """Train Logistic Regression (SAGA/OvR).

    \b
    Examples:
        sers train lr
        sers train lr --logreg-c 0.1 --n-splits 5
    """
    args = _build_train_args("logistic_regression", **kwargs)
    if logreg_c is not None:
        args += ["--logreg-c", str(logreg_c)]
    if logreg_max_iter is not None:
        args += ["--logreg-max-iter", str(logreg_max_iter)]
    run_script("models/legacy/scripts/train.py", args)


# ── Random Forest ──

@train.command("rf")
@_add_common_options
@click.option("--rf-n-estimators", type=int, default=None, help="Number of trees.")
@click.option("--rf-max-depth", type=int, default=None, help="Max depth.")
@click.option("--rf-min-samples-leaf", type=int, default=None)
def train_rf(rf_n_estimators, rf_max_depth, rf_min_samples_leaf, **kwargs):
    """Train Random Forest.

    \b
    Examples:
        sers train rf
        sers train rf --rf-n-estimators 500 --rf-max-depth 10
    """
    args = _build_train_args("random_forest", **kwargs)
    if rf_n_estimators is not None:
        args += ["--rf-n-estimators", str(rf_n_estimators)]
    if rf_max_depth is not None:
        args += ["--rf-max-depth", str(rf_max_depth)]
    if rf_min_samples_leaf is not None:
        args += ["--rf-min-samples-leaf", str(rf_min_samples_leaf)]
    run_script("models/legacy/scripts/train.py", args)


# ── XGBoost ──

@train.command("xgboost")
@_add_common_options
@click.option("--xgb-n-estimators", type=int, default=None)
@click.option("--xgb-max-depth", type=int, default=None)
@click.option("--xgb-learning-rate", type=float, default=None)
@click.option("--xgb-subsample", type=float, default=None)
@click.option("--xgb-colsample-bytree", type=float, default=None)
@click.option("--xgb-reg-lambda", type=float, default=None)
@click.option("--xgb-min-child-weight", type=float, default=None)
def train_xgboost(xgb_n_estimators, xgb_max_depth, xgb_learning_rate,
                  xgb_subsample, xgb_colsample_bytree, xgb_reg_lambda,
                  xgb_min_child_weight, **kwargs):
    """Train XGBoost.

    \b
    Examples:
        sers train xgboost
        sers train xgboost --xgb-n-estimators 300 --xgb-max-depth 5
    """
    args = _build_train_args("xgboost", **kwargs)
    xgb_map = {
        "--xgb-n-estimators": xgb_n_estimators,
        "--xgb-max-depth": xgb_max_depth,
        "--xgb-learning-rate": xgb_learning_rate,
        "--xgb-subsample": xgb_subsample,
        "--xgb-colsample-bytree": xgb_colsample_bytree,
        "--xgb-reg-lambda": xgb_reg_lambda,
        "--xgb-min-child-weight": xgb_min_child_weight,
    }
    for flag, val in xgb_map.items():
        if val is not None:
            args += [flag, str(val)]
    run_script("models/legacy/scripts/train.py", args)


# ── CNN1D ──

@train.command("cnn1d")
@_add_common_options
def train_cnn1d(**kwargs):
    """Train 1D CNN.

    \b
    Examples:
        sers train cnn1d --epochs 100 --lr 1e-3
    """
    args = _build_train_args("cnn1d", **kwargs)
    run_script("models/legacy/scripts/train.py", args)


# ── Stacking Ensemble ──

@train.command("stacking")
@click.option("--dry-run", is_flag=True, default=False,
              help="Quick test mode (1 fold, 3 base, 2 meta).")
@click.option("--val-group", default=None,
              help="Inference on out-of-training group (e.g. SPAN).")
@click.option("--meta-learner", default=None, show_default=True,
              help="Meta-learner type [elasticnet].")
@click.option("--no-shap", is_flag=True, default=False, help="Skip SHAP.")
@click.option("--no-cm", is_flag=True, default=False, help="Skip confusion matrix.")
@click.option("--shap-samples", type=int, default=None, help="SHAP background [100].")
def train_stacking(dry_run, val_group, meta_learner, no_shap, no_cm, shap_samples):
    """Train stacking ensemble (nested CV, multiple base + meta learners).

    \b
    Examples:
        sers train stacking
        sers train stacking --dry-run
        sers train stacking --val-group SPAN --meta-learner elasticnet
    """
    args = []
    if dry_run:
        args.append("--dry-run")
    if val_group:
        args += ["--val-group", val_group]
    if meta_learner:
        args += ["--meta-learner", meta_learner]
    if no_shap:
        args.append("--no-shap")
    if no_cm:
        args.append("--no-cm")
    if shap_samples is not None:
        args += ["--shap-samples", str(shap_samples)]
    run_script("scripts/training/train_usersnet.py", args)


# ── Multichannel ──

@train.command("multichannel")
@click.option("--epochs", type=int, default=None, help="Epochs [150].")
@click.option("--lr", type=float, default=None, help="Learning rate [5e-4].")
@click.option("--batch-size", type=int, default=None, help="Batch size [32].")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default=None)
@click.option("--aggregate", type=click.Choice(["medoid", "mean", "none"]),
              default=None, help="Aggregation [none].")
@click.option("--cancer-types", multiple=True)
@click.option("--non-cancer-groups", multiple=True)
def train_multichannel(epochs, lr, batch_size, device, aggregate,
                       cancer_types, non_cancer_groups):
    """Train multi-channel model.

    \b
    Examples:
        sers train multichannel
        sers train multichannel --epochs 200 --lr 1e-4
    """
    args = []
    if epochs is not None:
        args += ["--epochs", str(epochs)]
    if lr is not None:
        args += ["--lr", str(lr)]
    if batch_size is not None:
        args += ["--batch-size", str(batch_size)]
    if device:
        args += ["--device", device]
    if aggregate:
        args += ["--aggregate", aggregate]
    if cancer_types:
        args += ["--cancer-types"] + list(cancer_types)
    if non_cancer_groups:
        args += ["--non-cancer-groups"] + list(non_cancer_groups)
    run_script("models/legacy/scripts/train_multichannel.py", args)


# ── Train/Val/Test ──

@train.command("tvt")
@click.option("--n-repeats", type=int, default=None, help="Random splits [5].")
@click.option("--seed", type=int, default=None, help="Random seed [42].")
@click.option("--exclude-patients", default=None, help="Exclusion CSV.")
def train_tvt(n_repeats, seed, exclude_patients):
    """Train/Val/Test held-out split experiment (60/20/20).

    \b
    Examples:
        sers train tvt
        sers train tvt --n-repeats 10 --seed 123
    """
    args = []
    if n_repeats is not None:
        args += ["--n-repeats", str(n_repeats)]
    if seed is not None:
        args += ["--seed", str(seed)]
    if exclude_patients:
        args += ["--exclude-patients", exclude_patients]
    run_script("models/legacy/scripts/run_train_val_test.py", args)


# ---------- sers benchmark ----------

@click.command()
@click.argument("models", nargs=-1, required=True,
                type=click.Choice(["resnet18", "lr", "rf", "xgboost", "cnn1d"]))
@click.option("--n-splits", type=int, default=None, help="CV splits [5].")
@click.option("--aggregate", "-a",
              type=click.Choice(["medoid", "mean", "none"]), default=None)
@click.option("--epochs", type=int, default=None)
@click.option("--no-mlflow", is_flag=True, default=False)
def benchmark(models, n_splits, aggregate, epochs, no_mlflow):
    """Benchmark multiple models side-by-side.

    \b
    Examples:
        sers benchmark resnet18 lr xgboost
        sers benchmark resnet18 lr rf --n-splits 10
    """
    # Map short names to train.py names
    name_map = {"lr": "logistic_regression", "rf": "random_forest"}
    full_names = [name_map.get(m, m) for m in models]

    args = ["--benchmark-models"] + full_names
    if n_splits is not None:
        args += ["--n-splits", str(n_splits)]
    if aggregate:
        args += ["--aggregate", aggregate]
    if epochs is not None:
        args += ["--epochs", str(epochs)]
    if no_mlflow:
        args.append("--no-mlflow")
    run_script("models/legacy/scripts/train.py", args)
