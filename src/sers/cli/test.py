"""sers test — Model evaluation and visualization."""

import click

from ._run import run_script


@click.command()
@click.option("--input", "-i", default=None,
              help="Model directory [models/results/_archive/resnet18_medoid_v1].")
@click.option("--processed-csv", default=None,
              help="Processed spectra CSV [results/processed_spectra.csv].")
@click.option("--device", type=click.Choice(["auto", "cuda", "cpu"]), default=None)
@click.option("--tsne-dim", type=click.Choice(["2", "3"]), default=None,
              help="t-SNE dimensions [3].")
@click.option("--no-shap", is_flag=True, default=False, help="Skip SHAP analysis.")
@click.option("--no-feature-selection", is_flag=True, default=False)
@click.option("--no-gradcam", is_flag=True, default=False, help="Skip GradCAM.")
@click.option("--top-k-features", type=int, default=None, help="Top features [30].")
@click.option("--gradcam-samples", type=int, default=None, help="GradCAM samples [128].")
@click.option("--shap-samples", type=int, default=None, help="SHAP background [100].")
@click.option("--shap-explain", type=int, default=None, help="SHAP explain samples [200].")
@click.option("--val-group", default=None, help="Out-of-training group for inference.")
@click.option("--aggregate", "-a",
              type=click.Choice(["medoid", "mean", "none"]), default=None)
def test(input, processed_csv, device, tsne_dim, no_shap, no_feature_selection,
         no_gradcam, top_k_features, gradcam_samples, shap_samples, shap_explain,
         val_group, aggregate):
    """Evaluate trained model with ROC, confusion matrix, SHAP, t-SNE.

    \b
    Examples:
        sers test
        sers test -i results/training --no-shap
        sers test --top-k-features 50 --tsne-dim 2
    """
    args = []
    simple = {
        "--input": input,
        "--processed-csv": processed_csv,
        "--device": device,
        "--tsne-dim": tsne_dim,
        "--top-k-features": top_k_features,
        "--gradcam-samples": gradcam_samples,
        "--shap-samples": shap_samples,
        "--shap-explain": shap_explain,
        "--val-group": val_group,
        "--aggregate": aggregate,
    }
    for flag, val in simple.items():
        if val is not None:
            args += [flag, str(val)]
    if no_shap:
        args.append("--no-shap")
    if no_feature_selection:
        args.append("--no-feature-selection")
    if no_gradcam:
        args.append("--no-gradcam")
    run_script("models/legacy/scripts/test.py", args)
