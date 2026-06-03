"""sers analyze — Analysis & visualization commands."""

import click

from ._run import PROJECT_ROOT, run_script


LEGACY_EMBED_SCRIPT = "scripts/legacy/analysis/tsne_groups.py"


def _run_legacy_embed(args):
    """Run the historical embedding helper when it exists."""
    if not (PROJECT_ROOT / LEGACY_EMBED_SCRIPT).exists():
        raise click.ClickException(
            "Legacy embedding helper is not present in this checkout. "
            "Active analysis scripts are under scripts/analysis/stk_v2 and "
            "scripts/analysis/calibration."
        )
    run_script(LEGACY_EMBED_SCRIPT, args)


@click.group(invoke_without_command=True)
@click.pass_context
def analyze(ctx):
    """Analysis & visualization tools.

    \b
    Subcommands:
        sers analyze embed          Embedding visualization (t-SNE, UMAP, PCA)
        sers analyze tsne           t-SNE (shortcut for embed --method tsne)
        sers analyze umap           UMAP (shortcut for embed --method umap)
        sers analyze pca            PCA  (shortcut for embed --method pca)
        sers analyze explore        Data exploration
        sers analyze cross-inst     Cross-instrument analysis
        sers analyze equipment      Equipment QC analysis
    """
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


# ── Shared options for embedding commands ──

_embed_options = [
    click.option("--groups", "-g", multiple=True, required=True,
                 help="Groups to include (repeat for each)."),
    click.option("--input", "-i", default=None,
                 help="Processed spectra CSV [results/processed_spectra.csv]."),
    click.option("--output", "-o", default=None,
                 help="Output path (auto-generated if omitted)."),
    click.option("--dim", type=click.Choice(["2", "3"]), default="2",
                 show_default=True, help="Dimensions."),
    click.option("--aggregate", "-a",
                 type=click.Choice(["none", "mean", "medoid"]),
                 default="none", show_default=True,
                 help="Aggregation per patient."),
    click.option("--pca-init", type=int, default=50, show_default=True,
                 help="PCA pre-reduction dims (0 to skip)."),
    click.option("--interactive", is_flag=True, default=False,
                 help="Interactive HTML (3D only, requires plotly)."),
]


def _add_embed_options(func):
    for option in reversed(_embed_options):
        func = option(func)
    return func


def _build_embed_args(method, groups, input, output, dim, aggregate,
                      pca_init, interactive, **extra):
    """Build CLI args for the legacy embedding helper."""
    args = ["--method", method, "--groups"] + list(groups)
    args += ["--dim", dim]
    args += ["--aggregate", aggregate]
    args += ["--pca-init", str(pca_init)]
    if input:
        args += ["--input", input]
    if output:
        args += ["--output", output]
    if interactive:
        args.append("--interactive")
    for flag, val in extra.items():
        if val is not None:
            args += [f"--{flag.replace('_', '-')}", str(val)]
    return args


# ── Full embed command (all methods) ──

@analyze.command("embed")
@_add_embed_options
@click.option("--method", "-m",
              type=click.Choice(["tsne", "umap", "pca"]),
              default="tsne", show_default=True, help="Embedding method.")
@click.option("--perplexity", type=int, default=None, help="t-SNE perplexity [30].")
@click.option("--n-neighbors", type=int, default=None, help="UMAP n_neighbors [15].")
@click.option("--min-dist", type=float, default=None, help="UMAP min_dist [0.1].")
def embed(method, perplexity, n_neighbors, min_dist, **kwargs):
    """Embedding visualization (t-SNE, UMAP, PCA).

    \b
    Examples:
        sers analyze embed -g PRO -g NOR --method umap --dim 3 --interactive
        sers analyze embed -g PRO -g NOR --method pca
        sers analyze embed -g PRO -g NOR --method tsne --perplexity 50
    """
    args = _build_embed_args(method, **kwargs,
                             perplexity=perplexity,
                             n_neighbors=n_neighbors,
                             min_dist=min_dist)
    _run_legacy_embed(args)


# ── Shortcut commands ──

@analyze.command("tsne")
@_add_embed_options
@click.option("--perplexity", type=int, default=None, help="Perplexity [30].")
def analyze_tsne(perplexity, **kwargs):
    """t-SNE visualization for selected groups.

    \b
    Examples:
        sers analyze tsne -g YPAN -g SPAN -g CPAN -g NOR
        sers analyze tsne -g PRO -g NOR --dim 3 --interactive
        sers analyze tsne -g PRO -g NOR --perplexity 50
    """
    args = _build_embed_args("tsne", **kwargs, perplexity=perplexity)
    _run_legacy_embed(args)


@analyze.command("umap")
@_add_embed_options
@click.option("--n-neighbors", type=int, default=None, help="n_neighbors [15].")
@click.option("--min-dist", type=float, default=None, help="min_dist [0.1].")
def analyze_umap(n_neighbors, min_dist, **kwargs):
    """UMAP visualization for selected groups.

    \b
    Examples:
        sers analyze umap -g PRO -g NOR -g CRC
        sers analyze umap -g PRO -g NOR --dim 3 --interactive
        sers analyze umap -g PRO -g NOR --n-neighbors 30 --min-dist 0.3
    """
    args = _build_embed_args("umap", **kwargs,
                             n_neighbors=n_neighbors, min_dist=min_dist)
    _run_legacy_embed(args)


@analyze.command("pca")
@_add_embed_options
def analyze_pca(**kwargs):
    """PCA visualization for selected groups.

    \b
    Examples:
        sers analyze pca -g PRO -g NOR -g CRC
        sers analyze pca -g PRO -g NOR --dim 3 --interactive
    """
    args = _build_embed_args("pca", **kwargs)
    _run_legacy_embed(args)


# ── Other analysis commands ──

@analyze.command("explore")
def analyze_explore():
    """Run data exploration analysis."""
    run_script("scripts/legacy/analysis/explore_data.py")


@analyze.command("cross-inst")
def analyze_cross_inst():
    """Cross-instrument analysis."""
    run_script("scripts/legacy/analysis/cross_instrument_analysis.py")


@analyze.command("equipment")
def analyze_equipment():
    """Equipment QC analysis."""
    run_script("scripts/legacy/analysis/analyze_equipment.py")
