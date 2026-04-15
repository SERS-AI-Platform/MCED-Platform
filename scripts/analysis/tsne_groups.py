"""Dimensionality reduction visualization for SERS groups.

Supports t-SNE, UMAP, PCA.

Usage:
    python scripts/analysis/tsne_groups.py --groups YPAN SPAN CPAN NOR --method tsne
    python scripts/analysis/tsne_groups.py --groups PRO CRC NOR --method umap --dim 3
    python scripts/analysis/tsne_groups.py --groups PRO CRC NOR --method pca
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Register Korean font (Noto Sans CJK)
_KO_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _KO_FONT.exists():
    fm.fontManager.addfont(str(_KO_FONT))
    plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

# Load colors and categories from config/config.yaml
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"
with open(_CONFIG_PATH) as f:
    _config = yaml.safe_load(f)

_display = _config.get("display", {})
GROUP_COLORS = _display.get("group_colors", {})
_category_map = _display.get("category_map", {})

CANCER_GROUPS = {g for g, cat in _category_map.items() if cat == "cancer"}
NON_CANCER_GROUPS = {g for g, cat in _category_map.items()
                     if cat in ("non_cancer", "control")}
AMBIGUOUS_GROUPS = set(GROUP_COLORS.keys()) - CANCER_GROUPS - NON_CANCER_GROUPS


# ═══════════════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_filter(csv_path: str, groups: list[str],
                    aggregate: str = "none"):
    """Load processed spectra and filter to selected groups."""
    df = pd.read_csv(csv_path)
    feature_cols = [c for c in df.columns if c.startswith("x_")]

    available = set(df["group"].unique())
    missing = set(groups) - available
    if missing:
        print(f"Warning: groups not in data: {missing}")
        groups = [g for g in groups if g in available]

    df = df[df["group"].isin(groups)].copy()
    print(f"Loaded {len(df)} spectra from {len(groups)} groups")

    if aggregate == "mean":
        df = df.groupby(["group", "sample_id"])[feature_cols].mean().reset_index()
        print(f"  After mean aggregation: {len(df)} samples")
    elif aggregate == "medoid":
        rows = []
        for (grp, sid), sub in df.groupby(["group", "sample_id"]):
            X = sub[feature_cols].values
            if len(X) == 1:
                rows.append(sub.iloc[0])
                continue
            center = X.mean(axis=0)
            dists = np.linalg.norm(X - center, axis=1)
            rows.append(sub.iloc[dists.argmin()])
        df = pd.DataFrame(rows)
        print(f"  After medoid aggregation: {len(df)} samples")

    X = df[feature_cols].values
    group_labels = df["group"].values
    sample_ids = df["sample_id"].values
    return X, group_labels, sample_ids


def classify_binary(groups: np.ndarray) -> np.ndarray:
    """Assign binary labels: Cancer=1, Non-cancer=0, Ambiguous=-1."""
    labels = np.full(len(groups), -1, dtype=int)
    for i, g in enumerate(groups):
        if g in CANCER_GROUPS:
            labels[i] = 1
        elif g in NON_CANCER_GROUPS:
            labels[i] = 0
    return labels


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding methods
# ═══════════════════════════════════════════════════════════════════════════════

def run_embedding(X: np.ndarray, method: str, dim: int,
                  perplexity: int, pca_init: int,
                  n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    """Run dimensionality reduction (t-SNE, UMAP, or PCA)."""
    n = len(X)

    # PCA pre-reduction for t-SNE / UMAP
    if method != "pca" and pca_init > 0 and X.shape[1] > pca_init:
        print(f"  PCA pre-reduction: {X.shape[1]} → {pca_init}")
        X = PCA(n_components=pca_init, random_state=42).fit_transform(X)

    if method == "tsne":
        perp = min(perplexity, n - 1)
        print(f"  Running t-SNE (dim={dim}, perplexity={perp}, n={n})...")
        coords = TSNE(
            n_components=dim, random_state=42, perplexity=perp,
            max_iter=2000, learning_rate="auto", init="pca",
        ).fit_transform(X)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            print("Error: umap-learn not installed. Run: pip install umap-learn")
            sys.exit(1)
        nn = min(n_neighbors, n - 1)
        print(f"  Running UMAP (dim={dim}, n_neighbors={nn}, "
              f"min_dist={min_dist}, n={n})...")
        coords = umap.UMAP(
            n_components=dim, random_state=42,
            n_neighbors=nn, min_dist=min_dist,
        ).fit_transform(X)

    elif method == "pca":
        print(f"  Running PCA (dim={dim}, n={n})...")
        coords = PCA(n_components=dim, random_state=42).fit_transform(X)

    else:
        raise ValueError(f"Unknown method: {method}")

    return coords


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _count_patients(groups, sample_ids, mask):
    """Count unique patients (group + sample_id pair) in a boolean mask."""
    return len(set(zip(groups[mask], sample_ids[mask])))


def _method_label(method: str) -> str:
    return {"tsne": "t-SNE", "umap": "UMAP", "pca": "PCA"}[method]


def compute_cluster_scores(coords, groups, binary_labels):
    """Compute clustering quality metrics on the embedding."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

    results = {}

    # Binary (Cancer vs Non-cancer, excluding ambiguous=-1)
    mask = binary_labels >= 0
    if mask.sum() > 0 and len(set(binary_labels[mask])) > 1:
        results["binary"] = {
            "silhouette": silhouette_score(coords[mask], binary_labels[mask]),
            "calinski_harabasz": calinski_harabasz_score(coords[mask], binary_labels[mask]),
            "davies_bouldin": davies_bouldin_score(coords[mask], binary_labels[mask]),
        }

    # Per-group
    if len(set(groups)) > 1:
        results["groups"] = {
            "silhouette": silhouette_score(coords, groups),
            "calinski_harabasz": calinski_harabasz_score(coords, groups),
            "davies_bouldin": davies_bouldin_score(coords, groups),
        }

    return results


def print_scores(scores, method):
    """Pretty-print cluster quality scores."""
    label = _method_label(method)
    print(f"\n{'='*60}")
    print(f"  {label} Cluster Quality Scores")
    print(f"{'='*60}")
    for level, metrics in scores.items():
        level_name = "Binary (Cancer vs Non-cancer)" if level == "binary" else "Groups (per disease)"
        print(f"\n  {level_name}:")
        sil = metrics['silhouette']
        ch = metrics['calinski_harabasz']
        db = metrics['davies_bouldin']
        print(f"    Silhouette Score    : {sil:+.4f}  (range [-1, 1], higher = better)")
        print(f"    Calinski-Harabasz   : {ch:.1f}  (higher = better)")
        print(f"    Davies-Bouldin      : {db:.4f}  (lower = better)")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 2D plot (static PNG)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_2d(coords, groups, binary_labels, output_path, title_suffix="",
            sample_ids=None, method="tsne"):
    """Generate 2D figure with binary + group panels."""
    label = _method_label(method)
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Panel 1: Binary
    ax = axes[0]
    for bl, c, name in [(0, "#3498db", "Non-cancer"),
                         (1, "#e74c3c", "Cancer"),
                         (-1, "#95a5a6", "Other")]:
        m = binary_labels == bl
        if m.sum() == 0:
            continue
        n_pat = _count_patients(groups, sample_ids, m) if sample_ids is not None else m.sum()
        ax.scatter(coords[m, 0], coords[m, 1], c=c, s=20, alpha=0.6,
                   edgecolors="white", linewidth=0.3,
                   label=f"{name} (n={n_pat})")
    ax.set_title(f"{label} — Binary{title_suffix}", fontsize=14)
    ax.legend(fontsize=20)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel(f"{label} 1")
    ax.set_ylabel(f"{label} 2")

    # Panel 2: All groups
    ax = axes[1]
    for g in sorted(set(groups)):
        m = groups == g
        n_pat = _count_patients(groups, sample_ids, m) if sample_ids is not None else m.sum()
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=GROUP_COLORS.get(g, "#999999"), s=20, alpha=0.6,
                   edgecolors="white", linewidth=0.3,
                   label=f"{g} (n={n_pat})")
    ax.set_title(f"{label} — Groups{title_suffix}", fontsize=14)
    ax.legend(fontsize=9, ncol=2)
    ax.grid(True, alpha=0.2)
    ax.set_xlabel(f"{label} 1")
    ax.set_ylabel(f"{label} 2")

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3D plot (static PNG)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_3d(coords, groups, binary_labels, output_path, title_suffix="",
            sample_ids=None, method="tsne"):
    """Generate 3D figure with binary + group panels."""
    label = _method_label(method)
    fig = plt.figure(figsize=(18, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")

    for bl, c, name in [(0, "#3498db", "Non-cancer"),
                         (1, "#e74c3c", "Cancer"),
                         (-1, "#95a5a6", "Other")]:
        m = binary_labels == bl
        if m.sum() == 0:
            continue
        n_pat = _count_patients(groups, sample_ids, m) if sample_ids is not None else m.sum()
        ax1.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                    c=c, s=12, alpha=0.5, label=f"{name} (n={n_pat})")
    ax1.set_title(f"{label} 3D — Binary{title_suffix}")
    ax1.legend(fontsize=9)

    for g in sorted(set(groups)):
        m = groups == g
        n_pat = _count_patients(groups, sample_ids, m) if sample_ids is not None else m.sum()
        ax2.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                    c=GROUP_COLORS.get(g, "#999999"),
                    s=12, alpha=0.5, label=f"{g} (n={n_pat})")
    ax2.set_title(f"{label} 3D — Groups{title_suffix}")
    ax2.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3D Interactive (plotly HTML) — with patient ID hover
# ═══════════════════════════════════════════════════════════════════════════════

def plot_3d_interactive(coords, groups, binary_labels, output_path,
                        title_suffix="", sample_ids=None, method="tsne"):
    """Generate interactive 3D HTML with plotly. Hover shows patient ID."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    label = _method_label(method)

    fig = make_subplots(
        rows=2, cols=1,
        specs=[[{"type": "scatter3d"}], [{"type": "scatter3d"}]],
        subplot_titles=[
            f"{label} 3D — Binary{title_suffix}",
            f"{label} 3D — Groups{title_suffix}",
        ],
        vertical_spacing=0.05,
    )

    # Build per-point hover text with patient ID
    hover_ids = None
    if sample_ids is not None:
        hover_ids = np.array([f"{groups[i]}-{sample_ids[i]}"
                              for i in range(len(groups))])

    # Panel 1: Binary
    for bl, c, name in [(0, "#3498db", "Non-cancer"),
                         (1, "#e74c3c", "Cancer"),
                         (-1, "#95a5a6", "Other")]:
        m = binary_labels == bl
        if m.sum() == 0:
            continue
        n_pat = _count_patients(groups, sample_ids, m) if sample_ids is not None else m.sum()
        hover = hover_ids[m] if hover_ids is not None else None
        fig.add_trace(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
            mode="markers",
            marker=dict(size=1.5, color=c, opacity=0.6),
            name=f"{name} (n={n_pat})",
            legendgroup="binary",
            legendgrouptitle_text="Binary",
            text=hover,
            hovertemplate="%{text}<extra></extra>" if hover is not None else None,
        ), row=1, col=1)

    # Panel 2: Groups (row 2)
    for g in sorted(set(groups)):
        m = groups == g
        n_pat = _count_patients(groups, sample_ids, m) if sample_ids is not None else m.sum()
        hover = hover_ids[m] if hover_ids is not None else None
        fig.add_trace(go.Scatter3d(
            x=coords[m, 0], y=coords[m, 1], z=coords[m, 2],
            mode="markers",
            marker=dict(size=1.5, color=GROUP_COLORS.get(g, "#999999"), opacity=0.6),
            name=f"{g} (n={n_pat})",
            legendgroup="groups",
            legendgrouptitle_text="Groups",
            text=hover,
            hovertemplate="%{text}<extra></extra>" if hover is not None else None,
        ), row=2, col=1)

    fig.update_layout(
        title=f"SERS {label} 3D Interactive{title_suffix}",
        width=2000, height=1400,
        template="plotly_white",
        legend=dict(font=dict(size=20)),
    )

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    print(f"Saved (interactive): {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="Dimensionality reduction visualization (t-SNE / UMAP / PCA)")
    p.add_argument("--groups", nargs="+", required=True,
                   help="Groups to include (e.g. YPAN SPAN CPAN NOR)")
    p.add_argument("--input", "-i", default="results/processed_spectra.csv",
                   help="Processed spectra CSV")
    p.add_argument("--output", "-o", default=None,
                   help="Output path (auto-generated if omitted)")
    p.add_argument("--method", "-m", default="tsne",
                   choices=["tsne", "umap", "pca"],
                   help="Embedding method (default: tsne)")
    p.add_argument("--dim", type=int, default=2, choices=[2, 3],
                   help="Dimensions (default: 2)")
    p.add_argument("--perplexity", type=int, default=30,
                   help="t-SNE perplexity (default: 30)")
    p.add_argument("--n-neighbors", type=int, default=15,
                   help="UMAP n_neighbors (default: 15)")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="UMAP min_dist (default: 0.1)")
    p.add_argument("--aggregate", "-a", default="none",
                   choices=["none", "mean", "medoid"],
                   help="Aggregation per patient (default: none)")
    p.add_argument("--pca-init", type=int, default=50,
                   help="PCA pre-reduction dims (0 to skip, default: 50)")
    p.add_argument("--interactive", action="store_true", default=False,
                   help="Generate interactive HTML (3D only, requires plotly)")
    return p.parse_args()


def main():
    args = parse_args()

    X, groups, sample_ids = load_and_filter(args.input, args.groups, args.aggregate)
    binary = classify_binary(groups)
    coords = run_embedding(
        X, args.method, args.dim, args.perplexity, args.pca_init,
        n_neighbors=args.n_neighbors, min_dist=args.min_dist,
    )

    # Cluster quality scores
    scores = compute_cluster_scores(coords, groups, binary)
    print_scores(scores, args.method)

    tag = "_".join(sorted(args.groups))
    title = f" ({args.aggregate})" if args.aggregate != "none" else ""

    if args.interactive and args.dim == 3:
        out = (Path(args.output) if args.output
               else Path("results") / f"{args.method}_3d_{tag}.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        plot_3d_interactive(coords, groups, binary, str(out), title,
                            sample_ids=sample_ids, method=args.method)
    else:
        out = (Path(args.output) if args.output
               else Path("results") / f"{args.method}_{args.dim}d_{tag}.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.dim == 2:
            plot_2d(coords, groups, binary, str(out), title,
                    sample_ids=sample_ids, method=args.method)
        else:
            plot_3d(coords, groups, binary, str(out), title,
                    sample_ids=sample_ids, method=args.method)


if __name__ == "__main__":
    main()
