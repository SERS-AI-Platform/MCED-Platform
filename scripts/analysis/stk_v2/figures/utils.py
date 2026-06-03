#!/usr/bin/env python3
"""Utilities to reproduce uSERS-Net-style figures (Fig2/4/5) for STK-V2 stacking.

This module reconstructs *stacking* out-of-fold (OOF) predictions for each outer fold using:
- Global base-model outer-test predictions stored in oof_predictions.npz
- Inner-CV OOF predictions for outer-train stored in checkpoints/oof_<model>_outerfold<k>.npz
- Fold-specific best meta-learner names stored in best_ensemble_config.json

It then returns full-length arrays:
- s1_prob: (n,) cancer probabilities
- s2_prob: (n, n_classes) cancer-type probabilities (zeros for non-cancer by default)

Notes
-----
- If checkpoint files are missing, the module falls back to simple averaging across base models.
- Sex-constraint masking is *not* applied here because groups_arr/sex metadata is not guaranteed
  to be available in the saved artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


NATURE_TEXT = '#1A1A1A'
NATURE_MUTED = '#5F6368'
NATURE_GRID = '#D9DEE2'
NATURE_BLUE = '#2B6CB0'
NATURE_RED = '#C94842'
NATURE_TEAL = '#2A9D8F'
NATURE_GOLD = '#D99A21'
NATURE_PURPLE = '#7B3294'
NATURE_GRAY = '#6E7781'

NATURE_CANCER_COLORS = {
    'CRC': '#2B6CB0',
    'LUN': '#2A9D8F',
    'BLC': '#D99A21',
    'PRO': '#CC79A7',
    'PAN': '#C94842',
    'OVA': '#7B3294',
    'BRE': '#56A6D6',
}


def nature_rcparams() -> dict[str, object]:
    """Matplotlib defaults tuned for compact journal figures."""
    return {
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 8,
        'axes.titlesize': 9,
        'axes.labelsize': 8,
        'axes.titleweight': 'bold',
        'axes.labelcolor': NATURE_TEXT,
        'axes.edgecolor': NATURE_TEXT,
        'axes.linewidth': 0.65,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,
        'xtick.color': NATURE_TEXT,
        'ytick.color': NATURE_TEXT,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'legend.fontsize': 7,
        'legend.frameon': False,
        'legend.handlelength': 1.4,
        'legend.borderaxespad': 0.3,
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.04,
        'savefig.facecolor': 'white',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'svg.fonttype': 'none',
    }


def apply_nature_style() -> None:
    """Apply the shared STK-V2 journal figure style."""
    mpl.rcParams.update(nature_rcparams())


def style_axis(ax: plt.Axes, *, grid: str | None = None) -> None:
    """Apply light, print-friendly axes styling."""
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.65)
        ax.spines[side].set_color(NATURE_TEXT)
    ax.tick_params(axis='both', colors=NATURE_TEXT, width=0.6, length=3)
    if grid:
        ax.grid(axis=grid, color=NATURE_GRID, linewidth=0.55, alpha=0.75)
        ax.set_axisbelow(True)


def style_matrix_axis(ax: plt.Axes, n_rows: int, n_cols: int) -> None:
    """Draw crisp cell boundaries for matrix-style panels."""
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=0.9)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(axis='both', length=0)


def add_panel_label(ax: plt.Axes, label: str, *, x: float = -0.12, y: float = 1.08) -> None:
    ax.text(
        x, y, label,
        transform=ax.transAxes,
        ha='left', va='top',
        fontsize=11,
        fontweight='bold',
        color=NATURE_TEXT,
    )


def light_colormap(name: str, color: str) -> LinearSegmentedColormap:
    """Single-hue colormap with a white low end for annotated heatmaps."""
    return LinearSegmentedColormap.from_list(name, ['#FFFFFF', '#EAF1F6', color])


def softened_color(color: str, amount: float = 0.35) -> tuple[float, float, float]:
    """Blend a color toward white by amount."""
    rgb = np.array(to_rgb(color))
    return tuple((1 - amount) * rgb + amount * np.ones(3))


def save_nature_figure(
    fig: plt.Figure,
    out: Path,
    *,
    dpi: int = 300,
    aliases: list[Path] | None = None,
    save_pdf: bool = True,
) -> None:
    """Save PNG plus an editable PDF companion for manuscript workflows."""
    paths = [Path(out), *[Path(p) for p in aliases or []]]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white', edgecolor='none')
        if save_pdf and path.suffix.lower() != '.pdf':
            fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white', edgecolor='none')


def _make_meta_learners(n_classes: int):
    """Meta-learner factory mirroring the STK-V2 training script."""
    try:
        from xgboost import XGBClassifier
        xgb_available = True
    except Exception:
        xgb_available = False

    def _lr(task: str):
        return LogisticRegression(
            C=1.0,
            max_iter=2000,
            solver='lbfgs',
            multi_class='multinomial' if task == 'multiclass' else 'auto',
        )

    def _xgb(task: str):
        if not xgb_available:
            return _lr(task)
        kw = dict(n_estimators=200, max_depth=4, learning_rate=0.05, n_jobs=2, verbosity=0)
        if task == 'multiclass':
            kw.update(objective='multi:softprob', num_class=n_classes)
        else:
            kw.update(objective='binary:logistic')
        return XGBClassifier(**kw)

    def _rf(task: str):
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=5,
            class_weight='balanced_subsample',
            n_jobs=2,
            random_state=42,
        )

    def _elasticnet(task: str):
        return LogisticRegression(
            C=0.5,
            penalty='elasticnet',
            l1_ratio=0.5,
            max_iter=2000,
            solver='saga',
            multi_class='multinomial' if task == 'multiclass' else 'auto',
        )

    def _mlp(task: str):
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        )

    return {'lr': _lr, 'xgb': _xgb, 'rf': _rf, 'elasticnet': _elasticnet, 'mlp': _mlp}


def _load_best_per_fold(run_dir: Path) -> Dict[int, str]:
    cfg_path = run_dir / 'best_ensemble_config.json'
    with open(cfg_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return {int(k): v for k, v in cfg.get('best_per_fold', {}).items()}


def _infer_base_model_names_from_oof(oof: np.lib.npyio.NpzFile) -> List[str]:
    # keys like "lr_raw_s1" / "lr_raw_s2"
    names = sorted({k[:-3] for k in oof.files if k.endswith('_s1')})
    return names


def _safe_load_checkpoint(run_dir: Path, model_name: str, outer_fold: int):
    ckpt = run_dir / 'checkpoints' / f'oof_{model_name}_outerfold{outer_fold}.npz'
    if not ckpt.exists():
        return None
    d = np.load(ckpt)
    return d['s1_prob'], d['s2_prob']


def reconstruct_stacking_oof(
    run_dir: Path,
    n_classes: int = 7,
    outer_folds: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Reconstruct stacking OOF predictions and return (bl, ctl, s1_prob, s2_prob, order).

    Parameters
    ----------
    run_dir:
        STK-V2 output directory, e.g. results/training/stacking_v2
    n_classes:
        number of cancer classes
    outer_folds:
        outer CV folds (default 5)

    Returns
    -------
    bl: (n,) binary labels
    ctl: (n,) cancer-type labels (non-cancer = -1)
    s1_prob: (n,) stacking cancer probabilities
    s2_prob: (n, n_classes) stacking cancer-type probabilities
    base_models: list of base model names (used for meta features)
    """

    run_dir = Path(run_dir)
    oof_path = run_dir / 'oof_predictions.npz'
    if not oof_path.exists():
        raise FileNotFoundError(f'Not found: {oof_path}')

    oof = np.load(oof_path, allow_pickle=True)
    bl = oof['binary_labels'].astype(int)
    ctl = oof['cancer_type_labels'].astype(int)
    sample_ids = oof['sample_ids']

    base_models = _infer_base_model_names_from_oof(oof)

    # global base-model predictions on each sample (outer-test predictions)
    base_s1_global = {m: oof[f'{m}_s1'].astype(float) for m in base_models}
    base_s2_global = {m: oof[f'{m}_s2'].astype(float) for m in base_models}

    best_per_fold = _load_best_per_fold(run_dir)
    factories = _make_meta_learners(n_classes)

    n = len(bl)
    s1_stack = np.full(n, np.nan, dtype=float)
    s2_stack = np.zeros((n, n_classes), dtype=float)

    # Rebuild the same outer split as the training script:
    from sklearn.model_selection import StratifiedGroupKFold

    sgkf = StratifiedGroupKFold(n_splits=outer_folds, shuffle=True, random_state=42)
    splits = list(sgkf.split(np.zeros((n, 1)), bl, sample_ids))

    # If checkpoints are missing, we will later fall back to averaging.
    any_ckpt_missing = False

    for fold, (tr_idx, te_idx) in enumerate(splits):
        meta_name = best_per_fold.get(fold, 'rf')
        meta_factory = factories.get(meta_name)
        if meta_factory is None:
            meta_factory = factories['rf']

        # Build meta-feature matrices:
        # - train features come from inner-CV OOF checkpoints for this fold
        # - test features come from global base predictions at te_idx
        meta_s1_tr_cols = []
        meta_s2_tr_cols = []
        for m in base_models:
            ckpt = _safe_load_checkpoint(run_dir, m, fold)
            if ckpt is None:
                any_ckpt_missing = True
                break
            s1_tr, s2_tr = ckpt
            meta_s1_tr_cols.append(s1_tr.reshape(-1, 1))
            meta_s2_tr_cols.append(s2_tr)

        if any_ckpt_missing:
            break

        meta_s1_train = np.concatenate(meta_s1_tr_cols, axis=1)
        meta_s2_train = np.concatenate(meta_s2_tr_cols, axis=1)

        meta_s1_test = np.column_stack([base_s1_global[m][te_idx] for m in base_models])
        meta_s2_test = np.hstack([base_s2_global[m][te_idx] for m in base_models])

        meta_s1_train = np.nan_to_num(meta_s1_train, nan=0.0)
        meta_s2_train = np.nan_to_num(meta_s2_train, nan=0.0)
        meta_s1_test = np.nan_to_num(meta_s1_test, nan=0.0)
        meta_s2_test = np.nan_to_num(meta_s2_test, nan=0.0)

        y_bin_tr = bl[tr_idx]
        y_type_tr = ctl[tr_idx]

        # Stage 1 meta
        ml_s1 = meta_factory('binary')
        ml_s1.fit(meta_s1_train, y_bin_tr)
        try:
            s1_pred = ml_s1.predict_proba(meta_s1_test)[:, 1]
        except Exception:
            # fallback for estimators without predict_proba
            s1_pred = ml_s1.decision_function(meta_s1_test)
        s1_stack[te_idx] = s1_pred

        # Stage 2 meta (cancer only)
        cancer_tr = y_bin_tr == 1
        cancer_te_mask = bl[te_idx] == 1
        if cancer_tr.sum() > 5 and cancer_te_mask.sum() > 0:
            ml_s2 = meta_factory('multiclass')
            ml_s2.fit(meta_s2_train[cancer_tr], y_type_tr[cancer_tr])
            prob = ml_s2.predict_proba(meta_s2_test[cancer_te_mask])
            # map to full class space
            full = np.zeros((cancer_te_mask.sum(), n_classes), dtype=float)
            for j, cls in enumerate(ml_s2.classes_):
                if int(cls) < n_classes:
                    full[:, int(cls)] = prob[:, j]
            s2_stack[te_idx[cancer_te_mask]] = full

    if any_ckpt_missing or np.isnan(s1_stack).any():
        # Fallback: simple averaging across base models (still valid OOF because base predictions
        # are generated on outer-test sets)
        s1_stack = np.nanmean(np.column_stack([base_s1_global[m] for m in base_models]), axis=1)
        s2_stack = np.nanmean(np.stack([base_s2_global[m] for m in base_models], axis=0), axis=0)

    return bl, ctl, s1_stack, s2_stack, base_models


def get_palette_and_labels():
    """Try to reuse AACR/nature_style palette; otherwise use a local fallback."""
    # Desired order in the existing figure scripts
    order = ['CRC', 'LUN', 'BLC', 'PRO', 'PAN', 'OVA', 'BRE']
    internal_idx = {'PRO': 0, 'LUN': 1, 'CRC': 2, 'PAN': 3, 'OVA': 4, 'BRE': 5, 'BLC': 6}
    try:
        # If the old style module exists in your repo
        from nature_style import AACR_LABELS, CANCER_COLORS as AACR_CC  # type: ignore
        cancer_colors = dict(AACR_CC)
        cancer_colors.update({'BRE': '#D4527A', 'BLC': '#E8960C'})
        labels = [AACR_LABELS.get(c, c) for c in order]
    except Exception:
        cancer_colors = dict(NATURE_CANCER_COLORS)
        labels = order

    return order, internal_idx, cancer_colors, labels
