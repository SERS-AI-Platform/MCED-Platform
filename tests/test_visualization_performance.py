"""성능 그림 모듈: PNG만 저장, 고정 스타일, 부제목 규칙."""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from sers.visualization.performance import plot_confusion_matrix, plot_pr, plot_roc
from sers.visualization.style import CLASS_COLORS, apply_style, rcparams, save_png, subtitle_from

matplotlib.use("Agg")


def _data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    s = np.clip(y * 0.4 + rng.normal(0.3, 0.25, n), 0, 1)
    return y, s


def test_save_png_rewrites_other_suffixes(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = save_png(fig, tmp_path / "fig.pdf")
    assert out.suffix == ".png" and out.exists()
    assert not (tmp_path / "fig.pdf").exists()
    assert any("PNG" in str(w.message) for w in caught)


def test_save_png_extra_formats_only_when_explicit(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    save_png(fig, tmp_path / "fig.png", extra_formats=("pdf",))
    assert (tmp_path / "fig.png").exists() and (tmp_path / "fig.pdf").exists()


def test_subtitle_order_is_fixed():
    text = subtitle_from(n=112, n_pos=43, n_neg=69, model="LR", evaluation="5-fold OOF",
                         metric="AUC 0.82 [0.73, 0.89]")
    assert text == "n=112 (암 43 / 비암 69) · LR · 5-fold OOF · AUC 0.82 [0.73, 0.89]"
    assert subtitle_from(n=10, groups={"a": 4, "b": 6}) == "n=10 (a 4 / b 6)"


def test_roc_cm_pr_write_png_only(tmp_path):
    y, s = _data()
    plot_roc((y, s), title="ROC", model="LR", evaluation="OOF", auc_ci=(0.6, 0.9),
             output=tmp_path / "roc.png")
    plot_pr(y, s, title="PR", output=tmp_path / "pr.png")
    plot_confusion_matrix(y, (s >= 0.5).astype(int), title="CM",
                          class_keys=("non_cancer", "cancer"), threshold=0.5,
                          output=tmp_path / "cm.png")
    files = sorted(p.name for p in tmp_path.iterdir())
    assert files == ["cm.png", "pr.png", "roc.png"]
    plt.close("all")


def test_roc_single_curve_uses_cancer_color_and_subtitle():
    y, s = _data()
    fig, ax = plot_roc((y, s), title="ROC", model="LR", evaluation="OOF")
    colors = {line.get_color().lower() for line in ax.get_lines()}
    assert CLASS_COLORS["cancer"].lower() in colors
    texts = [t.get_text() for t in ax.texts]
    assert any(t.startswith("n=60 (암") and "LR" in t and "AUC" in t for t in texts)
    plt.close(fig)


def test_multi_roc_fixed_series_order():
    y, s = _data()
    y2, s2 = _data(seed=1)
    fig, ax = plot_roc([("A", y, s), ("B", y2, s2)], title="ROC", auc_ci={"A": (0.5, 0.9)})
    labels = [line.get_label() for line in ax.get_lines()]
    assert any(lbl.startswith("A — AUC") and "[" in lbl for lbl in labels)
    assert any(lbl.startswith("B — AUC") and "[" not in lbl for lbl in labels)
    plt.close(fig)


def test_apply_style_sets_font_stack_and_png_dpi():
    apply_style()
    assert matplotlib.rcParams["font.sans-serif"][0] == "Arial"
    assert matplotlib.rcParams["savefig.dpi"] == rcparams()["savefig.dpi"]
    assert matplotlib.rcParams["axes.spines.top"] is False
