"""Figure 2b: cancer-type discrimination peak feature importance."""

from __future__ import annotations

import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nature_style import (
    CANCER_COLORS,
    LINE_WIDTH,
    OUTPUT_DIR,
    apply_nature_style,
    apply_style,
    save_figure,
)
from stk_v2_fixed_non_ypan import (
    CANCER_TYPES,
    DISPLAY_CANCERS,
    add_peak_assignment_columns,
    compute_cancer_type_peak_feature_importance,
    load_plot_data,
)

apply_style()

PANEL_LABELS = {
    "PRO": "PRO",
    "BRE": "BRE",
    "OVA": "OVA",
    "LUN": "LUN",
    "CRC": "CRC",
    "PAN": "PAN",
    "BLC": "BLC",
}


def _add_panel_letter(ax, label: str) -> None:
    ax.text(
        0.012,
        0.36,
        label,
        transform=ax.transAxes,
        fontsize=8.5,
        fontweight="bold",
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.4},
    )


def _mean_stats(X: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray | int]:
    spectra = X[mask, 0, :]
    return {
        "mean": spectra.mean(axis=0),
        "sem": spectra.std(axis=0, ddof=1) / np.sqrt(len(spectra)),
        "n": int(len(spectra)),
    }


def _save_assignment_table(df: pd.DataFrame) -> None:
    out_dir = OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    table = add_peak_assignment_columns(df[df["rank"] <= 4]).copy()
    table["cancer_type"] = pd.Categorical(
        table["cancer_type"], categories=list(DISPLAY_CANCERS), ordered=True
    )
    table = (
        table.sort_values(["cancer_type", "rank"])[
            [
                "cancer_type",
                "rank",
                "peak_position_cm-1",
                "representative_assignment",
                "candidate_assignments",
                "shade_min_cm-1",
                "shade_max_cm-1",
                "feature_importance_score",
            ]
        ]
    ).copy()
    table.to_csv(
        os.path.join(out_dir, "fig2b_cancer_type_peak_table.csv"), index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(8.7, 5.25))
    ax.axis("off")
    display = table[
        [
            "cancer_type",
            "rank",
            "peak_position_cm-1",
            "representative_assignment",
            "candidate_assignments",
            "feature_importance_score",
        ]
    ].copy()
    display["feature_importance_score"] = display["feature_importance_score"].map(
        lambda x: f"{x:.2f}"
    )
    display["candidate_assignments"] = display["candidate_assignments"].map(
        lambda x: textwrap.fill(str(x), width=45)
    )
    display = display.rename(
        columns={
            "cancer_type": "Cancer",
            "rank": "Rank",
            "peak_position_cm-1": "Peak (cm$^{-1}$)",
            "representative_assignment": "Representative assignment",
            "candidate_assignments": "Candidate assignments",
            "feature_importance_score": "Feature importance",
        }
    )
    tbl = ax.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.075, 0.055, 0.105, 0.225, 0.39, 0.15],
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(4.8)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_linewidth(0.35)
        cell.set_edgecolor("#CCCCCC")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#F3F3F3")
    fig.tight_layout()
    save_figure(fig, "fig2b_cancer_type_peak_table")
    plt.close(fig)


def main() -> None:
    importance, wavenumbers, scores = compute_cancer_type_peak_feature_importance()
    X_3ch, _meta, wavenumbers, groups, _sample_ids, _yb, _yt = load_plot_data()
    cancer_pool = np.isin(groups, CANCER_TYPES)
    stats = {
        group: _mean_stats(X_3ch, groups == group)
        for group in DISPLAY_CANCERS
        if np.any(groups == group)
    }
    other_stats = {group: _mean_stats(X_3ch, cancer_pool & (groups != group)) for group in stats}
    _save_assignment_table(importance)

    order = [g for g in DISPLAY_CANCERS if g in stats]
    fig, axes = plt.subplots(
        len(order),
        1,
        figsize=(7.2, 1.05 * len(order)),
        sharex=True,
        gridspec_kw={"hspace": 0.14},
    )
    if len(order) == 1:
        axes = [axes]

    label_y_positions = []

    for i, (group, ax) in enumerate(zip(order, axes)):
        _add_panel_letter(ax, chr(ord("a") + i))
        color = CANCER_COLORS.get(group, "#333333")

        other = other_stats[group]
        ax.plot(
            wavenumbers,
            other["mean"],
            color="#B8B8B8",
            lw=LINE_WIDTH["thin"],
            alpha=0.9,
            label=f"Other cancers (n={other['n']})",
        )
        ax.fill_between(
            wavenumbers,
            other["mean"] - 1.96 * other["sem"],
            other["mean"] + 1.96 * other["sem"],
            color="#B8B8B8",
            alpha=0.14,
            linewidth=0,
        )

        st = stats[group]
        line_color = color
        line_width = 1.2
        ax.plot(
            wavenumbers,
            st["mean"],
            color=line_color,
            lw=line_width,
            label=f"{group} (n={st['n']})",
        )
        ax.fill_between(
            wavenumbers,
            st["mean"] - 1.96 * st["sem"],
            st["mean"] + 1.96 * st["sem"],
            color=color,
            alpha=0.12,
            linewidth=0,
        )

        y_min = min(float(np.min(st["mean"])), float(np.min(other["mean"])))
        y_max = max(float(np.max(st["mean"])), float(np.max(other["mean"])))
        y_pad = 0.08 * (y_max - y_min + 1e-9)
        y_range = y_max - y_min + 1e-9
        ax.set_ylim(y_min - y_pad, y_max + 0.58 * y_range)
        label_y = y_max + 0.10 * y_range
        label_y_positions.append(label_y)

        shade_peaks = importance[
            (importance["cancer_type"] == group) & (importance["rank"] <= 5)
        ].sort_values("rank")
        label_levels = {}
        recent: list[tuple[int, int]] = []
        for peak in shade_peaks["peak_position_cm-1"].astype(int):
            level = 0
            used = {lvl for prev_peak, lvl in recent if abs(peak - prev_peak) < 70}
            while level in used:
                level += 1
            label_levels[peak] = min(level, 2)
            recent.append((peak, label_levels[peak]))
        score = scores.get(group, np.zeros_like(wavenumbers))
        for _, row in shade_peaks.iterrows():
            peak = int(row["peak_position_cm-1"])
            lo = float(row["shade_min_cm-1"])
            hi = float(row["shade_max_cm-1"])
            band = (wavenumbers >= lo) & (wavenumbers <= hi)
            local_score = float(np.nanmax(score[band])) if band.any() else 0.0
            alpha = 0.08 + 0.24 * local_score
            ax.axvspan(lo, hi, color=color, alpha=alpha, zorder=0)
            label_level = label_levels.get(peak, 0)
            ax.text(
                peak,
                label_y + label_level * 0.13 * y_range,
                f"{peak}",
                ha="center",
                va="bottom",
                fontsize=6.8,
                fontweight="bold",
                color=color,
            )

        ax.text(
            0.985,
            0.47,
            PANEL_LABELS[group],
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
            color=line_color,
            alpha=0.9,
        )
        ax.legend(loc="upper right", frameon=False, fontsize=6.6, bbox_to_anchor=(0.95, 1.02))
        ax.set_ylabel("Intensity (a.u.)")
        apply_nature_style(ax)

    axes[-1].set_xlabel("Wavenumber (cm$^{-1}$)")
    axes[-1].set_xlim(float(wavenumbers.min()), float(wavenumbers.max()))
    fig.suptitle(
        "Cancer Type Discrimination Peak Feature Importance",
        fontsize=9.5,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(left=0.088, right=0.995, bottom=0.06, top=0.952, hspace=0.14)
    save_figure(fig, "fig2b_cancer_type_feature_importance")
    plt.close(fig)
    print("Figure 2 complete.")


if __name__ == "__main__":
    main()
