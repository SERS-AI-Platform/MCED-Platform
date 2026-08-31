from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Literal, TypeAlias

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

from scripts.db.aecd_model_lineage_timeline_core import ParameterChangeRow, TimelineRow

BLUE: Final = "#1F4E79"
GOLD: Final = "#B8860B"
ORANGE: Final = "#C55A11"
OLIVE: Final = "#6B7F2A"
PINK: Final = "#A64D79"
MUTED: Final = "#8A8F98"
GRID: Final = "#D8DEE7"
INK: Final = "#202833"
PALETTE: Final = (BLUE, GOLD, ORANGE, OLIVE, PINK)
MetricName: TypeAlias = Literal["cancer_screening_auc", "cancer_type_id_auc"]


def _groups(rows: Sequence[TimelineRow]) -> Mapping[str, tuple[TimelineRow, ...]]:
    grouped: dict[str, list[TimelineRow]] = {}
    for row in rows:
        grouped.setdefault(row.registry_model_name, []).append(row)
    return {name: tuple(items) for name, items in grouped.items()}


def _dated(rows: Sequence[TimelineRow]) -> tuple[TimelineRow, ...]:
    return tuple(row for row in rows if row.source_timestamp_epoch is not None)


def _utc(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


def _family_colors(groups: Mapping[str, Sequence[TimelineRow]]) -> Mapping[str, str]:
    ordered = sorted(groups, key=lambda name: (-len(groups[name]), name))
    return {
        name: PALETTE[index] if index < len(PALETTE) else MUTED
        for index, name in enumerate(ordered)
    }


def _short_family(value: str) -> str:
    return value.removeprefix("aecd-legacy-").removeprefix("aecd-production-")[:32]


def _short_parameter(value: str) -> str:
    return value.removeprefix("config_").removeprefix("model_params_")[:24]


def _format_date_axis(axis: Axes) -> None:
    axis.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axis.tick_params(axis="x", labelrotation=0)
    axis.grid(axis="both", color=GRID, linewidth=0.7, alpha=0.75)
    axis.set_axisbelow(True)


def _save(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_version_timeline(rows: Sequence[TimelineRow], output_path: Path) -> None:
    groups = _groups(rows)
    colors = _family_colors(groups)
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    ordered_families = sorted(groups, key=lambda name: (-len(groups[name]), name))
    max_order = max((row.history_order for row in rows), default=1)
    for family_index, family in enumerate(ordered_families):
        points = _dated(groups[family])
        if not points:
            continue
        x_values = [_utc(row.source_timestamp_epoch) for row in points if row.source_timestamp_epoch is not None]
        y_values = [row.history_order for row in points]
        color = colors[family]
        line_style = "-" if color != MUTED else ":"
        axes[0].plot(x_values, y_values, color=color, linewidth=1.6, linestyle=line_style, alpha=0.9)
        axes[0].scatter(x_values, y_values, color=color, edgecolor="white", linewidth=0.6, s=28, zorder=3)
        last = points[-1]
        if family_index < len(PALETTE):
            axes[0].annotate(
                f"{_short_family(family)}  r{last.registry_model_version}/{last.source_model_version}",
                (_utc(last.source_timestamp_epoch), last.history_order),
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=7.5,
                color=color,
                va="center",
            )
        sizes = [32 + 16 * min(row.hyperparameter_change_count, 8) for row in points]
        axes[1].scatter(
            x_values,
            [row.hyperparameter_change_count for row in points],
            color=color,
            edgecolor="white",
            linewidth=0.6,
            s=sizes,
            alpha=0.88,
            label=_short_family(family),
        )
    axes[0].set_title("Historical model version progression by source time", loc="left", color=INK, weight="bold")
    axes[0].set_ylabel("Family history order", color=INK)
    axes[0].set_ylim(0.5, max_order + 1.4)
    axes[1].set_title("Hyperparameter changes per historical transition", loc="left", color=INK, weight="bold")
    axes[1].set_ylabel("Changed hyperparameters", color=INK)
    axes[1].set_xlabel("Historical source timestamp (UTC)", color=INK)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(colors=INK)
        _format_date_axis(axis)
    missing_count = sum(row.source_time_status != "observed" for row in rows)
    fig.suptitle("Model lineage timeline", x=0.07, y=0.99, ha="left", fontsize=16, color=INK, weight="bold")
    fig.text(
        0.07,
        0.945,
        "Each line is a registered model family; labels show registry/source version at the latest dated point. "
        f"Point size below encodes changed hyperparameters. Undated bundles: {missing_count}.",
        fontsize=9,
        color="#56616F",
    )
    fig.subplots_adjust(left=0.07, right=0.82, top=0.89, bottom=0.09, hspace=0.3)
    _save(fig, output_path)


def save_parameter_change_figure(
    rows: Sequence[TimelineRow],
    changes: Sequence[ParameterChangeRow],
    output_path: Path,
) -> None:
    total_counts: Counter[str] = Counter()
    for change in changes:
        total_counts[change.parameter_name] += change.change_count
    top_parameters = [name for name, _ in total_counts.most_common(12)]
    families = sorted({row.registry_model_name for row in rows})
    matrix = [[0 for _ in top_parameters] for _ in families]
    family_index = {family: index for index, family in enumerate(families)}
    parameter_index = {parameter: index for index, parameter in enumerate(top_parameters)}
    for change in changes:
        if change.parameter_name in parameter_index:
            matrix[family_index[change.registry_model_name]][parameter_index[change.parameter_name]] = change.change_count
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": (0.9, 1.6)})
    if top_parameters:
        labels = [_short_parameter(name) for name in top_parameters]
        axes[0].barh(labels[::-1], [total_counts[name] for name in top_parameters[::-1]], color=BLUE)
        axes[0].set_xlabel("Transition count", color=INK)
        axes[0].set_title("Most frequently changed hyperparameters", loc="left", color=INK, weight="bold")
        axes[0].grid(axis="x", color=GRID, linewidth=0.7, alpha=0.75)
    else:
        axes[0].text(0.5, 0.5, "No hyperparameter transitions", ha="center", va="center", color=INK)
        axes[0].set_title("Hyperparameter changes", loc="left", color=INK, weight="bold")
    axes[0].spines[["top", "right", "left"]].set_visible(False)
    axes[0].tick_params(axis="y", length=0, colors=INK)
    axes[1].set_title("Family × hyperparameter change matrix", loc="left", color=INK, weight="bold")
    if top_parameters and families:
        max_count = max(max(values, default=0) for values in matrix)
        cmap = LinearSegmentedColormap.from_list("aecd_blue", ["#F1F5F9", BLUE])
        axes[1].imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=max(1, max_count))
        axes[1].set_xticks(range(len(top_parameters)), [_short_parameter(name) for name in top_parameters], rotation=45, ha="right")
        axes[1].set_yticks(range(len(families)), [_short_family(name) for name in families])
        for row_index, values in enumerate(matrix):
            for column_index, value in enumerate(values):
                if value:
                    axes[1].text(column_index, row_index, str(value), ha="center", va="center", fontsize=8, color=INK)
    else:
        axes[1].text(0.5, 0.5, "No matrix rows", ha="center", va="center", color=INK)
    axes[1].set_xlabel("Parameter name", color=INK)
    axes[1].set_ylabel("Registered model family", color=INK)
    for axis in axes:
        axis.tick_params(colors=INK)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Parameter evolution across historical model versions", x=0.07, y=0.99, ha="left", fontsize=16, color=INK, weight="bold")
    fig.text(0.07, 0.945, "Counts are transitions between adjacent source-time versions within each registered model family.", fontsize=9, color="#56616F")
    fig.subplots_adjust(left=0.07, right=0.98, top=0.88, bottom=0.16, wspace=0.35)
    _save(fig, output_path)


def _metric_value(row: TimelineRow, metric: MetricName) -> float | None:
    if metric == "cancer_screening_auc":
        return row.cancer_screening_auc
    return row.cancer_type_id_auc


def _plot_metric_axis(axis: Axes, rows: Sequence[TimelineRow], colors: Mapping[str, str], metric: MetricName, title: str) -> None:
    groups = _groups(rows)
    for family, family_rows in groups.items():
        points = tuple(row for row in _dated(family_rows) if _metric_value(row, metric) is not None)
        if not points:
            continue
        x_values = [_utc(row.source_timestamp_epoch) for row in points if row.source_timestamp_epoch is not None]
        y_values = [_metric_value(row, metric) for row in points]
        if len(points) >= 2:
            axis.plot(x_values, y_values, color=colors[family], linewidth=1.4, alpha=0.8)
        axis.scatter(x_values, y_values, color=colors[family], edgecolor="white", linewidth=0.6, s=34, label=_short_family(family))
    axis.axhline(0.5, color="#56616F", linestyle="--", linewidth=0.9, label="0.5 reference")
    axis.set_title(title, loc="left", color=INK, weight="bold")
    axis.set_ylim(0, 1)
    axis.set_ylabel("AUC", color=INK)
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=INK)
    _format_date_axis(axis)


def save_performance_timeline(rows: Sequence[TimelineRow], output_path: Path) -> None:
    groups = _groups(rows)
    colors = _family_colors(groups)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    _plot_metric_axis(axes[0], rows, colors, "cancer_screening_auc", "Cancer Screening ROC-AUC")
    _plot_metric_axis(axes[1], rows, colors, "cancer_type_id_auc", "Cancer Type ID ROC-AUC")
    axes[1].set_xlabel("Historical source timestamp (UTC)", color=INK)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.005, 1), frameon=False, fontsize=8)
    fig.suptitle("Historical performance metrics by source time", x=0.07, y=0.99, ha="left", fontsize=16, color=INK, weight="bold")
    fig.text(
        0.07,
        0.945,
        "Metrics are shown as lineage evidence, not as a direct benchmark across runs with different cohorts or aggregation rules. "
        "Cancer Screening may be hospital/measurement-confounded.",
        fontsize=8.5,
        color="#56616F",
    )
    fig.subplots_adjust(left=0.07, right=0.82, top=0.88, bottom=0.09, hspace=0.3)
    _save(fig, output_path)
