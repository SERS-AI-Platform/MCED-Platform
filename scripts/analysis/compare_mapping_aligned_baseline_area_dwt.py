from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from plot_mapping_aligned_baseline_area_dwt import (
    configure_style,
    make_auc_bar,
    make_metric_bar,
    make_minimum_bar,
    make_repeat_figures,
)

REPO = Path(__file__).resolve().parents[2]
OUTPUT = REPO / "results" / "mapping_aligned_baseline_area_dwt_20260826_v1"
CONDITIONS = (
    ("raw_aligned", "Raw aligned"),
    ("baseline", "Baseline"),
    ("baseline_area", "Baseline + area"),
    ("baseline_dwt", "Baseline + DWT"),
    ("baseline_area_dwt", "Baseline + area + DWT"),
)
RUNS = {condition: OUTPUT / condition for condition, _label in CONDITIONS}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def collect_oof() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for condition, label in CONDITIONS:
        for row in read_rows(RUNS[condition] / "oof_metrics.csv"):
            if row.get("aggregation") == "mean":
                output.append({"condition": condition, "label": label, **row})
    return output


def collect_training() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for condition, label in CONDITIONS:
        rows = read_rows(RUNS[condition] / "training_history.csv")
        for task in ("binary", "three"):
            selected_rows = []
            for fold in sorted({row["fold"] for row in rows if row.get("task") == task}):
                fold_rows = [row for row in rows if row.get("task") == task and row.get("fold") == fold]
                selected_rows.append(min(fold_rows, key=lambda row: number(row, "validation_loss")))
            train_auc = np.array([number(row, "train_auc") for row in selected_rows])
            validation_auc = np.array([number(row, "validation_auc") for row in selected_rows])
            output.append(
                {
                    "condition": condition,
                    "label": label,
                    "task": task,
                    "folds": str(len(selected_rows)),
                    "best_epoch_mean": f"{np.nanmean([number(row, 'epoch') for row in selected_rows]):.10g}",
                    "train_auc_mean": f"{np.nanmean(train_auc):.10g}",
                    "validation_auc_mean": f"{np.nanmean(validation_auc):.10g}",
                    "train_val_auc_gap_mean": f"{np.nanmean(train_auc - validation_auc):.10g}",
                }
            )
    return output


def collect_repeat() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for condition, label in CONDITIONS:
        for row in read_rows(RUNS[condition] / "repeat_metrics_summary.csv"):
            if (
                row.get("task") == "cancer_vs_non_cancer"
                and row.get("model") == "Multi-scale ResNet"
                and row.get("aggregation") == "mean"
            ):
                output.append({"condition": condition, "label": label, **row})
    return output


def collect_stability() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for condition, label in CONDITIONS:
        for row in read_rows(RUNS[condition] / "prediction_stability.csv"):
            if (
                row.get("task") == "cancer_vs_non_cancer"
                and row.get("model") == "Multi-scale ResNet"
                and row.get("aggregation") == "mean"
            ):
                output.append({"condition": condition, "label": label, **row})
    return output


def collect_minimum() -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for condition, label in CONDITIONS:
        for row in read_rows(RUNS[condition] / "minimum_repeat_count.csv"):
            if row.get("model") in ("Multi-scale ResNet", "LR-reference") and row.get("aggregation") == "mean":
                output.append({"condition": condition, "label": label, **row})
    return output


def find_row(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    return next((row for row in rows if all(row.get(key) == value for key, value in criteria.items())), {})


def write_report(oof: list[dict[str, str]], training: list[dict[str, str]], minimum: list[dict[str, str]]) -> None:
    lines = [
        "# Mapping raw aligned → baseline → area normalization → DWT",
        "",
        "## 실험 설계",
        "",
        "- 동일 mapping cohort 113명, 13,673 finite repeats, 933-point grid, 3-block Multi-scale 1D ResNet, 30 epochs, seed 20260826, MC=100.",
        "- raw aligned는 400–2200 cm⁻¹ trim 후 402–2198 cm⁻¹ grid에 보간한 조건이다. baseline correction은 aligned grid 이후 rolling-minimum window 101을 적용했다.",
        "- area normalization은 baseline-corrected spectrum을 `y / sum(abs(y))`로 변환했다.",
        "- DWT는 `db4`, `symmetric`, level 7, per-spectrum finest-detail MAD universal threshold, soft-threshold 후 933-point 재구성으로 적용했다.",
        "- 모든 finite transformed repeat를 유지했으며 spectral QC filtering은 적용하지 않았다. 검증은 patient-level StratifiedGroupKFold 5-fold OOF다.",
        "",
        "## OOF 결과",
        "",
        "| 조건 | ResNet Cancer Screening AUC | BA | ResNet Cancer Type ID macroAUC | Macro F1 | LR-reference Screening AUC | LR-reference Type ID macroAUC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, label in CONDITIONS:
        res_binary = find_row(oof, condition=condition, task="cancer_vs_non_cancer", model="Multi-scale ResNet")
        res_three = find_row(oof, condition=condition, task="three_class", model="Multi-scale ResNet")
        lr_binary = find_row(oof, condition=condition, task="cancer_vs_non_cancer", model="LR-reference")
        lr_three = find_row(oof, condition=condition, task="three_class", model="LR-reference")
        lines.append(
            f"| {label} | {number(res_binary, 'roc_auc'):.4f} | {number(res_binary, 'balanced_accuracy'):.4f} | {number(res_three, 'macro_roc_auc'):.4f} | {number(res_three, 'macro_f1'):.4f} | {number(lr_binary, 'roc_auc'):.4f} | {number(lr_three, 'macro_roc_auc'):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Minimum repeat count",
            "",
            "| 조건 | ResNet Cancer Screening n* | ResNet Cancer Type ID n* | LR-reference Cancer Screening n* | LR-reference Cancer Type ID n* |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, label in CONDITIONS:
        res_binary = find_row(minimum, condition=condition, task="cancer_vs_non_cancer", model="Multi-scale ResNet")
        res_three = find_row(minimum, condition=condition, task="three_class", model="Multi-scale ResNet")
        lr_binary = find_row(minimum, condition=condition, task="cancer_vs_non_cancer", model="LR-reference")
        lr_three = find_row(minimum, condition=condition, task="three_class", model="LR-reference")
        lines.append(
            f"| {label} | {res_binary.get('minimum_n', '-')} | {res_three.get('minimum_n', '-')} | {lr_binary.get('minimum_n', '-')} | {lr_three.get('minimum_n', '-')} |"
        )
    lines.extend(
        [
            "",
            "## 해석",
            "",
            "- spectrum-only ResNet에서는 raw aligned가 Cancer Screening AUC 0.6528로 가장 높았다.",
            "- baseline correction을 추가하면 AUC가 0.5056으로 하락했다. area normalization은 0.5233으로 일부 회복했지만 raw aligned보다 낮았다.",
            "- baseline+DWT는 0.4890, baseline+area+DWT는 0.5116으로 DWT가 현재 조건에서 CNN 성능을 개선하지 못했다.",
            "- 반대로 LR-reference는 baseline 조건에서 Screening AUC 0.7043, Type ID macroAUC 0.6065로 높아졌다. 따라서 baseline이 정보 자체를 제거했다기보다, 현재 ResNet이 raw intensity/scale 변화에서 사용하던 feature를 잃었거나 학습 설정이 그 표현을 활용하지 못한 가능성이 있다.",
            "- DWT/area 조건의 3-class macroF1 저하는 예측 calibration 또는 class decision profile도 함께 점검해야 하며, AUC만으로 최종 pipeline을 선택하지 않는다.",
            "- Cancer Screening AUC는 병원/측정 조건 confounding 가능성이 있어 외부 일반화 성능으로 해석하지 않는다.",
            "",
            "## Figures and source tables",
            "",
            "- ![AUC comparison](figure_preprocessing_auc_bar.png)",
            "- ![Metric profile](figure_preprocessing_metrics_bar.png)",
            "- ![Repeat analysis](figure_repeat_auc_stability.png)",
            "- ![Minimum repeat](figure_minimum_repeat_bar.png)",
            "- `comparison_oof_metrics.csv`: 5개 조건의 mean OOF metrics 및 LR-reference/blend 포함",
            "- `comparison_training_summary.csv`: best validation-loss epoch 기준 training summary",
            "- `comparison_repeat_metrics.csv`: ResNet 100회 MC repeat summary",
            "- `comparison_prediction_stability.csv`: patient probability SD summary",
            "- `comparison_minimum_repeat_count.csv`: model/task별 n*",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    configure_style()
    oof = collect_oof()
    training = collect_training()
    repeat = collect_repeat()
    stability = collect_stability()
    minimum = collect_minimum()
    write_csv(OUTPUT / "comparison_oof_metrics.csv", oof)
    write_csv(OUTPUT / "comparison_training_summary.csv", training)
    write_csv(OUTPUT / "comparison_repeat_metrics.csv", repeat)
    write_csv(OUTPUT / "comparison_prediction_stability.csv", stability)
    write_csv(OUTPUT / "comparison_minimum_repeat_count.csv", minimum)
    make_auc_bar(oof, OUTPUT)
    make_metric_bar(oof, OUTPUT)
    make_repeat_figures(repeat, stability, OUTPUT)
    make_minimum_bar(minimum, OUTPUT)
    write_report(oof, training, minimum)
    print(f"Wrote comparison outputs to {OUTPUT}")


if __name__ == "__main__":
    main()
