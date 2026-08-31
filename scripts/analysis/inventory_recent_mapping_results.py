from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "recent_experiment_inventory_20260826_v1"


SPECS: list[dict[str, Any]] = [
    {
        "directory": "mapping_multiscale_resnet_20260826_standard_repro",
        "category": "완료 학습",
        "group": "직접 비교: 3-block 전처리 ablation",
        "purpose": "기존 standard 전처리 재현 기준선",
        "status": "주 분석 결과",
        "note": "Trim→SG(11,3)→rolling-min baseline(101)→SNV.",
    },
    {
        "directory": "mapping_trim_only_20260826_v1",
        "category": "완료 학습",
        "group": "직접 비교: 3-block 전처리 ablation",
        "purpose": "trim만 적용한 CNN 전처리 ablation",
        "status": "주 분석 결과",
        "note": "400–2200 trim만 적용; signal correction은 적용하지 않음.",
    },
    {
        "directory": "mapping_no_trim_raw_20260826_v1",
        "category": "완료 학습",
        "group": "직접 비교: 3-block 전처리 ablation",
        "purpose": "trim하지 않은 raw spectrum 기준",
        "status": "주 분석 결과",
        "note": "full raw range를 933-point grid로 보간.",
    },
    {
        "directory": "mapping_multiscale_resnet_20260826_deep12_standard",
        "category": "완료 학습",
        "group": "직접 비교: depth 3 vs 12",
        "purpose": "standard 전처리에서 12-block 심화",
        "status": "주 분석 결과",
        "note": "3-block 기준선을 보존하고 12-block을 별도 학습.",
    },
    {
        "directory": "mapping_trim_only_20260826_deep12",
        "category": "완료 학습",
        "group": "직접 비교: depth 3 vs 12",
        "purpose": "trim-only에서 12-block 심화",
        "status": "주 분석 결과",
        "note": "trim-only 3-block의 깊이 증가 효과 검증.",
    },
    {
        "directory": "mapping_no_trim_raw_20260826_deep12",
        "category": "완료 학습",
        "group": "직접 비교: depth 3 vs 12",
        "purpose": "no-trim raw에서 12-block 심화",
        "status": "주 분석 결과",
        "note": "no-trim raw 3-block의 깊이 증가 효과 검증.",
    },
    {
        "directory": "mapping_preprocessing_summary_20260826_v1",
        "category": "요약/시각화",
        "group": "전처리 종합",
        "purpose": "3개 전처리 조건의 표·그래프 종합",
        "status": "파생 산출물",
        "note": "직접 비교표, delta, minimum repeat, Figure 1–7 포함.",
    },
    {
        "directory": "mapping_depth_comparison_20260826_v1",
        "category": "요약/시각화",
        "group": "depth 종합",
        "purpose": "3-block과 12-block의 성능·training gap 종합",
        "status": "파생 산출물",
        "note": "depth_oof_metrics.csv와 depth_training_summary.csv 포함.",
    },
    {
        "directory": "mapping_clinical_spectrum_logistic_20260825_v1",
        "category": "완료 학습",
        "group": "참고: clinical fusion",
        "purpose": "clinical covariates와 spectrum을 함께 쓰는 LR",
        "status": "참고 결과",
        "note": "PSA, UA pH, UA SG, microscopy WBC/RBC 5개 clinical feature.",
    },
    {
        "directory": "mapping_clinical_spectrum_logistic_20260825_smoke",
        "category": "smoke",
        "group": "참고: clinical fusion",
        "purpose": "clinical+spectrum LR pipeline 동작 확인",
        "status": "증거 제외",
        "note": "MC=2 smoke; 최종 성능 판단에 사용하지 않음.",
    },
    {
        "directory": "mapping_trim_only_20260826_smoke",
        "category": "smoke",
        "group": "pipeline smoke",
        "purpose": "trim-only ResNet pipeline 동작 확인",
        "status": "증거 제외",
        "note": "3 blocks, 1 epoch, MC=2.",
    },
    {
        "directory": "mapping_no_trim_raw_20260826_smoke",
        "category": "smoke",
        "group": "pipeline smoke",
        "purpose": "no-trim raw ResNet pipeline 동작 확인",
        "status": "증거 제외",
        "note": "3 blocks, 1 epoch, MC=2.",
    },
    {
        "directory": "mapping_multiscale_resnet_20260825_v1",
        "category": "중간 산출물",
        "group": "이전 실행 잔여물",
        "purpose": "standard 전처리 배열 생성/cache",
        "status": "완료 metric 없음",
        "note": "preprocessed_mapping_arrays.npz와 preprocessing_metadata.json만 존재; OOF metric 없음.",
    },
    {
        "directory": "mapping_repeat_average_patent_20260825_v1",
        "category": "반복성/신호 감사",
        "group": "참고: repeat averaging",
        "purpose": "121회 반복 평균·noise·peak detectability 감사",
        "status": "참고 결과",
        "note": "분류 모델 성능이 아니라 기존 average와 반복 평균의 신호 안정성 비교.",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_float(value)
    return "" if number is None else f"{number:.{digits}f}"


def metric_row(directory: Path, task: str, model: str) -> dict[str, str]:
    filenames = ["oof_metrics.csv", "combined_lr_oof_metrics.csv"]
    for filename in filenames:
        for row in read_csv(directory / filename):
            if (
                row.get("task") == task
                and row.get("model") == model
                and row.get("aggregation") == "mean"
            ):
                return row
    return {}


def minimum_repeat(directory: Path, task: str, model: str) -> str:
    filenames = ["minimum_repeat_count.csv", "combined_lr_minimum_repeat_count.csv"]
    for filename in filenames:
        rows = read_csv(directory / filename)
        for row in rows:
            if (
                row.get("task") == task
                and row.get("model") == model
                and row.get("aggregation") == "mean"
            ):
                return row.get("minimum_n", "")
    return ""


def metadata_summary(meta: dict[str, Any]) -> dict[str, str]:
    preprocessing = meta.get("preprocessing", {})
    if not isinstance(preprocessing, dict):
        preprocessing = {}
    trim = preprocessing.get("trim")
    trim_text = "none" if trim is None else str(trim)
    smooth = preprocessing.get("smooth")
    smooth_text = "none" if smooth is None else "SG(11,3)"
    baseline = preprocessing.get("baseline")
    baseline_text = "none" if baseline is None else "rolling-min(101)"
    normalization = preprocessing.get("normalization") or "none"
    grid = meta.get("grid", {})
    if not isinstance(grid, dict) or not grid:
        grid_text = ""
    else:
        grid_text = (
            f"{grid.get('min_cm-1', '')}–"
            f"{grid.get('max_cm-1', '')} ({grid.get('points', '')} points)"
        )
    return {
        "trim": trim_text,
        "smooth": smooth_text,
        "baseline": baseline_text,
        "normalization": str(normalization),
        "grid": grid_text,
    }


def link_for(path: Path, output: Path, label: str | None = None) -> str:
    relative = path.relative_to(output.parent).as_posix()
    return f"[{label or path.name}](../{relative})"


def inventory_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in SPECS:
        directory = RESULTS / spec["directory"]
        meta = read_json(directory / "run_metadata.json")
        if not meta:
            meta = read_json(directory / "preprocessing_metadata.json")
        pre = metadata_summary(meta)
        resnet_binary = metric_row(directory, "cancer_vs_non_cancer", "Multi-scale ResNet")
        resnet_three = metric_row(directory, "three_class", "Multi-scale ResNet")
        clinical_binary = metric_row(directory, "cancer_vs_non_cancer", "Clinical + Spectrum LR")
        clinical_three = metric_row(directory, "three_class", "Clinical + Spectrum LR")
        model_binary = resnet_binary or clinical_binary
        model_three = resnet_three or clinical_three
        model_name = (
            "Multi-scale 1D ResNet"
            if resnet_binary or resnet_three
            else "Clinical + Spectrum LR"
            if clinical_binary or clinical_three
            else str(meta.get("model", ""))
        )
        metrics_path = directory / "oof_metrics.csv"
        if not metrics_path.exists():
            metrics_path = directory / "combined_lr_oof_metrics.csv"
        rows.append(
            {
                "directory": spec["directory"],
                "modified": datetime.fromtimestamp(directory.stat().st_mtime).isoformat(
                    timespec="seconds"
                ),
                "category": spec["category"],
                "comparison_group": spec["group"],
                "purpose": spec["purpose"],
                "status": spec["status"],
                "model": model_name,
                "n_blocks": str(meta.get("n_blocks", "")),
                "epochs": str(meta.get("epochs", "")),
                "mc_iterations": str(meta.get("mc_iterations", "")),
                "seed": str(meta.get("seed", "")),
                "subjects": str(meta.get("subject_count", meta.get("included_subjects", ""))),
                "finite_repeats": str(meta.get("model_repeats", meta.get("input_repeats", ""))),
                "trim": pre["trim"],
                "smooth": pre["smooth"],
                "baseline": pre["baseline"],
                "normalization": pre["normalization"],
                "grid": pre["grid"],
                "cancer_screening_auc": fmt(model_binary.get("roc_auc")),
                "cancer_screening_balanced_accuracy": fmt(model_binary.get("balanced_accuracy")),
                "cancer_screening_sensitivity": fmt(model_binary.get("sensitivity")),
                "cancer_screening_specificity": fmt(model_binary.get("specificity")),
                "cancer_type_id_macro_auc": fmt(model_three.get("macro_roc_auc")),
                "cancer_type_id_balanced_accuracy": fmt(model_three.get("balanced_accuracy")),
                "cancer_type_id_macro_f1": fmt(model_three.get("macro_f1")),
                "minimum_repeat_n_cancer_screening": minimum_repeat(
                    directory,
                    "cancer_vs_non_cancer",
                    "Multi-scale ResNet" if resnet_binary else "Clinical + Spectrum LR",
                ),
                "minimum_repeat_n_cancer_type_id": minimum_repeat(
                    directory,
                    "three_class",
                    "Multi-scale ResNet" if resnet_three else "Clinical + Spectrum LR",
                ),
                "metric_file": str(metrics_path.relative_to(ROOT)) if metrics_path.exists() else "",
                "note": spec["note"],
            }
        )
    return rows


def markdown_metric(directory: Path, task: str, model: str, field: str) -> str:
    row = metric_row(directory, task, model)
    return fmt(row.get(field)) or "-"


def depth_summary_rows() -> list[dict[str, str]]:
    rows = read_csv(
        OUTPUT.parent / "mapping_depth_comparison_20260826_v1" / "depth_training_summary.csv"
    )
    return rows


def write_inventory_csv(rows: list[dict[str, str]]) -> None:
    path = OUTPUT / "experiment_inventory.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, str]]) -> None:
    standard = RESULTS / "mapping_multiscale_resnet_20260826_standard_repro"
    trim = RESULTS / "mapping_trim_only_20260826_v1"
    raw = RESULTS / "mapping_no_trim_raw_20260826_v1"
    deep_standard = RESULTS / "mapping_multiscale_resnet_20260826_deep12_standard"
    deep_trim = RESULTS / "mapping_trim_only_20260826_deep12"
    deep_raw = RESULTS / "mapping_no_trim_raw_20260826_deep12"
    clinical = RESULTS / "mapping_clinical_spectrum_logistic_20260825_v1"

    lines = [
        "# Recent Mapping Experiment Inventory",
        "",
        "생성일: 2026-08-26",
        "",
        "## 읽는 방법",
        "",
        "- 최근 `results/mapping_*` 14개를 학습 결과, 파생 요약, smoke, 중간 산출물, 반복성 감사로 분류했다.",
        "- 직접 순위 비교는 동일 cohort(113명, 13,673 finite repeats), seed 20260826, patient-level 5-fold OOF, 30 epochs, MC=100의 3-block/12-block ResNet 조건에만 적용한다.",
        "- 모든 finite repeat는 locked evaluation에 유지했다. standard의 QC 수치는 감사값이며 silent filtering이 아니다.",
        "- Cancer Screening AUC는 병원/측정 조건 confounding 가능성이 있어 외부 일반화 성능으로 해석하지 않는다.",
        "",
        "## 전체 인벤토리",
        "",
        "| 수정 시각 | 분류 | 폴더 | 목적 | 상태 | 핵심 조건 |",
        "|---|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["modified"], reverse=True):
        directory = RESULTS / row["directory"]
        report = directory / "REPORT.md"
        visual = directory / "VISUAL_REPORT.md"
        outputs = []
        if report.exists():
            outputs.append(link_for(report, OUTPUT, "REPORT"))
        if visual.exists():
            outputs.append(link_for(visual, OUTPUT, "VISUAL_REPORT"))
        output_text = ", ".join(outputs) or "metric 없음"
        condition = (
            f"{row['model']}; blocks={row['n_blocks'] or '-'}; epochs={row['epochs'] or '-'}; MC={row['mc_iterations'] or '-'}"
        )
        lines.append(
            f"| {row['modified']} | {row['category']} | `{row['directory']}` | {row['purpose']} | "
            f"{row['status']} ({output_text}) | {condition} |"
        )

    lines.extend(
        [
            "",
            "## 직접 비교 1: 3-block 전처리 ablation",
            "",
            "모델 입력은 933 points이며, ResNet은 121개 repeat spectrum을 학습하고 결과는 patient mean으로 집계했다. LR-reference는 patient mean spectrum 기준이다.",
            "",
            "| 조건 | 전처리 | ResNet Cancer Screening AUC / BA | ResNet Cancer Type ID macroAUC / macroF1 | LR-reference AUC / BA | LR-reference macroAUC / macroF1 | ResNet minimum n* (Screening / Type ID) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    direct = [
        ("standard", standard, "trim + SG + baseline + SNV"),
        ("trim-only", trim, "trim only"),
        ("no-trim raw", raw, "no trim / raw"),
    ]
    for name, directory, preprocessing in direct:
        res_b = metric_row(directory, "cancer_vs_non_cancer", "Multi-scale ResNet")
        res_t = metric_row(directory, "three_class", "Multi-scale ResNet")
        lr_b = metric_row(directory, "cancer_vs_non_cancer", "LR-reference")
        lr_t = metric_row(directory, "three_class", "LR-reference")
        lines.append(
            f"| {name} | {preprocessing} | {fmt(res_b.get('roc_auc'))} / {fmt(res_b.get('balanced_accuracy'))} | "
            f"{fmt(res_t.get('macro_roc_auc'))} / {fmt(res_t.get('macro_f1'))} | "
            f"{fmt(lr_b.get('roc_auc'))} / {fmt(lr_b.get('balanced_accuracy'))} | "
            f"{fmt(lr_t.get('macro_roc_auc'))} / {fmt(lr_t.get('macro_f1'))} | "
            f"{minimum_repeat(directory, 'cancer_vs_non_cancer', 'Multi-scale ResNet')} / "
            f"{minimum_repeat(directory, 'three_class', 'Multi-scale ResNet')} |"
        )

    lines.extend(
        [
            "",
            "### 해석",
            "",
            "- 3-block ResNet 기준 Cancer Screening AUC는 trim-only가 0.6528로 가장 높다.",
            "- standard 대비 trim-only는 +0.2110 AUC, +0.1188 BA이며, trim-only 대비 no-trim raw는 -0.0445 AUC다.",
            "- 따라서 현재 결과는 trim의 유용성과 SG+baseline+SNV bundle의 악화 가능성을 보여주지만, bundle 내부에서 어떤 연산이 원인인지는 분리하지 못한다.",
            "- minimum n*은 각 조건의 자기 121-repeat reference를 기준으로 한 상대 기준이다. 모델 자체의 절대 성능 순위와 동일하지 않다.",
            "",
            "## 직접 비교 2: 3-block vs 12-block",
            "",
            "| 전처리 | 3-block Screening AUC | 12-block Screening AUC | 변화 | 3-block Type ID macroAUC | 12-block macroAUC | 변화 | train–validation AUC gap (3→12) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, d3, d12, prep in [
        ("standard", standard, deep_standard, "standard"),
        ("trim-only", trim, deep_trim, "trim_only"),
        ("no-trim raw", raw, deep_raw, "no_trim_raw"),
    ]:
        b3 = metric_row(d3, "cancer_vs_non_cancer", "Multi-scale ResNet")
        b12 = metric_row(d12, "cancer_vs_non_cancer", "Multi-scale ResNet")
        t3 = metric_row(d3, "three_class", "Multi-scale ResNet")
        t12 = metric_row(d12, "three_class", "Multi-scale ResNet")
        gaps = []
        depth_rows = depth_summary_rows()
        for depth in ("3", "12"):
            match = next(
                (
                    row
                    for row in depth_rows
                    if row.get("condition") == prep
                    and row.get("depth") == depth
                    and row.get("task") == "binary"
                ),
                {},
            )
            gaps.append(fmt(match.get("train_val_auc_gap_mean")))
        auc3 = finite_float(b3.get("roc_auc"))
        auc12 = finite_float(b12.get("roc_auc"))
        macro3 = finite_float(t3.get("macro_roc_auc"))
        macro12 = finite_float(t12.get("macro_roc_auc"))
        delta_auc = "" if auc3 is None or auc12 is None else f"{auc12 - auc3:+.4f}"
        delta_macro = "" if macro3 is None or macro12 is None else f"{macro12 - macro3:+.4f}"
        lines.append(
            f"| {name} | {fmt(auc3)} | {fmt(auc12)} | {delta_auc} | {fmt(macro3)} | {fmt(macro12)} | {delta_macro} | {gaps[0]} → {gaps[1]} |"
        )
    lines.extend(
        [
            "",
            "### depth 결론",
            "",
            "- 12-block으로 깊이를 늘려도 binary CNN 성능은 회복되지 않았다. 특히 trim-only는 0.6528→0.5352로 하락했다.",
            "- trim-only의 train–validation AUC gap은 0.0018→0.1290으로 커져, 깊이 증가가 generalization을 악화시킨 패턴이다.",
            "- standard는 gap이 줄었지만 validation AUC 자체가 낮아 feature extraction이 충분히 회복된 것으로 볼 수 없다.",
            "",
            "## 참고 그룹: clinical fusion",
            "",
            "| 모델 | 입력/학습 단위 | Cancer Screening AUC / BA | Cancer Type ID macroAUC / BA / macroF1 | 판정 |",
            "|---|---|---:|---:|---|",
        ]
    )
    cb = metric_row(clinical, "cancer_vs_non_cancer", "Clinical + Spectrum LR")
    ct = metric_row(clinical, "three_class", "Clinical + Spectrum LR")
    lines.append(
        f"| Clinical + Spectrum LR | patient-mean spectrum + 5 clinical features로 학습; 각 repeat + clinical feature로 평가 | "
        f"{fmt(cb.get('roc_auc'))} / {fmt(cb.get('balanced_accuracy'))} | {fmt(ct.get('macro_roc_auc'))} / {fmt(ct.get('balanced_accuracy'))} / {fmt(ct.get('macro_f1'))} | "
        "전처리 ablation과 직접 비교 금지 |"
    )
    lines.extend(
        [
            "",
            "이 결과는 spectrum-only ResNet이 아니라 clinical information이 추가된 별도 입력 체계다. 따라서 ResNet의 feature extraction 성능을 직접 증명하거나 반증하는 비교군이 아니라, clinical+spectrum fusion의 참고 기준으로 둔다.",
            "",
            "## 참고 그룹: 반복 평균·signal audit",
            "",
            "- `mapping_repeat_average_patent_20260825_v1`: 113명, 121 repeats/subject, QC 통과 반복 수 중앙값 115개.",
            "- 기존 `_ave`와 121회 원시 산술평균의 normalized RMSE 중앙값은 0.0459, correlation 중앙값은 0.98759로 기록됐다.",
            "- 단일 repeat 대비 noise reduction 중앙값은 기존 `_ave` 49.8%, patent QC average 50.3%, 기존 방식 대비 추가 감소는 1.4%였다.",
            "- 이는 분류 AUC 실험이 아니라 측정 반복성과 평균화 효과를 보는 signal audit이다.",
            "",
            "## Smoke와 미완료 산출물",
            "",
            "- `mapping_trim_only_20260826_smoke`, `mapping_no_trim_raw_20260826_smoke`: 1 epoch/MC=2이므로 pipeline 실행 확인용이며 최종 성능 표에서 제외한다.",
            "- `mapping_clinical_spectrum_logistic_20260825_smoke`: MC=2 실행 확인용이며 full v1을 대체하지 않는다.",
            "- `mapping_multiscale_resnet_20260825_v1`: preprocessing arrays와 metadata만 남아 있고 `oof_metrics.csv`가 없으므로 completed model run으로 취급하지 않는다.",
            "",
            "## 기존 요약 파일과 시각화",
            "",
            "- [전처리 종합 시각화](../mapping_preprocessing_summary_20260826_v1/VISUAL_REPORT.md)",
            "- [depth 비교 시각화](../mapping_depth_comparison_20260826_v1/VISUAL_REPORT.md)",
            "- [전처리 직접 비교 표](../mapping_preprocessing_summary_20260826_v1/same_condition_performance.csv)",
            "- [depth OOF 표](../mapping_depth_comparison_20260826_v1/depth_oof_metrics.csv)",
            "- [상세 인벤토리 CSV](experiment_inventory.csv)",
            "",
            "## 역사적 참고값",
            "",
            "기존 결과 요약 파일에는 3/6/9-block historical run, position-aware pilot, clinical-only LR, clinical-input CNN, STK-v2 locked reference도 기록되어 있다. 이들은 seed, 입력 modality, validation artifact 또는 grid가 달라 현재 direct ablation 순위에 포함하지 않는다.",
            "",
            "- [historical_experiment_summary.csv](../mapping_preprocessing_summary_20260826_v1/historical_experiment_summary.csv)",
        ]
    )
    (OUTPUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = inventory_rows()
    write_inventory_csv(rows)
    write_report(rows)
    print(f"Wrote {OUTPUT / 'REPORT.md'}")
    print(f"Wrote {OUTPUT / 'experiment_inventory.csv'}")


if __name__ == "__main__":
    main()
