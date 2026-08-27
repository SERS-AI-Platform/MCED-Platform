from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from mapping_resnet_data import GROUP_TO_CLASS, PreparedSubject, write_csv
from mapping_resnet_model import _auc

REPO = Path(__file__).resolve().parents[2]


def run_locked_stk_reference(subjects: Sequence[PreparedSubject], out: Path) -> dict[str, object]:
    try:
        from sers_predict import StackingPredictor
    except (ImportError, ModuleNotFoundError) as exc:
        return {"status": "not_run", "reason": f"import failed: {exc}"}
    mapping_root = REPO / "data" / "mapping"
    label_by_ordinal: dict[int, str] = {}
    workbook = __import__("openpyxl").load_workbook(mapping_root / "clinical_df.xlsx", read_only=True, data_only=True)
    rows = list(workbook.active.iter_rows(values_only=True))
    columns = {str(value): index for index, value in enumerate(rows[0]) if value is not None}
    for ordinal, row in enumerate(rows[1:], start=1):
        label_by_ordinal[ordinal] = str(row[columns["solum_label"]])
    folder_map = {
        folder.name.replace(" ", "_"): folder
        for date_folder in mapping_root.glob("2026*_mapping")
        for folder in date_folder.iterdir()
        if folder.is_dir()
    }
    try:
        predictor = StackingPredictor(REPO / "artifacts" / "usersnet" / "current")
    except Exception as exc:
        return {"status": "not_run", "reason": f"artifact load failed: {exc}"}
    result_rows: list[dict[str, object]] = []
    for subject in subjects:
        label = label_by_ordinal[subject.ordinal]
        folder = folder_map[label]
        average_path = folder / f"{label.replace('_', ' ')}_ave.CSV"
        prediction = predictor.predict_single(average_path)
        result_rows.append(
            {
                "subject_ordinal": subject.ordinal,
                "group": subject.group,
                "status": prediction.get("status", "error"),
                "stk_v2_cancer_probability": prediction.get("cancer_probability", float("nan")),
                "model_variant": prediction.get("model_variant", "stacking_v2"),
            }
        )
    write_csv(out / "stk_v2_locked_external_patient_predictions.csv", result_rows)
    usable = [row for row in result_rows if row["status"] == "ok" and np.isfinite(float(row["stk_v2_cancer_probability"]))]
    y = np.array([GROUP_TO_CLASS[str(row["group"])] == GROUP_TO_CLASS["Prostate cancer"] for row in usable], dtype=int)
    p = np.array([float(row["stk_v2_cancer_probability"]) for row in usable], dtype=float)
    summary = {
        "status": "ok",
        "n_patients": len(usable),
        "roc_auc": _auc(y, np.column_stack([1 - p, p])) if len(np.unique(y)) == 2 else float("nan"),
        "note": "External locked STK-v2 prediction on instrument `_ave` files; not nested-CV and not directly comparable to mapping OOF models.",
    }
    (out / "stk_v2_locked_external_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_report(
    out: Path,
    metadata: dict[str, object],
    metrics: Sequence[dict[str, object]],
    minimum: Sequence[dict[str, object]],
    stk_summary: dict[str, object],
    args: argparse.Namespace,
) -> None:
    main_metrics = [row for row in metrics if row.get("aggregation") == "mean"]
    lines = [
        "# data/mapping Multi-scale 1D ResNet 평가",
        "",
        "## 결론 요약",
        "",
        f"- 분석 대상: {metadata['subject_count']}명, Control {metadata['group_counts']['Control']}명, Prostate disease control {metadata['group_counts']['Prostate disease control']}명, Prostate cancer {metadata['group_counts']['Prostate cancer']}명.",
        f"- 입력 반복: subject당 raw 121개, QC 통과 스펙트럼 중앙값 {metadata['qc_passed_per_subject']['median']:.0f}개.",
        f"- Multi-scale 1D ResNet residual block 수: **{metadata['n_blocks']}개**. 각 block은 k=3/5/7 branch를 사용하고 채널 stage는 32→64→128이다. position-aware={metadata['position_aware']}.",
        "- 외부 STK-v2는 production artifact를 그대로 사용한 Cancer Screening reference이며, mapping cohort OOF 모델과 직접적인 동일-run 비교가 아니다.",
        "",
        "## 전처리 및 검증",
        "",
        "- Trim 400–2200 cm⁻¹ → Savitzky-Golay smoothing (11, 3) → centered rolling-minimum baseline (101) → SNV → fixed common-grid linear interpolation.",
        f"- 실험 grid: {metadata['grid']['min_cm-1']:.1f}–{metadata['grid']['max_cm-1']:.1f} cm⁻¹, {metadata['grid']['points']} points, step {metadata['grid']['step_cm-1']:.5f} cm⁻¹.",
        "- QC는 baseline-corrected repeat set에 robust median/MAD outlier filter를 적용해 감사값으로 기록했다. 프로젝트 QC 게이트 원칙에 따라 모든 finite repeat를 121회 locked 평가에 유지했다.",
        "- Outer CV는 StratifiedGroupKFold 5-fold, group=patient ordinal; 동일 환자의 반복 스펙트럼은 train/test에 분리되지 않았다.",
        "- LR-reference는 기존 전립선 reference와 같은 StandardScaler + balanced Logistic Regression + inner 4-fold C search를 사용했다.",
        "",
        "## 주요 OOF 지표 (patient mean aggregation)",
        "",
        "| Task | Model | ROC-AUC | Macro ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Macro F1 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in main_metrics:
        def fmt(key: str) -> str:
            value = float(row.get(key, np.nan))
            return "-" if not np.isfinite(value) else f"{value:.3f}"

        lines.append(
            f"| {row['task']} | {row['model']} | {fmt('roc_auc')} | {fmt('macro_roc_auc')} | {fmt('balanced_accuracy')} | {fmt('sensitivity')} | {fmt('specificity')} | {fmt('macro_f1')} |"
        )
    lines.extend(
        [
            "",
            "## 최소 반복 횟수 판정",
            "",
            "- 121회 mean aggregation을 reference로 하고 AUC ≤0.01, balanced accuracy ≤0.02, Cancer Screening sensitivity ≤0.02, prediction agreement ≥95%, 다음 후보까지 AUC 증가량 ≤0.005를 적용했다.",
            "- 상세 판정표는 `minimum_repeat_count.csv`에 저장했다. `not_met`은 해당 모델/aggregation에서 모든 후보가 기준을 동시에 충족하지 못했다는 뜻이다.",
            "",
            "## STK-v2 reference 주의사항",
            "",
            f"- 상태: {stk_summary.get('status', 'unknown')}; patient 수: {stk_summary.get('n_patients', '-')}; external ROC-AUC: {stk_summary.get('roc_auc', '-')}",
            "- STK-v2 production artifact는 935-point grid와 10-base-model ElasticNet stack을 사용하고, 이 mapping 실험의 ResNet/LR는 933-point grid를 사용한다. 따라서 STK-v2 수치는 reference로만 해석한다.",
            "- STK-v2 production artifact에는 mapping의 세 그룹을 그대로 분류하는 3-class head가 없으므로, 3-class 비교는 mapping cohort에서 새로 학습한 LR/ResNet만 포함한다.",
            "- Cancer Screening AUC는 병원/측정 조건 confounding 가능성이 있어 외부 일반화 성능으로 해석하지 않는다.",
            "",
            "## 재현 정보",
            "",
            f"- run date: 2026-08-25; epochs={args.epochs}; Monte-Carlo iterations={args.mc_iterations}; seed={args.seed}.",
            "- 출력에는 raw patient identifier, source filename, 원시 스펙트럼 배열을 포함하지 않고 ordinal/집계 결과만 저장했다.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
