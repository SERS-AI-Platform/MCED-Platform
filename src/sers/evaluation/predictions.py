"""run 계열별 예측 파일 → 표준 `PredictionSet` 로더.

각 run 계열이 예측값을 다른 형식으로 저장한다:

- mapping (`results/mapping_*/patient_oof_predictions.csv`): long-format,
  task × model × aggregation × subject. OOF, 환자 단위 StratifiedGroupKFold.
- fixed-split (`results/training/*/fixed_test_predictions.npz`): `s1_prob`(이진),
  `s2_prob`(7클래스), `y_bin`, `y_type`(-1=비암), `sample_ids`, `groups`.
  held-out test. **`force_test_group`에 따라 test 세트 구성이 달라** run 간
  test 지표가 서로 비교 불가일 수 있다 — `cohort_id`에 그 설정을 넣어
  구분한다.

이 모듈은 파일 형식 차이를 여기서 끝내고, 이후 단계는 `PredictionSet`만 본다.
`s1`/`s2` 같은 파일 키는 여기서 `Task`로 변환된다.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sers.evaluation.schema import CohortSpec, Evaluation, Task


@dataclass(frozen=True, slots=True)
class PredictionSet:
    """한 (run, model, task)의 예측값. 이후 모든 계산의 입력 단위."""

    run_id: str
    model_name: str
    task: Task
    evaluation: Evaluation
    cohort: CohortSpec
    y_true: np.ndarray        # 이진: 0/1. 다중: 클래스 인덱스
    y_score: np.ndarray       # 이진: (n,). 다중: (n, k)
    groups: np.ndarray        # 재추출 단위(환자 ID)
    decision: np.ndarray | None = None  # 기록된 결정 라벨이 있으면 (임계값 의존 지표용)

    @property
    def is_binary(self) -> bool:
        return self.y_score.ndim == 1


# ---------------------------------------------------------------------------
# mapping
# ---------------------------------------------------------------------------

MAPPING_TASK = {
    "cancer_vs_non_cancer": Task.CANCER_SCREENING,
    "three_class": Task.PROSTATE_THREE_CLASS,
}


def load_mapping_oof(run_dir: Path, *, aggregations: tuple[str, ...] = ("mean",)
                     ) -> Iterator[PredictionSet]:
    """mapping run의 OOF 예측을 (task, model, aggregation)별로 낸다.

    `aggregations`는 replicate를 환자 점수로 합친 방식(mean/median/majority/
    trimmed_mean). mapping run card는 mean을 1차 비교로 쓴다. 이 값은 스키마의
    `aggregation`(재추출 단위)과 다른 개념이므로 model_name에 붙여 구분한다.
    """
    frame = pd.read_csv(run_dir / "patient_oof_predictions.csv", encoding="utf-8-sig")
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    groups_meta = meta.get("group_counts", {})

    for (task_key, model, agg), sub in frame.groupby(["task", "model", "aggregation"]):
        if agg not in aggregations or task_key not in MAPPING_TASK:
            continue
        task = MAPPING_TASK[task_key]
        y = sub["true_label"].to_numpy()
        if task is Task.CANCER_SCREENING:
            score = sub["probability_class_1"].to_numpy(dtype=float)
            n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
            cancer, control = ("Prostate cancer",), ("Control", "Prostate disease control")
        else:
            cols = sorted(c for c in sub.columns if c.startswith("probability_class_"))
            score = sub[cols].to_numpy(dtype=float)
            n_pos, n_neg = len(y), 0
            cancer, control = tuple(groups_meta), ()
        cohort = CohortSpec(
            cohort_id="mapping_prostate_113",
            site="mapping",
            cancer_groups=cancer, control_groups=control,
            n_patients=len(y), n_positive=n_pos, n_negative=n_neg,
            n_spectra=meta.get("input_repeats"),
            notes=f"group_counts={groups_meta}. OOF StratifiedGroupKFold by patient.",
        )
        yield PredictionSet(
            run_id=run_dir.name, model_name=f"{model} ({agg})", task=task,
            evaluation=Evaluation.OOF, cohort=cohort,
            y_true=y, y_score=score, groups=sub["subject_ordinal"].to_numpy(),
        )


# ---------------------------------------------------------------------------
# fixed-split (stacking)
# ---------------------------------------------------------------------------

def load_fixed_split_test(run_dir: Path) -> Iterator[PredictionSet]:
    """fixed-split run의 held-out test 예측을 screening / type-ID로 낸다.

    Cancer Type ID는 암 양성(y_bin==1) 표본에서만 정의된다 (비암은 y_type=-1).
    """
    z = np.load(run_dir / "fixed_test_predictions.npz", allow_pickle=True)
    summary = json.loads((run_dir / "fixed_split_run_summary.json").read_text())
    results = json.loads((run_dir / "fixed_split_results.json").read_text())

    y_bin = z["y_bin"].astype(int)
    groups_arr = z["groups"].astype(str)
    sample_ids = z["sample_ids"].astype(str)
    force = summary.get("force_test_group") or ""
    excl = tuple(summary.get("exclude_source_groups") or ())

    cancer = tuple(sorted({g for g, b in zip(groups_arr, y_bin) if b == 1}))
    control = tuple(sorted({g for g, b in zip(groups_arr, y_bin) if b == 0}))
    # test 세트 구성이 같은 run끼리만 같은 cohort_id를 갖도록 설정을 키에 넣는다
    cohort = CohortSpec(
        cohort_id=f"fixedsplit_test|force={force or 'none'}|excl={','.join(excl) or 'none'}",
        site="multi-site (Thermo)",
        cancer_groups=cancer, control_groups=control,
        n_patients=len(y_bin), n_positive=int(y_bin.sum()), n_negative=int((1 - y_bin).sum()),
        notes=f"held-out test. force_test_group={force!r}, exclude={excl}. "
              "환자 1명당 1행(replicate 사전 집계).",
    )
    model = f"STK-V2 meta={summary.get('best_meta', '?')}"

    yield PredictionSet(
        run_id=run_dir.name, model_name=model, task=Task.CANCER_SCREENING,
        evaluation=Evaluation.TEST_HELDOUT, cohort=cohort,
        y_true=y_bin, y_score=z["s1_prob"].astype(float), groups=sample_ids,
    )

    cancer_mask = y_bin == 1
    type_cohort = CohortSpec(
        cohort_id=cohort.cohort_id + "|cancer_only",
        site=cohort.site, cancer_groups=cancer, control_groups=(),
        n_patients=int(cancer_mask.sum()), n_positive=int(cancer_mask.sum()), n_negative=0,
        notes=f"Cancer Type ID는 암 양성만. 클래스 순서={results.get('cancer_types')}",
    )
    yield PredictionSet(
        run_id=run_dir.name, model_name=model, task=Task.CANCER_TYPE_ID,
        evaluation=Evaluation.TEST_HELDOUT, cohort=type_cohort,
        y_true=z["y_type"][cancer_mask].astype(int),
        y_score=z["s2_prob"][cancer_mask].astype(float),
        groups=sample_ids[cancer_mask],
    )
