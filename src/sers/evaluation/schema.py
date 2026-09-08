"""성능 보고 표준 스키마 — 코호트·평가종류·task를 필수로 못박는다.

왜 필요한가
-----------
같은 "AUC"가 run 계열마다 다른 이름으로 저장돼 있다
(`test_auc` / `roc_auc` / `val_s1_auc`). 더 위험한 건 **무엇을 측정한
AUC인지가 기록돼 있지 않다**는 점이다:

- `logs/experiment_runs.jsonl`은 val 지표만 담고 test 필드가 없는데,
  `fixed_split_run_summary.json`에는 test가 있다. 양쪽에서 "AUC"를 읽으면
  **val과 test가 조용히 섞인다.**
- `nested_cv_results.csv`의 `s1_auc`는 OOF인데 fixed-split의 test 지표와
  이름이 같다.
- mapping의 `three_class`(Control/전립선질환대조/전립선암)와 `s2`(7개 암종)는
  전혀 다른 task인데 둘 다 "다중분류"다.
- 코호트가 다르면 애초에 비교 대상이 아니다 (연세 췌장 30:30 vs
  보라매 전립선 41:7).

그래서 이 스키마는 `task`·`evaluation`·`cohort`를 **선택 필드가 아니라
필수**로 둔다. 하나라도 다르면 비교를 거부할 수 있게 하기 위함이다.

명명 규칙
---------
보고 레이어에서는 `s1`/`s2`를 쓰지 않는다
(`.claude/agents/experiment-runner.md`: 용어: "Cancer Screening" (Stage 1
아님), "Cancer Type ID" (Stage 2 아님)). 다만 npz/json **파일 스키마의**
`s1_prob`/`s2_prob` 키는 그대로 둔다 — 프로덕션 학습 코드가 쓰고 있어
바꾸면 깨진다. 변환은 로더 경계에서만 한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum

import pandas as pd

from sers.evaluation.bootstrap import MetricCI


class Task(StrEnum):
    """무엇을 예측하는 문제인가."""

    CANCER_SCREENING = "cancer_screening"      # 암 vs 비암 (이진)
    CANCER_TYPE_ID = "cancer_type_id"          # 암종 7분류 (암 양성 내에서)
    PROSTATE_THREE_CLASS = "prostate_three_class"  # Control/전립선질환대조/전립선암
    #                                            ↑ mapping 전용. CANCER_TYPE_ID와 절대 합치지 말 것


class Evaluation(StrEnum):
    """어떤 데이터로 잰 값인가. 서로 다르면 비교 불가."""

    OOF = "oof"                    # 교차검증 out-of-fold
    VAL = "val"                    # 모델 선택에 쓴 검증 세트
    TEST_HELDOUT = "test_heldout"  # 한 번만 채점한 홀드아웃
    EXTERNAL = "external"          # 별도 기관/전향 코호트


@dataclass(frozen=True, slots=True)
class CohortSpec:
    """어떤 사람들에게서 잰 값인가.

    코호트가 다르면 지표를 나란히 놓을 수 없다. 특히 `n_positive`/`n_negative`
    불균형은 특이도·PPV 신뢰구간을 크게 좌우한다 (보라매는 정상 7명이라
    특이도 CI가 [0.03, 0.51]이었다).
    """

    cohort_id: str                 # 예: "yonsei_prospective_2026"
    site: str                      # 예: "연세", "보라매"
    cancer_groups: tuple[str, ...]  # 예: ("YPAN",)
    control_groups: tuple[str, ...]  # 예: ("YNOR",)
    n_patients: int
    n_positive: int
    n_negative: int
    n_spectra: int | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_positive + self.n_negative != self.n_patients:
            raise ValueError(
                f"{self.cohort_id}: n_positive({self.n_positive}) + "
                f"n_negative({self.n_negative}) != n_patients({self.n_patients})"
            )


@dataclass(frozen=True, slots=True)
class MetricRow:
    """long-format 한 행 = 지표 하나. CSV 스키마의 단위."""

    run_id: str
    model_name: str
    task: Task
    evaluation: Evaluation
    aggregation: str          # "patient" | "spectrum"
    cohort: CohortSpec
    metric: str
    value: float
    ci_low: float
    ci_high: float
    ci_method: str
    n_boot: int
    n_units: int              # 실질 표본수 = 재추출 단위 수 (환자 수)
    n_rows: int               # 그 아래 행 수 (스펙트럼 수)

    @classmethod
    def from_ci(cls, ci: MetricCI, *, run_id: str, model_name: str, task: Task,
                evaluation: Evaluation, aggregation: str, cohort: CohortSpec) -> MetricRow:
        return cls(
            run_id=run_id, model_name=model_name, task=task, evaluation=evaluation,
            aggregation=aggregation, cohort=cohort, metric=ci.metric, value=ci.value,
            ci_low=ci.ci_low, ci_high=ci.ci_high, ci_method=ci.ci_method,
            n_boot=ci.n_boot, n_units=ci.n_units, n_rows=ci.n_rows,
        )

    def flatten(self) -> dict[str, object]:
        row = {k: v for k, v in asdict(self).items() if k != "cohort"}
        row["task"] = str(self.task)
        row["evaluation"] = str(self.evaluation)
        cohort = asdict(self.cohort)
        cohort["cancer_groups"] = "|".join(self.cohort.cancer_groups)
        cohort["control_groups"] = "|".join(self.cohort.control_groups)
        row.update({f"cohort_{k}": v for k, v in cohort.items()})
        return row


#: CSV 컬럼 순서 — 사람이 읽을 때 맥락(무엇을/누구에게/어떻게)이 먼저 오도록.
COLUMN_ORDER = [
    "run_id", "model_name", "task", "evaluation", "aggregation",
    "cohort_cohort_id", "cohort_site", "cohort_cancer_groups", "cohort_control_groups",
    "cohort_n_patients", "cohort_n_positive", "cohort_n_negative", "cohort_n_spectra",
    "n_units", "n_rows",
    "metric", "value", "ci_low", "ci_high", "ci_method", "n_boot",
    "cohort_notes",
]


def to_dataframe(rows: list[MetricRow]) -> pd.DataFrame:
    """표준 컬럼 순서의 DataFrame으로."""
    frame = pd.DataFrame([r.flatten() for r in rows])
    ordered = [c for c in COLUMN_ORDER if c in frame.columns]
    return frame[ordered + [c for c in frame.columns if c not in ordered]]


def write_metrics_csv(rows: list[MetricRow], path) -> None:
    """표준 스키마 CSV로 저장. 한글이 있으므로 utf-8-sig 고정."""
    to_dataframe(rows).to_csv(path, index=False, encoding="utf-8-sig")


@dataclass(frozen=True, slots=True)
class ComparabilityVerdict:
    comparable: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def check_comparable(a: MetricRow, b: MetricRow) -> ComparabilityVerdict:
    """두 지표 행을 나란히 놓아도 되는지. 하나라도 어긋나면 비교 거부.

    `~/.claude/skills/compare-runs`의 기존 4가지 검사(집계·암종세트·비대조군·
    표본수)를 보완한다. 여기서 막는 건 그 스킬이 다루지 않는 축이다.
    """
    reasons: list[str] = []
    if a.task != b.task:
        reasons.append(f"task 불일치: {a.task} vs {b.task}")
    if a.evaluation != b.evaluation:
        reasons.append(
            f"평가 종류 불일치: {a.evaluation} vs {b.evaluation} "
            "(OOF와 held-out test는 측정 대상이 다르다)"
        )
    if a.cohort.cohort_id != b.cohort.cohort_id:
        reasons.append(f"코호트 불일치: {a.cohort.cohort_id} vs {b.cohort.cohort_id}")
    if a.aggregation != b.aggregation:
        reasons.append(f"집계 단위 불일치: {a.aggregation} vs {b.aggregation}")
    if a.metric != b.metric:
        reasons.append(f"지표 불일치: {a.metric} vs {b.metric}")
    return ComparabilityVerdict(comparable=not reasons, reasons=tuple(reasons))
