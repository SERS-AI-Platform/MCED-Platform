"""표준 분류 지표 — 예측 확률과 정답에서 스칼라 하나를 내는 함수들.

`bootstrap.py`가 이 함수들을 콜러블로 받아 신뢰구간을 만든다. 지표를 추가할
때는 `MetricFn` 시그니처(`(y_true, y_score) -> float`)만 지키면 된다.

이름은 `schema.py`의 표준 지표명과 일치시킨다. `s1`/`s2` 같은 내부 약칭은
쓰지 않는다 (`.claude/agents/experiment-runner.md`: "Cancer Screening"
(Stage 1 아님), "Cancer Type ID" (Stage 2 아님)).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)

MetricFn = Callable[[np.ndarray, np.ndarray], float]

#: 임계값 기본값. **운영 임계값이 따로 있는 모델에는 쓰면 안 된다.**
#: 예: usersnet의 'balanced' 모드는 0.5가 아닌 자체 임계값을 쓴다
#: (artifacts/usersnet/v1.0.0/manifest.json: operating_modes.balanced.threshold).
#: 그럴 때는 `binary_metrics_at(threshold=...)`로 명시하거나, 이미 계산된
#: 예측 라벨을 점수처럼 넘긴다(0/1이면 임계값 0.5가 그대로 맞는다).
DEFAULT_THRESHOLD = 0.5


def _binarize(y_score: np.ndarray, threshold: float) -> np.ndarray:
    return (np.asarray(y_score, dtype=float) >= threshold).astype(int)


def _counts(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    pred = _binarize(y_score, threshold)
    true = np.asarray(y_true, dtype=int)
    tp = int(np.sum((true == 1) & (pred == 1)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    tn = int(np.sum((true == 0) & (pred == 0)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    return tp, fp, tn, fn


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC AUC. 한 클래스만 있으면 정의되지 않으므로 NaN."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def sensitivity(y_true: np.ndarray, y_score: np.ndarray,
                threshold: float = DEFAULT_THRESHOLD) -> float:
    tp, _fp, _tn, fn = _counts(y_true, y_score, threshold)
    return float(tp / (tp + fn)) if (tp + fn) else float("nan")


def specificity(y_true: np.ndarray, y_score: np.ndarray,
                threshold: float = DEFAULT_THRESHOLD) -> float:
    _tp, fp, tn, _fn = _counts(y_true, y_score, threshold)
    return float(tn / (tn + fp)) if (tn + fp) else float("nan")


def precision_ppv(y_true: np.ndarray, y_score: np.ndarray,
                  threshold: float = DEFAULT_THRESHOLD) -> float:
    tp, fp, _tn, _fn = _counts(y_true, y_score, threshold)
    return float(tp / (tp + fp)) if (tp + fp) else float("nan")


def npv(y_true: np.ndarray, y_score: np.ndarray,
        threshold: float = DEFAULT_THRESHOLD) -> float:
    _tp, _fp, tn, fn = _counts(y_true, y_score, threshold)
    return float(tn / (tn + fn)) if (tn + fn) else float("nan")


def f1(y_true: np.ndarray, y_score: np.ndarray,
       threshold: float = DEFAULT_THRESHOLD) -> float:
    return float(f1_score(y_true, _binarize(y_score, threshold), zero_division=0))


def balanced_accuracy(y_true: np.ndarray, y_score: np.ndarray,
                      threshold: float = DEFAULT_THRESHOLD) -> float:
    sens = sensitivity(y_true, y_score, threshold)
    spec = specificity(y_true, y_score, threshold)
    if np.isnan(sens) or np.isnan(spec):
        return float("nan")
    return float((sens + spec) / 2.0)


def binary_metrics_at(threshold: float) -> dict[str, MetricFn]:
    """지정한 임계값으로 고정한 지표 묶음.

    운영 임계값이 0.5가 아닌 모델에 필수다. 임계값 무관 지표(auc, pr_auc)는
    그대로 두고, 임계값 의존 지표만 부분 적용한다.
    """
    return {
        "auc": roc_auc,
        "pr_auc": pr_auc,
        "sensitivity": lambda y, s: sensitivity(y, s, threshold),
        "specificity": lambda y, s: specificity(y, s, threshold),
        "precision_ppv": lambda y, s: precision_ppv(y, s, threshold),
        "npv": lambda y, s: npv(y, s, threshold),
        "f1": lambda y, s: f1(y, s, threshold),
        "balanced_accuracy": lambda y, s: balanced_accuracy(y, s, threshold),
    }


#: 임계값 0.5 기준 기본 묶음. 운영 임계값이 다르면 `binary_metrics_at`을 쓸 것.
BINARY_METRICS: dict[str, MetricFn] = {
    "auc": roc_auc,
    "pr_auc": pr_auc,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "precision_ppv": precision_ppv,
    "npv": npv,
    "f1": f1,
    "balanced_accuracy": balanced_accuracy,
}


# ---------------------------------------------------------------------------
# 다중분류 (Cancer Type ID 등) — y_score는 (n, k) 확률 행렬
# ---------------------------------------------------------------------------

def macro_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """one-vs-rest macro AUC. 클래스가 하나뿐이거나 어떤 클래스가 비면 NaN."""
    y_score = np.asarray(y_score, dtype=float)
    if y_score.ndim != 2 or len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro",
                                   labels=list(range(y_score.shape[1]))))
    except ValueError:
        return float("nan")


def macro_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pred = np.asarray(y_score).argmax(axis=1)
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def multiclass_accuracy(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pred = np.asarray(y_score).argmax(axis=1)
    return float(np.mean(np.asarray(y_true) == pred))


#: 다중분류 기본 묶음.
MULTICLASS_METRICS: dict[str, MetricFn] = {
    "macro_auc": macro_roc_auc,
    "macro_f1": macro_f1,
    "accuracy": multiclass_accuracy,
}
