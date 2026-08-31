from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import label_binarize

CLASS_NAMES = ("control", "prostate disease control", "prostate")
CLASS_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
N_CLASSES = len(CLASS_NAMES)
OUTER_FOLDS = 5
INNER_FOLDS = 5
RANDOM_STATE = 20260825


class EvaluationError(RuntimeError):
    pass


def expanded_probability(model, values: np.ndarray) -> np.ndarray:
    probabilities = model.predict_proba(values)
    expanded = np.zeros((len(values), N_CLASSES), dtype=np.float64)
    for column, class_value in enumerate(model.classes_):
        expanded[:, int(class_value)] = probabilities[:, column]
    return expanded


def fit_model(name: str, factory, values, labels, weights):
    model = factory()
    if weights is None:
        model.fit(values, labels)
    elif name.startswith(("lr_", "ridge_")):
        model.fit(values, labels, logisticregression__sample_weight=weights)
    else:
        model.fit(values, labels, sample_weight=weights)
    return model


def macro_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        roc_auc_score(
            label_binarize(labels, classes=np.arange(N_CLASSES)),
            probabilities,
            average="macro",
        )
    )


def run_nested_multiclass(views, labels, groups, base_specs, weights=None):
    groups = np.asarray(groups, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    reference_view = next(iter(views.values()))
    outer = GroupKFold(n_splits=OUTER_FOLDS)
    oof = np.full((len(labels), N_CLASSES), np.nan, dtype=np.float64)
    fold_rows = []

    for fold, (train_index, test_index) in enumerate(
        outer.split(reference_view, labels, groups),
        start=1,
    ):
        train_groups = groups[train_index]
        train_weights = None if weights is None else weights[train_index]
        inner = GroupKFold(n_splits=INNER_FOLDS)
        meta_train = np.zeros(
            (len(train_index), len(base_specs) * N_CLASSES), dtype=np.float64
        )
        for model_index, (name, view_name, factory) in enumerate(base_specs):
            column = np.zeros((len(train_index), N_CLASSES), dtype=np.float64)
            train_values = views[view_name][train_index]
            train_labels = labels[train_index]
            for inner_train, inner_test in inner.split(
                train_values, train_labels, train_groups
            ):
                inner_weights = (
                    None if train_weights is None else train_weights[inner_train]
                )
                model = fit_model(
                    name,
                    factory,
                    train_values[inner_train],
                    train_labels[inner_train],
                    inner_weights,
                )
                column[inner_test] = expanded_probability(
                    model, train_values[inner_test]
                )
            start_column = model_index * N_CLASSES
            meta_train[:, start_column : start_column + N_CLASSES] = column

        meta = LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            random_state=RANDOM_STATE,
        )
        meta.fit(meta_train, labels[train_index], sample_weight=train_weights)
        meta_test = np.zeros(
            (len(test_index), len(base_specs) * N_CLASSES), dtype=np.float64
        )
        for model_index, (name, view_name, factory) in enumerate(base_specs):
            model = fit_model(
                name,
                factory,
                views[view_name][train_index],
                labels[train_index],
                train_weights,
            )
            start_column = model_index * N_CLASSES
            meta_test[:, start_column : start_column + N_CLASSES] = expanded_probability(
                model, views[view_name][test_index]
            )
        oof[test_index] = expanded_probability(meta, meta_test)
        fold_rows.append(
            {
                "fold": fold,
                "n_train_subjects": int(np.unique(groups[train_index]).size),
                "n_test_subjects": int(np.unique(groups[test_index]).size),
                "macro_ovr_auc": macro_auc(labels[test_index], oof[test_index]),
            }
        )

    if np.isnan(oof).any():
        raise EvaluationError("Nested CV did not produce probabilities for every row.")
    return oof, fold_rows


def subject_aggregate(groups: np.ndarray, labels: np.ndarray, probabilities: np.ndarray):
    unique_groups = np.unique(groups)
    subject_labels = np.zeros(len(unique_groups), dtype=np.int64)
    subject_probabilities = np.zeros((len(unique_groups), N_CLASSES), dtype=np.float64)
    for row, group in enumerate(unique_groups):
        mask = groups == group
        subject_labels[row] = labels[np.flatnonzero(mask)[0]]
        subject_probabilities[row] = probabilities[mask].mean(axis=0)
    return subject_labels, subject_probabilities


def evaluate_condition(name: str, views, labels, groups, base_specs, weights=None):
    oof, fold_rows = run_nested_multiclass(views, labels, groups, base_specs, weights)
    if weights is None:
        evaluation_labels, evaluation_probabilities = labels, oof
        unit = "subject-mean"
    else:
        evaluation_labels, evaluation_probabilities = subject_aggregate(
            groups, labels, oof
        )
        unit = "subject"
    predictions = evaluation_probabilities.argmax(axis=1)
    matrix = confusion_matrix(
        evaluation_labels, predictions, labels=np.arange(N_CLASSES)
    )
    class_auc = roc_auc_score(
        label_binarize(evaluation_labels, classes=np.arange(N_CLASSES)),
        evaluation_probabilities,
        average=None,
    )
    fold_auc = [row["macro_ovr_auc"] for row in fold_rows]
    summary = {
        "condition": name,
        "evaluation_unit": unit,
        "n_rows": int(len(evaluation_labels)),
        "macro_ovr_auc": macro_auc(evaluation_labels, evaluation_probabilities),
        "fold_macro_auc_mean": float(np.mean(fold_auc)),
        "fold_macro_auc_std": float(np.std(fold_auc)),
        "balanced_accuracy": float(
            balanced_accuracy_score(evaluation_labels, predictions)
        ),
        "macro_f1": float(
            f1_score(evaluation_labels, predictions, average="macro", zero_division=0)
        ),
        "control_ovr_auc": float(class_auc[0]),
        "prostate_disease_control_ovr_auc": float(class_auc[1]),
        "prostate_ovr_auc": float(class_auc[2]),
    }
    fold_rows = [row | {"condition": name} for row in fold_rows]
    return summary, fold_rows, matrix


def save_rows(rows: list[dict], path: Path) -> None:
    fieldnames = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
