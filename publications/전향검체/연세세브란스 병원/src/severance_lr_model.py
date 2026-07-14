from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from severance_data import REPO, YNOR_RAW, YPAN_RAW
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sers.config import load_config
from sers.io import make_fixed_grid, read_spectrum
from sers.preprocessing import preprocess_spectra


@dataclass(frozen=True, slots=True)
class SeveranceDataset:
    subject_ids: tuple[str, ...]
    source_groups: tuple[str, ...]
    spectra: np.ndarray
    labels: np.ndarray
    grid: np.ndarray


@dataclass(frozen=True, slots=True)
class LrEvaluationConfig:
    bin_width: int = 32
    regularization_c: float = 3.0
    n_splits: int = 5
    n_repeats: int = 100
    base_seed: int = 5000


@dataclass(frozen=True, slots=True)
class FoldResult:
    repeat: int
    fold: int
    auc: float
    test_subjects: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RepeatedCvResult:
    config: LrEvaluationConfig
    repeat_auc: np.ndarray
    mean_probability: np.ndarray
    probability_sd: np.ndarray
    folds: tuple[FoldResult, ...]


def bin_spectra(spectra: np.ndarray, width: int) -> np.ndarray:
    usable = (spectra.shape[1] // width) * width
    return spectra[:, :usable].reshape(len(spectra), -1, width).mean(axis=2)


def load_unique_severance_dataset() -> SeveranceDataset:
    raw: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}
    for group, root in (("YNOR", YNOR_RAW), ("YPAN", YPAN_RAW)):
        for path in sorted(root.glob("*.CSV")):
            if path.stem.lower().endswith("_ave"):
                continue
            label, replicate = path.stem.rsplit("_", maxsplit=1)
            sample_id = label.split()[-1]
            raw[(group, sample_id, replicate)] = read_spectrum(path)
    config = load_config(str(REPO / "config" / "config.yaml"))
    processed, _, grid = preprocess_spectra(
        raw, make_fixed_grid(config), config, qc_passed_keys=None
    )
    grouped: dict[tuple[str, str], list[np.ndarray]] = {}
    for (group, sample_id, _), spectrum in processed.items():
        grouped.setdefault((group, sample_id), []).append(spectrum)
    ordered = sorted(grouped.items())
    subject_ids = tuple(f"{group}_{sample_id}" for (group, sample_id), _ in ordered)
    source_groups = tuple(group for (group, _), _ in ordered)
    spectra = np.vstack([np.vstack(replicates).mean(axis=0) for _, replicates in ordered])
    labels = np.asarray([int(group == "YPAN") for group in source_groups], dtype=int)
    return SeveranceDataset(subject_ids, source_groups, spectra, labels, grid)


def evaluate_repeated_cv(dataset: SeveranceDataset, config: LrEvaluationConfig) -> RepeatedCvResult:
    features = bin_spectra(dataset.spectra, config.bin_width)
    probabilities = np.zeros((config.n_repeats, len(dataset.labels)), dtype=float)
    repeat_auc = np.zeros(config.n_repeats, dtype=float)
    folds: list[FoldResult] = []
    for repeat in range(config.n_repeats):
        splitter = StratifiedKFold(
            n_splits=config.n_splits,
            shuffle=True,
            random_state=config.base_seed + repeat,
        )
        for fold, (train_index, test_index) in enumerate(
            splitter.split(features, dataset.labels), start=1
        ):
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=config.regularization_c, max_iter=5000),
            )
            model.fit(features[train_index], dataset.labels[train_index])
            predicted = model.predict_proba(features[test_index])[:, 1]
            probabilities[repeat, test_index] = predicted
            folds.append(
                FoldResult(
                    repeat=repeat + 1,
                    fold=fold,
                    auc=float(roc_auc_score(dataset.labels[test_index], predicted)),
                    test_subjects=tuple(dataset.subject_ids[index] for index in test_index),
                )
            )
        repeat_auc[repeat] = roc_auc_score(dataset.labels, probabilities[repeat])
    return RepeatedCvResult(
        config=config,
        repeat_auc=repeat_auc,
        mean_probability=probabilities.mean(axis=0),
        probability_sd=probabilities.std(axis=0),
        folds=tuple(folds),
    )
