from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from click.testing import CliRunner

from scripts.training.train_raw_replicate_set import main as train_raw_set
from sers.raw_set.audit import audit_average_file
from sers.raw_set.data import (
    CohortSpec,
    PatientRecord,
    RawPatientDataset,
    load_patient_records,
)
from sers.raw_set.model import RawSetConfig, RawSetModel, RawSetTargets, raw_set_loss
from sers.raw_set.training import (
    FoldData,
    RawTrainingConfig,
    fit_raw_set,
    predict_raw_set,
)


def _write_spectrum(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    np.savetxt(path, np.column_stack([x, y]), delimiter=",", fmt="%.8f")


def test_average_target_is_usable_when_it_matches_five_replicates(tmp_path: Path) -> None:
    # Given
    x = np.linspace(400.0, 2200.0, 17)
    spectra = [np.sin(x / 100.0) + replicate for replicate in range(5)]
    for replicate, y in enumerate(spectra, start=1):
        _write_spectrum(tmp_path / f"NOR 1_{replicate}.CSV", x, y)
    average_dir = tmp_path / "Averaged data"
    average_dir.mkdir()
    average_path = average_dir / "NOR 1_ave.CSV"
    _write_spectrum(average_path, x, np.mean(spectra, axis=0))

    # When
    result = audit_average_file(average_path)

    # Then
    assert result.replicate_count == 5
    assert result.target_usable


def test_average_target_is_rejected_when_it_differs_from_replicates(tmp_path: Path) -> None:
    # Given
    x = np.linspace(400.0, 2200.0, 17)
    spectra = [np.sin(x / 100.0) + replicate for replicate in range(5)]
    for replicate, y in enumerate(spectra, start=1):
        _write_spectrum(tmp_path / f"DIA 9_{replicate}.CSV", x, y)
    average_dir = tmp_path / "Averaged data"
    average_dir.mkdir()
    average_path = average_dir / "DIA 9_ave.CSV"
    _write_spectrum(average_path, x, np.mean(spectra, axis=0) + 10.0)

    # When
    result = audit_average_file(average_path)

    # Then
    assert not result.target_usable
    assert result.normalized_rmse > 1e-4


def test_raw_set_model_is_invariant_to_replicate_order_and_padding() -> None:
    # Given
    torch.manual_seed(7)
    config = RawSetConfig(n_wavenumbers=64, channels=8, latent_dim=12)
    model = RawSetModel(config).eval()
    spectra = torch.randn(2, 5, 64)
    mask = torch.tensor([[True, True, True, True, True], [True, True, True, False, False]])

    # When
    original = model(spectra, mask)
    order = torch.tensor([2, 0, 4, 1, 3])
    permuted = model(spectra[:, order], mask[:, order])
    padded_changed = spectra.clone()
    padded_changed[1, 3:] = 1_000_000.0
    changed = model(padded_changed, mask)

    # Then
    torch.testing.assert_close(original.logits, permuted.logits)
    torch.testing.assert_close(original.logits[1], changed.logits[1])
    torch.testing.assert_close(original.attention.sum(dim=1), torch.ones(2))


def test_raw_set_loss_uses_only_valid_average_targets() -> None:
    # Given
    torch.manual_seed(11)
    config = RawSetConfig(n_wavenumbers=32, channels=8, latent_dim=12)
    model = RawSetModel(config)
    spectra = torch.randn(2, 5, 32)
    mask = torch.ones(2, 5, dtype=torch.bool)
    output = model(spectra, mask)
    target = torch.randn(2, 32)
    labels = torch.tensor([0.0, 1.0])

    # When
    with_target = raw_set_loss(
        output,
        RawSetTargets(
            labels=labels,
            average=target,
            average_mask=torch.tensor([True, False]),
            replicate_mask=mask,
        ),
    )
    without_target = raw_set_loss(
        output,
        RawSetTargets(
            labels=labels,
            average=target,
            average_mask=torch.zeros(2, dtype=torch.bool),
            replicate_mask=mask,
        ),
    )

    # Then
    assert with_target.reconstruction.item() > 0
    assert without_target.reconstruction.item() == pytest.approx(0.0)


def test_patient_loader_connects_numeric_replicates_and_trusted_average(
    tmp_path: Path,
) -> None:
    # Given
    cohort = tmp_path / "normal"
    cohort.mkdir()
    x = np.linspace(400.0, 2200.0, 17)
    spectra = [np.cos(x / 100.0) + replicate for replicate in range(5)]
    for replicate, y in enumerate(spectra, start=1):
        _write_spectrum(cohort / f"NOR 1_{replicate}.CSV", x, y)
    average_dir = cohort / "Averaged data"
    average_dir.mkdir()
    _write_spectrum(average_dir / "NOR 1_ave.CSV", x, np.mean(spectra, axis=0))
    spec = CohortSpec(folder="normal", group="NOR", cancer=False)

    # When
    records = load_patient_records(tmp_path, (spec,), x)

    # Then
    assert len(records) == 1
    assert records[0].replicates.shape == (5, 17)
    assert records[0].average_usable


def test_patient_dataset_pads_missing_replicates_without_using_derived_average() -> None:
    # Given
    records = [
        PatientRecord(
            patient_key="PRO:1",
            group="PRO",
            cancer=True,
            replicates=np.ones((3, 16), dtype=np.float32),
            average=np.full(16, 99.0, dtype=np.float32),
            average_usable=True,
        )
    ]

    # When
    item = RawPatientDataset(records, max_replicates=5)[0]

    # Then
    assert item.spectra.shape == (5, 16)
    assert item.replicate_mask.tolist() == [True, True, True, False, False]
    assert torch.all(item.spectra[3:] == 0)
    assert torch.all(item.average == 99.0)


def test_raw_set_training_produces_patient_level_probabilities() -> None:
    # Given
    rng = np.random.default_rng(19)
    records = []
    for index in range(8):
        cancer = index >= 4
        signal = rng.normal(size=(3, 32)).astype(np.float32) + 2.0 * cancer
        records.append(
            PatientRecord(
                patient_key=f"{'PRO' if cancer else 'NOR'}:{index}",
                group="PRO" if cancer else "NOR",
                cancer=cancer,
                replicates=signal,
                average=signal.mean(axis=0),
                average_usable=True,
            )
        )
    data = FoldData(
        train=tuple(records[index] for index in (0, 1, 2, 4, 5, 6)),
        validation=tuple(records[index] for index in (3, 7)),
    )
    config = RawTrainingConfig(
        model=RawSetConfig(n_wavenumbers=32, channels=8, latent_dim=12),
        epochs=1,
        batch_size=2,
        device="cpu",
    )

    # When
    fitted = fit_raw_set(data, config)
    predictions = predict_raw_set(fitted, data.validation)

    # Then
    assert predictions.probabilities.shape == (2,)
    assert np.isfinite(predictions.probabilities).all()


def test_training_cli_runs_with_click_options(tmp_path: Path) -> None:
    # Given
    data_root = tmp_path / "raw"
    prostate = data_root / "1. Prostate cancer (100개)"
    normal = data_root / "5. Normal (100개)"
    prostate.mkdir(parents=True)
    normal.mkdir()
    x = np.linspace(400.0, 2200.0, 32)
    for folder, group, offset in ((prostate, "PRO", 2.0), (normal, "NOR", 0.0)):
        for sample in (1, 2):
            for replicate in range(1, 6):
                y = np.sin(x / 100.0) + offset + sample * 0.01 + replicate * 0.001
                _write_spectrum(folder / f"{group} {sample}_{replicate}.CSV", x, y)
    out_dir = tmp_path / "output"

    # When
    result = CliRunner().invoke(
        train_raw_set,
        [
            "--data-root",
            str(data_root),
            "--config",
            "config/config.yaml",
            "--out-dir",
            str(out_dir),
            "--group",
            "PRO",
            "--group",
            "NOR",
            "--folds",
            "2",
            "--epochs",
            "1",
            "--grid-points",
            "128",
            "--device",
            "cpu",
        ],
    )

    # Then
    assert result.exit_code == 0, result.output
    assert (out_dir / "fold_metrics.csv").is_file()
