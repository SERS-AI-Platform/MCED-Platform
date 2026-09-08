from __future__ import annotations

import scripts.db.aecd_model_lineage_timeline_core as timeline


def test_build_timeline_orders_versions_and_exposes_changed_hyperparameters() -> None:
    # Given: two historical versions from one model family with one changed hyperparameter.
    first = timeline.TimelineRun(
        run_id="run-1",
        registry_model_name="aecd-legacy-resnet18_1d",
        registry_model_version="1",
        source_dir="results/v001",
        source_model_version="v001",
        model_display_name="ResNet18-1D",
        params={
            "timestamp": "2026-03-01T10:00:00",
            "config_learning_rate": "0.001",
            "config_resnet_blocks": "3",
        },
        metrics={"historical_val_s1_auc": 0.71},
    )
    second = timeline.TimelineRun(
        run_id="run-2",
        registry_model_name="aecd-legacy-resnet18_1d",
        registry_model_version="2",
        source_dir="results/v002",
        source_model_version="v002",
        model_display_name="ResNet18-1D",
        params={
            "timestamp": "2026-03-02T10:00:00",
            "config_learning_rate": "0.0005",
            "config_resnet_blocks": "3",
        },
        metrics={"historical_val_s1_auc": 0.72},
    )

    # When: the source-time timeline is built.
    rows = timeline.build_timeline((second, first))

    # Then: source chronology, normalized metric, and the changed parameter are explicit.
    assert [row.history_order for row in rows] == [1, 2]
    assert rows[0].source_timestamp_epoch < rows[1].source_timestamp_epoch
    assert rows[1].changed_hyperparameters == ("config_learning_rate",)
    assert rows[1].cancer_screening_auc == 0.72


def test_build_timeline_places_missing_source_time_after_observed_versions() -> None:
    # Given: one observed historical version and one checkpoint-only version without a timestamp.
    observed = timeline.TimelineRun(
        run_id="run-observed",
        registry_model_name="aecd-legacy-cnn1d_shallow",
        registry_model_version="1",
        source_dir="results/observed",
        source_model_version="v001",
        model_display_name="CNN1D-Shallow",
        params={"timestamp": "2026-03-01T10:00:00"},
        metrics={},
    )
    missing = timeline.TimelineRun(
        run_id="run-missing",
        registry_model_name="aecd-legacy-cnn1d_shallow",
        registry_model_version="2",
        source_dir="results/missing",
        source_model_version="unversioned",
        model_display_name="CNN1D-Shallow",
        params={},
        metrics={},
    )

    # When: the source-time timeline is built.
    rows = timeline.build_timeline((missing, observed))

    # Then: missing source time is visible and does not fabricate chronology.
    assert [row.run_id for row in rows] == ["run-observed", "run-missing"]
    assert rows[1].source_time_status == "missing"
    assert rows[1].source_timestamp_epoch is None


def test_parameter_change_rows_aggregate_changed_hyperparameters() -> None:
    # Given: timeline rows with repeated changes to two hyperparameters.
    rows = (
        timeline.TimelineRow(
            run_id="run-1",
            registry_model_name="family-a",
            registry_model_version="1",
            source_dir="results/v001",
            source_model_version="v001",
            model_display_name="Model A",
            source_timestamp="2026-03-01T10:00:00",
            source_timestamp_epoch=1772359200.0,
            source_time_status="observed",
            history_order=1,
            hyperparameter_change_count=0,
            changed_hyperparameters=(),
            cancer_screening_auc=0.70,
            cancer_type_id_auc=None,
        ),
        timeline.TimelineRow(
            run_id="run-2",
            registry_model_name="family-a",
            registry_model_version="2",
            source_dir="results/v002",
            source_model_version="v002",
            model_display_name="Model A",
            source_timestamp="2026-03-02T10:00:00",
            source_timestamp_epoch=1772445600.0,
            source_time_status="observed",
            history_order=2,
            hyperparameter_change_count=2,
            changed_hyperparameters=("config_learning_rate", "config_resnet_blocks"),
            cancer_screening_auc=0.72,
            cancer_type_id_auc=None,
        ),
        timeline.TimelineRow(
            run_id="run-3",
            registry_model_name="family-a",
            registry_model_version="3",
            source_dir="results/v003",
            source_model_version="v003",
            model_display_name="Model A",
            source_timestamp="2026-03-03T10:00:00",
            source_timestamp_epoch=1772532000.0,
            source_time_status="observed",
            history_order=3,
            hyperparameter_change_count=1,
            changed_hyperparameters=("config_learning_rate",),
            cancer_screening_auc=0.73,
            cancer_type_id_auc=None,
        ),
    )

    # When: the parameter-change matrix is built.
    changes = timeline.parameter_change_rows(rows)

    # Then: each parameter is counted independently and sorted by change frequency.
    assert changes[0].parameter_name == "config_learning_rate"
    assert changes[0].change_count == 2
    assert changes[1].parameter_name == "config_resnet_blocks"
    assert changes[1].change_count == 1
