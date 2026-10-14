"""run_guide_pipeline.py 기록용 인자 4개(--date-tag/--phase/--notes/--baseline-ref).

핵심: 인자를 생략하면 PL-3 당시 고정값과 문자 하나까지 같은 기록이 나와야 하고,
기존 산출물을 다른 태그로 재적재하려 하면 중단해야 한다.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "analysis" / "preprocessing_lab"
sys.path.insert(0, str(SCRIPT_DIR))

import run_guide_pipeline as G  # noqa: E402

META = {
    "run_name": "guide_aecd_production_20260907", "method_key": None, "condition": "production",
    "cohort": "aecd", "label": "현재 프로덕션: cal✓ despike✗ rolling_min SNV",
    "preprocessing": {}, "calibrate": True, "grid": "g", "model": "m", "cv": "cv",
    "seeds": [42], "primary_seed": 42, "calibration_shift_cm1": None, "data_query_filters": {},
    "qc": {"n_patients": 112, "final": 10413, "dropped": 3139},
    "started_at": "2026-09-07T00:00:00+00:00", "finished_at": "2026-09-07T01:00:00+00:00",
}


class _FakeCursor:
    def __init__(self, log: list) -> None:
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:
        self.log.append((sql, params))

    def fetchone(self):
        return None  # runs 행 없음 → create_run 경로


class _FakeConn(_FakeCursor):
    def cursor(self):
        return _FakeCursor(self.log)

    def commit(self) -> None:
        return None


class _FakeTracker:
    def __init__(self) -> None:
        self.sql: list = []
        self.created: dict = {}

    def _connect(self):
        return _FakeConn(self.sql)

    def select_method_id(self, key):
        return None

    def create_run(self, name, **fields):
        self.created = {"name": name, **fields}
        return 7

    def link_measurements(self, run_id, ids):
        return 0

    def finish_run(self, run_id, status, finished_at):
        return None

    def load_metric_rows(self, rows, git_commit=None):
        return 0

    def record_metrics(self, run_id, rows):
        return 0


def _load(tags: G.RecordTags) -> _FakeTracker:
    tracker = _FakeTracker()
    rows = [SimpleNamespace(cohort=SimpleNamespace(cancer_groups=("prostate",),
                                                   control_groups=("control",)))]
    out_dir = tags.out_root / "aecd" / "production"
    G._load_run(tracker, META, rows, out_dir, "abc1234", tags)
    return tracker


def _update_params(tracker: _FakeTracker) -> tuple:
    return next(p for sql, p in tracker.sql if sql.startswith("UPDATE experiment.runs"))


def test_defaults_reproduce_pl3_fixed_values() -> None:
    tags = G.record_tags(G.DEFAULT_TAGS.date_tag, G.DEFAULT_TAGS.phase, None,
                         G.DEFAULT_TAGS.baseline_ref)
    assert tags == G.DEFAULT_TAGS
    assert tags.out_root == G.OUT_ROOT
    assert G.run_name("aecd", "production", tags.date_tag) == "guide_aecd_production_20260907"

    tracker = _load(tags)
    out_dir = G.OUT_ROOT / "aecd" / "production"
    # 인자 추가 전 _load_run에 하드코딩돼 있던 문구 그대로
    legacy_notes = (f"PL-3 설계가이드 요인별 비교 [{META['cohort']}] {META['label']}. "
                    f"LR C=1.0 고정(PL-1 규약). 탐색 실험 — audit gate 면제(사용자 결정 2026-09-08), "
                    f"채택 근거 아님. 산출물 {out_dir.relative_to(G.REPO)}")
    assert tracker.created["notes"] == legacy_notes
    params = _update_params(tracker)
    assert params[3] == "PL-3" and params[5] == "cal_despike"


def test_custom_tags_change_only_record_fields() -> None:
    tags = G.record_tags("20261001", "PL-4", "그래프 승인 run", "cal_ps_si")
    assert tags.out_root.name == "guide_pipeline_20261001"
    assert G.run_name("aecd", "production", tags.date_tag) == "guide_aecd_production_20261001"

    tracker = _load(tags)
    assert tracker.created["notes"].startswith("그래프 승인 run [aecd]")
    assert "audit gate 면제" not in tracker.created["notes"]
    params = _update_params(tracker)
    assert params[3] == "PL-4" and params[5] == "cal_ps_si"


@pytest.mark.parametrize(("args", "message"), [
    (("2026/10/01", "PL-4", None, "cal_ps_si"), "date-tag"),
    (("20261001", " ", None, "cal_ps_si"), "phase"),
    (("20261001", "PL-4", " ", "cal_ps_si"), "notes"),
    (("20261001", "PL-4", None, "no_such_condition"), "baseline-ref"),
])
def test_invalid_tags_are_rejected(args: tuple, message: str) -> None:
    with pytest.raises(SystemExit, match=message):
        G.record_tags(*args)


def test_legacy_output_cannot_be_reloaded_with_other_tags(tmp_path: Path) -> None:
    legacy = dict(META)  # PL-3 산출물: record_tags 키 없음
    G._check_tags(legacy, G.DEFAULT_TAGS, tmp_path)
    with pytest.raises(SystemExit, match="record_tags"):
        G._check_tags(legacy, G.RecordTags(phase="PL-4"), tmp_path)


def test_saved_tags_must_match(tmp_path: Path) -> None:
    tags = G.RecordTags(date_tag="20261001", phase="PL-4", notes="n", baseline_ref="cal_ps_si")
    meta = {**META, "record_tags": G.dataclasses.asdict(tags)}
    G._check_tags(meta, tags, tmp_path)
    with pytest.raises(SystemExit, match="record_tags"):
        G._check_tags(meta, G.RecordTags(date_tag="20261001", phase="PL-5", notes="n",
                                         baseline_ref="cal_ps_si"), tmp_path)


def test_cohort_id_follows_date_tag() -> None:
    import pandas as pd

    pt = pd.DataFrame({"subject": ["a", "b"], "true_label": [1, 0], "prob": [0.9, 0.1],
                       "n_spectra": [100, 100]})
    default = G._cohort_spec(META, pt, "aecd", G.Task.CANCER_SCREENING)
    custom = G._cohort_spec(META, pt, "aecd", G.Task.CANCER_SCREENING, "20261001")
    assert default.cohort_id == "aecd_prostate3_20260907"
    assert custom.cohort_id == "aecd_prostate3_20261001"
