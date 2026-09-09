"""AECD API 기반 데이터 로더.

기존 파일 기반 로더(로컬 CSV + 임상 xlsx)를 대체하여, 스펙트럼/임상 정보를
AECD Data API에서 가져온다. **다운스트림 인터페이스는 그대로 유지**한다:

    - ClinicalSample, SubjectSpectrum  (dataclass)
    - preprocess_arrays()              (전처리 파이프라인)
    - build_boramae_subjects()         (subject 단위 평균 스펙트럼)
    - build_aligned_boramae_subjects() (calibrate_spectrum 적용 버전)
    - group_matrix()                   (그룹별 행렬)

⚠️  '# CONFIRM' 로 표시된 부분은 실제 API 응답 스키마 확인 후 확정 필요.
"""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Iterator

import httpx
import numpy as np
from scipy.signal import savgol_filter

# 로컬 전처리 유틸은 그대로 재사용 (API는 '원본에 가까운' 스펙트럼을 주는 것으로 가정)
from sers.io import read_spectrum  # noqa: E402
from sers.preprocessing import calibrate_spectrum  # noqa: E402
from sers.signal import baseline_correction, snv  # noqa: E402

# ------------------------------------------------------------------ #
# Configuration (notebook 블록과 동일 규약)
# ------------------------------------------------------------------ #
API_BASE_URL: Final = os.environ.get("AECD_API_BASE_URL", "http://127.0.0.1:8000")
API_KEY: Final = os.environ.get("AECD_API_KEY")
DEMO_MODE: Final = os.environ.get("AECD_NOTEBOOK_DEMO", "0") == "1"

# 쿼리 필터 (원하는 코호트를 지정; None이면 전체)
SITE_CODE: str | None = None
COHORT_GROUP: str | None = None
CANCER_TYPE: str | None = None
PAGE_SIZE: Final = 500
MAX_SPECTRA: Final = 100_000

# 모델 공용 그리드는 여전히 로컬 아티팩트를 사용 (필요하면 API로도 교체 가능)
REPO: Final = Path(__file__).resolve().parents[4]
MODEL_GRID: Final = REPO / "artifacts" / "usersnet" / "v1.0.0" / "common_grid.npy"

LABELS: Final = ["Control", "Biopsy-negative", "Prostate cancer"]
SHORT_LABELS: Final = ["Control", "Biopsy-negative", "Prostate"]
COLORS: Final = {
    "Control": "#2C7FB8",
    "Biopsy-negative": "#7A5195",
    "Prostate cancer": "#D95F02",
}

# aecd_platform clinical.diagnoses.cohort_group -> 분석 라벨.
# 값은 BORAMAE site의 실제 DB 값 (2026-09-09 확인: control 28 / prostate disease
# control 61 / prostate 53 / Drop 1 subjects). 'prostate disease control'이
# 논문의 Biopsy-negative(PSA↑/Bx−) 그룹이며, 리포 내 다른 AECD 분석 스크립트와
# 같은 규약이다.
GROUP_MAP: Final = {
    "control": "Control",
    "prostate disease control": "Biopsy-negative",
    "prostate": "Prostate cancer",
}
# v7 임상 워크북에서 제외 판정된 subject는 cohort_group='Drop'으로 적재된다.
EXCLUDED_COHORT_GROUPS: Final = frozenset({"Drop"})


def parse_group(cohort_group: str | None) -> str:
    """API/DB의 cohort_group 코드값을 분석 라벨로 변환한다.

    cohort_group이 없거나 제외 코드이면 "Excluded"를 돌려주고, 그 밖의 알 수 없는
    값은 조용히 버리지 않고 RuntimeError로 올린다 — 코드값이 바뀌었을 때 한 그룹이
    통째로 사라지는 것을 막기 위해서다.
    """
    if cohort_group is None or cohort_group in EXCLUDED_COHORT_GROUPS:
        return "Excluded"
    try:
        return GROUP_MAP[cohort_group]
    except KeyError:
        msg = f"Unknown cohort_group: {cohort_group!r}"
        raise RuntimeError(msg) from None


# ------------------------------------------------------------------ #
# Dataclasses (기존과 동일 시그니처 유지)
# ------------------------------------------------------------------ #
@dataclass(frozen=True, slots=True)
class ClinicalSample:
    label: str
    sample_no: str
    group: str
    grade_group: int | None
    excluded: bool


@dataclass(frozen=True, slots=True)
class SubjectSpectrum:
    sample: ClinicalSample
    mean_spectrum: np.ndarray
    replicate_spectra: np.ndarray


@dataclass(frozen=True, slots=True)
class CleanProSpectra:
    spectra: np.ndarray
    aligned_spectra: np.ndarray
    shifts: np.ndarray


# ------------------------------------------------------------------ #
# API client
# ------------------------------------------------------------------ #
class AecdApiClient:
    """AECD Data API용 얇은 httpx 래퍼 (페이지네이션 포함)."""

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        api_key: str | None = API_KEY,
        timeout: float = 60.0,
    ) -> None:
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            # CONFIRM: 인증 헤더 형식 (Bearer vs x-api-key)
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    # context manager
    def __enter__(self) -> "AecdApiClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _paginate(self, path: str, params: dict[str, Any] | None) -> Iterator[dict[str, Any]]:
        """page/page_size 기반 페이지네이션 (CONFIRM: cursor/offset이면 교체)."""
        page = 1
        fetched = 0
        while True:
            query = dict(params or {})
            query.update({"page": page, "page_size": PAGE_SIZE})
            resp = self._client.get(path, params=query)
            resp.raise_for_status()
            payload = resp.json()
            # CONFIRM: 응답 envelope 키 ("items" / "results" / "data")
            items = payload.get("items") or payload.get("results") or []
            if not items:
                break
            for item in items:
                yield item
                fetched += 1
                if fetched >= MAX_SPECTRA:
                    return
            # CONFIRM: 다음 페이지 판단 방식
            has_next = payload.get("has_next")
            if has_next is None:
                has_next = len(items) == PAGE_SIZE
            if not has_next:
                break
            page += 1

    def iter_spectra(
        self,
        *,
        site_code: str | None = None,
        cohort_group: str | None = None,
        cancer_type: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """스펙트럼 레코드 스트림. CONFIRM: 실제 엔드포인트 경로."""
        params: dict[str, Any] = {}
        if site_code:
            params["site_code"] = site_code
        if cohort_group:
            params["cohort_group"] = cohort_group
        if cancer_type:
            params["cancer_type"] = cancer_type
        yield from self._paginate("/v1/spectra", params)  # CONFIRM 경로


# ------------------------------------------------------------------ #
# 레코드 -> 배열/도메인 객체 변환
# ------------------------------------------------------------------ #
def _record_arrays(record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """스펙트럼 레코드에서 (wavenumber, intensity) 추출.

    CONFIRM: 실제 키 이름 (예: 'wavenumbers'/'intensities' 또는
             'x'/'y' 또는 'raman_shift'/'counts').
    """
    x = np.asarray(record["wavenumbers"], dtype=float)  # CONFIRM
    y = np.asarray(record["intensities"], dtype=float)  # CONFIRM
    return x, y


def _record_sample(record: dict[str, Any]) -> ClinicalSample:
    """스펙트럼/임상 레코드에서 ClinicalSample 구성.

    CONFIRM: 각 필드에 매핑되는 API 키.
    """
    label = str(record.get("solum_label") or record.get("subject_label") or record["subject_id"])
    grade_raw = record.get("grade_group")
    return ClinicalSample(
        label=label,
        sample_no=str(record.get("sample_no") or label.split()[-1]),
        group=parse_group(record.get("cohort_group")),
        grade_group=int(grade_raw) if isinstance(grade_raw, int) else None,
        # CONFIRM: API에 'excluded' 플래그가 있는지 (원본은 셀 배경색으로 판단)
        excluded=bool(record.get("excluded", False)),
    )


def _subject_key(record: dict[str, Any]) -> str:
    """리플리케이트를 묶는 subject 식별자. CONFIRM: 실제 키."""
    return str(record.get("subject_id") or record.get("solum_label"))


# ------------------------------------------------------------------ #
# 전처리 (기존과 동일 파이프라인 — 로직 변경 없음)
# ------------------------------------------------------------------ #
def preprocess_arrays(
    x: np.ndarray, y: np.ndarray, grid: np.ndarray
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    mask = (x >= float(grid.min())) & (x <= float(grid.max()))
    x_trim = x[mask]
    y_trim = y[mask]
    y_smooth = savgol_filter(y_trim, window_length=11, polyorder=3, mode="interp")
    y_base = baseline_correction(y_smooth, window=101)
    y_snv = snv(y_base)
    y_grid = np.interp(grid, x_trim, y_snv)
    return {
        "Raw": (x, y),
        "Trimmed": (x_trim, y_trim),
        "Smoothed": (x_trim, y_smooth),
        "Baseline corrected": (x_trim, y_base),
        "SNV + model grid": (grid, y_grid),
    }, y_grid


def preprocess_file(
    path: Path, grid: np.ndarray
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Local-file compat wrapper kept for `powder_comparison/data.py`.

    Restored 2026-09-02: the AECD-API migration (b6fcf97) dropped this
    while keeping `preprocess_arrays` — `powder_comparison` still reads
    raw replicate CSV files directly (not from the API), so it needs a
    path-based entry point. Behavior is identical to the pre-migration
    version: `read_spectrum` still lives in `sers.io`, untouched by the
    API migration.
    """
    return preprocess_arrays(*read_spectrum(path), grid)


def _preprocess_record(record: dict[str, Any], grid: np.ndarray) -> np.ndarray:
    return preprocess_arrays(*_record_arrays(record), grid)[1]


def _preprocess_aligned_record(
    record: dict[str, Any], grid: np.ndarray
) -> tuple[np.ndarray, float]:
    x, y = _record_arrays(record)
    aligned_x, aligned_y, shift = calibrate_spectrum(x, y, window=20.0)
    return preprocess_arrays(aligned_x, aligned_y, grid)[1], shift


# ------------------------------------------------------------------ #
# Subject 빌더 (기존 build_*_subjects 대응)
# ------------------------------------------------------------------ #
def _grouped_records(
    client: AecdApiClient,
    *,
    site_code: str | None,
    cohort_group: str | None,
    cancer_type: str | None,
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in client.iter_spectra(
        site_code=site_code, cohort_group=cohort_group, cancer_type=cancer_type
    ):
        grouped[_subject_key(record)].append(record)
    return grouped


def build_boramae_subjects(
    grid: np.ndarray,
    *,
    client: AecdApiClient | None = None,
    site_code: str | None = SITE_CODE,
    cohort_group: str | None = COHORT_GROUP,
    cancer_type: str | None = CANCER_TYPE,
) -> list[SubjectSpectrum]:
    """API에서 subject 단위 평균 스펙트럼 구성 (원본 함수 대체)."""
    owns_client = client is None
    client = client or AecdApiClient()
    try:
        grouped = _grouped_records(
            client, site_code=site_code, cohort_group=cohort_group, cancer_type=cancer_type
        )
        subjects: list[SubjectSpectrum] = []
        for _, records in grouped.items():
            sample = _record_sample(records[0])
            if sample.excluded or sample.group == "Excluded":
                continue
            mat = np.vstack([_preprocess_record(r, grid) for r in records])
            subjects.append(
                SubjectSpectrum(sample=sample, mean_spectrum=mat.mean(axis=0), replicate_spectra=mat)
            )
        return subjects
    finally:
        if owns_client:
            client.close()


def build_aligned_boramae_subjects(
    grid: np.ndarray,
    *,
    client: AecdApiClient | None = None,
    site_code: str | None = SITE_CODE,
    cohort_group: str | None = COHORT_GROUP,
    cancer_type: str | None = CANCER_TYPE,
) -> tuple[list[SubjectSpectrum], np.ndarray]:
    """calibrate_spectrum 적용 버전 (원본 aligned 함수 대체)."""
    owns_client = client is None
    client = client or AecdApiClient()
    try:
        grouped = _grouped_records(
            client, site_code=site_code, cohort_group=cohort_group, cancer_type=cancer_type
        )
        subjects: list[SubjectSpectrum] = []
        shifts: list[np.ndarray] = []
        for _, records in grouped.items():
            sample = _record_sample(records[0])
            if sample.excluded or sample.group == "Excluded":
                continue
            aligned = [_preprocess_aligned_record(r, grid) for r in records]
            matrix = np.vstack([item[0] for item in aligned])
            shifts.append(np.asarray([item[1] for item in aligned], dtype=float))
            subjects.append(
                SubjectSpectrum(
                    sample=sample, mean_spectrum=matrix.mean(axis=0), replicate_spectra=matrix
                )
            )
        return subjects, (np.vstack(shifts) if shifts else np.empty((0, 0)))
    finally:
        if owns_client:
            client.close()


def group_matrix(subjects: list[SubjectSpectrum], group: str) -> np.ndarray:
    return np.vstack([sub.mean_spectrum for sub in subjects if sub.sample.group == group])
