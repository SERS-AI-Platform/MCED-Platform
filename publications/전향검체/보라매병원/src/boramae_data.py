"""AECD API 기반 데이터 로더.

기존 파일 기반 로더(로컬 CSV + 임상 xlsx)를 대체하여, 스펙트럼/임상 정보를
AECD Data API에서 가져온다. **다운스트림 인터페이스는 그대로 유지**한다:

    - ClinicalSample, SubjectSpectrum  (dataclass)
    - preprocess_arrays()              (전처리 파이프라인)
    - build_boramae_subjects()         (subject 단위 평균 스펙트럼)
    - build_aligned_boramae_subjects() (calibrate_spectrum 적용 버전)
    - group_matrix()                   (그룹별 행렬)

API 계약은 `sers.aecd_api` (models.Spectrum / SpectrumPage, X-API-Key 인증,
limit/offset 페이지네이션)를 그대로 따르며, 응답은 같은 pydantic 모델로 파싱한다.
2026-09-09 실제 API/DB에 대해 확인.

데이터 범위 (2026-09-09 결정): 이 publication은 aecd_platform의 BORAMAE
2026-08-10~14 **mapping 측정(샘플당 121점)** 을 기준으로 한다. 7월 5-replicate
점 측정(BPRO/BNOR *.CSV)으로 만든 이전 산출물/수치는 이 로더로 재현되지 않으며,
mapping 데이터로 다시 생성한다. subject 평균 = 해당 subject의 모든 mapping 점 평균.
"""

from __future__ import annotations

import csv
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator

import httpx
import numpy as np
from scipy.signal import savgol_filter

from sers.aecd_api.models import Spectrum, SpectrumPage

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
# aecd_platform master.sites.site_code for this publication's cohort.
SITE_CODE: str | None = "BORAMAE"
COHORT_GROUP: str | None = None
CANCER_TYPE: str | None = None
PAGE_SIZE: Final = 500
MAX_SPECTRA: Final = 100_000

# 모델 공용 그리드는 여전히 로컬 아티팩트를 사용 (필요하면 API로도 교체 가능)
REPO: Final = Path(__file__).resolve().parents[4]
MODEL_GRID: Final = REPO / "artifacts" / "usersnet" / "v1.0.0" / "common_grid.npy"
OUT: Final = REPO / "publications" / "전향검체" / "보라매병원"
FIG_DIR: Final = OUT / "figures"
TABLE_DIR: Final = OUT / "tables"
# fig02 비교용 레거시 PRO 코호트 — 보라매 데이터가 아니라 별개 데이터셋(파일 기반 유지).
LEGACY_PRO_ROOT: Final = REPO / "data" / "raw_data" / "1. Prostate cancer (100개)"
CLEAN_MANIFEST: Final = REPO / "results" / "clean_cohort_20260605" / "clean_cohort_manifest.csv"

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
    # 전처리 단계 그림(fig01)용 원시 (wavenumber, intensity) 한 개 — 첫 mapping 점.
    example_raw: tuple[np.ndarray, np.ndarray] | None = None


@dataclass(frozen=True, slots=True)
class CleanProSpectra:
    spectra: np.ndarray
    aligned_spectra: np.ndarray
    shifts: np.ndarray


# ------------------------------------------------------------------ #
# API client
# ------------------------------------------------------------------ #
class AecdApiClient:
    """AECD Data API용 얇은 httpx 래퍼 (limit/offset 페이지네이션 포함).

    `http`에 기존 `httpx.Client`(예: FastAPI `TestClient`)를 주입하면 그 클라이언트를
    그대로 쓰고 닫지 않는다 — 계약 테스트에서 실제 서버 없이 앱에 직접 붙이기 위함.
    """

    def __init__(
        self,
        base_url: str = API_BASE_URL,
        api_key: str | None = API_KEY,
        timeout: float = 60.0,
        *,
        http: httpx.Client | None = None,
    ) -> None:
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key  # sers.aecd_api.app: APIKeyHeader(name="X-API-Key")
        self._owns_http = http is None
        self._client = http or httpx.Client(base_url=base_url, headers=headers, timeout=timeout)
        if http is not None:
            self._client.headers.update(headers)

    # context manager
    def __enter__(self) -> "AecdApiClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_http:
            self._client.close()

    def _paginate(self, path: str, params: dict[str, str]) -> Iterator[Spectrum]:
        """SpectrumPage(total/limit/offset/items) 기반 offset 페이지네이션."""
        offset = 0
        fetched = 0
        while True:
            query: dict[str, str | int] = dict(params)
            query.update({"limit": PAGE_SIZE, "offset": offset})
            resp = self._client.get(path, params=query)
            resp.raise_for_status()
            page = SpectrumPage.model_validate(resp.json())
            if not page.items:
                return
            for item in page.items:
                yield item
                fetched += 1
                if fetched >= MAX_SPECTRA:
                    return
            offset += len(page.items)
            if offset >= page.total:
                return

    def iter_spectra(
        self,
        *,
        site_code: str | None = None,
        cohort_group: str | None = None,
        cancer_type: str | None = None,
    ) -> Iterator[Spectrum]:
        """GET /v1/spectra 를 SpectrumFilters 이름 그대로 필터링해 스트리밍."""
        params: dict[str, str] = {}
        if site_code:
            params["site_code"] = site_code
        if cohort_group:
            params["cohort_group"] = cohort_group
        if cancer_type:
            params["cancer_type"] = cancer_type
        yield from self._paginate("/v1/spectra", params)


# ------------------------------------------------------------------ #
# 레코드 -> 배열/도메인 객체 변환
# ------------------------------------------------------------------ #
def _record_arrays(record: Spectrum) -> tuple[np.ndarray, np.ndarray]:
    """스펙트럼 레코드에서 (wavenumber, intensity) 추출."""
    return np.asarray(record.wavenumber, dtype=float), np.asarray(record.intensities, dtype=float)


def _record_sample(record: Spectrum) -> ClinicalSample:
    """스펙트럼 레코드에서 ClinicalSample 구성.

    API는 비식별 키(`subject:<id>`)만 주므로 label/sample_no도 그것을 쓴다 — 논문
    산출물에 원본 검체 코드(BPRO/BNOR n)가 남지 않는다. 제외 판정은 v7 워크북의
    cohort_group='Drop'으로 적재돼 있어 parse_group()이 "Excluded"로 돌려준다.
    """
    group = parse_group(record.cohort_group)
    return ClinicalSample(
        label=record.subject_key,
        sample_no=record.subject_key.split(":")[-1],
        group=group,
        grade_group=record.grade_group,
        excluded=group == "Excluded",
    )


def _subject_key(record: Spectrum) -> str:
    """리플리케이트를 묶는 subject 식별자."""
    return record.subject_key


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


def _preprocess_record(record: Spectrum, grid: np.ndarray) -> np.ndarray:
    return preprocess_arrays(*_record_arrays(record), grid)[1]


def _preprocess_aligned_record(
    record: Spectrum, grid: np.ndarray
) -> tuple[np.ndarray, float]:
    x, y = _record_arrays(record)
    aligned_x, aligned_y, shift = calibrate_spectrum(x, y, window=20.0)
    return preprocess_arrays(aligned_x, aligned_y, grid)[1], shift


# ------------------------------------------------------------------ #
# Subject 빌더 (원본 시그니처 유지: load_clinical_samples() -> build_*_subjects(samples, grid))
# ------------------------------------------------------------------ #
def _grouped_records(
    client: AecdApiClient,
    *,
    site_code: str | None,
    cohort_group: str | None,
    cancer_type: str | None,
) -> dict[str, list[Spectrum]]:
    grouped: dict[str, list[Spectrum]] = defaultdict(list)
    for record in client.iter_spectra(
        site_code=site_code, cohort_group=cohort_group, cancer_type=cancer_type
    ):
        grouped[_subject_key(record)].append(record)
    return grouped


def _fetch_grouped(
    client: AecdApiClient | None,
    site_code: str | None,
    cohort_group: str | None,
    cancer_type: str | None,
) -> dict[str, list[Spectrum]]:
    owns_client = client is None
    client = client or AecdApiClient()
    try:
        return _grouped_records(
            client, site_code=site_code, cohort_group=cohort_group, cancer_type=cancer_type
        )
    finally:
        if owns_client:
            client.close()


def load_clinical_samples(
    *,
    client: AecdApiClient | None = None,
    site_code: str | None = SITE_CODE,
    cohort_group: str | None = COHORT_GROUP,
    cancer_type: str | None = CANCER_TYPE,
) -> list[ClinicalSample]:
    """API에서 subject 단위 ClinicalSample 목록 (원본 xlsx 로더 대체).

    제외(Drop) subject도 excluded=True로 포함해 돌려준다 — 원본이 셀 배경색으로
    제외를 표시하되 목록에는 남겨두던 동작과 같다. 빌더가 걸러낸다.
    """
    grouped = _fetch_grouped(client, site_code, cohort_group, cancer_type)
    return [_record_sample(records[0]) for _, records in sorted(grouped.items())]


def build_boramae_subjects(
    samples: list[ClinicalSample],
    grid: np.ndarray,
    *,
    client: AecdApiClient | None = None,
    site_code: str | None = SITE_CODE,
    cohort_group: str | None = COHORT_GROUP,
    cancer_type: str | None = CANCER_TYPE,
) -> list[SubjectSpectrum]:
    """subject 단위 평균 스펙트럼. `samples`에 있는 subject만, 제외된 것은 건너뛴다."""
    grouped = _fetch_grouped(client, site_code, cohort_group, cancer_type)
    subjects: list[SubjectSpectrum] = []
    for sample in samples:
        if sample.excluded or sample.group == "Excluded":
            continue
        records = grouped.get(sample.label)
        if not records:
            msg = f"No spectra returned by the API for {sample.label}"
            raise RuntimeError(msg)
        mat = np.vstack([_preprocess_record(r, grid) for r in records])
        subjects.append(
            SubjectSpectrum(
                sample=sample,
                mean_spectrum=mat.mean(axis=0),
                replicate_spectra=mat,
                example_raw=_record_arrays(records[0]),
            )
        )
    return subjects


def build_aligned_boramae_subjects(
    samples: list[ClinicalSample],
    grid: np.ndarray,
    *,
    client: AecdApiClient | None = None,
    site_code: str | None = SITE_CODE,
    cohort_group: str | None = COHORT_GROUP,
    cancer_type: str | None = CANCER_TYPE,
) -> tuple[list[SubjectSpectrum], np.ndarray]:
    """calibrate_spectrum 적용 버전. 반환 shifts는 (subject, replicate) 배열."""
    grouped = _fetch_grouped(client, site_code, cohort_group, cancer_type)
    subjects: list[SubjectSpectrum] = []
    shifts: list[np.ndarray] = []
    for sample in samples:
        if sample.excluded or sample.group == "Excluded":
            continue
        records = grouped.get(sample.label)
        if not records:
            msg = f"No spectra returned by the API for {sample.label}"
            raise RuntimeError(msg)
        aligned = [_preprocess_aligned_record(r, grid) for r in records]
        matrix = np.vstack([item[0] for item in aligned])
        shifts.append(np.asarray([item[1] for item in aligned], dtype=float))
        subjects.append(
            SubjectSpectrum(
                sample=sample,
                mean_spectrum=matrix.mean(axis=0),
                replicate_spectra=matrix,
                example_raw=_record_arrays(records[0]),
            )
        )
    return subjects, np.vstack(shifts) if shifts else np.empty((0, 0))


def load_raw_replicates(
    samples: list[ClinicalSample],
    *,
    client: AecdApiClient | None = None,
    site_code: str | None = SITE_CODE,
    cohort_group: str | None = COHORT_GROUP,
    cancer_type: str | None = CANCER_TYPE,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """subject label -> 전처리하지 않은 (wavenumber, intensity) 목록 (mapping 점 순서).

    resolution/noise 재분석처럼 원시 replicate가 필요한 스크립트용. 제외된 subject는
    건너뛴다.
    """
    grouped = _fetch_grouped(client, site_code, cohort_group, cancer_type)
    raw: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for sample in samples:
        if sample.excluded or sample.group == "Excluded":
            continue
        records = grouped.get(sample.label)
        if not records:
            msg = f"No spectra returned by the API for {sample.label}"
            raise RuntimeError(msg)
        raw[sample.label] = [_record_arrays(r) for r in records]
    return raw


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------ #
# 레거시 PRO 비교 코호트 (fig02) — 파일 기반, b6fcf97 이전 구현 그대로
# ------------------------------------------------------------------ #
def preprocess_aligned_file(path: Path, grid: np.ndarray) -> tuple[np.ndarray, float]:
    x, y = read_spectrum(path)
    aligned_x, aligned_y, shift = calibrate_spectrum(x, y, window=20.0)
    return preprocess_arrays(aligned_x, aligned_y, grid)[1], shift


def clean_pro_ids() -> set[str]:
    with CLEAN_MANIFEST.open(encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        return {str(int(float(row["sample_id"]))) for row in rows if row["source_group"] == "PRO"}


def build_clean_pro_alignment(grid: np.ndarray) -> CleanProSpectra:
    selected = clean_pro_ids()
    grouped: dict[str, list[Path]] = {}
    for path in sorted(LEGACY_PRO_ROOT.glob("PRO *.CSV")):
        match = re.match(r"^PRO\s+([0-9]+)_([0-9]+|ave)\.CSV$", path.name)
        if match is not None and match.group(1) in selected and match.group(2) != "ave":
            grouped.setdefault(match.group(1), []).append(path)
    ordered = sorted(grouped, key=int)
    spectra = [
        np.vstack([preprocess_file(path, grid)[1] for path in grouped[item]]).mean(axis=0)
        for item in ordered
    ]
    aligned_items = [
        [preprocess_aligned_file(path, grid) for path in grouped[item]] for item in ordered
    ]
    aligned = [np.vstack([result[0] for result in items]).mean(axis=0) for items in aligned_items]
    shifts = [result[1] for items in aligned_items for result in items]
    return CleanProSpectra(np.vstack(spectra), np.vstack(aligned), np.asarray(shifts, dtype=float))


def build_clean_pro_subjects(grid: np.ndarray) -> np.ndarray:
    return build_clean_pro_alignment(grid).spectra


def group_matrix(subjects: list[SubjectSpectrum], group: str) -> np.ndarray:
    return np.vstack([sub.mean_spectrum for sub in subjects if sub.sample.group == group])
