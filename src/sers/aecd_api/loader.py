from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sers.aecd_api.models import Spectrum, SpectrumFilters
from sers.aecd_api.repository import AecdRepository

SpectrumKey = tuple[str, int, int]
SpectrumDict = dict[SpectrumKey, tuple[np.ndarray, np.ndarray]]

DEFAULT_PAGE_SIZE = 500


@dataclass(frozen=True, slots=True)
class AecdLoadResult:
    """Bridges aecd_platform rows into the shape preprocess_spectra/apply_stage1_qc expect.

    `measurement_ids` is the exact set of aecd_platform measurement_id values used —
    record it in preprocessing_lab_experiments.measurement_ids so a run can be traced
    back to the source records without a live cross-database foreign key.
    """

    spectra: SpectrumDict
    measurement_ids: list[int] = field(default_factory=list)
    skipped_missing_group: list[int] = field(default_factory=list)


def load_spectra_from_aecd(
    repository: AecdRepository,
    filters: SpectrumFilters,
    *,
    page_size: int = DEFAULT_PAGE_SIZE,
) -> AecdLoadResult:
    """Load spectra from aecd_platform via the read-only AecdRepository.

    Paginates through `repository.spectra(filters)` and reshapes each row into the
    `dict[(group, sample_id, replicate), (x, y)]` structure that `preprocess_spectra`
    and `apply_stage1_qc` already expect from file-based loading. Records with no
    `cohort_group` are skipped rather than assigned a fabricated group label — their
    measurement_ids are reported separately so the caller can decide what to do.
    """
    spectra: SpectrumDict = {}
    measurement_ids: list[int] = []
    skipped_missing_group: list[int] = []
    offset = filters.offset
    while True:
        page_filters = filters.model_copy(update={"limit": page_size, "offset": offset})
        page = repository.spectra(page_filters)
        if not page.items:
            break
        for item in page.items:
            if item.cohort_group is None:
                skipped_missing_group.append(item.measurement_id)
                continue
            key = _spectrum_key(item)
            spectra[key] = (
                np.asarray(item.wavenumber, dtype=float),
                np.asarray(item.intensities, dtype=float),
            )
            measurement_ids.append(item.measurement_id)
        offset += len(page.items)
        if offset >= page.total:
            break
    return AecdLoadResult(
        spectra=spectra,
        measurement_ids=measurement_ids,
        skipped_missing_group=skipped_missing_group,
    )


def _spectrum_key(item: Spectrum) -> SpectrumKey:
    assert item.cohort_group is not None
    return (item.cohort_group, _numeric_suffix(item.sample_key), item.replicate_number)


def _numeric_suffix(key: str) -> int:
    return int(key.rsplit(":", 1)[-1])
