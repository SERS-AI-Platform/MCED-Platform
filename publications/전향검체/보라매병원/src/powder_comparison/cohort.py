from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

FILE_PATTERN: Final = re.compile(
    r"^(?P<prefix>BPRO|BNOR)\s+(?P<sample_id>\d+)_(?P<replicate>\d+|ave)\.CSV$",
    re.IGNORECASE,
)
EXPECTED_REPLICATES: Final = frozenset(range(1, 6))


class CohortError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SubjectFiles:
    sample_id: int
    prefix: str
    replicates: tuple[Path, ...]
    order_index: int


@dataclass(frozen=True, slots=True)
class PairedSubject:
    sample_id: int
    clinical_group: str
    legacy: SubjectFiles
    powder: SubjectFiles


def collect_subject_files(root: Path) -> Mapping[int, SubjectFiles]:
    grouped: dict[tuple[str, int], dict[int, Path]] = {}
    for path in sorted(root.rglob("*.CSV")):
        match = FILE_PATTERN.match(path.name)
        if match is None or match["replicate"].lower() == "ave":
            continue
        prefix = match["prefix"].upper()
        sample_id = int(match["sample_id"])
        replicate = int(match["replicate"])
        key = (prefix, sample_id)
        bucket = grouped.setdefault(key, {})
        if replicate in bucket:
            raise CohortError(f"Duplicate replicate for {prefix} {sample_id}_{replicate}")
        bucket[replicate] = path

    ordered_keys = sorted(grouped, key=lambda item: (item[0] != "BNOR", item[1]))
    subjects: dict[int, SubjectFiles] = {}
    for order_index, (prefix, sample_id) in enumerate(ordered_keys, start=1):
        replicates = grouped[(prefix, sample_id)]
        if set(replicates) != EXPECTED_REPLICATES:
            raise CohortError(
                f"Incomplete replicate set for {prefix} {sample_id}: {sorted(replicates)}"
            )
        if sample_id in subjects:
            raise CohortError(f"Sample ID appears under multiple prefixes: {sample_id}")
        subjects[sample_id] = SubjectFiles(
            sample_id=sample_id,
            prefix=prefix,
            replicates=tuple(replicates[index] for index in sorted(replicates)),
            order_index=order_index,
        )
    return subjects


def pair_subjects(
    clinical_labels: Mapping[int, str],
    legacy: Mapping[int, SubjectFiles],
    powder: Mapping[int, SubjectFiles],
) -> tuple[PairedSubject, ...]:
    paired_ids = sorted(set(legacy) & set(powder) & set(clinical_labels))
    return tuple(
        PairedSubject(
            sample_id=sample_id,
            clinical_group=clinical_labels[sample_id],
            legacy=legacy[sample_id],
            powder=powder[sample_id],
        )
        for sample_id in paired_ids
    )
