from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Final

from .matching_persistence import stable_id

_GROUP_CODES: Final = frozenset({"group", "source_group", "group_code"})


@dataclass(frozen=True, slots=True)
class LabelMapping:
    observation_value: str
    label_value: str


@dataclass(frozen=True, slots=True)
class LabelRule:
    task_name: str
    label_definition_version: str
    observation_code: str
    label_source: str
    mappings: tuple[LabelMapping, ...]


@dataclass(frozen=True, slots=True)
class GroupCodeLabelRuleError(ValueError):
    observation_code: str

    def __str__(self) -> str:
        return "source group metadata cannot be used as clinical ground truth"


def _label_for(
    mappings: tuple[LabelMapping, ...],
    raw_value: str,
) -> str | None:
    for mapping in mappings:
        if mapping.observation_value == raw_value:
            return mapping.label_value
    return None


def derive_sample_labels(
    connection: sqlite3.Connection,
    rule: LabelRule,
) -> int:
    if rule.observation_code.casefold() in _GROUP_CODES:
        raise GroupCodeLabelRuleError(rule.observation_code)
    rows = connection.execute(
        """SELECT DISTINCT mc.sample_id, co.id, co.raw_value, co.text_value
           FROM match_candidates AS mc
           JOIN clinical_observations AS co
             ON co.clinical_event_id = mc.clinical_event_id
           WHERE mc.status = 'matched'
             AND (co.canonical_code = ? OR co.code = ?)
           ORDER BY mc.sample_id, co.id""",
        (rule.observation_code, rule.observation_code),
    )
    for row in rows:
        observed_value = row[2] if row[2] is not None else row[3]
        if observed_value is None:
            continue
        label_value = _label_for(rule.mappings, observed_value)
        if label_value is None:
            continue
        label_key = "|".join(
            (
                row[0],
                rule.task_name,
                rule.label_definition_version,
                label_value,
                rule.label_source,
            )
        )
        label_id = stable_id("sample-label", label_key)
        connection.execute(
            """INSERT OR IGNORE INTO sample_labels (
                   id, sample_id, task_name, label_definition_version,
                   label_value, label_source, status
               ) VALUES (?, ?, ?, ?, ?, ?, 'confirmed')""",
            (
                label_id,
                row[0],
                rule.task_name,
                rule.label_definition_version,
                label_value,
                rule.label_source,
            ),
        )
        connection.execute(
            """INSERT OR IGNORE INTO sample_label_evidence (
                   id, sample_label_id, clinical_observation_id
               ) VALUES (?, ?, ?)""",
            (
                stable_id("sample-label-evidence", f"{label_id}:{row[1]}"),
                label_id,
                row[1],
            ),
        )
    return int(
        connection.execute(
            """SELECT COUNT(*) FROM sample_labels
               WHERE task_name = ? AND label_definition_version = ?""",
            (rule.task_name, rule.label_definition_version),
        ).fetchone()[0]
    )
