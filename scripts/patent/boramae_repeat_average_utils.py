from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def stats(values: object) -> dict[str, float]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    if not len(data):
        return {key: float("nan") for key in ("mean", "median", "p25", "p75", "min", "max")}
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "p25": float(np.percentile(data, 25)),
        "p75": float(np.percentile(data, 75)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
