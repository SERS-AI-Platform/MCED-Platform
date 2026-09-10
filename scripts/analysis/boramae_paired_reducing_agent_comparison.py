"""Paired comparison: Boramae July liquid (old reducing agent) vs August mapping (new).

Same 112 subjects (DB cohort_group labels for both
July file labels are NOT used),
same preprocessing and nested-CV pipeline as the Boramae publication (AS-11), same
fold assignment. Three conditions:

- ``july_liquid``   : 2026-07-09 liquid set, 5 point replicates, local CSV (not in aecd_platform)
- ``aug_mapping``   : 2026-08-10~14 mapping run from aecd_platform, all 121 points per sample
- ``aug_mapping_5`` : the same mapping run, 5 random points per sample (replicate-count control)
- ``aug_mapping_qc``: the same mapping run, only points passing the two-stage QC PL-1 used
                      (Stage-1 intensity/cosmic/saturation on raw; Stage-2 corr-to-subject-mean
                      on preprocessed points) before its 0.79

Outputs go to results/boramae_paired_reducing_agent/ (git-ignored).

Run from the repo root with the AECD database reachable (source scripts/db/pghost.sh):
    PYTHONPATH=src python scripts/analysis/boramae_paired_reducing_agent_comparison.py
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Final

import numpy as np
from scipy.stats import wilcoxon

REPO: Final = Path(__file__).resolve().parents[2]
PUB_SRC: Final = REPO / "publications" / "전향검체" / "보라매병원" / "src"
sys.path.insert(0, str(PUB_SRC))

from boramae_data import MODEL_GRID, parse_group, preprocess_arrays  # noqa: E402
from powder_comparison.delong import correlated_auc_test  # noqa: E402
from prostate_comparison_model import OofResult, build_task, nested_oof  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from sers.aecd_api.models import SpectrumFilters  # noqa: E402
from sers.aecd_api.repository import DatabaseSettings, PostgresAecdRepository  # noqa: E402
from sers.io import read_spectrum  # noqa: E402
from sers.qc.qc import apply_per_spectrum_corr_qc, apply_stage1_qc  # noqa: E402

JULY_DIR: Final = REPO / "data" / "raw_data" / "20260709_BPRO,BNOR_1mW_0.05s_Ave100"
OUT: Final = REPO / "results" / "boramae_paired_reducing_agent"
SITE: Final = "BORAMAE"
LABELS: Final = ("Control", "Biopsy-negative", "Prostate cancer")
FILE_RE: Final = re.compile(r"^(BPRO|BNOR)\s+([0-9]+)_([0-9]+)\.CSV$", re.IGNORECASE)
SUBSET_POINTS: Final = 5
SEED: Final = 20260909
# nested_oof() fixes its fold seeds, so fold membership is set by subject order. With
# n=112 that alone moved screening AUC 0.79 -> 0.69 between two orderings, so every
# condition is evaluated over REPEATS random subject permutations and reported as a
# distribution. Repeat 0 is the identity order and provides the subject-level outputs.
REPEATS: Final = 10


def db_subjects() -> tuple[dict[str, dict], dict[int, list[tuple[np.ndarray, np.ndarray]]]]:
    """subject number -> {subject_id, cohort_group, label}; subject_id -> raw mapping spectra."""
    repository = PostgresAecdRepository(DatabaseSettings.from_environment())
    spectra: dict[int, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
    cohort: dict[int, str | None] = {}
    offset = 0
    while True:
        page = repository.spectra(SpectrumFilters(site_code=SITE, limit=500, offset=offset))
        for item in page.items:
            subject_id = int(item.subject_key.split(":")[1])
            cohort[subject_id] = item.cohort_group
            spectra[subject_id].append(
                (np.asarray(item.wavenumber, dtype=float), np.asarray(item.intensities, dtype=float))
            )
        offset += len(page.items)
        if not page.items or offset >= page.total:
            break
    # subject number (the digits in solum_label) links the July files to DB subjects
    import psycopg2

    settings = DatabaseSettings.from_environment()
    with psycopg2.connect(
        host=settings.host, port=settings.port, dbname=settings.database,
        user=settings.user, password=settings.password,
    ) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT subject.subject_id, sample.solum_label FROM master.samples AS sample "
            "JOIN master.subjects AS subject USING (subject_id) JOIN master.sites AS site USING (site_id) "
            "WHERE site.site_code = %s",
            (SITE,),
        )
        rows = cursor.fetchall()
    by_number: dict[str, dict] = {}
    for subject_id, solum_label in rows:
        number = str(int(re.search(r"([0-9]+)$", solum_label).group(1)))
        if subject_id in spectra:
            by_number[number] = {
                "subject_id": subject_id,
                "cohort_group": cohort[subject_id],
                "label": parse_group(cohort[subject_id]),
            }
    return by_number, spectra


def july_files() -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(JULY_DIR.rglob("*.CSV")):
        match = FILE_RE.match(path.name)
        if match is not None:
            files[str(int(match.group(2)))].append(path)
    return files


def july_session(paths: list[Path]) -> str:
    """Acquisition session from file mtime: the July set was measured in blocks
    (BNOR 07-09 11h / 18h, BPRO 07-09 18h -> 07-10 08h); label = 'MM-DD HHh'."""
    import datetime as _dt
    stamps = sorted(_dt.datetime.fromtimestamp(p.stat().st_mtime) for p in paths)
    return stamps[0].strftime("%m-%d %Hh")


def subject_mean(raw: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray) -> np.ndarray:
    return np.vstack([preprocess_arrays(x, y, grid)[1] for x, y in raw]).mean(axis=0)


def run_task(x: np.ndarray, labels: np.ndarray, task: str) -> tuple[OofResult, np.ndarray]:
    task_x, task_y, indices = build_task(x, labels, task)  # type: ignore[arg-type]
    return nested_oof(task_x, task_y), indices


CLASS_NAMES: Final = ("control", "biopsy_neg", "cancer")


def summarize(result: OofResult, task: str) -> dict[str, float]:
    keys = (
        ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "macro_f1")
        if task == "screening_binary"
        else ("macro_ovr_roc_auc", "balanced_accuracy", "macro_f1")
    )
    out: dict[str, float] = {}
    for key in keys:
        out[key] = result.metrics[key]
        low, high = result.intervals[key]
        out[f"{key}_ci_low"], out[f"{key}_ci_high"] = low, high
    if task == "three_group":
        # one-vs-rest AUC and recall per class (rows of the confusion matrix = true class)
        for k, name in enumerate(CLASS_NAMES):
            out[f"ovr_auc_{name}"] = float(roc_auc_score((result.y_true == k).astype(int), result.probabilities[:, k]))
            row = result.confusion[k]
            out[f"recall_{name}"] = float(row[k] / row.sum()) if row.sum() else float("nan")
            out[f"predicted_as_{name}"] = int(result.confusion[:, k].sum())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    grid = np.load(MODEL_GRID)
    rng = np.random.default_rng(SEED)

    by_number, mapping_raw = db_subjects()
    july = july_files()
    numbers = sorted((n for n in by_number if n in july and by_number[n]["label"] != "Excluded"), key=int)
    dropped = sorted((n for n in by_number if by_number[n]["label"] == "Excluded"), key=int)
    july_only = sorted((n for n in july if n not in by_number), key=int)
    print(f"paired subjects: {len(numbers)} | dropped: {dropped} | july-only (not in DB): {july_only}")

    labels = np.array([by_number[n]["label"] for n in numbers])
    conditions: dict[str, np.ndarray] = {"july_liquid": [], "aug_mapping": [], "aug_mapping_5": [], "aug_mapping_qc": []}
    replicate_counts: list[tuple[str, int, int, int, int]] = []
    sessions: list[str] = []
    # Stage-1 QC is relative (intensity gate vs the cohort), so it must see the whole set at
    # once - the way PL-1 ran it on all 13,552 mapping spectra - not one subject at a time.
    july_all = {("J", n, i): xy for n in numbers for i, xy in enumerate([read_spectrum(p) for p in sorted(july[n])])}
    mapping_all = {("M", n, i): xy for n in numbers for i, xy in enumerate(mapping_raw[by_number[n]["subject_id"]])}
    july_passed, _ = apply_stage1_qc(july_all)
    mapping_passed, qc_log = apply_stage1_qc(mapping_all)
    # Stage-2: per-point correlation to the mean of the subject's other points, on preprocessed
    # spectra (PL-1: 13,552 -> 11,331 mostly here). Keys are (group, subject, point) as the QC expects.
    processed_all = {key: preprocess_arrays(x, y, grid)[1] for key, (x, y) in mapping_all.items() if key in mapping_passed}
    mapping_passed2, corr_log = apply_per_spectrum_corr_qc(processed_all)
    print(f"stage1 QC pooled: mapping {len(mapping_passed)}/{len(mapping_all)} passed "
          f"({dict(qc_log['reason'].value_counts()) if len(qc_log) else {}}); july {len(july_passed)}/{len(july_all)}")
    print(f"stage2 corr QC: mapping {len(mapping_passed2)}/{len(processed_all)} passed")
    for n in numbers:
        subject_id = by_number[n]["subject_id"]
        liquid = [july_all[("J", n, i)] for i in range(len(july[n]))]
        mapping = mapping_raw[subject_id]
        subset_idx = rng.choice(len(mapping), size=SUBSET_POINTS, replace=False)
        qc_rows = [processed_all[("M", n, i)] for i in range(len(mapping)) if ("M", n, i) in mapping_passed2]
        liquid_passed = [i for i in range(len(liquid)) if ("J", n, i) in july_passed]
        conditions["july_liquid"].append(subject_mean(liquid, grid))
        conditions["aug_mapping"].append(subject_mean(mapping, grid))
        conditions["aug_mapping_5"].append(subject_mean([mapping[i] for i in subset_idx], grid))
        conditions["aug_mapping_qc"].append(np.vstack(qc_rows).mean(axis=0) if len(qc_rows) >= 3 else subject_mean(mapping, grid))
        replicate_counts.append((n, len(liquid), len(mapping), len(qc_rows), len(liquid_passed)))
        sessions.append(july_session(july[n]))
    matrices = {name: np.vstack(rows) for name, rows in conditions.items()}
    qc_kept = np.array([r[3] for r in replicate_counts])
    july_kept = np.array([r[4] for r in replicate_counts])
    print(f"stage1 QC: mapping points kept/subject median {np.median(qc_kept):.0f}/121 (min {qc_kept.min()}, max {qc_kept.max()}); "
          f"july replicates kept median {np.median(july_kept):.0f}/5")
    print("july sessions x label:", {sess: dict(zip(*np.unique(labels[np.array(sessions) == sess], return_counts=True))) for sess in sorted(set(sessions))})

    # --- per-condition models over REPEATS fold assignments (same permutation for every
    #     condition within a repeat, so paired tests compare like with like) ---
    n = len(labels)
    perm_rng = np.random.default_rng(SEED + 1)
    permutations = [np.arange(n)] + [perm_rng.permutation(n) for _ in range(REPEATS - 1)]
    records: list[dict] = []
    oof: dict[tuple[int, str, str], OofResult] = {}
    for repeat, perm in enumerate(permutations):
        inverse = np.argsort(perm)
        for task in ("screening_binary", "three_group"):
            for name, x in matrices.items():
                result, _ = run_task(x[perm], labels[perm], task)
                # store in canonical subject order
                result = OofResult(
                    y_true=result.y_true[inverse], y_pred=result.y_pred[inverse],
                    probabilities=result.probabilities[inverse], folds=result.folds[inverse],
                    metrics=result.metrics, intervals=result.intervals, confusion=result.confusion,
                )
                oof[(repeat, task, name)] = result
                records.append({"repeat": repeat, "task": task, "condition": name, "n": n, **summarize(result, task)})
        print(f"repeat {repeat}: " + " | ".join(
            f"{name} scr {oof[(repeat,'screening_binary',name)].metrics['roc_auc']:.3f} 3g {oof[(repeat,'three_group',name)].metrics['macro_ovr_roc_auc']:.3f}"
            for name in matrices))

    # --- paired tests on the same subjects (screening), per repeat ---
    truth = oof[(0, "screening_binary", "july_liquid")].y_true
    paired: list[dict] = []
    for repeat in range(REPEATS):
        for a, b in (("july_liquid", "aug_mapping"), ("july_liquid", "aug_mapping_5"), ("aug_mapping", "aug_mapping_5"), ("july_liquid", "aug_mapping_qc"), ("aug_mapping", "aug_mapping_qc")):
            pa = oof[(repeat, "screening_binary", a)].probabilities[:, 1]
            pb = oof[(repeat, "screening_binary", b)].probabilities[:, 1]
            delong = correlated_auc_test(truth, pa, pb)
            w = wilcoxon(pb - pa)
            paired.append({
                "repeat": repeat, "comparison": f"{a} -> {b}", "auc_a": delong.legacy_auc, "auc_b": delong.powder_auc,
                "delta_auc": delong.delta, "delong_z": delong.z_score, "delong_p": delong.p_value,
                "prob_shift_median": float(np.median(pb - pa)), "wilcoxon_p": float(w.pvalue),
                "subjects_flipped": int(((pa >= 0.5) != (pb >= 0.5)).sum()),
            })

    # --- across-repeat summary ---
    def dist(values: list[float]) -> dict[str, float]:
        arr = np.asarray(values, dtype=float)
        return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)), "min": float(arr.min()), "max": float(arr.max())}
    repeat_summary: list[dict] = []
    metric_keys = {
        "screening_binary": ("roc_auc", "balanced_accuracy", "sensitivity", "specificity", "macro_f1"),
        "three_group": ("macro_ovr_roc_auc", "balanced_accuracy", "macro_f1",
                        *[f"ovr_auc_{c}" for c in CLASS_NAMES], *[f"recall_{c}" for c in CLASS_NAMES], *[f"predicted_as_{c}" for c in CLASS_NAMES]),
    }
    per_repeat_rows = {(r["repeat"], r["task"], r["condition"]): r for r in records}
    for task, keys in metric_keys.items():
        for key in keys:
            for name in matrices:
                vals = [float(per_repeat_rows[(r, task, name)][key]) for r in range(REPEATS)]
                repeat_summary.append({"task": task, "metric": key, "condition": name, **dist(vals)})
    paired_summary: list[dict] = []
    for comp in ("july_liquid -> aug_mapping", "july_liquid -> aug_mapping_5", "aug_mapping -> aug_mapping_5", "july_liquid -> aug_mapping_qc", "aug_mapping -> aug_mapping_qc"):
        deltas = [p["delta_auc"] for p in paired if p["comparison"] == comp]
        pvals = [p["delong_p"] for p in paired if p["comparison"] == comp]
        paired_summary.append({"comparison": comp, **{f"delta_auc_{k}": v for k, v in dist(deltas).items()},
                               "repeats_delta_negative": int(sum(d < 0 for d in deltas)),
                               "repeats_delong_p_lt_0_05": int(sum(p < 0.05 for p in pvals)), "delong_p_median": float(np.median(pvals))})
    for row in repeat_summary + paired_summary:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    # --- batch check: were July Controls measured in their own session, and does the
    #     3-group model recognise them by session? ---
    sess_arr = np.array(sessions)
    batch_rows: list[dict] = []
    for cond in ("july_liquid", "aug_mapping"):
        for sess in sorted(set(sess_arr)):
            for k, name in enumerate(CLASS_NAMES):
                mask = (sess_arr == sess) & (labels == LABELS[k])
                if mask.sum() == 0:
                    continue
                recalls = [float((oof[(r, "three_group", cond)].y_pred[mask] == k).mean()) for r in range(REPEATS)]
                p_own = [float(oof[(r, "three_group", cond)].probabilities[mask, k].mean()) for r in range(REPEATS)]
                batch_rows.append({"condition": cond, "july_session": sess, "class": name, "n": int(mask.sum()),
                                   "recall_mean": float(np.mean(recalls)), "recall_sd": float(np.std(recalls, ddof=1)) if REPEATS > 1 else 0.0,
                                   "p_own_class_mean": float(np.mean(p_own))})
    for row in batch_rows:
        print("batch", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()})

    # --- spectral agreement between conditions (same subject) ---
    corr = [float(np.corrcoef(matrices["july_liquid"][i], matrices["aug_mapping"][i])[0, 1]) for i in range(len(numbers))]
    mean_by_group = {
        name: {lab: x[labels == lab].mean(axis=0).tolist() for lab in LABELS} for name, x in matrices.items()
    }

    # --- write ---
    with (OUT / "condition_metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(dict.fromkeys(key for record in records for key in record))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    with (OUT / "paired_tests.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired[0]))
        writer.writeheader()
        writer.writerows(paired)
    with (OUT / "repeat_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(repeat_summary[0]))
        writer.writeheader()
        writer.writerows(repeat_summary)
    with (OUT / "paired_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(paired_summary[0]))
        writer.writeheader()
        writer.writerows(paired_summary)
    with (OUT / "july_session_batch_check.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(batch_rows[0]))
        writer.writeheader()
        writer.writerows(batch_rows)
    with (OUT / "subject_oof.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subject_id", "label", "july_session", "n_july", "n_mapping", "n_mapping_qc", "p_cancer_july", "p_cancer_mapping", "p_cancer_mapping5", "spectral_corr_july_vs_mapping"])
        for i, n in enumerate(numbers):
            writer.writerow([
                by_number[n]["subject_id"], labels[i], sessions[i], replicate_counts[i][1], replicate_counts[i][2], replicate_counts[i][3],
                oof[(0, "screening_binary", "july_liquid")].probabilities[i, 1],
                oof[(0, "screening_binary", "aug_mapping")].probabilities[i, 1],
                oof[(0, "screening_binary", "aug_mapping_5")].probabilities[i, 1], corr[i],
            ])
    for task in ("screening_binary", "three_group"):
        for name in matrices:
            np.savetxt(OUT / f"confusion_{task}_{name}.csv", oof[(0, task, name)].confusion, fmt="%d", delimiter=",")
    (OUT / "summary.json").write_text(json.dumps({
        "n_paired": len(numbers), "dropped": dropped, "july_only_not_in_db": july_only,
        "labels": {lab: int((labels == lab).sum()) for lab in LABELS},
        "grid_points": int(len(grid)), "grid_range": [float(grid[0]), float(grid[-1])],
        "spectral_corr_july_vs_mapping": {"median": float(np.median(corr)), "p5": float(np.percentile(corr, 5)), "p95": float(np.percentile(corr, 95))},
        "mean_by_group": mean_by_group, "grid": grid.tolist(),
        "repeats": REPEATS, "metrics": records, "paired": paired,
        "repeat_summary": repeat_summary, "paired_summary": paired_summary, "batch_check": batch_rows,
        "qc_points_kept": {"median": float(np.median(qc_kept)), "min": int(qc_kept.min()), "max": int(qc_kept.max())},
        "july_sessions": {sess: {LABELS[k]: int(((sess_arr == sess) & (labels == LABELS[k])).sum()) for k in range(3)} for sess in sorted(set(sess_arr))},
    }, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
