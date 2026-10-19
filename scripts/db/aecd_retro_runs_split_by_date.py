"""Date the undated Thermo retro runs and split them into one run per measurement date.

thermo_retro runs (aecd_historical_measurement_load --source thermo_retro) were grouped by x-axis
grid because no acquisition date survived in the repo copy. The Thermo PC upload (raman02,
2026-10-16~19) carries the original folders, whose names start with the measurement date
(e.g. 20250326_Normal_1mW_0.05s_Ave100). Every spectrum of these runs matches exactly one uploaded
file by sha256, so its date is known.

Rule (user 2026-10-19: one run per date, like every other run): run = grid session x date.
The original run keeps its earliest date; each further date becomes a new run that copies the
original's instrument/conditions. measurement_id values do not change; only measurement_run_id
moves. Nothing else references these runs (run_calibrations / experiment.measurement_context: 0 rows).

Usage (repo root, after `source scripts/db/pghost.sh`; ledger = copy of
/mnt/nas/sers_raw/_ledger/uploads.csv):
    python scripts/db/aecd_retro_runs_split_by_date.py --ledger uploads.csv --mode plan|dry-run|commit
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from datetime import date

import psycopg2

LOADER_TAG = "loader=aecd_historical_measurement_load:thermo_retro"
UNKNOWN_RE = re.compile(r"measurement_date=unknown \([^)]*\)")
GROUPS_RE = re.compile(r"; group folders=[^;]*")
DATED_NOTE = "measurement_date from original Thermo PC folder name (raman02 upload, sha256 match; split by date 2026-10-19)"


def ledger_folders(path: str) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.reader(fh):
            if len(r) >= 6 and r[1] == "raman02" and "/" in r[2]:
                out[r[4]].add(r[2].split("/", 1)[0])
    return out


def plan(cur, folders: dict[str, set[str]]):
    cur.execute("""SELECT r.measurement_run_id, r.notes, m.measurement_id, rs.source_sha256
                   FROM measurement.runs r
                   JOIN measurement.measurements m USING (measurement_run_id)
                   JOIN measurement.raw_spectra rs USING (measurement_id)
                   WHERE r.measurement_date IS NULL AND r.notes LIKE %s""", (LOADER_TAG + "%",))
    rows = cur.fetchall()
    if not rows:
        raise SystemExit("no undated thermo_retro runs left")
    # run -> date -> (measurement ids, source folders)
    split: dict[int, dict[date, tuple[list[int], set[str]]]] = defaultdict(dict)
    notes = {}
    for run_id, note, mid, sha in rows:
        fs = folders.get(sha, set())
        if len(fs) != 1:
            raise RuntimeError(f"measurement {mid}: sha256 matches {len(fs)} uploaded folders")
        folder = next(iter(fs))
        if not re.match(r"20\d{6}_", folder):
            raise RuntimeError(f"measurement {mid}: folder has no date prefix: {folder}")
        d = date(int(folder[:4]), int(folder[4:6]), int(folder[6:8]))
        mids, srcs = split[run_id].setdefault(d, ([], set()))
        mids.append(mid)
        srcs.add(folder)
        notes[run_id] = note
    return split, notes, len(rows)


def dated_notes(note: str, day: date, folders: set[str], parent: int | None) -> str:
    note = UNKNOWN_RE.sub(DATED_NOTE, note)
    note = GROUPS_RE.sub("", note)
    note = re.sub(r"run_key=(thermo_retro_grid_[0-9.]+)", lambda m: f"run_key={m.group(1)}_{day:%Y%m%d}", note)
    note += "; original folders=" + " | ".join(sorted(folders))
    if parent is not None:
        note += f"; split_from_run={parent}"
    return note


def apply(cur, split, notes) -> list[int]:
    touched = []
    for run_id in sorted(split):
        days = sorted(split[run_id])
        for i, day in enumerate(days):
            mids, srcs = split[run_id][day]
            if i == 0:
                cur.execute("UPDATE measurement.runs SET measurement_date=%s, notes=%s WHERE measurement_run_id=%s",
                            (day, dated_notes(notes[run_id], day, srcs, None), run_id))
                touched.append(run_id)
                continue
            cur.execute("""INSERT INTO measurement.runs (instrument_id, calibration_id, reagent_lot_id, strip_lot_id,
                               measurement_date, operator_name, temperature_c, humidity_percent, laser_wavelength_nm,
                               laser_power_mw, exposure_time_s, objective_lens, notes)
                           SELECT instrument_id, calibration_id, reagent_lot_id, strip_lot_id, %s, operator_name,
                               temperature_c, humidity_percent, laser_wavelength_nm, laser_power_mw, exposure_time_s,
                               objective_lens, %s
                           FROM measurement.runs WHERE measurement_run_id=%s RETURNING measurement_run_id""",
                        (day, dated_notes(notes[run_id], day, srcs, run_id), run_id))
            new_id = cur.fetchone()[0]
            cur.execute("UPDATE measurement.measurements SET measurement_run_id=%s WHERE measurement_id = ANY(%s)",
                        (new_id, mids))
            if cur.rowcount != len(mids):
                raise RuntimeError(f"moved {cur.rowcount} of {len(mids)} measurements into run {new_id}")
            touched.append(new_id)
    return touched


def verify(cur, touched: list[int], split, total: int) -> None:
    cur.execute("""SELECT r.measurement_run_id, r.measurement_date, count(m.*)
                   FROM measurement.runs r JOIN measurement.measurements m USING (measurement_run_id)
                   WHERE r.measurement_run_id = ANY(%s) GROUP BY 1, 2 ORDER BY 2, 1""", (touched,))
    got = cur.fetchall()
    for row in got:
        print("  ", row[0], row[1], row[2])
    expected = sorted((d, len(v[0])) for days in split.values() for d, v in days.items())
    if sorted((d, n) for _, d, n in got) != expected or sum(n for *_, n in got) != total:
        raise RuntimeError("verification mismatch")
    cur.execute("SELECT count(*) FROM measurement.runs WHERE measurement_date IS NULL AND notes LIKE %s",
                (LOADER_TAG + "%",))
    if cur.fetchone()[0]:
        raise RuntimeError("undated thermo_retro runs remain")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--mode", choices=["plan", "dry-run", "commit"], default="plan")
    args = ap.parse_args()
    conn = psycopg2.connect(dbname=os.environ.get("PGDATABASE", "aecd_platform"))
    try:
        cur = conn.cursor()
        split, notes, total = plan(cur, ledger_folders(args.ledger))
        print(f"undated thermo_retro runs={len(split)} spectra={total} -> runs after split="
              f"{sum(len(v) for v in split.values())}")
        for run_id in sorted(split):
            for day in sorted(split[run_id]):
                mids, srcs = split[run_id][day]
                print(f"  run {run_id} {day} n={len(mids):5d}  {' | '.join(sorted(srcs))}")
        if args.mode == "plan":
            return
        touched = apply(cur, split, notes)
        verify(cur, touched, split, total)
        print(f"  undo: runs {sorted(split)} back to NULL date; new runs {sorted(set(touched) - set(split))}")
        if args.mode == "commit":
            conn.commit()
            print("COMMITTED")
        else:
            conn.rollback()
            print("dry-run OK — rolled back")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
