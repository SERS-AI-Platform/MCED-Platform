"""Load Thermo mapping runs 2026-09-11 ~ 2026-09-15 into aecd_platform measurement.*.

Source (copied by the lab, 2026-09-15):
    data/Thermo 260911~260915/    9/11, 9/14, 9/15 mapping + Reference (Si/PS calibration files)

Rules: same as scripts/db/aecd_mapping_batch_20260819_0910_load.py (confirmed with the user on
2026-09-11; the user asked on 2026-09-15 to load this batch "by the previous rules"):
    * run = measurement date x strip lot. Each date here used one lot (9/11 SK20260806A01 units 3-8,
      9/14 SK20260806A01 units 9-10 + SK20260806B01 units 1-4, 9/15 SK20260806C01 units 1-6), so
      9/14 becomes two runs. Run metadata comes from each date's 측정기록.txt.
    * file-name label wins over folder name ("LUN34" folders hold "LUN 34_*" files).
    * CPAN n -> PAN_n, KAPN -> KPAN, H.D(.) -> "H. D.".
    * KPAN (KIMS material-institute specimens, pancreatic only) are registered minimally under site KIMS.
    * "Simulated urine" per strip is loaded as control rows; point_no runs on across strips inside a run
      (strip is kept in source_uri). No MB controls in this batch.
    * Only Si calibrations are loaded (max of the Si _ave spectrum within +-12 cm-1). PS rows are left
      for the owner of the PS full-alignment method, as in the previous batch.
    * _ave files are not loaded.
    * SK20260806B01 / SK20260806C01 are new strip lots with unrecorded manufacture/expiry dates (NULL),
      like SK20260806A01 in the previous batch.

Usage (from repo root, after `source scripts/db/pghost.sh`):
    python scripts/db/aecd_mapping_batch_20260911_0915_load.py --mode plan
    python scripts/db/aecd_mapping_batch_20260911_0915_load.py --mode dry-run   # insert + verify + rollback
    python scripts/db/aecd_mapping_batch_20260911_0915_load.py --mode commit
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
BATCH_ROOT = DATA / "Thermo 260911~260915"
SOURCE_ROOTS = (BATCH_ROOT,)
REFERENCE_DIRS = (BATCH_ROOT / "Reference",)
DATE_RANGE = (date(2026, 9, 11), date(2026, 9, 15))

INSTRUMENT_NAME = "Thermofisher DXR3xi"
REAGENT_LOT = "BCCP0922"
STRIP = dict(strip_name="SOL-KIT", manufacturer="SoluM Healthcare", catalog_number="SHSK01")
STRIP_LOT_DATES = {
    "SK20260806A01": (None, None),
    "SK20260806B01": (None, None),
    "SK20260806C01": (None, None),
}

KIMS_SITE = ("KIMS", "Korea Institute of Materials Science")

SI_REFERENCE = 520.7
SI_WINDOW = 12.0
SI_TOLERANCE = 2.0
SI_NOTES = "Si 평균 스펙트럼 표준 피크 평가 (±12 cm-1 창 최댓값; 8/10~8/14 기존 DB 값 재현 확인)"

# (date, strip folder, sample folder) -> solum_label, where the file name is wrong. None known.
LABEL_OVERRIDES: dict[tuple[str, str, str], str] = {}
GROUP_ALIASES = {"CPAN": "PAN", "KAPN": "KPAN", "H.D.": "H. D.", "H.D": "H. D."}
CONTROL_PREFIXES = (
    (re.compile(r"^40 uM MB \d+$"), "MB_40uM"),
    (re.compile(r"^MB \d+$"), "MB_conc_unrecorded"),
    (re.compile(r"^Simulated urine$", re.I), "simulated_urine"),
)
FILE_RE = re.compile(r"^(?P<prefix>.+?)_(?P<ave>ave)?(?P<num>\d+)?$", re.I)


@dataclass(frozen=True)
class SpectrumFile:
    path: Path
    run_key: tuple[str, str]  # (yyyymmdd, strip lot)
    strip_unit: int
    role: str  # clinical | control
    label: str | None
    control_type: str | None
    file_point: int


@dataclass
class RunInfo:
    day: str
    lot: str
    source_root: Path
    operator: str
    temperature: float
    humidity: float
    units: list[int]


def parse_log(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def label_from_prefix(prefix: str) -> tuple[str, str | None, str | None]:
    prefix = prefix.strip()
    for pattern, ctype in CONTROL_PREFIXES:
        if pattern.match(prefix):
            return "control", None, ctype
    m = re.fullmatch(r"(.*?)[\s_]*(\d+)", prefix)
    if m is None:
        raise ValueError(f"cannot parse sample prefix: {prefix!r}")
    group = GROUP_ALIASES.get(m.group(1).strip(), m.group(1).strip())
    return "clinical", f"{group}_{int(m.group(2))}", None


def discover() -> tuple[list[SpectrumFile], dict[tuple[str, str], RunInfo], int]:
    files: list[SpectrumFile] = []
    runs: dict[tuple[str, str], RunInfo] = {}
    averages = 0
    for root in SOURCE_ROOTS:
        for mapping_dir in sorted(root.glob("*_mapping")):
            day = mapping_dir.name[:8]
            log = parse_log(mapping_dir / "측정기록.txt")
            if log["측정일자"] != day:
                raise ValueError(f"log date mismatch in {mapping_dir}")
            logged_strips = {s.strip() for s in log["Strip Lot"].split(",")}
            temp, hum = (float(v) for v in log["온/습도"].split(","))
            for dirpath, _, names in os.walk(mapping_dir):
                rel = Path(dirpath).relative_to(mapping_dir).parts
                for name in sorted(names):
                    if not name.upper().endswith(".CSV"):
                        continue
                    if not rel or not rel[0].startswith("SK"):
                        raise ValueError(f"file outside a strip folder: {dirpath}/{name}")
                    strip = rel[0]
                    if strip not in logged_strips:
                        raise ValueError(f"strip {strip} not listed in {mapping_dir}/측정기록.txt")
                    sample_folder = rel[-1]
                    m = FILE_RE.fullmatch(name[:-4])
                    if m is None:
                        raise ValueError(f"unrecognized file name: {dirpath}/{name}")
                    if m.group("ave") and not m.group("num"):
                        averages += 1
                        continue
                    if not m.group("num"):
                        raise ValueError(f"no point number: {dirpath}/{name}")
                    lot, unit = strip.rsplit("-", 1)
                    override = LABEL_OVERRIDES.get((day, strip, sample_folder))
                    if override:
                        role, label, ctype = "clinical", override, None
                    else:
                        role, label, ctype = label_from_prefix(m.group("prefix"))
                    key = (day, lot)
                    info = runs.get(key)
                    if info is None:
                        info = runs[key] = RunInfo(day, lot, root, log["측정자"], temp, hum, [])
                    if int(unit) not in info.units:
                        info.units.append(int(unit))
                    files.append(SpectrumFile(Path(dirpath) / name, key, int(unit), role, label,
                                              ctype, int(m.group("num"))))
    for info in runs.values():
        info.units.sort()
    return files, runs, averages


def assign_points(files: list[SpectrumFile]) -> list[tuple[SpectrumFile, int]]:
    """Clinical keeps the file point number; controls are renumbered across strips."""
    out: list[tuple[SpectrumFile, int]] = []
    counters: Counter[tuple] = Counter()
    ordered = sorted(files, key=lambda f: (f.run_key, f.role, f.control_type or "", f.label or "",
                                           f.strip_unit, f.file_point))
    for f in ordered:
        if f.role == "clinical":
            out.append((f, f.file_point))
        else:
            k = (f.run_key, f.control_type)
            counters[k] += 1
            out.append((f, counters[k]))
    keys = Counter((f.run_key, f.role, f.label, f.control_type, p) for f, p in out)
    dups = [k for k, n in keys.items() if n > 1]
    if dups:
        raise ValueError(f"duplicate measurement keys: {dups[:5]}")
    by_day = Counter((f.run_key[0], f.label, p) for f, p in out if f.role == "clinical")
    if any(n > 1 for n in by_day.values()):
        raise ValueError("clinical (day, label, point) collides across runs of one day")
    series = defaultdict(list)
    for f, p in out:
        series[(f.run_key, f.role, f.label, f.control_type)].append(p)
    gaps = [k for k, ps in series.items() if sorted(ps) != list(range(1, len(ps) + 1))]
    if gaps:
        raise ValueError(f"non-contiguous point numbers: {gaps[:5]}")
    return out


def si_calibration(day: str) -> dict:
    name = f"{day}_{day}Cali_Si_ave.CSV"
    path = next((d / name for d in REFERENCE_DIRS if (d / name).exists()), None)
    if path is None:
        raise FileNotFoundError(name)
    a = np.loadtxt(path, delimiter=",")
    m = (a[:, 0] >= SI_REFERENCE - SI_WINDOW) & (a[:, 0] <= SI_REFERENCE + SI_WINDOW)
    observed = round(float(a[m][np.argmax(a[m][:, 1]), 0]), 4)
    error = round(observed - SI_REFERENCE, 4)
    return dict(day=day, observed=observed, error=error, filename=name,
                result="pass" if abs(error) <= SI_TOLERANCE else "fail")


def read_csv(path: Path) -> tuple[list[float], list[float]]:
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all() or not np.all(np.diff(a[:, 0]) > 0):
        raise ValueError(f"bad spectrum: {path}")
    return a[:, 0].tolist(), a[:, 1].tolist()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_map(cur, sql, params=None) -> dict:
    cur.execute(sql, params)
    return {r[0]: r[1] for r in cur.fetchall()}


def load(conn, files, runs, points, cals) -> dict:
    cur = conn.cursor()
    cur.execute("SELECT instrument_id FROM measurement.instruments WHERE instrument_name=%s",
                (INSTRUMENT_NAME,))
    instrument_id = cur.fetchone()[0]
    cur.execute("SELECT reagent_lot_id FROM measurement.reagent_lots WHERE lot_number=%s", (REAGENT_LOT,))
    reagent_lot_id = cur.fetchone()[0]

    # guard: nothing from this source root may be loaded already
    cur.execute("SELECT count(*) FROM measurement.raw_spectra WHERE source_uri LIKE ANY(%s)",
                ([r.resolve().as_uri() + "/%" for r in SOURCE_ROOTS],))
    if cur.fetchone()[0]:
        raise RuntimeError("raw_spectra from this source root already exist")
    cur.execute("SELECT count(*) FROM measurement.runs WHERE measurement_date BETWEEN %s AND %s", DATE_RANGE)
    if cur.fetchone()[0]:
        raise RuntimeError(f"runs for {DATE_RANGE[0]}..{DATE_RANGE[1]} already exist")

    # 1) KPAN registration (site KIMS, subject/sample/diagnosis only) for labels not yet in master
    existing = fetch_map(cur, "SELECT solum_label, sample_id FROM master.samples")
    kpan = sorted({f.label for f, _ in points if f.label and f.label.startswith("KPAN_")} - existing.keys(),
                  key=lambda s: int(s.split("_")[1]))
    cur.execute("INSERT INTO master.sites (site_code, site_name) VALUES (%s,%s) "
                "ON CONFLICT (site_code) DO NOTHING", KIMS_SITE)
    cur.execute("SELECT site_id FROM master.sites WHERE site_code=%s", (KIMS_SITE[0],))
    kims_id = cur.fetchone()[0]
    for label in kpan:
        cur.execute("INSERT INTO master.subjects (site_id, patient_code) VALUES (%s,%s) RETURNING subject_id",
                    (kims_id, label))
        sid = cur.fetchone()[0]
        cur.execute("INSERT INTO master.samples (subject_id, solum_label, sample_type) VALUES (%s,%s,'urine')",
                    (sid, label))
        cur.execute("INSERT INTO clinical.diagnoses (subject_id, cohort_group, cancer_type) "
                    "VALUES (%s,'pancreatic','pancreatic')", (sid,))

    samples = fetch_map(cur, "SELECT solum_label, sample_id FROM master.samples")
    missing = sorted({f.label for f, _ in points if f.role == "clinical"} - samples.keys())
    if missing:
        raise RuntimeError(f"labels not in master.samples: {missing[:10]}")

    # 2) strip lots + units
    lot_ids = {}
    for lot in sorted({k[1] for k in runs}):
        mfg, exp = STRIP_LOT_DATES[lot]
        cur.execute(
            "INSERT INTO measurement.strip_lots (strip_name, manufacturer, catalog_number, lot_number,"
            " manufactured_date, expiration_date) VALUES (%(strip_name)s,%(manufacturer)s,"
            "%(catalog_number)s,%(lot)s,%(mfg)s,%(exp)s) "
            "ON CONFLICT (strip_name, manufacturer, catalog_number, lot_number) DO NOTHING",
            {**STRIP, "lot": lot, "mfg": mfg, "exp": exp})
        cur.execute("SELECT strip_lot_id FROM measurement.strip_lots WHERE lot_number=%s", (lot,))
        lot_ids[lot] = cur.fetchone()[0]
    for (day, lot), info in runs.items():
        for u in info.units:
            cur.execute("INSERT INTO measurement.strip_units (strip_lot_id, unit_number) VALUES (%s,%s) "
                        "ON CONFLICT DO NOTHING", (lot_ids[lot], u))

    # 3) Si calibrations
    cal_ids = {}
    for c in cals.values():
        operator = next(i.operator for i in runs.values() if i.day == c["day"])
        cur.execute(
            "INSERT INTO measurement.calibrations (instrument_id, calibration_type, standard_material,"
            " tolerance_cm1, result, operator_name, report_filename, notes, calibration_date,"
            " reference_peaks_cm1, observed_peaks_cm1, peak_errors_cm1) VALUES"
            " (%s,'raman_shift','Silicon (Si)',%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING calibration_id",
            (instrument_id, SI_TOLERANCE, c["result"], operator, c["filename"], SI_NOTES,
             date(int(c["day"][:4]), int(c["day"][4:6]), int(c["day"][6:])),
             [SI_REFERENCE], [c["observed"]], [c["error"]]))
        cal_ids[c["day"]] = cur.fetchone()[0]

    # 4) runs + run_calibrations
    run_ids = {}
    for key in sorted(runs):
        info = runs[key]
        notes = (f"source_mapping={info.day}_mapping; strip_lot={info.lot}; "
                 f"strip_units={','.join(map(str, info.units))}; "
                 f"source_root={info.source_root.relative_to(REPO)}")
        cur.execute(
            "INSERT INTO measurement.runs (instrument_id, reagent_lot_id, strip_lot_id, measurement_date,"
            " operator_name, temperature_c, humidity_percent, laser_wavelength_nm, laser_power_mw,"
            " exposure_time_s, objective_lens, notes) VALUES (%s,%s,%s,%s,%s,%s,%s,785,1,0.05,'20x',%s)"
            " RETURNING measurement_run_id",
            (instrument_id, reagent_lot_id, lot_ids[info.lot],
             date(int(info.day[:4]), int(info.day[4:6]), int(info.day[6:])),
             info.operator, info.temperature, info.humidity, notes))
        run_ids[key] = cur.fetchone()[0]
        cur.execute("INSERT INTO measurement.run_calibrations VALUES (%s,%s,%s)",
                    (run_ids[key], cal_ids[info.day], instrument_id))

    # 5) measurements + raw_spectra
    batch = 500
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        rows = [(run_ids[f.run_key], samples.get(f.label) if f.role == "clinical" else None, p,
                 f.role, f.control_type) for f, p in chunk]
        mids = execute_values(
            cur, "INSERT INTO measurement.measurements (measurement_run_id, sample_id, point_no,"
            " measurement_role, control_type) VALUES %s RETURNING measurement_id", rows, fetch=True)
        spectra = []
        for (f, _), (mid,) in zip(chunk, mids):
            x, y = read_csv(f.path)
            spectra.append((mid, f.path.resolve().as_uri(), f.path.name, "replicate", x, y,
                            len(x), x[0], x[-1], sha256(f.path)))
        execute_values(
            cur, "INSERT INTO measurement.raw_spectra (measurement_id, source_uri, raw_filename,"
            " source_kind, wavenumber, intensities, n_points, x_min, x_max, source_sha256) VALUES %s",
            spectra)
        print(f"  inserted {min(i + batch, len(points))}/{len(points)}", flush=True)
    kpan_ids = []
    if kpan:
        cur.execute("SELECT sample_id FROM master.samples WHERE solum_label = ANY(%s) ORDER BY 1", (kpan,))
        kpan_ids = [r[0] for r in cur.fetchall()]
    print(f"  undo ids: run_ids={sorted(run_ids.values())} kims_site_id={kims_id}"
          f" kpan_sample_ids={kpan_ids} si_calibration_ids={sorted(cal_ids.values())} strip_lot_ids={lot_ids}")
    return dict(run_ids=run_ids, kpan=len(kpan))


def verify(conn, run_ids: dict, expected: dict) -> None:
    cur = conn.cursor()
    ids = list(run_ids.values())
    cur.execute("""
        SELECT r.measurement_date, sl.lot_number, r.operator_name,
               count(m.*) FILTER (WHERE m.measurement_role='clinical') AS clinical,
               count(m.*) FILTER (WHERE m.measurement_role='control') AS control,
               count(DISTINCT m.sample_id) AS samples, count(rs.*) AS spectra,
               (SELECT count(*) FROM measurement.run_calibrations rc WHERE rc.measurement_run_id=r.measurement_run_id) AS cals
        FROM measurement.runs r JOIN measurement.strip_lots sl USING (strip_lot_id)
        LEFT JOIN measurement.measurements m USING (measurement_run_id)
        LEFT JOIN measurement.raw_spectra rs USING (measurement_id)
        WHERE r.measurement_run_id = ANY(%s)
        GROUP BY r.measurement_run_id, r.measurement_date, sl.lot_number, r.operator_name
        ORDER BY 1, 2""", (ids,))
    total = Counter()
    for row in cur.fetchall():
        print("  ", row)
        total["clinical"] += row[3]
        total["control"] += row[4]
        total["spectra"] += row[6]
        if row[7] != 1:
            raise RuntimeError(f"run without exactly one calibration: {row}")
    print("  totals:", dict(total), "expected:", expected)
    if (total["clinical"], total["control"], total["spectra"]) != (
            expected["clinical"], expected["control"], expected["clinical"] + expected["control"]):
        raise RuntimeError("verification count mismatch")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plan", "dry-run", "commit"], default="plan")
    args = ap.parse_args()

    files, runs, averages = discover()
    points = assign_points(files)
    cals = {day: si_calibration(day) for day in sorted({k[0] for k in runs})}
    role = Counter(f.role for f, _ in points)
    per_run = defaultdict(Counter)
    for f, _ in points:
        per_run[f.run_key][f.role] += 1
    print(f"files: {len(points)} (clinical {role['clinical']}, control {role['control']}),"
          f" averages skipped {averages}")
    for key in sorted(runs):
        i = runs[key]
        c = cals[i.day]
        print(f"  run {key[0]} {key[1]} units={i.units} op={i.operator} T={i.temperature} H={i.humidity}"
              f" clinical={per_run[key]['clinical']} control={per_run[key]['control']}"
              f" Si={c['observed']} ({c['error']:+.4f}, {c['result']})")
    labels = {f.label for f, _ in points if f.role == "clinical"}
    by_group = Counter(lbl.rsplit("_", 1)[0] for lbl in labels)
    print("clinical samples:", len(labels), dict(sorted(by_group.items())))
    per_sample = Counter(f.label for f, _ in points if f.role == "clinical")
    print("points per sample:", dict(Counter(per_sample.values())))
    if args.mode == "plan":
        return

    conn = psycopg2.connect(dbname=os.environ.get("PGDATABASE", "aecd_platform"))
    try:
        result = load(conn, files, runs, points, cals)
        verify(conn, result["run_ids"], dict(role))
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
