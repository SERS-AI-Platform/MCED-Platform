"""Load historical measurement sets into aecd_platform measurement.* (2026-09-11 decisions).

Dated sources:
    mbsu       thermo_standard_MB-SU_20260805-20260810/MB&SU (100 uM MB, simulated urine). Date = file-name date,
               corroborated by file mtime. Folder = preparation batch.
    april      03_sers_date_lot_balanced_acquisition/thermo_retest_12groups_20260416-20260519/2026MMDD_Urine test (4월 재측정) CSV. Date = file mtime date — mtimes are
               acquisition times here (hours of spread, ~90 s per spectrum) and match the folder
               dates except files measured on a later day (4/17, 4/27, 5/19), which get their own run.
    april_spa  the .SPA files of thermo_retest_12groups_20260416-20260519 (0. MB, 0. Reference). Date = acquisition time stored in
               the SPA file; rows attach to the existing april_YYYYMMDD run.
    raw_dated  thermo_retro_12groups_undated/BLC_1st_20260319-20260320 (3/19-20), thermo_retro_12groups_undated/BLC_2nd_20260407-20260409 (4/7-9),
               03_sers_date_lot_balanced_acquisition/thermo_boramae_liquid_BNOR-BPRO_20260709-20260710 (7/9-10). Date = file mtime (acquisition spread).
    july       20260715 powder reproducibility (one run per Sigma batch), 20260716 powder,
               20260720 YPAN (operator 엄찬호 from its 라만 분석 txt). Date = folder/log date.
    handheld   equipment_comparison_NOR_5devices_undated/handheld, Metrohm Mira P. Date = CreatedDate in each export.
    metabolite metabolite reference standards; date = original share folder (name + size match).

Undated sources (measurement_date NULL, notes carry measurement_date=unknown; needs
scripts/db/migrations/20260911_runs_measurement_date_nullable.sql):
    thermo_retro  thermo_retro_12groups_undated main groups; run = x-axis grid session.
    medical       ramcheck_retro_12groups_undated (NanoScope Ramcheck A1); run = delivered folder, Background = blank.
    metrohm       data/02_sers_primary_pooled_acquisition/metrohm_retro_NOR-PRO_undated wide CSVs (Mira P); run = delivered folder, one row per column.

Common rules: operator 'unrecorded' unless a log names one; reagent lot, strip lot, and
conditions NULL unless the folder name states them (1mW_0.05s); _ave and "Averaged data" are
not loaded; raw arrays are stored as-is (no shift, no preprocessing). Clinical labels are
normalised (CPAN->PAN, H.D.->"H. D.", KAPN->KPAN); Boramae BPRO/BNOR file prefixes are
pre-pathology numbers, so they are re-attached to the DB sample with the same number
(BPRO/BNOR numbers are disjoint in master.samples). Labels not in master.samples are skipped
and listed.

Usage (repo root, after `source scripts/db/pghost.sh`):
    python scripts/db/aecd_historical_measurement_load.py --source april --mode plan|dry-run|commit
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DATA = REPO / "data"
THERMO = "Thermofisher DXR3xi"
MIRA_P = "Metrohm Mira P"
RAMCHECK = "NanoScope Ramcheck A1"
# instruments registered on first use (user 2026-09-11: HandHeld = Metrohm Mira P,
# "Medical" = NanoScope Ramcheck A1; conditions to be added later)
INSTRUMENTS = {
    MIRA_P: dict(manufacturer="Metrohm", model_name="Mira P", location=None),
    RAMCHECK: dict(manufacturer="NanoScope", model_name="Ramcheck A1", location=None),
}
UNRECORDED = "unrecorded"
GROUP_ALIASES = {"CPAN": "PAN", "KAPN": "KPAN", "H.D.": "H. D.", "H.D": "H. D."}
REP_RE = re.compile(r"^(?P<stem>.+?)_(?P<rep>\d+|ave)?$", re.I)


@dataclass
class Run:
    key: str
    day: date | None
    operator: str = UNRECORDED
    instrument: str = THERMO
    laser_power_mw: float | None = None
    exposure_time_s: float | None = None
    laser_wavelength_nm: float | None = None
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class File:
    path: Path
    run: str
    role: str  # clinical | control | blank
    label: str | None = None
    control_type: str | None = None
    rep: int | None = None
    source_kind: str = "replicate"
    column: str | None = None  # wide CSV (several spectra per file): header of this spectrum


def source_uri(f: File) -> str:
    uri = f.path.resolve().as_uri()
    return uri if f.column is None else f"{uri}#column={f.column.replace(' ', '%20')}"


def read_wide_column(path: Path, column: str) -> tuple[list[float], list[float], str]:
    """One spectrum from a wide CSV (first column = wavenumber). source_sha256 is the hash of
    the extracted "x,y" lines, so identical spectra still share a hash like whole files do."""
    import csv
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.reader(fh))
    idx = rows[0].index(column)
    x = np.array([float(r[0]) for r in rows[1:]])
    y = np.array([float(r[idx]) for r in rows[1:]])
    if not np.isfinite(y).all() or not np.all(np.diff(x) > 0):
        raise ValueError(f"bad spectrum: {path}#{column}")
    text = "".join(f"{r[0]},{r[idx]}\n" for r in rows[1:])
    return x.tolist(), y.tolist(), hashlib.sha256(text.encode()).hexdigest()


def mdate(path: Path) -> date:
    return datetime.fromtimestamp(path.stat().st_mtime).date()


def csvs(root: Path, exts=(".csv",)) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts
                  and "Zone.Identifier" not in p.name and "Averaged data" not in p.parts)


def split_rep(path: Path) -> tuple[str, int | None, bool]:
    m = REP_RE.fullmatch(path.stem)
    if m is None:
        return path.stem, None, False
    rep = m.group("rep")
    if rep and rep.lower() == "ave":
        return m.group("stem"), None, True
    return m.group("stem"), int(rep) if rep else None, False


def clinical_label(stem: str) -> str | None:
    m = re.fullmatch(r"(.*?)[\s_]*(\d+)", stem.strip())
    if m is None:
        return None
    group = m.group(1).strip()
    group = GROUP_ALIASES.get(group, group)
    return f"{group}_{int(m.group(2))}"


# ───────────────────────── source adapters ─────────────────────────

def src_mbsu() -> tuple[dict[str, Run], list[File]]:
    root = DATA / "03_sers_date_lot_balanced_acquisition" / "thermo_standard_MB-SU_20260805-20260810" / "MB&SU"
    runs: dict[str, Run] = {}
    files: list[File] = []
    for p in csvs(root):
        stem, rep, ave = split_rep(p)
        if ave:
            continue
        m = re.fullmatch(r"(?P<what>.+?)_(?P<d>2026\d{4})_(?P<batch>B\d)_?", stem)
        if m is None:
            raise ValueError(f"unrecognised MB&SU file: {p}")
        day = datetime.strptime(m.group("d"), "%Y%m%d").date()
        if mdate(p) != day:
            raise ValueError(f"file-name date and mtime disagree: {p}")
        what = m.group("what").lower()
        ctype = "MB_100uM" if what.startswith("100 um mb") else (
            "simulated_urine" if "urine" in what else None)
        if ctype is None:
            raise ValueError(f"unknown control material: {p}")
        key = f"mbsu_{day:%Y%m%d}"
        run = runs.setdefault(key, Run(key, day, notes=[
            "control-only run", "source_root=data/03_sers_date_lot_balanced_acquisition/thermo_standard_MB-SU_20260805-20260810/MB&SU",
            "material lot label B1 (strip/solution batch, not resolved to a strip lot)"]))
        batch = f"prep_folder={p.parent.name}"
        if batch not in run.notes:
            run.notes.append(batch)
        files.append(File(p, key, "control", control_type=ctype))
    return runs, files


APRIL_CONTROL = {"PS": ("PS", "calibration"), "Si": ("Si", "calibration"),
                 "Si wafer": ("Si", "calibration"), "Si Wafer": ("Si", "calibration")}


def src_april() -> tuple[dict[str, Run], list[File]]:
    root = DATA / "03_sers_date_lot_balanced_acquisition" / "thermo_retest_12groups_20260416-20260519"
    runs: dict[str, Run] = {}
    files: list[File] = []
    for session in sorted(root.glob("2026*_Urine test")):
        for p in csvs(session):
            stem, rep, ave = split_rep(p)
            if ave:
                continue
            day = mdate(p)
            key = f"april_{day:%Y%m%d}"
            runs.setdefault(key, Run(key, day, notes=[
                "4월 재측정 (pre reducing-agent change; reagent identity not recorded)",
                f"source_root=data/03_sers_date_lot_balanced_acquisition/thermo_retest_12groups_20260416-20260519/{session.name}",
                "lot/paper layout: Downloads/검체 측정 Lot 정리.xlsx"]))
            if f"source_root=data/03_sers_date_lot_balanced_acquisition/thermo_retest_12groups_20260416-20260519/{session.name}" not in runs[key].notes:
                runs[key].notes.append(f"source_root=data/03_sers_date_lot_balanced_acquisition/thermo_retest_12groups_20260416-20260519/{session.name}")
            folder = p.parent.name
            if folder == "0. Blank":
                files.append(File(p, key, "blank", control_type="strip_blank"))
            elif folder == "0. Reference":
                name = re.sub(r"\s*\d+$", "", stem)
                ctype, kind = APRIL_CONTROL[name]
                files.append(File(p, key, "control", control_type=ctype, source_kind=kind))
            else:
                files.append(File(p, key, "clinical", label=clinical_label(stem), rep=rep))
    return runs, files


def src_raw_dated() -> tuple[dict[str, Run], list[File]]:
    runs: dict[str, Run] = {}
    files: list[File] = []
    sets = (
        ("rawblc", DATA / "02_sers_primary_pooled_acquisition" / "thermo_retro_12groups_undated" / "BLC_1st_20260319-20260320", None, None),
        ("rawblc0407", DATA / "02_sers_primary_pooled_acquisition" / "thermo_retro_12groups_undated" / "BLC_2nd_20260407-20260409", 1.0, 0.05),
        ("boramae0709", DATA / "03_sers_date_lot_balanced_acquisition" / "thermo_boramae_liquid_BNOR-BPRO_20260709-20260710", 1.0, 0.05),
    )
    for prefix, root, power, exposure in sets:
        for p in csvs(root):
            stem, rep, ave = split_rep(p)
            if ave:
                continue
            day = mdate(p)
            key = f"{prefix}_{day:%Y%m%d}"
            run = runs.setdefault(key, Run(key, day, laser_power_mw=power, exposure_time_s=exposure,
                                           laser_wavelength_nm=785 if power else None, notes=[
                f"source_root={root.relative_to(REPO)}",
                "conditions from folder name" if power else "conditions not recorded"]))
            if "Calibration_control" in p.parent.name:
                name = re.sub(r"\s*\d+$", "", stem)
                if name == "Sensor":
                    files.append(File(p, key, "blank", control_type="bare_sensor"))
                else:
                    ctype, kind = APRIL_CONTROL[name]
                    files.append(File(p, key, "control", control_type=ctype, source_kind=kind))
                continue
            files.append(File(p, key, "clinical", label=clinical_label(stem), rep=rep))
            if prefix == "boramae0709" and "Boramae file prefixes remapped by number" not in run.notes:
                run.notes.append("Boramae file prefixes remapped by number")
    return runs, files


def src_july() -> tuple[dict[str, Run], list[File]]:
    runs: dict[str, Run] = {}
    files: list[File] = []
    root = DATA / "04_machine_repeatability_tests" / "thermo_powder_reproducibility_BPRO_20260715"
    for p in csvs(root):
        stem, rep, ave = split_rep(p)
        if ave:
            continue
        m = re.fullmatch(r"Sigma (\d)_(.+)", stem)
        key = f"powder0715_sigma{m.group(1)}"
        runs.setdefault(key, Run(key, date(2026, 7, 15), notes=[
            "preparation=powder", f"reagent_batch=Sigma {m.group(1)} (powder reproducibility)",
            f"source_root={root.relative_to(REPO)}", "date from folder name; mtime is copy time"]))
        files.append(File(p, key, "clinical", label=clinical_label(m.group(2)), rep=rep))
    root = DATA / "05_not_yet_analyzed" / "thermo_powder_BNOR-BPRO_20260716"
    for p in csvs(root):
        stem, rep, ave = split_rep(p)
        if ave:
            continue
        runs.setdefault("powder0716", Run("powder0716", date(2026, 7, 16), notes=[
            "preparation=powder", f"source_root={root.relative_to(REPO)}",
            "date from folder name; mtime is copy time", "Boramae file prefixes remapped by number"]))
        files.append(File(p, "powder0716", "clinical", label=clinical_label(stem), rep=rep))
    root = (DATA / "05_not_yet_analyzed" / "thermo_YPAN_addition_undated" / "10-3. Y-Pancreatic cancer (추가)_20260721"
            / "20260720_Urine test_엄찬호_Thermo(YPAN)")
    run = runs.setdefault("ypan0720", Run("ypan0720", date(2026, 7, 20), operator="엄찬호", notes=[
        f"source_root={root.relative_to(REPO)}", "log: 20260720_라만 분석(YPAN).txt",
        "sensor manufactured 2025 (환원제 이슈로 과거 센서 사용)"]))
    timing: dict[str, str] = {}
    for p in csvs(root):
        stem, rep, ave = split_rep(p)
        if ave:
            continue
        m = re.fullmatch(r"YPAN_(.+?)_(\d+)", stem)
        label = f"YPAN_{int(m.group(2))}"
        timing[label] = m.group(1)
        files.append(File(p, "ypan0720", "clinical", label=label, rep=rep))
    run.notes.append("specimen timing: " + ", ".join(f"{k}={v}" for k, v in sorted(timing.items())))
    return runs, files


SPA_EPOCH = datetime(1899, 12, 31)


def read_spa(path: Path) -> tuple[np.ndarray, np.ndarray, datetime]:
    """Thermo OMNIC .SPA: acquisition time (uint32 s since 1899-12-31 UTC at offset 296),
    section directory at 304 (key 2 = header with n_points and x range, key 3 = float32 data).
    Validated on 2026-05-06 Reference: Si 520.95, PS 621.23/1001.14/1602.82, grid = CSV grid."""
    import struct
    from datetime import timedelta, timezone
    b = path.read_bytes()
    n_sections = struct.unpack("<H", b[294:296])[0]
    sections: dict[int, tuple[int, int]] = {}
    for i in range(n_sections):
        off = 304 + 16 * i
        sections.setdefault(b[off], struct.unpack("<II", b[off + 2:off + 10]))
    head, _ = sections[2]
    n = struct.unpack("<I", b[head + 4:head + 8])[0]
    x0, x1 = struct.unpack("<ff", b[head + 16:head + 24])
    data, _ = sections[3]
    y = np.frombuffer(b[data:data + 4 * n], dtype="<f4").astype(float)
    x = np.linspace(x0, x1, n)
    if x[0] > x[-1]:
        x, y = x[::-1], y[::-1]
    acquired = (datetime(1899, 12, 31, tzinfo=timezone.utc)
                + timedelta(seconds=struct.unpack("<I", b[296:300])[0])).astimezone()
    return x, y, acquired


def handheld_meta(path: Path) -> dict[str, str]:
    import csv
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {r[0]: (r[1] if len(r) > 1 else "") for r in csv.reader(fh) if r and r[0] != "Intensities"}


def src_april_spa() -> tuple[dict[str, Run], list[File]]:
    """.SPA files of thermo_retest_12groups_20260416-20260519 (0. MB of 4/16 session, 0. Reference of 5/06). Date is the
    acquisition time stored in the file; rows attach to the existing april_YYYYMMDD run."""
    runs: dict[str, Run] = {}
    files: list[File] = []
    for p in sorted((DATA / "03_sers_date_lot_balanced_acquisition" / "thermo_retest_12groups_20260416-20260519").rglob("*.SPA")):
        stem, rep, ave = split_rep(p)
        if ave:
            continue
        day = read_spa(p)[2].date()
        key = f"april_{day:%Y%m%d}"
        runs.setdefault(key, Run(key, day))
        if p.parent.name == "0. MB":
            files.append(File(p, key, "control", control_type="MB_conc_unrecorded"))
        elif p.parent.name == "0. Reference":
            ctype, kind = APRIL_CONTROL[re.sub(r"\s*\d+$", "", stem)]
            files.append(File(p, key, "control", control_type=ctype, source_kind=kind))
        else:
            raise ValueError(f"unexpected SPA location: {p}")
    return runs, files


def src_handheld() -> tuple[dict[str, Run], list[File]]:
    """equipment_comparison_NOR_5devices_undated/handheld: Metrohm Mira P exports; date = CreatedDate in the file."""
    root = DATA / "04_machine_repeatability_tests" / "equipment_comparison_NOR_5devices_undated" / "handheld"
    runs: dict[str, Run] = {}
    files: list[File] = []
    for p in csvs(root):
        meta = handheld_meta(p)
        m = re.fullmatch(r"(.+?)_(\d+|ave)", meta["Name"].strip())
        if m is None:
            raise ValueError(f"unexpected handheld Name: {p}")
        if m.group(2).lower() == "ave":
            continue
        if not meta["DeviceName"].startswith("Mira P"):
            raise ValueError(f"not a Mira P export: {p}")
        day = datetime.fromisoformat(meta["CreatedDate"]).date()
        key = f"handheld_{day:%Y%m%d}"
        runs.setdefault(key, Run(
            key, day, instrument=MIRA_P, laser_wavelength_nm=float(meta["Wavelength"]),
            exposure_time_s=float(meta["IntTime"]), notes=[
                "equipment comparison test (NOR 1-100 x 20)", f"source_root={root.relative_to(REPO)}",
                f"device={meta['DeviceName']}", f"LaserPower setting={meta['LaserPower']} (device level, not mW)",
                f"SmartTip={meta['SmartTipName']} SN {meta['SmartTipSerialNumber']}",
                f"device user={meta['UserName']}", f"LastCalibrationDate={meta['LastCalibrationDate']}"]))
        files.append(File(p, key, "clinical", label=clinical_label(m.group(1)), rep=int(m.group(2))))
    return runs, files


METABOLITE_SHARE = Path("/mnt/c/Users/user/OneDrive - solum/헬스케어-R BD - RnBD/퇴사자/▷보티낫린"
                        "/4. Metabolite/Experimental data")


def src_metabolite() -> tuple[dict[str, Run], list[File]]:
    """Metabolite reference standards. The local copy mixes three original measurement folders;
    each file is assigned to the share folder whose same-named file has the same size (share
    files are online-only, so content cannot be compared). Date/conditions = folder name.
    Un-suffixed files equal the mean of _1.._5 (checked on 69 sets) and are skipped like _ave."""
    root = DATA / "06_supporting_or_previous_outputs" / "thermo_metabolite_standards_20250828-20251203" / "Metabolite analysis_Thermo"
    share = {d: {f: (d / f).stat().st_size for f in os.listdir(d)}
             for d in METABOLITE_SHARE.iterdir() if d.is_dir()}
    runs: dict[str, Run] = {}
    files: list[File] = []
    for p in csvs(root):
        stem, rep, ave = split_rep(p)
        if ave or rep is None:
            continue
        origin = [d for d, names in share.items() if names.get(p.name) == p.stat().st_size]
        if len(origin) != 1:
            raise ValueError(f"cannot place {p.name} in one share folder: {[d.name for d in origin]}")
        folder = origin[0].name
        day = datetime.strptime(folder[:8], "%Y%m%d").date()
        key = f"metabolite_{folder[:8]}"
        runs.setdefault(key, Run(key, day, laser_wavelength_nm=785, laser_power_mw=1.0,
                                 exposure_time_s=0.05, notes=[
            "metabolite reference standards (no clinical sample)",
            f"source_root={root.relative_to(REPO)}",
            f"original folder=퇴사자/▷보티낫린/4. Metabolite/Experimental data/{folder}",
            "date and conditions from original folder name; file matched by name+size"]))
        files.append(File(p, key, "control", control_type=f"metabolite:{stem}", rep=rep))
    return runs, files


UNKNOWN_DATE = "measurement_date=unknown (no surviving acquisition timestamp; file times are copy/sync times)"
RAW_DATED_FOLDERS = ("BLC_1st_", "BLC_2nd_")  # 20260709 Boramae liquid moved to 03_ (2026-09-15)


def x_grid(path: Path, delimiter: str | None) -> tuple[int, float, float]:
    x = np.loadtxt(path, delimiter=delimiter, usecols=0)
    return len(x), round(float(x[0]), 4), round(float(x[-1]), 4)


def src_thermo_retro() -> tuple[dict[str, Run], list[File]]:
    """thermo_retro_12groups_undated main groups (retrospective Thermo). No date survives, so runs are the x-axis
    grid sessions (n_points, first x, last x): one grid = one instrument calibration state,
    which can span several days and cancer groups. No sample straddles two grids."""
    runs: dict[str, Run] = {}
    files: list[File] = []
    groups: dict[str, set[str]] = defaultdict(set)
    for folder in sorted((DATA / "02_sers_primary_pooled_acquisition" / "thermo_retro_12groups_undated").iterdir()):
        if not folder.is_dir() or folder.name.startswith(RAW_DATED_FOLDERS):
            continue
        for p in csvs(folder):
            stem, rep, ave = split_rep(p)
            if ave:
                continue
            grid = x_grid(p, ",")
            key = f"thermo_retro_grid_{grid[1]:.4f}"
            runs.setdefault(key, Run(key, None, notes=[
                UNKNOWN_DATE, f"x_grid_session=n{grid[0]}_{grid[1]:.4f}_{grid[2]:.4f}",
                "retrospective Thermo main set (02_sers_primary_pooled_acquisition/thermo_retro_12groups_undated); conditions not recorded"]))
            groups[key].add(folder.name)
            files.append(File(p, key, "clinical", label=clinical_label(stem), rep=rep))
    for key, names in groups.items():
        runs[key].notes.append("group folders=" + " | ".join(sorted(names)))
    return runs, files


def src_medical() -> tuple[dict[str, Run], list[File]]:
    """ramcheck_retro_12groups_undated (NanoScope Ramcheck A1). Every file shares one grid (2001 pts,
    100-3200 cm-1), so runs follow the 13 delivered folders. Background/ files are role
    'blank' (sensor_background); their pairing with the sample spectrum survives only in the
    file name and the /Background/ path. No wavenumber shift is applied at ingest."""
    runs: dict[str, Run] = {}
    files: list[File] = []
    root = DATA / "02_sers_primary_pooled_acquisition" / "ramcheck_retro_12groups_undated"
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        key = "medical_" + re.sub(r"[^0-9A-Za-z]+", "_", folder.name.split(" (")[0]).strip("_")
        runs[key] = Run(key, None, instrument=RAMCHECK, notes=[
            UNKNOWN_DATE, f"source_root={folder.relative_to(REPO)}",
            "single x grid for all Medical files (n2001, 100-3200); run = delivered folder",
            "Background/ = sensor background (blank rows, paired by file name)",
            "conditions to be added (user)"])
        for p in csvs(folder, exts=(".txt",)):
            stem, rep, ave = split_rep(p)
            if ave:
                continue
            if "Background" in p.parts:
                files.append(File(p, key, "blank", control_type="sensor_background"))
            else:
                files.append(File(p, key, "clinical", label=clinical_label(stem), rep=rep))
    return runs, files


def src_metrohm() -> tuple[dict[str, Run], list[File]]:
    """Metrohm (Mira P) retrospective NOR/PRO set, copied from the OneDrive share into
    data/02_sers_primary_pooled_acquisition/metrohm_retro_NOR-PRO_undated (see SOURCE.txt). Wide CSV: one file per sample, columns
    '<label>_1_5.00_<rep>' + '<label>_Average'; the replicate number is read from the column
    name (two files have shuffled column order). Average columns and 'Baseline corrected
    data' are not loaded. No timestamp survives, so date is NULL; run = delivered folder."""
    import csv
    root = DATA / "02_sers_primary_pooled_acquisition" / "metrohm_retro_NOR-PRO_undated"
    runs: dict[str, Run] = {}
    files: list[File] = []
    for group in ("Normal", "Prostate"):
        folder = root / group / "Raw data"
        key = f"metrohm_{group.lower()}"
        runs[key] = Run(key, None, instrument=MIRA_P, notes=[
            UNKNOWN_DATE, f"source_root={folder.relative_to(REPO)}",
            "copied from OneDrive 헬스케어-R BD - RnBD-DESKTOP-8VL414N/퇴사자/▷보티낫린/3. Urine/"
            "1. Raman analysis/3. Metrohm data (see data/02_sers_primary_pooled_acquisition/metrohm_retro_NOR-PRO_undated/SOURCE.txt)",
            "wide CSV, one column per replicate; column tag '5.00' kept in raw_filename (meaning not recorded)",
            "source_sha256 = sha256 of the extracted column",
            "instrument assumed Mira P (only Metrohm device on record; export has no device field)"])
        for p in csvs(folder):
            with p.open(encoding="utf-8-sig", newline="") as fh:
                header = next(csv.reader(fh))
            for col in header[1:]:
                m = re.fullmatch(r"(.+?)_\d+_[\d.]+_(\d+)", col)
                if m is None:
                    if not col.endswith("_Average"):
                        raise ValueError(f"unexpected column {col!r} in {p}")
                    continue
                files.append(File(p, key, "clinical", label=clinical_label(m.group(1)),
                                  rep=int(m.group(2)), column=col))
    return runs, files


SOURCES = {"mbsu": src_mbsu, "april": src_april, "raw_dated": src_raw_dated, "july": src_july,
           "april_spa": src_april_spa, "handheld": src_handheld, "metabolite": src_metabolite,
           "thermo_retro": src_thermo_retro, "medical": src_medical, "metrohm": src_metrohm}


def read_spectrum(path: Path) -> tuple[list[float], list[float]]:
    if path.suffix.lower() == ".spa":
        x, y, _ = read_spa(path)
    elif path.suffix.lower() == ".csv" and path.read_bytes()[:6] == b'"Name"':
        from scripts.db.aecd_spectrum_formats import read_spectrum as read_any
        data = read_any(path)
        x, y = data.wavenumber, data.intensities
    elif path.suffix.lower() == ".txt":
        a = np.loadtxt(path, ndmin=2)
        if a.shape[1] != 2 or len(a) < 2:
            raise ValueError(f"bad spectrum: {path}")
        x, y = a[:, 0], a[:, 1]
    else:
        return read_csv(path)
    if not np.isfinite(y).all() or not np.all(np.diff(x) > 0):
        raise ValueError(f"bad spectrum: {path}")
    return x.tolist(), y.tolist()


# ───────────────────────── common load ─────────────────────────

def resolve_labels(files: list[File], samples: dict[str, int]) -> tuple[list[File], Counter]:
    boramae = {int(k.split("_")[1]): k for k in samples if re.match(r"^(BPRO|BNOR)_\d+$", k)}
    out, skipped = [], Counter()
    for f in files:
        if f.role != "clinical":
            out.append(f)
            continue
        label = f.label
        m = re.fullmatch(r"(BPRO|BNOR)_(\d+)", label or "")
        if m:
            label = boramae.get(int(m.group(2)))
        if label is None or label not in samples:
            skipped[f.label] += 1
            continue
        out.append(File(f.path, f.run, f.role, label, None, f.rep, f.source_kind, f.column))
    return out, skipped


def fill_missing_rep(files: list[File]) -> list[File]:
    """A clinical file saved without its replicate number (e.g. "LUN 288_.CSV") takes the one
    missing number of its sample, only when exactly one file and one number are missing."""
    by_sample = defaultdict(list)
    for f in files:
        if f.role == "clinical":
            by_sample[(f.run, f.label)].append(f)
    fixed = {}
    for key, group in by_sample.items():
        unnumbered = [f for f in group if f.rep is None]
        if not unnumbered:
            continue
        have = {f.rep for f in group if f.rep is not None}
        missing = sorted(set(range(1, len(group) + 1)) - have)
        if len(unnumbered) != 1 or len(missing) != 1:
            raise ValueError(f"cannot infer replicate number: {key} {[f.path.name for f in unnumbered]}")
        f = unnumbered[0]
        fixed[f.path] = File(f.path, f.run, f.role, f.label, None, missing[0], f.source_kind, f.column)
        print(f"  replicate number filled: {f.path.relative_to(REPO)} -> {missing[0]}")
    return [fixed.get(f.path, f) for f in files]


def assign_points(files: list[File]) -> list[tuple[File, int]]:
    files = fill_missing_rep(files)
    out: list[tuple[File, int]] = []
    counter: Counter = Counter()
    for f in sorted(files, key=lambda f: (f.run, f.role, f.control_type or "", f.label or "",
                                           f.rep or 0, str(f.path))):
        if f.role == "clinical" and f.rep is not None:
            out.append((f, f.rep))
        else:
            k = (f.run, f.role, f.control_type, f.label)
            counter[k] += 1
            out.append((f, counter[k]))
    keys = Counter((f.run, f.role, f.label, f.control_type, p) for f, p in out)
    dups = [k for k, n in keys.items() if n > 1]
    if dups:
        raise ValueError(f"duplicate measurement keys ({len(dups)}): {dups[:5]}")
    return out


def is_all_nan(path: Path) -> bool:
    """Instrument exports of failed acquisitions: every intensity is '#NaN'."""
    if path.suffix.lower() != ".csv" or path.read_bytes()[:6] == b'"Name"':
        return False
    with path.open(encoding="utf-8-sig") as fh:
        values = [line.rstrip("\r\n").split(",", 1)[1] for line in fh if "," in line]
    return bool(values) and all(v.strip() in {"#NaN", "NaN", ""} for v in values)


def read_csv(path: Path) -> tuple[list[float], list[float]]:
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    if a.shape[1] != 2 or len(a) < 2 or not np.isfinite(a).all() or not np.all(np.diff(a[:, 0]) > 0):
        raise ValueError(f"bad spectrum: {path}")
    return a[:, 0].tolist(), a[:, 1].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=sorted(SOURCES), required=True)
    ap.add_argument("--mode", choices=["plan", "dry-run", "commit"], default="plan")
    args = ap.parse_args()

    runs, files = SOURCES[args.source]()
    conn = psycopg2.connect(dbname=os.environ.get("PGDATABASE", "aecd_platform"))
    cur = conn.cursor()
    cur.execute("SELECT solum_label, sample_id FROM master.samples")
    samples = dict(cur.fetchall())
    files, skipped = resolve_labels(files, samples)
    points = assign_points(files)
    per_run = defaultdict(Counter)
    for f, _ in points:
        per_run[f.run][f"{f.role}:{f.control_type}" if f.role != "clinical" else "clinical"] += 1
    print(f"source={args.source} runs={len(runs)} spectra={len(points)} "
          f"clinical samples={len({f.label for f, _ in points if f.role == 'clinical'})}")
    for key in sorted(runs, key=lambda k: (runs[k].day or date.min, k)):
        r = runs[key]
        print(f"  {key:22s} {r.day} op={r.operator} {dict(per_run[key])}")
    if skipped:
        print(f"  SKIPPED unregistered labels ({sum(skipped.values())} files): {dict(skipped)}")
    if args.mode == "plan":
        conn.close()
        return

    try:
        cur.execute("SELECT count(*) FROM measurement.raw_spectra WHERE source_uri = ANY(%s)",
                    ([source_uri(f) for f, _ in points],))
        if cur.fetchone()[0]:
            raise RuntimeError("some of these files are already in raw_spectra")
        for name in sorted({r.instrument for r in runs.values()} & INSTRUMENTS.keys()):
            spec = INSTRUMENTS[name]
            cur.execute(
                "INSERT INTO measurement.instruments (instrument_name, manufacturer, model_name,"
                " location, status) SELECT %s,%s,%s,%s,'active' WHERE NOT EXISTS"
                " (SELECT 1 FROM measurement.instruments WHERE instrument_name=%s)",
                (name, spec["manufacturer"], spec["model_name"], spec["location"], name))
        cur.execute("SELECT instrument_name, instrument_id FROM measurement.instruments")
        instruments = dict(cur.fetchall())
        run_ids = {}
        for key, r in sorted(runs.items()):
            if not per_run[key]:
                continue
            cur.execute("SELECT measurement_run_id FROM measurement.runs WHERE notes LIKE %s",
                        (f"loader=aecd_historical_measurement_load:%run_key={key};%",))
            existing = cur.fetchall()
            if len(existing) > 1:
                raise RuntimeError(f"run_key {key} matches several runs")
            if existing:
                run_ids[key] = existing[0][0]
                print(f"  attach to existing run {run_ids[key]} ({key})")
                continue
            cur.execute(
                "INSERT INTO measurement.runs (instrument_id, measurement_date, operator_name,"
                " laser_wavelength_nm, laser_power_mw, exposure_time_s, notes)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s) RETURNING measurement_run_id",
                (instruments[r.instrument], r.day, r.operator, r.laser_wavelength_nm,
                 r.laser_power_mw, r.exposure_time_s,
                 f"loader=aecd_historical_measurement_load:{args.source}; run_key={key}; "
                 + "; ".join(r.notes)))
            run_ids[key] = cur.fetchone()[0]
        # control/blank points continue after any rows already in a reused run
        cur.execute("""SELECT measurement_run_id, measurement_role, coalesce(control_type,''),
                              max(point_no) FROM measurement.measurements
                       WHERE measurement_run_id = ANY(%s) AND sample_id IS NULL GROUP BY 1,2,3""",
                    (list(run_ids.values()),))
        offset = {(r, role, ct): mx for r, role, ct, mx in cur.fetchall()}
        cur.execute("""SELECT count(*) FROM measurement.measurements
                       WHERE measurement_run_id = ANY(%s) AND sample_id = ANY(%s)""",
                    (list(run_ids.values()), [samples[f.label] for f, _ in points if f.role == "clinical"]))
        if cur.fetchone()[0]:
            raise RuntimeError("clinical rows for these samples already exist in the target runs")
        points = [(f, p + offset.get((run_ids[f.run], f.role, f.control_type or ""), 0)
                   if f.role != "clinical" else p) for f, p in points]
        invalid = {f.path for f, _ in points if is_all_nan(f.path)}
        count_sql = """SELECT count(m.*), count(rs.*) FROM measurement.measurements m
                       LEFT JOIN measurement.raw_spectra rs USING (measurement_id)
                       WHERE m.measurement_run_id = ANY(%s)"""
        cur.execute(count_sql, (list(run_ids.values()),))
        before_meas, before_raw = cur.fetchone()
        for path in sorted(invalid):
            print(f"  invalid (all-NaN) measurement, no raw_spectra: {path.relative_to(REPO)}")
        for i in range(0, len(points), 500):
            chunk = points[i:i + 500]
            mids = execute_values(
                cur, "INSERT INTO measurement.measurements (measurement_run_id, sample_id, point_no,"
                " measurement_role, control_type, status, invalid_reason) VALUES %s"
                " RETURNING measurement_id",
                [(run_ids[f.run], samples.get(f.label) if f.role == "clinical" else None, p,
                  f.role, f.control_type,
                  "invalid" if f.path in invalid else "acquired",
                  "all intensities #NaN in source file" if f.path in invalid else None)
                 for f, p in chunk], fetch=True)
            rows = []
            for (f, _), (mid,) in zip(chunk, mids):
                if f.path in invalid:
                    continue
                if f.column is None:
                    x, y = read_spectrum(f.path)
                    sha, name = hashlib.sha256(f.path.read_bytes()).hexdigest(), f.path.name
                else:
                    x, y, sha = read_wide_column(f.path, f.column)
                    name = f"{f.path.name}#{f.column}"
                rows.append((mid, source_uri(f), name, f.source_kind, x, y,
                             len(x), x[0], x[-1], sha))
            if rows:
                execute_values(
                    cur, "INSERT INTO measurement.raw_spectra (measurement_id, source_uri, raw_filename,"
                    " source_kind, wavenumber, intensities, n_points, x_min, x_max, source_sha256)"
                    " VALUES %s", rows)
        cur.execute(count_sql, (list(run_ids.values()),))
        after_meas, after_raw = cur.fetchone()
        n_meas, n_raw = after_meas - before_meas, after_raw - before_raw
        print(f"  verify: measurements={n_meas} raw_spectra={n_raw} expected={len(points)}"
              f" (invalid without spectrum: {len(invalid)})")
        if (n_meas, n_raw) != (len(points), len(points) - len(invalid)):
            raise RuntimeError("verification count mismatch")
        print(f"  undo ids: run_ids={sorted(run_ids.values())}")
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
