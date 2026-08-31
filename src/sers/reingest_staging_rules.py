import re
from importlib import import_module

pd = import_module("pandas")
GLEASON_PATTERN = re.compile(
    r"^\s*(\d+)\s*\(\s*([1-5])\s*\+\s*([1-5])\s*\)\s*$"
)


def map_pan_ajcc(t, n, m):
    if pd.isna(t) or pd.isna(n) or pd.isna(m):
        return "Unknown", "Missing TNM component"
    t = str(t).strip().upper()
    n = str(n).strip().upper()
    m = str(m).strip().upper()
    if t == "NX" or n == "NX" or m == "MX":
        return "Unknown", f"Cannot stage: {t}/{n}/{m} contains X (unknown)"
    if m == "M1":
        return "IV", f"{t} {n} {m} → Stage IV (distant metastasis)"
    if m == "M0":
        if t == "T4":
            return "III", f"{t} {n} {m} → Stage III (T4, locally advanced)"
        if n in ("N2", "N3"):
            return "III", f"{t} {n} {m} → Stage III (≥N2, regional spread)"
        if n == "N1":
            return "IIB", f"{t} {n} {m} → Stage IIB (N1, limited nodal)"
        if n == "N0":
            if t in ("T1", "T1A", "T1B", "T1C"):
                return "IA", f"{t} {n} {m} → Stage IA (T1 N0)"
            if t == "T2":
                return "IB", f"{t} {n} {m} → Stage IB (T2 N0)"
            if t == "T3":
                return "IIA", f"{t} {n} {m} → Stage IIA (T3 N0)"
    return "Unknown", f"Unmapped combination: {t} {n} {m}"


def parse_pro_tnm(val):
    if pd.isna(val):
        return None, None, None
    val = str(val).strip()
    t_match = re.search(r"[cp]?(T\d[a-c]?)", val, re.IGNORECASE)
    n_match = re.search(r"(N[0-2x])", val, re.IGNORECASE)
    m_match = re.search(r"(M[01][a-c]?)", val, re.IGNORECASE)
    t = t_match.group(1).upper() if t_match else None
    if n_match:
        n = n_match.group(1).upper()
    elif "NX" in val.upper():
        n = "NX"
    else:
        n = None
    m = m_match.group(1).upper() if m_match else None
    return t, n, m


def parse_gleason_components(
    val: str | int | float | None,
) -> tuple[int, int, int] | None:
    if pd.isna(val):
        return None
    match = GLEASON_PATTERN.fullmatch(str(val))
    if match is None:
        return None
    score = int(match.group(1))
    primary_pattern = int(match.group(2))
    secondary_pattern = int(match.group(3))
    if score != primary_pattern + secondary_pattern:
        return None
    return score, primary_pattern, secondary_pattern


def parse_gleason(val: str | int | float | None) -> int | None:
    components = parse_gleason_components(val)
    if components is None:
        return None
    return components[0]


def gleason_to_grade_group(val: str | int | float | None) -> int | None:
    components = parse_gleason_components(val)
    if components is None:
        return None
    score, primary_pattern, secondary_pattern = components
    if score <= 6:
        return 1
    if (primary_pattern, secondary_pattern) == (3, 4):
        return 2
    if (primary_pattern, secondary_pattern) == (4, 3):
        return 3
    if score == 8:
        return 4
    if score in (9, 10):
        return 5
    return None


def map_crc_ajcc(t, n, m):
    if pd.isna(t) or pd.isna(n) or pd.isna(m):
        return "Unknown"
    t = str(t).strip().upper()
    n = str(n).strip().upper()
    m = str(m).strip().upper()
    if m in ("M1", "M1A", "M1B", "M1C"):
        return "IV"
    if n in ("N0", "NX"):
        if t in ("T1", "T1A", "T1B", "T1C", "T2"):
            return "I"
        if t == "T3":
            return "IIA"
        if t == "T4A":
            return "IIB"
        if t == "T4B":
            return "IIC"
    elif n in ("N1", "N1A", "N1B", "N1C"):
        if t in ("T1", "T1A", "T1B", "T1C", "T2"):
            return "IIIA"
        if t in ("T3", "T4A"):
            return "IIIB"
        if t == "T4B":
            return "IIIC"
    elif n in ("N2", "N2A"):
        if t in ("T1", "T1A", "T1B", "T1C"):
            return "IIIA"
        if t in ("T2", "T3"):
            return "IIIB"
        if t in ("T4A", "T4B"):
            return "IIIC"
    elif n == "N2B":
        if t in ("T1", "T1A", "T1B", "T1C", "T2"):
            return "IIIB"
        if t in ("T3", "T4A", "T4B"):
            return "IIIC"
    return "Unknown"


def normalize_lun_stage(val):
    value = str(val).strip().upper().replace("STGAE", "STAGE").replace("STAGEI ", "STAGE ")
    match = re.search(r"(I{1,3}V?|IV)[A-B]?\d?", value)
    if match:
        stage = match.group(0)
        if stage.startswith("IV"):
            return "IV"
        if stage.startswith("III"):
            return "III"
        if stage.startswith("II"):
            return "II"
        if stage.startswith("I"):
            return "I"
    stage_number = value.replace("STAGE ", "").replace("STAGE", "").strip()
    if stage_number.startswith("1"):
        return "I"
    if stage_number.startswith("2"):
        return "II"
    if stage_number.startswith("3"):
        return "III"
    if stage_number.startswith("4"):
        return "IV"
    return "Unknown"
