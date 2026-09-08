from __future__ import annotations

import re
from datetime import datetime

_COMPACT_MONTH = re.compile(r"^\d{6}$")
_COMPACT_DATE = re.compile(r"^\d{8}$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}")


def safe_float(value: str) -> float | None:
    text = value.strip()
    if text in {"", "-", " "}:
        return None
    try:
        return float(text.removeprefix(">").removeprefix("<"))
    except ValueError:
        return None


def normalize_date(value: str) -> str | None:
    text = value.strip()
    if text == "":
        return None
    if _COMPACT_MONTH.fullmatch(text):
        return f"{text[:4]}-{text[4:6]}-01"
    if _COMPACT_DATE.fullmatch(text):
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    if _ISO_DATE.match(text):
        return text[:10]
    for date_format in ("%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(text, date_format).date().isoformat()
        except ValueError:
            continue
    return text
