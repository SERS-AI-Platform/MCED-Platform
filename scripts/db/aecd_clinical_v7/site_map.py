#!/usr/bin/env python3
"""Solum label -> collection site mapping for the normalized clinical workbook.

WHY this lives in code: the workbook's own ``Label definition`` sheet states the
hospital for every label range, but it does so as free text ("PRO_1 ~ PRO_100").
Encoding those ranges here makes the mapping reviewable and testable, and lets
the loader fail loudly when a new label falls outside every known range.

Site codes and names follow ``src/sers/master_data/clinical_source_contracts.py``
so that the platform and the file-based pipeline agree on site identity.

Protocol codes (SMCXD01/02/03/04/05/06/07, SMCMD06) are deliberately NOT modelled
here: a hospital runs several protocols (CBNUH alone spans four), so a protocol is
not a site. Protocol lives with the source-file contracts, not with master.sites.
"""

from __future__ import annotations

from typing import Final

SITE_NAMES: Final[dict[str, str]] = {
    "BORAMAE": "Seoul National University Boramae Hospital",
    "CBNUH": "Chungbuk National University Hospital",
    "IJBPH": "Inje University Busan Paik Hospital",
    "SAMSUNG": "Samsung Seoul Hospital",
    "SNUH": "Seoul National University Hospital",
    "SSMH": "Seoul St. Mary's Hospital",
    "YONSEI": "Yonsei Severance Hospital",
    "YPNUH": "Yangsan Pusan National University Hospital",
}

# (label prefix, first number, last number, site_code) -- inclusive range.
# Source: "Label definition" sheet of 전체환자_임상정보_정규화_v7.xlsx.
LABEL_RANGES: Final[tuple[tuple[str, int, int, str], ...]] = (
    ("NOR", 1, 100, "YPNUH"),
    ("NOR", 101, 400, "CBNUH"),
    ("DIA", 1, 100, "YPNUH"),
    ("HBP", 1, 100, "YPNUH"),
    ("H. D.", 1, 100, "YPNUH"),
    ("BLC", 1, 300, "CBNUH"),
    ("CRC", 1, 300, "CBNUH"),
    ("PRO", 1, 400, "CBNUH"),
    ("PAN", 1, 90, "CBNUH"),
    ("PAN", 91, 120, "SNUH"),
    ("BRE", 1, 330, "IJBPH"),
    ("OVA", 1, 30, "IJBPH"),
    ("OVA", 31, 70, "SNUH"),
    ("LUN", 1, 30, "SNUH"),
    ("LUN", 31, 200, "SSMH"),
    ("LUN", 201, 300, "SNUH"),
    ("SPAN", 1, 126, "SAMSUNG"),
    ("YNOR", 1, 60, "YONSEI"),
    ("YPAN", 1, 60, "YONSEI"),
    ("BNOR", 1, 160, "BORAMAE"),
    ("BPRO", 1, 160, "BORAMAE"),
)


def site_for_label(prefix: str, number: int) -> str | None:
    """Return the site_code for a label, or None when no range matches."""
    for range_prefix, low, high, site_code in LABEL_RANGES:
        if prefix == range_prefix and low <= number <= high:
            return site_code
    return None
