"""Fail-closed entry point for legacy row-order staging reproduction."""

import importlib
import sys


class UngovernedRowOrderLinkageError(RuntimeError):
    """Reject use of historical row-order linkage without explicit consent."""


LEGACY_OVERRIDE = "--allow-legacy-row-order-linkage"


def _require_legacy_override() -> None:
    if LEGACY_OVERRIDE not in sys.argv:
        raise UngovernedRowOrderLinkageError(
            "reingest_staging is a legacy non-governed utility that links CRC rows "
            "by order; use the governed clinical registry, or explicitly pass "
            "--allow-legacy-row-order-linkage for historical reproduction only"
        )
    sys.argv.remove(LEGACY_OVERRIDE)


_require_legacy_override()

importlib.import_module("sers.reingest_staging_workflow")
