import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.deployment import clinical_report


def test_browser_fallback_generates_pdf_when_weasyprint_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: a Windows browser capable of printing HTML to PDF.
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "C:/Program Files/Microsoft/Edge/msedge.exe" if name == "msedge" else None,
    )

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        output_argument = next(item for item in command if item.startswith("--print-to-pdf="))
        Path(output_argument.split("=", 1)[1]).write_bytes(b"%PDF-1.7\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # When: the HTML report is rendered through the browser fallback.
    pdf_bytes = clinical_report._generate_pdf_with_browser("<html><body>report</body></html>")

    # Then: a real PDF payload is returned to the download route.
    assert pdf_bytes.startswith(b"%PDF-")


def test_browser_fallback_uses_isolated_profile_when_default_profile_is_busy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given: the installed browser may already have its normal user profile open.
    monkeypatch.setattr(
        shutil,
        "which",
        lambda name: "C:/Program Files/Microsoft/Edge/msedge.exe" if name == "msedge" else None,
    )
    observed_command: list[str] = []

    def fake_run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
        observed_command.extend(command)
        output_argument = next(item for item in command if item.startswith("--print-to-pdf="))
        Path(output_argument.split("=", 1)[1]).write_bytes(b"%PDF-1.7\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    # When: the report is rendered by the browser fallback.
    clinical_report._generate_pdf_with_browser("<html><body>report</body></html>")

    # Then: the headless process uses a request-local profile instead of the busy default.
    profile_argument = next(
        (item for item in observed_command if item.startswith("--user-data-dir=")),
        None,
    )
    assert profile_argument is not None
    assert Path(profile_argument.split("=", 1)[1]).name == "browser-profile"
