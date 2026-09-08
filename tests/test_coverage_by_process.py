from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "quality" / "coverage_by_process.py"
SPEC = importlib.util.spec_from_file_location("coverage_by_process", MODULE_PATH)
coverage_by_process = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = coverage_by_process
SPEC.loader.exec_module(coverage_by_process)


def write_coverage_xml(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" ?>
<coverage>
  <packages>
    <package name="sers">
      <classes>
        <class filename="src/sers/cli/__init__.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="0"/>
          </lines>
        </class>
        <class filename="src/sers/scoring.py">
          <lines>
            <line number="1" hits="1"/>
            <line number="2" hits="1"/>
            <line number="3" hits="0"/>
          </lines>
        </class>
        <class filename="src/sers/qc/qc.py">
          <lines>
            <line number="1" hits="1"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>
""",
        encoding="utf-8",
    )


def test_process_report_groups_coverage_by_mced_process(tmp_path):
    coverage_xml = tmp_path / "coverage.xml"
    write_coverage_xml(coverage_xml)

    files = coverage_by_process.parse_coverage_xml(coverage_xml)
    report = {
        process.bucket.key: process
        for process in coverage_by_process.build_process_report(files)
    }

    assert report["cli_workflows"].covered == 1
    assert report["cli_workflows"].statements == 2
    assert report["risk_scoring"].covered == 2
    assert report["risk_scoring"].statements == 3
    assert report["quality_control"].covered == 1
    assert report["quality_control"].statements == 1


def test_markdown_report_keeps_deployment_software_scope_explicit(tmp_path):
    coverage_xml = tmp_path / "coverage.xml"
    write_coverage_xml(coverage_xml)

    files = coverage_by_process.parse_coverage_xml(coverage_xml)
    report = coverage_by_process.build_process_report(files)
    markdown = coverage_by_process.format_markdown(report)

    assert "Coverage By Process" in markdown
    assert "SSI/CTI risk scoring" in markdown
    assert "deployment_software" in markdown
    assert "moved to the SERS-Clinical-App repo" in markdown
