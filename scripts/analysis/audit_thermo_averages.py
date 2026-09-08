from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import click

from sers.raw_set.audit import AverageAuditPolicy, audit_average_file


@click.command()
@click.option(
    "--data-root",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/raw_data"),
    show_default=True,
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("results/raw_set/average_audit"),
    show_default=True,
)
@click.option("--max-normalized-rmse", type=float, default=1e-4, show_default=True)
def main(data_root: Path, out_dir: Path, max_normalized_rmse: float) -> None:
    policy = AverageAuditPolicy(max_normalized_rmse=max_normalized_rmse)
    average_paths = sorted(
        {
            *data_root.rglob("*_ave.CSV"),
            *data_root.rglob("*_ave.csv"),
        }
    )
    audits = [audit_average_file(path, policy) for path in average_paths]
    out_dir.mkdir(parents=True, exist_ok=True)

    row_path = out_dir / "average_audit.csv"
    with row_path.open("w", encoding="utf-8-sig", newline="") as handle:
        fieldnames = [
            "average_path",
            "replicate_count",
            "rmse",
            "normalized_rmse",
            "correlation",
            "shared_grid",
            "target_usable",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for audit in audits:
            writer.writerow(
                {
                    "average_path": str(audit.average_path.relative_to(data_root)),
                    "replicate_count": audit.replicate_count,
                    "rmse": audit.rmse,
                    "normalized_rmse": audit.normalized_rmse,
                    "correlation": audit.correlation,
                    "shared_grid": audit.shared_grid,
                    "target_usable": audit.target_usable,
                }
            )

    summary = {
        "data_root": str(data_root.resolve()),
        "policy": asdict(policy),
        "average_files": len(audits),
        "usable_targets": sum(audit.target_usable for audit in audits),
        "rejected_targets": sum(not audit.target_usable for audit in audits),
        "shared_grid": sum(audit.shared_grid for audit in audits),
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    click.echo(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
