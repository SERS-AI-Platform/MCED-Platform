"""sers compare — Head-to-head comparison report across models.

Reads artifact manifests and run metrics for each requested model, then
emits a markdown report + CSV metrics table under `results/comparisons/`.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import click

from sers.models._registry import MODEL_REGISTRY, get_spec


@click.command("compare")
@click.option(
    "--models",
    "models_arg",
    required=True,
    help="Comma-separated model keys (e.g. usersnet,resnet18,lr-fusion). "
    "Use 'all' to include every model with supports_compare=True.",
)
@click.option(
    "--report",
    "report_dir",
    default=None,
    help="Output directory. Default: results/comparisons/{timestamp}_{keys}/",
)
@click.option(
    "--metrics",
    default="auc_binary,auc_multi,f1,sensitivity,specificity",
    help="Comma-separated metric keys to include in the comparison table.",
)
def compare(models_arg: str, report_dir: str | None, metrics: str):
    """Compare multiple trained models head-to-head.

    \b
    Examples:
        sers compare --models usersnet,resnet18,lr-fusion
        sers compare --models all --report results/comparisons/full/
        sers compare --models usersnet,sersnet-ensemble --metrics auc_binary,f1
    """
    if models_arg.strip() == "all":
        keys = [k for k, s in MODEL_REGISTRY.items() if s.supports_compare]
    else:
        keys = [k.strip() for k in models_arg.split(",") if k.strip()]

    specs = [get_spec(k) for k in keys]
    metric_keys = [m.strip() for m in metrics.split(",") if m.strip()]

    if report_dir is None:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M")
        tag = "_".join(k for k in keys)[:40]
        report_dir = f"results/comparisons/{ts}_{tag}"

    out = Path(report_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for spec in specs:
        row = {
            "key": spec.key,
            "display_name": spec.display_name,
            "category": spec.category,
            "paper_name": spec.paper_name,
            "supports_clinical": spec.supports_clinical,
            "artifact_dir": spec.artifact_dir,
        }

        if spec.artifact_dir:
            manifest_path = Path(spec.artifact_dir) / "manifest.json"
            if manifest_path.exists():
                try:
                    manifest = json.loads(manifest_path.read_text())
                    flat_metrics = _flatten(manifest.get("metrics", {}))
                    for m in metric_keys:
                        row[m] = flat_metrics.get(m, "—")
                    row["version"] = manifest.get("version", "—")
                    row["trained_at"] = manifest.get("trained_at", "—")
                except Exception as exc:
                    row["error"] = f"manifest parse error: {exc}"
            else:
                row["error"] = f"no manifest at {manifest_path}"
        else:
            row["error"] = "no artifact_dir in registry"

        rows.append(row)

    # CSV
    import csv
    csv_path = out / "metrics_table.csv"
    all_keys = sorted({k for r in rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=all_keys)
        w.writeheader()
        w.writerows(rows)

    # Markdown
    md_path = out / "report.md"
    lines = [
        f"# Model Comparison — {', '.join(keys)}",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Metrics",
        "",
        "| Model | Category | Version | " + " | ".join(metric_keys) + " |",
        "|---|---|---|" + "|".join("---" for _ in metric_keys) + "|",
    ]
    for r in rows:
        vals = [str(r.get(m, "—")) for m in metric_keys]
        lines.append(
            f"| {r['display_name']} | {r['category']} | {r.get('version', '—')} | "
            + " | ".join(vals) + " |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Metrics sourced from each model's `artifacts/.../manifest.json`.",
        "- Missing cells (—) mean the manifest did not record that metric.",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"- **{r['display_name']}**: {r['error']}")
    md_path.write_text("\n".join(lines), encoding="utf-8")

    click.echo(f"Wrote {csv_path}")
    click.echo(f"Wrote {md_path}")


def _flatten(d: dict, parent: str = "") -> dict:
    """Flatten nested dict: {'holdout': {'auc': 0.9}} -> {'holdout.auc': 0.9, 'auc': 0.9}."""
    out: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            nested = _flatten(v, parent=k)
            out.update(nested)
            for nk, nv in nested.items():
                # Expose bare leaf keys too, last-writer-wins
                leaf = nk.split(".")[-1]
                out.setdefault(leaf, nv)
        else:
            full = f"{parent}.{k}" if parent else k
            out[full] = v
            out.setdefault(k, v)
    return out
