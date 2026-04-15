#!/usr/bin/env python3
"""
Migrate existing figures to centralized results/figures/ directory.

Copies (does NOT delete) PNG files from scattered locations to the new structure:
  results/figures/
  ├── training/{slug}/stage1/
  ├── training/{slug}/stage2/
  ├── analysis/{name}/
  └── weekend/{name}/

Usage:
    python scripts/migrate_figures.py --dry-run   # preview only
    python scripts/migrate_figures.py              # execute copy
"""

import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def training_dir_to_slug(train_dir: Path) -> str:
    results_training = RESULTS_DIR / "training"
    try:
        rel = train_dir.relative_to(results_training)
    except ValueError:
        # Try models/results path
        models_results = PROJECT_ROOT / "models" / "results"
        try:
            rel = train_dir.relative_to(models_results)
        except ValueError:
            rel = train_dir
    parts = [p for p in rel.parts if p not in (".", "..")]
    return "_".join(parts)


def collect_tasks():
    """Collect all (src, dst) pairs for figure migration."""
    tasks = []

    # 1. Training evaluation figures: results/training/**/evaluation/stage*/*.png
    for png in sorted(RESULTS_DIR.rglob("training/**/evaluation/**/*.png")):
        # Find the evaluation dir
        parts = png.parts
        eval_idx = None
        for i, p in enumerate(parts):
            if p == "evaluation":
                eval_idx = i
                break
        if eval_idx is None:
            continue

        train_dir = Path(*parts[:eval_idx])
        slug = training_dir_to_slug(train_dir)
        rel_in_eval = png.relative_to(Path(*parts[: eval_idx + 1]))
        dst = FIG_DIR / "training" / slug / rel_in_eval
        tasks.append((png, dst))

    # 2. Training root figures: results/training/**/*.png (not inside evaluation/)
    for png in sorted(RESULTS_DIR.glob("training/**/*.png")):
        if "evaluation" in png.parts:
            continue  # already handled above
        # Find parent training dir (one with training_summary.json or the immediate parent)
        rel = png.relative_to(RESULTS_DIR / "training")
        parts = list(rel.parts)
        if len(parts) <= 1:
            slug = parts[0].replace(".png", "") if parts else "root"
        else:
            slug = "_".join(parts[:-1])
        dst = FIG_DIR / "training" / slug / png.name
        tasks.append((png, dst))

    # 3. Analysis figures: results/{stacking_ensemble,multiview_*,cross_instrument,...}/*.png
    analysis_dirs = [
        "stacking_ensemble", "multiview_ensemble", "multiview_ablation",
        "multiview_derivative", "cross_instrument", "contrastive_sweep",
    ]
    for dirname in analysis_dirs:
        src_dir = RESULTS_DIR / dirname
        if not src_dir.exists():
            # Also check results/training/{dirname}
            src_dir = RESULTS_DIR / "training" / dirname
        if not src_dir.exists():
            continue
        for png in sorted(src_dir.rglob("*.png")):
            rel = png.relative_to(src_dir)
            dst = FIG_DIR / "analysis" / dirname / rel
            tasks.append((png, dst))

    # 4. Weekend experiment figures
    weekend_dir = RESULTS_DIR / "weekend_experiments"
    if weekend_dir.exists():
        for png in sorted(weekend_dir.rglob("*.png")):
            rel = png.relative_to(weekend_dir)
            parts = list(rel.parts)
            exp_name = parts[0] if parts else "unknown"
            rest = Path(*parts[1:]) if len(parts) > 1 else Path(png.name)
            dst = FIG_DIR / "weekend" / exp_name / rest
            tasks.append((png, dst))

    # 5. Feature study figures
    for study_dir in sorted(RESULTS_DIR.glob("feature_study_*")):
        for png in sorted(study_dir.rglob("*.png")):
            rel = png.relative_to(study_dir)
            dst = FIG_DIR / "analysis" / study_dir.name / rel
            tasks.append((png, dst))

    # 6. Existing results/figures/ (already in the right area but may need restructuring)
    # Skip — these are already in place

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Migrate figures to centralized directory")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, don't copy")
    args = parser.parse_args()

    tasks = collect_tasks()

    if not tasks:
        print("No figures to migrate.")
        return

    # Deduplicate
    seen = set()
    unique_tasks = []
    for src, dst in tasks:
        key = (str(src), str(dst))
        if key not in seen:
            seen.add(key)
            unique_tasks.append((src, dst))
    tasks = unique_tasks

    # Group by destination category
    categories = {}
    for src, dst in tasks:
        try:
            cat = dst.relative_to(FIG_DIR).parts[0]
        except (ValueError, IndexError):
            cat = "other"
        categories.setdefault(cat, []).append((src, dst))

    print(f"{'DRY RUN — ' if args.dry_run else ''}Figure Migration Summary")
    print("=" * 60)
    for cat, cat_tasks in sorted(categories.items()):
        print(f"\n  {cat}/ — {len(cat_tasks)} files")
        for src, dst in cat_tasks[:5]:
            print(f"    {src.relative_to(PROJECT_ROOT)}")
            print(f"      → {dst.relative_to(PROJECT_ROOT)}")
        if len(cat_tasks) > 5:
            print(f"    ... and {len(cat_tasks) - 5} more")

    print(f"\n  Total: {len(tasks)} files")

    if args.dry_run:
        print("\n  (dry run — no files copied)")
        return

    copied = 0
    skipped = 0
    for src, dst in tasks:
        if dst.exists():
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1

    print(f"\n  Copied: {copied}, Skipped (exists): {skipped}")
    print(f"  Output: {FIG_DIR.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()
