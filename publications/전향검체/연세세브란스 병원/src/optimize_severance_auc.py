# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy>=1.24",
#   "pandas>=2.0",
#   "pyyaml>=6.0",
#   "scikit-learn>=1.3",
#   "scipy>=1.10",
# ]
# ///
# ─── How to run ───
# uv run publications/전향검체/연세세브란스\ 병원/src/optimize_severance_auc.py
from __future__ import annotations

from severance_lr_outputs import generate_lr_outputs


def main() -> None:
    dataset, result = generate_lr_outputs()
    print(
        f"Severance optimized LR: n={len(dataset.labels)}, "
        f"repeated CV AUROC={result.repeat_auc.mean():.3f} +/- {result.repeat_auc.std():.3f}"
    )


if __name__ == "__main__":
    main()
