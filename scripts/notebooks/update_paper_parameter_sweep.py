#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["nbformat>=5.9"]
# ///

# ─── How to run ───
# 1. Install uv (if not installed):
#      curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run directly (no venv, no pip install needed):
#      uv run update_paper_parameter_sweep.py
# 3. Or make executable and run:
#      chmod +x update_paper_parameter_sweep.py && ./update_paper_parameter_sweep.py
# ──────────────────

from __future__ import annotations

from pathlib import Path

import nbformat

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "paper_2509_25964_parameter_sweep.ipynb"


def replace_once(source: str, old: str, new: str) -> str:
    if old not in source:
        raise ValueError(f"Notebook source fragment not found: {old[:80]!r}")
    return source.replace(old, new, 1)


def update_markdown(source: str) -> str:
    return replace_once(
        source,
        "- 기본 파일은 `results/clean_cohort_20260605/clean_processed_spectra.csv`입니다.\n  다른 cohort를 사용하려면 `DATA_PATH`만 바꿉니다.",
        "- 기본 입력은 PostgreSQL `public.raw_spectra`의 raw replicate입니다.\n  CSV fallback은 DB 연결 없이 Notebook 구조를 검증할 때만 사용합니다.",
    )


def update_setup_markdown(source: str) -> str:
    return replace_once(
        source,
        "이 Notebook은 기본적으로 CSV로 실행되어 DB 자격증명 없이도 재현됩니다. Windows에서\nDB를 사용하려면 **cell 3보다 먼저** 환경변수를 설정합니다.",
        "이 Notebook은 기본적으로 PostgreSQL raw DB에서 실행됩니다. DB 연결 없이 구조만\n검증하려면 `SERS_NOTEBOOK_DATA_SOURCE=csv`를 설정합니다. Windows에서 DB를\n사용하려면 **cell 3보다 먼저** 환경변수를 설정합니다.",
    )


def update_parameters(source: str) -> str:
    source = replace_once(
        source,
        'DATA_SOURCE = os.environ.get("SERS_NOTEBOOK_DATA_SOURCE", "csv").lower()',
        'DATA_SOURCE = os.environ.get("SERS_NOTEBOOK_DATA_SOURCE", "postgres").lower()',
    )
    return replace_once(
        source,
        'DB_GROUPS: set[str] | None = None\n',
        'DB_GROUPS: set[str] | None = None\nPROCESS_TRACE_SUBJECTS = 6\n',
    )


def update_loader(source: str) -> str:
    source = replace_once(
        source,
        '''            SELECT source_batch, group_code, sample_id, replicate,\n                   wavenumber, intensities\n''',
        '''            SELECT source_path, source_batch, source_kind, group_code,\n                   sample_id, replicate, wavenumber, intensities\n''',
    )
    source = replace_once(
        source,
        '''        "group": spectra["group_code"].astype(str).to_numpy(),\n        "sample_id": spectra["sample_id"].astype(str).to_numpy(),\n        "replicate": spectra["replicate"].astype(int).to_numpy(),\n        "source_batch": spectra["source_batch"].astype(str).to_numpy(),\n''',
        '''        "source_path": spectra["source_path"].astype(str).to_numpy(),\n        "source_batch": spectra["source_batch"].astype(str).to_numpy(),\n        "source_kind": spectra["source_kind"].astype(str).to_numpy(),\n        "group": spectra["group_code"].astype(str).to_numpy(),\n        "sample_id": spectra["sample_id"].astype(str).to_numpy(),\n        "replicate": spectra["replicate"].astype(int).to_numpy(),\n''',
    )
    source = replace_once(
        source,
        '''    if NORMALIZE_ROWS:\n        values = rowwise_zscore(values)\n    labels, label_names = make_labels(metadata)\n    return values, metadata, target_grid, label_names\n''',
        '''    aligned_values = values.copy()\n    if NORMALIZE_ROWS:\n        values = rowwise_zscore(values)\n    labels, label_names = make_labels(metadata)\n    return values, metadata, target_grid, label_names, aligned_values\n''',
    )
    source = replace_once(
        source,
        '''def load_spectra() -> tuple[np.ndarray, pd.DataFrame, np.ndarray, list[str]]:\n    if DATA_SOURCE == "csv":\n        raw_values, raw_metadata, raw_grid = load_csv_source(DATA_PATH)\n    elif DATA_SOURCE == "postgres":\n        raw_values, raw_metadata, raw_grid = load_postgres_source()\n    elif DATA_SOURCE == "supabase":\n        raw_values, raw_metadata, raw_grid = load_supabase_source()\n    else:\n        raise ValueError("DATA_SOURCE must be 'csv', 'postgres', or 'supabase'")\n    return prepare_spectra(raw_values, raw_metadata, raw_grid)\n\n\nX, metadata, grid, label_names = load_spectra()\n''',
        '''def load_spectra() -> tuple[\n    np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame,\n    np.ndarray, list[str], np.ndarray\n]:\n    if DATA_SOURCE == "csv":\n        raw_values, raw_metadata, raw_grid = load_csv_source(DATA_PATH)\n    elif DATA_SOURCE == "postgres":\n        raw_values, raw_metadata, raw_grid = load_postgres_source()\n    elif DATA_SOURCE == "supabase":\n        raw_values, raw_metadata, raw_grid = load_supabase_source()\n    else:\n        raise ValueError("DATA_SOURCE must be 'csv', 'postgres', or 'supabase'")\n    processed_values, processed_metadata, target_grid, labels, aligned_values = prepare_spectra(\n        raw_values, raw_metadata, raw_grid\n    )\n    return (\n        raw_values, raw_metadata, raw_grid, processed_values, processed_metadata,\n        target_grid, labels, aligned_values,\n    )\n\n\raw_values, raw_metadata, raw_grid, X, metadata, grid, label_names, aligned_values = load_spectra()\n''',
    )
    return source.replace(chr(13) + "aw_values", "raw_values")


PROCESSING_CELL = '''### 3. Raw-to-model processing trace

이 셀은 DB에서 가져온 raw intensity가 현재 설정에서 어떻게 모델 입력으로 바뀌는지
보여줍니다. 현재 Notebook의 실제 변환은 공통 grid 보간, `SPECTRUM_MODE` 범위 자르기,
`NORMALIZE_ROWS=True`일 때 row-wise z-score입니다. smoothing이나 baseline correction은
이 실험에 몰래 적용하지 않습니다.

```python
trace_subjects = metadata["subject_id"].drop_duplicates().head(PROCESS_TRACE_SUBJECTS)
trace_mask = metadata["subject_id"].isin(trace_subjects).to_numpy()
trace_indices = np.flatnonzero(trace_mask)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for index in trace_indices:
    row = metadata.iloc[index]
    label = f"{row['group']} {row['sample_id']} r{row['replicate']}"
    axes[0].plot(grid, aligned_values[index], alpha=0.35, linewidth=0.8, label=label)
    axes[1].plot(grid, X[index], alpha=0.35, linewidth=0.8, label=label)
axes[0].set_title("Raw DB intensity after common-grid interpolation and range selection")
axes[1].set_title("Model input after row-wise normalization")
axes[1].set_xlabel("Raman shift (cm⁻¹)")
for axis in axes:
    axis.set_ylabel("intensity")
    axis.grid(alpha=0.25)
axes[0].legend(loc="upper right", ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

trace_columns = [
    column for column in [
        "source_path", "source_batch", "source_kind", "group", "sample_id", "replicate", "subject_id"
    ] if column in metadata.columns
]
trace_table = metadata.loc[trace_mask, trace_columns].copy()
trace_table["raw_mean"] = aligned_values[trace_mask].mean(axis=1)
trace_table["raw_std"] = aligned_values[trace_mask].std(axis=1)
trace_table["model_mean"] = X[trace_mask].mean(axis=1)
trace_table["model_std"] = X[trace_mask].std(axis=1)
display(trace_table)

profile_columns = ["source_batch", "group"] if "source_batch" in metadata.columns else ["group"]
batch_profile = metadata.assign(
    raw_mean=aligned_values.mean(axis=1),
    raw_std=aligned_values.std(axis=1),
    model_mean=X.mean(axis=1),
    model_std=X.std(axis=1),
).groupby(profile_columns, as_index=False).agg(
    spectra=("subject_id", "size"),
    subjects=("subject_id", "nunique"),
    raw_std_median=("raw_std", "median"),
    model_std_median=("model_std", "median"),
)
display(batch_profile.head(50))
```
'''


PARAMETERS_SOURCE = '''RUN_MODE = "smoke"  # "smoke" or "full"

DATA_SOURCE = os.environ.get("SERS_NOTEBOOK_DATA_SOURCE", "postgres").lower()
DATA_PATH = PROJECT_ROOT / "results" / "clean_cohort_20260605" / "clean_processed_spectra.csv"
DB_DATASET_VERSION = "aecd_all_measurements_20260803"
DB_SOURCE_DOMAINS: set[str] | None = {"medical"}
DB_SOURCE_KINDS: set[str] | None = None
DB_ARTIFACT_ROLES: set[str] | None = {"raw"}
DB_GROUPS: set[str] | None = None
DB_INCLUDE_AVERAGES = False
DB_EXCLUDE_CALIBRATION = True
DB_LIMIT: int | None = None
PROCESS_TRACE_SUBJECTS = 6

TASK = "multiclass_group"  # "multiclass_group" or "binary_cancer"
TARGET_GROUPS: list[str] | None = None
EXCLUDE_GROUPS: set[str] = set()
CANCER_GROUPS: set[str] | None = None

SPECTRUM_MODE = "paper_like"  # "paper_like" or "native_grid"
PAPER_START_CM = 400.0
PAPER_END_CM = 2200.0
PAPER_STEP_CM = 1.0
NORMALIZE_ROWS = True

USE_BATCH_NORM = False
POOLING_DROPOUT = 0.5
CLASS_WEIGHTING = "balanced"  # "balanced" or "none"
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
LR_FACTOR = 0.7
LR_PATIENCE = 3
EARLY_STOPPING_PATIENCE = 5
SMOKE_EPOCHS = 3
FULL_EPOCHS = 30
NUM_WORKERS = 0

EXPERIMENT_GRID = [
    {"name": "m2_n1", "pool_kernel": 2, "pool_layers": 1},
    {"name": "m2_n3", "pool_kernel": 2, "pool_layers": 3},
    {"name": "m64_n1", "pool_kernel": 64, "pool_layers": 1},
]
MAX_CONFIGS = len(EXPERIMENT_GRID)
EPOCHS = SMOKE_EPOCHS if RUN_MODE == "smoke" else FULL_EPOCHS

TRAIN_NOISE_STD = 0.01
TRAIN_SCALE_RANGE = (0.95, 1.05)
TRAIN_SHIFT_MAX_CM = 0.0
EVAL_SHIFTS_CM = [0.0, 3.0, 10.0, 15.0, 30.0]

print({
    "data_source": DATA_SOURCE,
    "data_path": str(DATA_PATH),
    "db_dataset_version": DB_DATASET_VERSION,
    "db_source_domains": DB_SOURCE_DOMAINS,
    "db_artifact_roles": DB_ARTIFACT_ROLES,
    "task": TASK,
    "spectrum_mode": SPECTRUM_MODE,
    "epochs": EPOCHS,
    "configs": [c["name"] for c in EXPERIMENT_GRID[:MAX_CONFIGS]],
})'''


LOADER_SOURCE = '''def parse_wavenumber_columns(columns: list[str]) -> np.ndarray:
    values = []
    for column in columns:
        match = re.search(r"x_([-+0-9.eE]+)", column)
        if match is None:
            raise ValueError(f"Cannot parse wavenumber from {column}")
        values.append(float(match.group(1)))
    return np.asarray(values, dtype=np.float32)


def rowwise_zscore(values: np.ndarray) -> np.ndarray:
    mean = np.nanmean(values, axis=1, keepdims=True)
    std = np.nanstd(values, axis=1, keepdims=True)
    safe_std = np.where(std > 1e-8, std, 1.0)
    return np.nan_to_num((values - mean) / safe_std, nan=0.0, posinf=0.0, neginf=0.0)


def make_labels(metadata: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    if TASK == "binary_cancer":
        if CANCER_GROUPS is None:
            raise ValueError("Set CANCER_GROUPS explicitly before using binary_cancer")
        labels = metadata["group"].isin(CANCER_GROUPS).astype(np.int64).to_numpy()
        return labels, ["non_cancer", "cancer"]
    if TASK != "multiclass_group":
        raise ValueError(f"Unsupported TASK: {TASK}")
    label_names = sorted(metadata["group"].unique().tolist())
    mapping = {name: index for index, name in enumerate(label_names)}
    labels = metadata["group"].map(mapping).astype(np.int64).to_numpy()
    return labels, label_names


def load_csv_source(path: Path) -> tuple[list[np.ndarray], pd.DataFrame, list[np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(path)
    raw = pd.read_csv(path)
    spectral_columns = [column for column in raw.columns if column.startswith("x_")]
    if not spectral_columns:
        raise ValueError("No x_<wavenumber> spectral columns found")
    source_grid = parse_wavenumber_columns(spectral_columns)
    order = np.argsort(source_grid)
    source_grid = source_grid[order]
    values = raw.loc[:, np.asarray(spectral_columns)[order]].to_numpy(dtype=np.float32)
    metadata = raw.drop(columns=spectral_columns).copy()
    metadata["source_path"] = metadata.index.astype(str)
    metadata["source_batch"] = "csv"
    metadata["source_domain"] = "csv"
    metadata["source_kind"] = "csv"
    metadata["artifact_role"] = "processed"
    metadata["group"] = metadata["group"].astype(str)
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["subject_id"] = metadata["group"] + "::" + metadata["sample_id"]
    return [row for row in values], metadata, [source_grid] * len(values)


def _query_filter(
    clauses: list[str], parameters: list[object], column: str, values: set[str] | None
) -> None:
    if values is not None:
        clauses.append(f"{column} = ANY(%s)")
        parameters.append(sorted(values))


def load_postgres_source() -> tuple[list[np.ndarray], pd.DataFrame, list[np.ndarray]]:
    clauses = ["dataset_version = %s", "group_code IS NOT NULL"]
    parameters: list[object] = [DB_DATASET_VERSION]
    _query_filter(clauses, parameters, "source_domain", DB_SOURCE_DOMAINS)
    _query_filter(clauses, parameters, "source_kind", DB_SOURCE_KINDS)
    _query_filter(clauses, parameters, "artifact_role", DB_ARTIFACT_ROLES)
    if not DB_INCLUDE_AVERAGES:
        clauses.append("is_averaged = FALSE")
    if DB_EXCLUDE_CALIBRATION:
        clauses.append("artifact_role <> 'calibration_control'")
    if DB_GROUPS is not None:
        _query_filter(clauses, parameters, "group_code", DB_GROUPS)
    query = """
        SELECT source_path, source_domain, source_kind, source_batch,
               instrument_key, preparation, artifact_role, variant,
               group_code, sample_id, replicate, measurement_key,
               control_type, acquisition_date, wavenumber, intensities
        FROM public.aecd_measurements
        WHERE """ + " AND ".join(clauses) + " ORDER BY id"
    if DB_LIMIT is not None:
        query += " LIMIT %s"
        parameters.append(DB_LIMIT)

    password = os.environ.get("PGPASSWORD") or getpass("PostgreSQL password (not saved): ")
    if not password:
        raise RuntimeError("A PostgreSQL password is required")
    connection_kwargs = {
        "host": os.environ.get("PGHOST", "localhost"),
        "port": int(os.environ.get("PGPORT", "5432")),
        "dbname": os.environ.get("PGDATABASE", "sers_clinical"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": password,
    }
    with psycopg2.connect(**connection_kwargs) as connection:
        with connection.cursor() as cursor:
            cursor.execute(query, tuple(parameters))
            rows = cursor.fetchall()
    if not rows:
        raise ValueError("The PostgreSQL filters returned no spectra")

    columns = [
        "source_path", "source_domain", "source_kind", "source_batch",
        "instrument_key", "preparation", "artifact_role", "variant",
        "group", "sample_id", "replicate", "measurement_key",
        "control_type", "acquisition_date",
    ]
    metadata = pd.DataFrame([row[:14] for row in rows], columns=columns)
    metadata["group"] = metadata["group"].astype(str)
    metadata["sample_id"] = metadata["sample_id"].astype(str)
    metadata["subject_id"] = (
        metadata["source_domain"].astype(str) + "::"
        + metadata["source_batch"].astype(str) + "::"
        + metadata["group"] + "::" + metadata["sample_id"]
    )
    values = [np.asarray(row[15], dtype=np.float32) for row in rows]
    grids = [np.asarray(row[14], dtype=np.float32) for row in rows]
    print({
        "db_rows": len(rows),
        "db_dataset_version": DB_DATASET_VERSION,
        "db_source_domains": metadata["source_domain"].value_counts().to_dict(),
        "db_artifact_roles": metadata["artifact_role"].value_counts().to_dict(),
        "db_groups": metadata["group"].nunique(),
    })
    return values, metadata, grids


def prepare_spectra(
    raw_values: list[np.ndarray],
    raw_metadata: pd.DataFrame,
    raw_grids: list[np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    if not raw_values or len(raw_values) != len(raw_metadata) or len(raw_values) != len(raw_grids):
        raise ValueError("Spectrum values, metadata, and grids must have equal non-zero lengths")
    normalized_pairs = []
    for source_grid, values_row in zip(raw_grids, raw_values):
        order = np.argsort(source_grid)
        ordered = source_grid[order]
        if len(ordered) < 2 or not np.all(np.diff(ordered) > 0):
            raise ValueError("Every spectrum must have a strictly increasing wavenumber grid")
        normalized_pairs.append((ordered, np.asarray(values_row, dtype=np.float32)[order]))
    normalized_grids = [grid for grid, _ in normalized_pairs]
    if SPECTRUM_MODE == "paper_like":
        lower_bound = max(PAPER_START_CM, max(float(grid[0]) for grid in normalized_grids))
        upper_bound = min(PAPER_END_CM, min(float(grid[-1]) for grid in normalized_grids))
        lower = np.ceil(lower_bound / PAPER_STEP_CM) * PAPER_STEP_CM
        upper = np.floor(upper_bound / PAPER_STEP_CM) * PAPER_STEP_CM
        if upper <= lower:
            raise ValueError("No common wavenumber range remains after DB filtering")
        target_grid = np.arange(lower, upper + PAPER_STEP_CM / 2.0, PAPER_STEP_CM, dtype=np.float32)
    elif SPECTRUM_MODE == "native_grid":
        target_grid = normalized_grids[0]
        if any(
            len(grid) != len(target_grid) or not np.allclose(grid, target_grid, rtol=0.0, atol=1e-4)
            for grid in normalized_grids[1:]
        ):
            raise ValueError("native_grid requires identical wavenumber grids for all DB rows")
    else:
        raise ValueError(f"Unsupported SPECTRUM_MODE: {SPECTRUM_MODE}")
    values = np.vstack([
        np.interp(target_grid, grid, values_row).astype(np.float32)
        for grid, values_row in normalized_pairs
    ])
    metadata = raw_metadata.copy()
    keep = ~metadata["group"].isin(EXCLUDE_GROUPS).to_numpy()
    if TARGET_GROUPS is not None:
        keep &= metadata["group"].isin(TARGET_GROUPS).to_numpy()
    keep &= np.isfinite(values).all(axis=1)
    values = values[keep]
    metadata = metadata.loc[keep].reset_index(drop=True)
    aligned_values = values.copy()
    if NORMALIZE_ROWS:
        values = rowwise_zscore(values)
    labels, label_names = make_labels(metadata)
    return values, metadata, target_grid, label_names, aligned_values


def load_spectra() -> tuple[
    list[np.ndarray], pd.DataFrame, list[np.ndarray], np.ndarray, pd.DataFrame,
    np.ndarray, list[str], np.ndarray,
]:
    if DATA_SOURCE == "csv":
        raw_values, raw_metadata, raw_grids = load_csv_source(DATA_PATH)
    elif DATA_SOURCE == "postgres":
        raw_values, raw_metadata, raw_grids = load_postgres_source()
    else:
        raise ValueError("DATA_SOURCE must be 'postgres' or 'csv'")
    processed_values, processed_metadata, target_grid, labels, aligned_values = prepare_spectra(
        raw_values, raw_metadata, raw_grids
    )
    return (
        raw_values, raw_metadata, raw_grids, processed_values,
        processed_metadata, target_grid, labels, aligned_values,
    )


raw_values, raw_metadata, raw_grids, X, metadata, grid, label_names, aligned_values = load_spectra()
y, label_names = make_labels(metadata)
grid_step = float(np.median(np.diff(grid))) if len(grid) > 1 else 1.0
print({
    "X_shape": X.shape,
    "grid_cm-1": (float(grid[0]), float(grid[-1])),
    "grid_step_cm-1": grid_step,
    "subjects": int(metadata["subject_id"].nunique()),
    "classes": label_names,
})'''


def update_notebook_cells(notebook: nbformat.NotebookNode) -> None:
    for cell in notebook.cells:
        source = "".join(cell.get("source", []))
        if source.startswith("from __future__ import annotations"):
            if "import os\n" not in source:
                source = source.replace("import random\n", "import os\nimport random\n", 1)
            if "from getpass import getpass\n" not in source:
                source = source.replace("from pathlib import Path\n", "from getpass import getpass\nfrom pathlib import Path\n", 1)
            if "import psycopg2\n" not in source:
                source = source.replace("import pandas as pd\n", "import pandas as pd\nimport psycopg2\n", 1)
            cell["source"] = source
        elif source.startswith("RUN_MODE ="):
            cell["source"] = PARAMETERS_SOURCE
        elif source.startswith("def parse_wavenumber_columns"):
            cell["source"] = LOADER_SOURCE
        elif source.startswith("## Context & Methods"):
            cell["source"] = source.replace(
                "- 기본 파일은 `results/clean_cohort_20260605/clean_processed_spectra.csv`입니다.\n  다른 cohort를 사용하려면 `DATA_PATH`만 바꿉니다.",
                "- 기본 입력은 PostgreSQL `public.aecd_measurements`의 `medical + raw`입니다.\n  DB 조건은 Experiment parameters 셀에서 바꿀 수 있고, CSV는 fallback으로만 사용합니다.",
            )


def main() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    update_notebook_cells(notebook)
    for cell in notebook.cells:
        source = "".join(cell.get("source", []))
        if source.startswith("## Context & Methods"):
            if "- 기본 파일은" in source:
                cell["source"] = update_markdown(source)
        elif source.startswith("### 1. Experiment parameters"):
            if "기본적으로 CSV로" in source:
                cell["source"] = update_setup_markdown(source)
        elif source.startswith("RUN_MODE ="):
            if 'SERS_NOTEBOOK_DATA_SOURCE", "csv"' in source:
                cell["source"] = update_parameters(source)
        elif source.startswith("def parse_wavenumber_columns"):
            if "SELECT source_batch, group_code" in source:
                cell["source"] = update_loader(source)
        if cell.cell_type == "code":
            cell["source"] = cell["source"].replace(
                "\naw_values, raw_metadata, raw_grid, X, metadata, grid, label_names, aligned_values = load_spectra()",
                "\nraw_values, raw_metadata, raw_grid, X, metadata, grid, label_names, aligned_values = load_spectra()",
            )
            cell["source"] = cell["source"].replace(
                chr(13) + "aw_values, raw_metadata, raw_grid, X, metadata, grid, label_names, aligned_values = load_spectra()",
                "raw_values, raw_metadata, raw_grid, X, metadata, grid, label_names, aligned_values = load_spectra()",
            )

    insertion_index = next(
        index for index, cell in enumerate(notebook.cells)
        if "### 3. Data checks and subject-level split" in "".join(cell.get("source", []))
    )
    if not any("### 3. Raw-to-model processing trace" in "".join(cell.get("source", [])) for cell in notebook.cells):
        notebook.cells.insert(insertion_index, nbformat.v4.new_markdown_cell(PROCESSING_CELL))

    trace_index = next(
        index for index, cell in enumerate(notebook.cells)
        if "### 3. Raw-to-model processing trace" in "".join(cell.get("source", []))
    )
    trace_source = notebook.cells[trace_index].source
    if "```python" in trace_source:
        markdown_source, code_source = trace_source.split("```python", 1)
        code_source, _ = code_source.split("```", 1)
        notebook.cells[trace_index] = nbformat.v4.new_markdown_cell(markdown_source.strip())
        notebook.cells.insert(trace_index + 1, nbformat.v4.new_code_cell(code_source.strip()))
    else:
        trace_code = PROCESSING_CELL.split("```python", 1)[1].split("```", 1)[0].strip()
        for cell in notebook.cells:
            if cell.cell_type == "code" and cell.source.startswith("trace_subjects ="):
                cell["source"] = trace_code

    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.execution_count = None
            cell.outputs = []
    nbformat.write(notebook, NOTEBOOK)
    print(NOTEBOOK)


if __name__ == "__main__":
    main()
