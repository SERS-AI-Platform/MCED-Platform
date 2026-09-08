# Terminology & Coding Standards

This is the **prescriptive** development standard for SERS-AI / MCED-Platform: the **one** approved term or rule per concept. For values that change over time (cohort counts, model thresholds), this document gives the **rule** and names the **single source of truth (SSOT)** — it does **not** snapshot volatile numbers (those drift and go stale).

> **Scope** — Part A: domain & clinical terminology · Part B: code & naming conventions
> **Related** — contribution mechanics: [`CONTRIBUTING.md`](../../CONTRIBUTING.md) · onboarding: [`ONBOARDING.md`](../../ONBOARDING.md) · domain-rule SSOT: [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md)

## Enforcement legend

Every rule is tagged with **how** it is enforced:

- 🤖 **CI-enforced** — an automated check blocks merge (core import smoke; ruff `E,F,I,W`; configured mypy gate; pytest with coverage floor; process coverage summary; pre-commit Conventional Commits)
- 👁️ **Convention / review-only** — not auto-checked; relies on review and discipline

> ⚠️ **Enforcement honesty**: ruff currently selects only `E, F, I, W` (see `pyproject.toml`).
> mypy covers CLI/config/scoring plus `src/sers/master_data/**` and `src/sers/mlflow_tracking.py`; full-package coverage remains a migration backlog item.
> **Docstrings (`D`) and naming (`N`) are NOT auto-enforced** — they are 👁️ conventions, required by review, not by CI. Don't assume CI will catch a missing docstring or a bad name.

---

# Part A — Domain & Clinical Terminology

## A1. Two-stage pipeline

| Concept | Standard term | Avoid |
|---|---|---|
| Step 1 — cancer vs non-cancer | **Cancer Screening** | Stage 1 |
| Step 2 — subtype (output: *cancer-type pattern comparison*) | **Cancer Type ID** | Stage 2, confidence, diagnostic probability |

👁️ **Known deviation (migrate)**: internal code comments still use `Stage 1/2` (`scripts/deployment/sers_predict.py`, `scripts/generation/generate_monthly_ppt.py`). User-facing output is already compliant. *Patent/IP documents keep "Stage 1/2" as legal language — exempt.*

## A2. Clinical decision-rule terms (user-facing)

| Concept | Standard term | Avoid |
|---|---|---|
| User-facing cancer vs non-cancer score | **SSI score / SSI 점수** | raw probability, risk score |
| Fixed reference used for SSI interpretation | **SSI interpretation reference / SSI 해석 기준값** | operating mode, standalone decision threshold |
| At least 3 of 5 uploaded measurements are QC-valid replicates with `SSI_i > 4.0` | **Further evaluation recommended / 추가 확인 권고** | positive, strong positive, cancer diagnosed |
| Fewer than 3 of 5 uploaded measurements are QC-valid replicates with `SSI_i > 4.0` | **Below decision threshold / 기준 미만** | negative, normal |
| Step 2 output | **Cancer-type pattern comparison / 암종별 패턴 비교 결과** | probability of diagnosis, diagnostic confidence |

## A3. SSI score mechanics

- **SSI (user-facing score)**: patient-level adjunctive mean-signal score on a **0–10** scale with interpretation anchor **4.0** (stable code constants in `src/sers/scoring.py`: `SSI_MAX = 10.0`, `SSI_CUTOFF = 4.0`).
- **Rule**: the fixed model probability threshold maps each probability to SSI 4.0 via a piecewise-linear transform (`scoring.py: probability_to_ssi`). For a single replicate, probability below the threshold maps to SSI `[0, 4)` and probability above it maps to SSI `[4, 10]`. Patient SSI transforms the mean probability across QC-valid replicates.
- **Patient decision**: for a QC-valid test, the rounded patient mean SSI determines the action: `< 1.0` = Below Decision Threshold, `1.0–4.0` inclusive = Consider Further Evaluation, and `> 4.0` = Further Evaluation Recommended. Replicate-level threshold exceedance counts are retained as reference data only.
- **The internal probability cutoff is NOT the displayed SSI cutoff.** The UI/reports must not expose it as a selectable per-patient mode. The stored patient result holds `{score, threshold, decision rule, model version}` — never implying the user picked a different data mode.
- **SSOT for the exact probability threshold**: the deployed artifact's `manifest.json → operating_modes` — **not** this document. The deployed/manifest value is canonical (e.g., STK-V2 `balanced` is currently `0.60`).
- ℹ️ The previously-documented `0.4439` was a stale snapshot and has been retired — always read the live threshold from the manifest, never hardcode it here.

## A4. Cohort / group codes

- **Rule**: each cohort is a **3–4 letter UPPERCASE code** (e.g., `CRC`, `LUN`, `NOR`, `PRO`, `BLC`, `CPAN`).
- **Composite aliases (policy)**: **`PAN = CPAN + YPAN`**. **`SPAN` (Samsung, post-op) is excluded from screening** and must never be merged into PAN. **`NOR` includes `YNOR`**.
- **SSOT for codes, hospitals, aliases, and sample-count metadata**: `config/config.yaml` (`folder_to_group`, `group_metadata`, `group_aliases`) + [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md). **Do not hardcode counts in this doc — they drift.**
- **Only authoritative source-file → protocol mapping**: `src/sers/master_data/clinical_inventory.py`. `config/config.yaml: group_metadata.protocol` is a cohort-level discovery default, not authority for a physical file. SQL and documentation must mirror the registry and must not create a second mapping.
- **Protocol migration**: records traced to `SMCMD06_췌장암.xlsx` use `CPAN` / `SMCMD06`; all 19 SPAN CSV sources use `SMCXD02`; all 20 YPAN CSV sources plus YPAN/YNOR use `SMCXD04`; Boramae `BPRO` and `BNOR` use `SMCXD07`. Migrate legacy `SPAN_CRF` / `YPAN_CRF` protocol values only after exact source-file matching; do not rewrite identity from cohort aliases or row order.

## A5. Model name

- **STK-V2** (a.k.a. **uSERS-Net**) = the production **stacking ensemble** (base models + ElasticNet meta-learner, trained with nested CV). Code: `src/sers/models/usersnet/`; training: `scripts/training/train_usersnet.py`.

## A6. Replicate aggregation

- Two modes — **mean** and **medoid** (CLI `--aggregate {mean|medoid|none}`; `src/sers/qc/qc.py: find_medoid`, `select_medoid_spectra`). **Always state the aggregation mode when reporting results** (it is one of the four run-comparison keys; see A9).

## A7. Operating / threshold profiles

- Profiles: **`screening`** (high sensitivity), **`balanced`** (Youden's J), **`confirmatory`** (high specificity).
- These are **model-evaluation profiles, NOT per-patient data attributes.** Do not expose as a selectable patient mode.
- Deployment standard = **`balanced`** (webapp `STANDARD_DECISION_PROFILE`). `standard_balanced` is the **deployed-standard flag** for the balanced profile, not a separate profile.
- **SSOT for per-profile thresholds**: artifact `manifest.json → operating_modes` (values differ per model).

## A8. Patient-level interpretation

For a QC-valid test, the rounded patient mean SSI determines the screening action: `< 1.0` = Below Decision Threshold, `1.0–4.0` inclusive = Consider Further Evaluation, and `> 4.0` = Further Evaluation Recommended. Replicate-level cancer-signal counts remain reference information. Do **not** use "strong positive", "weak positive", or similar grading.

## A9. Other domain rules (SSOT)

Hospital-confound disclaimer, the run-comparison **4-key rule** (aggregation · cancer set · non-cancer set · sample count), QC handling, and CSV `utf-8-sig` encoding are defined in [`.github/copilot-instructions.md`](../../.github/copilot-instructions.md) and the Obsidian vault `01_Projects/SERS-AI/`.

---

# Part B — Code & Naming Conventions

Format: **concept → rule → real example → avoid**. Examples are verbatim from the codebase.

## B1. Naming

| Concept | Rule | Example (real) | Avoid | Enforce |
|---|---|---|---|---|
| Functions / methods | `snake_case`, verb-first | `read_spectrum`, `calculate_intensity_gate`, `run_qc_pipeline` | `camelCase`, noun-only names | 👁️ |
| Classes | `PascalCase` | `SpectrumID`, `PreprocessingConfig`, `DatasetResult` | `snake_case` class names | 👁️ |
| Module constants | `UPPER_CASE` at module top | `FINGERPRINT_REGION = (400, 2200)`, `SSI_CUTOFF = 4.0` | magic literals inline in functions | 👁️ |
| Private helpers | leading underscore `_` | `_group_spectra_by_sample` | exposing internals via `__all__` | 👁️ |
| Cohort codes | 3–4 letter UPPERCASE (see A4) | `CRC`, `CPAN` | lowercase / long names | 👁️ |
| Modules / files | `snake_case.py` | `calibration_transfer.py` | `CamelCase.py` | 👁️ |

## B2. Type annotations 👁️

- **Rule (going forward)**: annotate all public parameters **and** return types. Use **modern builtin generics** (`dict[...]`, `list[...]`, `tuple[...]`) and **`X | None`**. Type NumPy arrays as **`np.ndarray`**.
- **Example (verbatim)**:
  ```python
  def make_common_grid(
      x_arrays: list[np.ndarray],
      n_points: int | None = None,
      x_min: float | None = None,
      x_max: float | None = None,
  ) -> np.ndarray:
  ```
- **Avoid in new code**: `typing.Dict/List/Tuple/Optional`.
- ⚠️ **Known deviation (migrate)**: `src/sers/preprocessing.py` still uses `Dict[Tuple, ...]` and `Optional[float]`; some private helpers in `qc/qc.py` lack hints.

## B3. Docstrings 👁️

- **Rule**: **NumPy convention** (`Parameters` / `Returns` / `Notes` / `Raises`) on all public functions and classes.
- **Example (verbatim, abridged)**:
  ```python
  def trim_spectrum(x, y, region=FINGERPRINT_REGION):
      """
      Trim spectrum to specified wavenumber region.

      Parameters
      ----------
      x : np.ndarray
          Wavenumber values (cm⁻¹)
      y : np.ndarray
          Intensity values
      region : tuple of float
          (min_wavenumber, max_wavenumber) to keep

      Returns
      -------
      x_trimmed, y_trimmed : tuple of np.ndarray

      Notes
      -----
      Must be applied BEFORE smoothing/baseline to avoid edge artifacts.
      """
  ```
- **Avoid**: Google-style `Args:` / `Returns:` headers.

## B4. Core data structures

| Type | Rule | Definition (real) |
|---|---|---|
| Spectra container | canonical in-memory form | `dict[(group, sample_id, replicate), (x: np.ndarray, y: np.ndarray)]` — Medical data prepends `equipment` to the key |
| Spectrum identifier | `NamedTuple` | `SpectrumID(group: str, sample_id: str, replicate: int)` |
| Dataset bundle | `@dataclass` | `DatasetResult(spectra, metadata: pd.DataFrame, failed_files: list[Path])` |
| Configuration | **frozen** `@dataclass` hierarchy | `Config` → `PreprocessingConfig`, `QCConfig`, `ModelingConfig`, `MedicalConfig`, `DisplayConfig` |

- **Rule**: configuration is the **single source of thresholds/parameters — no hardcoded magic numbers** in modules; load from `config/config.yaml`. Use frozen dataclasses for config, `NamedTuple` for lightweight identifiers.

## B5. Public API

- **Rule**: the package's public surface is exactly what `src/sers/__init__.py: __all__` lists. Names with a `_` prefix are internal and may change without notice. Visualization helpers are **lazy-loaded** (via `__getattr__`) so model/training imports stay headless.

## B6. Spectrum filename convention

- **Format**: `GROUP<space|_>sampleid_replicate.ext` → e.g. **`CRC 001_1.CSV`** (group = 3–4 letter code; `sample_id` zero-padded; `replicate` 1–6). Parsed by `parse_filename()` → `SpectrumID`.
- **SSOT for per-group patterns**: `config/config.yaml: filename_pattern`.

## B7. Commits, lint, formatting (enforcement)

| Item | Rule | Enforce |
|---|---|---|
| Commit messages | Conventional Commits (`feat / fix / docs / refactor / test / chore / perf / build / ci`) | 🤖 pre-commit |
| Import smoke | Core public API imports without optional visualization dependencies | 🤖 CI |
| Lint | ruff selects **`E, F, I, W`** (line length 100; `E501/E402/E701/E722` ignored) | 🤖 CI |
| Type check | configured mypy checks CLI/config/scoring, `master_data/**`, and `mlflow_tracking.py` (`pyproject.toml: tool.mypy.files`) | 🤖 CI |
| Test coverage | pytest coverage floor starts at 35%; process summary is generated from `coverage.xml` | 🤖 CI |
| Docstrings (`D`), naming (`N`) | NumPy docstrings, snake_case — **not in ruff `select`** | 👁️ review only |
| Format | `ruff format` | 👁️ pre-commit (if installed) |
| CSV output encoding | `encoding="utf-8-sig"` (Excel/한글 BOM) | 👁️ |

## B8. Known deviations to migrate (backlog)

1. Legacy `typing.*` generics in `preprocessing.py` → modern builtins (B2).
2. `Stage 1/2` in internal comments → Cancer Screening / Cancer Type ID (A1; patents exempt).
3. Private helpers missing type hints (`qc/qc.py`).
4. Expand mypy from the governed CLI/master-data gate to the remaining `src/sers/` package.
5. Raise the coverage floor after adding tests for CLI execution, scoring, calibration, visualization, and deployment paths.

---

## SSOT pointers (where the live truth lives)

| Topic | Source of truth |
|---|---|
| Cohort codes · counts · hospitals · aliases | `config/config.yaml`, `.github/copilot-instructions.md` |
| Physical clinical source file → protocol | `src/sers/master_data/clinical_inventory.py` |
| Pipeline parameters (preprocess · QC · modeling) | `config/config.yaml` |
| Model operating-mode thresholds | deployed artifact `manifest.json` |
| Domain rules (hospital confound · run comparison · QC) | `.github/copilot-instructions.md`, vault `01_Projects/SERS-AI/` |
| Public API surface | `src/sers/__init__.py: __all__` |
| Contribution mechanics & workflow | `../CONTRIBUTING.md` |
