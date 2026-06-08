# Software Bill of Materials

## Document Metadata

| Field | Value |
| --- | --- |
| Product | sers-analysis |
| Version | 0.1.0 |
| Supplier | SOLUM Healthcare |
| License | Proprietary |
| Generated | 2026-06-08T22:57:54Z |
| SBOM standard | CycloneDX 1.5 JSON plus human-readable Markdown |
| Machine-readable file | `docs/sbom.cdx.json` |

## Scope

This SBOM covers the SERS spectral analysis pipeline package, locked Python
dependencies, and Docker deployment manifests present in this repository.

Inputs used:

- `pyproject.toml`
- `requirements.lock`
- `infra/Dockerfile`
- `infra/Dockerfile.api`
- `LICENSE`

Environment manifests reviewed:

- `config/environment.yml` (python>=3.10)
- `config/environment.lock.yml` (python=3.14.2=h32b2ec7_101_cp314)

Excluded from package license resolution:

- Raw or clinical data under `data/`
- Generated results under `results/`
- Runtime OS packages inside built container images
- Package license metadata not present in repository-local manifests

## Primary Component

| Component | Version | Type | Supplier | License |
| --- | --- | --- | --- | --- |
| sers-analysis | 0.1.0 | application | SOLUM Healthcare | Proprietary |

## Required Python Components

| Package | Version | Kind | Scope | License |
| --- | --- | --- | --- | --- |
| click | 8.4.1 | direct | runtime | NOASSERTION |
| joblib | 1.5.3 | transitive | transitive-runtime | NOASSERTION |
| numpy | 2.4.6 | direct | runtime | NOASSERTION |
| pandas | 3.0.3 | direct | runtime | NOASSERTION |
| python-dateutil | 2.9.0.post0 | transitive | transitive-runtime | NOASSERTION |
| pyyaml | 6.0.3 | direct | runtime | NOASSERTION |
| scikit-learn | 1.8.0 | direct | runtime | NOASSERTION |
| scipy | 1.17.1 | direct | runtime | NOASSERTION |
| six | 1.17.0 | transitive | transitive-runtime | NOASSERTION |
| threadpoolctl | 3.6.0 | transitive | transitive-runtime | NOASSERTION |
| tqdm | 4.67.3 | direct | runtime | NOASSERTION |

## Optional And Development Python Components

| Package | Version | Kind | Scope | License |
| --- | --- | --- | --- | --- |
| ast-serialize | 0.5.0 | transitive | transitive-optional | NOASSERTION |
| contourpy | 1.3.3 | transitive | transitive-optional | NOASSERTION |
| coverage | 7.14.1 | transitive | transitive-optional | NOASSERTION |
| cycler | 0.12.1 | transitive | transitive-optional | NOASSERTION |
| fonttools | 4.63.0 | transitive | transitive-optional | NOASSERTION |
| iniconfig | 2.3.0 | transitive | transitive-optional | NOASSERTION |
| kiwisolver | 1.5.0 | transitive | transitive-optional | NOASSERTION |
| librt | 0.11.0 | transitive | transitive-optional | NOASSERTION |
| lightgbm | 4.6.0 | direct | ml | NOASSERTION |
| matplotlib | 3.10.9 | direct | viz | NOASSERTION |
| mypy | 2.1.0 | direct | dev | NOASSERTION |
| mypy-extensions | 1.1.0 | transitive | transitive-optional | NOASSERTION |
| nvidia-nccl-cu12 | 2.30.4 | transitive | transitive-optional | NOASSERTION |
| packaging | 26.2 | transitive | transitive-optional | NOASSERTION |
| pathspec | 1.1.1 | transitive | transitive-optional | NOASSERTION |
| pillow | 12.2.0 | transitive | transitive-optional | NOASSERTION |
| pluggy | 1.6.0 | transitive | transitive-optional | NOASSERTION |
| pygments | 2.20.0 | transitive | transitive-optional | NOASSERTION |
| pyparsing | 3.3.2 | transitive | transitive-optional | NOASSERTION |
| pytest | 9.0.3 | direct | dev | NOASSERTION |
| pytest-cov | 7.1.0 | direct | dev | NOASSERTION |
| ruff | 0.15.15 | direct | dev | NOASSERTION |
| seaborn | 0.13.2 | direct | viz | NOASSERTION |
| typing-extensions | 4.15.0 | transitive | transitive-optional | NOASSERTION |
| xgboost | 3.2.0 | direct | ml | NOASSERTION |

## Container Components

### Base Images

| Image | Tag | Source | License |
| --- | --- | --- | --- |
| python | 3.11-slim | infra/Dockerfile, infra/Dockerfile.api | NOASSERTION |

### Dockerfile-Only pip Installs

| Package | Version | Source | License |
| --- | --- | --- | --- |
| fastapi | UNPINNED | infra/Dockerfile.api | NOASSERTION |
| joblib | see requirements.lock | infra/Dockerfile, infra/Dockerfile.api | NOASSERTION |
| matplotlib | see requirements.lock | infra/Dockerfile | NOASSERTION |
| python-multipart | UNPINNED | infra/Dockerfile.api | NOASSERTION |
| uvicorn | UNPINNED | infra/Dockerfile.api | NOASSERTION |

## Internal Runtime Artifacts

| Artifact path | Purpose | Supplier | License |
| --- | --- | --- | --- |
| `src/sers/` | Core SERS analysis library | SOLUM Healthcare | Proprietary |
| `scripts/deployment/sers_predict.py` | CLI inference entry point | SOLUM Healthcare | Proprietary |
| `scripts/deployment/sers_webapp.py` | Web/API inference entry point | SOLUM Healthcare | Proprietary |
| `artifacts/usersnet/` | Production model artifacts copied into Docker images | SOLUM Healthcare | Proprietary |
| `artifacts/baselines/` | Baseline artifacts copied into Docker images | SOLUM Healthcare | Proprietary |
| `config/config.yaml` | Runtime pipeline configuration | SOLUM Healthcare | Proprietary |

## Known Gaps And Follow-Up Items

1. `infra/Dockerfile.api` installs `fastapi`, `uvicorn`, and
   `python-multipart` without pinned versions. Pin these in `pyproject.toml`
   or a dedicated lock file before release.
2. Docker base images use `python:3.11-slim` tags without immutable digests.
   Pin image digests for release-grade reproducibility.
3. OS packages inside built images are not enumerated here because no image
   scan was performed. Run an image scanner such as Syft or Trivy against the
   built images for a deployment SBOM.
4. Third-party Python package licenses are recorded as `NOASSERTION` because
   this generator does not query package indexes or installed metadata.
5. Python version declarations are not fully aligned:
   `config/environment.lock.yml` locks `python=3.14.2=h32b2ec7_101_cp314`, Docker images use
   Python 3.11, and `pyproject.toml` declares
   `>=3.10`. Align these before
   regulatory or release use.

## Regeneration

Run:

```bash
python scripts/compliance/generate_sbom.py
```

The command rewrites `docs/SBOM.md` and `docs/sbom.cdx.json`.
