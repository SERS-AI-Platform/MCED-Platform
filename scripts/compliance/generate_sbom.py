#!/usr/bin/env python3
"""Generate SBOM artifacts for the SERS analysis project.

The generator intentionally uses only repository-local dependency manifests and
the Python standard library. Package licenses are therefore recorded as
NOASSERTION unless they are declared by this project itself.
"""

from __future__ import annotations

import json
import re
import tomllib
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
REQUIREMENTS_LOCK = ROOT / "requirements.lock"
DOCKERFILES = (ROOT / "infra" / "Dockerfile", ROOT / "infra" / "Dockerfile.api")
ENVIRONMENT = ROOT / "config" / "environment.yml"
ENVIRONMENT_LOCK = ROOT / "config" / "environment.lock.yml"
SBOM_MD = ROOT / "docs" / "compliance" / "SBOM.md"
SBOM_CDX = ROOT / "docs" / "compliance" / "sbom.cdx.json"

REQ_NAME_RE = re.compile(r"^\s*([A-Za-z0-9_.-]+)")
LOCK_PACKAGE_RE = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s]+)$")
FROM_RE = re.compile(r"^\s*FROM\s+([^\s]+)")
PIP_INSTALL_RE = re.compile(r"pip install .*?((?:[A-Za-z0-9_.-]+(?:\[[^\]]+\])?\s*)+)$")


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def requirement_name(requirement: str) -> str | None:
    match = REQ_NAME_RE.match(requirement)
    if not match:
        return None
    return normalize_name(match.group(1))


def load_project() -> dict[str, Any]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["project"]


def direct_dependency_sets(project: dict[str, Any]) -> tuple[set[str], dict[str, set[str]]]:
    required = {
        name
        for dep in project.get("dependencies", [])
        if (name := requirement_name(dep)) is not None
    }
    optional: dict[str, set[str]] = {}
    for extra, deps in project.get("optional-dependencies", {}).items():
        names = {
            name
            for dep in deps
            if (name := requirement_name(dep)) is not None
            and name != normalize_name(project["name"])
        }
        if names:
            optional[extra] = names
    return required, optional


def parse_requirements_lock() -> list[dict[str, Any]]:
    packages: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    for raw_line in REQUIREMENTS_LOCK.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        package_match = LOCK_PACKAGE_RE.match(line)
        if package_match:
            current = {
                "name": normalize_name(package_match.group(1)),
                "display_name": package_match.group(1),
                "version": package_match.group(2),
                "via": set(),
            }
            packages.append(current)
            continue

        if current is None or "#" not in line:
            continue

        comment = line.split("#", 1)[1].strip()
        if comment == "via":
            continue
        if comment.startswith("via "):
            comment = comment[4:].strip()
        if not comment:
            continue

        provider = comment.split("(", 1)[0].strip()
        if provider:
            current["via"].add(normalize_name(provider))

    for package in packages:
        package["via"] = sorted(package["via"])
    return packages


def classify_packages(
    packages: list[dict[str, Any]],
    required_direct: set[str],
    optional_direct: dict[str, set[str]],
    project_name: str,
) -> None:
    optional_names = set().union(*optional_direct.values()) if optional_direct else set()
    required = set(required_direct)
    optional = set(optional_names)

    via_map = {package["name"]: set(package["via"]) for package in packages}

    changed = True
    while changed:
        changed = False
        for name, providers in via_map.items():
            if name in required:
                continue
            if project_name in providers and name in required_direct:
                required.add(name)
                changed = True
            elif providers & required:
                required.add(name)
                changed = True

    changed = True
    while changed:
        changed = False
        for name, providers in via_map.items():
            if name in required or name in optional:
                continue
            if project_name in providers and name in optional_names:
                optional.add(name)
                changed = True
            elif providers & optional:
                optional.add(name)
                changed = True

    extra_by_name: dict[str, list[str]] = defaultdict(list)
    for extra, names in optional_direct.items():
        for name in names:
            extra_by_name[name].append(extra)

    for package in packages:
        name = package["name"]
        direct = name in required_direct or name in optional_names
        package["direct"] = direct
        package["scope"] = "required" if name in required else "optional"
        if direct and name in extra_by_name:
            package["extra"] = ",".join(sorted(extra_by_name[name]))
        elif direct:
            package["extra"] = "runtime"
        else:
            package["extra"] = "transitive-runtime" if package["scope"] == "required" else "transitive-optional"


def parse_dockerfiles() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    base_images: dict[str, dict[str, Any]] = {}
    docker_pip: dict[str, dict[str, Any]] = {}

    for dockerfile in DOCKERFILES:
        if not dockerfile.exists():
            continue
        lines = dockerfile.read_text(encoding="utf-8").splitlines()
        for line in lines:
            from_match = FROM_RE.match(line)
            if from_match:
                image = from_match.group(1)
                name, _, version = image.partition(":")
                base_images.setdefault(
                    image,
                    {
                        "name": name,
                        "version": version or "latest",
                        "sources": set(),
                    },
                )["sources"].add(dockerfile.relative_to(ROOT).as_posix())

            if "pip install" not in line:
                continue
            install_match = PIP_INSTALL_RE.search(line)
            if not install_match:
                continue
            for token in install_match.group(1).split():
                if token.startswith("-") or token in {".", "\".\"", "'."}:
                    continue
                package = normalize_name(token.split("[", 1)[0])
                if package in {"pip", "setuptools", "wheel"}:
                    continue
                docker_pip.setdefault(package, {"name": package, "sources": set()})["sources"].add(
                    dockerfile.relative_to(ROOT).as_posix()
                )

    for item in base_images.values():
        item["source"] = ", ".join(sorted(item.pop("sources")))
    for item in docker_pip.values():
        item["source"] = ", ".join(sorted(item.pop("sources")))

    return (
        sorted(base_images.values(), key=lambda item: (item["name"], item["version"])),
        sorted(docker_pip.values(), key=lambda item: item["name"]),
    )


def environment_python_version(path: Path) -> str:
    if not path.exists():
        return "UNKNOWN"
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("- python"):
            return line[2:].strip()
    return "UNKNOWN"


def component_ref(name: str, version: str | None = None, namespace: str = "pypi") -> str:
    if namespace == "project":
        return f"project:{name}@{version}"
    if version:
        return f"pkg:{namespace}/{name}@{version}"
    return f"pkg:{namespace}/{name}"


def make_cyclonedx(
    project: dict[str, Any],
    packages: list[dict[str, Any]],
    base_images: list[dict[str, str]],
    docker_pip: list[dict[str, str]],
    timestamp: str,
) -> dict[str, Any]:
    project_name = normalize_name(project["name"])
    project_ref = component_ref(project_name, project["version"], "project")
    package_names = {package["name"] for package in packages}

    components: list[dict[str, Any]] = []
    for package in packages:
        purl = component_ref(package["name"], package["version"])
        components.append(
            {
                "type": "library",
                "bom-ref": purl,
                "name": package["display_name"],
                "version": package["version"],
                "purl": purl,
                "scope": package["scope"],
                "licenses": [{"license": {"name": "NOASSERTION"}}],
                "properties": [
                    {"name": "sers:dependency-kind", "value": "direct" if package["direct"] else "transitive"},
                    {"name": "sers:dependency-scope", "value": package["extra"]},
                    {"name": "sers:source", "value": "requirements.lock"},
                ],
            }
        )

    for image in base_images:
        image_ref = component_ref(image["name"], image["version"], "docker")
        components.append(
            {
                "type": "container",
                "bom-ref": image_ref,
                "name": image["name"],
                "version": image["version"],
                "purl": image_ref,
                "licenses": [{"license": {"name": "NOASSERTION"}}],
                "properties": [
                    {"name": "sers:source", "value": image["source"]},
                    {"name": "sers:version-status", "value": "tag-only-digest-not-pinned"},
                ],
            }
        )

    for package in docker_pip:
        if package["name"] in package_names:
            continue
        purl = component_ref(package["name"])
        components.append(
            {
                "type": "library",
                "bom-ref": purl,
                "name": package["name"],
                "purl": purl,
                "scope": "required",
                "licenses": [{"license": {"name": "NOASSERTION"}}],
                "properties": [
                    {"name": "sers:source", "value": package["source"]},
                    {"name": "sers:version-status", "value": "unpinned"},
                ],
            }
        )

    dependencies: dict[str, set[str]] = defaultdict(set)
    for package in packages:
        package_ref = component_ref(package["name"], package["version"])
        if project_name in package["via"] or package["direct"]:
            dependencies[project_ref].add(package_ref)
        for provider in package["via"]:
            provider_pkg = next((item for item in packages if item["name"] == provider), None)
            if provider_pkg is not None:
                dependencies[component_ref(provider, provider_pkg["version"])].add(package_ref)

    for image in base_images:
        dependencies[project_ref].add(component_ref(image["name"], image["version"], "docker"))
    for package in docker_pip:
        if package["name"] not in package_names:
            dependencies[project_ref].add(component_ref(package["name"]))

    dependency_entries = [
        {"ref": ref, "dependsOn": sorted(depends_on)}
        for ref, depends_on in sorted(dependencies.items())
    ]

    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "serialNumber": f"urn:uuid:{uuid4()}",
        "version": 1,
        "metadata": {
            "timestamp": timestamp,
            "tools": [
                {
                    "vendor": "SOLUM Healthcare",
                    "name": "scripts/compliance/generate_sbom.py",
                    "version": "1",
                }
            ],
            "component": {
                "type": "application",
                "bom-ref": project_ref,
                "name": project["name"],
                "version": project["version"],
                "description": project.get("description", ""),
                "supplier": {"name": "SOLUM Healthcare"},
                "licenses": [{"license": {"name": "Proprietary"}}],
                "externalReferences": [
                    {"type": "vcs", "url": project.get("urls", {}).get("Repository", "")},
                    {"type": "website", "url": project.get("urls", {}).get("Homepage", "")},
                ],
                "properties": [
                    {"name": "sers:readme", "value": project.get("readme", "")},
                    {"name": "sers:requires-python", "value": project.get("requires-python", "")},
                ],
            },
        },
        "components": components,
        "dependencies": dependency_entries,
    }


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def make_markdown(
    project: dict[str, Any],
    packages: list[dict[str, Any]],
    base_images: list[dict[str, str]],
    docker_pip: list[dict[str, str]],
    timestamp: str,
) -> str:
    env_python = environment_python_version(ENVIRONMENT)
    env_lock_python = environment_python_version(ENVIRONMENT_LOCK)
    required_rows = [
        [pkg["display_name"], pkg["version"], "direct" if pkg["direct"] else "transitive", pkg["extra"], "NOASSERTION"]
        for pkg in packages
        if pkg["scope"] == "required"
    ]
    optional_rows = [
        [pkg["display_name"], pkg["version"], "direct" if pkg["direct"] else "transitive", pkg["extra"], "NOASSERTION"]
        for pkg in packages
        if pkg["scope"] == "optional"
    ]
    base_rows = [[item["name"], item["version"], item["source"], "NOASSERTION"] for item in base_images]
    docker_rows = [
        [
            item["name"],
            "see requirements.lock" if item["name"] in {pkg["name"] for pkg in packages} else "UNPINNED",
            item["source"],
            "NOASSERTION",
        ]
        for item in docker_pip
    ]

    return f"""# Software Bill of Materials

## Document Metadata

| Field | Value |
| --- | --- |
| Product | {project["name"]} |
| Version | {project["version"]} |
| Supplier | SOLUM Healthcare |
| License | Proprietary |
| Generated | {timestamp} |
| SBOM standard | CycloneDX 1.5 JSON plus human-readable Markdown |
| Machine-readable file | `docs/compliance/sbom.cdx.json` |

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

- `config/environment.yml` ({env_python})
- `config/environment.lock.yml` ({env_lock_python})

Excluded from package license resolution:

- Raw or clinical data under `data/`
- Generated results under `results/`
- Runtime OS packages inside built container images
- Package license metadata not present in repository-local manifests

## Primary Component

| Component | Version | Type | Supplier | License |
| --- | --- | --- | --- | --- |
| {project["name"]} | {project["version"]} | application | SOLUM Healthcare | Proprietary |

## Required Python Components

{markdown_table(["Package", "Version", "Kind", "Scope", "License"], required_rows)}

## Optional And Development Python Components

{markdown_table(["Package", "Version", "Kind", "Scope", "License"], optional_rows)}

## Container Components

### Base Images

{markdown_table(["Image", "Tag", "Source", "License"], base_rows)}

### Dockerfile-Only pip Installs

{markdown_table(["Package", "Version", "Source", "License"], docker_rows)}

## Internal Runtime Artifacts

| Artifact path | Purpose | Supplier | License |
| --- | --- | --- | --- |
| `src/sers/` | Core SERS analysis library | SOLUM Healthcare | Proprietary |
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
   `config/environment.lock.yml` locks `{env_lock_python}`, Docker images use
   Python 3.11, and `pyproject.toml` declares
   `{project.get("requires-python", "UNKNOWN")}`. Align these before
   regulatory or release use.

## Regeneration

Run:

```bash
python scripts/compliance/generate_sbom.py
```

The command rewrites `docs/compliance/SBOM.md` and `docs/compliance/sbom.cdx.json`.
"""


def main() -> None:
    project = load_project()
    required_direct, optional_direct = direct_dependency_sets(project)
    project_name = normalize_name(project["name"])
    packages = parse_requirements_lock()
    classify_packages(packages, required_direct, optional_direct, project_name)
    base_images, docker_pip = parse_dockerfiles()

    timestamp = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    SBOM_MD.write_text(
        make_markdown(project, packages, base_images, docker_pip, timestamp),
        encoding="utf-8",
    )
    SBOM_CDX.write_text(
        json.dumps(
            make_cyclonedx(project, packages, base_images, docker_pip, timestamp),
            ensure_ascii=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {SBOM_MD.relative_to(ROOT)}")
    print(f"Wrote {SBOM_CDX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
