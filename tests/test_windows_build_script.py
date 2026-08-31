from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "deployment" / "build_windows.bat"
PYINSTALLER_SPEC = PROJECT_ROOT / "scripts" / "deployment" / "sers_clinical.spec"
LAUNCHER = PROJECT_ROOT / "scripts" / "deployment" / "launcher.py"


def test_build_script_anchors_project_root_from_script_location() -> None:
    # Given: the Windows build script may be launched from any working directory.
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    # When: its startup commands are inspected.
    anchor_index = script.find('cd /d "%~dp0\\..\\.."')
    dependency_index = script.find("\npip install ")

    # Then: it anchors the working directory before resolving project paths.
    assert 0 <= anchor_index < dependency_index


def test_launcher_selects_actual_port_before_printing_banner() -> None:
    # Given: another app instance may already occupy the default port.
    source = LAUNCHER.read_text(encoding="utf-8")
    main_source = source[source.index("def main():") :]

    # When: launcher startup ordering is inspected.
    select_index = main_source.index("PORT = find_free_port(8080)")
    banner_index = main_source.index("print_banner()")

    # Then: the displayed URL is based on the port the new process will use.
    assert select_index < banner_index


def test_build_script_accepts_any_supported_model() -> None:
    # Given: uSERS-Net and LR fallback are independently valid production models.
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    # When: the model preflight contract is inspected.
    supported_checks = (
        'if exist "artifacts\\usersnet\\current\\manifest.json"',
        'if exist "artifacts\\usersnet\\v1.0.0\\manifest.json"',
        'if exist "artifacts\\baselines\\lr-fusion\\v1.0.0\\manifest.json"',
    )

    # Then: each supported location contributes to one any-model gate.
    assert all(check in script for check in supported_checks)
    assert "set MODEL_FOUND=0" in script
    assert "if %MODEL_FOUND% == 0" in script


def test_build_script_installs_runtime_yaml_dependency() -> None:
    # Given: application startup imports `yaml` through the SERS configuration module.
    script = BUILD_SCRIPT.read_text(encoding="utf-8")

    # When: Windows build dependencies are installed.
    dependency_line = next(line for line in script.splitlines() if line.startswith("pip install "))

    # Then: PyYAML is present before PyInstaller analyzes the application.
    assert "pyyaml" in dependency_line.casefold().split()


def test_pyinstaller_spec_resolves_versioned_usersnet_without_symlink() -> None:
    # Given: Windows Git checkouts may materialize `current` as a text pointer.
    spec = PYINSTALLER_SPEC.read_text(encoding="utf-8")

    # When: PyInstaller resolves the uSERS-Net source directory.
    current_index = spec.find("'current'")
    versioned_index = spec.find("'v1.0.0'")

    # Then: both the preferred pointer and versioned fallback are supported.
    assert "PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]" in spec
    assert current_index >= 0
    assert versioned_index > current_index
    assert "usersnet_source" in spec
    assert "pathex=[str(PROJECT_ROOT), str(PROJECT_ROOT / 'src')]" in spec


def test_pyinstaller_spec_collects_only_available_models() -> None:
    # Given: a valid deployment may contain either supported production model.
    spec = PYINSTALLER_SPEC.read_text(encoding="utf-8")

    # When: model data sources are assembled for PyInstaller.
    usersnet_gate = "if (usersnet_source / 'manifest.json').exists():"
    fallback_gate = "if (fallback_source / 'manifest.json').exists():"

    # Then: each model is included conditionally through the shared data list.
    assert usersnet_gate in spec
    assert fallback_gate in spec
    assert "model_datas" in spec


def test_pyinstaller_spec_packages_manual_but_not_mutable_database() -> None:
    spec = PYINSTALLER_SPEC.read_text(encoding="utf-8")

    assert "scripts' / 'deployment' / 'manuals'" in spec
    assert "clinical_data.db" not in spec


def test_pyinstaller_spec_includes_new_clinical_modules() -> None:
    spec = PYINSTALLER_SPEC.read_text(encoding="utf-8")

    required_modules = (
        "clinical_access",
        "clinical_db_core",
        "clinical_db_results",
        "clinical_db_schema",
        "clinical_db_sessions",
        "clinical_result_data",
        "clinical_validation",
    )
    assert all(f"scripts.deployment.{module}" in spec for module in required_modules)


def test_pyinstaller_spec_collects_xgboost_runtime_library() -> None:
    # Given: XGBoost loads its native DLL while restoring the packaged model.
    spec = PYINSTALLER_SPEC.read_text(encoding="utf-8")

    # When: PyInstaller assembles native binaries.
    # Then: the XGBoost dynamic library is explicitly passed to Analysis.
    assert "collect_data_files" in spec
    assert "collect_dynamic_libs" in spec
    assert "collect_data_files('xgboost')" in spec
    assert "collect_dynamic_libs('xgboost')" in spec
    assert "binaries=xgboost_binaries" in spec
    assert "*xgboost_datas" in spec


def test_pyinstaller_spec_excludes_training_and_source_trees() -> None:
    # Given: the clinical distribution must contain only runtime application assets.
    spec = PYINSTALLER_SPEC.read_text(encoding="utf-8")

    # When: bundled data paths are inspected.
    # Then: research and training source trees are not copied into the distribution.
    assert "train_usersnet.py" not in spec
    assert "PROJECT_ROOT / 'src' / 'sers'" not in spec
    assert "scripts/training" not in spec
