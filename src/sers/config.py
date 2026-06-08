"""Configuration management for SERS analysis pipeline.

Hierarchy:
    config.yaml → Config (frozen dataclass) → individual modules

QC thresholds are centralized in QCConfig. The qc module accepts
either a QCConfig object or individual parameters, but QCConfig
is the canonical source of truth.

QC Philosophy:
    SERS urine spectra are metabolite superpositions. Traditional SNR/SBR
    approaches fail. QC is based on:
    - Intensity gate (adaptive): "Did SERS enhancement work?"
    - Replicate reproducibility: RSD + Correlation

Changelog:

    v0.7.0 (2026-02) - Auto-detect WSL, convert Windows paths from .env
    v0.6.0 (2026-02) - Add DisplayConfig (colors, ordering, category map)
    v0.5.0 (2026-02) - Add intensity_gate_ratio to QCConfig
    v0.4.0 (2026-02) - Replicate-only QC: removed SBR/SNR
    v0.3.0 (2026-02) - SBR approach (deprecated)
    v0.1.0 (2025-01) - Initial release
"""

import logging
import os
import platform
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger(__name__)


# =============================================================================
# OS / WSL Detection and Path Conversion
# =============================================================================
def _is_wsl() -> bool:
    """Detect if running inside WSL (Windows Subsystem for Linux).

    Checks:
        1. /proc/version contains 'microsoft' or 'WSL'
        2. platform.uname().release contains 'microsoft'
    """
    try:
        with open("/proc/version", "r") as f:
            content = f.read().lower()
            if "microsoft" in content or "wsl" in content:
                return True
    except (FileNotFoundError, PermissionError):
        pass

    try:
        if "microsoft" in platform.uname().release.lower():
            return True
    except Exception:
        pass

    return False


# Cache the result at module load time (won't change during runtime)
IS_WSL: bool = _is_wsl()


def _win_to_wsl_path(win_path: str) -> str:
    """Convert a Windows path to WSL-compatible path.

    Examples
    --------
    >>> _win_to_wsl_path(r'C:\\Users\\user\\data')
    '/mnt/c/Users/user/data'

    >>> _win_to_wsl_path('D:\\Projects\\SERS')
    '/mnt/d/Projects/SERS'

    >>> _win_to_wsl_path('/mnt/c/already/wsl')  # Already WSL → no change
    '/mnt/c/already/wsl'

    >>> _win_to_wsl_path('relative/path')  # No drive letter → no change
    'relative/path'
    """
    path = win_path.strip()

    # Already a WSL/Unix path
    if path.startswith("/"):
        return path

    # Match Windows drive pattern: C:\ or C:/
    match = re.match(r"^([A-Za-z]):[/\\]", path)
    if match:
        drive = match.group(1).lower()
        rest = path[3:]  # Skip 'C:\'
        rest = rest.replace("\\", "/")
        return f"/mnt/{drive}/{rest}"

    # No drive letter — just fix slashes (relative path)
    return path.replace("\\", "/")


def _normalize_path(raw_path: str) -> str:
    """Normalize a path string based on current OS.

    If running in WSL and the path looks like a Windows path,
    automatically convert it.  Otherwise return as-is.
    """
    if IS_WSL:
        return _win_to_wsl_path(raw_path)
    return raw_path


# =============================================================================
# .env Loading
# =============================================================================
def _find_dotenv() -> Optional[Path]:
    """Find .env file by searching upward from current directory."""
    current = Path(__file__).absolute().parent
    for directory in [current, *current.parents]:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def _load_dotenv() -> None:
    dotenv_path = _find_dotenv()
    if dotenv_path is None:
        return
    logger.debug(f"Loading .env from {dotenv_path}")

    with open(dotenv_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")

            # WSL: skip path variables — they point to OneDrive internal
            # paths (YoonInsu_HealthCare_) that are inaccessible from WSL.
            # Paths are auto-derived from _PROJECT_ROOT instead.
            if IS_WSL and _looks_like_path(key):
                logger.debug(f"  WSL: skipping path var {key}")
                continue

            if key not in os.environ:
                os.environ[key] = value
                logger.debug(f"  Set {key}")


def _looks_like_path(env_key: str) -> bool:
    key_upper = env_key.upper()
    return any(
        token in key_upper
        for token in ("DIR", "ROOT", "PATH", "HOME", "FOLDER", "CONFIG_DIR")
    )


# =============================================================================
# Project Root and Paths
# =============================================================================
_load_dotenv()

# Use absolute() instead of resolve() to avoid OneDrive symlink issues
_PROJECT_ROOT = Path(__file__).absolute().parent.parent.parent  # src/sers/config.py → repo root

# Data paths
DATA_ROOT: Path = Path(_normalize_path(
    os.environ.get("SERS_DATA_ROOT", str(_PROJECT_ROOT / "data"))
))
RAW_DATA_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_RAW_DATA_DIR", str(DATA_ROOT / "raw_data"))
))
RAW_DATA_MEDICAL_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_RAW_DATA_MEDICAL_DIR", str(DATA_ROOT / "raw_data_medical"))
))
SERS_EQUIPMENT_TEST_DATA_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_EQUIPMENT_TEST_DATA_DIR", str(DATA_ROOT / "equipment_test_data"))
))
CLINICAL_DATA_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_CLINICAL_DATA_DIR", str(DATA_ROOT / "clinical_data"))
))
PROCESSED_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_PROCESSED_DIR", str(DATA_ROOT / "processed"))
))

# Output paths
RESULTS_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_RESULTS_DIR", str(_PROJECT_ROOT / "results"))
))
FIG_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_FIGURES_DIR", str(RESULTS_DIR / "figures"))
))
MODEL_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_MODEL_DIR", str(_PROJECT_ROOT / "models"))
))


def get_figure_dir(category: str, experiment_name: str) -> Path:
    """Return centralized figure directory. category: training|analysis|weekend"""
    d = FIG_DIR / category / experiment_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def training_dir_to_figure_slug(train_dir: Path) -> str:
    """Convert a training result path to a compact figure directory name.

    Example: results/training/fixed_grid_7cancer/logistic_regression/v001
             → fixed_grid_7cancer_logistic_regression_v001
    """
    train_dir = Path(train_dir).resolve()
    results_training = (RESULTS_DIR / "training").resolve()
    try:
        rel = train_dir.relative_to(results_training)
    except ValueError:
        # Fallback for paths outside results/training/
        rel = train_dir
    parts = [p for p in rel.parts if p not in (".", "..")]
    return "_".join(parts)
LOG_DIR: Path = Path(_normalize_path(
    os.environ.get("SERS_LOG_DIR", str(_PROJECT_ROOT / "logs"))
))

# Infrastructure
INFRA_DIR: Path = _PROJECT_ROOT / "infra"

# Documentation
DOCS_DIR: Path = _PROJECT_ROOT / "docs"

# Dashboard
DASHBOARD_DIR: Path = _PROJECT_ROOT / "dashboard"

# Config
CONFIG_DIR: Path = Path(_normalize_path(
    os.environ.get("CONFIG_DIR", str(_PROJECT_ROOT / "config"))
))

# MLflow
MLFLOW_TRACKING_URI: str = _normalize_path(
    os.environ.get("MLFLOW_TRACKING_URI", str(_PROJECT_ROOT / "mlruns"))
)
MLFLOW_EXPERIMENT_NAME: str = os.environ.get(
    "MLFLOW_EXPERIMENT_NAME", "sers-cancer-detection"
)

# Environment
SERS_ENV: str = os.environ.get("SERS_ENV", "development")
LOG_LEVEL: str = os.environ.get("SERS_LOG_LEVEL", "INFO")


# =============================================================================
# Configuration Dataclasses
# =============================================================================
@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing parameters (user-adjustable)."""

    do_trim: bool = True
    trim_region: tuple[float, float] = (400.0, 2200.0)
    do_smooth: bool = True
    smoothing_method: str = "savgol"
    smooth_window: int = 11
    smooth_poly: int = 3
    median_window: int = 5
    gaussian_sigma: float = 1.0
    moving_window: int = 5
    wavelet_threshold: float = 1.0
    wavelet_level: Optional[int] = None
    do_baseline: bool = True
    baseline_window: int = 101
    baseline_method: str = "rolling_min"
    baseline_als_lam: float = 1e6
    baseline_als_p: float = 0.01
    baseline_als_niter: int = 10
    baseline_arpls_lam: float = 1e5
    baseline_arpls_ratio: float = 1e-6
    baseline_arpls_niter: int = 50
    baseline_airpls_lam: float = 1e5
    baseline_airpls_niter: int = 15
    baseline_airpls_tol: float = 1e-3
    baseline_poly_order: int = 3
    baseline_poly_quantile: float = 0.2
    baseline_moving_quantile: float = 0.1
    baseline_clip_negative: bool = False
    use_snv: bool = True
    normalization: Optional[str] = None
    normalization_peak_wn: Optional[float] = None
    normalization_peak_window: float = 10.0
    normalization_emsc_order: int = 2
    fixed_grid: Optional[Dict[str, float]] = None
    # Wavenumber calibration (urea reference peak alignment)
    do_calibration: bool = False
    calibration_reference_wn: float = 1001.4
    calibration_window: float = 20.0


@dataclass(frozen=True)
class QCConfig:
    """Quality control thresholds and parameters.

    Single source of truth for all QC settings.

    Attributes
    ----------
    rsd_threshold : float
        Maximum acceptable RSD in % (default: 5.0)
    corr_threshold : float
        Minimum acceptable pairwise correlation (default: 0.95)
    expected_reps : int
        Expected number of replicates per sample (default: 5)
    fingerprint_region : tuple of float
        Region for analysis in cm⁻¹ (default: (400, 2200))
    intensity_gate_ratio : float
        Adaptive intensity gate threshold as fraction of median fp_mean.
        Spectra with fp_mean < median × ratio are flagged as
        SERS enhancement failure. (default: 0.1)

        Why 0.1?
        - Normal equipment shows p1/median ≈ 0.45–0.56
        - Enhancement failure shows p1/median < 0.1
        - ratio=0.1 catches catastrophic failures without
          flagging normal measurement variation
    """

    rsd_threshold: float = 5.0
    corr_threshold: float = 0.95
    expected_reps: int = 5
    fingerprint_region: Tuple[float, float] = (400.0, 2200.0)
    intensity_gate_ratio: float = 0.1

    # v2 two-stage QC (Phase 7, 2026-04-08) — NOR p5 derived
    per_spectrum_corr_threshold: float = 0.925
    min_reps_after_qc: int = 4
    cosmic_isolation: float = 2.0
    cosmic_height: float = 0.3
    saturation_plateau: int = 5

    def __post_init__(self) -> None:
        """Validate QC configuration values."""
        if self.rsd_threshold < 0:
            raise ValueError(f"rsd_threshold must be >= 0, got {self.rsd_threshold}")
        if not (0 < self.corr_threshold <= 1.0):
            raise ValueError(
                f"corr_threshold must be in (0, 1], got {self.corr_threshold}"
            )
        if self.fingerprint_region[0] >= self.fingerprint_region[1]:
            raise ValueError(
                f"fingerprint_region must be (low, high), got {self.fingerprint_region}"
            )
        if not (0 < self.intensity_gate_ratio < 1.0):
            raise ValueError(
                f"intensity_gate_ratio must be in (0, 1), got {self.intensity_gate_ratio}"
            )


@dataclass(frozen=True)
class ModelingConfig:
    """Model training parameters."""

    n_splits: int = 5
    use_pca: bool = False
    pca_components: int = 10
    random_state: int = 42


@dataclass(frozen=True)
class MedicalConfig:
    """Medical Raman instrument specific settings.

    Medical 장비 데이터는 BG 미차감 원본 + Background/ 폴더 구조.
    BG 차감 후 noise가 매우 낮으므로 SG smoothing 불필요.
    장비간 SERS enhancement 패턴이 상이하여 calibration skip.
    """

    file_pattern: str = "*.txt"
    read_from_subdir: str = "Background"
    exclude_patterns: List[str] = field(
        default_factory=lambda: ["*_ave.txt", "MultiData.txt", "*Zone.Identifier*"]
    )
    expected_reps: int = 6
    do_smooth: bool = False
    do_calibration: bool = False
    # Wavenumber shift to align Medical with Thermo reference frame.
    # Physical validation: urea peak (literature: 1001.4 cm⁻¹) appears at
    # ~1027 cm⁻¹ in Medical → +26 cm⁻¹ instrument offset.
    # Empirical 2nd-deriv correlation search confirms +28 cm⁻¹ (median across groups).
    # Applied as: shifted_wn = original_wn + wavenumber_shift
    wavenumber_shift: float = -28.0


@dataclass(frozen=True)
class EquipmentEntry:
    """Single equipment configuration with its sample group mapping."""

    equipment: str = ""
    sample_folder_to_group: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DisplayConfig:
    """Visualization display settings.

    Centralized colors, ordering, and category mappings for consistent
    plotting across the project.
    Loaded from config.yaml under 'display' section + dataset.group_metadata.

    Usage::

        dsp = config.display
        groups = dsp.active_groups(df['group'].unique())
        colors = dsp.palette_for(groups)
        cat = dsp.category_map.get(group, 'Unknown')
    """

    group_order: List[str] = field(default_factory=list)
    category_map: Dict[str, str] = field(default_factory=dict)
    group_colors: Dict[str, str] = field(default_factory=dict)
    category_colors: Dict[str, str] = field(default_factory=dict)
    replicate_colors: Dict[str, str] = field(default_factory=dict)
    group_metadata: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def active_groups(self, available) -> List[str]:
        """Return groups in display order that are available in the data."""
        return [g for g in self.group_order if g in available]

    def palette_for(self, groups: List[str]) -> List[str]:
        """Get color palette for given groups."""
        return [self.group_colors.get(g, "#333333") for g in groups]

    def category_for_group(self, group: str) -> Optional[str]:
        """Get category for a given group."""
        return self.category_map.get(group)

    def category_color(self, category: str) -> str:
        """Get color for a given category."""
        return self.category_colors.get(category, "#333333")


@dataclass(frozen=True)
class Config:
    """Main configuration container."""

    folder_to_group: Dict[str, str] = field(default_factory=dict)
    folder_to_group_medical: Dict[str, str] = field(default_factory=dict)
    equipment_folder_to_group: Dict[str, EquipmentEntry] = field(default_factory=dict)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    medical: MedicalConfig = field(default_factory=MedicalConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    display: DisplayConfig = field(default_factory=DisplayConfig)

    @property
    def project_root(self) -> Path:
        return _PROJECT_ROOT

    @property
    def data_root(self) -> Path:
        return DATA_ROOT

    @property
    def processed_dir(self) -> Path:
        return PROCESSED_DIR

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "Config":
        """Create Config from dictionary."""
        preprocessing_dict = config_dict.get("preprocessing", {})
        qc_dict = _parse_qc_dict(config_dict.get("qc", {}))
        modeling_dict = config_dict.get("modeling", {})
        dataset = config_dict.get("dataset", {})
        folder_mapping = dataset.get("folder_to_group", {})
        folder_mapping_medical = dataset.get("folder_to_group_medical", {})
        equipment_raw = dataset.get("equipment_folder_to_group", {})
        equipment_mapping = {k: EquipmentEntry(**v) for k, v in equipment_raw.items()}

        # Medical config
        medical_raw = dataset.get("medical", {})
        medical_config = MedicalConfig(**{
            k: v for k, v in medical_raw.items()
            if k in MedicalConfig.__dataclass_fields__
        }) if medical_raw else MedicalConfig()

        # Display: merge display config with group metadata
        display_raw = config_dict.get("display", {})
        group_metadata = dataset.get("group_metadata", {})
        display_config = _parse_display_dict(display_raw, group_metadata)

        return cls(
            folder_to_group=folder_mapping,
            folder_to_group_medical=folder_mapping_medical,
            equipment_folder_to_group=equipment_mapping,
            preprocessing=PreprocessingConfig(**preprocessing_dict),
            medical=medical_config,
            qc=QCConfig(**qc_dict),
            modeling=ModelingConfig(**modeling_dict),
            display=display_config,
        )


# =============================================================================
# Config Parsing Helpers
# =============================================================================
def _parse_qc_dict(raw_qc: Dict[str, Any]) -> Dict[str, Any]:
    """Parse QC config dict, handling legacy fields.

    Current fields:
        rsd_threshold, corr_threshold, expected_reps,
        fingerprint_region, intensity_gate_ratio

    Legacy (silently ignored):
        min_snr, snr_threshold, sbr_threshold,
        snr_method, background_region, signal_metric
    """
    result = {}

    deprecated = {
        "min_snr", "snr_threshold", "sbr_threshold",
        "snr_method", "background_region", "signal_metric",
    }

    for key, value in raw_qc.items():
        if key in deprecated:
            logger.debug(f"Ignoring deprecated QC config field: {key}")
            continue

        if key == "fingerprint_region" and isinstance(value, list):
            result[key] = tuple(value)
        else:
            result[key] = value

    return result


def _parse_display_dict(
    raw_display: Dict[str, Any],
    group_metadata: Dict[str, Any],
) -> DisplayConfig:
    """Parse display config, merging with group metadata."""
    kwargs: Dict[str, Any] = {}
    for key in [
        "group_order", "category_map", "group_colors",
        "category_colors", "replicate_colors",
    ]:
        if key in raw_display:
            kwargs[key] = raw_display[key]

    kwargs["group_metadata"] = group_metadata or {}

    return DisplayConfig(**kwargs)


# =============================================================================
# Config Loader
# =============================================================================
def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from YAML file.

    Parameters
    ----------
    path : str or Path, optional
        Path to config.yaml. Defaults to CONFIG_DIR/config.yaml

    Returns
    -------
    Config
        Frozen configuration object
    """
    if path is None:
        path = CONFIG_DIR / "config.yaml"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    return Config.from_dict(raw)


# Convenience: load default config on import if file exists
_default_config_path = CONFIG_DIR / "config.yaml"
cfg: Config | None = (
    load_config(_default_config_path) if _default_config_path.exists() else None
)
