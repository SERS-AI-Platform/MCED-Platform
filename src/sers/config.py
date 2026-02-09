"""Configuration management for SERS analysis pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import logging
import yaml

logger = logging.getLogger(__name__)



def _find_dotenv() -> Optional[Path]:
    """Find .env file by searching upward from current directory"""
    current = Path(__file__).resolve().parent
    for directory in [current, *current.parents]:
        candidate = directory / "*.env"
        if candidate.is_file():
            return candidate
    return None
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             
def _load_dotenv() -> None:
    dotenv_path = _find_dotenv()
    if dotenv_path is None:
        return
    logger.debug("Loading .env from {dot_env}")
    with open(dotenv_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\'")
            if key not in os.environ:
                os.environ[key] = value
                logger.debug(f" Set {key} from .env")

_load_dotenv()
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/sers/config.py → repo root

# Data paths
DATA_ROOT: Path = Path(os.environ.get("SERS_DATA_ROOT", _PROJECT_ROOT / "data"))
# SERS data
RAW_DATA_DIR: Path = Path(os.environ.get("SERS_RAW_DATA_DIR", DATA_ROOT / "raw_data"))
# 
CLINICAL_DATA_DIR: Path = Path(os.environ.get("SERS_CLINICAL_DATA_DIR", DATA_ROOT / "clinical_data"))
PROCESSED_DIR: Path = Path(os.environ.get("SERS_PROCESSED_DIR", DATA_ROOT / "processed"))

# Output paths
RESULTS_DIR: Path = Path(os.environ.get("SERS_RESULTS_DIR", _PROJECT_ROOT / "results"))
FIG_DIR: Path = Path(os.environ.get("SERS_FIGURES_DIR", _PROJECT_ROOT / "figures"))
MODEL_DIR: Path = Path(os.environ.get("SERS_MODEL_DIR", _PROJECT_ROOT / "models"))
LOG_DIR: Path = Path(os.environ.get("SERS_LOG_DIR", _PROJECT_ROOT / "logs"))

# Config
CONFIG_DIR: Path = Path(os.environ.get("CONFIG_DIR", _PROJECT_ROOT))

# MLflow
MLFLOW_TRACKING_URI: str = os.environ.get("MLFLOW_TRACKING_URI", str(_PROJECT_ROOT / "mlruns"))
MLFLOW_EXPERIMENT_NAME: str = os.environ.get("MLFLOW_EXPERIMENT_NAME", "sers-cancer-detection")

# Environment
SERS_ENV: str = os.environ.get("SERS_ENV", "development")
LOG_LEVEL: str = os.environ.get("SERS_LOG_LEVEL", "INFO")

@dataclass(frozen=True)
class PreprocessingConfig:
    """Preprocessing parameters (user-adjustable)."""
    do_smooth: bool = True
    smooth_window: int = 11
    smooth_poly: int = 3
    baseline_window: int = 101
    use_snv: bool = True


@dataclass(frozen=True)
class QCConfig:
    """Quality control thresholds (typically fixed)."""
    corr_threshold: float = 0.95
    min_snr: float = 3.0
    expected_reps: int = 5


@dataclass(frozen=True)
class ModelingConfig:
    """Model training parameters."""
    n_splits: int = 5
    use_pca: bool = False
    pca_components: int = 10
    random_state: int = 42


@dataclass(frozen=True)
class Config:
    """Main configuration container."""
    
    folder_to_group: Dict[str, str] = field(default_factory=dict)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    qc: QCConfig = field(default_factory=QCConfig)
    modeling : ModelingConfig = field(default_factory=ModelingConfig)
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary."""
        preprocessing_dict = config_dict.get('preprocessing', {})
        qc_dict = config_dict.get('qc', {})
        modeling_dict = config_dict.get('modeling', {}) 
        folder_mapping = config_dict.get('dataset', {}).get('folder_to_group', {})
        
        return cls(
            folder_to_group=folder_mapping,
            preprocessing=PreprocessingConfig(**preprocessing_dict),
            qc=QCConfig(**qc_dict),
            modeling=ModelingConfig(**modeling_dict)
        )        

def load_config(path: str | Path | None = None) -> Config:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    path : str or Path, optional
        Path to config.yaml. Defaults to CONFIG_DIR/config.yaml
    
    Returns
    -------
    Config
        Frozen configuration object
    
    Raises
    ------
    FileNotFoundError
        If config file doesn't exist
    """
    if path is None:
        path = CONFIG_DIR / "config.yaml"
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    
    return Config(
        folder_to_group=raw.get("dataset", {}).get("folder_to_group", {}),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        qc=QCConfig(**raw.get("qc", {})),
        modeling=ModelingConfig(**raw.get("modeling", {})),
    )

# Convenience: load default config on import if file exists
_default_config_path = CONFIG_DIR / "config.yaml"
cfg: Config | None = load_config(_default_config_path) if _default_config_path.exists() else None
