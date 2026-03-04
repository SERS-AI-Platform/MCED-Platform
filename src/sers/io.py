"""File I/O operations for SERS spectra."""

import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List,NamedTuple
from tqdm import tqdm
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class SpectrumID(NamedTuple):
    """Identifier for a spectrum file."""
    group: str
    sample_id: str
    replicate: int


def parse_filename(path: Path, fallback_group: str = "UNK") -> SpectrumID:
    """
    Extract (group, sample_id, replicate) from filename.
    
    Supported patterns:
    - 'GROUP ID_REP.csv' (space separator)
    - 'GROUP_ID_REP.csv' (underscore separator)
    - 'GROUP.SUB ID_REP.csv' (dotted group names)
    
    Parameters
    ----------
    path : Path
        Path to spectrum file
    fallback_group : str
        Group name to use if pattern doesn't match
    
    Returns
    -------
    SpectrumID
        Named tuple with (group, sample_id, replicate)
    
    Raises
    ------
    ValueError
        If filename cannot be parsed
    
    Examples
    --------
    >>> parse_filename(Path("CRC 001_1.csv"))
    SpectrumID(group='CRC', sample_id='001', replicate=1)
    
    >>> parse_filename(Path("H.D._042_3.csv"))
    SpectrumID(group='H.D.', sample_id='042', replicate=3)
    """
    stem = path.stem
    
    # Primary pattern: GROUP (space or _) ID _ REP
    pattern = r"^([A-Za-z]+(?:\.[A-Za-z]+)*\.?)[\s_]?([0-9]+)_([0-9]+)$"
    if m := re.match(pattern, stem):
        group, sid, rep = m.groups()
        return SpectrumID(group.strip().upper(), sid, int(rep))
    
    # Fallback: just numbers, use provided group
    if m := re.search(r"([0-9]+)_([0-9]+)$", stem):
        sid, rep = m.groups()
        return SpectrumID(fallback_group.upper(), sid, int(rep))
    
    raise ValueError(f"Cannot parse filename: {path.name}")


def read_spectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read 2-column spectrum CSV (raman_shift, intensity).
    
    Parameters
    ----------
    path : Path
        Path to CSV file with columns [raman_shift, intensity]
    
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (raman_shift, intensity) arrays sorted by raman_shift
    
    Raises
    ------
    ValueError
        If file has fewer than 2 columns or no valid data
    """
    df = pd.read_csv(path, sep=None, engine="python", header=None, usecols=[0, 1])
    df.columns = ["raman_shift", "intensity"]
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    
    if df.empty:
        raise ValueError(f"No valid numeric data in {path}")

    df = df.sort_values("raman_shift")
    return df["raman_shift"].to_numpy(), df["intensity"].to_numpy()


def find_spectra(
    directory: Path, 
    pattern: str = "*.csv",
    recursive: bool = True
) -> list[Path]:
    """
    Find all spectrum files in directory.
    
    Parameters
    ----------
    directory : Path
        Directory to search
    pattern : str
        Glob pattern for files
    recursive : bool
        If True, search subdirectories
    
    Returns
    -------
    list[Path]
        Sorted list of matching file paths
    """
    glob_method = directory.rglob if recursive else directory.glob
    return sorted(glob_method(pattern))


def make_common_grid(
    x_arrays: list[np.ndarray],
    n_points: int | None = None,
    x_min: float | None = None,
    x_max: float | None = None,
) -> np.ndarray:
    """
    Create common raman_shift grid from multiple spectra.
    
    Parameters
    ----------
    x_arrays : list[np.ndarray]
        List of raman_shift arrays
    n_points : int, optional
        Number of points in grid. Defaults to median length of inputs.
    x_min, x_max : float, optional
        Grid bounds. Defaults to intersection of all spectra.
    
    Returns
    -------
    np.ndarray
        Common raman_shift grid
    """
    if x_min is None:
        x_min = max(x.min() for x in x_arrays)
    if x_max is None:
        x_max = min(x.max() for x in x_arrays)
    if n_points is None:
        n_points = int(np.median([len(x) for x in x_arrays]))
    
    return np.linspace(x_min, x_max, n_points)



@dataclass
class DatasetResult:
    """Container for loaded dataset."""
    spectra: Dict[tuple, Tuple[np.ndarray, np.ndarray]]
    metadata: pd.DataFrame
    failed_files: List[Path]

    def __len__(self):
        return len(self.spectra)

    @property
    def has_equipment(self) -> bool:
        """Whether this dataset has equipment-level grouping."""
        return "equipment" in self.metadata.columns

    def groups(self) -> List[str]:
        """Unique sample groups in dataset."""
        if self.metadata.empty:
            return []
        return sorted(self.metadata["group"].unique())

    def equipments(self) -> List[str]:
        """Unique equipment names (equipment mode only)."""
        if not self.has_equipment:
            return []
        return sorted(self.metadata["equipment"].unique())

    def get_group(self, group: str) -> Dict:
        """Filter spectra by sample group."""
        rows = self.metadata[self.metadata["group"] == group]
        if self.has_equipment:
            keys = set(zip(rows["equipment"], rows["group"], rows["sample_id"], rows["replicate"]))
        else:
            keys = set(zip(rows["group"], rows["sample_id"], rows["replicate"]))
        return {k: v for k, v in self.spectra.items() if k in keys}

    def get_equipment(self, equipment: str) -> Dict:
        """Filter spectra by equipment name (equipment mode only)."""
        rows = self.metadata[self.metadata["equipment"] == equipment]
        keys = set(zip(rows["equipment"], rows["group"], rows["sample_id"], rows["replicate"]))
        return {k: v for k, v in self.spectra.items() if k in keys}

    def summary(self) -> pd.DataFrame:
        """Group-level summary statistics."""
        if self.metadata.empty:
            return pd.DataFrame()
        group_cols = ["equipment", "group"] if self.has_equipment else ["group"]
        return self.metadata.groupby(group_cols).agg(
            n_samples=("sample_id", "nunique"),
            n_spectra=("sample_id", "count"),
            x_min=("x_min", "min"),
            x_max=("x_max", "max"),
        ).reset_index()


def _collect_flat(
    data_dir: Path,
    folder_to_group: Dict[str, str],
    groups: Optional[List[str]],
    pattern: str,
) -> List[Tuple[Path, str, Optional[str]]]:
    """Collect files from flat structure: data_dir/group_folder/files.

    Returns list of (file_path, fallback_group, equipment_or_None).
    """
    folders = [d for d in data_dir.iterdir() if d.is_dir()]

    if groups is not None:
        groups_upper = [g.upper() for g in groups]
        folders = [
            d for d in folders
            if folder_to_group.get(d.name, "").upper() in groups_upper
        ]

    all_files = []
    for folder in folders:
        group = folder_to_group.get(folder.name, "UNK")
        files = find_spectra(folder, pattern=pattern, recursive=False)
        all_files.extend([(f, group, None) for f in files])
    return all_files


def _collect_equipment(
    data_dir: Path,
    equipment_mapping: Dict,
    groups: Optional[List[str]],
    pattern: str,
) -> List[Tuple[Path, str, Optional[str]]]:
    """Collect files from two-level structure: data_dir/equipment/sample_folder/files.

    Returns list of (file_path, fallback_group, equipment_name).
    """
    all_files = []
    groups_upper = [g.upper() for g in groups] if groups else None

    for eq_folder_name, entry in equipment_mapping.items():
        eq_dir = data_dir / eq_folder_name
        if not eq_dir.is_dir():
            logger.warning(f"Equipment folder not found: {eq_dir}")
            continue
        equipment_name = entry.equipment
        sample_mapping = entry.sample_folder_to_group

        if sample_mapping:
            # Two-level: equipment/sample_folder/files
            for sample_folder_name, sample_group in sample_mapping.items():
                if groups_upper and sample_group.upper() not in groups_upper:
                    continue
                sample_dir = eq_dir / sample_folder_name
                if not sample_dir.is_dir():
                    logger.warning(f"Sample folder not found: {sample_dir}")
                    continue
                files = find_spectra(sample_dir, pattern=pattern, recursive=False)
                all_files.extend([(f, sample_group, equipment_name) for f in files])
        else:
            # Flat: equipment/files (group inferred from filename)
            files = find_spectra(eq_dir, pattern=pattern, recursive=False)
            all_files.extend([(f, "UNK", equipment_name) for f in files])

    return all_files


def load_dataset(
    data_dir: Path,
    folder_to_group: Optional[Dict[str, str]] = None,
    equipment_mapping: Optional[Dict] = None,
    groups: Optional[List[str]] = None,
    pattern: str = "*.csv",
    show_progress: bool = True,
) -> DatasetResult:
    """
    Load all SERS spectra from directory structure.

    Supports two modes:
    - Flat mode (folder_to_group): data_dir/group_folder/files
    - Equipment mode (equipment_mapping): data_dir/equipment/sample_folder/files

    Exactly one of folder_to_group or equipment_mapping must be provided.

    Parameters
    ----------
    data_dir : Path
        Root directory containing subfolders
    folder_to_group : Dict[str, str], optional
        Flat mapping from folder names to group codes
    equipment_mapping : Dict[str, EquipmentEntry], optional
        Two-level mapping: equipment folder → EquipmentEntry with sample subfolder mapping
    groups : List[str], optional
        If provided, only load these groups. None = load all.
    pattern : str
        Glob pattern for spectrum files
    show_progress : bool
        Show tqdm progress bar

    Returns
    -------
    DatasetResult
        Container with spectra dict, metadata DataFrame, and failed files.
        When equipment_mapping is used, metadata includes an 'equipment' column.

    Examples
    --------
    >>> from sers.config import RAW_DATA_DIR, SERS_EQUIPMENT_TEST_DATA_DIR, load_config
    >>> config = load_config()
    >>> # Flat mode (raw data)
    >>> result = load_dataset(RAW_DATA_DIR, folder_to_group=config.folder_to_group)
    >>>
    >>> # Equipment mode
    >>> result = load_dataset(SERS_EQUIPMENT_TEST_DATA_DIR, equipment_mapping=config.equipment_folder_to_group)
    """
    if (folder_to_group is None) == (equipment_mapping is None):
        raise ValueError("Provide exactly one of folder_to_group or equipment_mapping")

    if folder_to_group is not None:
        all_files = _collect_flat(data_dir, folder_to_group, groups, pattern)
    else:
        all_files = _collect_equipment(data_dir, equipment_mapping, groups, pattern)

    logger.info(f"Loading {len(all_files)} spectra from {data_dir}")

    spectra = {}
    metadata = []
    failed_files = []

    iterator = tqdm(all_files, desc="Loading spectra", disable=not show_progress)

    for file_path, fallback_group, equipment in iterator:
        try:
            spec_id = parse_filename(file_path, fallback_group=fallback_group)
            x, y = read_spectrum(file_path)

            key = (spec_id.group, spec_id.sample_id, spec_id.replicate)
            if equipment is not None:
                key = (equipment, spec_id.group, spec_id.sample_id, spec_id.replicate)
            spectra[key] = (x, y)

            row = {
                "file": file_path.name,
                "folder": file_path.parent.name,
                "group": spec_id.group,
                "sample_id": spec_id.sample_id,
                "replicate": spec_id.replicate,
                "n_points": len(x),
                "x_min": x.min(),
                "x_max": x.max(),
                "y_min": y.min(),
                "y_max": y.max(),
            }
            if equipment is not None:
                row["equipment"] = equipment
            metadata.append(row)

        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            failed_files.append(file_path)
            continue

    meta_df = pd.DataFrame(metadata)

    logger.info(f"Loaded {len(spectra)} spectra, {len(failed_files)} failed")

    return DatasetResult(
        spectra=spectra,
        metadata=meta_df,
        failed_files=failed_files,
    )
