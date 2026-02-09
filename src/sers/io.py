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
    spectra: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]]
    metadata: pd.DataFrame
    failed_files: List[Path]
    
    def __len__(self):
        return len(self.spectra)
    
    def groups(self) -> List[str]:
        """Unique groups in dataset."""
        return sorted(set(g for g, _, _ in self.spectra.keys()))
    
    def get_group(self, group: str) -> Dict:
        """Filter spectra by group."""
        return {k: v for k, v in self.spectra.items() if k[0] == group}
    
    def summary(self) -> pd.DataFrame:
        """Group-level summary statistics."""
        return self.metadata.groupby("group").agg(
            n_samples=("sample_id", "nunique"),
            n_spectra=("sample_id", "count"),
            x_min=("x_min", "min"),
            x_max=("x_max", "max"),
        ).reset_index()


def load_dataset(
    data_dir: Path,
    folder_to_group: Dict[str, str],
    groups: Optional[List[str]] = None,
    pattern: str = "*.CSV",
    show_progress: bool = True,
) -> DatasetResult:
    """
    Load all SERS spectra from directory structure.
    
    Parameters
    ----------
    data_dir : Path
        Root directory containing group subfolders
    folder_to_group : Dict[str, str]
        Mapping from folder names to group codes (from config)
    groups : List[str], optional
        If provided, only load these groups. None = load all.
    pattern : str
        Glob pattern for spectrum files
    show_progress : bool
        Show tqdm progress bar
    
    Returns
    -------
    DatasetResult
        Container with spectra dict, metadata DataFrame, and failed files
    
    Examples
    --------
    >>> from sers.config import RAW_DATA_DIR, load_config
    >>> config = load_config()
    >>> result = load_dataset(RAW_DATA_DIR, config.folder_to_group)
    >>> print(result.summary())
    >>> 
    >>> # Load only normal samples
    >>> normal = load_dataset(RAW_DATA_DIR, config.folder_to_group, groups=["NOR"])
    """
    spectra = {}
    metadata = []
    failed_files = []
    
    # Find all folders to process
    folders = [d for d in data_dir.iterdir() if d.is_dir()]
    
    # Filter folders by group if specified
    if groups is not None:
        groups_upper = [g.upper() for g in groups]
        folders = [
            d for d in folders 
            if folder_to_group.get(d.name, "").upper() in groups_upper
        ]
    
    # Collect all files first (for progress bar accuracy)
    all_files = []
    for folder in folders:
        group = folder_to_group.get(folder.name, "UNK")
        files = find_spectra(folder, pattern=pattern, recursive=False)
        all_files.extend([(f, group) for f in files])
    
    logger.info(f"Loading {len(all_files)} spectra from {len(folders)} folders")
    
    # Load spectra
    iterator = tqdm(all_files, desc="Loading spectra", disable=not show_progress)
    
    for file_path, fallback_group in iterator:
        try:
            # Parse filename
            spec_id = parse_filename(file_path, fallback_group=fallback_group)
            
            # Read spectrum
            x, y = read_spectrum(file_path)
            
            # Store spectrum
            key = (spec_id.group, spec_id.sample_id, spec_id.replicate)
            spectra[key] = (x, y)
            
            # Collect metadata
            metadata.append({
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
            })
            
        except Exception as e:
            logger.warning(f"Failed to load {file_path.name}: {e}")
            failed_files.append(file_path)
            continue
    
    # Create metadata DataFrame
    meta_df = pd.DataFrame(metadata)
    
    logger.info(f"Loaded {len(spectra)} spectra, {len(failed_files)} failed")
    
    return DatasetResult(
        spectra=spectra,
        metadata=meta_df,
        failed_files=failed_files,
    )
