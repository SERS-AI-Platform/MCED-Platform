"""
Clinical data utilities for multimodal fusion models.

Provides:
    - load_clinical(): Load and prepare clinical CSV
    - build_id_mappings(): Build lookup tables for groups with non-standard IDs
    - merge_clinical_features(): Align clinical features with spectral data
    - ClinicalSERSDataset: Dataset subclass with clinical features
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from models.model import SERSDataset

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[4]

TIER1_COLS = ["age", "sex_numeric", "bmi"]


def load_clinical(project_root: Path = PROJECT_ROOT) -> pd.DataFrame:
    """Load standardized clinical CSV and add sex_numeric column."""
    path = project_root / "data" / "clinical_data" / "standardized" / "all_clinical_standardized.csv"
    clin = pd.read_csv(path)
    clin["disease_group"] = clin["disease_group"].replace({"PAN": "CPAN"})
    clin["sex_numeric"] = (clin["sex"] == "M").astype(float)
    logger.info(f"  Loaded clinical data: {len(clin)} patients from {path.name}")
    return clin


def build_id_mappings(project_root: Path = PROJECT_ROOT) -> Dict[str, str]:
    """Build solum_label → clinical patient_id mappings for groups
    where spectral sample IDs don't directly match clinical patient_ids.

    Handles:
        - BRE: hospital numbers → 'BRE N' via Excel SoluM Label
        - OVA: hospital numbers → 'OVA N' via Excel SoluM Label
        - BLC: spectral 'BLC N' → clinical 'stage-N' format (BLA group)
        - H.D.: spectral 'H.D. N' → clinical 'H. D. N' (space difference)
        - PAN: spectral 'PAN N' (aliased from CPAN/YPAN) → clinical 'CPAN N'/'YPAN N'
        - LUN 201-300: spectral has 300 but clinical only has 1-200
    """
    solum_to_clinical: Dict[str, str] = {}

    # --- BRE: Extract from Excel ---
    bre_path = project_root / "data" / "clinical_data" / "2. 유방암" / "SMCXD01_유방암.xlsx"
    if bre_path.exists():
        try:
            bre_xl = pd.read_excel(bre_path, header=None, skiprows=2)
            bre_map = bre_xl[[1, 4]].dropna()
            bre_map.columns = ["hospital_id", "solum_label"]
            bre_map["hospital_id"] = bre_map["hospital_id"].astype(int).astype(str)
            for _, row in bre_map.iterrows():
                solum_to_clinical[row["solum_label"]] = row["hospital_id"]
            logger.info(f"  BRE mapping: {len(bre_map)} entries loaded from Excel")
        except Exception as e:
            logger.warning(f"  BRE mapping failed: {e}")

    # --- OVA: Extract from two Excel files ---
    ova1_path = project_root / "data" / "clinical_data" / "3. 난소암" / "SMCXD01_난소암 1.xlsx"
    ova2_path = project_root / "data" / "clinical_data" / "3. 난소암" / "SMCXD01_난소암 2.xlsx"
    ova_count = 0
    if ova1_path.exists():
        try:
            ova1 = pd.read_excel(ova1_path, header=None, skiprows=2)
            ova1_map = ova1[[1, 4]].dropna()
            ova1_map.columns = ["hospital_id", "solum_label"]
            ova1_map["hospital_id"] = ova1_map["hospital_id"].astype(int).astype(str)
            for _, row in ova1_map.iterrows():
                solum_to_clinical[row["solum_label"]] = row["hospital_id"]
            ova_count += len(ova1_map)
        except Exception as e:
            logger.warning(f"  OVA file 1 mapping failed: {e}")

    if ova2_path.exists():
        try:
            ova2 = pd.read_excel(ova2_path, header=None, skiprows=1)
            ova2_map = ova2[[0, 3]].dropna()
            ova2_map.columns = ["solum_label", "hospital_id"]
            ova2_map["hospital_id"] = ova2_map["hospital_id"].astype(int).astype(str)
            for _, row in ova2_map.iterrows():
                solum_to_clinical[row["solum_label"]] = row["hospital_id"]
            ova_count += len(ova2_map)
        except Exception as e:
            logger.warning(f"  OVA file 2 mapping failed: {e}")

    if ova_count:
        logger.info(f"  OVA mapping: {ova_count} entries loaded from Excel")

    # --- BLC: Sequential numbering from BLA staging Excel ---
    blc_path = project_root / "data" / "clinical_data" / "11. 방광암" / "SMCXD06_방광암.xlsm"
    if blc_path.exists():
        try:
            blc_xl = pd.read_excel(blc_path, header=None, skiprows=2)
            blc_map = blc_xl[[0, 1]].dropna()
            blc_map.columns = ["stage_id", "hospital_id"]
            blc_map["hospital_id"] = blc_map["hospital_id"].astype(int).astype(str)
            # Sequential: 1기~2기-1 → BLC 1, ..., 3기~4기-10 → BLC 300
            for i, (_, row) in enumerate(blc_map.iterrows(), start=1):
                solum_to_clinical[f"BLC {i}"] = row["stage_id"]
            logger.info(f"  BLC mapping: {len(blc_map)} entries loaded from Excel")
        except Exception as e:
            logger.warning(f"  BLC mapping failed: {e}")

    # --- LUN 201-300: SoluM Label → 병원번호 from SMCXD06_폐암 3.xlsx ---
    lun3_path = project_root / "data" / "clinical_data" / "4. 폐암" / "SMCXD06_폐암 3.xlsx"
    if lun3_path.exists():
        try:
            lun3 = pd.read_excel(lun3_path, sheet_name="자원리스트", header=None, skiprows=1)
            lun3_map = lun3[[0, 3]].dropna()
            lun3_map.columns = ["solum_label", "hospital_id"]
            lun3_map["hospital_id"] = lun3_map["hospital_id"].astype(int).astype(str)
            for _, row in lun3_map.iterrows():
                solum_to_clinical[row["solum_label"]] = row["hospital_id"]
            logger.info(f"  LUN 201-300 mapping: {len(lun3_map)} entries loaded from Excel")
        except Exception as e:
            logger.warning(f"  LUN 201-300 mapping failed: {e}")

    # --- H.D.: space normalization ---
    for i in range(1, 201):
        solum_to_clinical[f"H.D. {i}"] = f"H. D. {i}"

    return solum_to_clinical


def merge_clinical_features(
    df_spec: pd.DataFrame,
    clin: pd.DataFrame,
    clinical_cols: List[str] = TIER1_COLS,
    project_root: Path = PROJECT_ROOT,
) -> np.ndarray:
    """Merge clinical features with spectral dataframe by patient_id.

    Applies multi-strategy matching:
        1. Direct key match: "{group} {sample_id}" → patient_id
        2. ID mapping: BRE/OVA (hospital ID), BLC (stage ID), H.D. (space fix)
        3. Alias fallback: PAN → CPAN/YPAN original group names

    Returns:
        np.ndarray of shape (N, len(clinical_cols)), aligned with df_spec rows.
        Missing BMI filled with median; remaining NaN filled with 0.
    """
    # Build primary lookup: patient_id -> {col: value}
    clin_lookup = {}
    for _, row in clin.iterrows():
        pid = str(row["patient_id"])
        clin_lookup[pid] = {c: row.get(c, np.nan) for c in clinical_cols}

    # Build ID mappings for non-standard groups
    id_mappings = build_id_mappings(project_root)

    n = len(df_spec)
    result = np.full((n, len(clinical_cols)), np.nan)
    matched = 0
    match_details = {}

    for i, (_, row) in enumerate(df_spec.iterrows()):
        group = row["group"]
        sample_id = str(row["sample_id"])
        key = f"{group} {sample_id}"

        # Strategy 1: Direct match
        if key in clin_lookup:
            for j, c in enumerate(clinical_cols):
                result[i, j] = clin_lookup[key][c]
            matched += 1
            match_details[group] = match_details.get(group, 0) + 1
            continue

        # Strategy 2: ID mapping (BRE/OVA hospital IDs, BLC stage IDs, H.D. space fix)
        if key in id_mappings:
            mapped_pid = id_mappings[key]
            if mapped_pid in clin_lookup:
                for j, c in enumerate(clinical_cols):
                    result[i, j] = clin_lookup[mapped_pid][c]
                matched += 1
                match_details[group] = match_details.get(group, 0) + 1
                continue

        # Strategy 3: Alias fallback (PAN → try CPAN, YPAN original names)
        if group == "PAN":
            group_raw = row.get("group_raw", "")
            if group_raw:
                raw_key = f"{group_raw} {sample_id}"
                if raw_key in clin_lookup:
                    for j, c in enumerate(clinical_cols):
                        result[i, j] = clin_lookup[raw_key][c]
                    matched += 1
                    match_details[group] = match_details.get(group, 0) + 1
                    continue
            # Try both CPAN and YPAN
            for prefix in ["CPAN", "YPAN"]:
                alt_key = f"{prefix} {sample_id}"
                if alt_key in clin_lookup:
                    for j, c in enumerate(clinical_cols):
                        result[i, j] = clin_lookup[alt_key][c]
                    matched += 1
                    match_details[group] = match_details.get(group, 0) + 1
                    break

    match_rate = matched / n * 100
    logger.info(f"  Clinical merge: {matched}/{n} matched ({match_rate:.1f}%)")
    for g, cnt in sorted(match_details.items()):
        total_g = (df_spec["group"] == g).sum()
        logger.info(f"    {g}: {cnt}/{total_g}")

    # Log unmatched groups
    all_groups = df_spec["group"].value_counts()
    for g in all_groups.index:
        if g not in match_details or match_details[g] < all_groups[g]:
            unmatched = all_groups[g] - match_details.get(g, 0)
            if unmatched > 0:
                logger.warning(f"    {g}: {unmatched} unmatched")

    # Clip BMI outliers (physiological range: 10-60)
    if "bmi" in clinical_cols:
        bmi_idx = clinical_cols.index("bmi")
        bmi_vals = result[:, bmi_idx]
        n_outlier = ((bmi_vals > 60) & ~np.isnan(bmi_vals)).sum()
        if n_outlier > 0:
            logger.warning(f"  BMI outliers clipped: {n_outlier} values > 60")
        bmi_vals = np.where((bmi_vals > 60) & ~np.isnan(bmi_vals), np.nan, bmi_vals)
        result[:, bmi_idx] = bmi_vals

    # Impute BMI with median if present in clinical_cols
    if "bmi" in clinical_cols:
        bmi_idx = clinical_cols.index("bmi")
        bmi_vals = result[:, bmi_idx]
        bmi_valid = bmi_vals[~np.isnan(bmi_vals)]
        if len(bmi_valid) > 0:
            median_bmi = np.median(bmi_valid)
            n_imputed = np.isnan(bmi_vals).sum()
            result[:, bmi_idx] = np.where(np.isnan(bmi_vals), median_bmi, bmi_vals)
            if n_imputed > 0:
                logger.info(f"  BMI imputed: {n_imputed} samples with median={median_bmi:.1f}")

    # Fill remaining NaN with per-column median (not 0, to avoid implicit masking)
    for col_idx in range(result.shape[1]):
        col = result[:, col_idx]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            valid = col[~nan_mask]
            fill_val = np.median(valid) if len(valid) > 0 else 0.0
            result[nan_mask, col_idx] = fill_val
            logger.info(f"  {clinical_cols[col_idx]} NaN filled: {nan_mask.sum()} samples with median={fill_val:.1f}")

    return result.astype(np.float32)


def compute_multichannel_spectra(X: np.ndarray) -> np.ndarray:
    """Compute 3-channel spectral input: raw + 1st derivative + 2nd derivative.

    Args:
        X: (N, L) raw spectra

    Returns:
        (N, 3, L) array — [raw, d1, d2]
    """
    d1 = np.gradient(X, axis=1)
    d2 = np.gradient(d1, axis=1)
    result = np.stack([X, d1, d2], axis=1)
    logger.info(
        f"  Multi-channel spectra: {X.shape} → {result.shape} "
        f"(raw + 1st_deriv + 2nd_deriv)"
    )
    return result


class ClinicalSERSDataset(torch.utils.data.Dataset):
    """Dataset with multi-channel spectra + clinical features.

    Supports both 1D (N, L) and multi-channel (N, C, L) spectral input.
    __getitem__ returns dict with 'spectra', 'clinical', 'binary_label', 'cancer_type_label'.
    """

    def __init__(
        self,
        spectra: np.ndarray,
        binary_labels: np.ndarray,
        cancer_type_labels: np.ndarray,
        clinical_features: np.ndarray,
        groups: Optional[list] = None,
        sample_ids: Optional[list] = None,
        augment: bool = False,
        noise_std: float = 0.02,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        shift_max: int = 3,
    ):
        self.spectra = torch.FloatTensor(spectra)  # (N, L) or (N, C, L)
        self.binary_labels = torch.FloatTensor(binary_labels).unsqueeze(-1)
        self.cancer_type_labels = torch.LongTensor(cancer_type_labels)
        self.clinical = torch.FloatTensor(clinical_features)
        self.groups = groups or ["UNK"] * len(spectra)
        self.sample_ids = sample_ids or [str(i) for i in range(len(spectra))]
        self.augment = augment
        self.noise_std = noise_std
        self.scale_range = scale_range
        self.shift_max = shift_max
        self.multichannel = self.spectra.dim() == 3

    def __len__(self):
        return len(self.spectra)

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Augment spectra. Works for both (L,) and (C, L) shapes."""
        # Gaussian noise (applied to all channels)
        x = x + torch.randn_like(x) * self.noise_std
        # Random intensity scaling
        scale = torch.empty(1).uniform_(*self.scale_range).item()
        x = x * scale
        # Random spectral shift (circular, same shift for all channels)
        shift = torch.randint(-self.shift_max, self.shift_max + 1, (1,)).item()
        if shift != 0:
            x = torch.roll(x, shifts=shift, dims=-1)
        return x

    def __getitem__(self, idx):
        spec = self.spectra[idx]
        if self.augment:
            spec = self._augment(spec)
        return {
            "spectra": spec,
            "binary_label": self.binary_labels[idx],
            "cancer_type_label": self.cancer_type_labels[idx],
            "clinical": self.clinical[idx],
        }
