from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .group_separation import GroupSeparationDiagnostics
from .signal_noise_ablation import AblationResult
from .signal_noise_lot import VarianceComponents
from .signal_noise_peak import PeakRepeatabilityDiagnostics


@dataclass(frozen=True, slots=True)
class EffectRetentionSummary:
    group_pair: str
    relevant_feature_count: int
    spearman_rho: float
    sign_agreement: float
    median_absolute_retention: float


@dataclass(frozen=True, slots=True)
class ImportantPeakRetention:
    peak_name: str
    shift_cm_1: float
    legacy_importance: float
    legacy_effect: float
    powder_effect: float
    absolute_effect_retention: float
    direction_agreement: bool
    legacy_presence_rate: float
    powder_presence_rate: float
    category: str


@dataclass(frozen=True, slots=True)
class PeakLandscape:
    legacy_presence_rate: np.ndarray
    powder_presence_rate: np.ndarray
    legacy_max_abs_effect: np.ndarray
    powder_max_abs_effect: np.ndarray
    categories: np.ndarray


@dataclass(frozen=True, slots=True)
class PatientNoiseLink:
    patient_id: str
    sample_id: int
    clinical_group: str
    powder_replicate_correlation: float
    powder_consensus_fraction: float
    native_cancer_probability: float
    native_screening_correct: bool


@dataclass(frozen=True, slots=True)
class SignalNoiseDiagnostics:
    repeatability: PeakRepeatabilityDiagnostics
    separation: GroupSeparationDiagnostics
    effect_summaries: tuple[EffectRetentionSummary, ...]
    important_peaks: tuple[ImportantPeakRetention, ...]
    landscape: PeakLandscape
    lot_variance: VarianceComponents
    screening_ablation: tuple[AblationResult, ...]
    three_group_ablation: tuple[AblationResult, ...]
    patient_links: tuple[PatientNoiseLink, ...]
    conclusions: tuple[str, ...]
