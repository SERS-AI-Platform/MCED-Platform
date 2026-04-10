"""
Cross-instrument calibration transfer for SERS spectra.

Provides paired-sample calibration methods to map spectra from one instrument
domain (e.g. Medical Raman) to another (e.g. Thermo) and vice versa.

All transformers operate on spectra that have already been resampled onto a
SHARED wavenumber grid (same number of channels, same wavenumber positions).
Use src.sers.preprocessing to put both domains on a common grid first.

Methods
-------
- AffineTransfer       : per-channel y = a*x + b (simplest baseline)
- PDSTransfer          : Piecewise Direct Standardization (chemometrics standard)
- OSCTransfer          : Orthogonal Signal Correction via paired-difference PCA
- IdentityTransfer     : no-op (raw baseline)

All transformers expose .fit(X_source, X_target) and .transform(X) where the
direction is "source → target" (i.e. transform takes a source-domain spectrum
and returns its estimate in the target domain).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


# =============================================================================
# Identity (raw baseline)
# =============================================================================
class IdentityTransfer:
    name = "raw"

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "IdentityTransfer":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X.copy()


# =============================================================================
# Affine — per-channel linear fit
# =============================================================================
@dataclass
class AffineTransfer:
    """Per-wavenumber linear fit:  x_target_i ≈ a_i * x_source_i + b_i.

    Robust, fast, and handles uniform baseline shift + scaling. Does NOT handle
    peak position shift or local resolution differences.
    """
    name: str = "affine"
    a: Optional[np.ndarray] = None
    b: Optional[np.ndarray] = None

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "AffineTransfer":
        assert X_source.shape == X_target.shape, "paired matrices must have equal shape"
        n, p = X_source.shape
        a = np.empty(p)
        b = np.empty(p)
        for j in range(p):
            xs = X_source[:, j]
            xt = X_target[:, j]
            var = xs.var()
            if var < 1e-12:
                a[j], b[j] = 1.0, xt.mean() - xs.mean()
            else:
                a[j] = np.cov(xs, xt, bias=True)[0, 1] / var
                b[j] = xt.mean() - a[j] * xs.mean()
        self.a = a
        self.b = b
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.a is None:
            raise RuntimeError("AffineTransfer not fit")
        return X * self.a + self.b


# =============================================================================
# PDS — Piecewise Direct Standardization
# =============================================================================
@dataclass
class PDSTransfer:
    """Piecewise Direct Standardization.

    For each target channel j, fit a local linear model that maps a window of
    source channels [j-w..j+w] to the target channel j. Solved with ridge
    regression to control overfit when paired sample count is small or windows
    are wide.

    Reference: Wang, Veltkamp & Kowalski, Anal. Chem. 1991, 63(23), 2750-2756.

    Parameters
    ----------
    half_window : int
        Number of channels on each side of the center to include (window size
        = 2*half_window + 1).
    ridge : float
        L2 regularization for the local regression. 0 = OLS.
    """
    half_window: int = 7
    ridge: float = 1.0
    name: str = "pds"
    F: Optional[np.ndarray] = None  # (p, p) sparse-banded transformation matrix

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "PDSTransfer":
        assert X_source.shape == X_target.shape
        n, p = X_source.shape
        w = self.half_window
        F = np.zeros((p, p), dtype=np.float64)

        # Center both for stable regression; we restore the mean offset per-channel
        mu_s = X_source.mean(axis=0)
        mu_t = X_target.mean(axis=0)
        Xs_c = X_source - mu_s
        Xt_c = X_target - mu_t

        for j in range(p):
            lo = max(0, j - w)
            hi = min(p, j + w + 1)
            A = Xs_c[:, lo:hi]                     # (n, k)
            y = Xt_c[:, j]                         # (n,)
            k = A.shape[1]
            # Ridge: (A.T A + λI) β = A.T y
            G = A.T @ A
            G.flat[:: k + 1] += self.ridge
            try:
                beta = np.linalg.solve(G, A.T @ y)
            except np.linalg.LinAlgError:
                beta = np.linalg.lstsq(A, y, rcond=None)[0]
            F[lo:hi, j] = beta

        self.F = F
        self.mu_s = mu_s
        self.mu_t = mu_t
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.F is None:
            raise RuntimeError("PDSTransfer not fit")
        return (X - self.mu_s) @ self.F + self.mu_t


# =============================================================================
# OSC — Orthogonal Signal Correction (paired-difference flavor)
# =============================================================================
@dataclass
class OSCTransfer:
    """Orthogonal Signal Correction by removing top-k PCs of the paired
    instrument-difference matrix.

    Idea: D = X_target - X_source captures instrument-induced variation that
    is, by construction, the bias subspace between domains on aligned samples.
    We project that subspace OUT of any incoming source spectrum, then add back
    the mean target shift.

    This is a simple, fast complement to PDS. It removes global multiplicative
    & baseline modes that PDS may not capture cleanly.
    """
    n_components: int = 2
    name: str = "osc"
    components_: Optional[np.ndarray] = None  # (k, p)
    mean_shift_: Optional[np.ndarray] = None  # (p,)

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "OSCTransfer":
        D = X_target - X_source                       # (n, p)
        self.mean_shift_ = D.mean(axis=0)
        Dc = D - self.mean_shift_
        # SVD on D — top-k right singular vectors are the bias directions
        try:
            _, s, Vt = np.linalg.svd(Dc, full_matrices=False)
        except np.linalg.LinAlgError:
            _, s, Vt = np.linalg.svd(Dc + 1e-9 * np.random.randn(*Dc.shape),
                                     full_matrices=False)
        k = min(self.n_components, Vt.shape[0])
        self.components_ = Vt[:k]                     # (k, p) orthonormal rows
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.components_ is None:
            raise RuntimeError("OSCTransfer not fit")
        # Subtract projection onto bias subspace, then add mean shift
        proj = (X @ self.components_.T) @ self.components_
        return X - proj + self.mean_shift_


# =============================================================================
# Composite — chain transformers (e.g. PDS then OSC)
# =============================================================================
@dataclass
class ChainTransfer:
    steps: list = field(default_factory=list)
    name: str = "chain"

    def fit(self, X_source: np.ndarray, X_target: np.ndarray) -> "ChainTransfer":
        cur = X_source
        for step in self.steps:
            step.fit(cur, X_target)
            cur = step.transform(cur)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        cur = X
        for step in self.steps:
            cur = step.transform(cur)
        return cur


# =============================================================================
# Diagnostics
# =============================================================================
def paired_rmse(X_source: np.ndarray, X_target: np.ndarray) -> float:
    """Mean per-spectrum RMSE between paired matrices."""
    diff = X_source - X_target
    return float(np.sqrt((diff ** 2).mean()))


def paired_correlation(X_source: np.ndarray, X_target: np.ndarray) -> float:
    """Mean per-spectrum Pearson correlation between paired matrices."""
    n = X_source.shape[0]
    corrs = np.empty(n)
    for i in range(n):
        a = X_source[i] - X_source[i].mean()
        b = X_target[i] - X_target[i].mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        corrs[i] = (a @ b) / denom if denom > 0 else 0.0
    return float(corrs.mean())
