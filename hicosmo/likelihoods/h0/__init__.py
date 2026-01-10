"""
H0 (Hubble Constant) Likelihoods
================================

Provides likelihood implementations for direct H0 measurements.

Available Datasets:
- SH0ESLikelihood: SH0ES distance ladder measurement
- H0LiCOWLikelihood: H0LiCOW strong lensing time delays

Quick Start:
    >>> from hicosmo.likelihoods import H0LiCOW
    >>> from hicosmo.models import LCDM
    >>> lensing = H0LiCOW(LCDM)
    >>> log_L = lensing(H0=70, Omega_m=0.3)
"""

from .shoes import SH0ESLikelihood
from .h0licow import H0LiCOWLikelihood

__all__ = [
    "SH0ESLikelihood",
    "H0LiCOWLikelihood",
]
