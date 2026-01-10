"""
CMB (Cosmic Microwave Background) Likelihoods
==============================================

Provides likelihood implementations for CMB distance priors.

Available Datasets:
- Planck2018DistancePriorsLikelihood: Planck 2018 compressed likelihood

Quick Start:
    >>> from hicosmo.likelihoods import Planck
    >>> from hicosmo.models import LCDM
    >>> cmb = Planck(LCDM)
    >>> log_L = cmb(H0=70, Omega_m=0.3, Omega_b=0.05)
"""

from .planck_distance import Planck2018DistancePriorsLikelihood

__all__ = [
    "Planck2018DistancePriorsLikelihood",
]
