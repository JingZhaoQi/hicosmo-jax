"""
HiCosmo Likelihood System
========================

Production-grade likelihood calculations for cosmological parameter estimation.
Supports multi-probe analyses with real observational data.

Key features:
- Type Ia Supernovae (Pantheon+, Union3)
- Baryon Acoustic Oscillations (BOSS, eBOSS)
- Cosmic Microwave Background (Planck, ACT, SPT)
- Weak Lensing (KiDS, DES, HSC)
- Hubble constant measurements (SH0ES, H0LiCOW)
- Multi-probe combination and covariance handling
"""

from .observational_data import ObservationalDataManager
from .likelihoods import LikelihoodCalculator
from .multi_probe import MultiProbeLikelihood

__all__ = [
    'ObservationalDataManager',
    'LikelihoodCalculator',
    'MultiProbeLikelihood'
]