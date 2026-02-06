"""
H0 (Hubble Constant) Likelihoods
================================

Provides likelihood implementations for direct H0 measurements.

Available Datasets:
- SH0ESLikelihood: SH0ES distance ladder measurement

Quick Start:
    >>> from hicosmo.likelihoods import SH0ESLikelihood
    >>> shoes = SH0ESLikelihood()
    >>> log_L = shoes(H0=73.0)
"""

from .shoes import SH0ESLikelihood

__all__ = [
    "SH0ESLikelihood",
]
