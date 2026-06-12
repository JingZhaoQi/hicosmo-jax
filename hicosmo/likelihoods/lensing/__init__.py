"""
Strong Gravitational Lensing Likelihoods
=========================================

Provides likelihood implementations for strong gravitational lensing time delays.

Available Classes:
- H0LiCOWLikelihood: H0LiCOW 6-lens time-delay distance analysis
- TDCOSMOLikelihood: Hierarchical TDCOSMO analysis
- ExternalLensLikelihood: External lens likelihood
- HierarchicalTDCOSMO: Full hierarchical model
- KappaPrior: External convergence prior

Quick Start:
    >>> from hicosmo.likelihoods import H0LiCOW, TDCOSMO
    >>> from hicosmo.models import LCDM
    >>> lensing = H0LiCOW(LCDM)
    >>> log_L = lensing(H0=70, Omega_m=0.3)
"""

from .h0licow import H0LiCOWLikelihood
from .tdcosmo import (
    ExternalLensLikelihood,
    HierarchicalTDCOSMO,
    KappaPrior,
    TDCOSMOLikelihood,
)

__all__ = [
    "H0LiCOWLikelihood",
    "TDCOSMOLikelihood",
    "ExternalLensLikelihood",
    "HierarchicalTDCOSMO",
    "KappaPrior",
    "HierarchicalGWLikelihood",
]


def __getattr__(name):
    # Lazy: hierarchical_helper depends on the optional GW module, which is
    # not part of the tracked repository. Importing it eagerly would make
    # `import hicosmo.likelihoods` crash on a fresh clone.
    if name == "HierarchicalGWLikelihood":
        from .hierarchical_helper import HierarchicalGWLikelihood

        return HierarchicalGWLikelihood
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
