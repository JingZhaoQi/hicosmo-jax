"""
Supernova (SN Ia) Likelihoods
=============================

Provides likelihood implementations for Type Ia supernova distance measurements.

Available Datasets:
- PantheonPlus: Pantheon+ sample (1701 SNe Ia)

Quick Start:
    >>> from hicosmo.likelihoods import SN_likelihood
    >>> from hicosmo.models import LCDM
    >>> sne = SN_likelihood(LCDM, "pantheon+", M_B="free")
    >>> log_L = sne(H0=70, Omega_m=0.3, M_B=-19.3)
"""

from .pantheonplus import PantheonPlusLikelihood, create_pantheonplus_likelihood
from .factory import (
    SN_likelihood,
    SNLikelihood,
    register_sn_dataset,
    available_sn_datasets,
)

__all__ = [
    # Likelihood classes
    "PantheonPlusLikelihood",
    "SNLikelihood",
    # Factory functions
    "SN_likelihood",
    "create_pantheonplus_likelihood",
    "register_sn_dataset",
    "available_sn_datasets",
]
