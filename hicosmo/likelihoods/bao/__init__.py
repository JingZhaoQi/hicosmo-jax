"""
BAO (Baryon Acoustic Oscillations) Likelihoods
===============================================

Provides likelihood implementations for BAO measurements from various surveys.

Available Datasets:
- DESI2024BAO: DESI DR1 2024 BAO measurements
- SDSSDR12BAO: SDSS DR12 consensus BAO
- SDSSDR16BAO: SDSS DR16 (eBOSS) BAO
- BOSSDR12BAO: BOSS DR12 BAO
- SixDFBAO: 6dF Galaxy Survey BAO
- CustomBAO: User-defined BAO measurements

Quick Start:
    >>> from hicosmo.likelihoods import BAO_likelihood
    >>> from hicosmo.models import LCDM
    >>> bao = BAO_likelihood(LCDM, "desi2024")
    >>> log_L = bao(H0=70, Omega_m=0.3)
"""

from .base import BAOCollection, BAOLikelihood
from .datasets import (
    BOSSDR12BAO,
    DESI2024BAO,
    SDSSDR12BAO,
    SDSSDR16BAO,
    CustomBAO,
    SixDFBAO,
    create_bao_likelihood,
    get_available_datasets,
)
from .factory import (
    BAO_likelihood,
    BAOLikelihoodFactory,
    register_bao_dataset,
    available_bao_datasets,
)

__all__ = [
    # Base classes
    "BAOLikelihood",
    "BAOCollection",
    # Specific datasets
    "DESI2024BAO",
    "SDSSDR12BAO",
    "SDSSDR16BAO",
    "BOSSDR12BAO",
    "SixDFBAO",
    "CustomBAO",
    # Factory functions
    "BAO_likelihood",
    "BAOLikelihoodFactory",
    "create_bao_likelihood",
    "get_available_datasets",
    "register_bao_dataset",
    "available_bao_datasets",
]
