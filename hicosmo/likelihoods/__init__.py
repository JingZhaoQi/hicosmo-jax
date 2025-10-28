"""
HIcosmo Likelihood System
========================

Clean, minimal likelihood implementations following CLAUDE.md design principles.
Currently available:
- PantheonPlus Type Ia supernova likelihood
- BAO (Baryon Acoustic Oscillations) likelihoods

Features:
- Simple interface: likelihood.log_likelihood(model)
- JAX-optimized performance
- Clean dependency management
"""

# Only import working, clean modules
from .pantheonplus import PantheonPlusLikelihood, create_pantheonplus_likelihood
from .bao_base import BAOLikelihood, BAOCollection
from .bao_datasets import (
    SDSSDR12BAO, SDSSDR16BAO, BOSSDR12BAO,
    DESI2024BAO, SixDFBAO, CustomBAO,
    get_available_datasets, create_bao_likelihood
)
from .h0licow import H0LiCOWLikelihood
from .planck_distance import Planck2018DistancePriorsLikelihood
from .sh0es import SH0ESLikelihood
from .tdcosmo import TDCOSMOLikelihood, KappaPrior

__all__ = [
    # Supernova
    'PantheonPlusLikelihood',
    'create_pantheonplus_likelihood',
    # BAO base
    'BAOLikelihood',
    'BAOCollection',
    # Specific BAO datasets
    'SDSSDR12BAO',
    'SDSSDR16BAO',
    'BOSSDR12BAO',
    'DESI2024BAO',
    'SixDFBAO',
    'CustomBAO',
    # BAO utilities
    'get_available_datasets',
    'create_bao_likelihood',
    # Strong lensing
    'H0LiCOWLikelihood',
    # CMB distance priors
    'Planck2018DistancePriorsLikelihood',
    'SH0ESLikelihood',
    'TDCOSMOLikelihood',
    'KappaPrior'
]
