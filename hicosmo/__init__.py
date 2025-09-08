"""
HiCosmo: High-performance cosmological parameter estimation framework.

A JAX-based cosmology package for parameter inference using modern MCMC methods.
"""

__version__ = "0.1.0"
__author__ = "HiCosmo Team"

from . import core
from . import models
from . import likelihoods
from . import samplers
from . import parameters
from . import fisher
from . import utils
from . import visualization

__all__ = [
    "core",
    "models",
    "likelihoods",
    "samplers",
    "parameters",
    "fisher",
    "utils",
    "visualization",
]