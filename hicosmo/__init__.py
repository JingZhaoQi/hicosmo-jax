"""
HIcosmo: High-performance universal cosmology framework.

HIcosmo (HI = neutral hydrogen, I = Roman numeral 1) is a universal 
cosmological parameter estimation framework with enhanced functionality 
and optimizations for neutral hydrogen cosmology and 21cm surveys. 
Built with JAX for high-performance computing and modern MCMC methods.
"""

__version__ = "0.1.0"
__author__ = "Jingzhao Qi"

from . import core
from . import models
from . import likelihoods
from . import samplers
from . import parameters
# from . import fisher  # Temporarily disabled during refactoring
from . import utils
from . import visualization

__all__ = [
    "core",
    "models",
    "likelihoods",
    "samplers",
    "parameters",
    # "fisher",  # Temporarily disabled
    "utils",
    "visualization",
]