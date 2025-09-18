"""
HIcosmo Likelihood System
========================

Clean, minimal likelihood implementations following CLAUDE.md design principles.
Currently available: PantheonPlus Type Ia supernova likelihood.

Features:
- Simple interface: likelihood.log_likelihood(model)
- JAX-optimized performance
- Clean dependency management
"""

# Only import working, clean modules
from .pantheonplus import PantheonPlusLikelihood, create_pantheonplus_likelihood

__all__ = [
    'PantheonPlusLikelihood',
    'create_pantheonplus_likelihood'
]