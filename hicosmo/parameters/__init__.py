"""
HIcosmo Parameter Management System
===================================

This module provides a unified parameter registry for managing cosmological
and nuisance parameters across MCMC sampling and Fisher forecasting.

Key Features
------------
- **Unified Parameter Management**: Single source of truth for all parameters
- **Smart Defaults**: Load Planck 2018, WMAP9 presets with one line
- **Automatic Nuisance Merging**: Extract nuisance parameters from likelihoods
- **Clean API**: 3-line parameter setup for typical use cases

Quick Start
-----------
>>> from hicosmo.parameters import ParameterRegistry
>>>
>>> # Load Planck 2018 cosmological parameters
>>> registry = ParameterRegistry.from_defaults('planck2018')
>>>
>>> # Choose which parameters to sample
>>> registry.set_free(['H0', 'Omega_m'])
>>>
>>> # Add a custom nuisance parameter
>>> registry.add('M_B', value=-19.23, free=True,
...              prior={'dist': 'uniform', 'min': -20, 'max': -18})

Examples
--------
**Example 1: SNe Ia analysis with SN_likelihood**

>>> from hicosmo.parameters import ParameterRegistry
>>> from hicosmo.likelihoods import SN_likelihood
>>>
>>> registry = ParameterRegistry.from_defaults('planck2018')
>>> registry.set_free(['H0', 'Omega_m'])
>>>
>>> # Automatically add nuisance parameters (M_B, alpha, beta)
>>> pantheon = SN_likelihood(LCDM, "pantheon+", M_B="free")
>>> registry.add_from_likelihood(pantheon)
>>>
>>> # Get free parameters for MCMC
>>> free_params = registry.get_free()
>>> print(list(free_params.keys()))
['H0', 'Omega_m', 'M_B', 'alpha', 'beta']

**Example 2: Custom parameter configuration**

>>> registry = ParameterRegistry()
>>> registry.add('H0', value=70, free=True,
...              prior={'dist': 'uniform', 'min': 50, 'max': 100},
...              latex_label=r'$H_0$')
>>> registry.add('Omega_m', value=0.3, free=True,
...              prior={'dist': 'normal', 'loc': 0.3, 'scale': 0.05})
>>> registry.add('Omega_b', value=0.049, free=False)  # Fixed parameter

Module Contents
---------------
"""

from .parameter import Parameter, DEFAULT_LATEX_LABELS
from .registry import ParameterRegistry
from .presets import (
    PLANCK_2018_PARAMS,
    PLANCK_2018_DEFAULTS,
    WMAP9_PARAMS,
    WMAP9_DEFAULTS,
    PRESET_CATALOG,
    get_preset,
    get_fiducial_values,
)
from .validation import (
    validate_parameter_name,
    validate_prior_dict,
    validate_bounds,
    suggest_similar_names,
    build_parameter_error_message,
)

__all__ = [
    # Core classes
    "Parameter",
    "ParameterRegistry",
    # Preset parameters (Single Source of Truth)
    "PLANCK_2018_PARAMS",
    "PLANCK_2018_DEFAULTS",
    "WMAP9_PARAMS",
    "WMAP9_DEFAULTS",
    "PRESET_CATALOG",
    "get_preset",
    "get_fiducial_values",
    # Validation utilities
    "validate_parameter_name",
    "validate_prior_dict",
    "validate_bounds",
    "suggest_similar_names",
    "build_parameter_error_message",
]
