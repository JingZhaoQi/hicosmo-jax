"""
HIcosmo Cosmological Models - Clean and Unified
===============================================

This package provides production-ready implementations of cosmological models
using JAX-first numerical tools.

Available models:
- LCDM: Standard Lambda-CDM model
- wCDM: Constant dark energy equation of state
- CPL: Chevallier-Polarski-Linder parametrization
- ILCDM: Interacting Lambda-CDM (energy transfer Q = βHρ_c)

All models now provide:
- JAX-first numerical kernels
- Complete parameter validation
- Full precision control
"""

from .lcdm import LCDM
from .wcdm import wCDM
from .cpl import CPL
from .ilcdm import ILCDM, planck2018_ilcdm
from .base import (
    CosmologyBase,
    compute_background_grid_for_model,
    compute_distances_from_E_z,
    make_compute_background_grid_traced,
    make_compute_grid_traced,
    make_sound_horizon_traced,
    make_sound_horizon_drag_traced,
    make_recombination_redshift_traced,
)
from .unified_parameters import CosmologicalParameters, PLANCK_2018, PLANCK_2015, WMAP9

__all__ = [
    "LCDM",
    "wCDM",
    "CPL",
    "ILCDM",
    "planck2018_ilcdm",
    "CosmologyBase",
    "compute_background_grid_for_model",
    "compute_distances_from_E_z",
    "make_compute_background_grid_traced",
    "make_compute_grid_traced",
    "make_sound_horizon_traced",
    "make_sound_horizon_drag_traced",
    "make_recombination_redshift_traced",
    "CosmologicalParameters",
    "PLANCK_2018",
    "PLANCK_2015",
    "WMAP9",
]
