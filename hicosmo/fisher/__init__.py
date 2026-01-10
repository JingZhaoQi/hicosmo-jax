"""
HIcosmo Fisher Matrix Analysis
=============================

Professional Fisher matrix analysis tools for cosmological parameter estimation.
Provides comprehensive forecasting capabilities for 21cm intensity mapping surveys.

Key features:
- Fisher information matrix calculation with JAX automatic differentiation
- Parameter constraint forecasting for future surveys
- 21cm intensity mapping survey optimization
- Multi-probe synergy analysis
"""

from .fisher_matrix import FisherMatrix, FisherMatrixConfig
from .numerical_derivatives import (
    compute_step_sizes,
    finite_difference_gradient,
    finite_difference_hessian,
    finite_difference_jacobian,
)
from .intensity_mapping import (
    IntensityMappingSurvey,
    IntensityMappingFisher,
    list_available_surveys,
    load_survey,
    run_forecast,
)

__all__ = [
    # Core Fisher matrix tools
    'FisherMatrix',
    'FisherMatrixConfig',

    # Numerical differentiation utilities
    'compute_step_sizes',
    'finite_difference_gradient',
    'finite_difference_hessian',
    'finite_difference_jacobian',

    # Intensity mapping forecasts
    'IntensityMappingSurvey',
    'IntensityMappingFisher',
    'list_available_surveys',
    'load_survey',
    'run_forecast',
]
