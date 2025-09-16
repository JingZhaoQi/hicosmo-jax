"""
HIcosmo Fisher Matrix Analysis
=============================

Professional Fisher matrix analysis tools for cosmological parameter estimation.
Provides comprehensive forecasting capabilities, survey optimization, and 
dark energy figure-of-merit calculations.

Key features:
- Fisher information matrix calculation with JAX automatic differentiation
- Parameter constraint forecasting for future surveys
- Survey optimization and experimental design
- Multi-probe synergy analysis
- Dark Energy Task Force (DETF) figure of merit
- Modified gravity and neutrino mass figures of merit
- Hubble tension and curvature constraint analysis
"""

from .fisher_matrix import FisherMatrix, FisherMatrixConfig
from .forecasting import (
    CosmologicalForecaster,
    SurveySpecification, 
    ExperimentalDesign
)
from .figures_of_merit import (
    DarkEnergyFiguresOfMerit,
    FigureOfMeritConfig
)

__all__ = [
    # Core Fisher matrix tools
    'FisherMatrix',
    'FisherMatrixConfig',
    
    # Forecasting tools
    'CosmologicalForecaster',
    'SurveySpecification',
    'ExperimentalDesign',
    
    # Figure of merit tools
    'DarkEnergyFiguresOfMerit', 
    'FigureOfMeritConfig'
]