"""
JAX-Cosmology Analysis Module

A unified, object-oriented interface for analyzing MCMC results from 
cosmological parameter estimation.

Main Classes:
    MCMCAnalysis: Primary analysis interface for single/multiple chains
    PlottingEngine: Professional plotting with GetDist integration
    StatisticsEngine: Parameter statistics and convergence diagnostics
    ColorSchemes: Professional color schemes for plotting

Usage:
    from analysis import MCMCAnalysis
    
    # Single chain analysis
    analysis = MCMCAnalysis('mcmc_results.h5')
    analysis.corner_plot(filename='corner.pdf')
    analysis.print_summary()
    
    # Multiple chain comparison
    analysis = MCMCAnalysis(['chain1.h5', 'chain2.h5'], 
                           names=['Model A', 'Model B'])
    analysis.comparison_plot(filename='comparison.pdf')
"""

from .core import (
    MCMCAnalysis,
    PlottingEngine, 
    StatisticsEngine,
    ColorSchemes,
    DataLoader
)

__version__ = "1.0.0"

__all__ = [
    'MCMCAnalysis',
    'PlottingEngine',
    'StatisticsEngine', 
    'ColorSchemes',
    'DataLoader'
]