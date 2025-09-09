"""
HiCosmo Power Spectrum Module
============================

Professional power spectrum calculations including:
- Linear matter power spectrum
- Non-linear corrections (Halofit, etc.)
- Baryon acoustic oscillations
- Cross-correlations and auto-correlations
- Redshift-space distortions
- Weak lensing power spectra
"""

from .linear_power import LinearPowerSpectrum
from .nonlinear_corrections import NonlinearCorrections
from .cross_correlations import CrossCorrelationCalculator

__all__ = [
    'LinearPowerSpectrum',
    'NonlinearCorrections', 
    'CrossCorrelationCalculator'
]