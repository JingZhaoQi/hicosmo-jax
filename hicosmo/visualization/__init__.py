"""
HiCosmo Visualization Package
============================

Professional plotting and visualization tools for cosmological analysis.

Key Features:
- Elegant corner plots with GetDist integration
- MCMC trace plots with convergence diagnostics  
- Cosmological function visualization
- Hubble diagram plots for supernova data
- Publication-ready styling inspired by core.py aesthetics

Quick Start:
-----------
```python
from hicosmo.visualization import quick_corner, quick_trace, HiCosmoPlotter

# Quick corner plot
fig = quick_corner(samples, labels=['H0', 'Omega_m', 'w'])

# Professional plotter with full control
plotter = HiCosmoPlotter()
fig = plotter.corner_plot(samples, truths=[70, 0.3, -1])
```
"""

from .plotting import (
    HiCosmoPlotter,
    quick_corner,
    quick_trace, 
    quick_hubble_diagram,
    setup_style,
    HICOSMO_COLORS,
    COLOR_CYCLE,
)

__all__ = [
    'HiCosmoPlotter',
    'quick_corner',
    'quick_trace',
    'quick_hubble_diagram', 
    'setup_style',
    'HICOSMO_COLORS',
    'COLOR_CYCLE',
]