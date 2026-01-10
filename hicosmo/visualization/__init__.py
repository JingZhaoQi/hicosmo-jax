"""
HIcosmo Visualization System
============================

Unified API for cosmological parameter visualization.

Usage (3 lines):
```python
from hicosmo.visualization import Plotter

plotter = Plotter('my_chain')
plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')
```

Multi-chain Comparison:
```python
plotter = Plotter(['chain1', 'chain2'], labels=['Planck', 'SH0ES'])
plotter.corner(filename='comparison.pdf')
```

Fisher Forecasts:
```python
plotter = Plotter.from_fisher(mean, cov, param_names=['w0', 'wa'])
plotter.corner(filename='forecast.pdf')
```

Author: Jingzhao Qi
"""

from .plotter import Plotter
from .styles import apply_style, qstyle
from .information_criteria import (
    aic,
    bic,
    log_evidence_harmonic_mean,
    information_criteria,
)

__all__ = [
    'Plotter',
    'apply_style',
    'qstyle',
    'aic',
    'bic',
    'log_evidence_harmonic_mean',
    'information_criteria',
]

__version__ = '3.0.0'
