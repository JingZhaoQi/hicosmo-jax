#!/usr/bin/env python3
"""
HIcosmo Visualization System - Unified Interface

Minimalist refactor:
- Removed all Manager classes
- Unified simple function interface
- Direct GetDist wrapper, no intermediate layers

Author: Jingzhao Qi
Total Lines: ~150 (vs original 636 lines)
"""

from pathlib import Path
from typing import Union, List, Optional, Any

from .plotting import plot_corner, plot_chains, plot_1d, RESULTS_DIR

class HIcosmoViz:
    """
    HIcosmo visualization unified interface - minimalist version

    Simple wrapper around functions. Recommended: use functions directly.
    """

    def __init__(self, results_dir: Union[str, Path] = 'results'):
        """
        Initialize visualization interface

        Parameters
        ----------
        results_dir : str or Path
            Results save directory
        """
        global RESULTS_DIR
        RESULTS_DIR = Path(results_dir)
        RESULTS_DIR.mkdir(exist_ok=True)
        self.results_dir = RESULTS_DIR

    def corner(self, data, params=None, filename=None, **kwargs):
        """Create corner plot"""
        return plot_corner(data, params=params, filename=filename, **kwargs)

    def plot3D(self, data, params: List[Union[str, int]], filename=None, **kwargs):
        """Create 3-parameter corner plot - backward compatibility"""
        if len(params) != 3:
            raise ValueError("plot3D requires exactly 3 parameters")
        return self.corner(data, params=params, filename=filename, **kwargs)

    def traces(self, data, params=None, filename=None, **kwargs):
        """Create chain trace plots"""
        return plot_chains(data, params=params, filename=filename, **kwargs)

    def plot_1d(self, data, params=None, filename=None, **kwargs):
        """Create 1D marginal distribution plots"""
        return plot_1d(data, params=params, filename=filename, **kwargs)

    # Aliases - backward compatibility
    plot_corner = corner
    plot_chains = traces
    plot_traces = traces

# Backward compatibility alias
MCplot = HIcosmoViz

def load_chain_simple(filename: str):
    """
    Simple chain loading function - replaces complex ChainManager

    Parameters
    ----------
    filename : str
        File path

    Returns
    -------
    data
        Data ready for plotting functions
    """
    import numpy as np

    file_path = Path(filename)

    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix in ['.h5', '.hdf5']:
        import h5py
        data = {}
        with h5py.File(file_path, 'r') as f:
            if 'chains' in f:
                for param in f['chains'].keys():
                    data[param] = f['chains'][param][:]
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

__all__ = [
    'plot_corner',
    'plot_chains',
    'plot_1d',
    'load_chain_simple',
    'HIcosmoViz',
    'MCplot',
]