#!/usr/bin/env python3
"""
HIcosmo Visualization System - User's Preferred Architecture
"""

from pathlib import Path
from typing import Union, List, Optional, Any

import numpy as np

from .plotting import (
    plot_corner,
    plot_chains,
    plot_1d,
    plot_fisher_contours,
    RESULTS_DIR,
)
from .mcplot_redesign import MCplot as MCplotNew
from getdist.gaussian_mixtures import GaussianND

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

    def fisher_contours(
        self,
        mean: Union[List[float], np.ndarray],
        covariance: Union[List[List[float]], np.ndarray],
        params: List[str],
        filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """Generate Fisher contour plot from mean/covariance."""
        return plot_fisher_contours(mean, covariance, params, filename=filename, **kwargs)

    # Aliases - backward compatibility
    plot_corner = corner
    plot_chains = traces
    plot_traces = traces

# User's preferred architecture - new MCplot class
MCplot = MCplotNew


class FisherPlot:
    """Lightweight Gaussian Fisher contour helper inspired by legacy qcosmc implementation."""

    def __init__(
        self,
        mean: Union[List[float], np.ndarray],
        covariance: Union[List[List[float]], np.ndarray],
        labels: List[str],
        legend: str = '',
        *,
        nsample: int = 200_000,
        style: str = 'modern',
    ) -> None:
        self.param_names = list(labels)
        self.style = style
        self.nsample = nsample
        self._samples: List[Any] = []
        self._legends: List[str] = []

        self.add_covariance(mean, covariance, legend or 'Forecast')

    def add_covariance(
        self,
        mean: Union[List[float], np.ndarray],
        covariance: Union[List[List[float]], np.ndarray],
        legend: str,
    ) -> None:
        mean_arr = np.asarray(mean, dtype=float)
        cov_arr = np.asarray(covariance, dtype=float)
        gaussian = GaussianND(mean_arr, cov_arr, names=self.param_names, labels=self.param_names)
        samples = gaussian.MCSamples(self.nsample)
        samples.label = legend
        self._samples.append(samples)
        self._legends.append(legend)

    def figure(
        self,
        filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        data = self._samples if len(self._samples) > 1 else self._samples[0]
        return plot_corner(
            data,
            style=self.style,
            filename=filename,
            labels=self._legends if len(self._samples) > 1 else None,
            **kwargs,
        )

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
            group_name = 'samples' if 'samples' in f else 'chains'
            if group_name in f:
                for param in f[group_name].keys():
                    data[param] = f[group_name][param][:]
        return data
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

__all__ = [
    'plot_corner',
    'plot_chains',
    'plot_1d',
    'plot_fisher_contours',
    'load_chain_simple',
    'HIcosmoViz',
    'FisherPlot',
    'MCplot',
]
