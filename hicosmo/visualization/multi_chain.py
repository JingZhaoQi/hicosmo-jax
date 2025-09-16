#!/usr/bin/env python3
"""
HIcosmo Multi-Chain Management System

A clean class-based approach for multi-chain visualization:
- Initialize with chain data and labels
- plot_corner([1,2,3]) for parameter indices
- plot_corner(['H0', 'Omega_m']) for parameter names
- Automatic legend and color management

Author: Jingzhao Qi
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt

try:
    from getdist import plots, MCSamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    raise ImportError("GetDist is required. Install with: pip install getdist")

from .plotting import _apply_qijing_style, _prepare_getdist_samples, _optimize_ticks, _get_results_dir
from .plotting import MODERN_COLORS, CLASSIC_COLORS

class MultiChain:
    """
    Multi-chain data manager for cosmological parameter analysis

    Initialize with chain data and labels, then use plot_corner([1,2,3])
    for parameter indices or plot_corner(['H0', 'Omega_m']) for names.
    """

    def __init__(self, chains: List[Any], labels: List[str], param_names: Optional[List[str]] = None):
        """
        Initialize multi-chain manager

        Parameters
        ----------
        chains : list
            List of chain data (files, arrays, dicts, etc.)
        labels : list of str
            Labels for each chain ['Planck 2018', 'SH0ES 2022', ...]
        param_names : list of str, optional
            Parameter names ['H0', 'Omega_m', 'sigma8', ...]. If None,
            will be inferred from first chain if possible.
        """
        if len(chains) != len(labels):
            raise ValueError(f"chains length ({len(chains)}) != labels length ({len(labels)})")

        self.chains = chains
        self.labels = labels
        self.n_chains = len(chains)

        # Convert to GetDist samples and infer parameter names
        self.samples_list = []
        for i, chain_data in enumerate(chains):
            samples = _prepare_getdist_samples(chain_data, None)
            samples.label = labels[i]
            self.samples_list.append(samples)

        # Get parameter names from first chain
        if param_names is not None:
            self.param_names = param_names
        else:
            self.param_names = self.samples_list[0].getParamNames().list()

        self.n_params = len(self.param_names)

    def plot_corner(self, params: Union[List[int], List[str]],
                   style: str = 'modern', filename: Optional[str] = None,
                   **kwargs) -> plt.Figure:
        """
        Create corner plot for specified parameters

        Parameters
        ----------
        params : list of int or list of str
            Parameter indices [0, 1, 2] or names ['H0', 'Omega_m']
        style : str
            Color style 'modern' or 'classic'
        filename : str, optional
            Save filename (auto-saves to caller's results/)
        **kwargs
            Additional arguments for GetDist

        Returns
        -------
        fig : matplotlib.Figure
            Corner plot figure with multi-chain comparison
        """
        # Apply professional style
        _apply_qijing_style()

        # Select colors
        colors = MODERN_COLORS if style == 'modern' else CLASSIC_COLORS

        # Convert parameter specification to indices
        if isinstance(params[0], str):
            # Parameter names provided
            param_indices = []
            for param_name in params:
                if param_name in self.param_names:
                    param_indices.append(self.param_names.index(param_name))
                else:
                    raise ValueError(f"Parameter '{param_name}' not found in {self.param_names}")
        else:
            # Parameter indices provided
            param_indices = params
            for idx in param_indices:
                if idx >= self.n_params:
                    raise ValueError(f"Parameter index {idx} >= n_params {self.n_params}")

        # Create subset of samples with only requested parameters
        selected_samples = []
        for samples in self.samples_list:
            # Get parameter names for the selected indices
            selected_param_names = [self.param_names[i] for i in param_indices]

            # Extract data for selected parameters
            selected_data = []
            for param_name in selected_param_names:
                param_index = samples.getParamNames().list().index(param_name)
                selected_data.append(samples.samples[:, param_index])

            # Create new MCSamples with selected parameters
            selected_samples_array = np.column_stack(selected_data)
            subset_samples = MCSamples(samples=selected_samples_array,
                                     names=selected_param_names,
                                     labels=[samples.getParamNames().parWithName(name).label
                                           for name in selected_param_names],
                                     label=samples.label)
            selected_samples.append(subset_samples)

        # Create GetDist plotter
        plotter = plots.get_subplot_plotter(width_inch=8)

        # Professional settings
        plotter.settings.axes_fontsize = 12
        plotter.settings.lab_fontsize = 14
        plotter.settings.legend_fontsize = 12
        plotter.settings.figure_legend_frame = False

        # Assign colors to chains
        contour_colors = colors[:self.n_chains]
        line_colors = colors[:self.n_chains]

        # Create triangle plot
        plotter.triangle_plot(selected_samples, filled=True,
                             contour_colors=contour_colors,
                             line_args=[{'color': c, 'lw': 2} for c in line_colors],
                             **kwargs)

        # Add legend
        plotter.add_legend(legend_labels=self.labels, legend_loc='upper right')

        # Optimize display
        _optimize_ticks(plotter.fig)
        plt.tight_layout()

        # Auto-save to caller's results/ directory
        if filename:
            results_dir = _get_results_dir()
            save_path = results_dir / filename
            if not save_path.suffix:
                save_path = save_path.with_suffix('.pdf')

            save_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.fig.savefig(save_path, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            print(f"Multi-chain corner plot saved to: {save_path}")

        return plotter.fig

    def plot_chains(self, params: Union[List[int], List[str]],
                   style: str = 'modern', filename: Optional[str] = None,
                   **kwargs) -> plt.Figure:
        """
        Create chain trace plots for convergence diagnostics

        Parameters
        ----------
        params : list of int or list of str
            Parameter indices or names to plot
        style : str
            Color style
        filename : str, optional
            Save filename
        **kwargs
            Additional arguments

        Returns
        -------
        fig : matplotlib.Figure
            Chain trace plots
        """
        _apply_qijing_style()
        colors = MODERN_COLORS if style == 'modern' else CLASSIC_COLORS

        # Convert params to indices if needed
        if isinstance(params[0], str):
            param_indices = [self.param_names.index(p) for p in params]
        else:
            param_indices = params

        n_params = len(param_indices)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params))
        if n_params == 1:
            axes = [axes]

        for i, param_idx in enumerate(param_indices):
            param_name = self.param_names[param_idx]

            for j, samples in enumerate(self.samples_list):
                chain_data = samples.samples[:, param_idx]
                axes[i].plot(chain_data, color=colors[j], alpha=0.8,
                           label=self.labels[j], lw=1.5)

            axes[i].set_ylabel(param_name)
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend()

        axes[-1].set_xlabel('Sample Index')
        plt.tight_layout()

        if filename:
            results_dir = _get_results_dir()
            save_path = results_dir / filename
            if not save_path.suffix:
                save_path = save_path.with_suffix('.pdf')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chain traces saved to: {save_path}")

        return fig

    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics for all chains

        Returns
        -------
        dict
            Summary statistics for each chain and parameter
        """
        summary = {
            'n_chains': self.n_chains,
            'labels': self.labels,
            'param_names': self.param_names,
            'chains': {}
        }

        for i, (samples, label) in enumerate(zip(self.samples_list, self.labels)):
            chain_summary = {
                'n_samples': samples.numrows,
                'parameters': {}
            }

            for j, param_name in enumerate(self.param_names):
                if j < samples.samples.shape[1]:
                    data = samples.samples[:, j]
                    chain_summary['parameters'][param_name] = {
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data)),
                        'median': float(np.median(data)),
                        'q16': float(np.percentile(data, 16)),
                        'q84': float(np.percentile(data, 84))
                    }

            summary['chains'][label] = chain_summary

        return summary

    def __repr__(self) -> str:
        return f"MultiChain({self.n_chains} chains, {self.n_params} parameters)"