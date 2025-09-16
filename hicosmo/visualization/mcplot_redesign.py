#!/usr/bin/env python3
"""
HIcosmo MCplot Class - User's Preferred Architecture

Designed for convenience with chain initialization:
chains = [['wCDM_SN','SN'], ['wCDM_BAO','BAO'], ['wCDM_CMB','CMB']]
pl = MCplot(chains)
pl.plot3D([1,2,3])  # 3-parameter corner plot
pl.plot2D([1,2])    # 2-parameter corner plot
pl.plot1D(0)        # 1D probability distribution
pl.results          # LaTeX formatted parameter results
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt

try:
    from getdist import plots, MCSamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    raise ImportError("GetDist is required. Install with: pip install getdist")

from .plotting import (_apply_qijing_style, _prepare_getdist_samples, _optimize_ticks,
                       _get_results_dir, MODERN_COLORS, CLASSIC_COLORS)

class MCplot:
    """
    Monte Carlo plot manager following user's preferred architecture

    Initialize with chains list, then use convenient methods:
    pl.plot3D([1,2,3]) - 3-parameter corner plot
    pl.plot2D([1,2])   - 2-parameter corner plot
    pl.plot1D(0)       - 1D probability distribution
    pl.results         - LaTeX parameter results
    """

    def __init__(self, chains: List[List[str]], style: str = 'modern'):
        """
        Initialize MCplot with chains information

        Parameters
        ----------
        chains : list of [chain_file, label] pairs
            Format: [['wCDM_SN','SN'], ['wCDM_BAO','BAO'], ...]
        style : str
            Color scheme ('modern' or 'classic')
        """
        self.chains_info = chains
        self.style = style
        self.colors = MODERN_COLORS if style == 'modern' else CLASSIC_COLORS

        # Load chain data and convert to GetDist samples
        self.samples_list = []
        self.labels = []

        for chain_file, label in chains:
            # Load chain data (assuming file exists)
            try:
                samples = _prepare_getdist_samples(chain_file, None)
                samples.label = label
                self.samples_list.append(samples)
                self.labels.append(label)
            except Exception as e:
                print(f"Warning: Could not load {chain_file}: {e}")

        if not self.samples_list:
            raise ValueError("No valid chains could be loaded")

        # Get parameter information from first chain
        self.param_names = self.samples_list[0].getParamNames().list()
        self.n_params = len(self.param_names)
        self.n_chains = len(self.samples_list)

        print(f"Loaded {self.n_chains} chains with {self.n_params} parameters")
        print(f"Parameters: {self.param_names}")
        print(f"Chain labels: {self.labels}")

    def plot_corner(self, params: Union[int, List[int]], filename: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create corner plot (supports 2D, 3D, or multi-parameter)

        Parameters
        ----------
        params : int or list of int
            Single parameter index (plots first 3 params) or list of parameter indices
        filename : str, optional
            Save filename
        **kwargs
            Additional GetDist parameters

        Returns
        -------
        fig : matplotlib.Figure
        """
        if isinstance(params, int):
            # Single parameter given, plot first 3 starting from that parameter
            param_indices = [params, params+1, params+2]
        else:
            param_indices = params

        return self._create_corner_plot(param_indices, filename, **kwargs)

    def plot_2D(self, params: List[int], filename: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create 2D contour plot (not corner plot)

        Parameters
        ----------
        params : list of int
            List of 2 parameter indices
        filename : str, optional
            Save filename
        **kwargs
            Additional GetDist parameters

        Returns
        -------
        fig : matplotlib.Figure
        """
        if len(params) != 2:
            raise ValueError("plot_2D requires exactly 2 parameters")

        _apply_qijing_style()

        # Get parameter names
        param1_name = self.param_names[params[0]]
        param2_name = self.param_names[params[1]]

        # Create single plotter for 2D plot
        plotter = plots.get_single_plotter(width_inch=6)
        plotter.settings.axes_fontsize = 12
        plotter.settings.lab_fontsize = 14
        plotter.settings.legend_fontsize = 12
        plotter.settings.figure_legend_frame = False

        # Use GetDist's plot_2d function directly with custom colors
        plotter.plot_2d(self.samples_list, param1_name, param2_name,
                       filled=True,
                       colors=self.colors[:self.n_chains],
                       **kwargs)

        # Add legend if labels exist (regardless of chain count)
        if self.labels:
            plotter.add_legend(legend_labels=self.labels, legend_loc='best')

        _optimize_ticks(plotter.fig)
        plt.tight_layout()

        # Auto-save
        if filename:
            results_dir = _get_results_dir()
            save_path = results_dir / filename
            if not save_path.suffix:
                save_path = save_path.with_suffix('.pdf')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.fig.savefig(save_path, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            print(f"2D plot saved to: {save_path}")

        return plotter.fig

    def plot_1D(self, param: int, filename: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Create 1D probability distribution plot

        Parameters
        ----------
        param : int
            Parameter index
        filename : str, optional
            Save filename
        **kwargs
            Additional GetDist parameters

        Returns
        -------
        fig : matplotlib.Figure
        """
        _apply_qijing_style()

        # Create single plotter
        plotter = plots.get_single_plotter(width_inch=8)
        plotter.settings.axes_fontsize = 12
        plotter.settings.lab_fontsize = 14
        plotter.settings.legend_fontsize = 12
        plotter.settings.figure_legend_frame = False

        # Get parameter name
        param_name = self.param_names[param]

        # Plot 1D distributions for all chains
        plotter.plot_1d(self.samples_list, param_name,
                        colors=self.colors[:self.n_chains], **kwargs)

        # Add legend if labels exist (regardless of chain count)
        if self.labels:
            plotter.add_legend(legend_labels=self.labels, legend_loc='best')

        _optimize_ticks(plotter.fig)
        plt.tight_layout()

        # Auto-save
        if filename:
            results_dir = _get_results_dir()
            save_path = results_dir / filename
            if not save_path.suffix:
                save_path = save_path.with_suffix('.pdf')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.fig.savefig(save_path, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            print(f"1D plot saved to: {save_path}")

        return plotter.fig

    def _create_corner_plot(self, param_indices: List[int], filename: Optional[str] = None, **kwargs) -> plt.Figure:
        """
        Internal method to create corner plots
        """
        _apply_qijing_style()

        # Validate parameter indices
        for idx in param_indices:
            if idx >= self.n_params:
                raise ValueError(f"Parameter index {idx} >= n_params {self.n_params}")

        # Create subset of samples with selected parameters
        selected_samples = []
        for samples in self.samples_list:
            # Get parameter names for selected indices
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
        plotter.settings.axes_fontsize = 12
        plotter.settings.lab_fontsize = 14
        plotter.settings.legend_fontsize = 12
        plotter.settings.figure_legend_frame = False

        # Assign colors
        contour_colors = self.colors[:self.n_chains]
        line_colors = self.colors[:self.n_chains]

        # Create triangle plot - GetDist will handle legend automatically
        plotter.triangle_plot(selected_samples, filled=True,
                             contour_colors=contour_colors,
                             line_args=[{'color': c, 'lw': 2} for c in line_colors],
                             **kwargs)

        _optimize_ticks(plotter.fig)
        plt.tight_layout()

        # Auto-save
        if filename:
            results_dir = _get_results_dir()
            save_path = results_dir / filename
            if not save_path.suffix:
                save_path = save_path.with_suffix('.pdf')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plotter.fig.savefig(save_path, dpi=300, bbox_inches='tight',
                               facecolor='white', edgecolor='none')
            print(f"Corner plot saved to: {save_path}")

        return plotter.fig

    @property
    def results(self) -> Dict[str, Any]:
        """
        Get LaTeX formatted parameter results

        Returns
        -------
        dict
            Parameter results in LaTeX format for each chain
        """
        results = {
            'chains': self.labels,
            'parameters': {}
        }

        for param_idx, param_name in enumerate(self.param_names):
            param_results = {}

            for chain_idx, (samples, label) in enumerate(zip(self.samples_list, self.labels)):
                if param_idx < samples.samples.shape[1]:
                    data = samples.samples[:, param_idx]
                    mean = np.mean(data)
                    std = np.std(data)
                    median = np.median(data)
                    q16 = np.percentile(data, 16)
                    q84 = np.percentile(data, 84)

                    # LaTeX formatted result
                    latex_result = f"${mean:.3f} \\pm {std:.3f}$"
                    confidence_68 = f"${median:.3f}^{{+{q84-median:.3f}}}_{{-{median-q16:.3f}}}$"

                    param_results[label] = {
                        'mean_std': latex_result,
                        'median_68': confidence_68,
                        'raw_values': {
                            'mean': float(mean),
                            'std': float(std),
                            'median': float(median),
                            'q16': float(q16),
                            'q84': float(q84)
                        }
                    }

            results['parameters'][param_name] = param_results

        return results

    def print_results(self):
        """
        Print LaTeX formatted results to console
        """
        results = self.results

        print("\n" + "="*60)
        print("PARAMETER RESULTS (LaTeX Format)")
        print("="*60)

        for param_name, param_data in results['parameters'].items():
            print(f"\n📊 Parameter: {param_name}")
            print("-" * 40)

            for chain_label, chain_results in param_data.items():
                print(f"  {chain_label}:")
                print(f"    Mean ± Std:     {chain_results['mean_std']}")
                print(f"    Median ± 68%:   {chain_results['median_68']}")

    def save_results_image(self, filename: Optional[str] = None) -> plt.Figure:
        """
        Save LaTeX formatted results as image file

        Parameters
        ----------
        filename : str, optional
            Save filename (defaults to 'results_table.pdf')

        Returns
        -------
        fig : matplotlib.Figure
        """
        results = self.results

        # Create figure for results table
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis('off')

        # Prepare table data
        table_data = []
        headers = ['Parameter', 'Chain'] + ['Mean ± Std', 'Median ± 68%']

        for param_name, param_data in results['parameters'].items():
            for i, (chain_label, chain_results) in enumerate(param_data.items()):
                row = [
                    param_name if i == 0 else '',
                    chain_label,
                    chain_results['mean_std'],
                    chain_results['median_68']
                ]
                table_data.append(row)
            # Add separator row
            if param_name != list(results['parameters'].keys())[-1]:
                table_data.append(['', '', '', ''])

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        bbox=[0, 0, 1, 1])

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)

        # Style headers
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # Style parameter name cells
        for i, row in enumerate(table_data):
            if row[0] and row[0] != '':  # Parameter name row
                table[(i+1, 0)].set_facecolor('#E8F5E8')
                table[(i+1, 0)].set_text_props(weight='bold')

        # Add title
        fig.suptitle('MCMC Parameter Estimation Results',
                    fontsize=16, fontweight='bold', y=0.95)

        # Add chains info
        chains_info = f"Chains: {', '.join(self.labels)}"
        fig.text(0.5, 0.02, chains_info, ha='center', fontsize=10, style='italic')

        plt.tight_layout()

        # Auto-save
        if filename is None:
            filename = 'results_table.pdf'

        results_dir = _get_results_dir()
        save_path = results_dir / filename
        if not save_path.suffix:
            save_path = save_path.with_suffix('.pdf')

        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Results table saved to: {save_path}")

        return fig

    def __repr__(self) -> str:
        return f"MCplot({self.n_chains} chains, {self.n_params} parameters)"