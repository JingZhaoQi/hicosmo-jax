#!/usr/bin/env python3
"""
HIcosmo Visualization System - Core Plotting Functions
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple

try:
    from getdist import plots, MCSamples
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    raise ImportError("GetDist is required. Install with: pip install getdist")

# Color schemes
MODERN_COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#592E83', '#0F7173', '#7B2D26']
CLASSIC_COLORS = ['#348ABD', '#7A68A6', '#E24A33', '#467821', '#ffb3a6', '#188487', '#A60628']

# LaTeX label mappings
LATEX_LABELS = {
    'H0': r'H_0 ~[\mathrm{km~s^{-1}~Mpc^{-1}}]',
    'Omega_m': r'\Omega_m',
    'Omega_b': r'\Omega_b',
    'sigma8': r'\sigma_8',
    'w': r'w',
    'w0': r'w_0',
    'wa': r'w_a',
}

# Results directory
def _get_results_dir():
    """Get results directory relative to calling script"""
    import inspect
    frame = inspect.currentframe()
    try:
        # Get first non-visualization module frame in call stack
        caller_frame = frame.f_back.f_back if frame.f_back else frame.f_back
        if caller_frame and 'visualization' not in caller_frame.f_code.co_filename:
            caller_dir = Path(caller_frame.f_code.co_filename).parent
        else:
            caller_dir = Path.cwd()
        results_dir = caller_dir / 'results'
        results_dir.mkdir(exist_ok=True)
        return results_dir
    finally:
        del frame

RESULTS_DIR = Path('results')  # fallback
RESULTS_DIR.mkdir(exist_ok=True)

def _apply_qijing_style():
    """Apply professional plotting style - based on FigStyle.py"""
    import seaborn as sns

    # Basic style configuration (merged best parts from original 5 styles)
    style_config = {
        'axes.linewidth': 1.2,
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 14,
        'legend.frameon': False,  # Key: frameless legend
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'savefig.facecolor': 'white',
        'font.size': 12,
        'lines.linewidth': 2.0,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'mathtext.fontset': 'cm',
        'mathtext.rm': 'serif',
    }

    plt.rcParams.update(style_config)

    # Set color cycle
    plt.gca().set_prop_cycle('color', MODERN_COLORS)


def _optimize_ticks(fig):
    """Smart tick optimization - prevents label overlap"""
    import matplotlib.font_manager as fm

    for ax in fig.get_axes():
        # Get axis dimensions in pixels
        bbox = ax.get_window_extent()
        width, height = bbox.width, bbox.height

        # Skip empty axes
        if width <= 0 or height <= 0:
            continue

        # Get current font size for tick labels
        try:
            font_size = plt.rcParams['xtick.labelsize']
            if isinstance(font_size, str) or font_size is None:
                font_size = 11  # default fallback
        except:
            font_size = 11  # safe fallback

        # Estimate average character width in pixels
        char_width = font_size * 0.6  # rough estimate

        # X-axis optimization
        if ax.get_xlabel() or len(ax.get_xticklabels()) > 0:
            # Estimate label width more conservatively (scientific notation + margins)
            estimated_label_width = char_width * 6.5  # longer for scientific notation
            # Use more generous spacing to prevent overlap (1.8x instead of 1.2x)
            max_x_ticks = max(3, int(width / (estimated_label_width * 1.8)))
            # Cap at more conservative limits
            optimal_x_ticks = min(max_x_ticks, 6)
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=optimal_x_ticks, prune='both'))

        # Y-axis optimization
        if ax.get_ylabel() or len(ax.get_yticklabels()) > 0:
            # Y-axis labels need more vertical space too
            line_height = font_size * 1.8  # more generous line spacing
            max_y_ticks = max(3, int(height / (line_height * 2.2)))
            optimal_y_ticks = min(max_y_ticks, 6)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=optimal_y_ticks, prune='both'))

        # Handle very small subplots
        if width < 120:
            # Very narrow - reduce ticks and rotate
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha('right')
        elif width < 180:
            # Narrow - just reduce ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune='both'))

        if height < 100:
            # Very short - minimal y-ticks
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune='both'))


def _load_chain_simple(filename: str):
    """Simple data loading - replaces complex ChainManager"""
    import numpy as np

    file_path = Path(filename)

    if file_path.suffix == '.npy':
        return np.load(file_path)
    elif file_path.suffix == '.npz':
        npz_file = np.load(file_path)
        if len(npz_file.files) == 1:
            return npz_file[npz_file.files[0]]
        return dict(npz_file)
    elif file_path.suffix in ['.txt', '.dat']:
        return np.loadtxt(file_path)
    elif file_path.suffix in ['.h5', '.hdf5']:
        try:
            import h5py
            data = {}
            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    data[key] = f[key][:]
            return data
        except ImportError:
            raise ImportError("h5py required for HDF5 files")
    else:
        try:
            return np.loadtxt(file_path)
        except:
            raise ValueError(f"Cannot load file format: {file_path.suffix}")


def _prepare_latex_labels(param_names: List[str]) -> List[str]:
    """Prepare LaTeX labels with auto-formatting"""
    labels = []
    for param in param_names:
        if param in LATEX_LABELS:
            latex_label = LATEX_LABELS[param]
        else:
            # Clean redundant $ symbols
            latex_label = param.replace('$', '')
            if '_' in latex_label:
                latex_label = latex_label.replace('_', '_{') + '}'
        labels.append(latex_label)
    return labels


def _prepare_getdist_samples(data, params=None) -> MCSamples:
    """Convert to GetDist format - replaces complex adapters"""
    if isinstance(data, MCSamples):
        return data

    # Select parameters
    if isinstance(data, str):
        data = _load_chain_simple(data)

    if params is not None:
        # Handle indices or names
        if isinstance(data, dict):
            if isinstance(params[0], int):
                # Convert 1-based index to 0-based index
                param_names = list(data.keys())
                selected_params = [param_names[i] for i in params]
            else:
                # Parameter name
                selected_params = params
            samples = np.column_stack([data[p] for p in selected_params])
            param_names = selected_params
        else:
            samples = data[:, params]
            param_names = [f'param_{i}' for i in params]
    else:
        if isinstance(data, dict):
            param_names = list(data.keys())
            samples = np.column_stack([data[k] for k in param_names])
        else:
            samples = data
            param_names = [f'param_{i}' for i in range(samples.shape[1])]

    # Prepare LaTeX labels
    labels = _prepare_latex_labels(param_names)

    return MCSamples(samples=samples, names=param_names, labels=labels)


def plot_corner(data, params=None, style='modern', filename=None, labels=None, **kwargs) -> plt.Figure:
    """
    Create corner plot - unified simple interface with multi-chain comparison support

    Parameters
    ----------
    data : various or list of various
        Single chain: chain data (file path, array, dict, ChainData object)
        Multi-chain: [data1, data2, ...] multi-chain data list
    params : list, optional
        Parameters to plot (indices or names)
    style : str
        Color scheme ('modern' or 'classic')
    filename : str, optional
        Save filename (auto-saves to caller's results/ directory)
    labels : list of str, optional
        Label list for multi-chain case ['Chain1', 'Chain2', ...]
    **kwargs
        Parameters passed to GetDist

    Returns
    -------
    fig : plt.Figure
        Corner plot with multi-chain comparison and legend support
    """
    # Apply style
    _apply_qijing_style()

    # Select colors
    colors = MODERN_COLORS if style == 'modern' else CLASSIC_COLORS

    # Detect multi-chain data
    is_multi_chain = isinstance(data, (list, tuple)) and len(data) > 1

    if is_multi_chain:
        # Multi-chain mode
        samples_list = []
        for i, chain_data in enumerate(data):
            samples = _prepare_getdist_samples(chain_data, params)
            # Set labels
            if labels and i < len(labels):
                samples.label = labels[i]
            else:
                samples.label = f'Chain {i+1}'
            samples_list.append(samples)

        # Assign different colors to each chain
        contour_colors = colors[:len(samples_list)]
        line_colors = colors[:len(samples_list)]
    else:
        # Single chain mode - backward compatible
        samples = _prepare_getdist_samples(data, params)
        samples_list = [samples]
        contour_colors = [colors[0]]
        line_colors = [colors[0]]

    # Create GetDist plotter
    plotter = plots.get_subplot_plotter(width_inch=8)

    # GetDist professional settings
    plotter.settings.axes_fontsize = 12
    plotter.settings.lab_fontsize = 14
    plotter.settings.legend_fontsize = 12
    plotter.settings.figure_legend_frame = False

    # Draw corner plot
    if is_multi_chain:
        # Multi-chain plotting - use different colors and legend
        plotter.triangle_plot(samples_list, filled=True,
                             contour_colors=contour_colors,
                             line_args=[{'color': c, 'lw': 2} for c in line_colors],
                             **kwargs)
        # Add legend
        if any(s.label for s in samples_list):
            plotter.add_legend(legend_labels=[s.label for s in samples_list],
                              legend_loc='upper right')
    else:
        # Single chain plotting
        plotter.triangle_plot(samples_list, filled=True,
                             contour_colors=contour_colors,
                             line_args={'color': line_colors[0], 'lw': 2},
                             **kwargs)

    # Optimize ticks
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
        print(f"Corner plot saved to: {save_path}")

    return plotter.fig


def plot_chains(data, params=None, style='modern', filename=None, **kwargs) -> plt.Figure:
    """
    Create chain trace plots for convergence diagnostics

    Parameters
    ----------
    data : various
        Chain data
    params : list, optional
        Parameters to plot
    style : str
        Color scheme
    filename : str, optional
        Save filename
    **kwargs
        Additional parameters

    Returns
    -------
    fig : plt.Figure
        Chain trace plots
    """
    _apply_qijing_style()
    colors = MODERN_COLORS if style == 'modern' else CLASSIC_COLORS

    samples = _prepare_getdist_samples(data, params)

    if params is None:
        n_params = min(6, samples.samples.shape[1])  # Max 6 parameters
        selected_params = list(range(n_params))
    else:
        selected_params = params

    fig, axes = plt.subplots(len(selected_params), 1, figsize=(10, 2*len(selected_params)))
    if len(selected_params) == 1:
        axes = [axes]

    for i, param_idx in enumerate(selected_params):
        chain_data = samples.samples[:, param_idx]
        axes[i].plot(chain_data, color=colors[0], alpha=0.8, lw=1.5)
        axes[i].set_ylabel(samples.getParamNames().list()[param_idx])
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Sample Index')
    plt.tight_layout()

    if filename:
        results_dir = _get_results_dir()
        save_path = results_dir / filename
        if not save_path.suffix:
            save_path = save_path.with_suffix('.pdf')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        print(f"Chain traces saved to: {save_path}")

    return fig


def plot_1d(data, params=None, style='modern', filename=None, **kwargs) -> plt.Figure:
    """
    Create 1D marginal distribution plots

    Parameters
    ----------
    data : various
        Chain data
    params : list, optional
        Parameters to plot
    style : str
        Color scheme
    filename : str, optional
        Save filename
    **kwargs
        Additional parameters

    Returns
    -------
    fig : plt.Figure
        1D marginal plots
    """
    _apply_qijing_style()
    colors = MODERN_COLORS if style == 'modern' else CLASSIC_COLORS

    samples = _prepare_getdist_samples(data, params)
    plotter = plots.get_single_plotter(width_inch=8)

    plotter.settings.axes_fontsize = 12
    plotter.settings.lab_fontsize = 14
    plotter.settings.legend_fontsize = 12
    plotter.settings.figure_legend_frame = False

    plotter.plots_1d(samples, **kwargs)
    _optimize_ticks(plotter.fig)
    plt.tight_layout()

    if filename:
        results_dir = _get_results_dir()
        save_path = results_dir / filename
        if not save_path.suffix:
            save_path = save_path.with_suffix('.pdf')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plotter.fig.savefig(save_path, dpi=300, bbox_inches='tight',
                           facecolor='white', edgecolor='none')
        print(f"1D plots saved to: {save_path}")

    return plotter.fig


# Backward compatible aliases
corner = plot_corner
traces = plot_chains
marginals = plot_1d