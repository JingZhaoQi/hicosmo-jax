"""
HiCosmo Visualization Module
===========================

Elegant plotting utilities inspired by professional matplotlib aesthetics.
Provides simple, publication-ready plots for MCMC analysis and cosmological functions.

Key Features:
- Corner plots with GetDist integration
- Trace plots with convergence diagnostics
- Cosmological function plots
- Professional styling and formatting
- Smart tick formatting and layout
"""

from typing import Dict, List, Optional, Union, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
import jax.numpy as jnp
from pathlib import Path

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False

try:
    import getdist
    from getdist import plots
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


# Professional color scheme
HICOSMO_COLORS = {
    'primary': '#2E86C1',      # Blue
    'secondary': '#E74C3C',    # Red  
    'accent': '#F39C12',       # Orange
    'success': '#27AE60',      # Green
    'warning': '#F1C40F',      # Yellow
    'dark': '#34495E',         # Dark gray
    'light': '#BDC3C7',       # Light gray
    'purple': '#8E44AD',       # Purple
    'teal': '#16A085',         # Teal
}

# Color cycles for multiple datasets
COLOR_CYCLE = [
    HICOSMO_COLORS['primary'],
    HICOSMO_COLORS['secondary'], 
    HICOSMO_COLORS['accent'],
    HICOSMO_COLORS['success'],
    HICOSMO_COLORS['purple'],
    HICOSMO_COLORS['teal'],
]


def setup_style() -> None:
    """Setup professional matplotlib style inspired by core.py aesthetics."""
    plt.style.use('default')
    
    # Professional settings
    plt.rcParams.update({
        # Figure settings
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        
        # Font settings
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times', 'serif'],
        'font.size': 11,
        'mathtext.fontset': 'cm',
        
        # Axes settings
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,
        'axes.prop_cycle': plt.cycler('color', COLOR_CYCLE),
        
        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        
        # Tick settings
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.minor.size': 2,
        'ytick.minor.size': 2,
        
        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.9,
        'legend.fontsize': 10,
        'legend.numpoints': 1,
        'legend.scatterpoints': 1,
    })


class HiCosmoPlotter:
    """
    Professional plotting class for cosmological analysis.
    
    Provides elegant, publication-ready plots with consistent styling.
    """
    
    def __init__(self, style: str = 'professional'):
        """Initialize plotter with specified style."""
        self.style = style
        setup_style()
        
    def corner_plot(self, 
                   samples: Union[np.ndarray, Dict[str, np.ndarray]],
                   labels: Optional[List[str]] = None,
                   truths: Optional[List[float]] = None,
                   title: Optional[str] = None,
                   save_path: Optional[Union[str, Path]] = None,
                   **kwargs) -> plt.Figure:
        """
        Create elegant corner plot for parameter posterior distributions.
        
        Args:
            samples: MCMC samples array or dictionary
            labels: Parameter labels
            truths: True parameter values (optional)
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional corner plot arguments
            
        Returns:
            matplotlib Figure object
        """
        if not HAS_CORNER:
            raise ImportError("corner package required for corner plots. Install with: pip install corner")
            
        # Handle different input formats
        if isinstance(samples, dict):
            param_names = list(samples.keys())
            samples_array = np.column_stack([samples[name] for name in param_names])
            if labels is None:
                labels = param_names
        else:
            samples_array = samples
            
        # Default settings
        corner_kwargs = {
            'bins': 30,
            'smooth': 1.0,
            'show_titles': True,
            'title_kwargs': {'fontsize': 11},
            'label_kwargs': {'fontsize': 12},
            'color': HICOSMO_COLORS['primary'],
            'hist_kwargs': {'alpha': 0.8},
            'contour_kwargs': {'colors': [HICOSMO_COLORS['primary']], 'linewidths': 1.2},
            'fill_contours': True,
            'levels': [0.68, 0.95],
            'plot_density': False,
            'plot_contours': True,
        }
        corner_kwargs.update(kwargs)
        
        # Create corner plot
        fig = corner.corner(
            samples_array,
            labels=labels,
            truths=truths,
            **corner_kwargs
        )
        
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
            
        # Adjust layout
        fig.subplots_adjust(top=0.92 if title else 0.98)
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
    
    def trace_plot(self,
                  samples: Union[np.ndarray, Dict[str, np.ndarray]],
                  labels: Optional[List[str]] = None,
                  title: Optional[str] = None,
                  save_path: Optional[Union[str, Path]] = None,
                  **kwargs) -> plt.Figure:
        """
        Create trace plots for MCMC diagnostics.
        
        Args:
            samples: MCMC samples (shape: [n_chains, n_samples, n_params] or dict)
            labels: Parameter labels
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
            
        Returns:
            matplotlib Figure object
        """
        # Handle different input formats
        if isinstance(samples, dict):
            param_names = list(samples.keys())
            n_params = len(param_names)
            if labels is None:
                labels = param_names
        else:
            n_params = samples.shape[-1] if samples.ndim > 1 else 1
            if labels is None:
                labels = [f'Parameter {i+1}' for i in range(n_params)]
        
        # Create subplots
        fig, axes = plt.subplots(
            n_params, 1, 
            figsize=(10, 2.5 * n_params),
            sharex=True
        )
        
        if n_params == 1:
            axes = [axes]
            
        # Plot traces
        for i, (ax, label) in enumerate(zip(axes, labels)):
            if isinstance(samples, dict):
                param_samples = samples[param_names[i]]
                if param_samples.ndim == 2:  # Multiple chains
                    for chain_idx in range(param_samples.shape[0]):
                        ax.plot(param_samples[chain_idx], 
                               alpha=0.8, 
                               color=COLOR_CYCLE[chain_idx % len(COLOR_CYCLE)],
                               linewidth=0.8)
                else:  # Single chain
                    ax.plot(param_samples, color=HICOSMO_COLORS['primary'], linewidth=0.8)
            else:
                if samples.ndim == 3:  # [n_chains, n_samples, n_params]
                    for chain_idx in range(samples.shape[0]):
                        ax.plot(samples[chain_idx, :, i],
                               alpha=0.8,
                               color=COLOR_CYCLE[chain_idx % len(COLOR_CYCLE)],
                               linewidth=0.8)
                else:  # [n_samples, n_params] or [n_samples]
                    sample_data = samples[:, i] if samples.ndim > 1 else samples
                    ax.plot(sample_data, color=HICOSMO_COLORS['primary'], linewidth=0.8)
            
            ax.set_ylabel(label, fontsize=12)
            ax.grid(True, alpha=0.3)
            
        # Set x-label for bottom plot
        axes[-1].set_xlabel('Iteration', fontsize=12)
        
        if title:
            fig.suptitle(title, fontsize=14, y=0.98)
            
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
    
    def cosmology_plot(self,
                      z: np.ndarray,
                      functions: Dict[str, np.ndarray],
                      xlabel: str = 'Redshift $z$',
                      ylabel: str = '',
                      title: Optional[str] = None,
                      save_path: Optional[Union[str, Path]] = None,
                      **kwargs) -> plt.Figure:
        """
        Plot cosmological functions (distances, Hubble parameter, etc.).
        
        Args:
            z: Redshift array
            functions: Dictionary of function name -> values
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            save_path: Path to save figure
            **kwargs: Additional plot arguments
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot each function
        for i, (name, values) in enumerate(functions.items()):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            ax.plot(z, values, label=name, color=color, linewidth=2, **kwargs)
            
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
            
        if len(functions) > 1:
            ax.legend(frameon=True, framealpha=0.9)
            
        ax.grid(True, alpha=0.3)
        
        # Smart tick formatting
        ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=8))
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
    
    def comparison_plot(self,
                       x: np.ndarray,
                       datasets: Dict[str, Dict[str, np.ndarray]],
                       xlabel: str = 'x',
                       ylabel: str = 'y',
                       title: Optional[str] = None,
                       save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create comparison plot for multiple models/datasets.
        
        Args:
            x: X-axis values
            datasets: Nested dict {model_name: {'y': values, 'yerr': errors (optional)}}
            xlabel: X-axis label
            ylabel: Y-axis label
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for i, (name, data) in enumerate(datasets.items()):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            
            if 'yerr' in data:
                ax.errorbar(x, data['y'], yerr=data['yerr'], 
                           label=name, color=color, 
                           linewidth=2, capsize=3, alpha=0.8)
            else:
                ax.plot(x, data['y'], label=name, color=color, linewidth=2)
                
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        if title:
            ax.set_title(title, fontsize=14)
            
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig
    
    def hubble_diagram(self,
                      z_obs: np.ndarray,
                      mu_obs: np.ndarray,
                      mu_err: Optional[np.ndarray] = None,
                      z_theory: Optional[np.ndarray] = None,
                      mu_theory: Optional[np.ndarray] = None,
                      model_name: str = 'Model',
                      title: str = 'Hubble Diagram',
                      save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create Hubble diagram plot for Type Ia supernovae.
        
        Args:
            z_obs: Observed redshifts
            mu_obs: Observed distance moduli
            mu_err: Distance modulus errors (optional)
            z_theory: Theory redshift grid (optional)
            mu_theory: Theory distance moduli (optional)
            model_name: Model name for legend
            title: Plot title
            save_path: Path to save figure
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot observations
        if mu_err is not None:
            ax.errorbar(z_obs, mu_obs, yerr=mu_err,
                       fmt='o', color=HICOSMO_COLORS['dark'],
                       alpha=0.6, markersize=3, capsize=0,
                       label='Observations')
        else:
            ax.scatter(z_obs, mu_obs, 
                      c=HICOSMO_COLORS['dark'], 
                      alpha=0.6, s=8, label='Observations')
        
        # Plot theory
        if z_theory is not None and mu_theory is not None:
            ax.plot(z_theory, mu_theory,
                   color=HICOSMO_COLORS['primary'],
                   linewidth=2, label=model_name)
        
        ax.set_xlabel('Redshift $z$', fontsize=12)
        ax.set_ylabel('Distance Modulus $\\mu$', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        ax.legend(frameon=True, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for x-axis if data spans large range
        if np.max(z_obs) / np.min(z_obs) > 100:
            ax.set_xscale('log')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path)
            
        return fig


def quick_corner(samples: Union[np.ndarray, Dict[str, np.ndarray]], 
                labels: Optional[List[str]] = None,
                save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Quick corner plot with default settings."""
    plotter = HiCosmoPlotter()
    return plotter.corner_plot(samples, labels=labels, save_path=save_path)


def quick_trace(samples: Union[np.ndarray, Dict[str, np.ndarray]],
               labels: Optional[List[str]] = None,
               save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Quick trace plot with default settings."""
    plotter = HiCosmoPlotter()
    return plotter.trace_plot(samples, labels=labels, save_path=save_path)


def quick_hubble_diagram(z_obs: np.ndarray, mu_obs: np.ndarray,
                        mu_err: Optional[np.ndarray] = None,
                        save_path: Optional[Union[str, Path]] = None) -> plt.Figure:
    """Quick Hubble diagram plot."""
    plotter = HiCosmoPlotter()
    return plotter.hubble_diagram(z_obs, mu_obs, mu_err=mu_err, save_path=save_path)