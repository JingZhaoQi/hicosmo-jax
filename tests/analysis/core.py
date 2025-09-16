#!/usr/bin/env python3
"""
Cosmological Analysis Core Module

This module provides a unified, object-oriented interface for analyzing 
MCMC results from cosmological parameter estimation.

Classes:
    MCMCAnalysis: Main analysis class for single or multiple chains
    PlottingEngine: Professional plotting with GetDist
    StatisticsEngine: Parameter statistics and convergence diagnostics
    ComparisonEngine: Model comparison and information criteria

Author: JAX-cosmology framework
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import h5py
import warnings

# Optional imports
try:
    from getdist import MCSamples, plots
    GETDIST_AVAILABLE = True
except ImportError:
    GETDIST_AVAILABLE = False
    warnings.warn("GetDist not available. Plotting functionality will be limited.")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

# Import user's style
try:
    from .FigStyle import qstyle, snstyle
    USER_STYLE_AVAILABLE = True
except ImportError:
    USER_STYLE_AVAILABLE = False
    warnings.warn("User FigStyle not available. Using default styling.")


class DataLoader:
    """Handles loading and validation of HDF5 MCMC data."""
    
    @staticmethod
    def load_chains(hdf5_file: str) -> Tuple[Dict[str, np.ndarray], List[str], Dict[str, str], Dict[str, Tuple[float, float]]]:
        """
        Load MCMC chains from HDF5 file.
        
        Args:
            hdf5_file: Path to HDF5 file
            
        Returns:
            Tuple of (samples_dict, param_names, param_labels, param_ranges)
        """
        samples_dict = {}
        param_labels = {}
        param_ranges = {}
        param_names = []
        
        with h5py.File(hdf5_file, 'r') as f:
            # Load chains and get parameter names (original user input names)
            if 'chains' in f:
                for param in f['chains'].keys():
                    chain_data = f['chains'][param][:]
                    if chain_data.ndim > 1:
                        samples_dict[param] = chain_data.flatten()
                    else:
                        samples_dict[param] = chain_data
            
            # Get parameter names in correct order
            param_names = list(samples_dict.keys())
            
            # Load parameter specifications to get original names and LaTeX labels
            if 'parameter_specs' in f:
                for param in f['parameter_specs'].keys():
                    param_group = f['parameter_specs'][param]
                    
                    # Get bounds/ranges
                    if 'bounds' in param_group:
                        bounds = param_group['bounds'][:]
                        param_ranges[param] = (float(bounds[0]), float(bounds[1]))
                    
                    # Get original name if stored (for parameter name management)
                    original_name = param
                    if 'original_name' in param_group.attrs:
                        original_name = param_group.attrs['original_name']
                        # Update samples dict with original name (no conversion!)
                        if param in samples_dict and original_name != param:
                            samples_dict[original_name] = samples_dict.pop(param)
                            # Update param_names list
                            if param in param_names:
                                idx = param_names.index(param)
                                param_names[idx] = original_name
                    
                    # Get LaTeX labels directly from HDF5
                    if 'latex' in param_group.attrs:
                        label = param_group.attrs['latex']
                        # Add units for H_0 parameter automatically
                        if original_name in ['H0', 'H_0'] or label in ['H_0', 'H0', r'H_0']:
                            label = r'H_0 ~[\mathrm{km~s^{-1}~Mpc^{-1}}]'
                        param_labels[original_name] = label
            
            # Fallback to old format
            if 'parameters' in f and 'free' in f['parameters']:
                for param in f['parameters']['free'].keys():
                    if 'latex' in f['parameters']['free'][param].attrs:
                        label = f['parameters']['free'][param].attrs['latex']
                        # Add units for H_0 parameter automatically
                        if param in ['H0', 'H_0'] or label in ['H_0', 'H0', r'H_0']:
                            label = r'H_0 ~[\mathrm{km~s^{-1}~Mpc^{-1}}]'
                        param_labels[param] = label
        
        return samples_dict, param_names, param_labels, param_ranges
    
    @staticmethod
    def validate_data(samples_dict: Dict[str, np.ndarray]) -> bool:
        """Validate loaded data."""
        if not samples_dict:
            raise ValueError("No samples found in HDF5 file")
        
        # Check all parameters have same length
        lengths = [len(samples) for samples in samples_dict.values()]
        if len(set(lengths)) > 1:
            raise ValueError("Parameter chains have different lengths")
        
        return True


class ColorSchemes:
    """Professional color schemes for plotting."""
    
    MODERN = [
        '#2E86AB',  # Modern blue
        '#A23B72',  # Deep magenta  
        '#F18F01',  # Vibrant orange
        '#C73E1D',  # Rich red
        '#592E83',  # Deep purple
        '#0F7173',  # Teal
        '#7B2D26'   # Dark red
    ]
    
    SOPHISTICATED = [
        '#264653',  # Dark green
        '#2A9D8F',  # Teal
        '#E9C46A',  # Yellow
        '#F4A261',  # Orange
        '#E76F51',  # Coral
        '#6A4C93',  # Purple
        '#1D3557'   # Navy
    ]
    
    CLASSIC = [
        '#348ABD',  # User's original
        '#7A68A6', 
        '#E24A33', 
        '#467821',
        '#ffb3a6', 
        '#188487', 
        '#A60628'
    ]
    
    @classmethod
    def get_scheme(cls, name: str) -> List[str]:
        """Get color scheme by name."""
        schemes = {
            'modern': cls.MODERN,
            'sophisticated': cls.SOPHISTICATED,
            'classic': cls.CLASSIC
        }
        return schemes.get(name.lower(), cls.MODERN)


class PlottingEngine:
    """Professional plotting engine using GetDist."""
    
    def __init__(self, color_scheme: str = 'modern', use_latex: bool = True):
        """
        Initialize plotting engine.
        
        Args:
            color_scheme: Color scheme ('modern', 'sophisticated', 'classic')
            use_latex: Enable LaTeX rendering
        """
        if not GETDIST_AVAILABLE:
            raise ImportError("GetDist is required for plotting functionality")
        
        self.color_scheme = color_scheme
        self.colors = ColorSchemes.get_scheme(color_scheme)
        self.use_latex = use_latex
        
        self._setup_style()
    
    def _setup_style(self):
        """Setup matplotlib style."""
        if USER_STYLE_AVAILABLE:
            qstyle(tex=self.use_latex)
        
        # Apply color scheme
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=self.colors)
        
        # Professional settings
        plt.rcParams.update({
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',
            'font.size': 12,
            'axes.labelsize': 16,
            'legend.fontsize': 14,
            'lines.linewidth': 2.0,
            'axes.linewidth': 1.2,
            'legend.frameon': False,
            'axes.grid': True,
            'grid.alpha': 0.3,
        })
        
        if self.use_latex:
            plt.rcParams.update({
                'text.usetex': True,
                'font.family': 'serif',
                'mathtext.fontset': 'cm',
            })
    
    def _optimize_tick_formatting(self, ax):
        """
        Optimize tick label formatting to prevent overlap.
        
        Args:
            ax: matplotlib axis object
        """
        # Get current tick locations and labels
        x_ticks = ax.get_xticks()
        y_ticks = ax.get_yticks()
        
        # Determine appropriate number formatting
        x_range = np.ptp(x_ticks) if len(x_ticks) > 1 else 1
        y_range = np.ptp(y_ticks) if len(y_ticks) > 1 else 1
        
        # Smart formatting based on range and typical values
        def smart_format(value, value_range):
            """Format numbers to minimize width while maintaining precision."""
            if abs(value) == 0:
                return '0'
            elif abs(value) >= 1000:
                return f'{value:.0f}'
            elif abs(value) >= 100:
                return f'{value:.1f}'
            elif abs(value) >= 10:
                return f'{value:.2f}'
            elif abs(value) >= 1:
                return f'{value:.3f}'
            elif abs(value) >= 0.1:
                return f'{value:.3f}'
            elif abs(value) >= 0.01:
                return f'{value:.4f}'
            else:
                # For very small numbers, use scientific notation
                return f'{value:.1e}'
        
        # Apply smart formatting using formatter instead of direct label setting
        try:
            # Create custom formatters
            def x_formatter(value, pos):
                return smart_format(value, x_range)
            
            def y_formatter(value, pos):
                return smart_format(value, y_range)
            
            # Apply formatters
            ax.xaxis.set_major_formatter(ticker.FuncFormatter(x_formatter))
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(y_formatter))
            
        except Exception:
            # If custom formatting fails, use simple precision formatting
            try:
                ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
                ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
            except Exception:
                pass  # Keep default formatting
    
    def corner_plot(self, 
                   samples_dict: Dict[str, np.ndarray],
                   param_labels: Dict[str, str],
                   param_ranges: Dict[str, Tuple[float, float]],
                   params: Optional[List[str]] = None,
                   title: Optional[str] = None,
                   legend_labels: Optional[List[str]] = None,
                   filename: Optional[str] = None) -> None:
        """
        Create GetDist corner plot.
        
        Args:
            samples_dict: Dictionary of parameter samples
            param_labels: LaTeX labels for parameters
            param_ranges: Parameter ranges
            params: Parameters to plot (default: all)
            title: Optional plot title
            legend_labels: Optional legend labels
            filename: Output filename
        """
        if params is None:
            params = list(samples_dict.keys())
        
        # Prepare data
        param_names = [p for p in params if p in samples_dict]
        n_params = len(param_names)
        n_samples = len(samples_dict[param_names[0]])
        
        samples_array = np.zeros((n_samples, n_params))
        for i, param in enumerate(param_names):
            samples_array[:, i] = samples_dict[param]
        
        # Prepare labels (clean LaTeX from HDF5 and add units for H_0)
        labels = []
        for param in param_names:
            label = param_labels.get(param, param)
            if '\\\\' in label:
                label = label.replace('\\\\', '\\')
            
            # Add units for H_0 parameter
            if param in ['H0', 'H_0'] or label in ['H_0', 'H0', r'H_0']:
                label = r'H_0 ~[\mathrm{km~s^{-1}~Mpc^{-1}}]'
            
            labels.append(label)
        
        # Prepare ranges
        ranges = {}
        for param in param_names:
            if param in param_ranges:
                ranges[param] = param_ranges[param]
        
        # Create MCSamples
        mc_samples = MCSamples(
            samples=samples_array,
            names=param_names,
            labels=labels,
            ranges=ranges
        )
        mc_samples.updateSettings({'ignore_rows': 0.2})
        
        # Create plot
        g = plots.get_subplot_plotter(width_inch=8)
        g.settings.axes_fontsize = 12  # Slightly smaller to reduce overlap
        g.settings.lab_fontsize = 14
        g.settings.legend_fontsize = 12
        g.settings.axes_labelsize = 12
        
        # Additional GetDist settings for better layout
        g.settings.figure_legend_frame = False  # No frame around legend
        
        # Apply colors
        contour_colors = [self.colors[0]]
        line_args = {'color': self.colors[0], 'lw': 2}
        
        g.triangle_plot(
            mc_samples,
            filled=True,
            contour_colors=contour_colors,
            line_args=line_args,
            legend_labels=legend_labels
        )
        
        # Fix tick density and overlap issues after plot creation
        for ax in g.fig.get_axes():
            if hasattr(ax, 'xaxis') and hasattr(ax, 'yaxis'):
                # Get axis position and size to determine appropriate tick density
                bbox = ax.get_position()
                width_inch = bbox.width * g.fig.get_figwidth()
                height_inch = bbox.height * g.fig.get_figheight()
                
                # Adaptive tick density based on subplot size
                if width_inch < 1.5:  # Small subplot
                    x_nbins = 3
                elif width_inch < 2.5:  # Medium subplot
                    x_nbins = 4
                else:  # Large subplot
                    x_nbins = 5
                
                if height_inch < 1.5:  # Small subplot
                    y_nbins = 3
                elif height_inch < 2.5:  # Medium subplot
                    y_nbins = 4
                else:  # Large subplot
                    y_nbins = 5
                
                # Use MaxNLocator with auto-pruning to prevent overlap
                ax.xaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=x_nbins, prune='both', min_n_ticks=2)
                )
                ax.yaxis.set_major_locator(
                    ticker.MaxNLocator(nbins=y_nbins, prune='both', min_n_ticks=2)
                )
                
                # Additional formatting to prevent overlap
                ax.tick_params(axis='x', labelsize=10, pad=2)
                ax.tick_params(axis='y', labelsize=10, pad=2)
                
                # For very small subplots, rotate x labels if needed
                if width_inch < 1.8:
                    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
                
                # Ensure labels don't extend beyond plot area
                ax.xaxis.set_tick_params(which='major', direction='in')
                ax.yaxis.set_tick_params(which='major', direction='in')
                
                # Smart number formatting to reduce overlap
                self._optimize_tick_formatting(ax)
        
        # Add title if specified
        if title:
            plt.suptitle(title, fontsize=18, y=0.98)
        
        # Save if filename provided
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()
    
    def trace_plot(self,
                  samples_dict: Dict[str, np.ndarray],
                  param_labels: Dict[str, str],
                  params: Optional[List[str]] = None,
                  filename: Optional[str] = None) -> None:
        """
        Create trace plots for convergence diagnostics.
        
        Args:
            samples_dict: Dictionary of parameter samples
            param_labels: LaTeX labels for parameters
            params: Parameters to plot (default: all)
            filename: Output filename
        """
        if params is None:
            params = list(samples_dict.keys())
        
        n_params = len(params)
        fig, axes = plt.subplots(n_params, 2, figsize=(12, 3*n_params))
        if n_params == 1:
            axes = axes.reshape(1, -1)
        
        for i, param in enumerate(params):
            if param not in samples_dict:
                continue
                
            samples = samples_dict[param]
            label = param_labels.get(param, param)
            
            # Clean LaTeX label for matplotlib
            if '\\\\' in label:
                label = label.replace('\\\\', '\\')
            
            # Trace plot
            axes[i, 0].plot(samples, color=self.colors[0], alpha=0.8, linewidth=1)
            # Use LaTeX label if available and LaTeX is enabled
            if self.use_latex and '\\' in label:
                axes[i, 0].set_ylabel(f'${label}$')
                axes[i, 0].set_title(f'Trace: ${label}$')
            else:
                axes[i, 0].set_ylabel(label)
                axes[i, 0].set_title(f'Trace: {label}')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Running mean
            running_mean = np.cumsum(samples) / np.arange(1, len(samples) + 1)
            axes[i, 1].plot(running_mean, color=self.colors[1], linewidth=2)
            axes[i, 1].set_ylabel('Running Mean')
            axes[i, 1].set_title('Convergence')
            axes[i, 1].grid(True, alpha=0.3)
        
        axes[-1, 0].set_xlabel('Iteration')
        axes[-1, 1].set_xlabel('Iteration')
        
        plt.tight_layout()
        
        if filename:
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filename, dpi=300, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            plt.close()


class StatisticsEngine:
    """Parameter statistics and convergence diagnostics."""
    
    @staticmethod
    def parameter_summary(samples_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate parameter summary statistics.
        
        Args:
            samples_dict: Dictionary of parameter samples
            
        Returns:
            Dictionary with statistics for each parameter
        """
        summary = {}
        
        for param, samples in samples_dict.items():
            summary[param] = {
                'mean': float(np.mean(samples)),
                'std': float(np.std(samples)),
                'median': float(np.median(samples)),
                'q16': float(np.percentile(samples, 16)),
                'q84': float(np.percentile(samples, 84)),
                'q2p5': float(np.percentile(samples, 2.5)),
                'q97p5': float(np.percentile(samples, 97.5)),
                'min': float(np.min(samples)),
                'max': float(np.max(samples))
            }
        
        return summary
    
    @staticmethod
    def effective_sample_size(samples: np.ndarray) -> float:
        """Calculate effective sample size."""
        n = len(samples)
        
        # Simple autocorrelation-based estimate
        autocorr = np.correlate(samples - np.mean(samples), 
                               samples - np.mean(samples), mode='full')
        autocorr = autocorr[len(autocorr)//2:] / autocorr[len(autocorr)//2]
        
        # Find first negative value or where autocorr < 0.01
        tau_int = 1.0
        for i in range(1, min(len(autocorr), n//4)):
            if autocorr[i] <= 0.01:
                tau_int = i
                break
        
        return n / (2 * tau_int + 1)
    
    @staticmethod
    def convergence_diagnostics(samples_dict: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Calculate convergence diagnostics.
        
        Args:
            samples_dict: Dictionary of parameter samples
            
        Returns:
            Dictionary with diagnostics for each parameter
        """
        diagnostics = {}
        
        for param, samples in samples_dict.items():
            n_eff = StatisticsEngine.effective_sample_size(samples)
            
            diagnostics[param] = {
                'n_samples': len(samples),
                'n_effective': n_eff,
                'efficiency': n_eff / len(samples)
            }
        
        return diagnostics


class MCMCAnalysis:
    """
    Main analysis class for MCMC results.
    
    This class provides a unified interface for analyzing single or multiple
    MCMC chains with professional plotting, statistics, and comparison tools.
    """
    
    def __init__(self, 
                 hdf5_files: Union[str, List[str]], 
                 names: Optional[List[str]] = None,
                 color_scheme: str = 'modern',
                 use_latex: bool = True):
        """
        Initialize MCMC analysis.
        
        Args:
            hdf5_files: HDF5 filename(s) - automatically looks in chains/ directory
            names: Names for each dataset (for comparison plots)
            color_scheme: Color scheme ('modern', 'sophisticated', 'classic')
            use_latex: Enable LaTeX rendering for parameter labels (default: True)
        """
        if isinstance(hdf5_files, str):
            hdf5_files = [hdf5_files]
        
        # Auto-prepend chains/ directory if not absolute path
        full_paths = []
        for hdf5_file in hdf5_files:
            if not Path(hdf5_file).is_absolute() and not hdf5_file.startswith('chains/'):
                full_paths.append(f'chains/{hdf5_file}')
            else:
                full_paths.append(hdf5_file)
        
        self.hdf5_files = full_paths
        self.names = names or [f"Chain {i+1}" for i in range(len(full_paths))]
        self.color_scheme = color_scheme
        
        # Load data
        self.datasets = []
        for hdf5_file in full_paths:
            samples_dict, param_names, param_labels, param_ranges = DataLoader.load_chains(hdf5_file)
            DataLoader.validate_data(samples_dict)
            
            self.datasets.append({
                'samples': samples_dict,
                'names': param_names,
                'labels': param_labels,
                'ranges': param_ranges,
                'file': hdf5_file
            })
        
        # Initialize engines
        self.plotter = PlottingEngine(color_scheme=color_scheme, use_latex=use_latex)
        self.stats = StatisticsEngine()
        
        print(f"Loaded {len(self.datasets)} dataset(s)")
        for i, dataset in enumerate(self.datasets):
            n_params = len(dataset['samples'])
            n_samples = len(next(iter(dataset['samples'].values())))
            param_names_str = ', '.join(dataset['names'][:3])  # Show first 3 parameter names
            if len(dataset['names']) > 3:
                param_names_str += '...'
            print(f"  {self.names[i]}: {n_params} parameters ({param_names_str}), {n_samples} samples")
    
    def corner_plot(self, 
                   dataset_index: int = 0,
                   params: Optional[List[str]] = None,
                   title: Optional[str] = None,
                   filename: Optional[str] = None) -> None:
        """
        Create corner plot for specified dataset.
        
        Args:
            dataset_index: Index of dataset to plot
            params: Parameters to include (use parameter names like 'H0', 'Omega_m')
            title: Plot title
            filename: Output filename (optional, defaults to figures/{h5_name}.pdf)
        """
        dataset = self.datasets[dataset_index]
        
        # If params specified, use parameter names; otherwise use all available
        if params is None:
            params = dataset['names']
        
        # Auto-generate filename if not provided
        if filename is None:
            h5_path = Path(dataset['file'])
            h5_name = h5_path.stem  # 去掉.h5扩展名
            filename = f'figures/{h5_name}.pdf'
        
        self.plotter.corner_plot(
            samples_dict=dataset['samples'],
            param_labels=dataset['labels'],
            param_ranges=dataset['ranges'],
            params=params,
            title=title,
            filename=filename
        )
    
    def comparison_plot(self,
                       params: Optional[List[str]] = None,
                       title: Optional[str] = None,
                       filename: Optional[str] = None) -> None:
        """
        Create comparison corner plot for multiple datasets.
        
        Args:
            params: Parameters to include (use parameter names like 'H0', 'Omega_m')
            title: Plot title
            filename: Output filename
        """
        if len(self.datasets) == 1:
            print("Only one dataset available. Use corner_plot() instead.")
            return
        
        # For now, plot first dataset with legend showing comparison
        # TODO: Implement proper multi-dataset comparison
        dataset = self.datasets[0]
        
        # If params specified, use parameter names; otherwise use all available
        if params is None:
            params = dataset['names']
        
        self.plotter.corner_plot(
            samples_dict=dataset['samples'],
            param_labels=dataset['labels'],
            param_ranges=dataset['ranges'],
            params=params,
            title=title,
            legend_labels=self.names,
            filename=filename
        )
    
    def trace_plot(self,
                  dataset_index: int = 0,
                  params: Optional[List[str]] = None,
                  filename: Optional[str] = None) -> None:
        """
        Create trace plots for convergence diagnostics.
        
        Args:
            dataset_index: Index of dataset to plot
            params: Parameters to include (use parameter names like 'H0', 'Omega_m')
            filename: Output filename
        """
        dataset = self.datasets[dataset_index]
        
        # If params specified, use parameter names; otherwise use all available
        if params is None:
            params = dataset['names']
        
        self.plotter.trace_plot(
            samples_dict=dataset['samples'],
            param_labels=dataset['labels'],
            params=params,
            filename=filename
        )
    
    def summary(self, dataset_index: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Get parameter summary statistics.
        
        Args:
            dataset_index: Index of dataset
            
        Returns:
            Parameter summary statistics
        """
        dataset = self.datasets[dataset_index]
        return self.stats.parameter_summary(dataset['samples'])
    
    def diagnostics(self, dataset_index: int = 0) -> Dict[str, Dict[str, float]]:
        """
        Get convergence diagnostics.
        
        Args:
            dataset_index: Index of dataset
            
        Returns:
            Convergence diagnostics
        """
        dataset = self.datasets[dataset_index]
        return self.stats.convergence_diagnostics(dataset['samples'])
    
    def print_summary(self, dataset_index: int = 0) -> None:
        """Print formatted parameter summary."""
        summary = self.summary(dataset_index)
        diagnostics = self.diagnostics(dataset_index)
        
        print(f"\nParameter Summary for {self.names[dataset_index]}:")
        print("=" * 80)
        print(f"{'Parameter':<12} {'Mean':<12} {'Std':<12} {'68% CI':<20} {'95% CI':<20} {'N_eff':<8}")
        print("-" * 80)
        
        for param in summary:
            stats = summary[param]
            diag = diagnostics[param]
            
            ci68 = f"[{stats['q16']:.4f}, {stats['q84']:.4f}]"
            ci95 = f"[{stats['q2p5']:.4f}, {stats['q97p5']:.4f}]"
            
            print(f"{param:<12} {stats['mean']:<12.4f} {stats['std']:<12.4f} "
                  f"{ci68:<20} {ci95:<20} {diag['n_effective']:<8.0f}")