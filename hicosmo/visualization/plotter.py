#!/usr/bin/env python3
"""
HIcosmo Unified Plotter Class
=============================

Single entry point for all visualization needs:
- MCMC chain visualization (corner plots, traces, marginals)
- Multi-chain comparison
- Fisher forecast contours

Core Usage (3 lines):
```python
from hicosmo.visualization import Plotter

plotter = Plotter('results/mcmc_chain.pkl')
plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')
```

Author: Jingzhao Qi
"""

from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from ..samplers.inference import MCMC
import numpy as np
import matplotlib.pyplot as plt

try:
    from getdist import plots, MCSamples
    from getdist.gaussian_mixtures import GaussianND
    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    raise ImportError("GetDist is required. Install with: pip install getdist")

from .plotting import (
    _apply_qijing_style,
    _prepare_getdist_samples,
    _prepare_latex_labels,
    _optimize_ticks,
    _safe_tight_layout,
    _get_results_dir,
    _load_chain_simple,
    MODERN_COLORS,
    CLASSIC_COLORS,
)
from ..samplers.constants import get_chains_dir


class Plotter:
    """
    HIcosmo unified visualization interface.

    Supports multiple data input formats:
    - Chain name: 'my_chain' (auto-loads from mcmc_chains/my_chain.h5)
    - Chain names list: ['chain1', 'chain2'] (multi-chain mode)
    - Dict: {'H0': arr, 'Omega_m': arr}
    - NumPy array: np.array((N, D))
    - MCSamples: GetDist native object

    Examples
    --------
    >>> # Single chain (simplest!)
    >>> plotter = Plotter('my_chain')
    >>> plotter.corner(['H0', 'Omega_m'], filename='corner.pdf')

    >>> # Multi-chain comparison
    >>> plotter = Plotter(
    ...     ['low_noise', 'high_noise'],
    ...     labels=['Low Noise', 'High Noise']
    ... )
    >>> plotter.corner(['a', 'b', 'c'], filename='comparison.pdf')

    >>> # From dict
    >>> plotter = Plotter({'H0': samples_h0, 'Omega_m': samples_om})

    >>> # Fisher forecast
    >>> plotter = Plotter.from_fisher(mean, cov, param_names=['w0', 'wa'])
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def __init__(
        self,
        data: Union[str, Path, np.ndarray, dict, List, MCSamples],
        *,
        labels: Optional[Union[List[str], Dict[str, str]]] = None,
        ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        chain_labels: Optional[List[str]] = None,
        style: str = 'modern',
        results_dir: Union[str, Path] = 'results',
        **kwargs,
    ):
        """
        Initialize Plotter with data.

        Parameters
        ----------
        data : various
            Chain data in various formats (see class docstring)
        labels : list or dict, optional
            Parameter LaTeX labels. If None, auto-extracted from file metadata.
        ranges : dict, optional
            Parameter plotting ranges {name: (min, max)}.
        chain_labels : list of str, optional
            Labels for multi-chain mode ['Planck', 'SH0ES', ...]
        style : str
            Color scheme ('modern' or 'classic')
        results_dir : str or Path
            Directory for saving plots
        """
        if 'cosmology' in kwargs:
            import warnings
            warnings.warn(
                "The 'cosmology' parameter is deprecated.",
                DeprecationWarning,
                stacklevel=2
            )

        self.style = style
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Handle smart label detection for multi-chain mode
        if self._is_chain_name_list(data) and isinstance(labels, list):
            chain_labels, labels = labels, None

        self._user_labels = labels
        self._user_ranges = ranges
        self._chain_labels = chain_labels
        self._file_metadata: Dict[str, Any] = {}
        self._samples_list: List[MCSamples] = []
        self._is_multichain = False
        self._param_names: List[str] = []
        self._raw_samples: Optional[dict] = None
        self._chain_name: Optional[str] = None
        self._chain_names: List[str] = []  # For multichain mode

        self._load_data(data)

    # =========================================================================
    # Class Methods (Factory)
    # =========================================================================

    @classmethod
    def from_fisher(
        cls,
        mean: Union[np.ndarray, List[float]],
        covariance: Union[np.ndarray, List[List[float]]],
        param_names: List[str],
        *,
        nsample: int = 200_000,
        **kwargs,
    ) -> 'Plotter':
        """Create Plotter from Fisher matrix results."""
        mean_arr = np.asarray(mean, dtype=float)
        cov_arr = np.asarray(covariance, dtype=float)

        if mean_arr.ndim != 1:
            raise ValueError("mean must be a 1-D array")
        if cov_arr.ndim != 2 or cov_arr.shape[0] != cov_arr.shape[1]:
            raise ValueError("covariance must be a square 2-D array")
        if cov_arr.shape[0] != mean_arr.shape[0]:
            raise ValueError("Mean and covariance dimensions do not match")
        if len(param_names) != mean_arr.shape[0]:
            raise ValueError("Number of parameter names must equal mean size")

        latex_labels = _prepare_latex_labels(param_names)
        gaussian = GaussianND(mean_arr, cov_arr, names=param_names, labels=latex_labels)
        samples = gaussian.MCSamples(nsample)
        return cls(samples, **kwargs)

    @classmethod
    def from_mcmc(cls, mcmc: 'MCMC', **kwargs) -> 'Plotter':
        """Create Plotter from MCMC object."""
        samples = mcmc.samples
        labels = mcmc.param_config.get_labels() if hasattr(mcmc, 'param_config') else None
        ranges = mcmc.param_config.get_bounds() if hasattr(mcmc, 'param_config') else None
        return cls(samples, labels=labels, ranges=ranges, **kwargs)

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def param_names(self) -> List[str]:
        """Get parameter names."""
        return self._param_names.copy()

    @property
    def is_multichain(self) -> bool:
        """Check if this is multi-chain data."""
        return self._is_multichain

    @property
    def _colors(self) -> List[str]:
        """Get color palette based on style."""
        return MODERN_COLORS if self.style == 'modern' else CLASSIC_COLORS

    # =========================================================================
    # Public Plotting Methods
    # =========================================================================

    def corner(
        self,
        params: Optional[Union[List[str], List[int]]] = None,
        *,
        filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> plt.Figure:
        """
        Create corner plot.

        Parameters
        ----------
        params : list, optional
            Parameters to plot. None = all parameters.
        filename : str or Path, optional
            Save filename (auto-saves to results_dir)
        **kwargs
            Additional arguments passed to GetDist
        """
        samples_list = self._select_params(params) if params else self._samples_list
        plotter = self._create_getdist_plotter(width_inch=8)

        n_chains = len(samples_list)
        colors = self._colors[:n_chains]

        plot_kwargs = {
            'filled': True,
            'contour_colors': colors,
            **kwargs
        }

        if self._is_multichain:
            plot_kwargs['line_args'] = [{'color': c, 'lw': 2} for c in colors]
            plot_kwargs['legend_labels'] = self._get_legend_labels(samples_list)
            plot_kwargs['legend_loc'] = 'upper right'
        else:
            plot_kwargs['line_args'] = {'color': colors[0], 'lw': 2}

        plotter.triangle_plot(samples_list, **plot_kwargs)
        return self._finalize_plot(plotter.fig, filename, 'Corner plot')

    def plot_1d(
        self,
        param: Union[str, int],
        *,
        filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Create 1D marginal distribution plot for a single parameter."""
        param_name = self._param_names[param] if isinstance(param, int) else param
        plotter = self._create_getdist_plotter(width_inch=6, single=True)

        plotter.plot_1d(
            self._samples_list,
            param_name,
            colors=self._colors[:len(self._samples_list)],
            **kwargs
        )
        self._add_multichain_legend(plotter)
        return self._finalize_plot(plotter.fig, filename, '1D plot')

    def plot_2d(
        self,
        param1: Union[str, int],
        param2: Union[str, int],
        *,
        filename: Optional[Union[str, Path]] = None,
        filled: bool = True,
        **kwargs,
    ) -> plt.Figure:
        """Create 2D contour plot for two parameters."""
        p1 = self._param_names[param1] if isinstance(param1, int) else param1
        p2 = self._param_names[param2] if isinstance(param2, int) else param2
        plotter = self._create_getdist_plotter(width_inch=6, single=True)

        plotter.plot_2d(
            self._samples_list,
            p1, p2,
            filled=filled,
            colors=self._colors[:len(self._samples_list)],
            **kwargs
        )
        self._add_multichain_legend(plotter)
        return self._finalize_plot(plotter.fig, filename, '2D contour plot')

    def traces(
        self,
        params: Optional[Union[List[str], List[int]]] = None,
        *,
        filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Create chain trace plots for convergence diagnostics."""
        _apply_qijing_style()

        selected = self._resolve_param_names(params) if params else self._param_names[:6]
        n_params = len(selected)
        fig, axes = plt.subplots(n_params, 1, figsize=(10, 2*n_params))
        axes = [axes] if n_params == 1 else axes

        for i, param_name in enumerate(selected):
            param_obj = self._samples_list[0].getParamNames().parWithName(param_name)
            label = f"${param_obj.label}$" if param_obj.label else f"${param_name}$"

            for j, samples in enumerate(self._samples_list):
                idx = samples.getParamNames().list().index(param_name)
                axes[i].plot(samples.samples[:, idx], color=self._colors[j], alpha=0.8, lw=1.5,
                           label=samples.label if self._is_multichain else None)

            axes[i].set_ylabel(label)
            axes[i].grid(True, alpha=0.3)
            if i == 0 and self._is_multichain:
                axes[i].legend(frameon=False)

        axes[-1].set_xlabel('Sample Index')
        return self._finalize_plot(fig, filename, 'Chain traces', use_getdist=False)

    def marginals(
        self,
        params: Optional[Union[List[str], List[int]]] = None,
        *,
        filename: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> plt.Figure:
        """Create 1D marginal distribution plots."""
        samples_list = self._select_params(params) if params else self._samples_list
        plotter = self._create_getdist_plotter(width_inch=8, single=True)

        plotter.plots_1d(
            samples_list,
            colors=self._colors[:len(samples_list)],
            legend_labels=self._get_legend_labels(samples_list),
            **kwargs
        )
        return self._finalize_plot(plotter.fig, filename, '1D marginals')

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all parameters."""
        samples = self._samples_list[0].samples
        n_params = min(len(self._param_names), samples.shape[1])
        return {
            self._param_names[i]: {
                'mean': float(np.mean(samples[:, i])),
                'std': float(np.std(samples[:, i])),
                'median': float(np.median(samples[:, i])),
                'q16': float(np.percentile(samples[:, i], 16)),
                'q84': float(np.percentile(samples[:, i], 84)),
            }
            for i in range(n_params)
        }

    def report(
        self,
        filename: Optional[Union[str, Path]] = None,
        save_corner: bool = True,
        corner_params: Optional[List[str]] = None,
    ):
        """Generate comprehensive analysis report."""
        if not self._samples_list:
            print("No samples loaded")
            return

        # Handle multichain mode - generate individual reports + combined corner
        if self._is_multichain and self._chain_names:
            # Generate individual report for each chain
            for i, chain_name in enumerate(self._chain_names):
                self._generate_single_report(
                    chain_name=chain_name,
                    samples_idx=i,
                    save_corner=False,  # Don't save individual corners
                    corner_params=corner_params,
                )
            # Generate ONE combined corner plot for comparison
            if save_corner:
                params = corner_params or [p for p in self._param_names if p != 'log_likelihood']
                combined_name = '+'.join(self._chain_names)
                self.corner(params, filename=f"{combined_name}_corner.pdf")
            return

        # Single chain mode
        chain_name = self._chain_name or "unknown"
        self._generate_single_report(
            chain_name=chain_name,
            samples_idx=0,
            filename=filename,
            save_corner=save_corner,
            corner_params=corner_params,
        )

    def _generate_single_report(
        self,
        chain_name: str,
        samples_idx: int = 0,
        filename: Optional[Union[str, Path]] = None,
        save_corner: bool = True,
        corner_params: Optional[List[str]] = None,
    ):
        """Generate report for a single chain."""
        chain_path = self._find_chain_path(chain_name)
        metadata = self._load_chain_metadata(chain_path) if chain_path else {}

        # Use specific samples for this chain
        original_samples = self._samples_list
        original_multichain = self._is_multichain
        if self._is_multichain:
            self._samples_list = [original_samples[samples_idx]]
            self._is_multichain = False

        lines = self._generate_report_header(chain_name, metadata)
        lines += self._generate_constraints_section(metadata)
        lines += self._generate_diagnostics_section()
        lines += self._generate_model_selection_section(chain_path)
        lines += ["", "---", "*Report generated by HIcosmo*"]

        # Restore samples list
        self._samples_list = original_samples
        self._is_multichain = original_multichain

        # Save report
        if filename is None:
            filename = f"{chain_name}_report.md"
        save_path = self.results_dir / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text('\n'.join(lines))
        print(f"Report saved to: {save_path}")

        # Auto-save corner plot (only for single chain mode)
        if save_corner and not original_multichain:
            params = corner_params or [p for p in self._param_names if p != 'log_likelihood']
            self.corner(params, filename=f"{chain_name}_corner.pdf")

    def __repr__(self) -> str:
        chain_info = f"{len(self._samples_list)} chains" if self._is_multichain else "single chain"
        return f"Plotter({chain_info}, {len(self._param_names)} parameters)"

    # =========================================================================
    # Private Helper Methods - Data Loading
    # =========================================================================

    def _is_chain_name_list(self, data) -> bool:
        """Check if data is a list of chain names (strings without extensions)."""
        return (
            isinstance(data, (list, tuple)) and len(data) > 0 and
            all(isinstance(item, str) and not Path(item).suffix for item in data)
        )

    def _load_data(self, data):
        """Process input data into MCSamples list."""
        # Single chain name
        if isinstance(data, str) and not Path(data).suffix:
            self._chain_name = data
            samples_dict, file_ranges, file_labels = self._load_h5_chain(data)
            self._user_ranges = self._user_ranges or file_ranges
            self._user_labels = self._user_labels or file_labels
            self._raw_samples = samples_dict
            data = samples_dict

        # Multiple chain names
        if self._is_chain_name_list(data):
            self._load_multichain_from_names(data)
            return

        # Multiple chains (dicts/arrays)
        if isinstance(data, (list, tuple)) and len(data) > 1:
            if not self._is_metadata_tuple(data):
                self._load_multichain_from_data(data)
                return

        # Single chain (various formats)
        self._load_single_chain(data)

    def _load_h5_chain(self, chain_name: str) -> Tuple[dict, Dict, Dict]:
        """Load samples, ranges and labels from mcmc_chains/{chain_name}.h5"""
        import h5py
        import json

        h5_path = get_chains_dir() / f'{chain_name}.h5'
        if not h5_path.exists():
            raise FileNotFoundError(f"Chain not found: {h5_path}")

        samples, ranges, labels = {}, {}, {}
        with h5py.File(h5_path, 'r') as f:
            if 'samples' in f:
                for param_name in f['samples'].keys():
                    samples[param_name] = np.array(f['samples'][param_name])

            if 'metadata' in f:
                if 'parameter_config' in f['metadata']:
                    try:
                        config = json.loads(f['metadata']['parameter_config'][()].decode('utf-8'))
                        for pname, pinfo in config.get('parameters', {}).items():
                            bounds = pinfo.get('bounds')
                            if bounds and len(bounds) >= 2:
                                ranges[pname] = (float(bounds[0]), float(bounds[1]))
                    except (json.JSONDecodeError, KeyError, AttributeError):
                        pass

                if 'labels' in f['metadata']:
                    for pname in f['metadata']['labels'].attrs.keys():
                        labels[pname] = f['metadata']['labels'].attrs[pname]

        if not samples:
            raise ValueError(f"No samples found in {h5_path}")
        return samples, ranges, labels

    def _load_multichain_from_names(self, chain_names: List[str]):
        """Load multiple chains from chain names."""
        self._chain_names = list(chain_names)
        loaded = [self._load_h5_chain(name) for name in chain_names]
        all_samples = [d[0] for d in loaded]
        effective_ranges = self._user_ranges or loaded[0][1] or self._compute_unified_ranges(all_samples)
        effective_labels = self._user_labels or loaded[0][2]
        self._build_multichain_samples(all_samples, effective_labels, effective_ranges)

    def _load_multichain_from_data(self, data_list):
        """Load multiple chains from data list (dicts/arrays)."""
        all_samples = [d for d in data_list if isinstance(d, (dict, np.ndarray))]
        unified_ranges = self._user_ranges or self._compute_unified_ranges(all_samples)
        self._build_multichain_samples(data_list, self._user_labels, unified_ranges)

    def _build_multichain_samples(self, data_list, labels, ranges):
        """Build MCSamples list from multiple chain data."""
        self._is_multichain = True
        for i, chain_data in enumerate(data_list):
            samples = _prepare_getdist_samples(chain_data, labels=labels, ranges=ranges)
            samples.label = self._chain_labels[i] if self._chain_labels and i < len(self._chain_labels) else f'Chain {i+1}'
            self._samples_list.append(samples)
        self._param_names = self._samples_list[0].getParamNames().list()

    def _load_single_chain(self, data):
        """Load single chain from various formats."""
        if isinstance(data, (str, Path)):
            loaded = _load_chain_simple(str(data))
            if isinstance(loaded, tuple) and len(loaded) == 2:
                data, self._file_metadata = loaded
            else:
                data = loaded

        if self._is_metadata_tuple(data):
            data, self._file_metadata = data

        effective_labels = self._user_labels or self._file_metadata.get('labels')
        effective_ranges = self._user_ranges or self._file_metadata.get('ranges')

        samples = _prepare_getdist_samples(data, labels=effective_labels, ranges=effective_ranges)
        self._samples_list = [samples]
        self._param_names = samples.getParamNames().list()

    def _is_metadata_tuple(self, data) -> bool:
        """Check if data is (samples, metadata) tuple."""
        return (isinstance(data, tuple) and len(data) == 2 and
                isinstance(data[1], dict) and ('labels' in data[1] or 'ranges' in data[1]))

    def _compute_unified_ranges(self, all_samples: List) -> Dict[str, Tuple[float, float]]:
        """Compute unified ranges from all chains."""
        if not all_samples or not isinstance(all_samples[0], dict):
            return {}

        param_names = list(all_samples[0].keys())
        unified = {}
        for param in param_names:
            values = []
            for chain_data in all_samples:
                if isinstance(chain_data, dict) and param in chain_data:
                    values.extend(chain_data[param].flatten())
            if values:
                arr = np.array(values)
                lo, hi = float(np.min(arr)), float(np.max(arr))
                pad = 0.05 * (hi - lo) if hi > lo else 0.1
                unified[param] = (lo - pad, hi + pad)
        return unified

    # =========================================================================
    # Private Helper Methods - Plotting
    # =========================================================================

    def _create_getdist_plotter(self, width_inch: int = 8, single: bool = False):
        """Create and configure GetDist plotter."""
        _apply_qijing_style()
        plotter = plots.get_single_plotter(width_inch=width_inch) if single else plots.get_subplot_plotter(width_inch=width_inch)
        plotter.settings.axes_fontsize = 12
        plotter.settings.lab_fontsize = 14
        plotter.settings.legend_fontsize = 12 if not single else 9
        plotter.settings.figure_legend_frame = False
        return plotter

    def _finalize_plot(self, fig: plt.Figure, filename, plot_type: str, use_getdist: bool = True) -> plt.Figure:
        """Apply final optimizations and save."""
        if use_getdist:
            _optimize_ticks(fig)
        _safe_tight_layout(fig)

        if filename:
            save_path = self.results_dir / filename
            if not save_path.suffix:
                save_path = save_path.with_suffix('.pdf')
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            print(f"{plot_type} saved to: {save_path}")
        return fig

    def _add_multichain_legend(self, plotter):
        """Add legend for multi-chain plots."""
        labels = self._get_legend_labels(self._samples_list)
        if not labels:
            return
        plotter.add_legend(labels, legend_loc='upper right')
        for ax in plotter.fig.axes:
            legend = ax.get_legend()
            if legend:
                legend.set_frame_on(False)
                for text in legend.get_texts():
                    text.set_fontsize(9)

    def _get_legend_labels(self, samples_list) -> Optional[List[str]]:
        """Get legend labels for samples list."""
        if self._is_multichain and any(s.label for s in samples_list):
            return [s.label for s in samples_list]
        return None

    def _resolve_param_names(self, params: Union[List[str], List[int]]) -> List[str]:
        """Convert parameter indices to names if needed."""
        if not params:
            return self._param_names
        if isinstance(params[0], int):
            return [self._param_names[i] for i in params if 0 <= i < len(self._param_names)]
        return list(params)

    def _select_params(self, params: Union[List[str], List[int]]) -> List[MCSamples]:
        """Create MCSamples subset with only requested parameters."""
        selected_names = self._resolve_param_names(params)
        result = []

        for samples in self._samples_list:
            available = samples.getParamNames().list()
            indices = [available.index(p) for p in selected_names if p in available]
            if not indices:
                continue

            actual_names = [available[i] for i in indices]
            labels_list = [samples.getParamNames().parWithName(p).label for p in actual_names]

            selected_ranges = self._extract_ranges_for_params(samples, actual_names)

            subset = MCSamples(
                samples=samples.samples[:, indices],
                names=actual_names,
                labels=labels_list,
                label=getattr(samples, 'label', None),
                ranges=selected_ranges or None,
            )
            result.append(subset)
        return result

    def _extract_ranges_for_params(self, samples: MCSamples, param_names: List[str]) -> Dict[str, Tuple[float, float]]:
        """Extract ranges for specified parameters from MCSamples."""
        if not hasattr(samples, 'ranges') or not samples.ranges:
            return {}
        ranges = {}
        for pname in param_names:
            lo, hi = samples.ranges.getLower(pname), samples.ranges.getUpper(pname)
            if lo is not None and hi is not None:
                ranges[pname] = (lo, hi)
        return ranges

    # =========================================================================
    # Private Helper Methods - Report Generation
    # =========================================================================

    def _find_chain_path(self, chain_name: str) -> Optional[Path]:
        """Find chain file path."""
        chains_dir = get_chains_dir()
        h5_file = chains_dir / f"{chain_name}.h5"
        return h5_file if h5_file.exists() else None

    def _load_chain_metadata(self, chain_path: Path) -> Dict:
        """Load metadata from chain file."""
        import h5py
        metadata = {}
        try:
            with h5py.File(chain_path, 'r') as f:
                if 'metadata' not in f:
                    return metadata
                for key in f['metadata'].keys():
                    item = f['metadata'][key]
                    if isinstance(item, h5py.Dataset):
                        val = item[()]
                        metadata[key] = val.decode('utf-8') if isinstance(val, bytes) else val
        except (OSError, KeyError, AttributeError):
            pass
        return metadata

    def _generate_report_header(self, chain_name: str, metadata: Dict) -> List[str]:
        """Generate report header section."""
        import json
        from datetime import datetime

        lines = [
            f"# MCMC Analysis Report: {chain_name}",
            "",
            f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Framework**: HIcosmo",
            "",
            "## Run Information",
            "",
            f"- **Chain Name**: `{chain_name}`",
        ]

        param_config = {}
        likelihood_info = {}

        if 'parameter_config' in metadata:
            try:
                param_config = json.loads(metadata['parameter_config'])
                mcmc_cfg = param_config.get('mcmc', {})
                for key, label in [('num_chains', 'Chains'), ('num_warmup', 'Warmup Steps'), ('num_samples', 'Samples per Chain')]:
                    if key in mcmc_cfg:
                        lines.append(f"- **{label}**: {mcmc_cfg[key]}")
            except (json.JSONDecodeError, KeyError, TypeError):
                pass
        lines.append("")

        # Likelihood Information
        if 'likelihood_info' in metadata:
            try:
                likelihood_info = json.loads(metadata['likelihood_info'])
            except (json.JSONDecodeError, TypeError):
                pass

        if likelihood_info:
            lines.extend([
                "## Likelihood",
                "",
            ])
            if 'class' in likelihood_info:
                lines.append(f"- **Class**: `{likelihood_info['class']}`")
            elif 'module' in likelihood_info:
                lines.append(f"- **Module**: `{likelihood_info['module']}`")
            if 'function_name' in likelihood_info and likelihood_info['function_name'] != 'unknown':
                lines.append(f"- **Function**: `{likelihood_info['function_name']}`")
            if 'description' in likelihood_info:
                lines.append(f"- **Description**: {likelihood_info['description']}")
            elif 'docstring' in likelihood_info:
                # Extract first line of docstring as description
                docstring = likelihood_info['docstring'].strip()
                first_line = docstring.split('\n')[0].strip() if docstring else ""
                if first_line:
                    lines.append(f"- **Description**: {first_line}")
            lines.append("")

        # Input Parameters
        if 'parameters' in param_config:
            lines.extend([
                "## Input Parameters",
                "",
                "| Parameter | Prior | Bounds | Free |",
                "|-----------|-------|--------|------|",
            ])
            for pname, pinfo in param_config['parameters'].items():
                bounds = pinfo.get('bounds', [None, None])
                bounds_str = f"[{bounds[0]}, {bounds[1]}]" if bounds and bounds[0] is not None else "N/A"
                free = "Yes" if pinfo.get('free', False) else "No"
                prior = pinfo.get('prior', 'uniform')
                lines.append(f"| {pname} | {prior} | {bounds_str} | {free} |")
            lines.append("")

        return lines

    def _generate_constraints_section(self, metadata: Dict) -> List[str]:
        """Generate parameter constraints section."""
        samples = self._samples_list[0]
        marge_stats = samples.getMargeStats()

        lines = [
            "## Parameter Constraints",
            "",
            r"| Parameter | Mean ± $\sigma$ | Median | 68% CI | 95% CI |",
            "|-----------|----------|--------|--------|--------|",
        ]

        latex_1sigma = []
        latex_2sigma = []

        for param_name in self._param_names:
            ps = marge_stats.parWithName(param_name)
            if not ps:
                continue

            param_obj = samples.getParamNames().parWithName(param_name)
            latex_label = param_obj.label if param_obj and param_obj.label else param_name
            display_label = f"${latex_label}$"

            limit_68 = ps.limits[0] if ps.limits and len(ps.limits) > 0 else None
            limit_95 = ps.limits[1] if ps.limits and len(ps.limits) > 1 else None

            ci68_lo = limit_68.lower if limit_68 else ps.mean - ps.err
            ci68_hi = limit_68.upper if limit_68 else ps.mean + ps.err
            ci95_lo = limit_95.lower if limit_95 else ps.mean - 2*ps.err
            ci95_hi = limit_95.upper if limit_95 else ps.mean + 2*ps.err

            idx = samples.getParamNames().list().index(param_name)
            median = float(np.median(samples.samples[:, idx]))

            lines.append(
                f"| {display_label} | {ps.mean:.4g} ± {ps.err:.4g} | {median:.4g} | "
                f"[{ci68_lo:.4g}, {ci68_hi:.4g}] | [{ci95_lo:.4g}, {ci95_hi:.4g}] |"
            )

            # Build LaTeX format strings
            err_up_1 = ci68_hi - ps.mean
            err_dn_1 = ps.mean - ci68_lo
            err_up_2 = ci95_hi - ps.mean
            err_dn_2 = ps.mean - ci95_lo

            if abs(err_up_1 - err_dn_1) < 0.1 * ps.err:
                latex_1sigma.append(f"${latex_label} = {ps.mean:.3g}\\pm {ps.err:.2g}$")
            else:
                latex_1sigma.append(f"${latex_label} = {ps.mean:.3g}^{{+{err_up_1:.2g}}}_{{-{err_dn_1:.2g}}}$")

            if abs(err_up_2 - err_dn_2) < 0.1 * 2 * ps.err:
                latex_2sigma.append(f"${latex_label} = {ps.mean:.3g}\\pm {2*ps.err:.2g}$")
            else:
                latex_2sigma.append(f"${latex_label} = {ps.mean:.3g}^{{+{err_up_2:.2g}}}_{{-{err_dn_2:.2g}}}$")

        lines.append("")

        # LaTeX Format section
        lines.extend([
            "### LaTeX Format",
            "",
            r"**1$\sigma$ constraints:**",
            "```latex",
        ])
        lines.extend(latex_1sigma)
        lines.extend([
            "```",
            "",
            r"**2$\sigma$ constraints:**",
            "```latex",
        ])
        lines.extend(latex_2sigma)
        lines.extend([
            "```",
            "",
        ])

        return lines

    def _generate_diagnostics_section(self) -> List[str]:
        """Generate MCMC diagnostics section."""
        samples = self._samples_list[0]
        n_samples = samples.samples.shape[0]

        lines = [
            "## MCMC Diagnostics",
            "",
            f"- **Total samples**: {n_samples:,}",
            f"- **Parameters**: {len(self._param_names)}",
            "",
            "### Effective Sample Size (ESS)",
            "",
            "| Parameter | ESS | ESS/N |",
            "|-----------|-----|-------|",
        ]

        for param_name in self._param_names:
            idx = samples.getParamNames().list().index(param_name)
            data = samples.samples[:, idx]
            ess = self._estimate_ess(data, n_samples)
            lines.append(f"| {param_name} | {ess} | {100.0*ess/n_samples:.2f}% |")
        lines.append("")
        return lines

    def _estimate_ess(self, data: np.ndarray, n_samples: int) -> int:
        """Estimate effective sample size."""
        try:
            from scipy import signal
            centered = data - np.mean(data)
            autocorr = signal.correlate(centered, centered, mode='full')
            mid = len(autocorr) // 2
            autocorr = autocorr[mid:] / autocorr[mid]
            tau = 1 + 2 * np.sum(autocorr[:min(mid, 1000)])
            return min(int(n_samples / max(tau, 1)), n_samples)
        except (ImportError, ValueError, ZeroDivisionError):
            return n_samples

    def _generate_model_selection_section(self, chain_path: Optional[Path]) -> List[str]:
        """Generate model selection criteria section."""
        if not chain_path or not chain_path.exists():
            return []

        try:
            from .information_criteria import information_criteria
            ic = information_criteria(chain_path)

            lines = [
                "## Model Selection Criteria",
                "",
                f"- **$\\chi^2_{{\\rm min}}$**: {ic.get('chi2_min', 0):.2f}",
                f"- **$\\log(L_{{\\rm max}})$**: {ic.get('log_likelihood_max', 0):.2f}",
            ]
            if 'num_data' in ic:
                lines.append(f"- **$N_{{\\rm data}}$**: {ic['num_data']}")
            lines.extend([
                "",
                "| Criterion | Value | Interpretation |",
                "|-----------|-------|----------------|",
            ])
            if 'aic' in ic:
                lines.append(f"| AIC | {ic['aic']:.2f} | Lower is better |")
            if 'bic' in ic:
                lines.append(f"| BIC | {ic['bic']:.2f} | Lower is better |")
            if 'log_evidence' in ic:
                lines.append(f"| log(Z) | {ic['log_evidence']:.2f} | Higher is better |")
            lines.extend([
                "",
                f"- **Free parameters (k)**: {ic.get('num_params', 'N/A')}",
            ])
            if 'num_data' in ic:
                lines.append(f"- **Data points (N)**: {ic['num_data']}")
            lines.append("")
            return lines
        except Exception:
            return []
