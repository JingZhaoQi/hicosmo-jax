#!/usr/bin/env python3
"""
HIcosmo Visualization System - Core Plotting Functions
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from contextlib import contextmanager

from ..utils.logging import get_logger

try:
    from getdist import plots, MCSamples

    HAS_GETDIST = True
except ImportError:
    HAS_GETDIST = False
    plots = None
    MCSamples = None

logger = get_logger(__name__)


@contextmanager
def _suppress_stdout():
    """Context manager to suppress stdout (used for getdist verbose messages)."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def _require_getdist() -> None:
    if not HAS_GETDIST:
        raise ImportError(
            "GetDist is required for HIcosmo plotting. Install with: pip install getdist"
        )


# Color schemes
MODERN_COLORS = [
    "#2E86AB",
    "#A23B72",
    "#F18F01",
    "#C73E1D",
    "#592E83",
    "#0F7173",
    "#7B2D26",
]
CLASSIC_COLORS = [
    "#348ABD",
    "#7A68A6",
    "#E24A33",
    "#467821",
    "#ffb3a6",
    "#188487",
    "#A60628",
]


def _safe_tight_layout(fig: plt.Figure) -> None:
    """Apply tight_layout while suppressing Matplotlib warnings about incompatible axes."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "This figure includes Axes that are not compatible with tight_layout",
            UserWarning,
        )
        # Use increased padding to prevent tick label overlap
        fig.tight_layout(pad=1.5, h_pad=0.8, w_pad=0.8)


# Results directory
def _get_results_dir():
    """Get results directory relative to calling script"""
    import inspect

    frame = inspect.currentframe()
    try:
        # Get first non-visualization module frame in call stack
        caller_frame = frame.f_back.f_back if frame.f_back else frame.f_back
        if caller_frame and "visualization" not in caller_frame.f_code.co_filename:
            caller_dir = Path(caller_frame.f_code.co_filename).parent
        else:
            caller_dir = Path.cwd()
        results_dir = caller_dir / "results"
        results_dir.mkdir(exist_ok=True)
        return results_dir
    finally:
        del frame


RESULTS_DIR = Path("results")  # fallback
RESULTS_DIR.mkdir(exist_ok=True)


def _apply_qijing_style():
    """Apply professional plotting style - based on FigStyle.py"""
    import seaborn as sns

    # Basic style configuration (merged best parts from original 5 styles)
    style_config = {
        "axes.linewidth": 1.2,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 14,
        "legend.frameon": False,  # Key: frameless legend
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 12,
        "lines.linewidth": 2.0,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
    }

    plt.rcParams.update(style_config)

    # Set color cycle (avoid plt.gca() which creates empty figure)
    from cycler import cycler

    plt.rcParams["axes.prop_cycle"] = cycler("color", MODERN_COLORS)


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
            font_size = plt.rcParams["xtick.labelsize"]
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
            ax.xaxis.set_major_locator(
                ticker.MaxNLocator(nbins=optimal_x_ticks, prune="both")
            )

        # Y-axis optimization
        if ax.get_ylabel() or len(ax.get_yticklabels()) > 0:
            # Y-axis labels need more vertical space too
            line_height = font_size * 1.8  # more generous line spacing
            max_y_ticks = max(3, int(height / (line_height * 2.2)))
            optimal_y_ticks = min(max_y_ticks, 6)
            ax.yaxis.set_major_locator(
                ticker.MaxNLocator(nbins=optimal_y_ticks, prune="both")
            )

        # Handle very small subplots
        if width < 120:
            # Very narrow - reduce ticks and rotate
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="both"))
            for label in ax.get_xticklabels():
                label.set_rotation(45)
                label.set_ha("right")
        elif width < 180:
            # Narrow - just reduce ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=4, prune="both"))

        if height < 100:
            # Very short - minimal y-ticks
            ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=3, prune="both"))


def _load_chain_simple(filename: str):
    """
    Load MCMC chain data with automatic metadata extraction.

    Returns
    -------
    dict or tuple
        If file contains metadata: (samples_dict, metadata_dict)
        Otherwise: samples_dict only

    The metadata dict may contain 'labels' and 'ranges' if the file
    was saved with MCMC.save_results().
    """
    import numpy as np
    import pickle

    file_path = Path(filename)

    if file_path.suffix == ".npy":
        return np.load(file_path)

    elif file_path.suffix == ".npz":
        npz_file = np.load(file_path, allow_pickle=True)

        # Check if this is a HIcosmo-formatted file with metadata
        if "_param_names" in npz_file.files:
            # HIcosmo format with metadata
            param_names = npz_file["_param_names"].tolist()
            labels_arr = (
                npz_file["_labels"].tolist()
                if "_labels" in npz_file.files
                else param_names
            )
            ranges_arr = npz_file["_ranges"] if "_ranges" in npz_file.files else None

            samples = {}
            for pname in param_names:
                key = f"samples_{pname}"
                if key in npz_file.files:
                    samples[pname] = npz_file[key]

            metadata = {
                "labels": dict(zip(param_names, labels_arr)),
                "ranges": (
                    dict(zip(param_names, [tuple(r) for r in ranges_arr]))
                    if ranges_arr is not None
                    else {}
                ),
            }
            return samples, metadata

        # Standard npz file
        if len(npz_file.files) == 1:
            return npz_file[npz_file.files[0]]
        return dict(npz_file)

    elif file_path.suffix in [".pkl", ".pickle"]:
        # Pickle format - check for HIcosmo metadata
        with open(file_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            if "samples" in data and "metadata" in data:
                # HIcosmo format with metadata
                metadata = data["metadata"]
                return data["samples"], metadata
            elif "samples" in data:
                # Has samples but no metadata
                return data["samples"]
        return data

    elif file_path.suffix in [".txt", ".dat"]:
        return np.loadtxt(file_path)

    elif file_path.suffix in [".h5", ".hdf5"]:
        try:
            import h5py

            data = {}
            metadata = {}

            with h5py.File(file_path, "r") as f:
                # Load samples
                if "samples" in f:
                    group = f["samples"]
                elif "chains" in f:
                    group = f["chains"]
                else:
                    group = f
                for key in group.keys():
                    data[str(key)] = group[key][:]

                # Load metadata if present
                if "metadata" in f:
                    meta_group = f["metadata"]

                    # Load labels
                    if "labels" in meta_group:
                        labels_group = meta_group["labels"]
                        metadata["labels"] = dict(labels_group.attrs)

                    # Load ranges
                    if "ranges" in meta_group:
                        ranges_group = meta_group["ranges"]
                        metadata["ranges"] = {}
                        for key in ranges_group.keys():
                            bounds = ranges_group[key][:]
                            metadata["ranges"][key] = (
                                float(bounds[0]),
                                float(bounds[1]),
                            )

            if metadata:
                return data, metadata
            return data

        except ImportError:
            raise ImportError("h5py required for HDF5 files")
    else:
        try:
            return np.loadtxt(file_path)
        except:
            raise ValueError(f"Cannot load file format: {file_path.suffix}")


def _strip_latex_dollars(label: str) -> str:
    """
    Strip $ symbols from LaTeX label for GetDist compatibility.

    GetDist automatically wraps labels in math environment, so user-provided
    labels like '$a$' or '$H_0$' would become '$$a$$' causing parse errors.
    """
    if label is None:
        return label
    # Remove leading/trailing $ symbols
    label = label.strip()
    if label.startswith("$") and label.endswith("$"):
        label = label[1:-1]
    # Also handle escaped dollars
    label = label.replace("\\$", "")
    return label


def _prepare_latex_labels(param_names: List[str]) -> List[str]:
    """Prepare LaTeX labels with simple auto-formatting for GetDist.

    Only does basic formatting (e.g., Omega_m -> Omega_{m}).
    Proper LaTeX labels should come from the chain file metadata.
    """
    labels = []
    for param in param_names:
        latex_label = _strip_latex_dollars(param)
        # Simple underscore formatting: Omega_m -> Omega_{m}
        if "_" in latex_label and "_{" not in latex_label:
            latex_label = latex_label.replace("_", "_{") + "}"
        labels.append(latex_label)
    return labels


def _normalize_range(
    bounds: Union[Tuple[float, float], List[float]],
) -> Tuple[float, float]:
    lo, hi = bounds
    return (float(lo), float(hi))


def _filter_ranges_for_params(
    param_names: List[str],
    range_dict: Optional[Dict[str, Tuple[float, float]]],
) -> Dict[str, Tuple[float, float]]:
    if not range_dict:
        return {}

    allowed = set(param_names)
    filtered: Dict[str, Tuple[float, float]] = {}
    for name, bounds in range_dict.items():
        if name not in allowed:
            warnings.warn(
                f"Ignoring plotting range for '{name}' (parameter not present)",
                RuntimeWarning,
            )
            continue
        filtered[name] = _normalize_range(bounds)
    return filtered


def _range_from_values(name: str, column: np.ndarray) -> Tuple[float, float]:
    finite = np.asarray(column, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        warnings.warn(
            f"Parameter '{name}' has no finite samples; using fallback range [-1, 1]",
            RuntimeWarning,
        )
        return (-1.0, 1.0)

    lo = float(finite.min())
    hi = float(finite.max())
    if hi > lo:
        pad = 0.02 * (hi - lo)
        return (lo - pad, hi + pad)

    center = lo
    pad = max(abs(center), 1.0) * 0.02
    warnings.warn(
        f"Parameter '{name}' samples collapse to a single value; expanding range symmetrically",
        RuntimeWarning,
    )
    return (center - pad, center + pad)


def _ensure_positive_range(
    name: str,
    bounds: Tuple[float, float],
    fallback_column: np.ndarray,
) -> Tuple[float, float]:
    lo, hi = bounds
    if hi > lo:
        return (lo, hi)

    warnings.warn(
        f"Plotting range for '{name}' has non-positive width; deriving range from samples",
        RuntimeWarning,
    )
    return _range_from_values(name, fallback_column)


def _merge_range_dicts(
    *range_dicts: Optional[Dict[str, Tuple[float, float]]]
) -> Dict[str, Tuple[float, float]]:
    merged: Dict[str, Tuple[float, float]] = {}
    for range_dict in range_dicts:
        if not range_dict:
            continue
        for key, bounds in range_dict.items():
            merged[key] = _normalize_range(bounds)
    return merged


def _coerce_ranges_arg(
    ranges: Optional[Union[Dict[str, Tuple[float, float]], List[Any], Tuple[Any, ...]]],
) -> Optional[Dict[str, Tuple[float, float]]]:
    if ranges is None:
        return None
    if isinstance(ranges, dict):
        return {key: _normalize_range(val) for key, val in ranges.items()}
    raise TypeError("ranges must be a dict mapping parameter names to (min, max)")


def _resolve_ranges(
    param_names: List[str],
    samples: np.ndarray,
    explicit_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Optional[Dict[str, Tuple[float, float]]]:
    arr = np.asarray(samples, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    explicit = _filter_ranges_for_params(param_names, explicit_ranges)
    resolved: Dict[str, Tuple[float, float]] = {}

    for idx, name in enumerate(param_names):
        column = arr[:, idx]
        if name in explicit:
            resolved[name] = _ensure_positive_range(name, explicit[name], column)
            continue
        resolved[name] = _range_from_values(name, column)

    return resolved if resolved else None


def _prepare_getdist_samples(
    data, params=None, labels=None, ranges=None, param_ranges=None, **kwargs
) -> MCSamples:
    """
    Convert to GetDist format with automatic metadata extraction.

    If data is loaded from a HIcosmo-saved file, labels and ranges are
    automatically extracted from the file metadata.

    Parameters
    ----------
    data : various
        Chain data (file path, array, dict, MCSamples, or (samples, metadata) tuple)
    params : list, optional
        Parameters to select
    labels : list or dict, optional
        User-provided labels (override file metadata)
    ranges : dict, optional
        User-provided ranges (override file metadata)
    param_ranges : dict, optional
        Additional ranges to merge (legacy parameter)
    **kwargs
        Additional arguments for MCSamples

    Returns
    -------
    MCSamples
        GetDist MCSamples object with labels and ranges
    """
    _require_getdist()
    # Track metadata from file
    file_metadata = {}

    if isinstance(data, MCSamples):
        merged_ranges = _merge_range_dicts(_coerce_ranges_arg(ranges), param_ranges)
        param_names = [param.name for param in data.getParamNames().names]
        resolved = _resolve_ranges(
            param_names, data.samples, explicit_ranges=merged_ranges
        )
        if resolved:
            copied = data.copy()
            copied.setRanges(resolved)
            return copied
        return data

    # Load from file if string path
    if isinstance(data, str):
        loaded = _load_chain_simple(data)
        # Check if metadata was returned
        if isinstance(loaded, tuple) and len(loaded) == 2:
            data, file_metadata = loaded
        else:
            data = loaded

    # Handle tuple format (samples, metadata) - e.g., from explicit passing
    if isinstance(data, tuple) and len(data) == 2 and isinstance(data[1], dict):
        if "labels" in data[1] or "ranges" in data[1]:
            data, file_metadata = data

    # Select parameters
    if params is not None:
        # Handle indices or names
        if isinstance(data, dict):
            if isinstance(params[0], int):
                # Convert 1-based index to 0-based index
                all_param_names = list(data.keys())
                selected_params = [all_param_names[i] for i in params]
            else:
                # Parameter name
                selected_params = params
            samples = np.column_stack([data[p] for p in selected_params])
            param_names = selected_params
        else:
            samples = data[:, params]
            param_names = [f"param_{i}" for i in params]
    else:
        if isinstance(data, dict):
            param_names = list(data.keys())
            samples = np.column_stack([data[k] for k in param_names])
        else:
            samples = data
            param_names = [f"param_{i}" for i in range(samples.shape[1])]

    # Prepare LaTeX labels with priority: user-provided > file metadata > auto-generated
    # Note: All labels must have $ stripped for GetDist compatibility
    if labels is not None:
        # User provided labels
        if isinstance(labels, dict):
            plot_labels = [_strip_latex_dollars(labels.get(p, p)) for p in param_names]
        else:
            plot_labels = [_strip_latex_dollars(l) for l in labels]
    elif "labels" in file_metadata:
        # Use labels from file metadata (strip $ for GetDist)
        file_labels = file_metadata["labels"]
        plot_labels = [_strip_latex_dollars(file_labels.get(p, p)) for p in param_names]
    else:
        # Auto-generate LaTeX labels (already stripped in _prepare_latex_labels)
        plot_labels = _prepare_latex_labels(param_names)

    # Merge ranges with priority: user-provided > file metadata
    file_ranges = file_metadata.get("ranges", {})
    merged_ranges = _merge_range_dicts(
        file_ranges, _coerce_ranges_arg(ranges), param_ranges
    )
    resolved_ranges = _resolve_ranges(
        param_names, samples, explicit_ranges=merged_ranges
    )

    # Create MCSamples with ranges support
    # IMPORTANT: Set ignore_rows=0.0 by default because NumPyro already discards warmup
    # This avoids "double burn-in" where GetDist would discard another 30% by default
    ignore_rows = kwargs.pop("ignore_rows", 0.0)
    with _suppress_stdout():  # Suppress getdist "Removed no burn in" message
        return MCSamples(
            samples=samples,
            names=param_names,
            labels=plot_labels,
            ranges=resolved_ranges,
            ignore_rows=ignore_rows,
            **kwargs,
        )
