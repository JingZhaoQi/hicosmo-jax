#!/usr/bin/env python3
"""
Configuration Constants for HIcosmo Samplers

Centralizes all configuration constants to eliminate hardcoding.
"""

from pathlib import Path
from typing import Union
import os

# ========== Global Output Directory ==========
# All relative paths (mcmc_chains/, results/) are relative to this directory.
# Default is current working directory. Use set_output_dir() to change.

_OUTPUT_BASE_DIR: Path = Path.cwd()


def set_output_dir(path: Union[str, Path, None] = None) -> Path:
    """
    Set the base directory for all HIcosmo outputs.

    All relative paths (mcmc_chains/, results/) will be created
    relative to this directory.

    Parameters
    ----------
    path : str, Path, or None
        The base directory. Can be:
        - A directory path (str or Path)
        - A file path (will use parent directory) - useful with __file__
        - None: reset to current working directory

    Returns
    -------
    Path
        The resolved output directory

    Examples
    --------
    >>> import hicosmo as hc
    >>> # Set to script's directory (most common use case)
    >>> hc.set_output_dir(__file__)
    >>>
    >>> # Set to specific directory
    >>> hc.set_output_dir('/path/to/outputs')
    >>>
    >>> # Reset to current working directory
    >>> hc.set_output_dir(None)
    """
    global _OUTPUT_BASE_DIR

    if path is None:
        _OUTPUT_BASE_DIR = Path.cwd()
    else:
        p = Path(path)
        # If it's a file (like __file__), use its parent directory
        if p.is_file() or p.suffix:
            _OUTPUT_BASE_DIR = p.parent.resolve()
        else:
            _OUTPUT_BASE_DIR = p.resolve()

    return _OUTPUT_BASE_DIR


def get_output_dir() -> Path:
    """Get the current output base directory."""
    return _OUTPUT_BASE_DIR


def get_chains_dir() -> Path:
    """Get the MCMC chains directory (output_dir/mcmc_chains)."""
    return _OUTPUT_BASE_DIR / "mcmc_chains"


def get_results_dir() -> Path:
    """Get the results directory (output_dir/results)."""
    return _OUTPUT_BASE_DIR / "results"


# ========== MCMC Sampling Defaults ==========

# Default number of MCMC samples
DEFAULT_NUM_SAMPLES = 2000

# Default number of parallel chains
DEFAULT_NUM_CHAINS = 4

# Warmup steps for different modes
DEFAULT_WARMUP_STANDARD = 2000      # Without optimization
DEFAULT_WARMUP_OPTIMIZED = 300      # With optimization

# ========== Optimization Defaults ==========

# Maximum iterations for optimization
DEFAULT_MAX_OPTIMIZATION_ITERATIONS = 1000

# Report progress every N iterations
OPTIMIZATION_PROGRESS_INTERVAL = 100

# Penalty factor for constraints
OPTIMIZATION_PENALTY_FACTOR = 1000.0

# ========== Checkpointing ==========

# Save checkpoint interval
DEFAULT_CHECKPOINT_INTERVAL = 1000

# Save checkpoint interval in seconds (time-based)
DEFAULT_CHECKPOINT_INTERVAL_SECONDS = 600.0

# Default checkpoint directory (use get_chains_dir() for dynamic path)
DEFAULT_CHECKPOINT_DIR = Path("mcmc_chains")  # Legacy, prefer get_chains_dir()

# File extension
CHECKPOINT_FILE_EXTENSION = ".h5"

# File patterns
CHECKPOINT_PATTERN_TEMPLATE = "{run_name}_step_*.h5"
FINAL_RESULT_PATTERN = "{run_name}.h5"

# ========== Random Number Generation ==========

# RNG seed modulo for 32-bit compatibility
RNG_SEED_MODULO = 2**32

# ========== Numerical Tolerances ==========

# R-hat convergence threshold
RHAT_CONVERGENCE_THRESHOLD = 1.1

# Minimum effective sample size
MIN_EFFECTIVE_SAMPLE_SIZE = 100

# ========== Multi-core Optimization ==========

# Default number of CPU devices for parallel execution
DEFAULT_NUM_CPU_DEVICES = 4

# Auto-detect CPU cores
AUTO_DETECT_CPU_CORES = True

# Warning threshold for device count mismatch
DEVICE_MISMATCH_WARNING_THRESHOLD = 2
