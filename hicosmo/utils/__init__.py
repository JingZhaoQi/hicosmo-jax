"""
HIcosmo Utilities Package
========================

Utility modules and constants for cosmological calculations.

IMPORTANT: This module uses lazy loading to avoid importing JAX too early.
This allows Config.init(num_devices=N) to work correctly.
"""

from __future__ import annotations

import importlib
from typing import Any

# These are safe to import immediately (no JAX dependency)
try:
    from .constants import *
except ImportError:
    pass

from .logging import configure_logging, get_logger

# Lazy-loaded attributes that depend on JAX
_LAZY_ATTRIBUTES = {
    # From manifest.py (imports jax, numpyro)
    "write_run_manifest": ("hicosmo.utils.manifest", "write_run_manifest"),
    # From jax_tools.py (imports jax)
    "trapezoid": ("hicosmo.utils.jax_tools", "trapezoid"),
    "simpson": ("hicosmo.utils.jax_tools", "simpson"),
    "integrate_simpson": ("hicosmo.utils.jax_tools", "integrate_simpson"),
    "integrate_logspace": ("hicosmo.utils.jax_tools", "integrate_logspace"),
    "gauss_legendre_nodes_weights": ("hicosmo.utils.jax_tools", "gauss_legendre_nodes_weights"),
    "gauss_legendre_integrate": ("hicosmo.utils.jax_tools", "gauss_legendre_integrate"),
    "gauss_legendre_integrate_batch": ("hicosmo.utils.jax_tools", "gauss_legendre_integrate_batch"),
    "cumulative_trapezoid": ("hicosmo.utils.jax_tools", "cumulative_trapezoid"),
    "integrate_batch_cumulative": ("hicosmo.utils.jax_tools", "integrate_batch_cumulative"),
    "integrate_segmented": ("hicosmo.utils.jax_tools", "integrate_segmented"),
    "integrate_adaptive_simpson": ("hicosmo.utils.jax_tools", "integrate_adaptive_simpson"),
    "gradient_1d": ("hicosmo.utils.jax_tools", "gradient_1d"),
    "finite_difference_grad": ("hicosmo.utils.jax_tools", "finite_difference_grad"),
    "grad": ("hicosmo.utils.jax_tools", "grad"),
    "jacobian": ("hicosmo.utils.jax_tools", "jacobian"),
    "hessian": ("hicosmo.utils.jax_tools", "hessian"),
    "newton_root": ("hicosmo.utils.jax_tools", "newton_root"),
    "bisection_root": ("hicosmo.utils.jax_tools", "bisection_root"),
    "rk4_step": ("hicosmo.utils.jax_tools", "rk4_step"),
    "odeint_rk4": ("hicosmo.utils.jax_tools", "odeint_rk4"),
    "solve_ivp_rk4": ("hicosmo.utils.jax_tools", "solve_ivp_rk4"),
}

__all__ = [
    "configure_logging",
    "get_logger",
] + list(_LAZY_ATTRIBUTES.keys())


def __getattr__(name: str) -> Any:
    """Lazy load JAX-dependent utilities to allow Config.init() to work first."""
    if name in _LAZY_ATTRIBUTES:
        module_name, attr = _LAZY_ATTRIBUTES[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'hicosmo.utils' has no attribute '{name}'")
