"""
HIcosmo: High-performance universal cosmology framework.

HIcosmo (HI = neutral hydrogen, I = Roman numeral 1) is a universal
cosmological parameter estimation framework with enhanced functionality
and optimizations for neutral hydrogen cosmology and 21cm surveys.
Built with JAX for high-performance computing and modern MCMC methods.
"""

from __future__ import annotations

import importlib
from typing import Any

__version__ = "0.1.0"
__author__ = "Jingzhao Qi"

_MODULE_ALIASES = {
    "models": "hicosmo.models",
    "likelihoods": "hicosmo.likelihoods",
    "samplers": "hicosmo.samplers",
    "parameters": "hicosmo.parameters",
    "runner": "hicosmo.runner",
    "utils": "hicosmo.utils",
    "visualization": "hicosmo.visualization",
}

_ATTRIBUTE_MAP = {
    # High-level API (3-line usage)
    "hicosmo": ("hicosmo.hicosmo", "hicosmo"),
    "list_likelihoods": ("hicosmo.hicosmo", "list_likelihoods"),
    "list_cosmologies": ("hicosmo.hicosmo", "list_cosmologies"),
    "configure_logging": ("hicosmo.utils.logging", "configure_logging"),
    "run_from_config": ("hicosmo.runner.executor", "run_from_config"),
    # Elegant initialization helpers
    "Config": ("hicosmo.samplers.init", "Config"),
    "init": ("hicosmo.samplers.init", "init"),
    "init_hicosmo": ("hicosmo.samplers.init", "init_hicosmo"),
    "setup_multicore_execution": ("hicosmo.samplers.init", "setup_multicore_execution"),
    # Output directory configuration
    "set_output_dir": ("hicosmo.samplers.constants", "set_output_dir"),
    "get_output_dir": ("hicosmo.samplers.constants", "get_output_dir"),
}

__all__ = list(_MODULE_ALIASES.keys()) + list(_ATTRIBUTE_MAP.keys())


def __getattr__(name: str) -> Any:  # pragma: no cover - simple trampoline
    if name in _ATTRIBUTE_MAP:
        module_name, attr = _ATTRIBUTE_MAP[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    if name in _MODULE_ALIASES:
        module = importlib.import_module(_MODULE_ALIASES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module 'hicosmo' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - trivial helper
    return sorted(
        list(__all__) + [k for k in globals().keys() if not k.startswith("_")]
    )
