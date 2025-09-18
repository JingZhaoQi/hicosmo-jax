"""HIcosmo MCMC sampling package with lazy attribute loading."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_ALIASES = {
    "core": "hicosmo.samplers.core",
    "inference": "hicosmo.samplers.inference",
    "config": "hicosmo.samplers.config",
    "utils": "hicosmo.samplers.utils",
    "persistence": "hicosmo.samplers.persistence",
    "multicore": "hicosmo.samplers.multicore",
    "init": "hicosmo.samplers.init",
}

_ATTRIBUTE_MAP = {
    # Core MCMC (Low-level)
    "MCMCSampler": ("hicosmo.samplers.core", "MCMCSampler"),
    "DiagnosticsTools": ("hicosmo.samplers.core", "DiagnosticsTools"),
    "run_mcmc": ("hicosmo.samplers.core", "run_mcmc"),
    # MCMC System (High-level)
    "MCMC": ("hicosmo.samplers.inference", "MCMC"),
    "quick_mcmc": ("hicosmo.samplers.inference", "quick_mcmc"),
    # Parameter Management
    "ParameterConfig": ("hicosmo.samplers.config", "ParameterConfig"),
    "AutoParameter": ("hicosmo.samplers.config", "AutoParameter"),
    # Advanced Tools
    "ParameterMapper": ("hicosmo.samplers.utils", "ParameterMapper"),
    "FunctionInspector": ("hicosmo.samplers.utils", "FunctionInspector"),
    "create_auto_model": ("hicosmo.samplers.utils", "create_auto_model"),
    "validate_likelihood_compatibility": (
        "hicosmo.samplers.utils",
        "validate_likelihood_compatibility",
    ),
    "analyze_likelihood_function": (
        "hicosmo.samplers.utils",
        "analyze_likelihood_function",
    ),
    # Checkpoint System
    "MCMCState": ("hicosmo.samplers.persistence", "MCMCState"),
    "CheckpointManager": ("hicosmo.samplers.persistence", "CheckpointManager"),
    "ResumeManager": ("hicosmo.samplers.persistence", "ResumeManager"),
    "create_likelihood_info": (
        "hicosmo.samplers.persistence",
        "create_likelihood_info",
    ),
    "create_data_info": ("hicosmo.samplers.persistence", "create_data_info"),
    # Multi-core Optimization
    "setup_multicore_execution": (
        "hicosmo.samplers.multicore",
        "setup_multicore_execution",
    ),
    "get_optimal_chain_count": (
        "hicosmo.samplers.multicore",
        "get_optimal_chain_count",
    ),
    "check_multicore_status": (
        "hicosmo.samplers.multicore",
        "check_multicore_status",
    ),
    "print_multicore_info": (
        "hicosmo.samplers.multicore",
        "print_multicore_info",
    ),
    # Elegant Initialization
    "Config": ("hicosmo.samplers.init", "Config"),
    "init_hicosmo": ("hicosmo.samplers.init", "init_hicosmo"),
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
    raise AttributeError(f"module 'hicosmo.samplers' has no attribute '{name}'")


def __dir__() -> list[str]:  # pragma: no cover - trivial helper
    return sorted(list(__all__) + [k for k in globals().keys() if not k.startswith("_")])
