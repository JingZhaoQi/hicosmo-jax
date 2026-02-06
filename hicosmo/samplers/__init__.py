"""HIcosmo MCMC sampling package with lazy attribute loading."""

from __future__ import annotations

import importlib
from typing import Any

_MODULE_ALIASES = {
    "inference": "hicosmo.samplers.inference",
    "config": "hicosmo.samplers.config",
    "utils": "hicosmo.samplers.utils",
    "persistence": "hicosmo.samplers.persistence",
    "init": "hicosmo.samplers.init",
}

_ATTRIBUTE_MAP = {
    # MCMC System (High-level)
    "MCMC": ("hicosmo.samplers.inference", "MCMC"),
    # Parameter Management
    "ParameterConfig": ("hicosmo.samplers.config", "ParameterConfig"),
    # Checkpoint System
    "MCMCState": ("hicosmo.samplers.persistence", "MCMCState"),
    "CheckpointManager": ("hicosmo.samplers.persistence", "CheckpointManager"),
    "ResumeManager": ("hicosmo.samplers.persistence", "ResumeManager"),
    "DEFAULT_CHECKPOINT_DIR": ("hicosmo.samplers.constants", "DEFAULT_CHECKPOINT_DIR"),
    "create_likelihood_info": (
        "hicosmo.samplers.persistence",
        "create_likelihood_info",
    ),
    "create_data_info": ("hicosmo.samplers.persistence", "create_data_info"),
    # Elegant Initialization
    "Config": ("hicosmo.samplers.init", "Config"),
    "init_hicosmo": ("hicosmo.samplers.init", "init_hicosmo"),
    "init": ("hicosmo.samplers.init", "init"),
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
    return sorted(
        list(__all__) + [k for k in globals().keys() if not k.startswith("_")]
    )
