"""Configuration-driven runner utilities for HIcosmo."""

from .components import (
    LIKELIHOOD_REGISTRY,
    SAMPLER_REGISTRY,
    THEORY_REGISTRY,
    ComponentRegistry,
    build_likelihoods,
    resolve_sampler,
    resolve_theory,
)
from .config import (
    ConfigError,
    build_parameter_registry,
    load_config,
    normalize_config,
    validate_config,
)
from .datasets import (
    DATASETS,
    ensure_dataset,
    get_dataset,
    resolve_data_root,
    resolve_dataset_path,
)
from .executor import run_from_config

__all__ = [
    "ConfigError",
    "ComponentRegistry",
    "THEORY_REGISTRY",
    "LIKELIHOOD_REGISTRY",
    "SAMPLER_REGISTRY",
    "DATASETS",
    "load_config",
    "normalize_config",
    "validate_config",
    "build_parameter_registry",
    "resolve_theory",
    "resolve_sampler",
    "build_likelihoods",
    "get_dataset",
    "resolve_data_root",
    "resolve_dataset_path",
    "ensure_dataset",
    "run_from_config",
]
