"""Parameter assembly helpers shared by public runners."""

from __future__ import annotations

from typing import List, Optional, Type

from ..models.base import CosmologyBase
from .registry import ParameterRegistry


def register_model_parameters(
    registry: ParameterRegistry, cosmology_class: Type[CosmologyBase]
) -> set[str]:
    """Add model-declared parameters to a registry and return their names."""
    model_param_names: set[str] = set()
    if not hasattr(cosmology_class, "get_parameters"):
        return model_param_names

    for param in cosmology_class.get_parameters():
        model_param_names.add(param.name)
        if param.name not in registry:
            registry.add(
                param.name,
                value=param.value,
                free=param.free,
                prior=param.prior,
                latex_label=param.latex_label,
            )
    return model_param_names


def apply_requested_free_params(
    registry: ParameterRegistry,
    free_params: Optional[List[str]],
    model_param_names: set[str],
) -> None:
    """
    Apply user-selected model free parameters without clobbering nuisance status.

    Model parameters are forced to match ``free_params``. Non-model parameters are
    only touched when explicitly named, which preserves likelihood-declared
    nuisance defaults such as BAO ``H0_rd`` or SN absolute-magnitude settings.
    """
    if not free_params:
        return

    missing = [name for name in free_params if name not in registry]
    if missing:
        raise KeyError(
            "Parameter(s) not found in registry: "
            + ", ".join(missing)
            + ". Add the relevant model or likelihood before setting them free."
        )

    requested = set(free_params)
    for name in model_param_names:
        if name not in registry:
            continue
        param = registry.get(name)
        param.free = name in requested
        if param.free and param.prior is None:
            raise ValueError(
                f"Cannot set '{name}' as free: no prior distribution defined."
            )

    for name in requested - model_param_names:
        param = registry.get(name)
        param.free = True
        if param.prior is None:
            raise ValueError(
                f"Cannot set '{name}' as free: no prior distribution defined."
            )
