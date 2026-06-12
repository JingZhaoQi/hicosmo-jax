"""Public API component resolution."""

from __future__ import annotations

from functools import reduce
from typing import Any, List, Tuple, Type, Union

from .models.base import CosmologyBase
from .runner.components import (
    LIKELIHOOD_REGISTRY,
    THEORY_REGISTRY,
    build_likelihoods,
    resolve_theory,
)


def resolve_cosmology_class(
    cosmology: Union[Type[CosmologyBase], str],
) -> Type[CosmologyBase]:
    """Resolve a public cosmology specification to a model class."""
    if isinstance(cosmology, str):
        try:
            return resolve_theory(cosmology)
        except KeyError as exc:
            raise ValueError(str(exc)) from exc
    return cosmology


def resolve_likelihoods(
    cosmology_class: Type[CosmologyBase],
    likelihood: Union[Any, List[Any], Tuple[Any, ...], str],
) -> List[Any]:
    """Resolve public likelihood spec(s) into instantiated likelihood objects."""
    like_list = (
        list(likelihood) if isinstance(likelihood, (list, tuple)) else [likelihood]
    )
    resolved: List[Any] = []

    for like in like_list:
        if isinstance(like, (str, dict)):
            resolved.extend(build_likelihoods([like], theory_class=cosmology_class))
            continue

        if hasattr(like, "cosmology_class") and like.cosmology_class is None:
            like.cosmology_class = cosmology_class
        resolved.append(like)

    if not resolved:
        raise ValueError("At least one likelihood must be provided.")
    return resolved


def combine_likelihoods(likelihoods: List[Any]) -> Any:
    """Combine likelihood objects with their public + operator."""
    if len(likelihoods) == 1:
        return likelihoods[0]
    return reduce(lambda a, b: a + b, likelihoods)


def list_likelihood_names() -> List[str]:
    """List public likelihood string identifiers."""
    return LIKELIHOOD_REGISTRY.list()


def list_cosmology_names() -> List[str]:
    """List public cosmology model class names."""
    names = set()
    for key in THEORY_REGISTRY.list():
        try:
            names.add(resolve_theory(key).__name__)
        except KeyError:
            continue
    return sorted(names)
