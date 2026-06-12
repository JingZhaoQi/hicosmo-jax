"""Derived-parameter post-processing utilities."""

from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np


def _sample_count(samples: Dict[str, Any]) -> int:
    if not samples:
        return 0
    first = next(iter(samples.values()))
    return len(first)


def _coerce_derived_array(value: Any, n_samples: int) -> np.ndarray | None:
    array = np.asarray(value)
    if array.ndim == 0:
        array = np.full(n_samples, float(array))
    if len(array) != n_samples:
        return None
    return array


def _add_derived_arrays(
    output: Dict[str, Any],
    derived: Dict[str, Any],
    *,
    sampled_names: set[str],
    derived_names: set[str],
    n_samples: int,
) -> None:
    for name, value in derived.items():
        if name in sampled_names or name in derived_names:
            continue
        array = _coerce_derived_array(value, n_samples)
        if array is None:
            continue
        output[name] = array
        derived_names.add(name)


def _compute_scalar_derived(
    likelihood: Any,
    samples: Dict[str, Any],
    *,
    sampled_names: set[str],
    derived_names: set[str],
    n_samples: int,
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, list[float]] = {}
    active_names: set[str] = set()

    for i in range(n_samples):
        params = {name: float(np.asarray(samples[name])[i]) for name in sampled_names}
        try:
            derived = likelihood.derived_parameters(**params)
        except Exception:
            continue
        if not derived:
            continue

        for name, value in derived.items():
            if name in sampled_names or name in derived_names:
                continue
            if name not in arrays:
                arrays[name] = []
                active_names.add(name)
            arrays[name].append(float(value))

    result = {}
    for name in active_names:
        if len(arrays[name]) == n_samples:
            result[name] = np.asarray(arrays[name])
    return result


def compute_derived_parameters(
    samples: Dict[str, Any], likelihoods: Iterable[Any]
) -> Dict[str, Any]:
    """Return samples plus non-conflicting derived parameters from likelihoods."""
    n_samples = _sample_count(samples)
    if n_samples == 0:
        return samples

    output = dict(samples)
    sampled_names = set(samples.keys())
    derived_names: set[str] = set()

    for likelihood in likelihoods:
        if hasattr(likelihood, "derived_parameters_vectorized"):
            try:
                derived = likelihood.derived_parameters_vectorized(samples)
            except Exception:
                derived = None
            if derived:
                _add_derived_arrays(
                    output,
                    derived,
                    sampled_names=sampled_names,
                    derived_names=derived_names,
                    n_samples=n_samples,
                )
                continue

        if not hasattr(likelihood, "derived_parameters"):
            continue

        derived = _compute_scalar_derived(
            likelihood,
            samples,
            sampled_names=sampled_names,
            derived_names=derived_names,
            n_samples=n_samples,
        )
        _add_derived_arrays(
            output,
            derived,
            sampled_names=sampled_names,
            derived_names=derived_names,
            n_samples=n_samples,
        )

    return output
