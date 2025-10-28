"""Numerical derivative utilities for Fisher matrix calculations.

Provides reusable, JAX-friendly finite-difference routines for gradients,
Hessians, and Jacobians. Designed to work with scalar or vector-valued
functions of cosmological parameters while respecting HIcosmo's precision
requirements.
"""

from __future__ import annotations

from typing import Callable, Tuple

import jax.numpy as jnp
from jax import config as jax_config


_X64_ENABLED = bool(jax_config.read("jax_enable_x64"))
_DEFAULT_DTYPE = jnp.float64 if _X64_ENABLED else jnp.float32

ArrayLikeFunc = Callable[[jnp.ndarray], jnp.ndarray]
ScalarFunc = Callable[[jnp.ndarray], float]


def compute_step_sizes(
    fiducial: jnp.ndarray,
    base_step: float,
    method: str = "adaptive"
) -> jnp.ndarray:
    """Return per-parameter step sizes for finite differencing."""
    x = jnp.asarray(fiducial, dtype=_DEFAULT_DTYPE)
    if method == "fixed":
        return jnp.full_like(x, base_step)
    if method == "relative":
        return base_step * jnp.maximum(1.0, jnp.abs(x))
    if method == "optimal":
        return base_step * jnp.maximum(1.0, jnp.sqrt(jnp.abs(x) + 1e-8))
    if method == "adaptive":
        return base_step * (1.0 + jnp.abs(x))
    raise ValueError(f"Unknown step method '{method}'")


def finite_difference_gradient(
    func: ScalarFunc,
    point: jnp.ndarray,
    steps: jnp.ndarray
) -> jnp.ndarray:
    """Central-difference gradient of a scalar function."""
    point = jnp.asarray(point, dtype=_DEFAULT_DTYPE)
    steps = jnp.asarray(steps, dtype=_DEFAULT_DTYPE)
    grad = []
    for idx in range(point.size):
        direction = jnp.zeros_like(point).at[idx].set(1.0)
        f_plus = func(point + direction * steps[idx])
        f_minus = func(point - direction * steps[idx])
        grad_val = (f_plus - f_minus) / (2.0 * steps[idx])
        grad.append(grad_val)
    return jnp.asarray(grad)


def finite_difference_hessian(
    func: ScalarFunc,
    point: jnp.ndarray,
    steps: jnp.ndarray
) -> jnp.ndarray:
    """Central-difference Hessian of a scalar function."""
    point = jnp.asarray(point, dtype=_DEFAULT_DTYPE)
    steps = jnp.asarray(steps, dtype=_DEFAULT_DTYPE)
    n_params = point.size
    f0 = func(point)
    hessian = jnp.zeros((n_params, n_params), dtype=_DEFAULT_DTYPE)

    for i in range(n_params):
        ei = jnp.zeros_like(point).at[i].set(1.0)
        f_plus = func(point + ei * steps[i])
        f_minus = func(point - ei * steps[i])
        second_derivative = (f_plus - 2.0 * f0 + f_minus) / (steps[i] ** 2)
        hessian = hessian.at[i, i].set(second_derivative)

        for j in range(i + 1, n_params):
            ej = jnp.zeros_like(point).at[j].set(1.0)
            f_pp = func(point + ei * steps[i] + ej * steps[j])
            f_pm = func(point + ei * steps[i] - ej * steps[j])
            f_mp = func(point - ei * steps[i] + ej * steps[j])
            f_mm = func(point - ei * steps[i] - ej * steps[j])
            mixed = (f_pp - f_pm - f_mp + f_mm) / (4.0 * steps[i] * steps[j])
            hessian = hessian.at[i, j].set(mixed)
            hessian = hessian.at[j, i].set(mixed)

    return hessian


def finite_difference_jacobian(
    func: ArrayLikeFunc,
    point: jnp.ndarray,
    steps: jnp.ndarray
) -> jnp.ndarray:
    """Central-difference Jacobian for vector-valued functions."""
    point = jnp.asarray(point, dtype=_DEFAULT_DTYPE)
    steps = jnp.asarray(steps, dtype=_DEFAULT_DTYPE)
    baseline = jnp.asarray(func(point), dtype=_DEFAULT_DTYPE)
    if baseline.ndim != 1:
        raise ValueError("Jacobian calculation expects 1D output vector")

    m = baseline.size
    n = point.size
    jacobian = jnp.zeros((m, n), dtype=_DEFAULT_DTYPE)

    for idx in range(n):
        direction = jnp.zeros_like(point).at[idx].set(1.0)
        f_plus = jnp.asarray(func(point + direction * steps[idx]))
        f_minus = jnp.asarray(func(point - direction * steps[idx]))
        column = (f_plus - f_minus) / (2.0 * steps[idx])
        jacobian = jacobian.at[:, idx].set(column)

    return jacobian
