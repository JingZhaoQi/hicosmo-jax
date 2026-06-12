"""JAX-first numerical utilities for HIcosmo.

This module centralizes common math operations (integration, differentiation,
root finding, ODE solving) so the rest of the codebase can reuse a consistent,
JAX-friendly implementation.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple
from functools import lru_cache

import jax
import jax.numpy as jnp
from jax import lax, vmap

Array = jnp.ndarray


# -----------------------------------------------------------------------------
# Integration utilities
# -----------------------------------------------------------------------------


def trapezoid(y: Array, x: Array, axis: int = -1) -> Array:
    """Trapezoidal integration wrapper (JAX)."""
    return jnp.trapezoid(y, x, axis=axis)


def simpson(y: Array, x: Array) -> Array:
    """Composite Simpson integration for tabulated samples (JAX)."""
    y = jnp.asarray(y)
    x = jnp.asarray(x)
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape for Simpson integration")
    if y.size < 3:
        raise ValueError(
            "At least three sample points are required for Simpson integration"
        )
    if (y.size - 1) % 2 == 1:
        y = y[:-1]
        x = x[:-1]
    step = (x[-1] - x[0]) / (y.size - 1)
    s = y[0] + y[-1] + 4.0 * jnp.sum(y[1:-1:2]) + 2.0 * jnp.sum(y[2:-2:2])
    return s * step / 3.0


def integrate_simpson(
    func: Callable[[Array], Array],
    a: float,
    b: float,
    *,
    num: int = 512,
) -> Array:
    """Integrate ``func`` between ``a`` and ``b`` using Simpson's rule (JAX)."""
    if num % 2 == 1:
        num += 1
    x = jnp.linspace(a, b, num + 1)
    y = func(x)
    result = simpson(y, x)
    return jnp.where(b <= a, jnp.asarray(0.0), result)


def integrate_logspace(
    func: Callable[[Array], Array],
    k_min: float,
    k_max: float,
    *,
    num: int = 512,
) -> Array:
    """Integrate ``func`` on a logarithmic grid for positive arguments."""
    if k_min <= 0:
        raise ValueError("k_min must be positive for log-space integration")
    if num % 2 == 1:
        num += 1
    x = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), num + 1)
    y = func(x)
    log_x = jnp.log(x)
    integrand = y * x
    return simpson(integrand, log_x)


_GAUSS_LEGENDRE_NODES_WEIGHTS: Dict[int, Tuple[Array, Array]] = {
    8: (
        jnp.array(
            [
                -0.9602898565,
                -0.7966664774,
                -0.5255324099,
                -0.1834346425,
                0.1834346425,
                0.5255324099,
                0.7966664774,
                0.9602898565,
            ]
        ),
        jnp.array(
            [
                0.1012285363,
                0.2223810345,
                0.3137066459,
                0.3626837834,
                0.3626837834,
                0.3137066459,
                0.2223810345,
                0.1012285363,
            ]
        ),
    ),
    12: (
        jnp.array(
            [
                -0.9815606342,
                -0.9041172564,
                -0.7699026741,
                -0.5873179543,
                -0.3678314990,
                -0.1252334085,
                0.1252334085,
                0.3678314990,
                0.5873179543,
                0.7699026741,
                0.9041172564,
                0.9815606342,
            ]
        ),
        jnp.array(
            [
                0.0471753364,
                0.1069393260,
                0.1600783286,
                0.2031674267,
                0.2334925365,
                0.2491470458,
                0.2491470458,
                0.2334925365,
                0.2031674267,
                0.1600783286,
                0.1069393260,
                0.0471753364,
            ]
        ),
    ),
    16: (
        jnp.array(
            [
                -0.9894009349,
                -0.9445750231,
                -0.8656312024,
                -0.7554044084,
                -0.6178762444,
                -0.4580167776,
                -0.2816035507,
                -0.0950125098,
                0.0950125098,
                0.2816035507,
                0.4580167776,
                0.6178762444,
                0.7554044084,
                0.8656312024,
                0.9445750231,
                0.9894009349,
            ]
        ),
        jnp.array(
            [
                0.0271524594,
                0.0622535239,
                0.0951585117,
                0.1246289712,
                0.1495959888,
                0.1691565194,
                0.1826034150,
                0.1894506105,
                0.1894506105,
                0.1826034150,
                0.1691565194,
                0.1495959888,
                0.1246289712,
                0.0951585117,
                0.0622535239,
                0.0271524594,
            ]
        ),
    ),
}


def gauss_legendre_nodes_weights(order: int) -> Tuple[Array, Array]:
    """Return Gauss-Legendre nodes/weights for supported orders."""
    if order not in _GAUSS_LEGENDRE_NODES_WEIGHTS:
        raise ValueError(f"Unsupported Gauss-Legendre order: {order}")
    return _GAUSS_LEGENDRE_NODES_WEIGHTS[order]


def gauss_legendre_integrate(
    integrand: Callable[[Array], Array],
    a: float,
    b: float,
    *,
    order: int = 12,
) -> Array:
    """Gauss-Legendre integration on [a, b] with fixed order."""
    nodes, weights = gauss_legendre_nodes_weights(order)
    b_arr = jnp.asarray(b)
    a_arr = jnp.asarray(a)
    z_mid = 0.5 * (b_arr + a_arr)
    z_half = 0.5 * (b_arr - a_arr)
    z_nodes = z_mid[..., None] + z_half[..., None] * nodes
    integrand_flat = vmap(integrand)(jnp.reshape(z_nodes, (-1,)))
    integrand_vals = jnp.reshape(integrand_flat, z_nodes.shape)
    return z_half * jnp.sum(weights * integrand_vals, axis=-1)


def gauss_legendre_integrate_batch(
    integrand: Callable[[Array], Array],
    z_array: Array,
    *,
    z_min: float = 0.0,
    order: int = 12,
) -> Array:
    """Batch Gauss-Legendre integration for multiple upper limits."""
    integrate_single = lambda z_max: gauss_legendre_integrate(
        integrand, z_min, z_max, order=order
    )
    return vmap(integrate_single)(z_array)


def cumulative_trapezoid(y: Array, x: Array) -> Array:
    """Cumulative trapezoidal integral with zero at the first point."""
    y = jnp.asarray(y)
    x = jnp.asarray(x)
    dx = x[1:] - x[:-1]
    avg = 0.5 * (y[:-1] + y[1:])
    cumulative = jnp.cumsum(avg * dx)
    return jnp.concatenate([jnp.array([0.0]), cumulative])


def interp_linspace(x_query: Array, x_grid: Array, values: Array) -> Array:
    """Linear interpolation on an EQUALLY-SPACED grid.

    Numerically equivalent to ``jnp.interp(x_query, x_grid, values)`` when
    ``x_grid`` is a uniform ``jnp.linspace`` (out-of-range queries clamp to
    the endpoint values, matching jnp.interp), but indexing is O(1) arithmetic
    instead of searchsorted, and the AD reverse path is a plain scatter-add.
    On the SN likelihood gradient this is ~30% faster end to end; jnp.interp's
    vjp dominated the gradient cost of the whole joint likelihood.

    Do NOT use with non-uniform grids — results would be silently wrong.
    """
    x_query = jnp.asarray(x_query)
    x_grid = jnp.asarray(x_grid)
    values = jnp.asarray(values)
    dx = x_grid[1] - x_grid[0]
    pos = (x_query - x_grid[0]) / dx
    idx = jnp.clip(jnp.floor(pos).astype(jnp.int32), 0, x_grid.shape[0] - 2)
    w = jnp.clip(pos - idx, 0.0, 1.0)
    return values[idx] * (1.0 - w) + values[idx + 1] * w


def integrate_batch_cumulative(
    func: Callable[[Array], Array],
    z_array: Array,
    *,
    z_min: float = 0.0,
    z_max: float | None = None,
    n_grid: int = 4096,
) -> Array:
    """Batch integration using a cumulative grid + interpolation."""
    z_array = jnp.asarray(z_array)
    z_max_val = jnp.max(z_array) if z_max is None else jnp.asarray(z_max)
    z_max_val = jnp.maximum(z_max_val, z_min)
    grid = jnp.linspace(z_min, z_max_val, n_grid)
    values = func(grid)
    cumulative = cumulative_trapezoid(values, grid)
    return jnp.interp(z_array, grid, cumulative)


def integrate_segmented(
    func: Callable[[Array], Array],
    a: float,
    b: float,
    *,
    n_segments: int = 8,
    order: int = 12,
) -> Array:
    """Segmented Gauss-Legendre integration on [a, b]."""
    edges = jnp.linspace(a, b, n_segments + 1)
    seg_a = edges[:-1]
    seg_b = edges[1:]

    def integrate_single(seg_bounds):
        seg_start, seg_end = seg_bounds
        return gauss_legendre_integrate(func, seg_start, seg_end, order=order)

    return jnp.sum(vmap(integrate_single)(jnp.stack([seg_a, seg_b], axis=1)))


def integrate_adaptive_simpson(
    func: Callable[[Array], Array],
    a: float,
    b: float,
    *,
    num: int = 64,
    max_num: int = 4096,
    tol: float = 1e-6,
) -> Array:
    """Adaptive Simpson integration using a fixed max grid and lax.while_loop."""
    if num % 2 == 1:
        raise ValueError("num must be even for Simpson integration")
    if max_num % num != 0:
        raise ValueError("max_num must be divisible by num")
    if max_num % 2 == 1:
        raise ValueError("max_num must be even for Simpson integration")

    weights, n_levels = _cached_simpson_weights(num, max_num)

    # Full grid evaluation once
    x_full = jnp.linspace(a, b, max_num + 1)
    y_full = func(x_full)

    integrals = (b - a) * (weights @ y_full)
    if n_levels == 1:
        return integrals[0]

    def cond(state):
        level, done = state
        return jnp.logical_and(level < n_levels - 1, jnp.logical_not(done))

    def body(state):
        level, done = state
        err = jnp.abs(integrals[level] - integrals[level - 1])
        done = err < tol
        next_level = jnp.where(done, level, level + 1)
        return next_level, done

    level, _ = lax.while_loop(cond, body, (1, False))
    return integrals[level]


@lru_cache(maxsize=16)
def _cached_simpson_weights(num: int, max_num: int) -> Tuple[Array, int]:
    """Precompute Simpson weights for all refinement levels."""
    levels = []
    n_val = num
    while n_val <= max_num:
        levels.append(n_val)
        n_val *= 2

    weights = []
    for n_val in levels:
        stride = max_num // n_val
        idx = jnp.arange(0, max_num + 1, stride)
        w_base = jnp.ones(n_val + 1)
        w_base = w_base.at[1:-1:2].set(4.0)
        w_base = w_base.at[2:-2:2].set(2.0)
        h = 1.0 / n_val
        w_full = jnp.zeros(max_num + 1)
        w_full = w_full.at[idx].set(w_base * h / 3.0)
        weights.append(w_full)
    return jnp.stack(weights, axis=0), len(levels)


# -----------------------------------------------------------------------------
# Differentiation utilities
# -----------------------------------------------------------------------------


def gradient_1d(values: Array, coords: Array) -> Array:
    """Simple first-derivative on 1D grids (JAX)."""
    values = jnp.asarray(values)
    coords = jnp.asarray(coords)
    if values.size < 2:
        return jnp.zeros_like(values)
    if values.size == 2:
        first = (values[1] - values[0]) / (coords[1] - coords[0])
        last = first
        return jnp.array([first, last])

    # Second-order accurate endpoints using 3-point Lagrange interpolation
    x0, x1, x2 = coords[0], coords[1], coords[2]
    f0, f1, f2 = values[0], values[1], values[2]
    c0 = (2.0 * x0 - x1 - x2) / ((x0 - x1) * (x0 - x2))
    c1 = (x0 - x2) / ((x1 - x0) * (x1 - x2))
    c2 = (x0 - x1) / ((x2 - x0) * (x2 - x1))
    first = c0 * f0 + c1 * f1 + c2 * f2

    xa, xb, xc = coords[-3], coords[-2], coords[-1]
    fa, fb, fc = values[-3], values[-2], values[-1]
    c0 = (xc - xb) / ((xa - xb) * (xa - xc))
    c1 = (xc - xa) / ((xb - xa) * (xb - xc))
    c2 = (2.0 * xc - xa - xb) / ((xc - xa) * (xc - xb))
    last = c0 * fa + c1 * fb + c2 * fc

    # Interior points (second-order accurate, non-uniform spacing)
    x_im1 = coords[:-2]
    x_i = coords[1:-1]
    x_ip1 = coords[2:]
    h1 = x_i - x_im1
    h2 = x_ip1 - x_i
    c_im1 = -h2 / (h1 * (h1 + h2))
    c_i = (h2 - h1) / (h1 * h2)
    c_ip1 = h1 / (h2 * (h1 + h2))
    interior = c_im1 * values[:-2] + c_i * values[1:-1] + c_ip1 * values[2:]

    return jnp.concatenate([jnp.array([first]), interior, jnp.array([last])])


def finite_difference_grad(
    func: Callable[[Array], Array],
    x: Array,
    *,
    step: float = 1e-5,
) -> Array:
    """Central finite-difference gradient for scalar outputs."""
    x = jnp.asarray(x, dtype=float)
    n = x.size

    def body(i, grad_out):
        direction = jnp.zeros_like(x).at[i].set(1.0)
        f_plus = func(x + direction * step)
        f_minus = func(x - direction * step)
        return grad_out.at[i].set((f_plus - f_minus) / (2.0 * step))

    return lax.fori_loop(0, n, body, jnp.zeros_like(x))


def grad(func: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return jax.grad(func)


def jacobian(func: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return jax.jacobian(func)


def hessian(func: Callable[[Array], Array]) -> Callable[[Array], Array]:
    return jax.hessian(func)


# -----------------------------------------------------------------------------
# Root finding utilities
# -----------------------------------------------------------------------------


def newton_root(
    func: Callable[[Array], Array],
    x0: Array,
    *,
    tol: float = 1e-8,
    max_iter: int = 64,
    deriv: Callable[[Array], Array] | None = None,
) -> Array:
    """Newton-Raphson root finding with optional derivative."""
    dfunc = deriv or jax.grad(func)

    def cond(state):
        _, fx, i = state
        return jnp.logical_and(i < max_iter, jnp.abs(fx) > tol)

    def body(state):
        x, fx, i = state
        dfx = dfunc(x)
        step = fx / dfx
        x_new = x - step
        fx_new = func(x_new)
        return x_new, fx_new, i + 1

    f0 = func(x0)
    x_out, _, _ = lax.while_loop(cond, body, (x0, f0, 0))
    return x_out


def bisection_root(
    func: Callable[[Array], Array],
    a: Array,
    b: Array,
    *,
    tol: float = 1e-8,
    max_iter: int = 128,
) -> Array:
    """Bisection root finding (requires sign change on [a, b])."""
    fa = func(a)
    fb = func(b)
    same_sign = jnp.sign(fa) * jnp.sign(fb) > 0

    def cond(state):
        a_val, b_val, _, _, i = state
        return jnp.logical_and(i < max_iter, (b_val - a_val) / 2.0 > tol)

    def body(state):
        a_val, b_val, fa_val, fb_val, i = state
        mid = 0.5 * (a_val + b_val)
        fm = func(mid)
        left = jnp.sign(fa_val) * jnp.sign(fm) <= 0
        a_next = jnp.where(left, a_val, mid)
        b_next = jnp.where(left, mid, b_val)
        fa_next = jnp.where(left, fa_val, fm)
        fb_next = jnp.where(left, fm, fb_val)
        return a_next, b_next, fa_next, fb_next, i + 1

    a_out, b_out, _, _, _ = lax.while_loop(cond, body, (a, b, fa, fb, 0))
    root = 0.5 * (a_out + b_out)
    return jnp.where(same_sign, jnp.nan, root)


# -----------------------------------------------------------------------------
# ODE utilities
# -----------------------------------------------------------------------------


def rk4_step(
    func: Callable[[Array, Array], Array], t: Array, y: Array, dt: Array
) -> Array:
    """One RK4 step."""
    k1 = func(t, y)
    k2 = func(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = func(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def odeint_rk4(
    func: Callable[[Array, Array], Array],
    y0: Array,
    t_grid: Array,
) -> Array:
    """Integrate ODE y' = f(t, y) on a fixed grid using RK4."""
    t_grid = jnp.asarray(t_grid)
    y0 = jnp.asarray(y0)
    dt = t_grid[1:] - t_grid[:-1]

    def step(y, t_dt):
        t, dt_val = t_dt
        y_next = rk4_step(func, t, y, dt_val)
        return y_next, y_next

    _, ys = lax.scan(step, y0, (t_grid[:-1], dt))
    return jnp.concatenate([jnp.expand_dims(y0, 0), ys], axis=0)


def solve_ivp_rk4(
    func: Callable[[Array, Array], Array],
    y0: Array,
    t0: float,
    t1: float,
    *,
    n_steps: int = 256,
) -> Tuple[Array, Array]:
    """Convenience fixed-step RK4 solver on [t0, t1]."""
    t_grid = jnp.linspace(t0, t1, n_steps + 1)
    y_grid = odeint_rk4(func, y0, t_grid)
    return t_grid, y_grid


# -----------------------------------------------------------------------------
# Cosmology utilities
# -----------------------------------------------------------------------------


@jax.jit
def sound_horizon_drag_eh98(
    H0: Array,
    Omega_m: Array,
    Omega_b: Array,
    T_cmb: Array,
) -> Array:
    """Sound horizon at drag epoch (Eisenstein & Hu 1998, CAMB-calibrated).

    The original EH98 formula has ~2.7% systematic error compared to CAMB.
    A constant calibration factor of 0.9736 is applied to match CAMB results.
    This factor is derived from comparison across multiple cosmologies and
    is stable to within 0.02% (see DESI 2024 VI, arXiv:2404.03002).
    """
    # CAMB calibration factor: CAMB_rd / EH98_rd = 0.9736 (stable across cosmologies)
    CAMB_CALIBRATION = 0.9736

    h = H0 / 100.0
    Omega_m_h2 = Omega_m * h**2
    Omega_b_h2 = Omega_b * h**2
    theta_cmb = T_cmb / 2.7

    b1 = 0.313 * Omega_m_h2 ** (-0.419) * (1.0 + 0.607 * Omega_m_h2**0.674)
    b2 = 0.238 * Omega_m_h2**0.223
    z_d = (
        1291.0
        * Omega_m_h2**0.251
        / (1.0 + 0.659 * Omega_m_h2**0.828)
        * (1.0 + b1 * Omega_b_h2**b2)
    )

    z_eq = 2.50e4 * Omega_m_h2 * theta_cmb**-4
    k_eq = 7.46e-2 * Omega_m_h2 * theta_cmb**-2

    R_eq = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z_eq)
    R_d = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z_d)

    sqrt_term = jnp.sqrt(6.0 / R_eq)
    log_arg = (jnp.sqrt(1.0 + R_d) + jnp.sqrt(R_d + R_eq)) / (1.0 + jnp.sqrt(R_eq))
    rd_eh98 = (2.0 / (3.0 * k_eq)) * sqrt_term * jnp.log(log_arg)

    return rd_eh98 * CAMB_CALIBRATION


@jax.jit
def sound_horizon_eh98(
    H0: Array,
    Omega_m: Array,
    Omega_b: Array,
    T_cmb: Array,
    z: Array,
) -> Array:
    """Sound horizon at redshift z (Eisenstein & Hu 1998 approximation)."""
    h = H0 / 100.0
    Omega_m_h2 = Omega_m * h**2
    Omega_b_h2 = Omega_b * h**2
    theta_cmb = T_cmb / 2.7

    z_eq = 2.50e4 * Omega_m_h2 * theta_cmb**-4
    k_eq = 7.46e-2 * Omega_m_h2 * theta_cmb**-2

    R_eq = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z_eq)
    R_z = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z)

    sqrt_term = jnp.sqrt(6.0 / R_eq)
    log_arg = (jnp.sqrt(1.0 + R_z) + jnp.sqrt(R_z + R_eq)) / (1.0 + jnp.sqrt(R_eq))
    return (2.0 / (3.0 * k_eq)) * sqrt_term * jnp.log(log_arg)


__all__ = [
    "trapezoid",
    "simpson",
    "integrate_simpson",
    "integrate_logspace",
    "gauss_legendre_nodes_weights",
    "gauss_legendre_integrate",
    "gauss_legendre_integrate_batch",
    "cumulative_trapezoid",
    "integrate_batch_cumulative",
    "integrate_segmented",
    "integrate_adaptive_simpson",
    "gradient_1d",
    "finite_difference_grad",
    "grad",
    "jacobian",
    "hessian",
    "newton_root",
    "bisection_root",
    "rk4_step",
    "odeint_rk4",
    "solve_ivp_rk4",
    "sound_horizon_drag_eh98",
    "sound_horizon_eh98",
    "interp_linspace",
]
