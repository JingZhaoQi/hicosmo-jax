"""Numerical integration utilities used across HIcosmo.

These helpers avoid external dependencies (e.g. SciPy) while providing
sufficient accuracy for cosmological calculations that rely on repeated
one-dimensional integrals.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def simpson(y: np.ndarray, x: np.ndarray) -> float:
    """Composite Simpson integration for tabulated samples.

    Parameters
    ----------
    y : array_like
        Function values sampled at points ``x``.
    x : array_like
        Sample points (must be increasing).

    Returns
    -------
    float
        Approximation of ``∫ y(x) dx`` over the sampled interval.
    """
    if y.shape != x.shape:
        raise ValueError("y and x must have the same shape for Simpson integration")
    if y.size < 3:
        raise ValueError("At least three sample points are required for Simpson integration")

    # Simpson's rule requires an even number of intervals (odd number of samples)
    if (y.size - 1) % 2 == 1:
        y = y[:-1]
        x = x[:-1]

    h = np.diff(x)
    if not np.all(h > 0):
        raise ValueError("x must be strictly increasing for Simpson integration")

    # Use uniform spacing assumption for simplicity; for mildly non-uniform grids
    # this definition still yields accurate results because we operate in linear
    # space with many subdivisions.
    step = (x[-1] - x[0]) / (y.size - 1)
    s = y[0] + y[-1] + 4.0 * np.sum(y[1:-1:2]) + 2.0 * np.sum(y[2:-2:2])
    return s * step / 3.0


def integrate_simpson(func: Callable[[np.ndarray], np.ndarray], a: float, b: float, *,
                      num: int = 512) -> float:
    """Integrate ``func`` between ``a`` and ``b`` using Simpson's rule."""
    if b <= a:
        return 0.0
    # Simpson requires even number of intervals => num must be even
    if num % 2 == 1:
        num += 1
    x = np.linspace(a, b, num + 1)
    y = func(x)
    return simpson(y, x)


def integrate_logspace(func: Callable[[np.ndarray], np.ndarray], k_min: float,
                       k_max: float, *, num: int = 512) -> float:
    """Integrate ``func`` on a logarithmic grid for positive arguments."""
    if k_min <= 0:
        raise ValueError("k_min must be positive for log-space integration")
    if num % 2 == 1:
        num += 1
    x = np.logspace(np.log10(k_min), np.log10(k_max), num + 1)
    y = func(x)
    log_x = np.log(x)
    # Change of variable: ∫ f(k) dk = ∫ f(k(log)) k d(log k)
    integrand = y * x
    return simpson(integrand, log_x)
