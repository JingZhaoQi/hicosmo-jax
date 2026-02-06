"""TDCOSMO hierarchical strong-lensing likelihood components."""

from __future__ import annotations

import dataclasses
import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import erf, erfinv, logsumexp

from ...utils.constants import c_km_s
from ...utils.logging import get_logger
from ..base import Likelihood, NuisanceList

logger = get_logger(__name__)
_DTYPE = jnp.float32
_TWO_PI = jnp.array(2.0 * jnp.pi, dtype=_DTYPE)
_EPS = jnp.array(1e-12, dtype=_DTYPE)
_NEG_LARGE = jnp.array(-1e12, dtype=_DTYPE)

_HERMITE_NODES = jnp.asarray(
    [
        -2.0201828704560856,
        -0.9585724646138185,
        0.0,
        0.9585724646138185,
        2.0201828704560856,
    ],
    dtype=_DTYPE,
)
_HERMITE_WEIGHTS = jnp.asarray(
    [
        0.01995324205904591,
        0.39361932315224116,
        0.9453087204829419,
        0.39361932315224116,
        0.01995324205904591,
    ],
    dtype=_DTYPE,
)
_HERMITE_NORM = jnp.sqrt(jnp.pi)


def _soft_indicator(
    x: jnp.ndarray, low: jnp.ndarray, high: jnp.ndarray, sharpness: float = 100.0
) -> jnp.ndarray:
    """Soft indicator function: ~1 if low <= x < high, ~0 otherwise.

    Uses sigmoid functions to create a differentiable approximation of the
    indicator function. The sharpness parameter controls how close to a
    step function this is (higher = sharper transition).

    NOTE: Uses moderate sharpness (100) for numerical stability. Higher values
    cause sigmoid overflow leading to NaN gradients.
    """
    # Clip sharpness*delta to prevent overflow in sigmoid
    # sigmoid(x) overflows for |x| > ~88 in float32
    max_arg = 20.0  # Safe range for sigmoid

    left_arg = sharpness * (x - low)
    right_arg = sharpness * (high - x)

    # Clip arguments to prevent overflow
    left_arg = jnp.clip(left_arg, -max_arg, max_arg)
    right_arg = jnp.clip(right_arg, -max_arg, max_arg)

    left = jax.nn.sigmoid(left_arg)
    right = jax.nn.sigmoid(right_arg)
    return left * right


def _bspline_basis_vector(
    x: jnp.ndarray, knots: jnp.ndarray, degree: int
) -> jnp.ndarray:
    """Evaluate all B-spline basis functions at a scalar x.

    Matches SciPy FITPACK spline evaluation when used with the (tx, ty, c) from
    `scipy.interpolate.RectBivariateSpline(...).tck`, with the important caveat
    that the caller should clamp `x` to the spline domain to emulate the constant
    extension used by `RectBivariateSpline` at out-of-range values.

    NOTE: Uses soft indicator functions for JAX autodiff compatibility.
    The sharpness parameter (1000.0) provides < 0.1% error vs hard step function
    while maintaining well-defined gradients for NUTS/HMC samplers.
    """

    x = jnp.asarray(x, dtype=_DTYPE)
    t = jnp.asarray(knots, dtype=_DTYPE)
    k = int(degree)
    n_basis = t.shape[0] - k - 1
    i = jnp.arange(n_basis, dtype=jnp.int32)

    t_i = t[i]
    t_ip1 = t[i + 1]

    # Use soft indicator instead of hard boolean comparison for differentiability
    n0 = _soft_indicator(x, t_i, t_ip1, sharpness=1000.0)

    # Handle boundary: when x == t[-1], the last basis function should be 1
    # Use soft version: if x is very close to t[-1], boost the last basis
    last_basis_mask = (i == n_basis - 1).astype(_DTYPE)
    boundary_boost = jax.nn.sigmoid(1000.0 * (x - t[-1] + _EPS)) * last_basis_mask
    n0 = n0 + boundary_boost * (1.0 - n0)

    n = n0
    for d in range(1, k + 1):
        denom1 = t[i + d] - t_i
        # Safe division: replace zero denom with 1.0 to avoid NaN in gradient computation
        # The where() below will select 0.0 anyway, but both branches are differentiated
        safe_denom1 = jnp.where(denom1 != 0.0, denom1, 1.0)
        term1 = jnp.where(denom1 != 0.0, (x - t_i) / safe_denom1 * n, 0.0)

        n_next = jnp.concatenate([n[1:], jnp.zeros((1,), dtype=_DTYPE)], axis=0)
        denom2 = t[i + d + 1] - t_ip1
        # Same safe division pattern
        safe_denom2 = jnp.where(denom2 != 0.0, denom2, 1.0)
        term2 = jnp.where(denom2 != 0.0, (t[i + d + 1] - x) / safe_denom2 * n_next, 0.0)

        n = term1 + term2
    return n


def _rect_bivariate_spline_eval(
    x: jnp.ndarray,
    y: jnp.ndarray,
    *,
    x_axis: jnp.ndarray,
    y_axis: jnp.ndarray,
    tx: jnp.ndarray,
    ty: jnp.ndarray,
    coeffs: jnp.ndarray,
    kx: int = 3,
    ky: int = 3,
) -> jnp.ndarray:
    """Evaluate a tensor-product B-spline for each bin.

    Parameters
    ----------
    x, y : scalar
        Evaluation coordinates.
    x_axis, y_axis : arrays
        Original tabulated axes; used only for clamping to emulate FITPACK's
        constant extension behavior.
    tx, ty : arrays
        Knot vectors from FITPACK (RectBivariateSpline.tck).
    coeffs : array
        Coefficients with shape (n_bins, n_basis_x, n_basis_y).
    """

    x_c = jnp.clip(jnp.asarray(x, dtype=_DTYPE), x_axis[0], x_axis[-1])
    y_c = jnp.clip(jnp.asarray(y, dtype=_DTYPE), y_axis[0], y_axis[-1])
    bx = _bspline_basis_vector(x_c, tx, int(kx))
    by = _bspline_basis_vector(y_c, ty, int(ky))
    return jnp.einsum("i,bij,j->b", bx, coeffs, by).astype(_DTYPE)


def _interp_1d_batch(
    x: jnp.ndarray, x_axis: jnp.ndarray, values: jnp.ndarray
) -> jnp.ndarray:
    """Fast batched 1D interpolation for values with shape (n_meas, n_axis)."""
    x = jnp.asarray(x, dtype=_DTYPE)
    x_axis = jnp.asarray(x_axis, dtype=_DTYPE)
    x = jnp.clip(x, x_axis[0], x_axis[-1])
    idx = jnp.searchsorted(x_axis, x, side="right") - 1
    idx = jnp.clip(idx, 0, x_axis.shape[0] - 2)
    x0 = x_axis[idx]
    x1 = x_axis[idx + 1]
    t = (x - x0) / jnp.maximum(x1 - x0, _EPS)
    v0 = jnp.take(values, idx, axis=1)  # (n_meas, n_draws)
    v1 = jnp.take(values, idx + 1, axis=1)
    return jnp.swapaxes((1.0 - t) * v0 + t * v1, 0, 1).astype(_DTYPE)


def _interp_2d_batch(
    x: jnp.ndarray,
    y: jnp.ndarray,
    x_axis: jnp.ndarray,
    y_axis: jnp.ndarray,
    values: jnp.ndarray,
) -> jnp.ndarray:
    """Batched bilinear interpolation for values with shape (n_meas, n_x, n_y)."""

    def _interp_single(xi, yi):
        xi = jnp.clip(xi, x_axis[0], x_axis[-1])
        yi = jnp.clip(yi, y_axis[0], y_axis[-1])

        i = jnp.searchsorted(x_axis, xi, side="right") - 1
        j = jnp.searchsorted(y_axis, yi, side="right") - 1
        i = jnp.clip(i, 0, x_axis.shape[0] - 2)
        j = jnp.clip(j, 0, y_axis.shape[0] - 2)

        x0 = x_axis[i]
        x1 = x_axis[i + 1]
        y0 = y_axis[j]
        y1 = y_axis[j + 1]

        t = (xi - x0) / jnp.maximum(x1 - x0, _EPS)
        u = (yi - y0) / jnp.maximum(y1 - y0, _EPS)

        v00 = values[:, i, j]
        v10 = values[:, i + 1, j]
        v01 = values[:, i, j + 1]
        v11 = values[:, i + 1, j + 1]

        return (
            (1.0 - t) * (1.0 - u) * v00
            + t * (1.0 - u) * v10
            + (1.0 - t) * u * v01
            + t * u * v11
        ).astype(_DTYPE)

    return jax.vmap(_interp_single)(x, y)


def _build_rect_bivariate_spline_coeffs(
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    grid_stack: np.ndarray,
    *,
    kx: int = 3,
    ky: int = 3,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Pre-compute FITPACK spline coefficients for a stack of 2D grids.

    Returns (tx, ty, coeffs_stack) where coeffs_stack has shape
    (n_bins, n_basis_x, n_basis_y). If SciPy is unavailable, returns (None, None, None).
    """

    try:
        from scipy.interpolate import RectBivariateSpline  # type: ignore
    except Exception:
        return None, None, None

    x_axis = np.asarray(x_axis, dtype="float64")
    y_axis = np.asarray(y_axis, dtype="float64")
    grid_stack = np.asarray(grid_stack, dtype="float64")
    if grid_stack.ndim != 3:
        raise ValueError("grid_stack must have shape (n_bins, n_x, n_y)")

    tx: Optional[np.ndarray] = None
    ty: Optional[np.ndarray] = None
    coeffs_list: list[np.ndarray] = []
    for idx in range(grid_stack.shape[0]):
        spline = RectBivariateSpline(
            x_axis,
            y_axis,
            grid_stack[idx],
            kx=int(kx),
            ky=int(ky),
            s=0.0,
        )
        tx_i, ty_i, c = spline.tck
        if tx is None:
            tx = np.asarray(tx_i, dtype="float32")
            ty = np.asarray(ty_i, dtype="float32")
        coeffs_list.append(
            np.asarray(c, dtype="float32").reshape(
                (len(x_axis), len(y_axis)), order="C"
            )
        )

    return tx, ty, np.stack(coeffs_list, axis=0)


def _gaussian_logpdf(
    value: jnp.ndarray, mean: jnp.ndarray, sigma: jnp.ndarray
) -> jnp.ndarray:
    """Log-density of a normal distribution with stability guards."""

    sigma = jnp.where(sigma > 0.0, sigma, jnp.nan)
    diff = (value - mean) / sigma
    return -0.5 * diff**2 - jnp.log(sigma * jnp.sqrt(_TWO_PI))


@dataclass
class KappaPrior:
    mean: float = 0.0
    # NOTE: By default we do *not* add an extra prior on kappa_ext because the
    # lens-specific `kappa_pdf` already represents the external convergence
    # distribution used in the official TDCOSMO analysis. Users can optionally
    # provide a non-zero sigma to incorporate additional information.
    sigma: float = 0.0

    def logpdf(self, value: jnp.ndarray) -> jnp.ndarray:
        if self.sigma <= 0.0:
            return jnp.array(0.0, dtype=_DTYPE)
        diff = (value - self.mean) / self.sigma
        return -0.5 * diff**2 - jnp.log(self.sigma * jnp.sqrt(_TWO_PI))


@dataclass
class TDCOSMOLensData:
    name: str
    z_lens: float
    z_source: float
    lambda_scaling: float
    ddt_centers: jnp.ndarray
    ddt_weights: jnp.ndarray
    bandwidth: float
    sigma_v_obs: jnp.ndarray
    cov_meas: jnp.ndarray
    cov_j_sqrt: jnp.ndarray
    j_model: jnp.ndarray
    ani_params: jnp.ndarray
    # Kinematic scaling grid. Most lenses provide a 1D grid in a_ani (beta_ani),
    # but some (e.g. RXJ1131) provide a 2D grid in (a_ani, gamma_pl).
    #
    # - `ani_scaling` stores the 1D scaling for backward compatibility (for 2D
    #   lenses this is the gamma_pl-marginalized scaling using the lens prior if
    #   available, or a flat weight over the tabulated gamma axis).
    # - `ani_scaling_2d` stores the full (a_ani, gamma_pl) grid when available.
    ani_scaling: jnp.ndarray  # shape (n_meas, n_ani)
    kappa_centers: jnp.ndarray
    kappa_pdf: jnp.ndarray
    kappa_min: float
    kappa_max: float
    ddt_norm_factor: float
    gamma_pl_params: Optional[jnp.ndarray] = None  # shape (n_gamma,)
    ani_scaling_2d: Optional[jnp.ndarray] = None  # shape (n_meas, n_ani, n_gamma)
    gamma_pl_prior_mean: Optional[float] = None
    gamma_pl_prior_sigma: Optional[float] = None
    gamma_pl_pivot: Optional[float] = None
    gamma_pl_ddt_slope: Optional[float] = None
    ani_spline_tx: Optional[jnp.ndarray] = None
    ani_spline_ty: Optional[jnp.ndarray] = None
    ani_spline_coeffs: Optional[jnp.ndarray] = None  # (n_meas, n_ani, n_gamma)
    # Optional axisymmetric JAM correction draws (sigma_axi/sigma_sph). When provided,
    # this is sampled (deterministically) alongside other distributions and applied as
    # a multiplicative scaling of sigma_v predictions.
    vel_disp_scaling_samples: Optional[jnp.ndarray] = None

    def ddt_logpdf(self, value: jnp.ndarray) -> jnp.ndarray:
        value_exp = jnp.expand_dims(value, axis=-1)
        centers = jnp.asarray(self.ddt_centers, dtype=_DTYPE)
        diff = (value_exp - centers) / self.bandwidth
        log_kernel = -0.5 * diff**2 - jnp.log(self.bandwidth * jnp.sqrt(_TWO_PI))
        log_weights = jnp.log(self.ddt_weights + _EPS)
        return logsumexp(log_kernel + log_weights, axis=-1) - self.ddt_norm_factor

    def kappa_logpdf(self, kappa: jnp.ndarray) -> jnp.ndarray:
        pdf = jnp.interp(kappa, self.kappa_centers, self.kappa_pdf)
        pdf = jnp.maximum(pdf, _EPS)
        return jnp.log(pdf)

    def anisotropy_scaling(
        self,
        ani_param: Optional[jnp.ndarray],
        gamma_pl: Optional[jnp.ndarray] = None,
        *,
        use_spline: bool = True,
    ) -> jnp.ndarray:
        if ani_param is None:
            return jnp.ones(self.sigma_v_obs.shape, dtype=_DTYPE)
        ani_param = jnp.asarray(ani_param, dtype=_DTYPE)
        if (
            self.gamma_pl_params is None
            or self.ani_scaling_2d is None
            or gamma_pl is None
        ):
            ani_param = jnp.clip(ani_param, self.ani_params[0], self.ani_params[-1])
            return jax.vmap(lambda row: jnp.interp(ani_param, self.ani_params, row))(
                self.ani_scaling
            ).astype(_DTYPE)

        gamma_pl = jnp.asarray(gamma_pl, dtype=_DTYPE)
        if (
            use_spline
            and self.ani_spline_tx is not None
            and self.ani_spline_ty is not None
            and self.ani_spline_coeffs is not None
        ):
            return _rect_bivariate_spline_eval(
                ani_param,
                gamma_pl,
                x_axis=self.ani_params,
                y_axis=self.gamma_pl_params,
                tx=self.ani_spline_tx,
                ty=self.ani_spline_ty,
                coeffs=self.ani_spline_coeffs,
            )

        x_axis = self.ani_params
        y_axis = self.gamma_pl_params
        values = self.ani_scaling_2d  # (n_meas, n_ani, n_gamma)

        ani_param = jnp.clip(ani_param, x_axis[0], x_axis[-1])
        gamma_pl = jnp.clip(gamma_pl, y_axis[0], y_axis[-1])

        i = jnp.searchsorted(x_axis, ani_param, side="right") - 1
        j = jnp.searchsorted(y_axis, gamma_pl, side="right") - 1
        i = jnp.clip(i, 0, x_axis.shape[0] - 2)
        j = jnp.clip(j, 0, y_axis.shape[0] - 2)

        x0 = x_axis[i]
        x1 = x_axis[i + 1]
        y0 = y_axis[j]
        y1 = y_axis[j + 1]

        t = (ani_param - x0) / jnp.maximum(x1 - x0, _EPS)
        u = (gamma_pl - y0) / jnp.maximum(y1 - y0, _EPS)

        v00 = values[:, i, j]
        v10 = values[:, i + 1, j]
        v01 = values[:, i, j + 1]
        v11 = values[:, i + 1, j + 1]

        return (
            (1.0 - t) * (1.0 - u) * v00
            + t * (1.0 - u) * v10
            + (1.0 - t) * u * v01
            + t * u * v11
        ).astype(_DTYPE)

    def anisotropy_scaling_batch(
        self,
        ani_param: jnp.ndarray,
        gamma_pl: Optional[jnp.ndarray] = None,
        *,
        use_spline: bool = True,
    ) -> jnp.ndarray:
        """Vectorized anisotropy scaling for an array of ani_param values."""
        ani_param = jnp.asarray(ani_param, dtype=_DTYPE)
        if ani_param.ndim == 0:
            return self.anisotropy_scaling(ani_param, gamma_pl, use_spline=use_spline)[
                None, :
            ]

        if (
            self.gamma_pl_params is None
            or self.ani_scaling_2d is None
            or gamma_pl is None
        ):
            # Fast batched 1D interpolation: returns (n_draws, n_meas)
            ani_param = jnp.clip(ani_param, self.ani_params[0], self.ani_params[-1])
            idx = jnp.searchsorted(self.ani_params, ani_param, side="right") - 1
            idx = jnp.clip(idx, 0, self.ani_params.shape[0] - 2)
            x0 = self.ani_params[idx]
            x1 = self.ani_params[idx + 1]
            t = (ani_param - x0) / jnp.maximum(x1 - x0, _EPS)
            v0 = jnp.take(self.ani_scaling, idx, axis=1)  # (n_meas, n_draws)
            v1 = jnp.take(self.ani_scaling, idx + 1, axis=1)
            return jnp.swapaxes((1.0 - t) * v0 + t * v1, 0, 1).astype(_DTYPE)

        gamma_pl = jnp.asarray(gamma_pl, dtype=_DTYPE)
        if (
            use_spline
            and self.ani_spline_tx is not None
            and self.ani_spline_ty is not None
            and self.ani_spline_coeffs is not None
        ):
            return jax.vmap(
                lambda a, g: _rect_bivariate_spline_eval(
                    a,
                    g,
                    x_axis=self.ani_params,
                    y_axis=self.gamma_pl_params,
                    tx=self.ani_spline_tx,
                    ty=self.ani_spline_ty,
                    coeffs=self.ani_spline_coeffs,
                )
            )(ani_param, gamma_pl)
        return _interp_2d_batch(
            ani_param,
            gamma_pl,
            self.ani_params,
            self.gamma_pl_params,
            self.ani_scaling_2d,
        )


def _downsample_pdf_quantiles(
    centers: np.ndarray,
    weights: np.ndarray,
    max_nodes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Downsample a discrete PDF with deterministic quantile points.

    This approximates sampling from the distribution (like hierarc's Monte Carlo
    marginalization) but keeps the likelihood deterministic (important for JAX / HMC).
    """
    centers = np.asarray(centers, dtype="float64")
    weights = np.asarray(weights, dtype="float64")
    if centers.ndim != 1 or weights.ndim != 1:
        raise ValueError("centers and weights must be 1D arrays")
    if centers.shape[0] != weights.shape[0]:
        raise ValueError("centers and weights must have the same length")
    if centers.shape[0] <= max_nodes:
        return centers.astype("float32"), weights.astype("float32")

    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        idx = np.linspace(0, centers.shape[0] - 1, max_nodes).round().astype(int)
        idx = np.unique(idx)
        return centers[idx].astype("float32"), np.ones(idx.shape[0], dtype="float32")

    probs = weights / total
    cdf = np.cumsum(probs)
    cdf[-1] = 1.0

    quantiles = (np.arange(max_nodes, dtype="float64") + 0.5) / float(max_nodes)
    idx = np.searchsorted(cdf, quantiles, side="left")
    idx = np.clip(idx, 0, centers.shape[0] - 1)

    unique_idx, counts = np.unique(idx, return_counts=True)
    centers_ds = centers[unique_idx]
    weights_ds = counts.astype("float64") / float(max_nodes)
    return centers_ds.astype("float32"), weights_ds.astype("float32")


def _downsample_rows_evenly(samples: np.ndarray, max_rows: int) -> np.ndarray:
    """Deterministically subsample rows from an (N, ...) array for speed.

    Used for large empirical distributions (e.g. axisymmetric JAM correction draws).
    """
    arr = np.asarray(samples)
    if arr.shape[0] <= max_rows:
        return arr
    idx = np.linspace(0, arr.shape[0] - 1, max_rows).round().astype(int)
    idx = np.unique(idx)
    return arr[idx]


def _load_processed_lens(
    path: Path,
    lambda_scaling: float,
    *,
    max_kappa_nodes: Optional[int] = None,
) -> TDCOSMOLensData:
    """Load processed lens data, supporting both old and TDCOSMO2025 formats."""
    with path.open("rb") as f:
        data = pickle.load(f)

    # Lambda scaling from lens properties (r_eff/theta_E - 1) is required for the
    # alpha_lambda correlation term in the hierarchical MST model. The official
    # TDCOSMO2025 release stores this in `kwargs_lens_properties`; fall back to the
    # caller-provided value for legacy processed datasets.
    lambda_scaling_value = float(lambda_scaling)
    if "lambda_scaling_property" in data:
        try:
            lambda_scaling_value = float(data["lambda_scaling_property"])
        except (TypeError, ValueError):
            lambda_scaling_value = float(lambda_scaling)
    else:
        props = data.get("kwargs_lens_properties", {}) or {}
        try:
            r_eff = float(props.get("r_eff", 0.0))
            theta_e = float(props.get("theta_E", 0.0))
            if theta_e != 0.0:
                lambda_scaling_value = r_eff / theta_e - 1.0
        except (TypeError, ValueError):
            lambda_scaling_value = float(lambda_scaling)

    ddt_samples = data["ddt_samples"].astype("float32")
    weights = data.get("ddt_weights")
    if weights is None:
        weights = np.ones_like(ddt_samples)
    weights = weights.astype("float32")
    nbins = data.get("nbins_hist", 200)
    hist, edges = np.histogram(ddt_samples, bins=nbins, weights=weights, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = edges[1:] - edges[:-1]
    mask = hist > 0.0
    centers = centers[mask]
    hist = hist[mask]
    widths = bin_widths[mask]
    hist_weights = hist * widths
    hist_weights = hist_weights / np.sum(hist_weights)

    ddt_sigma_norm = float(np.std(ddt_samples))
    ddt_sigma_norm = max(ddt_sigma_norm, 1e-12)
    ddt_norm_factor = np.log(1.0 / (ddt_sigma_norm * np.sqrt(2.0 * np.pi)))

    sigma_v_obs = np.asarray(data["sigma_v_measurement"], dtype="float32")
    cov_meas = np.asarray(data["error_cov_measurement"], dtype="float32")
    cov_j_sqrt = np.asarray(data["error_cov_j_sqrt"], dtype="float32")
    if cov_meas.ndim == 0:
        cov_meas = np.array([[cov_meas]], dtype="float32")
    if cov_j_sqrt.ndim == 0:
        cov_j_sqrt = np.array([[cov_j_sqrt]], dtype="float32")
    j_model = np.asarray(data["j_model"], dtype="float32")

    # Handle anisotropy scaling - support both old and new TDCOSMO2025 format.
    gamma_pl_params = None
    ani_scaling_2d = None
    gamma_pl_prior_mean = None
    gamma_pl_prior_sigma = None
    ani_spline_tx = None
    ani_spline_ty = None
    ani_spline_coeffs = None
    gamma_pl_pivot = data.get("gamma_pl_pivot")
    gamma_pl_ddt_slope = data.get("gamma_pl_ddt_slope")
    if "j_kin_scaling_param_axes" in data and "j_kin_scaling_grid_list" in data:
        # TDCOSMO2025 format
        axes = data["j_kin_scaling_param_axes"]
        grids = data["j_kin_scaling_grid_list"]
        param_list = data.get("kin_scaling_param_list", ["a_ani"])

        # Check if we have 2D grid (a_ani × gamma_pl) or 1D (a_ani only)
        if len(param_list) >= 2 and "gamma_pl" in param_list:
            # 2D grid (a_ani × gamma_pl). Keep the full grid to enable proper
            # marginalization of the likelihood (not of the scaling factors).
            if param_list[0] == "a_ani":
                a_ani_idx, gamma_pl_idx = 0, 1
            else:
                a_ani_idx, gamma_pl_idx = 1, 0

            ani_params = np.asarray(axes[a_ani_idx], dtype="float32")
            gamma_pl_axis = np.asarray(axes[gamma_pl_idx], dtype="float32")
            gamma_pl_params = gamma_pl_axis

            for item in data.get("prior_list", []) or []:
                if (
                    isinstance(item, (list, tuple))
                    and len(item) >= 3
                    and item[0] == "gamma_pl"
                ):
                    gamma_pl_prior_mean = float(item[1])
                    gamma_pl_prior_sigma = float(item[2])
                    break

            gamma_weights = None
            if (
                gamma_pl_prior_mean is not None
                and gamma_pl_prior_sigma is not None
                and gamma_pl_prior_sigma > 0
            ):
                weights = np.exp(
                    -0.5
                    * ((gamma_pl_axis - gamma_pl_prior_mean) / gamma_pl_prior_sigma)
                    ** 2
                )
                if np.all(np.isfinite(weights)) and float(np.sum(weights)) > 0:
                    gamma_weights = weights / np.sum(weights)
            if gamma_weights is None:
                gamma_weights = np.ones_like(gamma_pl_axis) / float(len(gamma_pl_axis))

            scaling_2d_list = []
            ani_scaling_list = []
            for grid in grids:
                grid = np.asarray(grid, dtype="float32")
                if a_ani_idx == 0:
                    grid_norm = grid  # (n_ani, n_gamma)
                else:
                    grid_norm = grid.T  # (n_ani, n_gamma)
                scaling_2d_list.append(grid_norm)
                ani_scaling_list.append(
                    np.sum(grid_norm * gamma_weights[None, :], axis=1)
                )

            ani_scaling_2d = np.stack(
                scaling_2d_list, axis=0
            )  # (n_meas, n_ani, n_gamma)
            ani_scaling = np.stack(ani_scaling_list, axis=0)  # (n_meas, n_ani)

            tx, ty, coeffs = _build_rect_bivariate_spline_coeffs(
                ani_params,
                gamma_pl_axis,
                ani_scaling_2d,
            )
            if tx is not None and ty is not None and coeffs is not None:
                ani_spline_tx = tx
                ani_spline_ty = ty
                ani_spline_coeffs = coeffs
        else:
            # 1D grid: only a_ani (most lenses)
            ani_params = np.asarray(axes[0], dtype="float32")
            ani_scaling_list = []
            for grid in grids:
                grid = np.asarray(grid, dtype="float32")
                # grid is 1D array of length n_ani
                ani_scaling_list.append(grid.flatten())
            ani_scaling = np.stack(ani_scaling_list, axis=0)
    elif "ani_param_array" in data:
        # Old format: 1D anisotropy scaling
        ani_params = np.asarray(data.get("ani_param_array", [0.0]), dtype="float32")
        scaling_list = data.get("ani_scaling_array_list", None)
        if scaling_list is None or len(scaling_list) == 0:
            ani_scaling = np.ones((len(sigma_v_obs), len(ani_params)), dtype="float32")
        else:
            ani_scaling = np.stack(
                [np.asarray(row, dtype="float32") for row in scaling_list], axis=0
            )
    else:
        # Fallback: no anisotropy scaling
        ani_params = np.array([0.0], dtype="float32")
        ani_scaling = np.ones((len(sigma_v_obs), 1), dtype="float32")

    kappa_edges = np.asarray(data["kappa_bin_edges"], dtype="float32")
    kappa_pdf = np.asarray(data["kappa_pdf"], dtype="float32")
    kappa_centers = 0.5 * (kappa_edges[:-1] + kappa_edges[1:])
    if max_kappa_nodes is not None:
        kappa_centers, kappa_pdf = _downsample_pdf_quantiles(
            kappa_centers, kappa_pdf, max_kappa_nodes
        )

    return TDCOSMOLensData(
        name=data["name"],
        z_lens=float(data["z_lens"]),
        z_source=float(data["z_source"]),
        lambda_scaling=lambda_scaling_value,
        ddt_centers=jnp.asarray(centers, dtype=_DTYPE),
        ddt_weights=jnp.asarray(hist_weights, dtype=_DTYPE),
        bandwidth=float(data.get("bandwidth", 20.0)),
        sigma_v_obs=jnp.asarray(sigma_v_obs, dtype=_DTYPE),
        cov_meas=jnp.asarray(cov_meas, dtype=_DTYPE),
        cov_j_sqrt=jnp.asarray(cov_j_sqrt, dtype=_DTYPE),
        j_model=jnp.asarray(j_model, dtype=_DTYPE),
        ani_params=jnp.asarray(ani_params, dtype=_DTYPE),
        ani_scaling=jnp.asarray(ani_scaling, dtype=_DTYPE),
        kappa_centers=jnp.asarray(kappa_centers, dtype=_DTYPE),
        kappa_pdf=jnp.asarray(kappa_pdf, dtype=_DTYPE),
        kappa_min=float(kappa_centers.min()),
        kappa_max=float(kappa_centers.max()),
        ddt_norm_factor=ddt_norm_factor,
        gamma_pl_params=(
            jnp.asarray(gamma_pl_params, dtype=_DTYPE)
            if gamma_pl_params is not None
            else None
        ),
        ani_scaling_2d=(
            jnp.asarray(ani_scaling_2d, dtype=_DTYPE)
            if ani_scaling_2d is not None
            else None
        ),
        gamma_pl_prior_mean=gamma_pl_prior_mean,
        gamma_pl_prior_sigma=gamma_pl_prior_sigma,
        gamma_pl_pivot=(float(gamma_pl_pivot) if gamma_pl_pivot is not None else None),
        gamma_pl_ddt_slope=(
            float(gamma_pl_ddt_slope) if gamma_pl_ddt_slope is not None else None
        ),
        ani_spline_tx=(
            jnp.asarray(ani_spline_tx, dtype=_DTYPE)
            if ani_spline_tx is not None
            else None
        ),
        ani_spline_ty=(
            jnp.asarray(ani_spline_ty, dtype=_DTYPE)
            if ani_spline_ty is not None
            else None
        ),
        ani_spline_coeffs=(
            jnp.asarray(ani_spline_coeffs, dtype=_DTYPE)
            if ani_spline_coeffs is not None
            else None
        ),
        vel_disp_scaling_samples=None,
    )


def _sanitize(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", name)


class TDCOSMOLikelihood(Likelihood):
    """
    TDCOSMO strong-lensing likelihood with hierarchical modeling.

    Full hierarchical Bayesian analysis of strong-lensing time delays,
    including mass-sheet transformation (MST) and stellar anisotropy parameters.

    Parameters
    ----------
    cosmology_class : type, optional
        Cosmology model class (LCDM, wCDM, CPL). Required for direct callable interface.
    lens_names : iterable of str, optional
        Subset of lenses to use. If None, uses all available.
    lens_priors : dict, optional
        Custom KappaPrior for each lens (external convergence).
    lambda_bounds : tuple
        Bounds for internal MST parameter (default: 0.5-1.5).
    anisotropy_bounds : tuple
        Bounds for anisotropy parameter (default: 0.1-5.0).

    Nuisance Parameters
    -------------------
    - lambda_int_mean : Internal MST population mean (~1.0)
    - lambda_int_sigma : Internal MST scatter (~0.05)
    - alpha_lambda : Slope of λ with R_eff/θ_E (~0.0)
    - a_ani_mean : Stellar anisotropy mean (~1.0)
    - a_ani_sigma : Anisotropy scatter (~0.1)

    Examples
    --------
    >>> from hicosmo.likelihoods import TDCOSMOLikelihood
    >>> from hicosmo.models import LCDM
    >>> tdcosmo = TDCOSMOLikelihood(cosmology_class=LCDM)
    >>> log_L = tdcosmo(H0=70, Omega_m=0.3, lambda_int_mean=1.0, a_ani_mean=1.0)

    >>> # Get nuisance parameters for MCMC
    >>> nuisance = tdcosmo.nuisance_parameters()
    >>> # Returns: [('lambda_int_mean', 1.0, 0.5, 1.5), ...]
    """

    def __init__(
        self,
        cosmology_class: Optional[type] = None,
        lens_names: Optional[Iterable[str]] = None,
        lens_priors: Optional[Dict[str, KappaPrior]] = None,
        data_path: Optional[str] = None,
        name: Optional[str] = None,
        lambda_bounds: Tuple[float, float] = (0.5, 1.5),
        anisotropy_bounds: Tuple[float, float] = (0.1, 5.0),
        gamma_pl_bounds: Tuple[float, float] = (1.1, 2.9),
        normalized: bool = False,
        gamma_pl_sampling: bool = True,
        log_scatter_prior: bool = True,
        omega_m_prior: Optional[Tuple[float, float]] = None,
        default_omega_b: float = 0.049,
        use_tdcosmo2025: bool = False,
        anisotropy_model: str = "const",
        anisotropy_parameterization: str = "beta",
        kin_axi_correction: bool = False,
        sigma_sys_error_include: bool = False,
        num_distribution_draws: int = 200,
        distribution_seed: int = 0,
        use_spline: bool = True,
        max_kappa_nodes: Optional[int] = None,
        max_vel_disp_nodes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self.cosmology_class = cosmology_class
        self.use_tdcosmo2025 = use_tdcosmo2025
        self.anisotropy_model = anisotropy_model
        self.anisotropy_parameterization = str(anisotropy_parameterization).upper()
        self.kin_axi_correction = bool(kin_axi_correction)
        self.use_spline = bool(use_spline)
        # Match hierarc: sigma_sys_error_include=False by default (TDCOSMO only mode)
        # Set to True when combining with external lens samples (SLACS/SL2S)
        self.sigma_sys_error_include = bool(sigma_sys_error_include)
        # Match hierarc defaults (CosmoLikelihood(normalized=False)) for Fig. 8–12.
        # When normalized=False, the kinematics likelihood omits the parameter-dependent
        # Gaussian normalization term (log det covariance). This is what TDCOSMO2025
        # uses in its public sampling scripts.
        self.normalized = bool(normalized)
        # Match hierarc: power-law slope(s) are sampled as explicit parameters (and
        # only enter the Monte Carlo marginalization through the kinematic scaling
        # and the Ddt shift). This is required to reproduce the public chains.
        self.gamma_pl_sampling = bool(gamma_pl_sampling)
        if (
            self.anisotropy_model == "const"
            and self.anisotropy_parameterization
            not in {
                "BETA",
                "TAN_RAD",
            }
        ):
            raise ValueError(
                "anisotropy_parameterization must be 'beta' or 'TAN_RAD' for "
                "anisotropy_model='const'."
            )
        self.num_distribution_draws = int(num_distribution_draws)
        if self.num_distribution_draws <= 0:
            raise ValueError("num_distribution_draws must be a positive integer.")
        self.distribution_seed = int(distribution_seed)
        base = (np.arange(self.num_distribution_draws, dtype="float32") + 0.5) / float(
            self.num_distribution_draws
        )
        rng = np.random.default_rng(self.distribution_seed)
        self._quantile_base = jnp.asarray(base, dtype=_DTYPE)
        self._quantile_perm_lam = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_ani = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_kappa = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_vel = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_gamma = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        if max_kappa_nodes is None:
            if use_tdcosmo2025:
                # hierarc uses `num_distribution_draws=200` for LOS sampling; matching
                # that scale keeps sampling fast while remaining accurate.
                max_kappa_nodes = 200
        elif max_kappa_nodes <= 0:
            # Explicit opt-out (use all original nodes).
            max_kappa_nodes = None
        self.max_kappa_nodes = max_kappa_nodes
        if max_vel_disp_nodes is None:
            if self.kin_axi_correction and use_tdcosmo2025:
                # Empirical JAM correction draws can be huge; we only need enough
                # support to match num_distribution_draws.
                max_vel_disp_nodes = self.num_distribution_draws
        elif max_vel_disp_nodes <= 0:
            max_vel_disp_nodes = None
        self.max_vel_disp_nodes = max_vel_disp_nodes

        # Determine data path
        if data_path is not None:
            self.data_path = Path(data_path)
        elif use_tdcosmo2025:
            # Use TDCOSMO2025_public data
            # Go up 3 levels: lensing/ -> likelihoods/ -> hicosmo/ -> project root
            tdcosmo2025_path = (
                Path(__file__).resolve().parents[3]
                / "TDCOSMO2025_public"
                / "TDCOSMO_sample"
            )
            if tdcosmo2025_path.exists():
                self.data_path = tdcosmo2025_path
            else:
                raise ValueError(
                    f"TDCOSMO2025_public data not found at {tdcosmo2025_path}. "
                    "Please clone: git clone https://github.com/TDCOSMO/TDCOSMO2025_public"
                )
        else:
            # Go up 2 levels: lensing/ -> likelihoods/ -> hicosmo/
            self.data_path = (
                Path(__file__).resolve().parents[2] / "data" / "tdcosmo" / "processed"
            )

        # Determine file suffix based on data format
        if use_tdcosmo2025:
            file_suffix = f"_{anisotropy_model}_processed.pkl"
        else:
            file_suffix = "_processed.pkl"

        # Find available lenses
        raw_names = (
            list(lens_names)
            if lens_names is not None
            else [
                p.stem.replace(file_suffix.replace(".pkl", ""), "")
                for p in sorted(self.data_path.glob(f"*{file_suffix}"))
            ]
        )
        if not raw_names:
            raise ValueError(f"No TDCOSMO lenses found in {self.data_path}")

        self.lens_data: Dict[str, TDCOSMOLensData] = {}
        scaling_map = self._load_lambda_scaling()
        for lname in raw_names:
            sanitized = _sanitize(lname)
            scaling = scaling_map.get(sanitized, 0.0)
            data = _load_processed_lens(
                self.data_path / f"{lname}{file_suffix}",
                scaling,
                max_kappa_nodes=self.max_kappa_nodes,
            )
            data = dataclasses.replace(data, name=sanitized)
            self.lens_data[sanitized] = data
        self.lens_names = list(self.lens_data.keys())
        self._gamma_param_map: Dict[str, str] = {}
        if self.gamma_pl_sampling:
            for lname, lens in self.lens_data.items():
                if lens.gamma_pl_params is None or lens.ani_scaling_2d is None:
                    continue
                # Use lens-specific names to avoid collisions across likelihood components.
                self._gamma_param_map[lname] = f"gamma_pl_{lname}"

        if self.kin_axi_correction and use_tdcosmo2025:
            # Go up 3 levels: lensing/ -> likelihoods/ -> hicosmo/ -> project root
            corr_path = (
                Path(__file__).resolve().parents[3]
                / "TDCOSMO2025_public"
                / "kin_axi_jam_scaling"
                / "tdcosmo_correction.pickle"
            )
            if corr_path.exists():
                with corr_path.open("rb") as f:
                    corr_list = pickle.load(f)
                corr_map = {
                    _sanitize(entry.get("name", "")): np.asarray(
                        entry.get("correction_combined", []), dtype="float32"
                    )
                    for entry in corr_list
                }
                for lname, lens in list(self.lens_data.items()):
                    corr = corr_map.get(lname)
                    if corr is None or corr.size == 0:
                        continue
                    if self.max_vel_disp_nodes is not None:
                        corr = _downsample_rows_evenly(
                            corr, int(self.max_vel_disp_nodes)
                        )
                    self.lens_data[lname] = dataclasses.replace(
                        lens,
                        vel_disp_scaling_samples=jnp.asarray(corr, dtype=_DTYPE),
                    )
            else:
                raise ValueError(
                    f"Axisymmetric correction requested but not found: {corr_path}"
                )

        self.lens_priors: Dict[str, KappaPrior] = {}
        custom_priors = lens_priors or {}
        for key, prior in custom_priors.items():
            self.lens_priors[_sanitize(key)] = prior
        for name in self.lens_names:
            self.lens_priors.setdefault(name, KappaPrior())
        if use_tdcosmo2025 and max_kappa_nodes is not None and max_kappa_nodes < 16:
            raise ValueError("max_kappa_nodes must be >= 16 for stable integration.")

        # For TDCOSMO2025, set sensible default anisotropy bounds depending on
        # the chosen parameterization.
        #
        # Note: the processed likelihood grids are tabulated in beta_ani for the
        # constant anisotropy model. When sampling in TAN_RAD (sigma_t/sigma_r),
        # we transform to beta_ani internally.
        if use_tdcosmo2025 and anisotropy_bounds == (0.1, 5.0):
            if (
                self.anisotropy_model == "const"
                and self.anisotropy_parameterization == "TAN_RAD"
            ):
                # Paper/Table 3 baseline prior: U(0.87, 1.12) on <sigma_t/sigma_r>.
                anisotropy_bounds = (0.87, 1.12)
            else:
                grid_min = float(
                    min(float(l.ani_params[0]) for l in self.lens_data.values())
                )
                grid_max = float(
                    max(float(l.ani_params[-1]) for l in self.lens_data.values())
                )
                anisotropy_bounds = (grid_min, grid_max)

        if lambda_bounds[0] >= lambda_bounds[1]:
            raise ValueError("lambda_bounds must satisfy min < max")
        if anisotropy_bounds[0] >= anisotropy_bounds[1]:
            raise ValueError("anisotropy_bounds must satisfy min < max")
        self.lambda_bounds = lambda_bounds
        self.anisotropy_bounds = anisotropy_bounds
        self.gamma_pl_bounds = gamma_pl_bounds
        self.log_scatter_prior = log_scatter_prior
        self.omega_m_prior = omega_m_prior
        self.default_omega_b = default_omega_b
        super().__init__(
            name=name or "tdcosmo", data_path=str(self.data_path), **kwargs
        )
        self.initialize()

        self._packed_data = None
        self._use_packed = False
        if self.lens_data and (not self.use_spline):
            self._build_packed_data()

        # Print summary
        if cosmology_class is not None:
            logger.info(
                f"TDCOSMO loaded: {len(self.lens_names)} lenses (hierarchical mode)"
            )
            logger.info(f"  Lenses: {', '.join(self.lens_names)}")
            logger.info(f"  Nuisance parameters: {len(self.nuisance_parameters())}")
            logger.info(f"  Lambda bounds: {self.lambda_bounds}")
            logger.info(f"  Anisotropy bounds: {self.anisotropy_bounds}")

        # Initialize JIT-compiled likelihood (created on first call)
        self._jitted_call = None
        self._jit_param_names = None

    def _distance_tuple(
        self, cosmology, z_lens: float, z_source: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Compute angular diameter distances (D_d, D_s, D_ds) for a lens system."""
        # Use cosmology's standard distance methods
        # Note: z_lens and z_source are data constants (not traced), so float() is safe
        # but we avoid it for cleaner code
        dd = jnp.asarray(cosmology.angular_diameter_distance(z_lens), dtype=_DTYPE)
        ds = jnp.asarray(cosmology.angular_diameter_distance(z_source), dtype=_DTYPE)

        # For D_ds, use angular_diameter_distance_between if available
        if hasattr(cosmology, "angular_diameter_distance_between"):
            dds = jnp.asarray(
                cosmology.angular_diameter_distance_between(z_lens, z_source),
                dtype=_DTYPE,
            )
        else:
            # Compute D_ds from comoving distances
            d_c_l = cosmology.comoving_distance(z_lens)
            d_c_s = cosmology.comoving_distance(z_source)
            dds = jnp.asarray((d_c_s - d_c_l) / (1.0 + z_source), dtype=_DTYPE)

        return dd, ds, dds

    def _default_dataset_name(self) -> str:
        return "tdcosmo"

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self) -> Dict[str, Any]:
        return {}

    def theory(self, cosmology, **kwargs):
        raise NotImplementedError

    def _gaussian_nodes(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        bounds: Tuple[float, float],
        override: Optional[float] = None,
        *,
        truncate: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if override is not None:
            node = jnp.asarray(override, dtype=_DTYPE)
            return node[None], jnp.array([1.0], dtype=_DTYPE)

        sigma = jnp.asarray(sigma, dtype=_DTYPE)
        mean = jnp.asarray(mean, dtype=_DTYPE)
        sigma = jnp.maximum(sigma, jnp.array(1e-6, dtype=_DTYPE))

        nodes = mean + jnp.sqrt(2.0) * sigma * _HERMITE_NODES
        weights = _HERMITE_WEIGHTS / _HERMITE_NORM
        if truncate:
            lower, upper = bounds
            in_bounds = (nodes >= lower) & (nodes <= upper)
            weights = weights * in_bounds.astype(_DTYPE)
        weights = weights / jnp.maximum(jnp.sum(weights), _EPS)
        return nodes, weights

    def _kappa_nodes(
        self,
        lens: TDCOSMOLensData,
        override: Optional[float] = None,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        if override is not None:
            node = jnp.asarray(override, dtype=_DTYPE)
            return node[None], jnp.array([1.0], dtype=_DTYPE)

        weights = lens.kappa_pdf / jnp.maximum(jnp.sum(lens.kappa_pdf), _EPS)
        return lens.kappa_centers, weights

    @staticmethod
    def _standard_normal_cdf(x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))

    @staticmethod
    def _standard_normal_ppf(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.clip(p, _EPS, 1.0 - _EPS)
        return jnp.sqrt(2.0) * erfinv(2.0 * p - 1.0)

    def _normal_draws(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        quantiles: jnp.ndarray,
        *,
        override: Optional[float] = None,
    ) -> jnp.ndarray:
        """Deterministic draws from an (untruncated) normal distribution."""
        if override is not None:
            return jnp.full_like(quantiles, jnp.asarray(override, dtype=_DTYPE))
        mean = jnp.asarray(mean, dtype=_DTYPE)
        sigma = jnp.asarray(sigma, dtype=_DTYPE)
        sigma = jnp.maximum(sigma, jnp.array(1e-6, dtype=_DTYPE))
        return mean + sigma * self._standard_normal_ppf(quantiles)

    def _truncated_normal_draws(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        bounds: Tuple[float, float],
        quantiles: jnp.ndarray,
        *,
        override: Optional[float] = None,
    ) -> jnp.ndarray:
        if override is not None:
            return jnp.full_like(quantiles, jnp.asarray(override, dtype=_DTYPE))

        mean = jnp.asarray(mean, dtype=_DTYPE)
        sigma = jnp.asarray(sigma, dtype=_DTYPE)
        sigma = jnp.maximum(sigma, jnp.array(1e-6, dtype=_DTYPE))
        lower = jnp.array(bounds[0], dtype=_DTYPE)
        upper = jnp.array(bounds[1], dtype=_DTYPE)

        a = (lower - mean) / sigma
        b = (upper - mean) / sigma
        cdf_a = self._standard_normal_cdf(a)
        cdf_b = self._standard_normal_cdf(b)
        mass = jnp.maximum(cdf_b - cdf_a, _EPS)
        p = cdf_a + quantiles * mass
        z = self._standard_normal_ppf(p)
        return mean + sigma * z

    def _distribution_quantiles(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        base = self._quantile_base
        return (
            base[self._quantile_perm_lam],
            base[self._quantile_perm_ani],
            base[self._quantile_perm_kappa],
            base[self._quantile_perm_vel],
            base[self._quantile_perm_gamma],
        )

    def _build_packed_data(self) -> None:
        """Pack TDCOSMO lens data into fixed-size JAX arrays for vectorized evaluation."""
        lenses = [self.lens_data[name] for name in self.lens_names]
        if not lenses:
            return

        n_lens = len(lenses)
        max_n_meas = max(int(l.sigma_v_obs.shape[0]) for l in lenses)
        max_n_ani = max(int(l.ani_params.shape[0]) for l in lenses)
        max_n_gamma = max(
            int(l.gamma_pl_params.shape[0]) if l.gamma_pl_params is not None else 1
            for l in lenses
        )
        max_n_ddt = max(int(l.ddt_centers.shape[0]) for l in lenses)
        max_n_kappa = max(int(l.kappa_centers.shape[0]) for l in lenses)

        num_draws = int(self.num_distribution_draws)

        def _pad_axis(axis: np.ndarray, size: int, *, pad_val: float) -> np.ndarray:
            if axis.size >= size:
                return axis[:size]
            pad = np.full((size - axis.size,), pad_val, dtype=axis.dtype)
            return np.concatenate([axis, pad], axis=0)

        sigma_v_obs = np.zeros((n_lens, max_n_meas), dtype="float32")
        j_model = np.zeros_like(sigma_v_obs)
        cov_meas = np.zeros((n_lens, max_n_meas, max_n_meas), dtype="float32")
        cov_j_sqrt = np.zeros_like(cov_meas)
        mask_meas = np.zeros_like(sigma_v_obs)

        ani_params = np.zeros((n_lens, max_n_ani), dtype="float32")
        ani_n = np.ones((n_lens,), dtype="int32")
        ani_scaling = np.ones((n_lens, max_n_meas, max_n_ani), dtype="float32")
        ani_scaling_2d = np.ones(
            (n_lens, max_n_meas, max_n_ani, max_n_gamma), dtype="float32"
        )
        ani_bound_low = np.zeros((n_lens,), dtype="float32")
        ani_bound_high = np.zeros((n_lens,), dtype="float32")
        gamma_params = np.zeros((n_lens, max_n_gamma), dtype="float32")
        gamma_n = np.ones((n_lens,), dtype="int32")
        has_gamma = np.zeros((n_lens,), dtype="float32")
        gamma_prior_mean = np.zeros((n_lens,), dtype="float32")
        gamma_prior_sigma = np.zeros((n_lens,), dtype="float32")
        gamma_pivot = np.zeros((n_lens,), dtype="float32")
        gamma_slope = np.zeros((n_lens,), dtype="float32")

        ddt_centers = np.zeros((n_lens, max_n_ddt), dtype="float32")
        ddt_weights = np.zeros((n_lens, max_n_ddt), dtype="float32")
        ddt_n = np.ones((n_lens,), dtype="int32")
        ddt_norm_factor = np.zeros((n_lens,), dtype="float32")
        bandwidth = np.zeros((n_lens,), dtype="float32")

        kappa_centers = np.zeros((n_lens, max_n_kappa), dtype="float32")
        kappa_pdf = np.zeros((n_lens, max_n_kappa), dtype="float32")
        kappa_n = np.ones((n_lens,), dtype="int32")
        kappa_min = np.zeros((n_lens,), dtype="float32")
        kappa_max = np.zeros((n_lens,), dtype="float32")
        kappa_prior_mean = np.zeros((n_lens,), dtype="float32")
        kappa_prior_sigma = np.zeros((n_lens,), dtype="float32")

        vel_samples = np.ones((n_lens, num_draws, max_n_meas), dtype="float32")
        vel_n = np.ones((n_lens,), dtype="int32")

        z_lens = np.zeros((n_lens,), dtype="float32")
        z_source = np.zeros((n_lens,), dtype="float32")
        lambda_scaling = np.zeros((n_lens,), dtype="float32")

        for i, lens in enumerate(lenses):
            n_meas = int(lens.sigma_v_obs.shape[0])
            mask_meas[i, :n_meas] = 1.0
            sigma_v_obs[i, :n_meas] = np.asarray(lens.sigma_v_obs)
            j_model[i, :n_meas] = np.asarray(lens.j_model)
            cov_meas[i, :n_meas, :n_meas] = np.asarray(lens.cov_meas)
            cov_j_sqrt[i, :n_meas, :n_meas] = np.asarray(lens.cov_j_sqrt)

            ani_axis = np.asarray(lens.ani_params, dtype="float32")
            ani_params[i] = _pad_axis(ani_axis, max_n_ani, pad_val=float(ani_axis[-1]))
            ani_n[i] = int(ani_axis.shape[0])
            ani_scaling[i, :n_meas, : ani_axis.shape[0]] = np.asarray(
                lens.ani_scaling, dtype="float32"
            )
            if self.anisotropy_model == "const":
                beta_min = float(ani_axis[0])
                beta_max = float(ani_axis[-1])
                if self.anisotropy_parameterization == "TAN_RAD":
                    r_max = math.sqrt(max(1.0 - beta_min, 0.0))
                    ani_bound_low[i] = -r_max
                    ani_bound_high[i] = r_max
                else:
                    ani_bound_low[i] = beta_min
                    ani_bound_high[i] = beta_max
            else:
                ani_bound_low[i] = float(self.anisotropy_bounds[0])
                ani_bound_high[i] = float(self.anisotropy_bounds[1])

            if lens.gamma_pl_params is not None and lens.ani_scaling_2d is not None:
                has_gamma[i] = 1.0
                gamma_axis = np.asarray(lens.gamma_pl_params, dtype="float32")
                gamma_params[i] = _pad_axis(
                    gamma_axis, max_n_gamma, pad_val=float(gamma_axis[-1])
                )
                gamma_n[i] = int(gamma_axis.shape[0])
                ani_scaling_2d[
                    i, :n_meas, : ani_axis.shape[0], : gamma_axis.shape[0]
                ] = np.asarray(lens.ani_scaling_2d, dtype="float32")
                if lens.gamma_pl_prior_mean is not None:
                    gamma_prior_mean[i] = float(lens.gamma_pl_prior_mean)
                if lens.gamma_pl_prior_sigma is not None:
                    gamma_prior_sigma[i] = float(lens.gamma_pl_prior_sigma)
                if lens.gamma_pl_pivot is not None:
                    gamma_pivot[i] = float(lens.gamma_pl_pivot)
                if lens.gamma_pl_ddt_slope is not None:
                    gamma_slope[i] = float(lens.gamma_pl_ddt_slope)

            centers = np.asarray(lens.ddt_centers, dtype="float32")
            weights = np.asarray(lens.ddt_weights, dtype="float32")
            ddt_centers[i] = _pad_axis(centers, max_n_ddt, pad_val=float(centers[-1]))
            ddt_weights[i, : centers.shape[0]] = weights
            ddt_n[i] = int(centers.shape[0])
            ddt_norm_factor[i] = float(lens.ddt_norm_factor)
            bandwidth[i] = float(lens.bandwidth)

            kappa_axis = np.asarray(lens.kappa_centers, dtype="float32")
            kappa_centers[i] = _pad_axis(
                kappa_axis, max_n_kappa, pad_val=float(kappa_axis[-1])
            )
            kappa_pdf[i, : kappa_axis.shape[0]] = np.asarray(
                lens.kappa_pdf, dtype="float32"
            )
            kappa_n[i] = int(kappa_axis.shape[0])
            kappa_min[i] = float(lens.kappa_min)
            kappa_max[i] = float(lens.kappa_max)
            prior = self.lens_priors.get(lens.name)
            if prior is not None and prior.sigma > 0.0:
                kappa_prior_mean[i] = float(prior.mean)
                kappa_prior_sigma[i] = float(prior.sigma)

            if lens.vel_disp_scaling_samples is not None:
                vel = np.asarray(lens.vel_disp_scaling_samples, dtype="float32")
                if vel.ndim == 1:
                    vel = vel[:, None]
                if vel.shape[0] < num_draws:
                    pad = np.repeat(vel[-1:, :], num_draws - vel.shape[0], axis=0)
                    vel = np.concatenate([vel, pad], axis=0)
                elif vel.shape[0] > num_draws:
                    vel = _downsample_rows_evenly(vel, num_draws)
                vel_n[i] = int(vel.shape[0])
                vel_samples[i, : vel.shape[0], : vel.shape[1]] = vel

            z_lens[i] = float(lens.z_lens)
            z_source[i] = float(lens.z_source)
            lambda_scaling[i] = float(lens.lambda_scaling)

        self._packed_data = {
            "sigma_v_obs": jnp.asarray(sigma_v_obs, dtype=_DTYPE),
            "j_model": jnp.asarray(j_model, dtype=_DTYPE),
            "cov_meas": jnp.asarray(cov_meas, dtype=_DTYPE),
            "cov_j_sqrt": jnp.asarray(cov_j_sqrt, dtype=_DTYPE),
            "mask_meas": jnp.asarray(mask_meas, dtype=_DTYPE),
            "ani_params": jnp.asarray(ani_params, dtype=_DTYPE),
            "ani_scaling": jnp.asarray(ani_scaling, dtype=_DTYPE),
            "ani_scaling_2d": jnp.asarray(ani_scaling_2d, dtype=_DTYPE),
            "ani_n": jnp.asarray(ani_n, dtype=jnp.int32),
            "ani_bound_low": jnp.asarray(ani_bound_low, dtype=_DTYPE),
            "ani_bound_high": jnp.asarray(ani_bound_high, dtype=_DTYPE),
            "gamma_params": jnp.asarray(gamma_params, dtype=_DTYPE),
            "gamma_n": jnp.asarray(gamma_n, dtype=jnp.int32),
            "has_gamma": jnp.asarray(has_gamma, dtype=_DTYPE),
            "gamma_prior_mean": jnp.asarray(gamma_prior_mean, dtype=_DTYPE),
            "gamma_prior_sigma": jnp.asarray(gamma_prior_sigma, dtype=_DTYPE),
            "gamma_pivot": jnp.asarray(gamma_pivot, dtype=_DTYPE),
            "gamma_slope": jnp.asarray(gamma_slope, dtype=_DTYPE),
            "ddt_centers": jnp.asarray(ddt_centers, dtype=_DTYPE),
            "ddt_weights": jnp.asarray(ddt_weights, dtype=_DTYPE),
            "ddt_n": jnp.asarray(ddt_n, dtype=jnp.int32),
            "ddt_norm_factor": jnp.asarray(ddt_norm_factor, dtype=_DTYPE),
            "bandwidth": jnp.asarray(bandwidth, dtype=_DTYPE),
            "kappa_centers": jnp.asarray(kappa_centers, dtype=_DTYPE),
            "kappa_pdf": jnp.asarray(kappa_pdf, dtype=_DTYPE),
            "kappa_n": jnp.asarray(kappa_n, dtype=jnp.int32),
            "kappa_min": jnp.asarray(kappa_min, dtype=_DTYPE),
            "kappa_max": jnp.asarray(kappa_max, dtype=_DTYPE),
            "kappa_prior_mean": jnp.asarray(kappa_prior_mean, dtype=_DTYPE),
            "kappa_prior_sigma": jnp.asarray(kappa_prior_sigma, dtype=_DTYPE),
            "vel_samples": jnp.asarray(vel_samples, dtype=_DTYPE),
            "vel_n": jnp.asarray(vel_n, dtype=jnp.int32),
            "z_lens": jnp.asarray(z_lens, dtype=_DTYPE),
            "z_source": jnp.asarray(z_source, dtype=_DTYPE),
            "lambda_scaling": jnp.asarray(lambda_scaling, dtype=_DTYPE),
        }
        self._use_packed = True

    def _anisotropy_to_grid(self, value: jnp.ndarray) -> jnp.ndarray:
        """Map sampled anisotropy parameter to the beta_ani grid used by the data."""
        value = jnp.asarray(value, dtype=_DTYPE)
        if self.anisotropy_model != "const":
            return value
        if self.anisotropy_parameterization == "BETA":
            return value
        # TAN_RAD parameterization: sample r = sigma_t/sigma_r, convert to beta_ani.
        return 1.0 - value**2

    def _kappa_draws(
        self,
        lens: TDCOSMOLensData,
        quantiles: jnp.ndarray,
        *,
        override: Optional[float] = None,
    ) -> jnp.ndarray:
        if override is not None:
            return jnp.full_like(quantiles, jnp.asarray(override, dtype=_DTYPE))

        weights = lens.kappa_pdf / jnp.maximum(jnp.sum(lens.kappa_pdf), _EPS)
        cdf = jnp.cumsum(weights)
        cdf = cdf / jnp.maximum(cdf[-1], _EPS)
        idx = jnp.searchsorted(cdf, quantiles, side="left")
        idx = jnp.clip(idx, 0, lens.kappa_centers.shape[0] - 1)
        return lens.kappa_centers[idx]

    def _vel_disp_scaling_draws(
        self, lens: TDCOSMOLensData, quantiles: jnp.ndarray
    ) -> Optional[jnp.ndarray]:
        samples = lens.vel_disp_scaling_samples
        if samples is None:
            return None
        n = jnp.asarray(samples.shape[0], dtype=jnp.int32)
        idx = jnp.floor(quantiles * n.astype(_DTYPE)).astype(jnp.int32)
        idx = jnp.clip(idx, 0, n - 1)
        return samples[idx]

    def _integrated_lens_loglike_draws_packed(
        self,
        cosmology,
        *,
        lambda_mean: jnp.ndarray,
        lambda_sigma: jnp.ndarray,
        alpha_lambda: jnp.ndarray,
        a_mean: jnp.ndarray,
        a_sigma: jnp.ndarray,
        sigma_v_sys_error: jnp.ndarray,
        lambda_overrides: jnp.ndarray,
        ani_overrides: jnp.ndarray,
        kappa_overrides: jnp.ndarray,
        gamma_overrides: jnp.ndarray,
    ) -> jnp.ndarray:
        """Vectorized TDCOSMO likelihood using packed data arrays."""
        packed = self._packed_data
        if packed is None:
            return jnp.array(0.0, dtype=_DTYPE)

        q_lam, q_ani, q_kappa, q_vel, q_gamma = self._distribution_quantiles()
        ppf_lam = self._standard_normal_ppf(q_lam)

        z_lens = packed["z_lens"]
        z_source = packed["z_source"]
        lambda_scaling = packed["lambda_scaling"]

        dd, ds, dds = jax.vmap(lambda zl, zs: self._distance_tuple(cosmology, zl, zs))(
            z_lens, z_source
        )
        ddt = (1.0 + z_lens) * dd * ds / jnp.maximum(dds, _EPS)

        lambda_loc = lambda_mean + alpha_lambda * lambda_scaling
        gamma_lower = jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE)
        gamma_upper = jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE)

        def lens_loglike(
            sigma_v_obs,
            cov_meas,
            cov_j_sqrt,
            j_model,
            mask_meas,
            ani_params,
            ani_scaling,
            ani_scaling_2d,
            ani_n,
            ani_bound_low,
            ani_bound_high,
            gamma_params,
            gamma_n,
            has_gamma,
            gamma_prior_mean,
            gamma_prior_sigma,
            gamma_pivot,
            gamma_slope,
            ddt_centers,
            ddt_weights,
            ddt_n,
            ddt_norm,
            bandwidth,
            kappa_centers,
            kappa_pdf,
            kappa_n,
            kappa_prior_mean,
            kappa_prior_sigma,
            vel_samples,
            vel_n,
            z_lens_l,
            ddt_l,
            dd_l,
            lambda_loc_l,
            lambda_override_l,
            ani_override_l,
            kappa_override_l,
            gamma_override_l,
        ):
            override_mask = jnp.isfinite(lambda_override_l)
            lambda_draws = jnp.where(
                override_mask, lambda_override_l, lambda_loc_l + lambda_sigma * ppf_lam
            )

            ani_override_mask = jnp.isfinite(ani_override_l)
            ani_draws = jnp.where(
                ani_override_mask,
                ani_override_l,
                self._truncated_normal_draws(
                    a_mean, a_sigma, (ani_bound_low, ani_bound_high), q_ani
                ),
            )

            kappa_override_mask = jnp.isfinite(kappa_override_l)
            idx = jnp.arange(kappa_centers.shape[0])
            mask = (idx < kappa_n).astype(_DTYPE)
            weights = kappa_pdf * mask
            weights = weights / jnp.maximum(jnp.sum(weights), _EPS)
            cdf = jnp.cumsum(weights)
            cdf = cdf / jnp.maximum(cdf[-1], _EPS)
            ii = jnp.searchsorted(cdf, q_kappa, side="left")
            ii = jnp.clip(ii, 0, kappa_centers.shape[0] - 1)
            kappa_draws_pdf = kappa_centers[ii]
            kappa_draws = jnp.where(
                kappa_override_mask,
                jnp.full_like(q_kappa, kappa_override_l),
                kappa_draws_pdf,
            )

            gamma_override_mask = jnp.isfinite(gamma_override_l)

            def _gamma_draws(_):
                return jnp.where(
                    gamma_override_mask,
                    jnp.full_like(q_gamma, gamma_override_l),
                    jax.lax.cond(
                        gamma_prior_sigma > 0.0,
                        lambda __: self._truncated_normal_draws(
                            gamma_prior_mean,
                            gamma_prior_sigma,
                            self.gamma_pl_bounds,
                            q_gamma,
                        ),
                        lambda __: gamma_lower + (gamma_upper - gamma_lower) * q_gamma,
                        operand=None,
                    ),
                )

            gamma_draws = jax.lax.cond(
                has_gamma > 0.5, _gamma_draws, lambda __: q_gamma * 0.0, operand=None
            )

            ani_grid = self._anisotropy_to_grid(ani_draws)

            def _interp_1d(_):
                ani_min = ani_params[0]
                ani_max = jnp.take(ani_params, ani_n - 1)
                ani_grid_c = jnp.clip(ani_grid, ani_min, ani_max)
                idx = jnp.searchsorted(ani_params, ani_grid_c, side="right") - 1
                idx = jnp.clip(idx, 0, ani_n - 2)
                x0 = ani_params[idx]
                x1 = ani_params[idx + 1]
                t = (ani_grid_c - x0) / jnp.maximum(x1 - x0, _EPS)
                v0 = jnp.take(ani_scaling, idx, axis=1)
                v1 = jnp.take(ani_scaling, idx + 1, axis=1)
                return jnp.swapaxes((1.0 - t) * v0 + t * v1, 0, 1)

            def _interp_2d(_):
                ani_min = ani_params[0]
                ani_max = jnp.take(ani_params, ani_n - 1)
                gamma_min = gamma_params[0]
                gamma_max = jnp.take(gamma_params, gamma_n - 1)
                ani_grid_c = jnp.clip(ani_grid, ani_min, ani_max)
                gamma_grid_c = jnp.clip(gamma_draws, gamma_min, gamma_max)

                def _interp_one(a, g):
                    i = jnp.searchsorted(ani_params, a, side="right") - 1
                    j = jnp.searchsorted(gamma_params, g, side="right") - 1
                    i = jnp.clip(i, 0, ani_n - 2)
                    j = jnp.clip(j, 0, gamma_n - 2)
                    x0 = ani_params[i]
                    x1 = ani_params[i + 1]
                    y0 = gamma_params[j]
                    y1 = gamma_params[j + 1]
                    t = (a - x0) / jnp.maximum(x1 - x0, _EPS)
                    u = (g - y0) / jnp.maximum(y1 - y0, _EPS)
                    v00 = ani_scaling_2d[:, i, j]
                    v10 = ani_scaling_2d[:, i + 1, j]
                    v01 = ani_scaling_2d[:, i, j + 1]
                    v11 = ani_scaling_2d[:, i + 1, j + 1]
                    return (
                        (1.0 - t) * (1.0 - u) * v00
                        + t * (1.0 - u) * v10
                        + (1.0 - t) * u * v01
                        + t * u * v11
                    )

                return jax.vmap(_interp_one)(ani_grid_c, gamma_grid_c)

            kin_scalings = jax.lax.cond(has_gamma > 0.5, _interp_2d, _interp_1d, None)

            idx_vel = jnp.floor(q_vel * vel_n.astype(_DTYPE)).astype(jnp.int32)
            idx_vel = jnp.clip(idx_vel, 0, vel_n - 1)
            vel_draws = vel_samples[idx_vel]
            kin_scalings = kin_scalings * vel_draws**2

            sqrt_kin = jnp.sqrt(kin_scalings)
            scaling_mats = sqrt_kin[:, :, None] * sqrt_kin[:, None, :]

            lambda_tot = lambda_draws * (1.0 - kappa_draws)
            lambda_tot = jnp.maximum(lambda_tot, jnp.array(1e-4, dtype=_DTYPE))

            ddt_eff = ddt_l * lambda_tot
            ddt_eff = jax.lax.cond(
                (has_gamma > 0.5) & (gamma_slope != 0.0),
                lambda _: ddt_eff - (gamma_draws - gamma_pivot) * gamma_slope,
                lambda _: ddt_eff,
                operand=None,
            )

            diff_ddt = (ddt_eff[:, None] - ddt_centers[None, :]) / jnp.maximum(
                bandwidth, _EPS
            )
            log_kernel = -0.5 * diff_ddt**2 - jnp.log(bandwidth * jnp.sqrt(_TWO_PI))
            log_weights = jnp.log(ddt_weights + _EPS)
            ddt_log = logsumexp(log_kernel + log_weights, axis=-1) - ddt_norm

            ds_dds = ddt_eff / dd_l / jnp.array(1.0 + z_lens_l, dtype=_DTYPE)
            ds_dds = jnp.maximum(ds_dds, jnp.array(0.0, dtype=_DTYPE))

            sigma_model = (
                jnp.sqrt(j_model[None, :] * ds_dds[:, None]) * sqrt_kin * c_km_s
            )
            cov_model = (
                cov_j_sqrt[None, :, :]
                * scaling_mats
                * ds_dds[:, None, None]
                * c_km_s**2
            )
            cov_total = cov_model + cov_meas[None, :, :]
            cov_total = jnp.where(
                sigma_v_sys_error > 0,
                cov_total
                + jnp.outer(
                    sigma_v_obs * sigma_v_sys_error,
                    sigma_v_obs * sigma_v_sys_error,
                )[None, :, :],
                cov_total,
            )

            mask_mat = mask_meas[None, :, None] * mask_meas[None, None, :]
            cov_total = cov_total * mask_mat
            eye = jnp.eye(cov_total.shape[-1], dtype=_DTYPE)[None, :, :]
            cov_total = cov_total + eye * (1.0 - mask_meas)[None, None, :]
            cov_total = cov_total + eye * jnp.array(1e-6, dtype=_DTYPE)

            delta = (sigma_v_obs[None, :] - sigma_model) * mask_meas[None, :]
            L = jnp.linalg.cholesky(cov_total)
            y = jax.scipy.linalg.solve_triangular(L, delta, lower=True)
            chi2 = jnp.sum(y**2, axis=-1)

            if self.normalized:
                logdet = 2.0 * jnp.sum(
                    jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1
                )
                n_meas = jnp.sum(mask_meas)
                kin_log = -0.5 * (chi2 + logdet + n_meas * jnp.log(_TWO_PI))
            else:
                kin_log = -0.5 * chi2

            loglike = ddt_log + kin_log

            kappa_prior_mask = kappa_prior_sigma > 0.0
            diff = (kappa_draws - kappa_prior_mean) / jnp.maximum(
                kappa_prior_sigma, _EPS
            )
            prior_term = -0.5 * diff**2 - jnp.log(
                jnp.maximum(kappa_prior_sigma, _EPS) * jnp.sqrt(_TWO_PI)
            )
            loglike = loglike + jnp.where(kappa_prior_mask, prior_term, 0.0)

            loglike = jnp.where(jnp.isfinite(loglike), loglike, _NEG_LARGE)
            return logsumexp(loglike) - jnp.log(
                jnp.array(self.num_distribution_draws, dtype=_DTYPE)
            )

        loglikes = jax.vmap(lens_loglike)(
            packed["sigma_v_obs"],
            packed["cov_meas"],
            packed["cov_j_sqrt"],
            packed["j_model"],
            packed["mask_meas"],
            packed["ani_params"],
            packed["ani_scaling"],
            packed["ani_scaling_2d"],
            packed["ani_n"],
            packed["ani_bound_low"],
            packed["ani_bound_high"],
            packed["gamma_params"],
            packed["gamma_n"],
            packed["has_gamma"],
            packed["gamma_prior_mean"],
            packed["gamma_prior_sigma"],
            packed["gamma_pivot"],
            packed["gamma_slope"],
            packed["ddt_centers"],
            packed["ddt_weights"],
            packed["ddt_n"],
            packed["ddt_norm_factor"],
            packed["bandwidth"],
            packed["kappa_centers"],
            packed["kappa_pdf"],
            packed["kappa_n"],
            packed["kappa_prior_mean"],
            packed["kappa_prior_sigma"],
            packed["vel_samples"],
            packed["vel_n"],
            packed["z_lens"],
            ddt,
            dd,
            lambda_loc,
            lambda_overrides,
            ani_overrides,
            kappa_overrides,
            gamma_overrides,
        )

        return jnp.sum(loglikes)

    def _integrated_lens_loglike_draws(
        self,
        lens: TDCOSMOLensData,
        ddt: jnp.ndarray,
        dd: jnp.ndarray,
        *,
        lambda_loc: jnp.ndarray,
        lambda_sigma: jnp.ndarray,
        a_mean: jnp.ndarray,
        a_sigma: jnp.ndarray,
        sigma_v_sys_error: jnp.ndarray = jnp.array(0.05, dtype=_DTYPE),
        lambda_override: Optional[float] = None,
        ani_override: Optional[float] = None,
        kappa_override: Optional[float] = None,
        gamma_pl: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        q_lam, q_ani, q_kappa, q_vel, q_gamma = self._distribution_quantiles()

        # hierarc draws lambda_mst from an (untruncated) Gaussian. The bounds in
        # Table 3 apply to the *population mean* (lambda_mst), not to per-lens draws.
        lambda_draws = self._normal_draws(
            lambda_loc,
            lambda_sigma,
            q_lam,
            override=lambda_override,
        )
        # NOTE: hierArc applies truncation/rejection to anisotropy draws based on the
        # interpolation bounds of the kinematic scaling grid (beta_ani). For the
        # TAN_RAD parameterization, this corresponds to truncating the drawn
        # sigma_t/sigma_r values by |r| <= sqrt(1 - beta_min).
        if self.anisotropy_model == "const":
            beta_min = lens.ani_params[0]
            beta_max = lens.ani_params[-1]
            if self.anisotropy_parameterization == "TAN_RAD":
                r_max = jnp.sqrt(
                    jnp.maximum(1.0 - beta_min, jnp.array(0.0, dtype=_DTYPE))
                )
                r_min = jnp.sqrt(
                    jnp.maximum(1.0 - beta_max, jnp.array(0.0, dtype=_DTYPE))
                )
                # For current TDCOSMO grids beta_max=1 so r_min==0; keep symmetric
                # bounds to match hierArc's squaring in the beta conversion.
                ani_bounds = (-r_max, r_max)
            else:
                ani_bounds = (beta_min, beta_max)
        else:
            ani_bounds = self.anisotropy_bounds

        ani_draws = self._truncated_normal_draws(
            a_mean,
            a_sigma,
            ani_bounds,
            q_ani,
            override=ani_override,
        )
        kappa_draws = self._kappa_draws(lens, q_kappa, override=kappa_override)

        gamma_draws: Optional[jnp.ndarray] = None
        if lens.gamma_pl_params is not None and lens.ani_scaling_2d is not None:
            if gamma_pl is not None:
                gamma_draws = jnp.full_like(q_lam, jnp.asarray(gamma_pl, dtype=_DTYPE))
            else:
                if (
                    lens.gamma_pl_prior_mean is not None
                    and lens.gamma_pl_prior_sigma is not None
                    and lens.gamma_pl_prior_sigma > 0
                ):
                    gamma_draws = self._truncated_normal_draws(
                        jnp.array(lens.gamma_pl_prior_mean, dtype=_DTYPE),
                        jnp.array(lens.gamma_pl_prior_sigma, dtype=_DTYPE),
                        self.gamma_pl_bounds,
                        q_gamma,
                    )
                else:
                    gamma_lower = jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE)
                    gamma_upper = jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE)
                    gamma_draws = gamma_lower + (gamma_upper - gamma_lower) * q_gamma

        ani_grid = self._anisotropy_to_grid(ani_draws)
        if gamma_draws is None:
            kin_scalings = lens.anisotropy_scaling_batch(
                ani_grid, use_spline=self.use_spline
            )
        else:
            kin_scalings = lens.anisotropy_scaling_batch(
                ani_grid, gamma_draws, use_spline=self.use_spline
            )
        vel_draws = self._vel_disp_scaling_draws(lens, q_vel)
        if vel_draws is not None:
            if vel_draws.ndim == 1:
                vel_factor = vel_draws[:, None]
            else:
                vel_factor = vel_draws
            kin_scalings = kin_scalings * vel_factor**2
        sqrt_kin = jnp.sqrt(kin_scalings)
        scaling_mats = sqrt_kin[:, :, None] * sqrt_kin[:, None, :]

        lambda_tot = lambda_draws * (1.0 - kappa_draws)
        # Match hierarc: clamp combined MST to avoid non-physical values.
        lambda_tot = jnp.maximum(lambda_tot, jnp.array(1e-4, dtype=_DTYPE))

        ddt_eff = ddt * lambda_tot
        if (
            gamma_draws is not None
            and lens.gamma_pl_pivot is not None
            and lens.gamma_pl_ddt_slope is not None
        ):
            ddt_eff = (
                ddt_eff - (gamma_draws - lens.gamma_pl_pivot) * lens.gamma_pl_ddt_slope
            )
        ddt_log = lens.ddt_logpdf(ddt_eff)

        ds_dds = ddt_eff / dd / jnp.array(1.0 + lens.z_lens, dtype=_DTYPE)
        ds_dds = jnp.maximum(ds_dds, jnp.array(0.0, dtype=_DTYPE))

        sigma_model = (
            jnp.sqrt(lens.j_model[None, :] * ds_dds[:, None]) * sqrt_kin * c_km_s
        )
        cov_model = (
            lens.cov_j_sqrt[None, :, :]
            * scaling_mats
            * ds_dds[:, None, None]
            * c_km_s**2
        )
        # Add systematic velocity dispersion error only when sigma_v_sys_error > 0
        # (hierarc pattern: sigma_sys_error_include controls whether this is applied)
        cov_total = cov_model + lens.cov_meas[None, :, :]
        cov_total = jnp.where(
            sigma_v_sys_error > 0,
            cov_total
            + jnp.outer(
                lens.sigma_v_obs * sigma_v_sys_error,
                lens.sigma_v_obs * sigma_v_sys_error,
            )[None, :, :],
            cov_total,
        )
        n = lens.sigma_v_obs.shape[0]
        cov_total = cov_total + jnp.eye(n, dtype=_DTYPE)[None, :, :] * jnp.array(
            1e-6, dtype=_DTYPE
        )

        delta = lens.sigma_v_obs[None, :] - sigma_model
        if n == 1:
            sigma2 = cov_total[:, 0, 0]
            delta0 = delta[:, 0]
            if not self.normalized:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2),
                    _NEG_LARGE,
                )
            else:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2 + jnp.log(sigma2) + jnp.log(_TWO_PI)),
                    _NEG_LARGE,
                )
        else:
            L = jnp.linalg.cholesky(cov_total)
            y = jax.scipy.linalg.solve_triangular(L, delta, lower=True)
            chi2 = jnp.sum(y**2, axis=-1)
            if not self.normalized:
                kin_log = -0.5 * chi2
            else:
                logdet = 2.0 * jnp.sum(
                    jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1
                )
                kin_log = -0.5 * (chi2 + logdet + n * jnp.log(_TWO_PI))

        loglike = ddt_log + kin_log

        kappa_prior = self.lens_priors[lens.name]
        if kappa_prior.sigma > 0.0:
            loglike = loglike + jax.vmap(kappa_prior.logpdf)(kappa_draws)

        loglike = jnp.where(jnp.isfinite(loglike), loglike, _NEG_LARGE)
        return logsumexp(loglike) - jnp.log(
            jnp.array(self.num_distribution_draws, dtype=_DTYPE)
        )

    def _integrated_lens_loglike(
        self,
        lens: TDCOSMOLensData,
        ddt: jnp.ndarray,
        dd: jnp.ndarray,
        lambda_nodes: jnp.ndarray,
        lambda_weights: jnp.ndarray,
        ani_nodes: jnp.ndarray,
        ani_weights: jnp.ndarray,
        kappa_nodes: jnp.ndarray,
        kappa_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        # Pre-compute log weights.
        log_w_lam = jnp.log(lambda_weights + _EPS)  # (n_lam,)
        log_w_ani = jnp.log(ani_weights + _EPS)  # (n_ani,)
        log_w_kap = jnp.log(kappa_weights + _EPS)  # (n_kap,)

        # Optional additional kappa prior (disabled by default).
        kappa_prior = self.lens_priors[lens.name]
        if kappa_prior.sigma > 0.0:
            log_w_kap = log_w_kap + jax.vmap(kappa_prior.logpdf)(kappa_nodes)

        # Pre-compute kinematic scaling for each anisotropy node to avoid repeating
        # interpolation inside the (kappa, lambda) loops.
        kin_scalings = jax.vmap(
            lambda a: lens.anisotropy_scaling(a, use_spline=self.use_spline)
        )(ani_nodes)
        sqrt_kin = jnp.sqrt(kin_scalings)
        scaling_mats = jax.vmap(lambda s: jnp.outer(s, s))(sqrt_kin)

        def loglike_marginal_kappa(
            lam: jnp.ndarray, kin_scaling: jnp.ndarray, scaling_mat: jnp.ndarray
        ) -> jnp.ndarray:
            loglikes = jax.vmap(
                lambda kap: self._single_lens_loglike_with_kin_scaling(
                    lens=lens,
                    ddt=ddt,
                    dd=dd,
                    kappa=kap,
                    lambda_mst=lam,
                    kin_scaling=kin_scaling,
                    scaling_mat=scaling_mat,
                )
            )(kappa_nodes)
            return logsumexp(log_w_kap + loglikes)

        def loglike_marginal_lambda(
            kin_scaling: jnp.ndarray, scaling_mat: jnp.ndarray
        ) -> jnp.ndarray:
            return jax.vmap(
                lambda lam: loglike_marginal_kappa(lam, kin_scaling, scaling_mat)
            )(lambda_nodes)

        inner_totals = jax.vmap(loglike_marginal_lambda)(kin_scalings, scaling_mats)

        terms = inner_totals + log_w_ani[:, None] + log_w_lam[None, :]
        return logsumexp(terms.ravel())

    def log_likelihood(self, cosmology, **kwargs) -> float:
        include_priors = kwargs.pop("include_priors", True)

        lambda_mean = jnp.asarray(kwargs.get("lambda_int_mean", 1.0), dtype=_DTYPE)
        lambda_sigma = jnp.asarray(kwargs.get("lambda_int_sigma", 0.05), dtype=_DTYPE)
        alpha_lambda = jnp.asarray(kwargs.get("alpha_lambda", 0.0), dtype=_DTYPE)
        a_mean = jnp.asarray(kwargs.get("a_ani_mean", 1.0), dtype=_DTYPE)
        a_sigma = jnp.asarray(kwargs.get("a_ani_sigma", 0.1), dtype=_DTYPE)

        # Only extract sigma_v_sys_error when sigma_sys_error_include=True
        if self.sigma_sys_error_include:
            sigma_v_sys = jnp.asarray(
                kwargs.get("sigma_v_sys_error", 0.05), dtype=_DTYPE
            )
        else:
            sigma_v_sys = jnp.array(0.0, dtype=_DTYPE)

        lambda_lower = jnp.array(self.lambda_bounds[0], dtype=_DTYPE)
        lambda_upper = jnp.array(self.lambda_bounds[1], dtype=_DTYPE)
        a_lower = jnp.array(self.anisotropy_bounds[0], dtype=_DTYPE)
        a_upper = jnp.array(self.anisotropy_bounds[1], dtype=_DTYPE)

        total = jnp.array(0.0, dtype=_DTYPE)

        # Enforce positive scatters through an immediate penalty when violated.
        total = total + jnp.where(lambda_sigma > 0.0, 0.0, _NEG_LARGE)
        total = total + jnp.where(a_sigma > 0.0, 0.0, _NEG_LARGE)
        if self.sigma_sys_error_include:
            total = total + jnp.where(sigma_v_sys > 0.0, 0.0, _NEG_LARGE)

        lambda_sigma_safe = jnp.where(
            lambda_sigma > 0.0, lambda_sigma, jnp.array(1.0, dtype=_DTYPE)
        )
        a_sigma_safe = jnp.where(a_sigma > 0.0, a_sigma, jnp.array(1.0, dtype=_DTYPE))
        sigma_v_sys_safe = jnp.where(
            sigma_v_sys > 0.0, sigma_v_sys, jnp.array(1.0, dtype=_DTYPE)
        )

        if self._use_packed and self._packed_data is not None and self.use_tdcosmo2025:
            packed = self._packed_data
            lambda_overrides = []
            ani_overrides = []
            kappa_overrides = []
            gamma_overrides = []
            for name in self.lens_names:
                lambda_key = f"lambda_int_{name}"
                ani_key = f"a_ani_{name}"
                legacy_ani_key = f"ani_param_{name}"
                kappa_key = f"kappa_ext_{name}"
                gamma_key = self._gamma_param_map.get(name)
                lambda_overrides.append(
                    jnp.asarray(kwargs.get(lambda_key, jnp.nan), dtype=_DTYPE)
                )
                ani_overrides.append(
                    jnp.asarray(
                        kwargs.get(ani_key, kwargs.get(legacy_ani_key, jnp.nan)),
                        dtype=_DTYPE,
                    )
                )
                kappa_overrides.append(
                    jnp.asarray(kwargs.get(kappa_key, jnp.nan), dtype=_DTYPE)
                )
                if gamma_key is None:
                    gamma_overrides.append(jnp.array(jnp.nan, dtype=_DTYPE))
                else:
                    gamma_overrides.append(
                        jnp.asarray(kwargs.get(gamma_key, jnp.nan), dtype=_DTYPE)
                    )

            lambda_overrides = jnp.stack(lambda_overrides, axis=0)
            ani_overrides = jnp.stack(ani_overrides, axis=0)
            kappa_overrides = jnp.stack(kappa_overrides, axis=0)
            gamma_overrides = jnp.stack(gamma_overrides, axis=0)

            total = total + jnp.sum(
                self._bounds_penalty(lambda_overrides, lambda_lower, lambda_upper)
            )
            total = total + jnp.sum(
                self._bounds_penalty(ani_overrides, a_lower, a_upper)
            )
            total = total + jnp.sum(
                self._bounds_penalty(
                    kappa_overrides, packed["kappa_min"], packed["kappa_max"]
                )
            )
            total = total + jnp.sum(
                self._bounds_penalty(
                    gamma_overrides,
                    jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE),
                    jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE),
                )
            )

            total = total + self._integrated_lens_loglike_draws_packed(
                cosmology,
                lambda_mean=lambda_mean,
                lambda_sigma=lambda_sigma,
                alpha_lambda=alpha_lambda,
                a_mean=a_mean,
                a_sigma=a_sigma,
                sigma_v_sys_error=sigma_v_sys,
                lambda_overrides=lambda_overrides,
                ani_overrides=ani_overrides,
                kappa_overrides=kappa_overrides,
                gamma_overrides=gamma_overrides,
            )

            if include_priors:
                gamma_prior_sigma = packed["gamma_prior_sigma"]
                gamma_prior_mean = packed["gamma_prior_mean"]
                has_gamma = packed["has_gamma"] > 0.5
                mask = (
                    has_gamma
                    & jnp.isfinite(gamma_overrides)
                    & (gamma_prior_sigma > 0.0)
                )
                diff = (gamma_overrides - gamma_prior_mean) / jnp.maximum(
                    gamma_prior_sigma, _EPS
                )
                total = total + jnp.sum(jnp.where(mask, -0.5 * diff**2, 0.0))
        else:
            for name, lens in self.lens_data.items():
                kappa_key = f"kappa_ext_{name}"
                ani_key = f"a_ani_{name}"
                legacy_ani_key = f"ani_param_{name}"
                lambda_key = f"lambda_int_{name}"
                gamma_key = self._gamma_param_map.get(name)

                lambda_loc = lambda_mean + alpha_lambda * jnp.asarray(
                    lens.lambda_scaling, dtype=_DTYPE
                )

                lambda_override = kwargs.get(lambda_key, None)
                ani_override = kwargs.get(ani_key, kwargs.get(legacy_ani_key, None))
                kappa_override = kwargs.get(kappa_key, None)
                gamma_override = kwargs.get(gamma_key, None) if gamma_key else None

                if lambda_override is not None:
                    total = total + self._bounds_penalty(
                        jnp.asarray(lambda_override, dtype=_DTYPE),
                        lambda_lower,
                        lambda_upper,
                    )
                if ani_override is not None:
                    total = total + self._bounds_penalty(
                        jnp.asarray(ani_override, dtype=_DTYPE), a_lower, a_upper
                    )
                if kappa_override is not None:
                    total = total + self._bounds_penalty(
                        jnp.asarray(kappa_override, dtype=_DTYPE),
                        jnp.array(lens.kappa_min, dtype=_DTYPE),
                        jnp.array(lens.kappa_max, dtype=_DTYPE),
                    )
                if gamma_override is not None:
                    total = total + self._bounds_penalty(
                        jnp.asarray(gamma_override, dtype=_DTYPE),
                        jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE),
                        jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE),
                    )

                dd, ds, dds = self._distance_tuple(
                    cosmology, lens.z_lens, lens.z_source
                )
                ddt = (1.0 + lens.z_lens) * dd * ds / dds

                if self.use_tdcosmo2025:
                    loglike = self._integrated_lens_loglike_draws(
                        lens,
                        ddt,
                        dd,
                        lambda_loc=lambda_loc,
                        lambda_sigma=lambda_sigma,
                        a_mean=a_mean,
                        a_sigma=a_sigma,
                        sigma_v_sys_error=sigma_v_sys,
                        lambda_override=lambda_override,
                        ani_override=ani_override,
                        kappa_override=kappa_override,
                        gamma_pl=gamma_override,
                    )
                else:
                    lambda_nodes, lambda_weights = self._gaussian_nodes(
                        lambda_loc,
                        lambda_sigma,
                        self.lambda_bounds,
                        override=lambda_override,
                    )
                    ani_nodes, ani_weights = self._gaussian_nodes(
                        a_mean,
                        a_sigma,
                        self.anisotropy_bounds,
                        override=ani_override,
                        truncate=True,
                    )
                    kappa_nodes, kappa_weights = self._kappa_nodes(
                        lens, override=kappa_override
                    )

                    loglike = self._integrated_lens_loglike(
                        lens,
                        ddt,
                        dd,
                        lambda_nodes,
                        lambda_weights,
                        ani_nodes,
                        ani_weights,
                        kappa_nodes,
                        kappa_weights,
                    )
                total = total + loglike

                # Lens-specific priors (hierarc PriorLikelihood): unnormalized Gaussians.
                if (
                    include_priors
                    and gamma_override is not None
                    and lens.gamma_pl_prior_mean is not None
                    and lens.gamma_pl_prior_sigma is not None
                    and lens.gamma_pl_prior_sigma > 0
                ):
                    diff = (
                        jnp.asarray(gamma_override, dtype=_DTYPE)
                        - jnp.array(lens.gamma_pl_prior_mean, dtype=_DTYPE)
                    ) / jnp.array(  # noqa: E501
                        lens.gamma_pl_prior_sigma, dtype=_DTYPE
                    )
                    total = total - 0.5 * diff**2

        if include_priors:
            if self.log_scatter_prior:
                total = total - jnp.log(lambda_sigma_safe) - jnp.log(a_sigma_safe)
                if self.sigma_sys_error_include:
                    total = total - jnp.log(sigma_v_sys_safe)
            # hierarc uses a log-uniform prior on the OM/GOM anisotropy scale radius
            # (a_ani > 0). For the constant anisotropy model (including TAN_RAD
            # parameterization), the prior is flat (Table 3), so we must *not*
            # apply the 1/a_ani term.
            if self.anisotropy_model != "const":
                total = total + jnp.where(
                    a_mean > 0.0,
                    -jnp.log(a_mean),
                    _NEG_LARGE,
                )
            if self.omega_m_prior is not None:
                om = jnp.asarray(cosmology.params.get("Omega_m"), dtype=_DTYPE)
                mu, sigma = self.omega_m_prior
                total = total + _gaussian_logpdf(
                    om,
                    jnp.array(mu, dtype=_DTYPE),
                    jnp.array(sigma, dtype=_DTYPE),
                )

        return total

    def _single_lens_loglike(
        self,
        lens: TDCOSMOLensData,
        ddt: jnp.ndarray,
        dd: jnp.ndarray,
        kappa: jnp.ndarray,
        lambda_mst: jnp.ndarray,
        ani_param: Optional[jnp.ndarray],
    ) -> jnp.ndarray:
        kin_scaling = lens.anisotropy_scaling(ani_param, use_spline=self.use_spline)
        return self._single_lens_loglike_with_kin_scaling(
            lens=lens,
            ddt=ddt,
            dd=dd,
            kappa=kappa,
            lambda_mst=lambda_mst,
            kin_scaling=kin_scaling,
        )

    def _single_lens_loglike_with_kin_scaling(
        self,
        *,
        lens: TDCOSMOLensData,
        ddt: jnp.ndarray,
        dd: jnp.ndarray,
        kappa: jnp.ndarray,
        lambda_mst: jnp.ndarray,
        kin_scaling: jnp.ndarray,
        scaling_mat: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        # Match hierarc's TransformedCosmography + KinLikelihood convention:
        # - Apply MST (internal + external) to the model-predicted Ddt.
        # - Kinematics constrains Ds/Dds derived from the (shifted) Ddt and Dd.
        lambda_tot = jnp.asarray(lambda_mst, dtype=_DTYPE) * (
            1.0 - jnp.asarray(kappa, dtype=_DTYPE)
        )
        # hierarc clamps lambda_tot instead of rejecting samples.
        lambda_tot = jnp.maximum(lambda_tot, jnp.array(1e-4, dtype=_DTYPE))
        ddt_eff = ddt * lambda_tot
        ddt_log = lens.ddt_logpdf(ddt_eff)

        # Match hierarc: kinematics uses the same displaced Ddt (including kappa_ext)
        # when converting to Ds/Dds.
        ds_dds = jnp.maximum(
            ddt_eff / dd / (1.0 + lens.z_lens),
            jnp.array(0.0, dtype=_DTYPE),
        )
        sigma_model = jnp.sqrt(lens.j_model * ds_dds * kin_scaling) * c_km_s

        if scaling_mat is None:
            scaling_mat = jnp.outer(jnp.sqrt(kin_scaling), jnp.sqrt(kin_scaling))
        cov_model = lens.cov_j_sqrt * scaling_mat * ds_dds * c_km_s**2
        cov_total = cov_model + lens.cov_meas
        n = lens.sigma_v_obs.shape[0]
        cov_total = cov_total + jnp.eye(n, dtype=_DTYPE) * jnp.array(1e-6, dtype=_DTYPE)

        delta = lens.sigma_v_obs - sigma_model
        if n == 1:
            sigma2 = cov_total[0, 0]
            delta0 = delta[0]
            if not self.normalized:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2),
                    _NEG_LARGE,
                )
            else:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2 + jnp.log(sigma2) + jnp.log(_TWO_PI)),
                    _NEG_LARGE,
                )
        else:
            # Cholesky decomposition: faster and more stable than solve + slogdet
            L = jnp.linalg.cholesky(cov_total)
            # Solve L @ y = delta, then chi2 = y @ y
            y = jax.scipy.linalg.solve_triangular(L, delta, lower=True)
            chi2 = jnp.dot(y, y)
            if not self.normalized:
                kin_log = -0.5 * chi2
            else:
                # logdet(cov) = 2 * sum(log(diag(L)))
                logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                kin_log = -0.5 * (chi2 + logdet + n * jnp.log(_TWO_PI))

        invalid = ~jnp.isfinite(kin_log)
        kin_log = jnp.where(jnp.isfinite(kin_log), kin_log, _NEG_LARGE)

        loglike = ddt_log + kin_log
        loglike = jnp.where(jnp.isfinite(loglike), loglike, _NEG_LARGE)
        return jnp.where(invalid, _NEG_LARGE, loglike)

    @staticmethod
    def _bounds_penalty(
        value: jnp.ndarray, lower: jnp.ndarray, upper: jnp.ndarray
    ) -> jnp.ndarray:
        lower = jnp.asarray(lower, dtype=_DTYPE)
        upper = jnp.asarray(upper, dtype=_DTYPE)
        return jnp.where((value < lower) | (value > upper), _NEG_LARGE, 0.0)

    def __call__(self, **params) -> float:
        """
        Callable interface for MCMC sampling.

        Parameters
        ----------
        **params : dict
            All parameters including cosmological and nuisance parameters.
            Cosmological parameters are passed to cosmology_class dynamically.

        Returns
        -------
        float
            Log-likelihood value.
        """
        if self.cosmology_class is None:
            raise ValueError(
                "cosmology_class not set. Create with: "
                "TDCOSMOLikelihood(cosmology_class=LCDM)"
            )

        # Build JIT-compiled function on first call
        if self._jitted_call is None:
            self._build_jitted_call(params)

        # Extract parameter values in the correct order
        param_values = [params.get(name, 0.0) for name in self._jit_param_names]

        # Call JIT-compiled function
        return self._jitted_call(*param_values)

    def _get_cosmology_param_names(self) -> set:
        """Get parameter names from cosmology class dynamically (no hardcoding!)."""
        if hasattr(self, "_cached_cosmo_param_names"):
            return self._cached_cosmo_param_names

        # Try to get from class method first
        if hasattr(self.cosmology_class, "get_parameters"):
            params = self.cosmology_class.get_parameters()
            names = {p.name for p in params}
        else:
            # Fallback: inspect __init__ signature
            import inspect

            sig = inspect.signature(self.cosmology_class.__init__)
            names = {p for p in sig.parameters.keys() if p != "self" and p != "kwargs"}

        self._cached_cosmo_param_names = names
        return names

    def _build_jitted_call(self, sample_params: dict) -> None:
        """
        Build JIT-compiled likelihood function dynamically.

        This creates a JIT function that:
        1. Takes all parameter values as positional arguments
        2. Creates cosmology instance internally
        3. Computes full log_likelihood with Python loop unrolled by JAX

        No hardcoding of parameter names - they are determined from:
        - cosmology_class.get_parameters() or __init__ signature
        - sample_params keys (for nuisance parameters)
        """
        cosmo_param_names = self._get_cosmology_param_names()

        # Determine all parameter names from sample_params
        # Order: cosmology params first, then nuisance params
        cosmo_names = [k for k in sample_params.keys() if k in cosmo_param_names]
        nuisance_names = [k for k in sample_params.keys() if k not in cosmo_param_names]
        all_param_names = cosmo_names + nuisance_names

        self._jit_param_names = all_param_names

        cosmo_class = self.cosmology_class
        n_cosmo = len(cosmo_names)

        def _impl(*args):
            cosmo_vals = args[:n_cosmo]
            nuisance_vals = args[n_cosmo:]

            cosmo_dict = {name: val for name, val in zip(cosmo_names, cosmo_vals)}
            cosmology = cosmo_class(**cosmo_dict)

            nuisance_dict = {
                name: val for name, val in zip(nuisance_names, nuisance_vals)
            }
            return self.log_likelihood(cosmology, include_priors=True, **nuisance_dict)

        # JIT compile the implementation
        self._jitted_call = jax.jit(_impl)

    @property
    def nuisance_parameters(self) -> list:
        """
        Return nuisance parameters as Parameter objects.

        This allows MCMC samplers to automatically register these parameters
        with proper LaTeX labels for plotting.

        Returns
        -------
        List[Parameter]
            List of nuisance Parameter objects with full metadata.

        Examples
        --------
        >>> tdcosmo = TDCOSMOLikelihood(cosmology_class=LCDM)
        >>> for p in tdcosmo.nuisance_parameters():
        ...     print(f"{p.name}: {p.value} ({p.prior['min']}, {p.prior['max']}) [{p.latex_label}]")
        """
        from ...parameters import Parameter

        if self.anisotropy_model == "const":
            if self.anisotropy_parameterization == "TAN_RAD":
                a_mean_default = 1.0
                a_sigma_default = 0.1
                a_mean_label = r"$\langle\sigma_{\rm t}/\sigma_{\rm r}\rangle$"
                a_sigma_label = r"$\sigma(\sigma_{\rm t}/\sigma_{\rm r})$"
                a_mean_desc = "Mean constant anisotropy parameter (sigma_t/sigma_r)"
                a_sigma_desc = "Scatter in constant anisotropy (sigma_t/sigma_r)"
            else:
                a_mean_default = 0.5 * (
                    self.anisotropy_bounds[0] + self.anisotropy_bounds[1]
                )
                a_sigma_default = 0.2
                a_mean_label = r"$\langle\beta_{\rm ani}\rangle$"
                a_sigma_label = r"$\sigma(\beta_{\rm ani})$"
                a_mean_desc = "Mean constant anisotropy parameter (beta_ani)"
                a_sigma_desc = "Scatter in constant anisotropy (beta_ani)"
        else:
            a_mean_default = 1.0
            a_sigma_default = 0.1
            a_mean_label = r"$\langle a_{\rm ani}\rangle$"
            a_sigma_label = r"$\sigma(a_{\rm ani})$"
            a_mean_desc = "Stellar anisotropy population mean"
            a_sigma_desc = "Anisotropy population scatter"

        params = [
            Parameter(
                name="lambda_int_mean",
                value=1.0,
                free=True,
                prior={
                    "dist": "uniform",
                    "min": self.lambda_bounds[0],
                    "max": self.lambda_bounds[1],
                },
                latex_label=r"$\langle\lambda_{\rm int}\rangle$",
                description="Internal MST population mean",
            ),
            Parameter(
                name="lambda_int_sigma",
                value=0.05,
                free=True,
                prior={"dist": "uniform", "min": 0.001, "max": 0.5},
                latex_label=r"$\sigma(\lambda_{\rm int})$",
                description="Internal MST population scatter",
            ),
            Parameter(
                name="alpha_lambda",
                value=0.0,
                free=True,
                prior={"dist": "uniform", "min": -1.0, "max": 1.0},
                latex_label=r"$\alpha_\lambda$",
                description="Slope of lambda_int with R_eff/theta_E",
            ),
            Parameter(
                name="a_ani_mean",
                value=a_mean_default,
                free=True,
                prior={
                    "dist": "uniform",
                    "min": self.anisotropy_bounds[0],
                    "max": self.anisotropy_bounds[1],
                },
                latex_label=a_mean_label,
                description=a_mean_desc,
            ),
            Parameter(
                name="a_ani_sigma",
                value=a_sigma_default,
                free=True,
                prior={"dist": "uniform", "min": 0.01, "max": 1.0},
                latex_label=a_sigma_label,
                description=a_sigma_desc,
            ),
        ]
        # Only include sigma_v_sys_error when sigma_sys_error_include=True
        # (matches hierarc behavior for combining with external lenses)
        if self.sigma_sys_error_include:
            params.append(
                Parameter(
                    name="sigma_v_sys_error",
                    value=0.05,
                    free=True,
                    prior={"dist": "uniform", "min": 0.01, "max": 0.5},
                    latex_label=r"$\sigma_{v,\rm sys}$",
                    description="Fractional systematic velocity dispersion error",
                )
            )
        if self.gamma_pl_sampling:
            for lname, lens in self.lens_data.items():
                if lens.gamma_pl_params is None or lens.ani_scaling_2d is None:
                    continue
                # Match Table 3 prior bounds for gamma_pl.
                params.append(
                    Parameter(
                        name=self._gamma_param_map[lname],
                        value=float(
                            lens.gamma_pl_prior_mean
                            if lens.gamma_pl_prior_mean is not None
                            else 2.0
                        ),
                        free=True,
                        prior={
                            "dist": "uniform",
                            "min": self.gamma_pl_bounds[0],
                            "max": self.gamma_pl_bounds[1],
                        },
                        latex_label=rf"\gamma_{{\rm pl}}^{{\rm {lname}}}",
                        description=f"Power-law slope for lens {lname}",
                    )
                )
        return NuisanceList(params)

    def default_parameters(self) -> Dict[str, float]:
        """Return a physically motivated default parameter dictionary."""
        return {p.name: p.value for p in self.nuisance_parameters()}

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        derived: Dict[str, float] = {}
        for name, lens in self.lens_data.items():
            dd, ds, dds = self._distance_tuple(cosmology, lens.z_lens, lens.z_source)
            dd = float(dd)
            ds = float(ds)
            dds = float(dds)
            ddt = (1.0 + lens.z_lens) * dd * ds / dds
            derived[f"Ddt_{name}"] = ddt
            derived[f"Dd_{name}"] = dd
        return derived

    def _load_lambda_scaling(self) -> Dict[str, float]:
        csv_path = self.data_path.parents[0] / "tdcosmo_sample.csv"
        if not csv_path.exists():
            return {}
        import csv

        mapping: Dict[str, float] = {}
        with csv_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                raw = row.get("name")
                if raw is None:
                    continue
                sanitized = _sanitize(raw)
                try:
                    r_eff = float(row.get("r_eff", 0.0))
                    theta_e = float(row.get("theta_E", 1.0))
                except ValueError:
                    continue
                if theta_e == 0.0:
                    scaling = 0.0
                else:
                    scaling = r_eff / theta_e - 1.0
                mapping[sanitized] = scaling
        return mapping

    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        nuisance_info = [
            {
                "name": p.name,
                "default": p.value,
                "min": p.prior["min"],
                "max": p.prior["max"],
                "latex": p.latex_label,
                "description": p.description,
            }
            for p in self.nuisance_parameters()
        ]
        return {
            "name": self.name,
            "type": "TDCOSMO_Hierarchical",
            "n_lenses": len(self.lens_names),
            "lenses": self.lens_names,
            "nuisance_parameters": nuisance_info,
            "lambda_bounds": self.lambda_bounds,
            "anisotropy_bounds": self.anisotropy_bounds,
            "cosmology_class": (
                self.cosmology_class.__name__ if self.cosmology_class else None
            ),
        }

    # __add__ and __radd__ inherited from Likelihood base class

    def __repr__(self) -> str:
        cosmo_name = self.cosmology_class.__name__ if self.cosmology_class else "None"
        return (
            f"TDCOSMOLikelihood({len(self.lens_names)} lenses, "
            f"cosmology_class={cosmo_name})"
        )


# =============================================================================
# External Lens Likelihood (SLACS/SL2S) - Kinematic Only
# =============================================================================


@dataclass
class ExternalLensData:
    """
    Data structure for external (non-time-delay) lenses.

    These lenses provide only velocity dispersion constraints, not Ddt.
    They help break the H0-λ_int degeneracy by constraining λ_int independently.

    The key physics: σ² ∝ λ_int × J × (D_s/D_ds)
    Since D_s/D_ds depends only on Ω_m (not H0), these lenses constrain λ_int.
    """

    name: str
    z_lens: float
    z_source: float
    lambda_scaling: float  # r_eff/theta_E - 1
    sigma_v_obs: jnp.ndarray  # Measured velocity dispersion (n_bins,)
    cov_meas: jnp.ndarray  # Measurement covariance (n_bins, n_bins)
    cov_j_sqrt: jnp.ndarray  # J-factor uncertainty sqrt
    j_model: jnp.ndarray  # J factor at fiducial parameters (n_bins,)

    # Anisotropy scaling grid
    ani_params: jnp.ndarray  # Grid of anisotropy parameter values
    ani_scaling: jnp.ndarray  # J scaling for each anisotropy value (n_bins, n_ani)

    # Power-law slope grid (for external lenses with gamma_pl marginalization)
    gamma_pl_params: Optional[jnp.ndarray] = None  # Grid of gamma_pl values
    ani_scaling_2d: Optional[jnp.ndarray] = None  # (n_bins, n_ani, n_gamma)
    gamma_pl_prior_mean: Optional[float] = None
    gamma_pl_prior_sigma: Optional[float] = None
    ani_spline_tx: Optional[jnp.ndarray] = None
    ani_spline_ty: Optional[jnp.ndarray] = None
    ani_spline_coeffs: Optional[jnp.ndarray] = None  # (n_bins, n_ani, n_gamma)

    # External convergence (line of sight)
    kappa_los_type: str = "NONE"  # "NONE", "PDF", "GEV"
    kappa_pdf: Optional[jnp.ndarray] = None
    kappa_centers: Optional[jnp.ndarray] = None
    kappa_bin_edges: Optional[jnp.ndarray] = None
    kappa_cdf: Optional[jnp.ndarray] = None
    kappa_gev_mu: Optional[float] = None
    kappa_gev_sigma: Optional[float] = None
    kappa_gev_xi: Optional[float] = None
    # Optional axisymmetric JAM correction draws (sigma_axi/sigma_sph). For IFU data,
    # this may be an array per kinematic bin with shape (n_draws, n_bins).
    vel_disp_scaling_samples: Optional[jnp.ndarray] = None

    def anisotropy_scaling(
        self,
        ani_param: Optional[jnp.ndarray],
        gamma_pl: Optional[jnp.ndarray] = None,
        *,
        use_spline: bool = True,
    ) -> jnp.ndarray:
        """Interpolate J scaling for given anisotropy parameter (and gamma_pl if provided)."""
        if ani_param is None:
            return jnp.ones(self.sigma_v_obs.shape, dtype=_DTYPE)

        ani_param = jnp.asarray(ani_param, dtype=_DTYPE)
        if (
            self.gamma_pl_params is None
            or self.ani_scaling_2d is None
            or gamma_pl is None
        ):
            ani_param = jnp.clip(ani_param, self.ani_params[0], self.ani_params[-1])

            # ani_scaling has shape (n_bins, n_ani)
            # Interpolate for each kinematic bin
            def interp_bin(row):
                return jnp.interp(ani_param, self.ani_params, row)

            return jax.vmap(interp_bin)(self.ani_scaling).astype(_DTYPE)

        gamma_pl = jnp.asarray(gamma_pl, dtype=_DTYPE)
        if (
            use_spline
            and self.ani_spline_tx is not None
            and self.ani_spline_ty is not None
            and self.ani_spline_coeffs is not None
        ):
            return _rect_bivariate_spline_eval(
                ani_param,
                gamma_pl,
                x_axis=self.ani_params,
                y_axis=self.gamma_pl_params,
                tx=self.ani_spline_tx,
                ty=self.ani_spline_ty,
                coeffs=self.ani_spline_coeffs,
            )

        x_axis = self.ani_params
        y_axis = self.gamma_pl_params
        values = self.ani_scaling_2d  # (n_bins, n_ani, n_gamma)

        ani_param = jnp.clip(ani_param, x_axis[0], x_axis[-1])
        gamma_pl = jnp.clip(gamma_pl, y_axis[0], y_axis[-1])

        i = jnp.searchsorted(x_axis, ani_param, side="right") - 1
        j = jnp.searchsorted(y_axis, gamma_pl, side="right") - 1
        i = jnp.clip(i, 0, x_axis.shape[0] - 2)
        j = jnp.clip(j, 0, y_axis.shape[0] - 2)

        x0 = x_axis[i]
        x1 = x_axis[i + 1]
        y0 = y_axis[j]
        y1 = y_axis[j + 1]

        t = (ani_param - x0) / jnp.maximum(x1 - x0, _EPS)
        u = (gamma_pl - y0) / jnp.maximum(y1 - y0, _EPS)

        v00 = values[:, i, j]
        v10 = values[:, i + 1, j]
        v01 = values[:, i, j + 1]
        v11 = values[:, i + 1, j + 1]

        return (
            (1.0 - t) * (1.0 - u) * v00
            + t * (1.0 - u) * v10
            + (1.0 - t) * u * v01
            + t * u * v11
        ).astype(_DTYPE)

    def anisotropy_scaling_batch(
        self,
        ani_param: jnp.ndarray,
        gamma_pl: Optional[jnp.ndarray] = None,
        *,
        use_spline: bool = True,
    ) -> jnp.ndarray:
        """Vectorized anisotropy scaling for an array of ani_param values."""
        ani_param = jnp.asarray(ani_param, dtype=_DTYPE)
        if ani_param.ndim == 0:
            return self.anisotropy_scaling(ani_param, gamma_pl, use_spline=use_spline)[
                None, :
            ]

        if (
            self.gamma_pl_params is None
            or self.ani_scaling_2d is None
            or gamma_pl is None
        ):
            ani_param = jnp.clip(ani_param, self.ani_params[0], self.ani_params[-1])
            idx = jnp.searchsorted(self.ani_params, ani_param, side="right") - 1
            idx = jnp.clip(idx, 0, self.ani_params.shape[0] - 2)
            x0 = self.ani_params[idx]
            x1 = self.ani_params[idx + 1]
            t = (ani_param - x0) / jnp.maximum(x1 - x0, _EPS)
            v0 = jnp.take(self.ani_scaling, idx, axis=1)
            v1 = jnp.take(self.ani_scaling, idx + 1, axis=1)
            return jnp.swapaxes((1.0 - t) * v0 + t * v1, 0, 1).astype(_DTYPE)

        gamma_pl = jnp.asarray(gamma_pl, dtype=_DTYPE)
        if (
            use_spline
            and self.ani_spline_tx is not None
            and self.ani_spline_ty is not None
            and self.ani_spline_coeffs is not None
        ):
            return jax.vmap(
                lambda a, g: _rect_bivariate_spline_eval(
                    a,
                    g,
                    x_axis=self.ani_params,
                    y_axis=self.gamma_pl_params,
                    tx=self.ani_spline_tx,
                    ty=self.ani_spline_ty,
                    coeffs=self.ani_spline_coeffs,
                )
            )(ani_param, gamma_pl)
        return _interp_2d_batch(
            ani_param,
            gamma_pl,
            self.ani_params,
            self.gamma_pl_params,
            self.ani_scaling_2d,
        )


def _load_external_lens(data: dict, name: str) -> ExternalLensData:
    """Load a single external lens from hierarc-processed dictionary."""
    sigma_v_obs = np.asarray(data["sigma_v_measurement"], dtype="float32")
    cov_meas = np.asarray(data["error_cov_measurement"], dtype="float32")
    cov_j_sqrt = np.asarray(data["error_cov_j_sqrt"], dtype="float32")
    j_model = np.asarray(data["j_model"], dtype="float32")

    if cov_meas.ndim == 0:
        cov_meas = np.array([[cov_meas]], dtype="float32")
    if cov_j_sqrt.ndim == 0:
        cov_j_sqrt = np.array([[cov_j_sqrt]], dtype="float32")
    if cov_j_sqrt.ndim == 1:
        cov_j_sqrt = np.diag(cov_j_sqrt)

    # Extract gamma_pl prior from prior_list (if present). If missing, we treat
    # gamma_pl marginalization as uniform over the provided grid (matching the
    # official SLACS KCWI configuration where the gamma_pl prior is removed).
    gamma_pl_prior_mean = None
    gamma_pl_prior_sigma = None
    for prior in data.get("prior_list", []) or []:
        if isinstance(prior, list) and len(prior) >= 3 and prior[0] == "gamma_pl":
            gamma_pl_prior_mean = float(prior[1])
            gamma_pl_prior_sigma = float(prior[2])
            break

    # Extract anisotropy scaling grid
    # SLACS/SL2S typically have 2D grids: (a_ani, gamma_pl)
    # We marginalize over gamma_pl using its prior
    kin_params = data.get("kin_scaling_param_list", ["a_ani"])
    j_axes = data.get("j_kin_scaling_param_axes", [])
    j_grids = data.get("j_kin_scaling_grid_list", [])

    # Find parameter indices
    ani_idx = None
    gamma_idx = None
    for i, param in enumerate(kin_params):
        if param == "a_ani":
            ani_idx = i
        elif param == "gamma_pl":
            gamma_idx = i

    # Default: no anisotropy scaling
    ani_params = np.array([0.0], dtype="float32")  # Single value = no interpolation
    ani_scaling = np.ones((len(sigma_v_obs), 1), dtype="float32")
    gamma_pl_params = None

    ani_scaling_2d = None
    ani_spline_tx = None
    ani_spline_ty = None
    ani_spline_coeffs = None
    if ani_idx is not None and len(j_axes) > ani_idx:
        ani_params = np.asarray(j_axes[ani_idx], dtype="float32")
        gamma_axis = None
        gamma_weights = None
        if gamma_idx is not None and len(j_axes) > gamma_idx:
            gamma_axis = np.asarray(j_axes[gamma_idx], dtype="float32")
            gamma_pl_params = gamma_axis
            if (
                gamma_pl_prior_mean is not None
                and gamma_pl_prior_sigma is not None
                and gamma_pl_prior_sigma > 0
            ):
                weights = np.exp(
                    -0.5
                    * ((gamma_axis - gamma_pl_prior_mean) / gamma_pl_prior_sigma) ** 2
                )
                if np.all(np.isfinite(weights)) and float(np.sum(weights)) > 0:
                    gamma_weights = weights / np.sum(weights)
            if gamma_weights is None:
                gamma_weights = np.ones_like(gamma_axis) / float(len(gamma_axis))
        else:
            gamma_pl_params = None

        if len(j_grids) > 0:
            # j_grids is list of arrays, one per kinematic bin
            # Shape depends on parameter ordering
            ani_scaling_list = []
            ani_scaling_2d_list = []

            for grid in j_grids:
                grid = np.asarray(grid, dtype="float32")

                if grid.ndim == 1:
                    # Already 1D (only anisotropy), just use it
                    ani_scaling_list.append(grid)
                elif grid.ndim == 2:
                    # 2D grid: keep the full grid for proper marginalization of
                    # the likelihood (not of the scaling). We still store a 1D
                    # gamma-marginalized scaling for backward compatibility.
                    if gamma_axis is None:
                        ani_scaling_list.append(np.mean(grid, axis=-1))
                    else:
                        if ani_idx == 0:
                            grid_norm = grid  # (n_ani, n_gamma)
                        else:
                            grid_norm = grid.T  # (n_ani, n_gamma)
                        ani_scaling_2d_list.append(grid_norm)
                        ani_scaling_list.append(
                            np.sum(grid_norm * gamma_weights[None, :], axis=1)
                        )
                else:
                    # Unexpected shape, use ones
                    ani_scaling_list.append(np.ones(len(ani_params), dtype="float32"))

            ani_scaling = np.stack(ani_scaling_list, axis=0)
            if ani_scaling_2d_list:
                ani_scaling_2d = np.stack(ani_scaling_2d_list, axis=0)
                if gamma_axis is not None:
                    tx, ty, coeffs = _build_rect_bivariate_spline_coeffs(
                        ani_params,
                        gamma_axis,
                        ani_scaling_2d,
                    )
                    if tx is not None and ty is not None and coeffs is not None:
                        ani_spline_tx = tx
                        ani_spline_ty = ty
                        ani_spline_coeffs = coeffs

            # Ensure shape is (n_bins, n_ani)
            if ani_scaling.shape[1] != len(ani_params):
                # Shape mismatch, fall back to ones
                ani_scaling = np.ones(
                    (len(sigma_v_obs), len(ani_params)), dtype="float32"
                )
        else:
            ani_scaling = np.ones((len(sigma_v_obs), len(ani_params)), dtype="float32")
            gamma_pl_params = None
            ani_scaling_2d = None

    # Lambda scaling from lens properties
    props = data.get("kwargs_lens_properties", {})
    r_eff = props.get("r_eff", 1.0)
    theta_e = props.get("theta_E", 1.0)
    lambda_scaling = r_eff / theta_e - 1.0 if theta_e > 0 else 0.0

    # External convergence
    kappa_los_type = data.get("los_distribution_individual", "NONE")
    kappa_kwargs = data.get("kwargs_los_individual", {})

    kappa_pdf = None
    kappa_centers = None
    kappa_bin_edges = None
    kappa_cdf = None
    kappa_gev_mu = None
    kappa_gev_sigma = None
    kappa_gev_xi = None

    if kappa_los_type == "PDF":
        bin_edges = np.asarray(kappa_kwargs.get("bin_edges", []), dtype="float32")
        pdf_array = np.asarray(kappa_kwargs.get("pdf_array", []), dtype="float32")
        if len(bin_edges) > 1:
            norm = float(np.sum(pdf_array))
            if norm > 0:
                weights = pdf_array / norm
            else:
                weights = np.ones_like(pdf_array) / float(len(pdf_array))
            cdf = np.zeros_like(bin_edges, dtype="float32")
            cdf[1:] = np.cumsum(weights, dtype="float32")
            cdf[-1] = 1.0
            kappa_bin_edges = bin_edges.astype("float32")
            kappa_cdf = cdf
            kappa_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
            kappa_pdf = pdf_array
    elif kappa_los_type == "GEV":
        kappa_gev_mu = kappa_kwargs.get("mean", 0.0)
        kappa_gev_sigma = kappa_kwargs.get("sigma", 0.05)
        kappa_gev_xi = kappa_kwargs.get("xi", 0.0)

    vel_disp_scaling = data.get("vel_disp_scaling_distributions")
    vel_disp_scaling_samples = None
    if vel_disp_scaling is not None:
        vel_disp_scaling_samples = np.asarray(vel_disp_scaling, dtype="float32")

    return ExternalLensData(
        name=name,
        z_lens=float(data["z_lens"]),
        z_source=float(data["z_source"]),
        lambda_scaling=lambda_scaling,
        sigma_v_obs=jnp.asarray(sigma_v_obs, dtype=_DTYPE),
        cov_meas=jnp.asarray(cov_meas, dtype=_DTYPE),
        cov_j_sqrt=jnp.asarray(cov_j_sqrt, dtype=_DTYPE),
        j_model=jnp.asarray(j_model, dtype=_DTYPE),
        ani_params=jnp.asarray(ani_params, dtype=_DTYPE),
        ani_scaling=jnp.asarray(ani_scaling, dtype=_DTYPE),
        gamma_pl_params=(
            jnp.asarray(gamma_pl_params, dtype=_DTYPE)
            if gamma_pl_params is not None
            else None
        ),
        ani_scaling_2d=(
            jnp.asarray(ani_scaling_2d, dtype=_DTYPE)
            if ani_scaling_2d is not None
            else None
        ),
        gamma_pl_prior_mean=gamma_pl_prior_mean,
        gamma_pl_prior_sigma=gamma_pl_prior_sigma,
        ani_spline_tx=(
            jnp.asarray(ani_spline_tx, dtype=_DTYPE)
            if ani_spline_tx is not None
            else None
        ),
        ani_spline_ty=(
            jnp.asarray(ani_spline_ty, dtype=_DTYPE)
            if ani_spline_ty is not None
            else None
        ),
        ani_spline_coeffs=(
            jnp.asarray(ani_spline_coeffs, dtype=_DTYPE)
            if ani_spline_coeffs is not None
            else None
        ),
        kappa_los_type=kappa_los_type,
        kappa_pdf=(
            jnp.asarray(kappa_pdf, dtype=_DTYPE) if kappa_pdf is not None else None
        ),
        kappa_centers=(
            jnp.asarray(kappa_centers, dtype=_DTYPE)
            if kappa_centers is not None
            else None
        ),
        kappa_bin_edges=(
            jnp.asarray(kappa_bin_edges, dtype=_DTYPE)
            if kappa_bin_edges is not None
            else None
        ),
        kappa_cdf=(
            jnp.asarray(kappa_cdf, dtype=_DTYPE) if kappa_cdf is not None else None
        ),
        kappa_gev_mu=kappa_gev_mu,
        kappa_gev_sigma=kappa_gev_sigma,
        kappa_gev_xi=kappa_gev_xi,
        vel_disp_scaling_samples=(
            jnp.asarray(vel_disp_scaling_samples, dtype=_DTYPE)
            if vel_disp_scaling_samples is not None
            else None
        ),
    )


class ExternalLensLikelihood(Likelihood):
    """
    External lens likelihood for SLACS/SL2S samples.

    These lenses provide kinematic constraints only (no time-delay distances).
    They help break the H0-λ_int degeneracy by constraining λ_int independently.

    Physics:
    - σ²_model = c² × J × (D_s/D_ds) × λ_int × kin_scaling
    - D_s/D_ds depends only on Ω_m (NOT H0!)
    - So by fitting σ_obs, we constrain λ_int without constraining H0
    - Combined with time-delay lenses (which constrain H0 × λ_int), we break degeneracy

    Parameters
    ----------
    cosmology_class : type
        Cosmology model class (LCDM, wCDM, etc.)
    data_path : str
        Path to processed external lens data (SLACS/SL2S pkl files)
    sample_type : str
        "SLACS" or "SL2S" - determines which sample to load
    lens_names : list, optional
        Subset of lenses to use. If None, uses all available.
    """

    def __init__(
        self,
        cosmology_class: Optional[type] = None,
        data_path: Optional[str] = None,
        sample_type: str = "SLACS",
        lens_names: Optional[Iterable[str]] = None,
        lambda_bounds: Tuple[float, float] = (0.5, 1.5),
        anisotropy_bounds: Tuple[float, float] = (0.1, 5.0),
        gamma_pl_bounds: Tuple[float, float] = (1.1, 2.9),
        anisotropy_parameterization: str = "beta",
        normalized: bool = False,
        kin_axi_correction: bool = False,
        remove_gamma_pl_prior: bool = True,
        gamma_pl_sampling: bool = True,
        num_distribution_draws: int = 200,
        distribution_seed: int = 0,
        use_spline: bool = True,
        max_vel_disp_nodes: Optional[int] = None,
        use_quality_data_only: bool = True,
        use_selected_lens_only: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.cosmology_class = cosmology_class
        self.sample_type = sample_type
        self.sample_type_upper = sample_type.upper()
        self.lambda_bounds = lambda_bounds
        self.anisotropy_parameterization = str(anisotropy_parameterization).upper()
        self.anisotropy_bounds = anisotropy_bounds
        self.gamma_pl_bounds = gamma_pl_bounds
        # Match hierarc defaults (CosmoLikelihood(normalized=False)) for Fig. 8–12.
        self.normalized = bool(normalized)
        self.kin_axi_correction = bool(kin_axi_correction)
        self.remove_gamma_pl_prior = bool(remove_gamma_pl_prior)
        self.gamma_pl_sampling = bool(gamma_pl_sampling)
        self.use_spline = bool(use_spline)
        if (
            self.anisotropy_parameterization == "TAN_RAD"
            and self.anisotropy_bounds
            == (
                0.1,
                5.0,
            )
        ):
            # Paper/Table 3 baseline prior for constant anisotropy in TAN_RAD.
            self.anisotropy_bounds = (0.87, 1.12)

        self.num_distribution_draws = int(num_distribution_draws)
        if self.num_distribution_draws <= 0:
            raise ValueError("num_distribution_draws must be a positive integer.")

        self.use_quality_data_only = bool(use_quality_data_only)
        self.use_selected_lens_only = bool(use_selected_lens_only)
        self.distribution_seed = int(distribution_seed)
        base = (np.arange(self.num_distribution_draws, dtype="float32") + 0.5) / float(
            self.num_distribution_draws
        )
        rng = np.random.default_rng(self.distribution_seed)
        self._quantile_base = jnp.asarray(base, dtype=_DTYPE)
        self._quantile_perm_lam = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_ani = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_kappa = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_vel = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        self._quantile_perm_gamma = jnp.asarray(
            rng.permutation(self.num_distribution_draws), dtype=jnp.int32
        )
        if max_vel_disp_nodes is None:
            if self.kin_axi_correction:
                max_vel_disp_nodes = self.num_distribution_draws
        elif max_vel_disp_nodes <= 0:
            max_vel_disp_nodes = None
        self.max_vel_disp_nodes = max_vel_disp_nodes

        # Default data path based on sample type
        # Go up 3 levels: lensing/ -> likelihoods/ -> hicosmo/ -> project root
        if data_path is None:
            base = Path(__file__).resolve().parents[3] / "TDCOSMO2025_public"
            if self.sample_type_upper == "SLACS":
                data_path = (
                    base / "ExternalLenses" / "SLACS" / "slacs_kcwi_const_processed.pkl"
                )
            elif self.sample_type_upper == "SL2S":
                data_path = (
                    base / "ExternalLenses" / "SL2S" / "sl2s_const_processed_all.pkl"
                )
            else:
                raise ValueError(f"Unknown sample_type: {sample_type}")
        self.data_path = Path(data_path)

        # Load lens data
        self.lens_data: Dict[str, ExternalLensData] = {}
        self._load_external_data(lens_names)
        self._gamma_param_map: Dict[str, str] = {}
        if self.gamma_pl_sampling:
            for lname, lens in self.lens_data.items():
                if lens.gamma_pl_params is None or lens.ani_scaling_2d is None:
                    continue
                self._gamma_param_map[lname] = f"gamma_pl_{lname}"

        super().__init__(
            name=name or f"external_{sample_type.lower()}",
            data_path=str(self.data_path),
            **kwargs,
        )
        self.initialize()

        if cosmology_class is not None:
            logger.info(
                f"External lens loaded: {len(self.lens_data)} lenses ({sample_type})"
            )
            logger.info(f"  Lenses: {', '.join(self.lens_data.keys())}")

        self._packed_data = None
        self._use_packed = False
        if self.lens_data and (not self.use_spline) and (not self.normalized):
            self._build_packed_data()

        self._jitted_call = None
        self._jit_param_names = None

    def _load_external_data(self, lens_names: Optional[Iterable[str]]) -> None:
        """Load external lens data from pkl file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"External lens data not found: {self.data_path}")

        with self.data_path.open("rb") as f:
            data_list = pickle.load(f)

        # ------------------------------------------------------------
        # Optional: axisymmetric JAM correction draws (sigma_axi/sigma_sph)
        # ------------------------------------------------------------
        vel_disp_map: Optional[Dict[str, np.ndarray]] = None
        if self.kin_axi_correction:
            # Go up 3 levels: lensing/ -> likelihoods/ -> hicosmo/ -> project root
            correction_root = (
                Path(__file__).resolve().parents[3]
                / "TDCOSMO2025_public"
                / "kin_axi_jam_scaling"
            )
            if self.sample_type_upper == "SLACS":
                correction_path = correction_root / "kcwi_correction.pickle"
            elif self.sample_type_upper == "SL2S":
                correction_path = correction_root / "sl2s_correction.pickle"
            else:
                correction_path = None

            if correction_path is None or not correction_path.exists():
                raise ValueError(
                    "kin_axi_correction requested but correction file not found for "
                    f"sample_type={self.sample_type_upper}."
                )

            with correction_path.open("rb") as f:
                corr_list = pickle.load(f)
            vel_disp_map = {
                _sanitize(entry.get("name", "")): np.asarray(
                    entry.get("correction_combined", []), dtype="float32"
                )
                for entry in corr_list
            }

        # ------------------------------------------------------------
        # Optional: LOS kappa_ext distributions (matches likelihood_sampling.py)
        # ------------------------------------------------------------
        sl2s_gev_map: Optional[Dict[str, Tuple[float, float, float]]] = None
        if self.sample_type_upper == "SL2S":
            gev_path = self.data_path.parent / "kappa_ext" / "sl2s_los_gev.csv"
            sl2s_gev_map = {}
            if gev_path.exists():
                import csv

                with gev_path.open("r", newline="") as fh:
                    reader = csv.DictReader(fh)
                    for row in reader:
                        try:
                            name = str(row.get("name", ""))
                            mu = float(row.get("mu_kext", 0.0))
                            xi = float(row.get("xi_kext", 0.0))
                            log_sigma = float(row.get("log_sigma_kext", math.log(0.05)))
                            sl2s_gev_map[name] = (mu, math.exp(log_sigma), xi)
                        except (TypeError, ValueError):
                            continue

        if lens_names is not None:
            target_names = set(lens_names)
        elif self.use_selected_lens_only:
            # Match the paper's default sample selection (likelihood_sampling.py).
            if self.sample_type_upper == "SLACS":
                target_names = {
                    "SDSSJ0029-0055",
                    "SDSSJ0037-0942",
                    "SDSSJ1112+0826",
                    "SDSSJ1204+0358",
                    "SDSSJ1250+0523",
                    "SDSSJ1306+0600",
                    "SDSSJ1402+6321",
                    "SDSSJ1531-0105",
                    "SDSSJ1621+3931",
                    "SDSSJ1627-0053",
                    "SDSSJ1630+4520",
                }
            elif self.sample_type_upper == "SL2S":
                target_names = {
                    "SL2SJ0226-0420",
                    "SL2SJ0855-0147",
                    "SL2SJ0904-0059",
                    "SL2SJ2221+0115",
                }
            else:
                target_names = None
        else:
            target_names = None

        for lens_dict in data_list:
            name = lens_dict.get("name", "unknown")
            if target_names is not None and name not in target_names:
                continue
            if self.use_quality_data_only:
                flag_imaging = float(lens_dict.get("flag_imaging", 1))
                flag_ifu = float(lens_dict.get("flag_ifu", 1))
                if flag_imaging < 1 or flag_ifu < 1:
                    continue
            # Work on a shallow copy to avoid mutating the raw list.
            lens_payload = dict(lens_dict)

            # Remove IFU gamma_pl prior for SLACS lenses (official default).
            if self.sample_type_upper == "SLACS" and self.remove_gamma_pl_prior:
                prior_list = list(lens_payload.get("prior_list", []) or [])
                for idx, item in enumerate(prior_list):
                    if (
                        isinstance(item, (list, tuple))
                        and len(item) > 0
                        and item[0] == "gamma_pl"
                    ):
                        prior_list.pop(idx)
                        break
                lens_payload["prior_list"] = prior_list

            # Attach LOS kappa_ext if missing in the processed dict.
            if lens_payload.get("los_distribution_individual") is None:
                if self.sample_type_upper == "SLACS":
                    kappa_dir = self.data_path.parent / "kappa_ext"
                    kappa_choice_ending = (
                        "_computed_1innermask_nobeta_zgap-1.0_-1.0_fiducial_120_gal_120_"
                        "oneoverr_23.0_med_increments2_2_emptymsk.cat"
                    )
                    kappa_bins = np.linspace(-0.05, 0.2, 50, dtype="float32")
                    filepath = kappa_dir / f"kappahist_{name}{kappa_choice_ending}"
                    if filepath.exists():
                        try:
                            output = np.loadtxt(filepath, delimiter=" ", skiprows=1)
                            kappa_sample = output[:, 0]
                            kappa_weights = output[:, 1]
                            kappa_pdf, kappa_bin_edges = np.histogram(
                                kappa_sample,
                                weights=kappa_weights,
                                bins=kappa_bins,
                                density=True,
                            )
                            lens_payload["los_distribution_individual"] = "PDF"
                            lens_payload["kwargs_los_individual"] = {
                                "bin_edges": kappa_bin_edges.astype("float32"),
                                "pdf_array": kappa_pdf.astype("float32"),
                            }
                        except Exception:
                            pass
                elif self.sample_type_upper == "SL2S" and sl2s_gev_map is not None:
                    params = sl2s_gev_map.get(name)
                    if params is not None:
                        mu, sigma, xi = params
                        lens_payload["los_distribution_individual"] = "GEV"
                        lens_payload["kwargs_los_individual"] = {
                            "mean": float(mu),
                            "sigma": float(sigma),
                            "xi": float(xi),
                        }

            sanitized = _sanitize(name)
            if vel_disp_map is not None:
                corr = vel_disp_map.get(sanitized)
                if corr is not None and corr.size > 0:
                    if self.max_vel_disp_nodes is not None:
                        corr = _downsample_rows_evenly(
                            corr, int(self.max_vel_disp_nodes)
                        )
                    lens_payload["vel_disp_scaling_distributions"] = corr
            self.lens_data[sanitized] = _load_external_lens(lens_payload, sanitized)

        self.lens_names = list(self.lens_data.keys())

    def _build_packed_data(self) -> None:
        """Pack external lens data into fixed-size JAX arrays for vectorized evaluation."""
        lenses = list(self.lens_data.values())
        if not lenses:
            return

        n_lens = len(lenses)
        max_n_meas = max(int(l.sigma_v_obs.shape[0]) for l in lenses)
        max_n_ani = max(int(l.ani_params.shape[0]) for l in lenses)
        max_n_gamma = max(
            int(l.gamma_pl_params.shape[0]) if l.gamma_pl_params is not None else 1
            for l in lenses
        )
        max_n_kappa = max(
            int(l.kappa_centers.shape[0]) if l.kappa_centers is not None else 1
            for l in lenses
        )

        num_draws = int(self.num_distribution_draws)

        def _pad_axis(axis: np.ndarray, size: int) -> np.ndarray:
            if axis.size >= size:
                return axis[:size]
            pad_val = axis[-1] if axis.size > 0 else 0.0
            pad = np.full((size - axis.size,), pad_val, dtype=axis.dtype)
            return np.concatenate([axis, pad], axis=0)

        sigma_v_obs = np.zeros((n_lens, max_n_meas), dtype="float32")
        j_model = np.zeros_like(sigma_v_obs)
        cov_meas = np.zeros((n_lens, max_n_meas, max_n_meas), dtype="float32")
        cov_j_sqrt = np.zeros_like(cov_meas)
        mask_meas = np.zeros_like(sigma_v_obs)

        ani_params = np.zeros((n_lens, max_n_ani), dtype="float32")
        ani_n = np.ones((n_lens,), dtype="int32")
        ani_scaling = np.ones((n_lens, max_n_meas, max_n_ani), dtype="float32")
        ani_scaling_2d = np.ones(
            (n_lens, max_n_meas, max_n_ani, max_n_gamma), dtype="float32"
        )
        gamma_params = np.zeros((n_lens, max_n_gamma), dtype="float32")
        gamma_n = np.ones((n_lens,), dtype="int32")
        has_gamma = np.zeros((n_lens,), dtype="float32")
        gamma_prior_mean = np.zeros((n_lens,), dtype="float32")
        gamma_prior_sigma = np.zeros((n_lens,), dtype="float32")

        ani_min = np.zeros((n_lens,), dtype="float32")
        ani_max = np.zeros((n_lens,), dtype="float32")

        kappa_type = np.zeros((n_lens,), dtype="int32")
        kappa_centers = np.zeros((n_lens, max_n_kappa), dtype="float32")
        kappa_pdf = np.zeros((n_lens, max_n_kappa), dtype="float32")
        kappa_edges = np.zeros((n_lens, max_n_kappa + 1), dtype="float32")
        kappa_cdf = np.zeros((n_lens, max_n_kappa + 1), dtype="float32")
        kappa_n = np.ones((n_lens,), dtype="int32")
        kappa_mu = np.zeros((n_lens,), dtype="float32")
        kappa_sigma = np.zeros((n_lens,), dtype="float32")
        kappa_xi = np.zeros((n_lens,), dtype="float32")

        vel_samples = np.ones((n_lens, num_draws, max_n_meas), dtype="float32")
        vel_n = np.ones((n_lens,), dtype="int32")

        z_lens = np.zeros((n_lens,), dtype="float32")
        z_source = np.zeros((n_lens,), dtype="float32")
        lambda_scaling = np.zeros((n_lens,), dtype="float32")

        for i, lens in enumerate(lenses):
            n_meas = int(lens.sigma_v_obs.shape[0])
            mask_meas[i, :n_meas] = 1.0
            sigma_v_obs[i, :n_meas] = np.asarray(lens.sigma_v_obs)
            j_model[i, :n_meas] = np.asarray(lens.j_model)
            cov_meas[i, :n_meas, :n_meas] = np.asarray(lens.cov_meas)
            cov_j_sqrt[i, :n_meas, :n_meas] = np.asarray(lens.cov_j_sqrt)

            ani_axis = np.asarray(lens.ani_params, dtype="float32")
            ani_min[i] = float(ani_axis[0])
            ani_max[i] = float(ani_axis[-1])
            ani_params[i] = _pad_axis(ani_axis, max_n_ani)
            ani_n[i] = int(ani_axis.shape[0])
            ani_scaling[i, :n_meas, : ani_axis.shape[0]] = np.asarray(
                lens.ani_scaling, dtype="float32"
            )

            if lens.gamma_pl_params is not None and lens.ani_scaling_2d is not None:
                has_gamma[i] = 1.0
                gamma_axis = np.asarray(lens.gamma_pl_params, dtype="float32")
                gamma_n[i] = int(gamma_axis.shape[0])
                gamma_params[i] = _pad_axis(gamma_axis, max_n_gamma)
                ani_scaling_2d[
                    i, :n_meas, : ani_axis.shape[0], : gamma_axis.shape[0]
                ] = np.asarray(lens.ani_scaling_2d, dtype="float32")
                if lens.gamma_pl_prior_mean is not None:
                    gamma_prior_mean[i] = float(lens.gamma_pl_prior_mean)
                if lens.gamma_pl_prior_sigma is not None:
                    gamma_prior_sigma[i] = float(lens.gamma_pl_prior_sigma)

            if lens.kappa_los_type == "PDF":
                kappa_type[i] = 1
                if lens.kappa_centers is not None and lens.kappa_pdf is not None:
                    kappa_axis = np.asarray(lens.kappa_centers, dtype="float32")
                    kappa_centers[i] = _pad_axis(kappa_axis, max_n_kappa)
                    kappa_pdf[i, : kappa_axis.shape[0]] = np.asarray(
                        lens.kappa_pdf, dtype="float32"
                    )
                    kappa_n[i] = int(kappa_axis.shape[0])
                if lens.kappa_bin_edges is not None and lens.kappa_cdf is not None:
                    edge_axis = np.asarray(lens.kappa_bin_edges, dtype="float32")
                    cdf_axis = np.asarray(lens.kappa_cdf, dtype="float32")
                    kappa_edges[i] = _pad_axis(edge_axis, max_n_kappa + 1)
                    kappa_cdf[i] = _pad_axis(cdf_axis, max_n_kappa + 1)
                elif lens.kappa_centers is not None and lens.kappa_pdf is not None:
                    centers = np.asarray(lens.kappa_centers, dtype="float32")
                    if centers.size > 1:
                        edges = np.concatenate(
                            [
                                [centers[0] - 0.5 * (centers[1] - centers[0])],
                                0.5 * (centers[:-1] + centers[1:]),
                                [centers[-1] + 0.5 * (centers[-1] - centers[-2])],
                            ],
                            axis=0,
                        )
                    else:
                        width = 1e-3
                        edges = np.array(
                            [centers[0] - width, centers[0] + width], dtype="float32"
                        )
                    norm = float(np.sum(lens.kappa_pdf))
                    if norm > 0:
                        weights = np.asarray(lens.kappa_pdf, dtype="float32") / norm
                    else:
                        weights = np.ones_like(centers) / float(len(centers))
                    cdf_axis = np.zeros_like(edges, dtype="float32")
                    cdf_axis[1:] = np.cumsum(weights, dtype="float32")
                    cdf_axis[-1] = 1.0
                    kappa_edges[i] = _pad_axis(edges, max_n_kappa + 1)
                    kappa_cdf[i] = _pad_axis(cdf_axis, max_n_kappa + 1)
            elif lens.kappa_los_type == "GEV":
                kappa_type[i] = 2
                if lens.kappa_gev_mu is not None:
                    kappa_mu[i] = float(lens.kappa_gev_mu)
                if lens.kappa_gev_sigma is not None:
                    kappa_sigma[i] = float(lens.kappa_gev_sigma)
                if lens.kappa_gev_xi is not None:
                    kappa_xi[i] = float(lens.kappa_gev_xi)
            else:
                kappa_type[i] = 0

            if lens.vel_disp_scaling_samples is not None:
                vel = np.asarray(lens.vel_disp_scaling_samples, dtype="float32")
                if vel.ndim == 1:
                    vel = vel[:, None]
                if vel.shape[0] < num_draws:
                    pad = np.repeat(vel[-1:, :], num_draws - vel.shape[0], axis=0)
                    vel = np.concatenate([vel, pad], axis=0)
                elif vel.shape[0] > num_draws:
                    vel = _downsample_rows_evenly(vel, num_draws)
                vel_n[i] = int(vel.shape[0])
                vel_samples[i, : vel.shape[0], : vel.shape[1]] = vel

            z_lens[i] = float(lens.z_lens)
            z_source[i] = float(lens.z_source)
            lambda_scaling[i] = float(lens.lambda_scaling)

        self._packed_data = {
            "sigma_v_obs": jnp.asarray(sigma_v_obs, dtype=_DTYPE),
            "j_model": jnp.asarray(j_model, dtype=_DTYPE),
            "cov_meas": jnp.asarray(cov_meas, dtype=_DTYPE),
            "cov_j_sqrt": jnp.asarray(cov_j_sqrt, dtype=_DTYPE),
            "mask_meas": jnp.asarray(mask_meas, dtype=_DTYPE),
            "ani_params": jnp.asarray(ani_params, dtype=_DTYPE),
            "ani_scaling": jnp.asarray(ani_scaling, dtype=_DTYPE),
            "ani_scaling_2d": jnp.asarray(ani_scaling_2d, dtype=_DTYPE),
            "ani_n": jnp.asarray(ani_n, dtype=jnp.int32),
            "gamma_params": jnp.asarray(gamma_params, dtype=_DTYPE),
            "gamma_n": jnp.asarray(gamma_n, dtype=jnp.int32),
            "has_gamma": jnp.asarray(has_gamma, dtype=_DTYPE),
            "gamma_prior_mean": jnp.asarray(gamma_prior_mean, dtype=_DTYPE),
            "gamma_prior_sigma": jnp.asarray(gamma_prior_sigma, dtype=_DTYPE),
            "ani_min": jnp.asarray(ani_min, dtype=_DTYPE),
            "ani_max": jnp.asarray(ani_max, dtype=_DTYPE),
            "kappa_type": jnp.asarray(kappa_type, dtype=jnp.int32),
            "kappa_centers": jnp.asarray(kappa_centers, dtype=_DTYPE),
            "kappa_pdf": jnp.asarray(kappa_pdf, dtype=_DTYPE),
            "kappa_edges": jnp.asarray(kappa_edges, dtype=_DTYPE),
            "kappa_cdf": jnp.asarray(kappa_cdf, dtype=_DTYPE),
            "kappa_n": jnp.asarray(kappa_n, dtype=jnp.int32),
            "kappa_mu": jnp.asarray(kappa_mu, dtype=_DTYPE),
            "kappa_sigma": jnp.asarray(kappa_sigma, dtype=_DTYPE),
            "kappa_xi": jnp.asarray(kappa_xi, dtype=_DTYPE),
            "vel_samples": jnp.asarray(vel_samples, dtype=_DTYPE),
            "vel_n": jnp.asarray(vel_n, dtype=jnp.int32),
            "z_lens": jnp.asarray(z_lens, dtype=_DTYPE),
            "z_source": jnp.asarray(z_source, dtype=_DTYPE),
            "lambda_scaling": jnp.asarray(lambda_scaling, dtype=_DTYPE),
        }
        self._use_packed = True

    def _default_dataset_name(self) -> str:
        return f"external_{self.sample_type.lower()}"

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self) -> Dict[str, Any]:
        return {}

    def theory(self, cosmology, **kwargs):
        raise NotImplementedError

    def _distance_ratio(self, cosmology, z_lens: float, z_source: float) -> jnp.ndarray:
        """
        Compute D_s / D_ds ratio for external lenses.

        This ratio depends only on Ω_m, NOT on H0!
        """
        ds = jnp.asarray(cosmology.angular_diameter_distance(z_source), dtype=_DTYPE)

        if hasattr(cosmology, "angular_diameter_distance_between"):
            dds = jnp.asarray(
                cosmology.angular_diameter_distance_between(z_lens, z_source),
                dtype=_DTYPE,
            )
        else:
            d_c_l = cosmology.comoving_distance(z_lens)
            d_c_s = cosmology.comoving_distance(z_source)
            dds = jnp.asarray((d_c_s - d_c_l) / (1.0 + z_source), dtype=_DTYPE)

        return ds / jnp.maximum(dds, _EPS)

    def _distribution_quantiles(
        self,
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        base = self._quantile_base
        return (
            base[self._quantile_perm_lam],
            base[self._quantile_perm_ani],
            base[self._quantile_perm_kappa],
            base[self._quantile_perm_vel],
            base[self._quantile_perm_gamma],
        )

    def _anisotropy_to_grid(self, value: jnp.ndarray) -> jnp.ndarray:
        """Map sampled anisotropy parameter to the beta_ani grid used by the data."""
        value = jnp.asarray(value, dtype=_DTYPE)
        if self.anisotropy_parameterization == "TAN_RAD":
            return 1.0 - value**2
        return value

    @staticmethod
    def _standard_normal_cdf(x: jnp.ndarray) -> jnp.ndarray:
        return 0.5 * (1.0 + erf(x / jnp.sqrt(2.0)))

    @staticmethod
    def _standard_normal_ppf(p: jnp.ndarray) -> jnp.ndarray:
        p = jnp.clip(p, _EPS, 1.0 - _EPS)
        return jnp.sqrt(2.0) * erfinv(2.0 * p - 1.0)

    def _truncated_normal_draws(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        bounds: Tuple[float, float],
        quantiles: jnp.ndarray,
    ) -> jnp.ndarray:
        mean = jnp.asarray(mean, dtype=_DTYPE)
        sigma = jnp.asarray(sigma, dtype=_DTYPE)
        sigma = jnp.maximum(sigma, jnp.array(1e-6, dtype=_DTYPE))
        lower = jnp.array(bounds[0], dtype=_DTYPE)
        upper = jnp.array(bounds[1], dtype=_DTYPE)

        a = (lower - mean) / sigma
        b = (upper - mean) / sigma
        cdf_a = self._standard_normal_cdf(a)
        cdf_b = self._standard_normal_cdf(b)
        mass = jnp.maximum(cdf_b - cdf_a, _EPS)
        p = cdf_a + quantiles * mass
        z = self._standard_normal_ppf(p)
        return mean + sigma * z

    def _normal_draws(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        quantiles: jnp.ndarray,
        *,
        override: Optional[float] = None,
    ) -> jnp.ndarray:
        """Deterministic draws from an (untruncated) normal distribution."""
        if override is not None:
            return jnp.full_like(quantiles, jnp.asarray(override, dtype=_DTYPE))
        mean = jnp.asarray(mean, dtype=_DTYPE)
        sigma = jnp.asarray(sigma, dtype=_DTYPE)
        sigma = jnp.maximum(sigma, jnp.array(1e-6, dtype=_DTYPE))
        return mean + sigma * self._standard_normal_ppf(quantiles)

    @staticmethod
    def _gev_ppf(p: jnp.ndarray, *, mu: float, sigma: float, xi: float) -> jnp.ndarray:
        """Inverse CDF for the GEV distribution used for SL2S LOS kappa."""
        p = jnp.clip(p, _EPS, 1.0 - _EPS)
        mu_a = jnp.array(mu, dtype=_DTYPE)
        sigma_a = jnp.maximum(
            jnp.array(sigma, dtype=_DTYPE), jnp.array(1e-6, dtype=_DTYPE)
        )
        xi_a = jnp.array(xi, dtype=_DTYPE)

        t = -jnp.log(p)
        gumbel = mu_a - sigma_a * jnp.log(t)
        xi_safe = jnp.where(jnp.abs(xi_a) < 1e-6, jnp.array(1.0, dtype=_DTYPE), xi_a)
        gev = mu_a + sigma_a * (1.0 - t**xi_safe) / xi_safe
        return jnp.where(jnp.abs(xi_a) < 1e-6, gumbel, gev)

    def _kappa_draws(
        self, lens: ExternalLensData, quantiles: jnp.ndarray
    ) -> jnp.ndarray:
        if (
            lens.kappa_los_type == "PDF"
            and lens.kappa_bin_edges is not None
            and lens.kappa_cdf is not None
        ):
            cdf = lens.kappa_cdf
            edges = lens.kappa_bin_edges
            idx = jnp.searchsorted(cdf, quantiles, side="right") - 1
            idx = jnp.clip(idx, 0, cdf.shape[0] - 2)
            cdf0 = cdf[idx]
            cdf1 = cdf[idx + 1]
            edge0 = edges[idx]
            edge1 = edges[idx + 1]
            t = (quantiles - cdf0) / jnp.maximum(cdf1 - cdf0, _EPS)
            return edge0 + t * (edge1 - edge0)
        if (
            lens.kappa_los_type == "PDF"
            and lens.kappa_pdf is not None
            and lens.kappa_centers is not None
        ):
            weights = lens.kappa_pdf / jnp.maximum(jnp.sum(lens.kappa_pdf), _EPS)
            cdf = jnp.cumsum(weights)
            cdf = cdf / jnp.maximum(cdf[-1], _EPS)
            idx = jnp.searchsorted(cdf, quantiles, side="left")
            idx = jnp.clip(idx, 0, lens.kappa_centers.shape[0] - 1)
            return lens.kappa_centers[idx]
        if (
            lens.kappa_los_type == "GEV"
            and lens.kappa_gev_mu is not None
            and lens.kappa_gev_sigma is not None
            and lens.kappa_gev_xi is not None
        ):
            return self._gev_ppf(
                quantiles,
                mu=float(lens.kappa_gev_mu),
                sigma=float(lens.kappa_gev_sigma),
                xi=float(lens.kappa_gev_xi),
            )
        return jnp.zeros_like(quantiles)

    def _vel_disp_scaling_draws(
        self, lens: ExternalLensData, quantiles: jnp.ndarray
    ) -> Optional[jnp.ndarray]:
        samples = lens.vel_disp_scaling_samples
        if samples is None:
            return None
        n = jnp.asarray(samples.shape[0], dtype=jnp.int32)
        idx = jnp.floor(quantiles * n.astype(_DTYPE)).astype(jnp.int32)
        idx = jnp.clip(idx, 0, n - 1)
        return samples[idx]

    def _integrated_external_lens_loglike_draws(
        self,
        lens: ExternalLensData,
        ds_dds: jnp.ndarray,
        *,
        lambda_loc: jnp.ndarray,
        lambda_sigma: jnp.ndarray,
        a_mean: jnp.ndarray,
        a_sigma: jnp.ndarray,
        gamma_pl: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        q_lam, q_ani, q_kappa, q_vel, q_gamma = self._distribution_quantiles()
        # Match hierarc: per-lens lambda_mst draws are from an (untruncated) Gaussian.
        lambda_draws = self._normal_draws(
            lambda_loc,
            lambda_sigma,
            q_lam,
        )
        beta_min = lens.ani_params[0]
        beta_max = lens.ani_params[-1]
        if self.anisotropy_parameterization == "TAN_RAD":
            r_max = jnp.sqrt(jnp.maximum(1.0 - beta_min, jnp.array(0.0, dtype=_DTYPE)))
            r_min = jnp.sqrt(jnp.maximum(1.0 - beta_max, jnp.array(0.0, dtype=_DTYPE)))
            _ = r_min  # beta_max is 1 for current grids; keep for completeness.
            ani_bounds = (-r_max, r_max)
        else:
            ani_bounds = (beta_min, beta_max)
        ani_draws = self._truncated_normal_draws(a_mean, a_sigma, ani_bounds, q_ani)
        kappa_draws = self._kappa_draws(lens, q_kappa)

        gamma_draws: Optional[jnp.ndarray] = None
        if lens.gamma_pl_params is not None and lens.ani_scaling_2d is not None:
            if gamma_pl is not None:
                gamma_draws = jnp.full_like(q_lam, jnp.asarray(gamma_pl, dtype=_DTYPE))
            else:
                if (
                    lens.gamma_pl_prior_mean is not None
                    and lens.gamma_pl_prior_sigma is not None
                    and lens.gamma_pl_prior_sigma > 0
                ):
                    gamma_draws = self._truncated_normal_draws(
                        jnp.array(lens.gamma_pl_prior_mean, dtype=_DTYPE),
                        jnp.array(lens.gamma_pl_prior_sigma, dtype=_DTYPE),
                        self.gamma_pl_bounds,
                        q_gamma,
                    )
                else:
                    gamma_lower = jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE)
                    gamma_upper = jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE)
                    gamma_draws = gamma_lower + (gamma_upper - gamma_lower) * q_gamma

        ani_grid = self._anisotropy_to_grid(ani_draws)
        if gamma_draws is None:
            kin_scalings = lens.anisotropy_scaling_batch(
                ani_grid, use_spline=self.use_spline
            )
        else:
            kin_scalings = lens.anisotropy_scaling_batch(
                ani_grid, gamma_draws, use_spline=self.use_spline
            )
        vel_draws = self._vel_disp_scaling_draws(lens, q_vel)
        if vel_draws is not None:
            if vel_draws.ndim == 1:
                vel_factor = vel_draws[:, None]
            else:
                vel_factor = vel_draws
            kin_scalings = kin_scalings * vel_factor**2
        sqrt_kin = jnp.sqrt(kin_scalings)
        scaling_mats = sqrt_kin[:, :, None] * sqrt_kin[:, None, :]

        lambda_tot = lambda_draws * (1.0 - kappa_draws)
        lambda_tot = jnp.maximum(lambda_tot, jnp.array(1e-4, dtype=_DTYPE))
        ds_dds_eff = ds_dds * lambda_tot

        sigma_model = (
            jnp.sqrt(lens.j_model[None, :] * ds_dds_eff[:, None]) * sqrt_kin * c_km_s
        )
        cov_model = (
            lens.cov_j_sqrt[None, :, :]
            * scaling_mats
            * ds_dds_eff[:, None, None]
            * c_km_s**2
        )
        cov_total = cov_model + lens.cov_meas[None, :, :]
        n = lens.sigma_v_obs.shape[0]
        cov_total = cov_total + jnp.eye(n, dtype=_DTYPE)[None, :, :] * jnp.array(
            1e-6, dtype=_DTYPE
        )

        delta = lens.sigma_v_obs[None, :] - sigma_model
        if n == 1:
            sigma2 = cov_total[:, 0, 0]
            delta0 = delta[:, 0]
            if not self.normalized:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2),
                    _NEG_LARGE,
                )
            else:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2 + jnp.log(sigma2) + jnp.log(_TWO_PI)),
                    _NEG_LARGE,
                )
        else:
            L = jnp.linalg.cholesky(cov_total)
            y = jax.scipy.linalg.solve_triangular(L, delta, lower=True)
            chi2 = jnp.sum(y**2, axis=-1)
            if not self.normalized:
                kin_log = -0.5 * chi2
            else:
                logdet = 2.0 * jnp.sum(
                    jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)), axis=-1
                )
                kin_log = -0.5 * (chi2 + logdet + n * jnp.log(_TWO_PI))

        loglike = jnp.where(jnp.isfinite(kin_log), kin_log, _NEG_LARGE)
        return logsumexp(loglike) - jnp.log(
            jnp.array(self.num_distribution_draws, dtype=_DTYPE)
        )

    def _integrated_external_lens_loglike_draws_packed(
        self,
        cosmology,
        *,
        lambda_mean: jnp.ndarray,
        lambda_sigma: jnp.ndarray,
        alpha_lambda: jnp.ndarray,
        a_mean: jnp.ndarray,
        a_sigma: jnp.ndarray,
        gamma_overrides: jnp.ndarray,
    ) -> jnp.ndarray:
        """Vectorized external lens likelihood using packed data arrays."""
        packed = self._packed_data
        if packed is None:
            return jnp.array(0.0, dtype=_DTYPE)

        q_lam, q_ani, q_kappa, q_vel, q_gamma = self._distribution_quantiles()
        ppf_lam = self._standard_normal_ppf(q_lam)

        z_lens = packed["z_lens"]
        z_source = packed["z_source"]
        lambda_scaling = packed["lambda_scaling"]

        ds_dds = jax.vmap(lambda zl, zs: self._distance_ratio(cosmology, zl, zs))(
            z_lens, z_source
        )

        lambda_loc = lambda_mean + alpha_lambda * lambda_scaling

        def lens_loglike(
            sigma_v_obs,
            cov_meas,
            cov_j_sqrt,
            j_model,
            mask_meas,
            ani_params,
            ani_scaling,
            ani_scaling_2d,
            ani_n,
            gamma_params,
            gamma_n,
            has_gamma,
            gamma_prior_mean,
            gamma_prior_sigma,
            ani_min,
            ani_max,
            kappa_type,
            kappa_centers,
            kappa_pdf,
            kappa_edges,
            kappa_cdf,
            kappa_n,
            kappa_mu,
            kappa_sigma,
            kappa_xi,
            vel_samples,
            vel_n,
            ds_dds_l,
            lambda_loc_l,
            gamma_override_l,
        ):
            # Lambda MST draws
            lambda_draws = lambda_loc_l + lambda_sigma * ppf_lam

            # Anisotropy draws (bounds from interpolation grid)
            if self.anisotropy_parameterization == "TAN_RAD":
                r_max = jnp.sqrt(
                    jnp.maximum(1.0 - ani_min, jnp.array(0.0, dtype=_DTYPE))
                )
                r_min = jnp.sqrt(
                    jnp.maximum(1.0 - ani_max, jnp.array(0.0, dtype=_DTYPE))
                )
                _ = r_min
                ani_bounds = (-r_max, r_max)
            else:
                ani_bounds = (ani_min, ani_max)
            ani_draws = self._truncated_normal_draws(a_mean, a_sigma, ani_bounds, q_ani)

            # Kappa draws
            def _kappa_draws_pdf(_):
                idx = jnp.searchsorted(kappa_cdf, q_kappa, side="right") - 1
                idx = jnp.clip(idx, 0, kappa_n - 1)
                cdf0 = kappa_cdf[idx]
                cdf1 = kappa_cdf[idx + 1]
                edge0 = kappa_edges[idx]
                edge1 = kappa_edges[idx + 1]
                t = (q_kappa - cdf0) / jnp.maximum(cdf1 - cdf0, _EPS)
                return edge0 + t * (edge1 - edge0)

            def _kappa_draws_gev(_):
                return self._gev_ppf(
                    q_kappa,
                    mu=kappa_mu,
                    sigma=kappa_sigma,
                    xi=kappa_xi,
                )

            def _kappa_draws_none(_):
                return jnp.zeros_like(q_kappa)

            kappa_draws = jax.lax.switch(
                kappa_type,
                (_kappa_draws_none, _kappa_draws_pdf, _kappa_draws_gev),
                operand=None,
            )

            # Gamma_pl draws (optional)
            gamma_lower = jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE)
            gamma_upper = jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE)
            gamma_override_mask = jnp.isfinite(gamma_override_l)

            def _gamma_draws(_):
                return jnp.where(
                    gamma_override_mask,
                    jnp.full_like(q_lam, gamma_override_l),
                    jax.lax.cond(
                        gamma_prior_sigma > 0.0,
                        lambda __: self._truncated_normal_draws(
                            gamma_prior_mean,
                            gamma_prior_sigma,
                            self.gamma_pl_bounds,
                            q_gamma,
                        ),
                        lambda __: gamma_lower + (gamma_upper - gamma_lower) * q_gamma,
                        operand=None,
                    ),
                )

            gamma_draws = jax.lax.cond(
                has_gamma > 0.5, _gamma_draws, lambda __: q_lam * 0.0, operand=None
            )

            # Map anisotropy parameterization to beta grid
            if self.anisotropy_parameterization == "TAN_RAD":
                ani_grid = 1.0 - ani_draws**2
            else:
                ani_grid = ani_draws

            # Interpolate kinematic scaling
            if gamma_draws is None:
                idx = jnp.searchsorted(ani_params, ani_grid, side="right") - 1
                idx = jnp.clip(idx, 0, ani_n - 2)
                x0 = ani_params[idx]
                x1 = ani_params[idx + 1]
                t = (ani_grid - x0) / jnp.maximum(x1 - x0, _EPS)
                v0 = jnp.take(ani_scaling, idx, axis=1)
                v1 = jnp.take(ani_scaling, idx + 1, axis=1)
                kin_scalings = jnp.swapaxes((1.0 - t) * v0 + t * v1, 0, 1)
            else:

                def _interp_one(a, g):
                    i = jnp.searchsorted(ani_params, a, side="right") - 1
                    j = jnp.searchsorted(gamma_params, g, side="right") - 1
                    i = jnp.clip(i, 0, ani_n - 2)
                    j = jnp.clip(j, 0, gamma_n - 2)
                    x0 = ani_params[i]
                    x1 = ani_params[i + 1]
                    y0 = gamma_params[j]
                    y1 = gamma_params[j + 1]
                    t = (a - x0) / jnp.maximum(x1 - x0, _EPS)
                    u = (g - y0) / jnp.maximum(y1 - y0, _EPS)
                    v00 = ani_scaling_2d[:, i, j]
                    v10 = ani_scaling_2d[:, i + 1, j]
                    v01 = ani_scaling_2d[:, i, j + 1]
                    v11 = ani_scaling_2d[:, i + 1, j + 1]
                    return (
                        (1.0 - t) * (1.0 - u) * v00
                        + t * (1.0 - u) * v10
                        + (1.0 - t) * u * v01
                        + t * u * v11
                    )

                kin_scalings = jax.vmap(_interp_one)(ani_grid, gamma_draws)

            # Axisymmetric correction draws
            idx_vel = jnp.floor(q_vel * vel_n.astype(_DTYPE)).astype(jnp.int32)
            idx_vel = jnp.clip(idx_vel, 0, vel_n - 1)
            vel_draws = vel_samples[idx_vel]
            kin_scalings = kin_scalings * vel_draws**2

            sqrt_kin = jnp.sqrt(kin_scalings)
            scaling_mats = sqrt_kin[:, :, None] * sqrt_kin[:, None, :]

            lambda_tot = lambda_draws * (1.0 - kappa_draws)
            lambda_tot = jnp.maximum(lambda_tot, jnp.array(1e-4, dtype=_DTYPE))
            ds_dds_eff = ds_dds_l * lambda_tot

            sigma_model = (
                jnp.sqrt(j_model[None, :] * ds_dds_eff[:, None]) * sqrt_kin * c_km_s
            )
            cov_model = (
                cov_j_sqrt[None, :, :]
                * scaling_mats
                * ds_dds_eff[:, None, None]
                * c_km_s**2
            )
            cov_total = cov_model + cov_meas[None, :, :]
            mask_mat = mask_meas[None, :, None] * mask_meas[None, None, :]
            cov_total = cov_total * mask_mat
            eye = jnp.eye(cov_total.shape[-1], dtype=_DTYPE)[None, :, :]
            cov_total = cov_total + eye * (1.0 - mask_meas)[None, None, :]
            cov_total = cov_total + eye * jnp.array(1e-6, dtype=_DTYPE)

            delta = (sigma_v_obs[None, :] - sigma_model) * mask_meas[None, :]
            L = jnp.linalg.cholesky(cov_total)
            y = jax.scipy.linalg.solve_triangular(L, delta, lower=True)
            chi2 = jnp.sum(y**2, axis=-1)
            kin_log = -0.5 * chi2

            loglike = jnp.where(jnp.isfinite(kin_log), kin_log, _NEG_LARGE)
            return logsumexp(loglike) - jnp.log(
                jnp.array(self.num_distribution_draws, dtype=_DTYPE)
            )

        loglikes = jax.vmap(lens_loglike)(
            packed["sigma_v_obs"],
            packed["cov_meas"],
            packed["cov_j_sqrt"],
            packed["j_model"],
            packed["mask_meas"],
            packed["ani_params"],
            packed["ani_scaling"],
            packed["ani_scaling_2d"],
            packed["ani_n"],
            packed["gamma_params"],
            packed["gamma_n"],
            packed["has_gamma"],
            packed["gamma_prior_mean"],
            packed["gamma_prior_sigma"],
            packed["ani_min"],
            packed["ani_max"],
            packed["kappa_type"],
            packed["kappa_centers"],
            packed["kappa_pdf"],
            packed["kappa_edges"],
            packed["kappa_cdf"],
            packed["kappa_n"],
            packed["kappa_mu"],
            packed["kappa_sigma"],
            packed["kappa_xi"],
            packed["vel_samples"],
            packed["vel_n"],
            ds_dds,
            lambda_loc,
            gamma_overrides,
        )

        return jnp.sum(loglikes)

    def _single_external_lens_loglike(
        self,
        lens: ExternalLensData,
        ds_dds: jnp.ndarray,
        lambda_mst: jnp.ndarray,
        kin_scaling: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute kinematic log-likelihood for a single external lens.

        σ²_model = c² × J × (D_s/D_ds) × λ_int × kin_scaling
        """
        # Compute model velocity dispersion
        # Note: lambda_mst enters directly (no kappa_ext for external lenses by default)
        sigma_model = (
            jnp.sqrt(lens.j_model * ds_dds * lambda_mst * kin_scaling) * c_km_s
        )

        # Build total covariance
        sqrt_kin = jnp.sqrt(kin_scaling)
        scaling_mat = jnp.outer(sqrt_kin, sqrt_kin)
        cov_model = lens.cov_j_sqrt * scaling_mat * ds_dds * lambda_mst * c_km_s**2
        cov_total = cov_model + lens.cov_meas
        n = lens.sigma_v_obs.shape[0]
        cov_total = cov_total + jnp.eye(n, dtype=_DTYPE) * jnp.array(1e-6, dtype=_DTYPE)

        # Compute chi-squared
        delta = lens.sigma_v_obs - sigma_model
        if n == 1:
            sigma2 = cov_total[0, 0]
            delta0 = delta[0]
            if not self.normalized:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2),
                    _NEG_LARGE,
                )
            else:
                kin_log = jnp.where(
                    sigma2 > 0.0,
                    -0.5 * (delta0**2 / sigma2 + jnp.log(sigma2) + jnp.log(_TWO_PI)),
                    _NEG_LARGE,
                )
        else:
            L = jnp.linalg.cholesky(cov_total)
            y = jax.scipy.linalg.solve_triangular(L, delta, lower=True)
            chi2 = jnp.dot(y, y)
            if not self.normalized:
                kin_log = -0.5 * chi2
            else:
                logdet = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
                kin_log = -0.5 * (chi2 + logdet + n * jnp.log(_TWO_PI))

        return jnp.where(jnp.isfinite(kin_log), kin_log, _NEG_LARGE)

    def _gaussian_nodes(
        self,
        mean: jnp.ndarray,
        sigma: jnp.ndarray,
        bounds: Tuple[float, float],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate Gauss-Hermite quadrature nodes for marginalization."""
        sigma = jnp.maximum(sigma, jnp.array(1e-6, dtype=_DTYPE))
        nodes = mean + jnp.sqrt(2.0) * sigma * _HERMITE_NODES
        weights = _HERMITE_WEIGHTS / _HERMITE_NORM
        weights = weights / jnp.maximum(jnp.sum(weights), _EPS)
        return nodes, weights

    def _integrated_external_lens_loglike(
        self,
        lens: ExternalLensData,
        ds_dds: jnp.ndarray,
        lambda_nodes: jnp.ndarray,
        lambda_weights: jnp.ndarray,
        ani_nodes: jnp.ndarray,
        ani_weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute marginalized log-likelihood for external lens.

        Marginalizes over λ_int and anisotropy distributions.
        """
        log_w_lam = jnp.log(lambda_weights + _EPS)
        log_w_ani = jnp.log(ani_weights + _EPS)

        # Pre-compute kinematic scalings for each anisotropy node
        kin_scalings = jax.vmap(
            lambda a: lens.anisotropy_scaling(a, use_spline=self.use_spline)
        )(ani_nodes)

        def loglike_for_params(
            lam: jnp.ndarray, kin_scaling: jnp.ndarray
        ) -> jnp.ndarray:
            return self._single_external_lens_loglike(lens, ds_dds, lam, kin_scaling)

        # Compute log-likelihood for all (lambda, ani) combinations
        def loglike_for_lambda(lam: jnp.ndarray) -> jnp.ndarray:
            loglikes = jax.vmap(lambda ks: loglike_for_params(lam, ks))(kin_scalings)
            return logsumexp(log_w_ani + loglikes)

        lambda_loglikes = jax.vmap(loglike_for_lambda)(lambda_nodes)
        return logsumexp(log_w_lam + lambda_loglikes)

    def log_likelihood(self, cosmology, **kwargs) -> float:
        """Compute total log-likelihood for all external lenses."""
        lambda_mean = jnp.asarray(kwargs.get("lambda_int_mean", 1.0), dtype=_DTYPE)
        lambda_sigma = jnp.asarray(kwargs.get("lambda_int_sigma", 0.05), dtype=_DTYPE)
        alpha_lambda = jnp.asarray(kwargs.get("alpha_lambda", 0.0), dtype=_DTYPE)
        a_mean = jnp.asarray(kwargs.get("a_ani_mean", 1.0), dtype=_DTYPE)
        a_sigma = jnp.asarray(kwargs.get("a_ani_sigma", 0.1), dtype=_DTYPE)

        total = jnp.array(0.0, dtype=_DTYPE)

        # Enforce positive scatters
        total = total + jnp.where(lambda_sigma > 0.0, 0.0, _NEG_LARGE)
        total = total + jnp.where(a_sigma > 0.0, 0.0, _NEG_LARGE)

        if self._use_packed and self._packed_data is not None:
            packed = self._packed_data
            overrides = []
            for name in self.lens_names:
                gamma_key = self._gamma_param_map.get(name)
                if gamma_key is None:
                    overrides.append(jnp.array(jnp.nan, dtype=_DTYPE))
                else:
                    gamma_override = kwargs.get(gamma_key, jnp.nan)
                    overrides.append(jnp.asarray(gamma_override, dtype=_DTYPE))
            gamma_overrides = jnp.stack(overrides, axis=0)

            gamma_penalty = TDCOSMOLikelihood._bounds_penalty(  # noqa: SLF001
                gamma_overrides,
                jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE),
                jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE),
            )
            total = total + jnp.sum(gamma_penalty)

            total = total + self._integrated_external_lens_loglike_draws_packed(
                cosmology,
                lambda_mean=lambda_mean,
                lambda_sigma=lambda_sigma,
                alpha_lambda=alpha_lambda,
                a_mean=a_mean,
                a_sigma=a_sigma,
                gamma_overrides=gamma_overrides,
            )

            gamma_prior_sigma = packed["gamma_prior_sigma"]
            gamma_prior_mean = packed["gamma_prior_mean"]
            has_gamma = packed["has_gamma"] > 0.5
            mask = has_gamma & jnp.isfinite(gamma_overrides) & (gamma_prior_sigma > 0.0)
            diff = (gamma_overrides - gamma_prior_mean) / jnp.maximum(
                gamma_prior_sigma, _EPS
            )
            gamma_prior_term = jnp.where(mask, -0.5 * diff**2, 0.0)
            total = total + jnp.sum(gamma_prior_term)
            return total

        for name, lens in self.lens_data.items():
            # Compute lens-specific lambda location
            lambda_loc = lambda_mean + alpha_lambda * jnp.asarray(
                lens.lambda_scaling, dtype=_DTYPE
            )
            gamma_key = self._gamma_param_map.get(name)
            gamma_override = kwargs.get(gamma_key, None) if gamma_key else None
            if gamma_override is not None:
                total = total + TDCOSMOLikelihood._bounds_penalty(  # noqa: SLF001
                    jnp.asarray(gamma_override, dtype=_DTYPE),
                    jnp.array(self.gamma_pl_bounds[0], dtype=_DTYPE),
                    jnp.array(self.gamma_pl_bounds[1], dtype=_DTYPE),
                )

            # Compute D_s/D_ds ratio (independent of H0!)
            ds_dds = self._distance_ratio(cosmology, lens.z_lens, lens.z_source)

            # Add lens contribution
            loglike = self._integrated_external_lens_loglike_draws(
                lens,
                ds_dds,
                lambda_loc=lambda_loc,
                lambda_sigma=lambda_sigma,
                a_mean=a_mean,
                a_sigma=a_sigma,
                gamma_pl=gamma_override,
            )
            total = total + loglike
            if (
                gamma_override is not None
                and lens.gamma_pl_prior_mean is not None
                and lens.gamma_pl_prior_sigma is not None
                and lens.gamma_pl_prior_sigma > 0
            ):
                diff = (
                    jnp.asarray(gamma_override, dtype=_DTYPE)
                    - jnp.array(lens.gamma_pl_prior_mean, dtype=_DTYPE)
                ) / jnp.array(lens.gamma_pl_prior_sigma, dtype=_DTYPE)
                total = total - 0.5 * diff**2

        return total

    def __call__(self, **params) -> float:
        """Callable interface for MCMC sampling."""
        if self.cosmology_class is None:
            raise ValueError("cosmology_class not set")

        if self._jitted_call is None:
            self._build_jitted_call(params)

        param_values = [params.get(name, 0.0) for name in self._jit_param_names]
        return self._jitted_call(*param_values)

    def _get_cosmology_param_names(self) -> set:
        """Get parameter names from cosmology class."""
        if hasattr(self, "_cached_cosmo_param_names"):
            return self._cached_cosmo_param_names

        if hasattr(self.cosmology_class, "get_parameters"):
            params = self.cosmology_class.get_parameters()
            names = {p.name for p in params}
        else:
            import inspect

            sig = inspect.signature(self.cosmology_class.__init__)
            names = {p for p in sig.parameters.keys() if p != "self" and p != "kwargs"}

        self._cached_cosmo_param_names = names
        return names

    def _build_jitted_call(self, sample_params: dict) -> None:
        """Build JIT-compiled likelihood function."""
        cosmo_param_names = self._get_cosmology_param_names()
        cosmo_names = [k for k in sample_params.keys() if k in cosmo_param_names]
        nuisance_names = [k for k in sample_params.keys() if k not in cosmo_param_names]
        all_param_names = cosmo_names + nuisance_names

        self._jit_param_names = all_param_names
        cosmo_class = self.cosmology_class
        n_cosmo = len(cosmo_names)

        def _impl(*args):
            cosmo_vals = args[:n_cosmo]
            nuisance_vals = args[n_cosmo:]
            cosmo_dict = {name: val for name, val in zip(cosmo_names, cosmo_vals)}
            cosmology = cosmo_class(**cosmo_dict)
            nuisance_dict = {
                name: val for name, val in zip(nuisance_names, nuisance_vals)
            }
            return self.log_likelihood(cosmology, **nuisance_dict)

        self._jitted_call = jax.jit(_impl)

    @property
    def nuisance_parameters(self) -> list:
        """Return unique external-lens nuisance parameters (gamma_pl per lens when present)."""
        from ...parameters import Parameter

        if not self.gamma_pl_sampling:
            return NuisanceList()
        params: list = []
        for lname, lens in self.lens_data.items():
            if lens.gamma_pl_params is None or lens.ani_scaling_2d is None:
                continue
            params.append(
                Parameter(
                    name=self._gamma_param_map[lname],
                    value=float(
                        lens.gamma_pl_prior_mean
                        if lens.gamma_pl_prior_mean is not None
                        else 2.0
                    ),
                    free=True,
                    prior={
                        "dist": "uniform",
                        "min": self.gamma_pl_bounds[0],
                        "max": self.gamma_pl_bounds[1],
                    },
                    latex_label=rf"\gamma_{{\rm pl}}^{{\rm {lname}}}",
                    description=f"Power-law slope for external lens {lname}",
                )
            )
        return NuisanceList(params)

    def __repr__(self) -> str:
        cosmo_name = self.cosmology_class.__name__ if self.cosmology_class else "None"
        return (
            f"ExternalLensLikelihood({len(self.lens_data)} {self.sample_type} lenses, "
            f"cosmology_class={cosmo_name})"
        )


# =============================================================================
# Combined Hierarchical TDCOSMO Likelihood
# =============================================================================


class HierarchicalTDCOSMO(Likelihood):
    """
    Combined hierarchical TDCOSMO likelihood with external lens samples.

    This is the full TDCOSMO 2025 analysis combining:
    - Time-delay lenses (TDCOSMO): Constrain H0 × λ_int through Ddt
    - External lenses (SLACS/SL2S): Constrain λ_int through kinematics

    The combination breaks the H0-λ_int degeneracy!

    Parameters
    ----------
    cosmology_class : type
        Cosmology model class (LCDM, wCDM, etc.)
    tdcosmo_path : str, optional
        Path to TDCOSMO processed lens data
    include_slacs : bool
        Include SLACS external lenses (default: True)
    include_sl2s : bool
        Include SL2S external lenses (default: True)
    slacs_path : str, optional
        Path to SLACS processed data
    sl2s_path : str, optional
        Path to SL2S processed data

    Examples
    --------
    >>> from hicosmo.models import LCDM
    >>> from hicosmo.likelihoods import HierarchicalTDCOSMO
    >>> tdcosmo = HierarchicalTDCOSMO(
    ...     cosmology_class=LCDM,
    ...     include_slacs=True,
    ...     include_sl2s=True,
    ... )
    >>> log_L = tdcosmo(H0=70, Omega_m=0.3, lambda_int_mean=1.0, a_ani_mean=1.0)
    """

    def __init__(
        self,
        cosmology_class: Optional[type] = None,
        tdcosmo_path: Optional[str] = None,
        include_slacs: bool = True,
        include_sl2s: bool = True,
        slacs_path: Optional[str] = None,
        sl2s_path: Optional[str] = None,
        slacs_names: Optional[Iterable[str]] = None,
        sl2s_names: Optional[Iterable[str]] = None,
        tdcosmo_names: Optional[Iterable[str]] = None,
        lambda_bounds: Tuple[float, float] = (0.5, 1.5),
        anisotropy_bounds: Tuple[float, float] = (0.1, 5.0),
        use_tdcosmo2025: Optional[bool] = None,
        anisotropy_model: str = "const",
        anisotropy_parameterization: str = "beta",
        kin_axi_correction: bool = False,
        remove_gamma_pl_prior: bool = True,
        num_distribution_draws: int = 200,
        max_kappa_nodes: Optional[int] = None,
        max_vel_disp_nodes: Optional[int] = None,
        use_spline: bool = True,
        use_quality_data_only: bool = True,
        use_selected_lens_only: bool = True,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.cosmology_class = cosmology_class
        self.lambda_bounds = lambda_bounds
        self.anisotropy_bounds = anisotropy_bounds

        if use_tdcosmo2025 is None:
            if tdcosmo_path is not None:
                use_tdcosmo2025 = False
            else:
                # Go up 3 levels: lensing/ -> likelihoods/ -> hicosmo/ -> project root
                tdcosmo2025_root = (
                    Path(__file__).resolve().parents[3]
                    / "TDCOSMO2025_public"
                    / "TDCOSMO_sample"
                )
                use_tdcosmo2025 = tdcosmo2025_root.exists()

        # Initialize TDCOSMO time-delay likelihood
        self.tdcosmo = TDCOSMOLikelihood(
            cosmology_class=cosmology_class,
            lens_names=tdcosmo_names,
            data_path=tdcosmo_path,
            lambda_bounds=lambda_bounds,
            anisotropy_bounds=anisotropy_bounds,
            use_tdcosmo2025=bool(use_tdcosmo2025),
            anisotropy_model=anisotropy_model,
            anisotropy_parameterization=anisotropy_parameterization,
            kin_axi_correction=kin_axi_correction,
            num_distribution_draws=num_distribution_draws,
            max_kappa_nodes=max_kappa_nodes,
            max_vel_disp_nodes=max_vel_disp_nodes,
            use_spline=use_spline,
        )

        # Initialize external lens likelihoods
        self.external_likelihoods: list = []

        if include_slacs:
            try:
                slacs = ExternalLensLikelihood(
                    cosmology_class=cosmology_class,
                    data_path=slacs_path,
                    sample_type="SLACS",
                    lens_names=slacs_names,
                    lambda_bounds=lambda_bounds,
                    anisotropy_bounds=anisotropy_bounds,
                    anisotropy_parameterization=anisotropy_parameterization,
                    kin_axi_correction=kin_axi_correction,
                    remove_gamma_pl_prior=remove_gamma_pl_prior,
                    num_distribution_draws=num_distribution_draws,
                    max_vel_disp_nodes=max_vel_disp_nodes,
                    use_spline=use_spline,
                    use_quality_data_only=use_quality_data_only,
                    use_selected_lens_only=use_selected_lens_only,
                )
                self.external_likelihoods.append(slacs)
            except FileNotFoundError as e:
                logger.warning(f"SLACS data not found: {e}")

        if include_sl2s:
            try:
                sl2s = ExternalLensLikelihood(
                    cosmology_class=cosmology_class,
                    data_path=sl2s_path,
                    sample_type="SL2S",
                    lens_names=sl2s_names,
                    lambda_bounds=lambda_bounds,
                    anisotropy_bounds=anisotropy_bounds,
                    anisotropy_parameterization=anisotropy_parameterization,
                    kin_axi_correction=kin_axi_correction,
                    remove_gamma_pl_prior=remove_gamma_pl_prior,
                    num_distribution_draws=num_distribution_draws,
                    max_vel_disp_nodes=max_vel_disp_nodes,
                    use_spline=use_spline,
                    use_quality_data_only=use_quality_data_only,
                    use_selected_lens_only=use_selected_lens_only,
                )
                self.external_likelihoods.append(sl2s)
            except FileNotFoundError as e:
                logger.warning(f"SL2S data not found: {e}")

        # Count total lenses
        n_tdcosmo = len(self.tdcosmo.lens_names)
        n_external = sum(len(ext.lens_names) for ext in self.external_likelihoods)

        super().__init__(
            name=name or "hierarchical_tdcosmo",
            data_path=str(self.tdcosmo.data_path),
            **kwargs,
        )
        self.initialize()

        logger.info("\nHierarchical TDCOSMO initialized:")
        logger.info(f"  Time-delay lenses: {n_tdcosmo}")
        logger.info(f"  External lenses: {n_external}")
        logger.info(f"  Total: {n_tdcosmo + n_external} lenses")
        logger.info("  This breaks the H0-λ_int degeneracy!")

        self._jitted_call = None
        self._jit_param_names = None

    def _default_dataset_name(self) -> str:
        return "hierarchical_tdcosmo"

    def _load_data(self) -> None:
        return

    def _setup_covariance(self) -> None:
        return

    def get_requirements(self) -> Dict[str, Any]:
        return {}

    def theory(self, cosmology, **kwargs):
        raise NotImplementedError

    def log_likelihood(self, cosmology, **kwargs) -> float:
        """Compute combined log-likelihood from all lens samples."""
        # TDCOSMO time-delay contribution
        total = self.tdcosmo.log_likelihood(cosmology, **kwargs)

        # External lens contributions
        for ext in self.external_likelihoods:
            total = total + ext.log_likelihood(cosmology, **kwargs)

        return total

    def __call__(self, **params) -> float:
        """Callable interface for MCMC sampling."""
        if self.cosmology_class is None:
            raise ValueError("cosmology_class not set")

        if self._jitted_call is None:
            self._build_jitted_call(params)

        param_values = [params.get(name, 0.0) for name in self._jit_param_names]
        return self._jitted_call(*param_values)

    def _get_cosmology_param_names(self) -> set:
        """Get parameter names from cosmology class."""
        return self.tdcosmo._get_cosmology_param_names()

    def _build_jitted_call(self, sample_params: dict) -> None:
        """Build JIT-compiled likelihood function."""
        cosmo_param_names = self._get_cosmology_param_names()
        cosmo_names = [k for k in sample_params.keys() if k in cosmo_param_names]
        nuisance_names = [k for k in sample_params.keys() if k not in cosmo_param_names]
        all_param_names = cosmo_names + nuisance_names

        self._jit_param_names = all_param_names
        cosmo_class = self.cosmology_class
        n_cosmo = len(cosmo_names)

        # Get references to likelihoods
        tdcosmo_likelihood = self.tdcosmo
        external_likelihoods = self.external_likelihoods

        def _impl(*args):
            cosmo_vals = args[:n_cosmo]
            nuisance_vals = args[n_cosmo:]
            cosmo_dict = {name: val for name, val in zip(cosmo_names, cosmo_vals)}
            cosmology = cosmo_class(**cosmo_dict)
            nuisance_dict = {
                name: val for name, val in zip(nuisance_names, nuisance_vals)
            }

            # Combined likelihood
            total = tdcosmo_likelihood.log_likelihood(cosmology, **nuisance_dict)
            for ext in external_likelihoods:
                total = total + ext.log_likelihood(cosmology, **nuisance_dict)
            return total

        self._jitted_call = jax.jit(_impl)

    @property
    def nuisance_parameters(self) -> list:
        """Return nuisance parameters (shared hyperparameters + external gamma_pl when present)."""
        params = list(self.tdcosmo.nuisance_parameters())
        for ext in self.external_likelihoods:
            params.extend(ext.nuisance_parameters())
        return NuisanceList(params)

    def __repr__(self) -> str:
        cosmo_name = self.cosmology_class.__name__ if self.cosmology_class else "None"
        n_td = len(self.tdcosmo.lens_names)
        n_ext = sum(len(ext.lens_names) for ext in self.external_likelihoods)
        return (
            f"HierarchicalTDCOSMO({n_td} TD + {n_ext} external lenses, "
            f"cosmology_class={cosmo_name})"
        )
