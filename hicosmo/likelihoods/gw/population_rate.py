"""Pure JAX helpers for GW population modeling."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Any, Protocol, Optional, Union

import jax.numpy as jnp
from jax import config as jax_config
from jax.scipy.stats import norm

jax_config.update("jax_enable_x64", True)
from ...utils.jax_tools import trapezoid


ArrayLike = Union[float, np.ndarray, jnp.ndarray]


class CosmologyProtocol(Protocol):
    params: dict

    def luminosity_distance(self, z: ArrayLike) -> jnp.ndarray: ...
    def dVc_dz(self, z: ArrayLike) -> jnp.ndarray: ...
    def ddL_dz(self, z: ArrayLike) -> jnp.ndarray: ...
    def dl_to_z(self, d_l: ArrayLike) -> jnp.ndarray: ...


@dataclass
class CosmologyAdapter:
    """Thin wrapper enforcing the protocol for any hicosmo cosmology."""

    cosmology: CosmologyProtocol
    zmin: float = 1e-4
    zmax: float = 10.0
    n_grid: int = 6000

    def __post_init__(self):
        self._build_lookup()

    def _build_lookup(self):
        grid = jnp.geomspace(self.zmin, self.zmax, num=max(self.n_grid - 1, 2))
        z_grid = jnp.concatenate((jnp.asarray([self.zmin * 0.1]), grid))
        dl_grid = jnp.asarray(
            self.cosmology.luminosity_distance(z_grid), dtype=jnp.float64
        )
        dl_grid = jnp.clip(dl_grid, 1e-12, None)
        self._log10_dl = jnp.log10(dl_grid)
        self._log10_z = jnp.log10(jnp.clip(z_grid, 1e-12, None))
        self._min_dl = dl_grid[0]
        self._max_dl = dl_grid[-1]

    def luminosity_distance(self, z):
        return jnp.asarray(self.cosmology.luminosity_distance(z))

    def dVc_dz(self, z):
        return jnp.asarray(self.cosmology.dVc_dz(z))

    def ddL_dz(self, z):
        return jnp.asarray(self.cosmology.ddL_dz(z))

    def dl_to_z(self, d_l):
        arr = jnp.asarray(d_l, dtype=jnp.float64)
        flat = arr.reshape(-1)
        safe = jnp.clip(flat, self._min_dl, self._max_dl)
        log_dl = jnp.log10(safe)
        log_z = jnp.interp(log_dl, self._log10_dl, self._log10_z)
        z = jnp.power(10.0, log_z)
        return z.reshape(arr.shape)


def detector2source(mass1_det, mass2_det, d_l, cosmology: CosmologyAdapter):
    z = cosmology.dl_to_z(d_l)
    inv = 1.0 / (1.0 + z)
    return mass1_det * inv, mass2_det * inv, z


def detector2source_jacobian(z, cosmology: CosmologyAdapter):
    return jnp.abs(jnp.power(1.0 + z, 2.0) * cosmology.ddL_dz(z))


def detector2source_jacobian_q(z, cosmology: CosmologyAdapter):
    return jnp.abs((1.0 + z) * cosmology.ddL_dz(z))


def detector2source_jacobian_single_mass(z, cosmology: CosmologyAdapter):
    return jnp.abs((1.0 + z) * cosmology.ddL_dz(z))


def _highpass_filter_jnp(mass, mmin, delta_m):
    mass = jnp.asarray(mass)
    delta_m = float(delta_m)
    if delta_m <= 0.0:
        return jnp.ones_like(mass)
    mprime = mass - mmin
    window_region = (mass > mmin) & (mass < mmin + delta_m)
    safe_mprime = jnp.where(window_region, mprime, 1.0)
    term = delta_m / safe_mprime + delta_m / (safe_mprime - delta_m)
    effe_prime = jnp.exp(
        jnp.nan_to_num(term, nan=jnp.inf, posinf=jnp.inf, neginf=-jnp.inf)
    )
    window = 1.0 / (effe_prime + 1.0)
    window = jnp.where(mass <= mmin, 0.0, window)
    window = jnp.where(mass >= mmin + delta_m, 1.0, window)
    return window


def _highpass_filter_np(mass, mmin, delta_m):
    mass = np.asarray(mass, dtype=np.float64)
    if delta_m <= 0.0:
        return np.ones_like(mass)
    mprime = mass - mmin
    window = np.ones_like(mass)
    select_window = (mass > mmin) & (mass < mmin + delta_m)
    safe = np.where(select_window, mprime, 1.0)
    term = delta_m / safe + delta_m / (safe - delta_m)
    effe_prime = np.exp(np.nan_to_num(term, nan=np.inf, posinf=np.inf, neginf=-np.inf))
    window = 1.0 / (effe_prime + 1.0)
    window[mass <= mmin] = 0.0
    window[mass >= mmin + delta_m] = 1.0
    return window


def _logsumexp_pair(a, b):
    return jnp.logaddexp(a, b)


def _safe_float(value: Any) -> Any:
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


class _PowerLawDistribution:
    def __init__(self, mmin: float, mmax: float, exponent: float):
        self.mmin = float(mmin)
        self.mmax = float(mmax)
        self.alpha = float(exponent)
        if abs(self.alpha + 1.0) < 1e-9:
            self._log_norm = jnp.log(self.mmax / self.mmin)
        else:
            norm_val = (
                self.mmax ** (self.alpha + 1.0) - self.mmin ** (self.alpha + 1.0)
            ) / (self.alpha + 1.0)
            self._log_norm = jnp.log(norm_val)

    def _in_support(self, m):
        return (m >= self.mmin) & (m <= self.mmax)

    def log_pdf_jnp(self, m):
        m = jnp.asarray(m)
        log_pdf = self.alpha * jnp.log(m) - self._log_norm
        return jnp.where(self._in_support(m), log_pdf, -jnp.inf)

    def log_pdf_np(self, m):
        arr = np.asarray(m, dtype=np.float64)
        log_pdf = self.alpha * np.log(arr) - float(self._log_norm)
        mask = (arr >= self.mmin) & (arr <= self.mmax)
        out = np.full_like(arr, -np.inf, dtype=np.float64)
        out[mask] = log_pdf[mask]
        return out

    def cdf_jnp(self, m):
        m = jnp.asarray(m)
        if abs(self.alpha + 1.0) < 1e-9:
            cdf = jnp.log(m / self.mmin) / jnp.log(self.mmax / self.mmin)
        else:
            num = m ** (self.alpha + 1.0) - self.mmin ** (self.alpha + 1.0)
            den = self.mmax ** (self.alpha + 1.0) - self.mmin ** (self.alpha + 1.0)
            cdf = num / den
        cdf = jnp.clip(cdf, 0.0, 1.0)
        cdf = jnp.where(m <= self.mmin, 0.0, cdf)
        cdf = jnp.where(m >= self.mmax, 1.0, cdf)
        return cdf


class _TruncatedGaussian:
    def __init__(self, mu: float, sigma: float, mmin: float, mmax: float):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.mmin = float(mmin)
        self.mmax = float(mmax)
        a = (self.mmin - self.mu) / self.sigma
        b = (self.mmax - self.mu) / self.sigma
        self._log_norm = jnp.log(
            jnp.sqrt(2.0 * jnp.pi) * self.sigma * (norm.cdf(b) - norm.cdf(a))
        )

    def _in_support(self, m):
        return (m >= self.mmin) & (m <= self.mmax)

    def log_pdf_jnp(self, m):
        m = jnp.asarray(m)
        log_pdf = -0.5 * ((m - self.mu) / self.sigma) ** 2 - self._log_norm
        return jnp.where(self._in_support(m), log_pdf, -jnp.inf)

    def cdf_jnp(self, m):
        m = jnp.asarray(m)
        a = (self.mmin - self.mu) / self.sigma
        b = (self.mmax - self.mu) / self.sigma
        num = norm.cdf((m - self.mu) / self.sigma) - norm.cdf(a)
        den = norm.cdf(b) - norm.cdf(a)
        cdf = num / den
        cdf = jnp.clip(cdf, 0.0, 1.0)
        cdf = jnp.where(m <= self.mmin, 0.0, cdf)
        cdf = jnp.where(m >= self.mmax, 1.0, cdf)
        return cdf


class _PowerLawGaussianDistribution:
    def __init__(
        self,
        alpha: float,
        mmin: float,
        mmax: float,
        mu_g: float,
        sigma_g: float,
        lambda_peak: float,
    ):
        self.mmin = float(mmin)
        self.mmax = float(mmax)
        self.lambda_peak = float(lambda_peak)
        self.powerlaw = _PowerLawDistribution(mmin, mmax, -alpha)
        max_gauss = mu_g + 5.0 * sigma_g
        self.gaussian = _TruncatedGaussian(mu_g, sigma_g, mmin, max_gauss)

    def log_pdf_jnp(self, m):
        lam = jnp.clip(self.lambda_peak, 1e-12, 1 - 1e-12)
        log_pl = self.powerlaw.log_pdf_jnp(m) + jnp.log1p(-lam)
        log_gauss = self.gaussian.log_pdf_jnp(m) + jnp.log(lam)
        return _logsumexp_pair(log_pl, log_gauss)

    def cdf_jnp(self, m):
        lam = jnp.clip(self.lambda_peak, 1e-12, 1 - 1e-12)
        return (1.0 - lam) * self.powerlaw.cdf_jnp(m) + lam * self.gaussian.cdf_jnp(m)


class _LowpassSmoothedDistribution:
    def __init__(self, base_dist, delta_m: float, num_grid: int = 1000):
        self.base = base_dist
        self.delta_m = float(max(delta_m, 0.0))
        self.bottom = float(base_dist.mmin)
        self.norm = jnp.asarray(1.0, dtype=jnp.float64)
        self.integral_now = jnp.asarray(0.0, dtype=jnp.float64)
        self.base_cdf_edge = jnp.asarray(0.0, dtype=jnp.float64)
        self.midpoints: Optional[jnp.ndarray] = None
        self.cdf_numeric: Optional[jnp.ndarray] = None

        if self.delta_m > 0.0:
            grid = jnp.linspace(
                self.bottom,
                self.bottom + self.delta_m,
                num_grid,
                dtype=jnp.float64,
            )
            base_pdf = jnp.exp(self.base.log_pdf_jnp(grid))
            window = _highpass_filter_jnp(grid, self.bottom, self.delta_m)
            integral_before = trapezoid(base_pdf, grid)
            integral_now = trapezoid(base_pdf * window, grid)
            self.norm = jnp.asarray(
                1.0 - integral_before + integral_now, dtype=jnp.float64
            )
            self.integral_now = jnp.asarray(integral_now, dtype=jnp.float64)
            self.base_cdf_edge = self.base.cdf_jnp(self.bottom + self.delta_m)

            midpoints = 0.5 * (grid[:-1] + grid[1:])
            smooth_pdf = jnp.exp(
                self.base.log_pdf_jnp(midpoints)
                + jnp.log(_highpass_filter_jnp(midpoints, self.bottom, self.delta_m))
                - jnp.log(self.norm)
            )
            cdf_numeric = jnp.cumsum(smooth_pdf * (grid[1:] - grid[:-1]))
            self.midpoints = midpoints
            self.cdf_numeric = cdf_numeric

    def log_pdf_jnp(self, m):
        if self.delta_m <= 0.0:
            return self.base.log_pdf_jnp(m)
        window = _highpass_filter_jnp(m, self.bottom, self.delta_m)
        return self.base.log_pdf_jnp(m) + jnp.log(window) - jnp.log(self.norm)

    def log_cdf_jnp(self, m):
        return jnp.log(self.cdf_jnp(m))

    def cdf_jnp(self, m):
        if self.delta_m <= 0.0:
            return self.base.cdf_jnp(m)
        if self.midpoints is None or self.cdf_numeric is None:
            return self.base.cdf_jnp(m)

        m = jnp.asarray(m, dtype=jnp.float64)
        edge = self.bottom + self.delta_m
        interp = jnp.interp(
            m,
            self.midpoints,
            self.cdf_numeric,
            left=0.0,
            right=self.cdf_numeric[-1],
        )
        tail = (
            self.integral_now + self.base.cdf_jnp(m) - self.base_cdf_edge
        ) / self.norm
        cdf_val = jnp.where(
            m <= self.bottom,
            0.0,
            jnp.where(m <= edge, interp, tail),
        )
        return jnp.clip(cdf_val, 0.0, 1.0)


class MassPriorEvaluator:
    """Pure JAX implementation of icarogw's PowerLawPeak + low-pass pairing."""

    population_parameters = [
        "alpha",
        "beta",
        "mmin",
        "mmax",
        "delta_m",
        "mu_g",
        "sigma_g",
        "lambda_peak",
    ]

    def __init__(self):
        self.params: dict[str, float] = {}
        self._m1_lowpass: _LowpassSmoothedDistribution | None = None
        self._m2_lowpass: _LowpassSmoothedDistribution | None = None

    def update(self, **params):
        missing = [k for k in self.population_parameters if k not in params]
        if missing:
            raise ValueError(f"Missing mass-prior parameters: {missing}")
        self.params = {k: float(params[k]) for k in self.population_parameters}
        alpha = self.params["alpha"]
        beta = self.params["beta"]
        mmin = self.params["mmin"]
        mmax = self.params["mmax"]
        delta_m = self.params["delta_m"]
        mu_g = self.params["mu_g"]
        sigma_g = self.params["sigma_g"]
        lambda_peak = self.params["lambda_peak"]

        m1_dist = _PowerLawGaussianDistribution(
            alpha, mmin, mmax, mu_g, sigma_g, lambda_peak
        )
        m2_dist = _PowerLawDistribution(mmin, mmax, beta)
        self._m1_lowpass = _LowpassSmoothedDistribution(m1_dist, delta_m)
        self._m2_lowpass = _LowpassSmoothedDistribution(m2_dist, delta_m)

    def log_prob(self, m1_source, m2_source):
        if self._m1_lowpass is None or self._m2_lowpass is None:
            raise RuntimeError(
                "MassPriorEvaluator.update() must be called before log_prob()."
            )
        m1 = jnp.asarray(m1_source)
        m2 = jnp.asarray(m2_source)
        log_p1 = self._m1_lowpass.log_pdf_jnp(m1)
        log_p2 = self._m2_lowpass.log_pdf_jnp(m2)
        log_cdf = self._m2_lowpass.log_cdf_jnp(m1)
        log_prob = log_p1 + log_p2 - log_cdf
        mmin = self.params["mmin"]
        mmax = self.params["mmax"]
        valid = (m1 >= mmin) & (m1 <= mmax) & (m2 >= mmin) & (m2 <= mmax) & (m2 <= m1)
        return jnp.where(valid, log_prob, -jnp.inf)


class MadauRateEvaluator:
    """Exact JAX translation of icarogw's md_rate."""

    population_parameters = ["gamma", "kappa", "zp"]

    def __init__(self):
        self.params: dict[str, float] = {}

    def update(self, **params):
        missing = [k for k in self.population_parameters if k not in params]
        if missing:
            raise ValueError(f"Missing rate parameters: {missing}")
        self.params = {k: _safe_float(params[k]) for k in self.population_parameters}

    def log_rate(self, z):
        if not self.params:
            raise RuntimeError(
                "MadauRateEvaluator.update() must be called before log_rate()."
            )
        z = jnp.asarray(z)
        gamma = self.params["gamma"]
        kappa = self.params["kappa"]
        zp = self.params["zp"]
        log_norm = jnp.log1p((1.0 + zp) ** (-gamma - kappa))
        log_num = gamma * jnp.log1p(z)
        ratio = (1.0 + z) / (1.0 + zp)
        log_den = jnp.log1p(ratio ** (gamma + kappa))
        return log_norm + log_num - log_den


class GWPopulationRateModel:
    def __init__(
        self,
        mass_prior=None,
        rate_evolution=None,
        scale_free=True,
        zmax=10.0,
        grid_size=6000,
    ):
        self.mass_prior = mass_prior or MassPriorEvaluator()
        self.rate_evolution = rate_evolution or MadauRateEvaluator()
        self.scale_free = scale_free
        self.zmax = zmax
        self.grid_size = grid_size
        self.population_parameters = [
            "alpha",
            "beta",
            "mmin",
            "mmax",
            "delta_m",
            "mu_g",
            "sigma_g",
            "lambda_peak",
            "gamma",
            "kappa",
            "zp",
        ]
        if not scale_free:
            self.population_parameters.append("R0")
        self.R0 = 1.0
        self._population_only_parameters = (
            list(self.mass_prior.population_parameters)
            + list(self.rate_evolution.population_parameters)
            + ([] if scale_free else ["R0"])
        )
        self._cached_population: dict[str, Any] = {}
        self.cosmology: Optional[CosmologyAdapter] = None

    def update(self, cosmology, **params):
        """Update both cosmology and population parameters."""
        self.set_cosmology(cosmology)
        self.update_population(**params)

    def set_cosmology(self, cosmology) -> None:
        if cosmology is None:
            raise ValueError("cosmology must be provided")
        required = ["luminosity_distance", "dVc_dz", "ddL_dz", "dl_to_z"]
        missing = [name for name in required if not hasattr(cosmology, name)]
        if missing:
            raise ValueError(
                "Cosmology is missing required methods: " + ", ".join(missing)
            )
        self.cosmology = cosmology

    def update_population(self, **params) -> None:
        """Update mass/rate population parameters (no cosmology update)."""
        missing = [p for p in self._population_only_parameters if p not in params]
        if missing:
            raise ValueError(f"Missing population parameters: {missing}")
        self.mass_prior.update(**params)
        self.rate_evolution.update(**params)
        if not self.scale_free:
            self.R0 = _safe_float(params.get("R0", 1.0))
        self._cached_population = {
            k: params[k] for k in self._population_only_parameters
        }

    @property
    def population_only_parameters(self) -> list[str]:
        return list(self._population_only_parameters)

    def log_rate_PE(self, prior, luminosity_distance, mass_1, mass_2):
        if self.cosmology is None:
            raise RuntimeError("Cosmology not set. Call set_cosmology() first.")
        prior = jnp.asarray(prior, dtype=jnp.float64)
        d_l = jnp.asarray(luminosity_distance, dtype=jnp.float64)
        mass_1 = jnp.asarray(mass_1, dtype=jnp.float64)
        mass_2 = jnp.asarray(mass_2, dtype=jnp.float64)

        m1_src, m2_src, z = detector2source(mass_1, mass_2, d_l, self.cosmology)
        log_mass = self.mass_prior.log_prob(m1_src, m2_src)
        log_rate = self.rate_evolution.log_rate(z)
        log_dVc = jnp.log(self.cosmology.dVc_dz(z))
        log_jac = jnp.log(detector2source_jacobian(z, self.cosmology))
        log_weights = (
            log_mass + log_rate + log_dVc - jnp.log1p(z) - jnp.log(prior) - log_jac
        )
        if not self.scale_free:
            log_weights = log_weights + jnp.log(self.R0)
        return log_weights

    def log_rate_injections(self, prior, luminosity_distance, mass_1, mass_2):
        return self.log_rate_PE(prior, luminosity_distance, mass_1, mass_2)

    def expected_detections(
        self, prior, luminosity_distance, mass_1, mass_2, ntotal, Tobs
    ):
        log_w = self.log_rate_injections(prior, luminosity_distance, mass_1, mass_2)
        pseudo_rate = jnp.exp(jnp.logaddexp.reduce(log_w)) / ntotal
        return pseudo_rate * Tobs


__all__ = [
    "CosmologyAdapter",
    "detector2source",
    "detector2source_jacobian",
    "detector2source_jacobian_q",
    "detector2source_jacobian_single_mass",
    "MassPriorEvaluator",
    "MadauRateEvaluator",
    "GWPopulationRateModel",
]
