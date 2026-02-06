"""Gravitational-wave population models shared across likelihood modules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import jax.numpy as jnp
from jax import jit, vmap
from jax.scipy.stats import norm

__all__ = [
    "MassPrior",
    "PowerLawMass",
    "BrokenPowerLawMass",
    "PowerLawPeak",
    "BNSMassPrior",
    "RateEvolution",
    "MadauRate",
]


class MassPrior(ABC):
    """Abstract base class for binary mass distributions."""

    parameter_names: tuple[str, ...] = ()

    @abstractmethod
    def log_prob(self, m1: jnp.ndarray, m2: jnp.ndarray, **params: Any) -> jnp.ndarray:
        """Return log p(m1, m2 | theta)."""
        raise NotImplementedError


class PowerLawMass(MassPrior):
    """Factorized power-law distribution for (m1, m2)."""

    parameter_names = ("alpha", "beta", "mmin", "mmax")

    @staticmethod
    @jit
    def _log_norm_m1(alpha: float, mmin: float, mmax: float) -> float:
        # Use exponent = -alpha so that alpha > 0 corresponds to m^{-alpha}
        exp = -alpha
        is_special = jnp.abs(exp + 1.0) < 1e-6
        normal_norm = jnp.log((mmax ** (exp + 1) - mmin ** (exp + 1)) / (exp + 1))
        special_norm = jnp.log(jnp.log(mmax / mmin))
        return jnp.where(is_special, special_norm, normal_norm)

    @staticmethod
    @jit
    def _log_norm_m2(beta: float, mmin: float, m1: float) -> float:
        is_degenerate = jnp.abs(m1 - mmin) < 1e-9
        is_special_beta = jnp.abs(beta + 1.0) < 1e-6
        normal_norm = jnp.log((m1 ** (beta + 1) - mmin ** (beta + 1)) / (beta + 1))
        special_norm = jnp.log(jnp.log(m1 / mmin))
        norm = jnp.where(is_special_beta, special_norm, normal_norm)
        return jnp.where(is_degenerate, -1e10, norm)

    def log_prob(self, m1: jnp.ndarray, m2: jnp.ndarray, **params: Any) -> jnp.ndarray:
        alpha = params.get("alpha", -2.35)
        beta = params.get("beta", 1.0)
        mmin = params.get("mmin", 5.0)
        mmax = params.get("mmax", 100.0)

        m1 = jnp.atleast_1d(m1)
        m2 = jnp.atleast_1d(m2)

        log_p_m1_unnorm = -alpha * jnp.log(m1)
        log_norm_m1 = self._log_norm_m1(alpha, mmin, mmax)

        log_p_m2_unnorm = beta * jnp.log(m2)
        log_norm_m2 = vmap(lambda m1_val: self._log_norm_m2(beta, mmin, m1_val))(m1)

        in_range_m1 = (m1 >= mmin) & (m1 <= mmax)
        in_range_m2 = (m2 >= mmin) & (m2 <= m1)
        in_range = in_range_m1 & in_range_m2

        log_p = jnp.where(
            in_range,
            log_p_m1_unnorm - log_norm_m1 + log_p_m2_unnorm - log_norm_m2,
            -jnp.inf,
        )

        return log_p


class BrokenPowerLawMass(MassPrior):
    """Broken power-law distribution for primary mass with low-mass smoothing."""

    parameter_names = ("alpha_1", "alpha_2", "beta", "mmin", "mmax", "b", "delta_m")

    @staticmethod
    @jit
    def _log_norm_powerlaw(exp: float, mmin: float, mmax: float) -> float:
        eps = 1e-6
        is_special = jnp.abs(exp + 1.0) < eps
        normal_norm = jnp.log((mmax ** (exp + 1) - mmin ** (exp + 1)) / (exp + 1))
        special_norm = jnp.log(jnp.log(mmax / mmin))
        return jnp.where(is_special, special_norm, normal_norm)

    @staticmethod
    @jit
    def _smoothing_function(m: jnp.ndarray, mmin: float, delta_m: float) -> jnp.ndarray:
        safe_m = jnp.maximum(m, mmin + 1e-10)
        term1 = delta_m / (safe_m - mmin)
        term2 = delta_m / (safe_m - mmin - delta_m)
        log_smoothing = -jnp.logaddexp(0.0, term1 + term2)
        return jnp.where(m >= mmin, log_smoothing, -jnp.inf)

    def log_prob(self, m1: jnp.ndarray, m2: jnp.ndarray, **params: Any) -> jnp.ndarray:
        alpha1 = params.get("alpha_1", 3.0)
        alpha2 = params.get("alpha_2", 6.0)
        beta = params.get("beta", 1.0)
        mmin = params.get("mmin", 5.0)
        mmax = params.get("mmax", 100.0)
        b = params.get("b", 0.5)
        delta_m = params.get("delta_m", 4.0)

        m1 = jnp.atleast_1d(m1)
        m2 = jnp.atleast_1d(m2)

        m_break = mmin + b * (mmax - mmin)
        valid_break = (b >= 0.0) & (b <= 1.0) & (m_break > mmin) & (m_break < mmax)

        exp1 = -alpha1
        exp2 = -alpha2
        log_norm1 = self._log_norm_powerlaw(exp1, mmin, m_break)
        log_norm2 = self._log_norm_powerlaw(exp2, m_break, mmax)
        log_pdf1 = exp1 * jnp.log(m1) - log_norm1
        log_pdf2 = exp2 * jnp.log(m1) - log_norm2
        log_pdf1_break = exp1 * jnp.log(m_break) - log_norm1
        log_pdf2_break = exp2 * jnp.log(m_break) - log_norm2
        log_norm = jnp.log1p(jnp.exp(log_pdf1_break - log_pdf2_break))
        log_p_m1 = (
            jnp.logaddexp(
                log_pdf1,
                log_pdf2 + log_pdf1_break - log_pdf2_break,
            )
            - log_norm
        )

        log_p_m2_unnorm = beta * jnp.log(m2)
        log_norm_m2 = vmap(
            lambda m1_val: PowerLawMass._log_norm_m2(beta, mmin, m1_val)
        )(m1)
        log_p_m2_given_m1 = log_p_m2_unnorm - log_norm_m2

        log_S_m1 = self._smoothing_function(m1, mmin, delta_m)
        log_S_m2 = self._smoothing_function(m2, mmin, delta_m)

        in_range_m1 = (m1 >= mmin) & (m1 <= mmax)
        in_range_m2 = (m2 >= mmin) & (m2 <= m1)
        in_range = in_range_m1 & in_range_m2 & valid_break

        log_p = jnp.where(
            in_range,
            log_p_m1 + log_p_m2_given_m1 + log_S_m1 + log_S_m2,
            -jnp.inf,
        )
        return log_p


class PowerLawPeak(MassPrior):
    """Power-law + Gaussian peak mass distribution with smoothing."""

    parameter_names = (
        "alpha",
        "beta",
        "mmin",
        "mmax",
        "delta_m",
        "mu_g",
        "sigma_g",
        "lambda_peak",
    )

    @staticmethod
    @jit
    def _powerlaw_component(
        m1: jnp.ndarray, alpha: float, mmin: float, mmax: float
    ) -> jnp.ndarray:
        log_p_unnorm = -alpha * jnp.log(m1)
        is_unity = jnp.abs(alpha - 1.0) < 1e-6
        norm_regular = jnp.log(
            (mmax ** (1 - alpha) - mmin ** (1 - alpha)) / (1 - alpha)
        )
        norm_unity = jnp.log(jnp.log(mmax / mmin))
        log_norm = jnp.where(is_unity, norm_unity, norm_regular)
        return log_p_unnorm - log_norm

    @staticmethod
    @jit
    def _gaussian_component(
        m1: jnp.ndarray, mu_g: float, sigma_g: float, mmin: float, mmax: float
    ) -> jnp.ndarray:
        log_p_unnorm = -0.5 * ((m1 - mu_g) / sigma_g) ** 2
        cdf_max = norm.cdf((mmax - mu_g) / sigma_g)
        cdf_min = norm.cdf((mmin - mu_g) / sigma_g)
        log_norm = jnp.log(jnp.sqrt(2 * jnp.pi) * sigma_g * (cdf_max - cdf_min))
        return log_p_unnorm - log_norm

    @staticmethod
    @jit
    def _smoothing_function(m: jnp.ndarray, mmin: float, delta_m: float) -> jnp.ndarray:
        safe_m = jnp.maximum(m, mmin + 1e-10)
        term1 = delta_m / (safe_m - mmin)
        term2 = delta_m / (safe_m - mmin - delta_m)
        log_smoothing = -jnp.logaddexp(0.0, term1 + term2)
        return jnp.where(m >= mmin, log_smoothing, -jnp.inf)

    def log_prob(self, m1: jnp.ndarray, m2: jnp.ndarray, **params: Any) -> jnp.ndarray:
        alpha = params.get("alpha", 3.78)
        beta = params.get("beta", 0.81)
        mmin = params.get("mmin", 4.98)
        mmax = params.get("mmax", 112.5)
        delta_m = params.get("delta_m", 4.8)
        mu_g = params.get("mu_g", 32.27)
        sigma_g = params.get("sigma_g", 3.88)
        lambda_peak = params.get("lambda_peak", 0.03)

        m1 = jnp.atleast_1d(m1)
        m2 = jnp.atleast_1d(m2)

        log_p_pl = self._powerlaw_component(m1, alpha, mmin, mmax)
        log_p_gauss = self._gaussian_component(m1, mu_g, sigma_g, mmin, mmax)
        log_p_m1 = jnp.logaddexp(
            log_p_pl + jnp.log(1 - lambda_peak),
            log_p_gauss + jnp.log(lambda_peak),
        )

        log_p_m2_unnorm = beta * jnp.log(m2)
        log_norm_m2 = vmap(
            lambda m1_val: PowerLawMass._log_norm_m2(beta, mmin, m1_val)
        )(m1)
        log_p_m2_given_m1 = log_p_m2_unnorm - log_norm_m2

        log_S_m1 = self._smoothing_function(m1, mmin, delta_m)
        log_S_m2 = self._smoothing_function(m2, mmin, delta_m)

        in_range_m1 = (m1 >= mmin) & (m1 <= mmax)
        in_range_m2 = (m2 >= mmin) & (m2 <= m1)
        in_range = in_range_m1 & in_range_m2

        log_p = jnp.where(
            in_range,
            log_p_m1 + log_p_m2_given_m1 + log_S_m1 + log_S_m2,
            -jnp.inf,
        )

        return log_p

    def log_pdf_chirp_mass(self, mc_source: jnp.ndarray, **params: Any) -> jnp.ndarray:
        alpha = params.get("alpha", 3.78)
        mmin = params.get("mmin", 4.98)
        mmax = params.get("mmax", 112.5)
        delta_m = params.get("delta_m", 4.8)
        mu_g = params.get("mu_g", 32.27)
        sigma_g = params.get("sigma_g", 3.88)
        lambda_peak = params.get("lambda_peak", 0.03)

        mc_source = jnp.atleast_1d(mc_source)
        q_avg = 0.7
        m1_approx = mc_source * (1 + q_avg) ** (1.0 / 5.0)

        log_p_pl = self._powerlaw_component(m1_approx, alpha, mmin, mmax)
        log_p_gauss = self._gaussian_component(m1_approx, mu_g, sigma_g, mmin, mmax)
        log_p_m1 = jnp.logaddexp(
            log_p_pl + jnp.log(1 - lambda_peak),
            log_p_gauss + jnp.log(lambda_peak),
        )

        log_S = self._smoothing_function(m1_approx, mmin, delta_m)
        in_range = (m1_approx >= mmin) & (m1_approx <= mmax)
        return jnp.where(in_range, log_p_m1 + log_S, -jnp.inf)


class RateEvolution(ABC):
    """Abstract base class for merger-rate evolution R(z)."""

    parameter_names: tuple[str, ...] = ()

    @abstractmethod
    def rate(self, z: jnp.ndarray, **params: Any) -> jnp.ndarray:
        raise NotImplementedError

    def log_rate(self, z: jnp.ndarray, **params: Any) -> jnp.ndarray:
        return jnp.log(self.rate(z, **params) + 1e-300)


class MadauRate(RateEvolution):
    """Madau-Dickinson star-formation-inspired merger rate."""

    parameter_names = ("gamma", "kappa", "zp")

    @staticmethod
    @jit
    def _rate_jax(z: jnp.ndarray, gamma: float, kappa: float, zp: float) -> jnp.ndarray:
        return (1 + z) ** gamma / (1 + ((1 + z) / (1 + zp)) ** kappa)

    def rate(self, z: jnp.ndarray, **params: Any) -> jnp.ndarray:
        gamma = params.get("gamma", 2.7)
        kappa = params.get("kappa", params.get("Madau_k", params.get("k", 2.9)))
        zp = params.get("zp", params.get("Madau_zp", 2.47))
        return self._rate_jax(z, gamma, kappa, zp)

    def log_rate(self, z: jnp.ndarray, **params: Any) -> jnp.ndarray:
        return super().log_rate(z, **params)


class BNSMassPrior(MassPrior):
    """Simple BNS power-law mass prior in source frame."""

    parameter_names = ("mminns", "mmaxns", "alphans")

    @staticmethod
    @jit
    def _log_norm(alpha: float, mmin: float, mmax: float) -> float:
        is_special = jnp.abs(alpha + 1.0) < 1e-6
        normal_norm = jnp.log((mmax ** (alpha + 1) - mmin ** (alpha + 1)) / (alpha + 1))
        special_norm = jnp.log(jnp.log(mmax / mmin))
        return jnp.where(is_special, special_norm, normal_norm)

    def log_prob(self, m1: jnp.ndarray, m2: jnp.ndarray, **params: Any) -> jnp.ndarray:
        alphans = params.get("alphans", 0.0)
        mminns = params.get("mminns", 1.0)
        mmaxns = params.get("mmaxns", 3.0)

        m1 = jnp.atleast_1d(m1)
        m2 = jnp.atleast_1d(m2)

        log_norm = self._log_norm(alphans, mminns, mmaxns)
        log_p_m1 = alphans * jnp.log(m1 / mminns) - log_norm
        log_p_m2 = alphans * jnp.log(m2 / mminns) - log_norm

        in_range = (m1 >= mminns) & (m1 <= mmaxns) & (m2 >= mminns) & (m2 <= mmaxns)
        return jnp.where(in_range, log_p_m1 + log_p_m2, -jnp.inf)
