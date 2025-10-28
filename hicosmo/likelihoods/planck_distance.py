"""Planck 2018 distance prior likelihood compatible with HIcosmo and JAX."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import jax.numpy as jnp

from .base import Likelihood
from ..utils.constants import c_km_s


@dataclass
class PlanckDistancePriorData:
    R: float = 1.750235
    l_a: float = 301.4707
    omega_b_h2: float = 0.02235976
    inv_cov: jnp.ndarray = field(
        default_factory=lambda: jnp.array(
            [
                [9.43923971e04, -1.3604913e03, 1.6645172916e06],
                [-1.3604913e03, 1.6143490e02, 3.6716180e03],
                [1.6645172916e06, 3.6716180e03, 7.97191825162e07],
            ],
            dtype=jnp.float32,
        )
    )

    def data_vector(self) -> jnp.ndarray:
        return jnp.array([self.R, self.l_a, self.omega_b_h2], dtype=jnp.float32)


class Planck2018DistancePriorsLikelihood(Likelihood):
    """Compressed Planck 2018 TT,TE,EE+lowE distance prior."""

    def __init__(
        self,
        dataset: Optional[PlanckDistancePriorData] = None,
        name: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.dataset = dataset or PlanckDistancePriorData()
        self.obs_vec = self.dataset.data_vector()
        self.inv_cov = self.dataset.inv_cov
        super().__init__(name=name or "planck2018_distance", data_path=None, **kwargs)
        self.initialize()

    def _default_dataset_name(self) -> str:
        return "planck2018_distance"

    def _load_data(self) -> None:  # type: ignore[override]
        return

    def _setup_covariance(self) -> None:  # type: ignore[override]
        return

    def get_requirements(self) -> Dict[str, Any]:  # type: ignore[override]
        return {}

    def theory(self, cosmology, **kwargs):  # type: ignore[override]
        raise NotImplementedError

    def log_likelihood(self, cosmology, **kwargs) -> float:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float32)
        Omega_m = jnp.asarray(cosmology.params["Omega_m"], dtype=jnp.float32)
        Omega_b = cosmology.params.get("Omega_b")
        if Omega_b is None:
            raise ValueError("Planck distance priors require Omega_b in the cosmology parameters.")
        Omega_b = jnp.asarray(Omega_b, dtype=jnp.float32)

        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2
        omega_m_h2 = Omega_m * h**2

        g1 = 0.0783 * omega_b_h2**(-0.238) / (1.0 + 39.5 * omega_b_h2**0.763)
        g2 = 0.560 / (1.0 + 21.1 * omega_b_h2**1.81)
        z_star = 1048.0 * (1.0 + 0.00124 * omega_b_h2**(-0.738)) * (1.0 + g1 * omega_m_h2**g2)

        D_A = self._angular_diameter_distance(cosmology, z_star)
        r_s = self._sound_horizon(cosmology, z_star, omega_b_h2)

        shift_R = jnp.sqrt(Omega_m) * (H0 / c_km_s) * D_A * (1.0 + z_star)
        l_a = jnp.pi * D_A * (1.0 + z_star) / r_s

        model_vec = jnp.stack([shift_R, l_a, omega_b_h2])
        diff = model_vec - self.obs_vec
        chi2 = diff @ (self.inv_cov @ diff)
        return -0.5 * chi2

    def get_derived_params(self, cosmology) -> Dict[str, float]:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float32)
        Omega_m = jnp.asarray(cosmology.params["Omega_m"], dtype=jnp.float32)
        Omega_b = jnp.asarray(cosmology.params["Omega_b"], dtype=jnp.float32)
        h = H0 / 100.0
        omega_b_h2 = Omega_b * h**2
        omega_m_h2 = Omega_m * h**2

        g1 = 0.0783 * omega_b_h2**(-0.238) / (1.0 + 39.5 * omega_b_h2**0.763)
        g2 = 0.560 / (1.0 + 21.1 * omega_b_h2**1.81)
        z_star = 1048.0 * (1.0 + 0.00124 * omega_b_h2**(-0.738)) * (1.0 + g1 * omega_m_h2**g2)

        D_A = self._angular_diameter_distance(cosmology, z_star)
        r_s = self._sound_horizon(cosmology, z_star, omega_b_h2)
        shift_R = jnp.sqrt(Omega_m) * (H0 / c_km_s) * D_A * (1.0 + z_star)
        l_a = jnp.pi * D_A * (1.0 + z_star) / r_s

        return {
            "R": float(shift_R),
            "l_a": float(l_a),
            "omega_b_h2": float(omega_b_h2),
            "z_star": float(z_star),
            "r_s_zstar": float(r_s),
        }

    def _angular_diameter_distance(self, cosmology, z: jnp.ndarray) -> jnp.ndarray:
        z = jnp.asarray(z, dtype=jnp.float32)
        z_grid = jnp.linspace(0.0, z, 16384)
        H_vals = cosmology.H_z(z_grid)
        integrand = c_km_s / H_vals
        D_C = self._trapz(integrand, z_grid)
        return D_C / (1.0 + z)

    def _sound_horizon(self, cosmology, z_star: jnp.ndarray, omega_b_h2: jnp.ndarray) -> jnp.ndarray:
        H0 = jnp.asarray(cosmology.params["H0"], dtype=jnp.float32)
        h = H0 / 100.0
        Tcmb = jnp.asarray(cosmology.params.get("T_cmb", 2.7255), dtype=jnp.float32)

        theta_cmb = Tcmb / 2.7
        Omega_m_h2 = jnp.asarray(cosmology.params["Omega_m"], dtype=jnp.float32) * h**2
        Omega_b_h2 = omega_b_h2

        z_eq = 2.50e4 * Omega_m_h2 * theta_cmb**-4
        k_eq = 7.46e-2 * Omega_m_h2 * theta_cmb**-2

        R_eq = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z_eq)
        R_star = 31.5 * Omega_b_h2 * theta_cmb**-4 * (1000.0 / z_star)

        sqrt_term = jnp.sqrt(6.0 / R_eq)
        log_arg = (jnp.sqrt(1.0 + R_star) + jnp.sqrt(R_star + R_eq)) / (1.0 + jnp.sqrt(R_eq))
        return (2.0 / (3.0 * k_eq)) * sqrt_term * jnp.log(log_arg)

    @staticmethod
    def _trapz(y: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
        dx = x[1:] - x[:-1]
        avg = 0.5 * (y[:-1] + y[1:])
        return jnp.sum(avg * dx)
