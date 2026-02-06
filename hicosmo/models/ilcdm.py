"""
Interacting Lambda-CDM (ILCDM) cosmological model.

Dark matter and dark energy exchange energy via Q = βHρ_c.
Reference: Zhang et al. (2021), ApJ 918:56
"""

from typing import Dict, Union, Optional
import warnings
import jax.numpy as jnp

from .lcdm import LCDM
from .base import register_cosmology_model


@register_cosmology_model
class ILCDM(LCDM):
    """
    Interacting Lambda-CDM: CDM and DE interact via Q = βHρ_c.

    Physical equations:
    - ρ_c(a) = ρ_c0 × a^(-3+β)
    - E²(z) = Ω_Λ + Ω_b(1+z)³ + Ω_r(1+z)⁴ + Ω_c[β/(β-1) + (1/(1-β))(1+z)^(3-3β)]
    """

    def __init__(
        self,
        H0: float = 67.36,
        Omega_m: float = 0.3153,
        Omega_b: Optional[float] = None,
        Omega_k: Optional[float] = None,
        beta: float = 0.0,
        **kwargs,
    ):
        if Omega_b is None:
            Omega_b = 0.0493

        # ILCDM assumes flat universe
        if Omega_k is not None and abs(Omega_k) > 1e-10:
            warnings.warn(f"ILCDM assumes Omega_k=0, ignoring {Omega_k}", UserWarning)

        if not -1.0 < beta < 1.0:
            raise ValueError(f"beta={beta} outside valid range (-1, 1)")

        super().__init__(
            H0=H0, Omega_m=Omega_m, Omega_b=Omega_b, Omega_k=0.0, beta=beta, **kwargs
        )

    @staticmethod
    def E_z(z: Union[float, jnp.ndarray], params: Dict) -> jnp.ndarray:
        """E(z) for ILCDM with interacting dark sector."""
        from .base import compute_omega_r

        z_arr = jnp.asarray(z)
        one_plus_z = 1.0 + z_arr

        Omega_m = params["Omega_m"]
        Omega_b = params.get("Omega_b", 0.0493)
        Omega_r = compute_omega_r(params)
        Omega_Lambda = params.get("Omega_Lambda", 1.0 - Omega_m - Omega_r)
        beta = params.get("beta", 0.0)

        Omega_c = Omega_m - Omega_b

        # Standard terms
        term_DE = Omega_Lambda
        term_baryon = Omega_b * one_plus_z**3
        term_radiation = Omega_r * one_plus_z**4

        # CDM term: modified by interaction (for β→0, reduces to Ω_c(1+z)³)
        beta_abs = jnp.abs(beta)
        beta_safe = jnp.where(beta_abs < 1e-10, 1e-10, beta)
        coeff1 = beta_safe / (beta_safe - 1.0)
        coeff2 = 1.0 / (1.0 - beta_safe)
        ilcdm_cdm = Omega_c * (coeff1 + coeff2 * one_plus_z ** (3.0 - 3.0 * beta_safe))
        term_cdm = jnp.where(beta_abs < 1e-10, Omega_c * one_plus_z**3, ilcdm_cdm)

        return jnp.sqrt(term_DE + term_baryon + term_radiation + term_cdm)

    @classmethod
    def get_parameters(cls):
        """Return ILCDM parameters (LCDM + beta)."""
        from ..parameters import Parameter

        params = LCDM.get_parameters()
        params.append(
            Parameter(
                name="beta",
                value=0.0,
                free=False,
                prior={"dist": "uniform", "min": -1.0, "max": 1.0},
                latex_label=r"$\beta$",
                description="Interaction coupling (β>0: CDM→DE)",
            )
        )
        return params


def planck2018_ilcdm(beta: float = 0.0, **kwargs) -> ILCDM:
    """Create ILCDM with Planck 2018 parameters."""
    defaults = {
        "H0": 67.36,
        "Omega_m": 0.3153,
        "Omega_b": 0.0493,
        "sigma8": 0.8111,
        "n_s": 0.9649,
        "beta": beta,
    }
    defaults.update(kwargs)
    return ILCDM(**defaults)
