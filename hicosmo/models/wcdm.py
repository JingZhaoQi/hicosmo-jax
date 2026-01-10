"""
wCDM cosmological model with constant dark energy equation of state.

Minimal implementation: only defines what differs from LCDM.
"""

from typing import Dict, Union
import jax.numpy as jnp

from .lcdm import LCDM
from .base import register_cosmology_model, compute_omega_r


@register_cosmology_model
class wCDM(LCDM):
    """wCDM: LCDM with constant dark energy equation of state w0 ≠ -1."""

    def __init__(self, w0: float = -1.0, **kwargs):
        super().__init__(w0=w0, **kwargs)

    @staticmethod
    def E_z(z: Union[float, jnp.ndarray], params: Dict) -> jnp.ndarray:
        """E(z) for wCDM: includes (1+z)^(3(1+w0)) dark energy term."""
        z_arr = jnp.asarray(z)
        one_plus_z = 1.0 + z_arr

        Omega_m = params['Omega_m']
        Omega_k = params.get('Omega_k', 0.0)
        Omega_r = compute_omega_r(params)
        Omega_Lambda = params.get('Omega_Lambda', 1.0 - Omega_m - Omega_k - Omega_r)
        w0 = params.get('w0', -1.0)

        E_squared = (
            Omega_m * one_plus_z**3 +
            Omega_r * one_plus_z**4 +
            Omega_k * one_plus_z**2 +
            Omega_Lambda * one_plus_z**(3.0 * (1.0 + w0))
        )
        return jnp.sqrt(E_squared)

    def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Dark energy equation of state: constant w0."""
        return jnp.full_like(jnp.asarray(z), self.params.get('w0', -1.0))

    @classmethod
    def get_parameters(cls):
        """Return wCDM parameters (LCDM + w0)."""
        from ..parameters import Parameter

        params = LCDM.get_parameters()
        params.append(
            Parameter(
                name='w0',
                value=-1.0,
                free=False,
                prior={'dist': 'uniform', 'min': -2.5, 'max': 0.0},
                latex_label=r'$w_0$',
                description='Dark energy equation of state (constant)'
            )
        )
        return params
