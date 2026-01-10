"""
CPL (Chevallier-Polarski-Linder) dark energy model.

Minimal implementation: w(z) = w0 + wa * z/(1+z)
"""

from typing import Dict, Union
import jax.numpy as jnp

from .lcdm import LCDM
from .base import register_cosmology_model, compute_omega_r


@register_cosmology_model
class CPL(LCDM):
    """CPL: time-evolving dark energy w(z) = w0 + wa * z/(1+z)."""

    def __init__(self, w0: float = -1.0, wa: float = 0.0, **kwargs):
        super().__init__(w0=w0, wa=wa, **kwargs)

    @staticmethod
    def E_z(z: Union[float, jnp.ndarray], params: Dict) -> jnp.ndarray:
        """E(z) for CPL with evolving dark energy."""
        z_arr = jnp.asarray(z)
        one_plus_z = 1.0 + z_arr

        Omega_m = params['Omega_m']
        Omega_k = params.get('Omega_k', 0.0)
        Omega_r = compute_omega_r(params)
        Omega_Lambda = params.get('Omega_Lambda', 1.0 - Omega_m - Omega_k - Omega_r)
        w0 = params.get('w0', -1.0)
        wa = params.get('wa', 0.0)

        # CPL dark energy evolution: ρ_DE(z)/ρ_DE(0) = (1+z)^{3(1+w0+wa)} * exp(-3wa*z/(1+z))
        # Reference: Chevallier & Polarski (2001), Linder (2003)
        f_DE = jnp.exp(-3.0 * wa * z_arr / one_plus_z) * one_plus_z**(3.0 * (1.0 + w0 + wa))

        E_squared = (
            Omega_m * one_plus_z**3 +
            Omega_r * one_plus_z**4 +
            Omega_k * one_plus_z**2 +
            Omega_Lambda * f_DE
        )
        return jnp.sqrt(E_squared)

    def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Dark energy equation of state: w(z) = w0 + wa * z/(1+z)."""
        z_arr = jnp.asarray(z)
        w0 = self.params.get('w0', -1.0)
        wa = self.params.get('wa', 0.0)
        return w0 + wa * z_arr / (1.0 + z_arr)

    @classmethod
    def get_parameters(cls):
        """Return CPL parameters (LCDM + w0 + wa)."""
        from ..parameters import Parameter

        params = LCDM.get_parameters()
        params.extend([
            Parameter(
                name='w0',
                value=-1.0,
                free=False,
                prior={'dist': 'uniform', 'min': -3.0, 'max': 0.0},
                latex_label=r'$w_0$',
                description='Dark energy equation of state at z=0'
            ),
            Parameter(
                name='wa',
                value=0.0,
                free=False,
                prior={'dist': 'uniform', 'min': -3.0, 'max': 3.0},
                latex_label=r'$w_a$',
                description='Dark energy equation of state evolution'
            ),
        ])
        return params
