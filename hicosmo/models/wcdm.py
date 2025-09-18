"""
wCDM cosmological model with constant dark energy equation of state.

Inherits all functionality from LCDM, only overrides E_z() and w_z() methods.
Uses FastIntegration engine for optimal performance.
"""

from typing import Dict, Union, Optional, Literal
import jax.numpy as jnp
from jax import jit
from functools import partial

from .lcdm import LCDM


class wCDM(LCDM):
    """
    wCDM model extending LCDM with constant dark energy equation of state.

    Only parameter added: w (dark energy equation of state)
    All other functionality inherited from LCDM parent class.
    """

    def __init__(self, w: float = -1.0, **kwargs):
        self.w = w
        kwargs['w'] = w
        super().__init__(**kwargs)


    def E_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Override E_z for wCDM: includes w parameter in dark energy evolution."""
        z = jnp.asarray(z)
        one_plus_z = 1.0 + z

        # Get parameters from instance
        Omega_m = self.params['Omega_m']
        Omega_r = self.params.get('Omega_r', 0.0)
        Omega_k = self.params.get('Omega_k', 0.0)
        Omega_Lambda = self.params['Omega_Lambda']
        w = self.w

        # Components
        matter_term = Omega_m * one_plus_z**3
        radiation_term = Omega_r * one_plus_z**4
        curvature_term = Omega_k * one_plus_z**2
        de_term = Omega_Lambda * one_plus_z**(3.0 * (1.0 + w))

        E_squared = matter_term + radiation_term + curvature_term + de_term

        return jnp.sqrt(E_squared)

    def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Override w_z for wCDM: returns constant w."""
        z = jnp.asarray(z)
        return jnp.full_like(z, self.w)








