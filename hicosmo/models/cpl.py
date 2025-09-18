"""
CPL (Chevallier-Polarski-Linder) dark energy model.

Inherits all functionality from LCDM, only overrides E_z() and w_z() methods.
Uses FastIntegration engine for optimal performance.
"""

from typing import Dict, Union, Optional, Literal
import jax.numpy as jnp
from jax import jit
from functools import partial

from .lcdm import LCDM


class CPL(LCDM):
    """
    CPL model extending LCDM with time-evolving dark energy equation of state.

    Parameters added: w0, wa for CPL parametrization w(z) = w0 + wa*z/(1+z)
    All other functionality inherited from LCDM parent class.
    """

    def __init__(self, w0: float = -1.0, wa: float = 0.0, **kwargs):
        self.w0 = w0
        self.wa = wa
        kwargs['w0'] = w0
        kwargs['wa'] = wa
        super().__init__(**kwargs)

    def _dark_energy_evolution(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Dark energy density evolution for CPL model."""
        z = jnp.asarray(z)
        one_plus_z = 1.0 + z
        exp_factor = jnp.exp(3.0 * self.wa * z / one_plus_z)
        power_factor = one_plus_z**(3.0 * (1.0 + self.w0 + self.wa))
        return exp_factor * power_factor

    def E_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Override E_z for CPL: includes w0, wa parameters in dark energy evolution."""
        z = jnp.asarray(z)
        one_plus_z = 1.0 + z

        # Get parameters from instance
        Omega_m = self.params['Omega_m']
        Omega_r = self.params.get('Omega_r', 0.0)
        Omega_k = self.params.get('Omega_k', 0.0)
        Omega_Lambda = self.params['Omega_Lambda']

        # Components
        matter_term = Omega_m * one_plus_z**3
        radiation_term = Omega_r * one_plus_z**4
        curvature_term = Omega_k * one_plus_z**2

        # Dark energy with CPL evolution
        f_DE = self._dark_energy_evolution(z)
        de_term = Omega_Lambda * f_DE

        E_squared = matter_term + radiation_term + curvature_term + de_term

        return jnp.sqrt(E_squared)

    def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Override w_z for CPL: w(z) = w0 + wa * z/(1+z)."""
        z = jnp.asarray(z)
        return self.w0 + self.wa * z / (1.0 + z)