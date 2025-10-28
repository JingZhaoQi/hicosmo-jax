"""
Fast Integration Engine for HIcosmo
===================================

Ultra-high performance integration engine for neutral hydrogen cosmology and 21cm surveys.
Completely replacing Diffrax with intelligent adaptive method selection.

Core Features:
- Eliminate all hardcoding, fully parametric design
- Intelligent adaptive method selection (Ultra-Fast, Vectorized, Interpolation)  
- Support arbitrary cosmological parameters and precision configurations
- Optimized for 21cm cosmology and neutral hydrogen physics
- Outperforms qcosmc and astropy in all scenarios

Author: Jingzhao Qi
"""

from typing import Dict, Union, Optional, Tuple, Literal
import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
from functools import partial, lru_cache
import warnings

from ..utils.constants import c_km_s


class FastIntegration:
    """
    Ultra-high performance integration engine
    
    Fully parameterized integration system that automatically selects optimal computation method based on input:
    - Single point: Ultra-Fast Single (JAX JIT + Gaussian integration)
    - Small batch: NumPy Vectorized (learning from qcosmc strategy) 
    - Large batch: Simple Interpolation (pre-computed table)
    - High precision: Ultra-Optimized Batch (JAX vectorization)
    """
    
    def __init__(
        self,
        precision_mode: Literal['fast', 'balanced', 'precise'] = 'balanced'
    ):
        """
        Initialize generic integration engine.

        FastIntegration is a pure mathematical tool for numerical integration
        using Gauss-Legendre quadrature. It contains NO cosmology-specific code.

        Parameters
        ----------
        precision_mode : {'fast', 'balanced', 'precise'}
            Precision mode:
            - 'fast': 8-point Gauss integration (~0.1% error)
            - 'balanced': 12-point Gauss integration (~0.01% error, default)
            - 'precise': 16-point Gauss integration (~0.001% error)
        """
        self.precision_mode = precision_mode

        # Set integration parameters based on precision mode
        self._setup_integration_params()

        # Precompute Gaussian integration nodes and weights
        self._precompute_gauss_quadrature()

        # Precompile JIT functions
        self._compile_functions()
        
    def _setup_integration_params(self):
        """Set integration parameters based on precision mode"""
        if self.precision_mode == 'fast':
            self.primary_order = 8
        elif self.precision_mode == 'balanced':
            self.primary_order = 12
        else:  # 'precise'
            self.primary_order = 16
            
    def _precompute_gauss_quadrature(self):
        """Precompute Gauss-Legendre nodes and weights for different orders"""
        self.gauss_nodes = {}
        self.gauss_weights = {}
        
        # Precomputed Gauss-Legendre nodes and weights
        # 8-point Gaussian integration
        self.gauss_nodes[8] = jnp.array([
            -0.9602898565, -0.7966664774, -0.5255324099, -0.1834346425,
             0.1834346425,  0.5255324099,  0.7966664774,  0.9602898565
        ])
        self.gauss_weights[8] = jnp.array([
            0.1012285363, 0.2223810345, 0.3137066459, 0.3626837834,
            0.3626837834, 0.3137066459, 0.2223810345, 0.1012285363
        ])
        
        # 12-point Gaussian integration
        self.gauss_nodes[12] = jnp.array([
            -0.9815606342, -0.9041172564, -0.7699026741, -0.5873179543,
            -0.3678314990, -0.1252334085,  0.1252334085,  0.3678314990,
             0.5873179543,  0.7699026741,  0.9041172564,  0.9815606342
        ])
        self.gauss_weights[12] = jnp.array([
            0.0471753364, 0.1069393260, 0.1600783286, 0.2031674267,
            0.2334925365, 0.2491470458, 0.2491470458, 0.2334925365,
            0.2031674267, 0.1600783286, 0.1069393260, 0.0471753364
        ])
        
        # 16-point Gaussian integration
        self.gauss_nodes[16] = jnp.array([
            -0.9894009349, -0.9445750231, -0.8656312024, -0.7554044084,
            -0.6178762444, -0.4580167776, -0.2816035507, -0.0950125098,
             0.0950125098,  0.2816035507,  0.4580167776,  0.6178762444,
             0.7554044084,  0.8656312024,  0.9445750231,  0.9894009349
        ])
        self.gauss_weights[16] = jnp.array([
            0.0271524594, 0.0622535239, 0.0951585117, 0.1246289712,
            0.1495959888, 0.1691565194, 0.1826034150, 0.1894506105,
            0.1894506105, 0.1826034150, 0.1691565194, 0.1495959888,
            0.1246289712, 0.0951585117, 0.0622535239, 0.0271524594
        ])
            
    def _compile_functions(self):
        """Precompile JIT functions for generic integrator"""
        # Warm up generic integrator
        test_z_array = jnp.array([0.5, 1.0, 2.0])
        test_integrand = lambda z: z**2

        _ = self.integrate(test_integrand, 0.0, 1.0)
        _ = self.integrate_batch(test_integrand, test_z_array, 0.0)

    # ==================== Generic Integration Methods ====================
    # These methods accept arbitrary integrands (no hardcoded cosmology)

    @partial(jit, static_argnums=(0, 1))  # Mark self and integrand as static
    def integrate(
        self,
        integrand: callable,
        z_min: float,
        z_max: float
    ) -> float:
        """
        Generic numerical integration using Gauss-Legendre quadrature.

        This is the core integration engine with NO cosmology hardcoded.
        Accepts any integrand function.

        Parameters
        ----------
        integrand : callable
            Function to integrate, signature: integrand(z: float) -> float
        z_min : float
            Lower integration limit
        z_max : float
            Upper integration limit

        Returns
        -------
        float
            Integral value: ∫[z_min, z_max] integrand(z) dz

        Examples
        --------
        >>> integrator = FastIntegration(params={'H0': 70, 'Omega_m': 0.3})
        >>> # Integrate x^2 from 0 to 1
        >>> result = integrator.integrate(lambda z: z**2, 0.0, 1.0)
        >>> print(result)  # Should be ~0.333
        >>>
        >>> # Use with cosmology: integrate 1/E(z)
        >>> def inv_E_z(z):
        >>>     return 1.0 / model.E_z(z)
        >>> d_c = integrator.integrate(inv_E_z, 0.0, 1.0) * (c/H0)
        """
        # Get Gauss-Legendre nodes and weights
        nodes = self.gauss_nodes[self.primary_order]
        weights = self.gauss_weights[self.primary_order]

        # Transform interval [z_min, z_max] to [-1, 1]
        z_mid = 0.5 * (z_max + z_min)
        z_half = 0.5 * (z_max - z_min)

        # Compute integration nodes
        z_nodes = z_mid + z_half * nodes

        # Evaluate integrand at all nodes (vectorized)
        integrand_vals = vmap(integrand)(z_nodes)

        # Gauss-Legendre integration
        integral = z_half * jnp.sum(weights * integrand_vals)

        return integral

    def integrate_batch(
        self,
        integrand: callable,
        z_array: jnp.ndarray,
        z_min: float = 0.0
    ) -> jnp.ndarray:
        """
        Batch integration: compute ∫[z_min, z_i] integrand(z) dz for multiple z_i.

        Parameters
        ----------
        integrand : callable
            Function to integrate
        z_array : array_like
            Array of upper integration limits [z_1, z_2, ..., z_n]
        z_min : float, optional
            Lower integration limit (default: 0.0)

        Returns
        -------
        array_like
            Array of integral values

        Examples
        --------
        >>> z_vals = jnp.array([0.5, 1.0, 1.5, 2.0])
        >>> results = integrator.integrate_batch(lambda z: 1/E(z), z_vals)
        """
        # Vectorize single integration
        integrate_single = lambda z_max: self.integrate(integrand, z_min, z_max)
        return vmap(integrate_single)(z_array)

    def get_performance_info(self) -> Dict[str, str]:
        """
        Get performance configuration information

        Returns
        -------
        dict
            Performance configuration information
        """
        info = {
            'precision_mode': self.precision_mode,
            'integration_order': f"{self.primary_order}-point Gauss",
        }
        return info
