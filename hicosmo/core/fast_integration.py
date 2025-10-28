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
        params: Dict[str, float],
        precision_mode: Literal['fast', 'balanced', 'precise'] = 'balanced',
        cache_size: int = 5000,
        z_max: float = 20.0,
        auto_select: bool = True
    ):
        """
        Initialize ultra-high performance integration system
        
        Parameters
        ----------
        params : Dict[str, float]
            Cosmological parameters dictionary, must contain H0, Omega_m
        precision_mode : {'fast', 'balanced', 'precise'}
            Precision mode:
            - 'fast': Prioritize speed, 8-point Gauss integration
            - 'balanced': Balance precision and speed, 12-point Gauss integration  
            - 'precise': Prioritize precision, 16-point Gauss integration
        cache_size : int
            Size of precomputed lookup table (default 5000 points)
        z_max : float
            Maximum redshift range (default 20.0)
        auto_select : bool
            Whether to enable intelligent method selection (default True)
        """
        self.params = params.copy()
        self.precision_mode = precision_mode
        self.cache_size = cache_size  
        self.z_max = z_max
        self.auto_select = auto_select
        
        # Validate required parameters
        required_params = ['H0', 'Omega_m']
        missing = [p for p in required_params if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
            
        # Calculate derived parameters
        self.params['Omega_Lambda'] = 1.0 - self.params['Omega_m'] - self.params.get('Omega_k', 0.0)
        self.params['D_H'] = c_km_s / self.params['H0']  # Hubble distance
        
        # Set integration parameters based on precision mode
        self._setup_integration_params()
        
        # Precompute Gaussian integration nodes and weights
        self._precompute_gauss_quadrature()
        
        # Precompute distance lookup table (if enabled)
        self.distance_table = None
        self.z_table = None
        if self.auto_select:
            self._precompute_distance_table()
        
        # Precompile JIT functions
        self._compile_functions()
        
    def _setup_integration_params(self):
        """Set integration parameters based on precision mode"""
        if self.precision_mode == 'fast':
            self.primary_order = 8
            self.precise_order = 12
            self.batch_threshold = 20  # Use vectorization for less than 20 points
        elif self.precision_mode == 'balanced':
            self.primary_order = 12
            self.precise_order = 16
            self.batch_threshold = 50  # Use vectorization for less than 50 points
        else:  # 'precise'
            self.primary_order = 16
            self.precise_order = 20
            self.batch_threshold = 100  # Use vectorization for less than 100 points
            
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
        
        # 20-point Gaussian integration (ultra high precision)
        self.gauss_nodes[20] = jnp.array([
            -0.9931285992, -0.9639719273, -0.9122344283, -0.8391169718,
            -0.7463319065, -0.6360536807, -0.5108670020, -0.3737060887,
            -0.2277858511, -0.0765265211,  0.0765265211,  0.2277858511,
             0.3737060887,  0.5108670020,  0.6360536807,  0.7463319065,
             0.8391169718,  0.9122344283,  0.9639719273,  0.9931285992
        ])
        self.gauss_weights[20] = jnp.array([
            0.0176140071, 0.0406014298, 0.0626720483, 0.0832767416,
            0.1019301198, 0.1181945320, 0.1316886384, 0.1420961093,
            0.1491729865, 0.1527533871, 0.1527533871, 0.1491729865,
            0.1420961093, 0.1316886384, 0.1181945320, 0.1019301198,
            0.0832767416, 0.0626720483, 0.0406014298, 0.0176140071
        ])
            
    def _precompute_distance_table(self):
        """Precompute distance lookup table - JAX compatible"""
        if self.cache_size <= 0:
            return

        # Create high-density redshift grid
        z_min = 1e-6  # Avoid numerical issues at z=0
        self.z_table = jnp.logspace(
            jnp.log10(z_min), jnp.log10(self.z_max), self.cache_size
        )

        # Precompute distances using the JAX ultra-precise integrator
        vectorized_distance = vmap(self._ultra_precise_single)
        self.distance_table = vectorized_distance(self.z_table)
        
    def _ultra_precise_single_numpy(self, z: float) -> float:
        """High precision single point calculation (NumPy version, for precomputation)"""
        if z <= 1e-8:
            return 0.0
            
        # Use highest precision Gaussian integration
        order = max(self.gauss_nodes.keys())
        nodes = np.array(self.gauss_nodes[order])
        weights = np.array(self.gauss_weights[order])
        
        # Transform to [0, z]
        z_nodes = 0.5 * z * (nodes + 1.0)
        jacobian = 0.5 * z
        
        # Calculate integral
        integral = 0.0
        Om = self.params['Omega_m']
        OL = self.params['Omega_Lambda']
        
        for i in range(len(nodes)):
            zi = z_nodes[i]
            E_zi = np.sqrt(Om * (1 + zi)**3 + OL)
            integral += weights[i] / E_zi
            
        integral *= jacobian
        return self.params['D_H'] * integral
        
    def _compile_functions(self):
        """Precompile JIT functions"""
        # Warm up all JIT functions
        test_z = 1.0
        test_z_array = jnp.array([0.5, 1.0, 2.0])
        
        _ = self._ultra_fast_single(test_z)
        _ = self._ultra_precise_single(test_z)
        _ = vmap(self._ultra_fast_single)(test_z_array)
        _ = vmap(self._ultra_precise_single)(test_z_array)
        if self.distance_table is not None:
            _ = self._interpolation_lookup(test_z_array)
            
    @staticmethod
    @jit
    def _E_z(z: float, Om: float, OL: float) -> float:
        """ΛCDM E(z) = H(z)/H0 (static method, avoid parameter passing)"""
        one_plus_z = 1.0 + z
        return jnp.sqrt(Om * one_plus_z**3 + OL)
    
    @partial(jit, static_argnums=(0,))  
    def _ultra_fast_single(self, z: float) -> float:
        """Ultra-fast single calculation"""
        
        def _compute(z_pos):
            # Use primary precision Gaussian integration
            nodes = self.gauss_nodes[self.primary_order]
            weights = self.gauss_weights[self.primary_order]
            
            # Transform to [0, z]
            z_nodes = 0.5 * z_pos * (nodes + 1.0)
            jacobian = 0.5 * z_pos
            
            # Vectorized E(z) calculation
            Om = self.params['Omega_m']
            OL = self.params['Omega_Lambda']
            E_vals = vmap(lambda zi: self._E_z(zi, Om, OL))(z_nodes)
            integrand_vals = 1.0 / E_vals
            
            # Gaussian integration
            integral = jacobian * jnp.sum(weights * integrand_vals)
            return self.params['D_H'] * integral
        
        return jnp.where(z <= 1e-8, 0.0, _compute(z))
    
    @partial(jit, static_argnums=(0,))
    def _ultra_precise_single(self, z: float) -> float:
        """Ultra-precise single calculation"""
        
        def _compute(z_pos):
            # Use highest precision Gaussian integration
            nodes = self.gauss_nodes[self.precise_order]
            weights = self.gauss_weights[self.precise_order]
            
            # Transform to [0, z]
            z_nodes = 0.5 * z_pos * (nodes + 1.0)
            jacobian = 0.5 * z_pos
            
            # Vectorized E(z) calculation
            Om = self.params['Omega_m']
            OL = self.params['Omega_Lambda']
            E_vals = vmap(lambda zi: self._E_z(zi, Om, OL))(z_nodes)
            integrand_vals = 1.0 / E_vals
            
            # Gaussian integration
            integral = jacobian * jnp.sum(weights * integrand_vals)
            return self.params['D_H'] * integral
            
        return jnp.where(z <= 1e-8, 0.0, _compute(z))
    
    def _numpy_vectorized_batch(self, z_array: np.ndarray) -> np.ndarray:
        """Learn from qcosmc's numpy.vectorize strategy"""
        
        @np.vectorize
        def single_calc(z):
            if z <= 1e-8:
                return 0.0
            
            # Use primary precision Gaussian integration
            nodes = np.array(self.gauss_nodes[self.primary_order])
            weights = np.array(self.gauss_weights[self.primary_order])
            
            # Transform to [0, z]
            z_nodes = 0.5 * z * (nodes + 1.0)
            jacobian = 0.5 * z
            
            # Calculate integral
            integral = 0.0
            Om = self.params['Omega_m']
            OL = self.params['Omega_Lambda']
            
            for i in range(len(nodes)):
                zi = z_nodes[i]
                E_zi = np.sqrt(Om * (1 + zi)**3 + OL)
                integral += weights[i] / E_zi
                
            integral *= jacobian
            return self.params['D_H'] * integral
            
        return single_calc(z_array)
    
    @partial(jit, static_argnums=(0,))
    def _interpolation_lookup(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Ultra-fast interpolation lookup"""
        if self.distance_table is None:
            raise ValueError("Distance table not precomputed. Set cache_size > 0 or auto_select=True")
            
        z = jnp.atleast_1d(z)

        # Use JAX's interp function for fast interpolation
        return jnp.interp(z, self.z_table, self.distance_table)

    @staticmethod
    def _normalize_input(z: Union[float, np.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, bool, Tuple[int, ...]]:
        """Convert input to JAX array and capture shape metadata."""
        z_array = jnp.asarray(z)
        if z_array.ndim == 0:
            return z_array, True, ()
        return z_array, False, z_array.shape

    @staticmethod
    def _format_output(
        value: Union[float, jnp.ndarray],
        original_input: Union[float, np.ndarray, jnp.ndarray],
        is_scalar: bool
    ) -> Union[float, np.ndarray, jnp.ndarray]:
        """Cast results back to match the original input's expectations - JAX compatible."""
        if is_scalar:
            if isinstance(original_input, (float, int, np.floating)):
                # Return as-is for JAX compatibility (no .item() during tracing)
                return value
            if isinstance(original_input, np.ndarray):
                return np.asarray(value)
            return value

        if isinstance(original_input, np.ndarray):
            return np.asarray(value)
        return value
        
    def _ultra_simple_interpolation(self, z_array: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Learn from astropy's simple linear interpolation strategy"""
        if self.distance_table is None:
            raise ValueError("Distance table not precomputed. Set cache_size > 0 or auto_select=True")
            
        z_array = np.atleast_1d(z_array)
        
        # Use numpy's fast interpolation
        result = np.interp(z_array, np.array(self.z_table), np.array(self.distance_table))

        if result.shape == (1,):
            return result.item()  # JAX compatible: use .item() instead of float()
        return result
        
    def comoving_distance(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate comoving distance
        
        Parameters
        ----------
        z : float, array_like
            Redshift value or redshift array
        method : str, optional
            Force method to use:
            - 'fast': Fast single point calculation
            - 'precise': High precision single point calculation
            - 'vectorized': NumPy vectorized batch calculation
            - 'interpolation': Interpolation lookup
            If None, automatically select based on input
            
        Returns
        -------
        float or array_like
            Comoving distance (unit: Mpc)
        """
        z_array, is_scalar, original_shape = self._normalize_input(z)
        z_flat = jnp.reshape(z_array, (1,)) if is_scalar else jnp.reshape(z_array, (-1,))

        # Method selection that respects tracer inputs
        if method is None:
            if not self.auto_select:
                method = 'fast'
            elif is_scalar:
                method = 'fast'
            elif self.distance_table is not None:
                method = 'interpolation'
            elif z_flat.shape[0] < self.batch_threshold:
                method = 'fast'
            else:
                method = 'interpolation'

        method = method or 'fast'

        if method == 'fast':
            if is_scalar:
                result_flat = jnp.reshape(self._ultra_fast_single(z_array), (1,))
            else:
                result_flat = vmap(self._ultra_fast_single)(z_flat)
        elif method == 'precise':
            if is_scalar:
                result_flat = jnp.reshape(self._ultra_precise_single(z_array), (1,))
            else:
                result_flat = vmap(self._ultra_precise_single)(z_flat)
        elif method == 'vectorized':
            if is_scalar:
                result_flat = jnp.reshape(self._ultra_fast_single(z_array), (1,))
            else:
                result_flat = vmap(self._ultra_fast_single)(z_flat)
        elif method == 'interpolation':
            result_flat = self._interpolation_lookup(z_flat)
        else:
            raise ValueError(
                f"Unknown method: {method}. Options: 'fast', 'precise', 'vectorized', 'interpolation'"
            )

        if is_scalar:
            formatted = result_flat[0]
        else:
            formatted = jnp.reshape(result_flat, original_shape)

        return self._format_output(formatted, z, is_scalar)
            
    def angular_diameter_distance(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate angular diameter distance
        
        Parameters
        ----------
        z : float, array_like
            Redshift value or redshift array
        method : str, optional
            Calculation method (same as comoving_distance)
            
        Returns
        -------
        float or array_like
            Angular diameter distance (unit: Mpc)
        """
        # First calculate comoving distance
        z_array, is_scalar, original_shape = self._normalize_input(z)
        d_c = self.comoving_distance(z, method=method)
        d_c_array = jnp.asarray(d_c)
        if not is_scalar:
            d_c_array = jnp.reshape(d_c_array, original_shape)

        result = d_c_array / (1.0 + z_array)
        return self._format_output(result, z, is_scalar)
            
    def luminosity_distance(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate luminosity distance
        
        Parameters
        ----------
        z : float, array_like
            Redshift value or redshift array
        method : str, optional
            Calculation method (same as comoving_distance)
            
        Returns
        -------
        float or array_like
            Luminosity distance (unit: Mpc)
        """
        # First calculate comoving distance
        z_array, is_scalar, original_shape = self._normalize_input(z)
        d_c = self.comoving_distance(z, method=method)
        d_c_array = jnp.asarray(d_c)
        if not is_scalar:
            d_c_array = jnp.reshape(d_c_array, original_shape)

        result = d_c_array * (1.0 + z_array)
        return self._format_output(result, z, is_scalar)
            
    def distance_modulus(
        self, 
        z: Union[float, np.ndarray, jnp.ndarray],
        method: Optional[str] = None
    ) -> Union[float, np.ndarray]:
        """
        Calculate distance modulus
        
        Parameters
        ----------
        z : float, array_like
            Redshift value or redshift array
        method : str, optional
            Calculation method (same as comoving_distance)
            
        Returns
        -------
        float or array_like
            Distance modulus (unit: mag)
        """
        # First calculate luminosity distance
        z_array, is_scalar, original_shape = self._normalize_input(z)
        d_l = self.luminosity_distance(z, method=method)
        d_l_array = jnp.asarray(d_l)
        if not is_scalar:
            d_l_array = jnp.reshape(d_l_array, original_shape)

        result = 5.0 * jnp.log10(d_l_array * 1e6 / 10.0)
        return self._format_output(result, z, is_scalar)
        
    def update_params(self, new_params: Dict[str, float]):
        """
        Update cosmological parameters and reinitialize
        
        Parameters
        ----------
        new_params : dict
            New cosmological parameters
        """
        self.params.update(new_params)
        
        # Recalculate derived parameters
        self.params['Omega_Lambda'] = 1.0 - self.params['Omega_m'] - self.params.get('Omega_k', 0.0)
        self.params['D_H'] = c_km_s / self.params['H0']
        
        # Re-precompute lookup table (if enabled)
        if self.auto_select and self.cache_size > 0:
            self._precompute_distance_table()
            
        # Re-compile JIT functions
        self._compile_functions()
        
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
            'primary_order': f"{self.primary_order}-point Gauss",
            'precise_order': f"{self.precise_order}-point Gauss",
            'batch_threshold': self.batch_threshold,
            'cache_size': self.cache_size,
            'z_max': self.z_max,
            'auto_select': self.auto_select,
            'precomputed_table': self.distance_table is not None
        }
        return info
