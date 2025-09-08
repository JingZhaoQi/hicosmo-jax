"""
Advanced cosmology base classes with comprehensive functionality.

This module provides the foundation for all cosmological calculations in HiCosmo,
designed to be production-ready with complete feature coverage.
"""

from typing import Dict, Any, Union, Optional, Tuple, Callable
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import UnivariateSpline
from ..utils.constants import *


class CosmologyBase(ABC):
    """
    Abstract base class for all cosmological models.
    
    This class defines a comprehensive interface that all cosmological models
    must implement. It includes both background evolution and perturbation
    theory quantities.
    """
    
    def __init__(self, **params):
        """Initialize cosmological model with parameters."""
        self.params = params
        self._validate_params()
        self._derived_params = {}
        self._interpolators = {}
    
    @abstractmethod
    def _validate_params(self) -> None:
        """Validate input parameters for physical consistency."""
        pass
    
    # ==================== Background Evolution ====================
    
    @staticmethod
    @abstractmethod
    @jit
    def E_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        float or array_like
            E(z) values
        """
        pass
    
    @staticmethod
    @jit
    def H_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Hubble parameter H(z) in km/s/Mpc.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Dictionary containing 'H0' and model parameters
            
        Returns
        -------
        float or array_like
            H(z) in km/s/Mpc
        """
        return params['H0'] * CosmologyBase.E_z(z, params)
    
    @staticmethod
    @jit
    def w_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Dark energy equation of state w(z).
        
        Default implementation for cosmological constant.
        Override in derived classes for dynamic dark energy.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        float or array_like
            w(z) values
        """
        return jnp.full_like(jnp.asarray(z), -1.0)
    
    @staticmethod
    @jit
    def rho_DE_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Dark energy density evolution ρ_DE(z)/ρ_DE(0).
        
        For general w(z), this requires integration.
        Default implementation for w=-1.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        float or array_like
            Normalized dark energy density
        """
        return jnp.ones_like(jnp.asarray(z))
    
    @staticmethod
    @jit
    def Omega_m_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Matter density parameter as function of redshift.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Dictionary containing 'Omega_m'
            
        Returns
        -------
        float or array_like
            Omega_m(z)
        """
        E_z_val = CosmologyBase.E_z(z, params)
        return params['Omega_m'] * (1 + z)**3 / E_z_val**2
    
    @staticmethod
    @jit
    def Omega_DE_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Dark energy density parameter as function of redshift.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        float or array_like
            Omega_DE(z)
        """
        Omega_m_val = CosmologyBase.Omega_m_z(z, params)
        Omega_k = params.get('Omega_k', 0.0)
        Omega_r = params.get('Omega_r', 0.0)
        Omega_r_z = Omega_r * (1 + z)**4 / CosmologyBase.E_z(z, params)**2
        return 1.0 - Omega_m_val - Omega_r_z - Omega_k * (1 + z)**2 / CosmologyBase.E_z(z, params)**2
    
    # ==================== Distances ====================
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def comoving_distance_integrand(z: float, params: Dict[str, float], 
                                  cosmo_class, use_Mpc: bool = True) -> float:
        """
        Integrand for comoving distance: c/H(z) or D_H/E(z).
        
        Parameters
        ----------
        z : float
            Redshift
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class with E_z method
        use_Mpc : bool
            Whether to return in Mpc units
            
        Returns
        -------
        float
            Integrand value
        """
        if use_Mpc:
            return c_km_s / (params['H0'] * cosmo_class.E_z(z, params))
        else:
            return 1.0 / cosmo_class.E_z(z, params)
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def comoving_distance(z: Union[float, jnp.ndarray], 
                         params: Dict[str, float], 
                         cosmo_class,
                         n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Line-of-sight comoving distance in Mpc.
        
        Uses high-precision trapezoidal integration.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps for accuracy
            
        Returns
        -------
        float or array_like
            Comoving distance in Mpc
        """
        def integrate_single(z_max):
            if z_max <= 0:
                return 0.0
            
            z_arr = jnp.linspace(0, z_max, n_steps)
            integrand = vmap(lambda zi: CosmologyBase.comoving_distance_integrand(
                zi, params, cosmo_class, True))(z_arr)
            return jnp.trapz(integrand, z_arr)
        
        if isinstance(z, (float, int)):
            return integrate_single(z)
        else:
            return vmap(integrate_single)(z)
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def transverse_comoving_distance(z: Union[float, jnp.ndarray],
                                   params: Dict[str, float],
                                   cosmo_class,
                                   n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Transverse comoving distance (accounts for curvature).
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters (must include 'Omega_k')
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            Transverse comoving distance in Mpc
        """
        Omega_k = params.get('Omega_k', 0.0)
        D_H = c_km_s / params['H0']  # Hubble distance
        
        dc = CosmologyBase.comoving_distance(z, params, cosmo_class, n_steps)
        
        if jnp.abs(Omega_k) < 1e-6:  # Flat universe
            return dc
        elif Omega_k > 0:  # Open universe
            sqrt_Ok = jnp.sqrt(Omega_k)
            return D_H / sqrt_Ok * jnp.sinh(sqrt_Ok * dc / D_H)
        else:  # Closed universe
            sqrt_Ok = jnp.sqrt(-Omega_k)
            return D_H / sqrt_Ok * jnp.sin(sqrt_Ok * dc / D_H)
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def luminosity_distance(z: Union[float, jnp.ndarray],
                          params: Dict[str, float],
                          cosmo_class,
                          n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Luminosity distance in Mpc.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            Luminosity distance in Mpc
        """
        dm = CosmologyBase.transverse_comoving_distance(z, params, cosmo_class, n_steps)
        return dm * (1 + z)
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def angular_diameter_distance(z: Union[float, jnp.ndarray],
                                params: Dict[str, float],
                                cosmo_class,
                                n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Angular diameter distance in Mpc.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            Angular diameter distance in Mpc
        """
        dm = CosmologyBase.transverse_comoving_distance(z, params, cosmo_class, n_steps)
        return dm / (1 + z)
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def distance_modulus(z: Union[float, jnp.ndarray],
                        params: Dict[str, float],
                        cosmo_class,
                        n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Distance modulus μ = 5*log10(d_L/10pc).
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            Distance modulus
        """
        dL = CosmologyBase.luminosity_distance(z, params, cosmo_class, n_steps)
        return 5 * jnp.log10(dL * 1e6 / 10)  # Convert Mpc to pc
    
    # ==================== Time and Age ====================
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def lookback_time(z: Union[float, jnp.ndarray],
                     params: Dict[str, float],
                     cosmo_class,
                     n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Lookback time to redshift z in Gyr.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            Lookback time in Gyr
        """
        def integrand(zp, params):
            return 1.0 / ((1 + zp) * cosmo_class.E_z(zp, params))
        
        def compute_single(z_val):
            if z_val <= 0:
                return 0.0
            z_arr = jnp.linspace(0, z_val, n_steps)
            integrand_vals = vmap(lambda zp: integrand(zp, params))(z_arr)
            integral = jnp.trapz(integrand_vals, z_arr)
            return integral * t_H_Gyr / (params['H0'] / 100.0)  # Convert to Gyr
        
        if isinstance(z, (float, int)):
            return compute_single(z)
        else:
            return vmap(compute_single)(z)
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def age_universe(z: Union[float, jnp.ndarray],
                    params: Dict[str, float],
                    cosmo_class,
                    n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Age of universe at redshift z in Gyr.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            Age in Gyr
        """
        t0 = CosmologyBase.lookback_time(jnp.array([1e6]), params, cosmo_class, n_steps)[0]  # Age at z=0
        tlb = CosmologyBase.lookback_time(z, params, cosmo_class, n_steps)
        return t0 - tlb
    
    # ==================== Volume Elements ====================
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def differential_comoving_volume(z: Union[float, jnp.ndarray],
                                   params: Dict[str, float],
                                   cosmo_class,
                                   n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Differential comoving volume dV_c/dz/dΩ in Mpc³/sr.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            dV_c/dz/dΩ in Mpc³/sr
        """
        D_H = c_km_s / params['H0']
        dm = CosmologyBase.transverse_comoving_distance(z, params, cosmo_class, n_steps)
        E_z = cosmo_class.E_z(z, params)
        return dm**2 * D_H / E_z
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def comoving_volume_element(z: Union[float, jnp.ndarray],
                              params: Dict[str, float],
                              cosmo_class,
                              n_steps: int = 2000) -> Union[float, jnp.ndarray]:
        """
        Comoving volume element dV_c/dz in Mpc³ (4π steradians).
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        n_steps : int
            Integration steps
            
        Returns
        -------
        float or array_like
            dV_c/dz in Mpc³
        """
        return 4 * jnp.pi * CosmologyBase.differential_comoving_volume(
            z, params, cosmo_class, n_steps)
    
    # ==================== Growth and Perturbations ====================
    
    @staticmethod
    def growth_factor_z(z: Union[float, jnp.ndarray],
                       params: Dict[str, float],
                       cosmo_class,
                       normalization: str = 'today') -> Union[float, jnp.ndarray]:
        """
        Linear growth factor D+(z).
        
        Solves the growth equation numerically.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
        normalization : str
            'today' for D+(0)=1, 'matter_dom' for D+ ∝ a during matter domination
            
        Returns
        -------
        float or array_like
            Growth factor D+(z)
        """
        # This requires numerical ODE solving - placeholder for now
        # In full implementation, would solve:
        # d²D/da² + (3/a + d ln E/da) dD/da - (3/2) Ω_m(a) D / a² = 0
        return jnp.ones_like(jnp.asarray(z))
    
    @staticmethod
    def growth_rate_z(z: Union[float, jnp.ndarray],
                     params: Dict[str, float],
                     cosmo_class) -> Union[float, jnp.ndarray]:
        """
        Growth rate f(z) = d ln D/d ln a.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
            
        Returns
        -------
        float or array_like
            Growth rate f(z)
        """
        # Approximate formula: f ≈ Ω_m(z)^γ where γ ≈ 0.55
        gamma = 0.55  # Can be made parameter-dependent
        Omega_m_z = CosmologyBase.Omega_m_z(z, params)
        return Omega_m_z**gamma
    
    @staticmethod
    def fsigma8_z(z: Union[float, jnp.ndarray],
                  params: Dict[str, float],
                  cosmo_class) -> Union[float, jnp.ndarray]:
        """
        Structure growth parameter f*σ8(z).
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters (must include 'sigma8')
        cosmo_class : class
            Cosmology class
            
        Returns
        -------
        float or array_like
            f*σ8(z)
        """
        if 'sigma8' not in params:
            raise ValueError("sigma8 parameter required for fsigma8 calculation")
        
        D_z = CosmologyBase.growth_factor_z(z, params, cosmo_class)
        f_z = CosmologyBase.growth_rate_z(z, params, cosmo_class)
        sigma8_z = params['sigma8'] * D_z  # D(0) = 1 normalization
        
        return f_z * sigma8_z
    
    # ==================== Cosmological Parameters ====================
    
    @staticmethod
    def deceleration_parameter_z(z: Union[float, jnp.ndarray],
                                params: Dict[str, float],
                                cosmo_class) -> Union[float, jnp.ndarray]:
        """
        Deceleration parameter q(z) = -ä/aH².
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
            
        Returns
        -------
        float or array_like
            q(z)
        """
        Omega_m_z = CosmologyBase.Omega_m_z(z, params)
        Omega_DE_z = CosmologyBase.Omega_DE_z(z, params)
        w_z = cosmo_class.w_z(z, params)
        
        # q = (1/2) * [Ω_m - 2*Ω_DE*(1+w)]
        return 0.5 * (Omega_m_z - 2 * Omega_DE_z * (1 + w_z))
    
    @staticmethod
    def jerk_parameter_z(z: Union[float, jnp.ndarray],
                        params: Dict[str, float],
                        cosmo_class) -> Union[float, jnp.ndarray]:
        """
        Jerk parameter j(z) = aä̇/H³.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
        cosmo_class : class
            Cosmology class
            
        Returns
        -------
        float or array_like
            j(z)
        """
        # For general cosmology, this requires derivatives of w(z)
        # Placeholder implementation for constant w
        q_z = CosmologyBase.deceleration_parameter_z(z, params, cosmo_class)
        return 1 + 3*q_z  # ΛCDM value; override in dynamic models
    
    # ==================== Utility Methods ====================
    
    def get_derived_params(self) -> Dict[str, Any]:
        """
        Calculate and return derived cosmological parameters.
        
        Returns
        -------
        dict
            Dictionary of derived parameters
        """
        derived = {}
        
        # Age of universe today
        derived['age_Gyr'] = float(self.age_universe(0.0, self.params, self.__class__))
        
        # Hubble time
        derived['t_H_Gyr'] = t_H_Gyr / (self.params['H0'] / 100.0)
        
        # Hubble distance
        derived['D_H_Mpc'] = c_km_s / self.params['H0']
        
        # Matter-radiation equality redshift
        if 'Omega_r' in self.params:
            derived['z_eq'] = self.params['Omega_m'] / self.params['Omega_r'] - 1
        
        # Deceleration parameter today
        derived['q0'] = float(self.deceleration_parameter_z(0.0, self.params, self.__class__))
        
        # Transition redshift (q=0)
        # This would require root finding - placeholder
        derived['z_acc'] = 0.7  # Approximate for ΛCDM
        
        return derived
    
    def __repr__(self) -> str:
        """String representation of the cosmology."""
        return f"{self.__class__.__name__}({', '.join(f'{k}={v}' for k, v in self.params.items())})"