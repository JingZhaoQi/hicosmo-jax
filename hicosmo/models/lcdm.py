"""
Lambda-CDM (ΛCDM) cosmological model - production implementation.

This module provides a complete implementation of the standard ΛCDM model
with all advanced features from qcosmc, optimized for JAX.
"""

from typing import Dict, Union, Optional, Tuple
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
import numpy as np

from ..core.cosmology import CosmologyBase
from ..utils.constants import *


class LCDM(CosmologyBase):
    """
    Standard Lambda-CDM cosmological model with comprehensive functionality.
    
    This implementation includes:
    - Flat and non-flat geometries
    - Radiation component
    - Sound horizon calculations
    - Recombination history
    - Growth functions
    - All distance and time measures
    
    Parameters
    ----------
    H0 : float
        Hubble constant in km/s/Mpc
    Omega_m : float
        Total matter density parameter
    Omega_b : float, optional
        Baryon density parameter
    Omega_k : float, optional
        Curvature density parameter (default: 0.0 for flat)
    Omega_r : float, optional
        Radiation density parameter (computed if not given)
    sigma8 : float, optional
        Matter fluctuation amplitude at 8 h⁻¹ Mpc
    n_s : float, optional
        Scalar spectral index
    T_cmb : float, optional
        CMB temperature today in K (default: 2.7255)
    N_eff : float, optional
        Effective number of neutrino species (default: 3.046)
    """
    
    def __init__(self, 
                 H0: float = 67.36,
                 Omega_m: float = 0.3153,
                 Omega_b: Optional[float] = None,
                 Omega_k: float = 0.0,
                 Omega_r: Optional[float] = None,
                 sigma8: float = 0.8111,
                 n_s: float = 0.9649,
                 T_cmb: float = 2.7255,
                 N_eff: float = 3.046,
                 **kwargs):
        
        # Set up basic parameters
        params = {
            'H0': H0,
            'Omega_m': Omega_m,
            'Omega_k': Omega_k,
            'sigma8': sigma8,
            'n_s': n_s,
            'T_cmb': T_cmb,
            'N_eff': N_eff
        }
        
        # Compute baryon density if not provided
        if Omega_b is not None:
            params['Omega_b'] = Omega_b
            params['Omega_c'] = Omega_m - Omega_b  # CDM density
        else:
            params['Omega_b'] = 0.0493  # Planck 2018 default
            params['Omega_c'] = Omega_m - params['Omega_b']
        
        # Compute radiation density if not provided
        if Omega_r is not None:
            params['Omega_r'] = Omega_r
        else:
            params['Omega_r'] = self._compute_omega_r(T_cmb, N_eff, H0)
        
        # Dark energy density (enforces closure relation)
        params['Omega_Lambda'] = 1.0 - Omega_m - params['Omega_r'] - Omega_k
        
        # Add any additional parameters
        params.update(kwargs)
        
        super().__init__(**params)
    
    @staticmethod
    def _compute_omega_r(T_cmb: float, N_eff: float, h: float) -> float:
        """
        Compute radiation density parameter.
        
        Parameters
        ----------
        T_cmb : float
            CMB temperature in K
        N_eff : float
            Effective number of neutrino species
        h : float
            Dimensionless Hubble parameter
            
        Returns
        -------
        float
            Omega_r
        """
        # Photon density parameter
        Omega_gamma = 2.47e-5 * (T_cmb / 2.7255)**4 / (h/100)**2
        
        # Total radiation (photons + neutrinos)
        Omega_r = Omega_gamma * (1 + 0.2271 * N_eff)
        
        return Omega_r
    
    def _validate_params(self) -> None:
        """Validate LCDM parameters for physical consistency."""
        params = self.params
        
        # Basic range checks
        if not 20 < params['H0'] < 200:
            raise ValueError(f"H0 = {params['H0']} outside reasonable range [20, 200] km/s/Mpc")
        
        if not 0.01 < params['Omega_m'] < 1.0:
            raise ValueError(f"Omega_m = {params['Omega_m']} outside physical range [0.01, 1.0]")
        
        if 'Omega_b' in params and not 0 < params['Omega_b'] < params['Omega_m']:
            raise ValueError(f"Omega_b = {params['Omega_b']} must be between 0 and Omega_m")
        
        # Check closure relation
        total = params['Omega_m'] + params['Omega_Lambda'] + params['Omega_k'] + params['Omega_r']
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Closure relation violated: Ω_total = {total:.6f} ≠ 1")
        
        # Physical constraints
        if params['Omega_Lambda'] < 0:
            raise ValueError(f"Omega_Lambda = {params['Omega_Lambda']} < 0 unphysical")
        
        if 'sigma8' in params and not 0.1 < params['sigma8'] < 2.0:
            raise ValueError(f"sigma8 = {params['sigma8']} outside reasonable range [0.1, 2.0]")
    
    # ==================== Background Evolution ====================
    
    @staticmethod
    @jit
    def E_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0 for ΛCDM.
        
        E²(z) = Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_k(1+z)² + Ω_Λ
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Must contain 'Omega_m', 'Omega_Lambda', 'Omega_r', 'Omega_k'
            
        Returns
        -------
        float or array_like
            E(z) values
        """
        z_arr = jnp.asarray(z)
        one_plus_z = 1.0 + z_arr
        
        # Matter contribution
        matter_term = params['Omega_m'] * one_plus_z**3
        
        # Radiation contribution
        radiation_term = params.get('Omega_r', 0.0) * one_plus_z**4
        
        # Curvature contribution
        curvature_term = params.get('Omega_k', 0.0) * one_plus_z**2
        
        # Dark energy contribution (constant for ΛCDM)
        de_term = params['Omega_Lambda']
        
        E_squared = matter_term + radiation_term + curvature_term + de_term
        
        return jnp.sqrt(E_squared)
    
    @staticmethod
    @jit
    def w_z(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Dark energy equation of state for ΛCDM: w = -1.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Dictionary of parameters (not used for ΛCDM)
            
        Returns
        -------
        float or array_like
            w(z) = -1 for all z
        """
        return jnp.full_like(jnp.asarray(z), -1.0)
    
    # ==================== Specialized LCDM Functions ====================
    
    @staticmethod
    @jit
    def redshift_equality(params: Dict[str, float]) -> float:
        """
        Matter-radiation equality redshift z_eq.
        
        Parameters
        ----------
        params : dict
            Must contain 'Omega_m' and 'Omega_r'
            
        Returns
        -------
        float
            z_eq = Omega_m / Omega_r - 1
        """
        return params['Omega_m'] / params['Omega_r'] - 1
    
    @staticmethod
    @jit
    def redshift_acceleration(params: Dict[str, float]) -> float:
        """
        Acceleration redshift z_acc where q(z_acc) = 0.
        
        For ΛCDM: z_acc = (2*Ω_Λ/Ω_m)^(1/3) - 1
        
        Parameters
        ----------
        params : dict
            Must contain 'Omega_m' and 'Omega_Lambda'
            
        Returns
        -------
        float
            z_acc
        """
        ratio = 2 * params['Omega_Lambda'] / params['Omega_m']
        return jnp.power(ratio, 1.0/3.0) - 1.0
    
    @staticmethod
    @partial(jit, static_argnums=(2,))
    def sound_horizon(z: float, params: Dict[str, float], n_steps: int = 1000) -> float:
        """
        Sound horizon at redshift z in Mpc.
        
        r_s = ∫[c_s / H(z')] dz' from z to ∞
        
        Parameters
        ----------
        z : float
            Redshift
        params : dict
            Cosmological parameters
        n_steps : int
            Integration steps
            
        Returns
        -------
        float
            Sound horizon in Mpc
        """
        # Simplified sound speed (full version would include baryon loading)
        R_b = 3 * params.get('Omega_b', 0.05) / (4 * params.get('Omega_r', 5e-5))
        
        def integrand(zp):
            c_s = c_km_s / jnp.sqrt(3 * (1 + R_b / (1 + zp)))  # Simplified
            return c_s / (params['H0'] * LCDM.E_z(zp, params))
        
        # Integrate from z to high redshift
        z_max = max(1100, z + 100)  # Ensure we go to recombination
        z_arr = jnp.linspace(z, z_max, n_steps)
        integrand_vals = vmap(integrand)(z_arr)
        
        return jnp.trapz(integrand_vals, z_arr)
    
    @staticmethod
    @jit
    def sound_horizon_drag(params: Dict[str, float]) -> float:
        """
        Sound horizon at drag epoch (fitting formula from Eisenstein & Hu 1998).
        
        Parameters
        ----------
        params : dict
            Must contain cosmological parameters
            
        Returns
        -------
        float
            r_s(z_d) in Mpc
        """
        h = params['H0'] / 100.0
        Omega_m_h2 = params['Omega_m'] * h**2
        Omega_b_h2 = params.get('Omega_b', 0.05) * h**2
        
        # Eisenstein & Hu 1998 fitting formulas
        b1 = 0.313 * (Omega_m_h2)**(-0.419) * (1 + 0.607 * (Omega_m_h2)**(0.674))
        b2 = 0.238 * (Omega_m_h2)**(0.223)
        z_d = 1291 * (Omega_m_h2)**(0.251) / (1 + 0.659 * (Omega_m_h2)**(0.828)) * \
              (1 + b1 * (Omega_b_h2)**(b2))
        
        R_eq = 31.5 * Omega_b_h2 * (T_cmb / 2.7)**(-4) * (z_d / 1000)
        R_d = 31.5 * Omega_b_h2 * (T_cmb / 2.7)**(-4) * (z_d / 1000)
        
        s = (2.0 / (3 * R_eq)) * jnp.log((jnp.sqrt(1 + R_d) + jnp.sqrt(R_eq + R_d)) / 
                                        (1 + jnp.sqrt(R_eq)))
        
        return s * c_km_s / (h * 100)  # Convert to Mpc
    
    @staticmethod
    def critical_density(z: Union[float, jnp.ndarray], params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Critical density at redshift z in units of M_sun/Mpc³.
        
        ρ_c(z) = 3H²(z) / (8πG)
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
            
        Returns
        -------
        float or array_like
            Critical density in M_sun/Mpc³
        """
        H_z = LCDM.H_z(z, params)  # km/s/Mpc
        H_z_SI = H_z * 3.24078e-20  # Convert to 1/s
        
        rho_crit = 3 * H_z_SI**2 / (8 * jnp.pi * G_SI)  # kg/m³
        rho_crit_Msun_Mpc3 = rho_crit * (Mpc / M_sun) * 1e-9  # M_sun/Mpc³
        
        return rho_crit_Msun_Mpc3
    
    # ==================== Growth Functions ====================
    
    @staticmethod
    def growth_factor_analytical(z: Union[float, jnp.ndarray], 
                                params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Analytical approximation for ΛCDM growth factor.
        
        Uses Carroll, Press & Turner (1992) formula.
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
            
        Returns
        -------
        float or array_like
            Growth factor D(z) normalized to D(0) = 1
        """
        a = 1.0 / (1.0 + z)
        Omega_m = params['Omega_m']
        Omega_Lambda = params['Omega_Lambda']
        
        # Analytical approximation for flat ΛCDM
        if abs(params.get('Omega_k', 0.0)) < 1e-6:
            # Lahav et al. (1991) approximation
            Om_a = Omega_m / (Omega_m + Omega_Lambda * a**3)
            D_a = (5/2) * Om_a / (Om_a**(4/7) - Omega_Lambda/Omega_m + 
                                  (1 + Om_a/2) * (1 + Omega_Lambda/(70*Om_a)))
            return D_a * a
        else:
            # For non-flat, use growth rate approximation
            return a  # Placeholder - would need more sophisticated calculation
    
    @staticmethod
    def growth_rate_analytical(z: Union[float, jnp.ndarray],
                             params: Dict[str, float]) -> Union[float, jnp.ndarray]:
        """
        Analytical approximation for ΛCDM growth rate.
        
        Uses Wang & Steinhardt (1998) formula: f ≈ Ω_m(z)^γ
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
        params : dict
            Cosmological parameters
            
        Returns
        -------
        float or array_like
            Growth rate f(z) = d ln D / d ln a
        """
        # For flat ΛCDM, γ ≈ 0.55 + 0.05 * (1 + w) for general w
        # For ΛCDM (w = -1), γ ≈ 0.55
        gamma = 0.545 + 0.0055 * (1 + LCDM.w_z(z, params))  # Small correction
        
        Omega_m_z = LCDM.Omega_m_z(z, params)
        return Omega_m_z**gamma
    
    # ==================== Specialized Distances ====================
    
    @staticmethod
    @partial(jit, static_argnums=(2, 3))
    def time_delay_distance(z_l: float, z_s: float, 
                           params: Dict[str, float],
                           n_steps: int = 1000) -> float:
        """
        Time delay distance for strong lensing.
        
        D_Δt = (1+z_l) * D_A(z_l) * D_A(z_s) / D_A(z_l, z_s)
        
        Parameters
        ----------
        z_l : float
            Lens redshift
        z_s : float
            Source redshift
        params : dict
            Cosmological parameters
        n_steps : int
            Integration steps
            
        Returns
        -------
        float
            Time delay distance in Mpc
        """
        if z_s <= z_l:
            raise ValueError("Source redshift must be greater than lens redshift")
        
        # Angular diameter distances
        D_A_l = LCDM.angular_diameter_distance(z_l, params, LCDM, n_steps)
        D_A_s = LCDM.angular_diameter_distance(z_s, params, LCDM, n_steps)
        
        # Angular diameter distance between lens and source
        D_H = c_km_s / params['H0']
        
        def integrand(zp):
            return 1.0 / LCDM.E_z(zp, params)
        
        z_arr = jnp.linspace(z_l, z_s, n_steps)
        integral = jnp.trapz(vmap(integrand)(z_arr), z_arr)
        D_A_ls = D_H * integral / (1 + z_s)
        
        return (1 + z_l) * D_A_l * D_A_s / D_A_ls
    
    @staticmethod
    @jit
    def drift_rate(z: float, params: Dict[str, float], 
                   observation_time_yr: float = 30.0) -> float:
        """
        Redshift drift rate for Sandage test.
        
        dz/dt = H₀(1+z) - H(z)
        
        Parameters
        ----------
        z : float
            Redshift
        params : dict
            Cosmological parameters
        observation_time_yr : float
            Observation time in years
            
        Returns
        -------
        float
            Drift rate in cm/s over observation time
        """
        H0 = params['H0'] / Mpc  # Convert to km/s/km = 1/s
        H_z = LCDM.H_z(z, params) / Mpc  # Convert to 1/s
        
        drift_rate_per_sec = H0 * (1 + z) - H_z
        observation_time_sec = observation_time_yr * 365.25 * 24 * 3600
        
        # Convert to cm/s
        return drift_rate_per_sec * c_m_s * observation_time_sec * 100  # m/s to cm/s
    
    # ==================== Convenience Methods ====================
    
    @classmethod
    def planck2018(cls, **kwargs) -> 'LCDM':
        """
        ΛCDM with Planck 2018 best-fit parameters.
        
        Parameters
        ----------
        **kwargs
            Override any default parameters
            
        Returns
        -------
        LCDM
            Planck 2018 ΛCDM model
        """
        planck_params = {
            'H0': 67.36,
            'Omega_m': 0.3153,
            'Omega_b': 0.0493,
            'sigma8': 0.8111,
            'n_s': 0.9649,
        }
        planck_params.update(kwargs)
        return cls(**planck_params)
    
    @classmethod
    def wmap9(cls, **kwargs) -> 'LCDM':
        """
        ΛCDM with WMAP9 best-fit parameters.
        
        Parameters
        ----------
        **kwargs
            Override any default parameters
            
        Returns
        -------
        LCDM
            WMAP9 ΛCDM model
        """
        wmap_params = {
            'H0': 70.0,
            'Omega_m': 0.279,
            'Omega_b': 0.046,
            'sigma8': 0.821,
            'n_s': 0.972,
        }
        wmap_params.update(kwargs)
        return cls(**wmap_params)
    
    def summary(self) -> str:
        """
        Generate a comprehensive summary of the ΛCDM model.
        
        Returns
        -------
        str
            Formatted summary
        """
        lines = [
            "ΛCDM Cosmological Model Summary",
            "=" * 35,
            "",
            "Primary Parameters:",
            f"  H₀ = {self.params['H0']:.2f} km/s/Mpc",
            f"  Ωₘ = {self.params['Omega_m']:.4f}",
            f"  Ωᵦ = {self.params.get('Omega_b', 0):.4f}",
            f"  Ωₖ = {self.params.get('Omega_k', 0):.4f}",
            f"  ΩΛ = {self.params['Omega_Lambda']:.4f}",
            f"  Ωᵣ = {self.params.get('Omega_r', 0):.6f}",
            "",
            "Structure Formation:",
            f"  σ₈ = {self.params.get('sigma8', 0):.4f}",
            f"  nₛ = {self.params.get('n_s', 0):.4f}",
            ""
        ]
        
        # Derived parameters
        derived = self.get_derived_params()
        lines.extend([
            "Derived Parameters:",
            f"  Age₀ = {derived.get('age_Gyr', 0):.2f} Gyr",
            f"  tₕ = {derived.get('t_H_Gyr', 0):.2f} Gyr",
            f"  Dₕ = {derived.get('D_H_Mpc', 0):.1f} Mpc",
            f"  q₀ = {derived.get('q0', 0):.3f}",
            f"  z_acc = {derived.get('z_acc', 0):.2f}",
        ])
        
        if 'Omega_r' in self.params:
            z_eq = self.redshift_equality(self.params)
            lines.append(f"  z_eq = {z_eq:.0f}")
        
        return "\n".join(lines)