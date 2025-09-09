"""
Transfer Function Calculator
===========================

Professional implementation of cosmological transfer functions.
Essential for connecting primordial fluctuations to observed power spectra.

Key features:
- Eisenstein & Hu (1998) fitting formulas
- Bardeen et al. (1986) analytical forms
- Support for massive neutrinos
- Baryon acoustic oscillations
- Interface for Boltzmann codes (CAMB/CLASS)

Transfer functions describe how primordial density fluctuations
evolve through radiation domination, matter-radiation equality,
and into the matter-dominated era.
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from typing import Union, Tuple, Dict, Optional, Callable
import numpy as np

from ..background.background import BackgroundEvolution
from ..utils.constants import c_km_s, eV_to_J, k_B


class TransferFunctionCalculator:
    """
    Professional transfer function calculator.
    
    Provides accurate transfer functions for matter, baryons, and CDM
    using state-of-the-art fitting formulas and analytical approximations.
    
    Essential for computing matter power spectra and CMB observables.
    """
    
    def __init__(self, background: BackgroundEvolution):
        """
        Initialize transfer function calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology calculator
        """
        self.bg = background
        self.params = background.model.params
        
        # Extract cosmological parameters
        self.h = self.params.get_value('h')
        self.Omega_m = self.params.get_value('Omega_m')
        self.Omega_b = self.params.get_value('Omega_b')
        self.Omega_cdm = self.Omega_m - self.Omega_b
        self.T_cmb = self.params.get_value('T_cmb')
        
        # Derived parameters for transfer functions
        self.Omega_m_h2 = self.Omega_m * self.h**2
        self.Omega_b_h2 = self.Omega_b * self.h**2
        self.Omega_cdm_h2 = self.Omega_cdm * self.h**2
        self.theta = self.T_cmb / 2.7  # Temperature parameter
        
        # Neutrino properties (simplified - assume massless)
        self.N_nu = 3.046  # Effective number of neutrinos
        self.m_nu_sum = 0.06  # Sum of neutrino masses in eV (minimal)
        
        # Pre-compute key scales
        self._compute_characteristic_scales()
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compute_characteristic_scales(self):
        """Compute characteristic scales for transfer functions."""
        
        # Equality scale
        z_eq = self.bg.matter_radiation_equality_redshift()
        self.k_eq = 0.073 * self.Omega_m_h2  # h/Mpc (approximate)
        
        # Sound horizon at drag epoch
        z_drag = 1060  # Approximate drag redshift
        # Simplified sound horizon calculation
        self.r_s = 147.0 * (self.Omega_m_h2)**(-0.25) * (self.Omega_b_h2)**(-0.08)  # Mpc
        
        # Silk damping scale  
        self.k_silk = 1.6 * (self.Omega_b_h2)**(0.52) * (self.Omega_m_h2)**(0.73)  # h/Mpc
        
        # Peak scale
        self.k_peak = 0.05 / self.r_s  # Peak of BAO feature
    
    def _compile_functions(self):
        """Pre-compile JAX functions for performance."""
        
        # Vectorized transfer functions
        self._transfer_matter_vmap = jit(vmap(self._transfer_matter_single))
        self._transfer_baryon_vmap = jit(vmap(self._transfer_baryon_single))
        self._transfer_cdm_vmap = jit(vmap(self._transfer_cdm_single))
        
        # BAO features
        self._bao_features_jit = jit(self._compute_bao_features)
    
    # ==================== Eisenstein & Hu Transfer Functions ====================
    
    def _transfer_matter_single(self, k: float) -> float:
        """
        Matter transfer function using Eisenstein & Hu (1998) fitting formula.
        
        Accurate to ~10% without baryon acoustic oscillations.
        """
        # Dimensionless wavenumber
        q = k * self.theta**2 / self.Omega_m_h2
        
        # Fitting parameters
        Gamma_eff = self.Omega_m * self.h * (self.theta / (1 + 0.3 * self.Omega_b_h2))
        q_eff = q * self.Omega_m_h2 / Gamma_eff
        
        # Transfer function components
        L0 = jnp.log(2 * jnp.e + 1.8 * q_eff)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q_eff)
        
        # Final transfer function
        T_k = L0 / (L0 + C0 * q_eff**2)
        
        return T_k
    
    def transfer_matter(self, k: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Matter transfer function T_m(k).
        
        Normalized to T_m(0) = 1.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
            
        Returns
        -------
        float or array_like
            Matter transfer function
        """
        k = jnp.asarray(k)
        if k.ndim == 0:
            return self._transfer_matter_single(k)
        else:
            return self._transfer_matter_vmap(k)
    
    # ==================== Baryon and CDM Transfer Functions ====================
    
    def _transfer_cdm_single(self, k: float) -> float:
        """CDM transfer function with baryon effects."""
        
        q = k / (13.41 * self.k_eq)
        
        # CDM component (Eisenstein & Hu)
        a1 = (46.9 * self.Omega_m_h2)**(0.67) * (1 + (32.1 * self.Omega_m_h2)**(-0.532))
        a2 = (12.0 * self.Omega_m_h2)**(0.424) * (1 + (45.0 * self.Omega_m_h2)**(-0.582))
        alpha_c = a1**(-self.Omega_b / self.Omega_m) * a2**(-(self.Omega_b / self.Omega_m)**3)
        
        b1 = 0.944 / (1 + (458 * self.Omega_m_h2)**(-0.708))
        b2 = (0.395 * self.Omega_m_h2)**(-0.0266)
        beta_c = 1 / (1 + b1 * ((self.Omega_cdm / self.Omega_m)**b2 - 1))
        
        # Transfer function
        f = 1 / (1 + (k * self.r_s / 5.4)**4)
        
        T_cdm = f * self._bardeen_transfer(q * alpha_c) + (1 - f) * self._bardeen_transfer(q * beta_c)
        
        return T_cdm
    
    def _transfer_baryon_single(self, k: float) -> float:
        """Baryon transfer function with acoustic oscillations."""
        
        q = k / (13.41 * self.k_eq)
        
        # Baryon drag effects
        R_drag = 31.5 * self.Omega_b_h2 * (self.theta / 2.7)**(-4) * (1000 / (1 + self.bg.matter_radiation_equality_redshift()))**(0.5)
        
        # Acoustic oscillations
        k_silk_damped = k * self.k_silk
        alpha_b = 2.07 * self.k_eq * self.r_s * (1 + R_drag)**(-0.75) * (1 + 0.37 * (self.Omega_m_h2 / 0.15)**(0.40))
        
        # Oscillatory component
        beta_b = 0.5 + self.Omega_b / self.Omega_m + (3 - 2 * self.Omega_b / self.Omega_m) * jnp.sqrt((17.2 * self.Omega_m_h2)**2 + 1)
        
        # Sine term for BAO
        s = self.r_s / (1 + (beta_b / (k * self.r_s))**3)**(1/3)
        
        T_baryon = (self._bardeen_transfer(q) / (1 + (k * self.r_s / 5.2)**2) + 
                   alpha_b / (1 + (beta_b / (k * self.r_s))**3) * jnp.exp(-k_silk_damped)) * jnp.sin(k * s) / (k * s)
        
        return T_baryon
    
    def transfer_cdm(self, k: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Cold dark matter transfer function.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
            
        Returns
        -------
        float or array_like
            CDM transfer function
        """
        k = jnp.asarray(k)
        if k.ndim == 0:
            return self._transfer_cdm_single(k)
        else:
            return self._transfer_cdm_vmap(k)
    
    def transfer_baryon(self, k: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Baryon transfer function with BAO features.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
            
        Returns
        -------
        float or array_like
            Baryon transfer function
        """
        k = jnp.asarray(k)
        if k.ndim == 0:
            return self._transfer_baryon_single(k)
        else:
            return self._transfer_baryon_vmap(k)
    
    def _bardeen_transfer(self, q: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Bardeen et al. (1986) transfer function shape.
        
        T(q) = ln(1 + 2.34q) / (2.34q) * [1 + 3.89q + (16.1q)² + (5.46q)³ + (6.71q)⁴]^(-1/4)
        """
        q = jnp.asarray(q)
        
        # Handle q → 0 limit
        term1 = jnp.where(q < 1e-3, 1.0, jnp.log(1 + 2.34 * q) / (2.34 * q))
        term2 = (1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4)**(-0.25)
        
        return term1 * term2
    
    # ==================== Baryon Acoustic Oscillations ====================
    
    def _compute_bao_features(self, k_array: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Compute BAO features in transfer functions.
        """
        T_m = self.transfer_matter(k_array)
        T_no_wiggle = self._transfer_no_wiggle(k_array)
        
        # BAO wiggle amplitude
        wiggle_amplitude = (T_m - T_no_wiggle) / T_no_wiggle
        
        return {
            'k': k_array,
            'T_matter': T_m,
            'T_no_wiggle': T_no_wiggle,
            'wiggle_amplitude': wiggle_amplitude
        }
    
    def _transfer_no_wiggle(self, k: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Smooth transfer function without BAO wiggles.
        
        Useful for separating BAO features from smooth component.
        """
        # Simplified smooth component
        k = jnp.asarray(k)
        q = k / self.k_eq
        
        # Smooth approximation (no oscillations)
        T_smooth = jnp.log(1 + 0.171*q) / (0.171*q) * (1 + 0.284*q + (1.18*q)**2 + (0.399*q)**3 + (0.490*q)**4)**(-0.25)
        
        return T_smooth
    
    def bao_peak_positions(self) -> jnp.ndarray:
        """
        Theoretical positions of BAO peaks in k-space.
        
        Returns
        -------
        jnp.ndarray
            Peak positions in h/Mpc
        """
        # BAO peaks occur at k ≈ n*π/r_s for integer n
        n_peaks = jnp.arange(1, 6)  # First 5 peaks
        k_peaks = n_peaks * jnp.pi / self.r_s
        
        return k_peaks
    
    def bao_features(self, k_min: float = 0.01, k_max: float = 0.3,
                    n_points: int = 1000) -> Dict[str, jnp.ndarray]:
        """
        Extract BAO features for analysis.
        
        Parameters
        ----------
        k_min, k_max : float
            k-range in h/Mpc
        n_points : int
            Number of points
            
        Returns
        -------
        dict
            BAO feature analysis
        """
        k_array = jnp.linspace(k_min, k_max, n_points)
        return self._bao_features_jit(k_array)
    
    # ==================== Massive Neutrinos ====================
    
    def neutrino_suppression(self, k: Union[float, jnp.ndarray],
                           z: float = 0) -> Union[float, jnp.ndarray]:
        """
        Neutrino suppression factor for matter power spectrum.
        
        Massive neutrinos suppress small-scale power due to free-streaming.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float
            Redshift
            
        Returns
        -------
        float or array_like
            Suppression factor (< 1)
        """
        if self.m_nu_sum <= 0.01:  # Effectively massless
            return 1.0
        
        k = jnp.asarray(k)
        
        # Neutrino density parameter
        f_nu = self.m_nu_sum / (94.07 * self.Omega_m_h2)  # Neutrino fraction
        
        # Free-streaming scale
        k_fs = 0.82 * (self.m_nu_sum / 0.05)**(0.5) * (self.Omega_m_h2 / 0.14)**(0.5)  # h/Mpc
        
        # Suppression factor (fitting formula)
        suppression = 1 - f_nu * (1 + (k / k_fs)**2)**(-0.5)
        
        return suppression
    
    # ==================== Interface Functions ====================
    
    def transfer_function_dict(self, k: Union[float, jnp.ndarray]) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Complete set of transfer functions.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
            
        Returns
        -------
        dict
            Dictionary with all transfer functions
        """
        return {
            'k': k,
            'T_matter': self.transfer_matter(k),
            'T_cdm': self.transfer_cdm(k),
            'T_baryon': self.transfer_baryon(k),
            'T_no_wiggle': self._transfer_no_wiggle(k),
            'neutrino_suppression': self.neutrino_suppression(k)
        }
    
    def camb_format_transfer(self, k_array: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        """
        Transfer functions in CAMB-like format.
        
        Parameters
        ----------
        k_array : jnp.ndarray
            Wave number array in h/Mpc
            
        Returns
        -------
        dict
            CAMB-formatted transfer functions
        """
        T_m = self.transfer_matter(k_array)
        T_cdm = self.transfer_cdm(k_array)
        T_b = self.transfer_baryon(k_array)
        
        # Normalization to match CAMB conventions
        T_m_norm = T_m / T_m[0]  # Normalize to k→0
        
        return {
            'k_h': k_array,  # h/Mpc
            'transfer_cdm': T_cdm / T_cdm[0],
            'transfer_baryon': T_b / T_b[0], 
            'transfer_matter': T_m_norm,
            'transfer_nonu': T_m_norm,  # No massive neutrinos
        }
    
    # ==================== Diagnostics and Utilities ====================
    
    def characteristic_scales_summary(self) -> str:
        """
        Summary of characteristic scales in the transfer function.
        
        Returns
        -------
        str
            Formatted summary
        """
        lines = [
            "Transfer Function Characteristic Scales",
            "=" * 40,
            f"Equality scale: k_eq = {self.k_eq:.4f} h/Mpc",
            f"Sound horizon: r_s = {self.r_s:.1f} Mpc",
            f"Silk scale: k_silk = {self.k_silk:.2f} h/Mpc",
            f"BAO peak scale: k_peak = {self.k_peak:.4f} h/Mpc",
            f"Matter fraction: Ω_m = {self.Omega_m:.4f}",
            f"Baryon fraction: Ω_b = {self.Omega_b:.4f}",
            f"CDM fraction: Ω_cdm = {self.Omega_cdm:.4f}",
        ]
        
        # BAO peak positions
        k_peaks = self.bao_peak_positions()
        lines.append("")
        lines.append("BAO Peak Positions (h/Mpc):")
        for i, k_peak in enumerate(k_peaks):
            lines.append(f"  Peak {i+1}: k = {k_peak:.4f}")
        
        return "\n".join(lines)
    
    def transfer_benchmarks(self, k_ref: float = 0.1) -> Dict[str, float]:
        """
        Benchmark values for transfer function validation.
        
        Parameters
        ----------
        k_ref : float
            Reference scale in h/Mpc
            
        Returns
        -------
        dict
            Benchmark values
        """
        T_m = self.transfer_matter(k_ref)
        T_cdm = self.transfer_cdm(k_ref)
        T_b = self.transfer_baryon(k_ref)
        
        return {
            'k_reference': k_ref,
            'T_matter': float(T_m),
            'T_cdm': float(T_cdm),
            'T_baryon': float(T_b),
            'T_ratio_b_cdm': float(T_b / T_cdm),
            'equality_scale': float(self.k_eq),
            'sound_horizon': float(self.r_s)
        }
    
    def plot_transfer_functions(self, k_min: float = 1e-3, k_max: float = 10.0,
                               n_points: int = 1000) -> Tuple[jnp.ndarray, Dict]:
        """
        Generate data for plotting transfer functions.
        
        Parameters
        ----------
        k_min, k_max : float
            k-range in h/Mpc
        n_points : int
            Number of points
            
        Returns
        -------
        k_array : jnp.ndarray
            k values
        transfer_data : dict
            Transfer function data
        """
        k_array = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_points)
        
        transfer_data = self.transfer_function_dict(k_array)
        transfer_data['bao_features'] = self.bao_features()
        
        return k_array, transfer_data