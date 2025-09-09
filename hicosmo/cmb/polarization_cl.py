"""
CMB Polarization Power Spectrum Calculator
==========================================

Professional implementation of CMB polarization anisotropy calculations.
Computes E-mode, B-mode, and temperature-polarization cross-correlation spectra.

Key features:
- E-mode polarization from Thomson scattering
- B-mode polarization from gravitational waves (tensor perturbations)
- Temperature-E mode cross-correlations (C_l^TE)
- Reionization bump modeling
- Lensing-induced B-modes

Polarization observables provide:
- Independent tests of cosmological models
- Constraints on inflation (tensor-to-scalar ratio r)
- Reionization physics
- Gravitational lensing reconstruction
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.scipy.special import spherical_jn
from functools import partial
from typing import Union, Tuple, Dict, Optional, Callable
import numpy as np

from ..background.background import BackgroundEvolution
from ..background.thermodynamics import ThermodynamicsCalculator
from ..perturbations.linear_perturbations import LinearPerturbations
from .temperature_cl import TemperaturePowerSpectrum


class PolarizationPowerSpectrum:
    """
    CMB polarization power spectrum calculator.
    
    Computes polarization anisotropies from:
    - Thomson scattering during recombination
    - Tensor perturbations (gravitational waves)
    - Reionization effects  
    - Gravitational lensing
    
    Provides accurate predictions for E-modes, B-modes, and TE correlations.
    """
    
    def __init__(self, background: BackgroundEvolution,
                 thermodynamics: ThermodynamicsCalculator,
                 perturbations: LinearPerturbations,
                 temperature_cl: TemperaturePowerSpectrum):
        """
        Initialize polarization calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology
        thermodynamics : ThermodynamicsCalculator
            Recombination physics
        perturbations : LinearPerturbations
            Linear perturbation theory
        temperature_cl : TemperaturePowerSpectrum
            Temperature anisotropy calculator
        """
        self.bg = background
        self.thermo = thermodynamics
        self.pert = perturbations
        self.temp_cl = temperature_cl
        self.params = background.model.params
        
        # Polarization parameters
        self.tau_reio = self.params.get_value('tau_reio')
        self.T_cmb = self.params.get_value('T_cmb')
        
        # Tensor parameters (primordial gravitational waves)
        self.r_tensor = 0.0  # Tensor-to-scalar ratio (can be set by user)
        
        # Recombination parameters from temperature calculator
        self.z_star = self.temp_cl.z_star
        self.eta_star = self.temp_cl.eta_star
        self.D_A_star = self.temp_cl.D_A_star
        self.theta_s = self.temp_cl.theta_s
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile JAX functions for performance."""
        
        # Vectorized polarization calculations
        self._cl_EE_vmap = jit(vmap(self._cl_EE_single))
        self._cl_BB_vmap = jit(vmap(self._cl_BB_single))
        self._cl_TE_vmap = jit(vmap(self._cl_TE_single))
        
        # Transfer functions
        self._polarization_transfer_jit = jit(self._polarization_transfer_function)
        
        # Tensor modes
        self._tensor_transfer_jit = jit(self._tensor_transfer_function)
    
    # ==================== E-Mode Polarization ====================
    
    def _polarization_transfer_function(self, k: float, l: int, mode: str = 'E') -> float:
        """
        Polarization transfer function for E and B modes.
        
        Polarization arises from Thomson scattering of quadrupole anisotropy.
        
        Parameters
        ----------
        k : float
            Wave number in Mpc^-1
        l : int
            Multipole moment
        mode : str
            'E' for E-mode, 'B' for B-mode
            
        Returns
        -------
        float
            Transfer function amplitude
        """
        if l < 2:
            return 0.0
        
        # Comoving distance to last scattering
        chi_star = self.bg.distances.comoving_distance(self.z_star)
        x = k * chi_star
        
        # Spherical Bessel functions
        if l < 100:
            j_l = spherical_jn(l, x)
            j_l_minus2 = spherical_jn(max(0, l-2), x) if l >= 2 else 0.0
            j_l_plus2 = spherical_jn(l+2, x)
        else:
            # Asymptotic approximations for large l
            phase_l = x - l*jnp.pi/2 - jnp.pi/4
            j_l = jnp.sqrt(2/(jnp.pi*x)) * jnp.cos(phase_l)
            
            phase_minus2 = x - (l-2)*jnp.pi/2 - jnp.pi/4
            j_l_minus2 = jnp.sqrt(2/(jnp.pi*x)) * jnp.cos(phase_minus2) if l >= 2 else 0.0
            
            phase_plus2 = x - (l+2)*jnp.pi/2 - jnp.pi/4
            j_l_plus2 = jnp.sqrt(2/(jnp.pi*x)) * jnp.cos(phase_plus2)
        
        # Polarization coupling coefficients
        # E-mode couples to quadrupole (l±2 terms)
        if mode == 'E':
            # E-mode transfer function
            coupling = jnp.sqrt((l-1)*l*(l+1)*(l+2)) / (2*l + 1)
            T_pol = coupling * (j_l_minus2 - j_l_plus2) / 4
            
            # Include Thomson scattering and visibility
            visibility_factor = self.thermo.visibility_function(self.z_star)
            T_pol *= visibility_factor
            
        elif mode == 'B':
            # B-mode from scalar perturbations is zero (parity)
            # B-modes only from tensors or lensing
            T_pol = 0.0
        else:
            T_pol = 0.0
        
        return T_pol
    
    def _cl_EE_single(self, l: int) -> float:
        """
        E-mode polarization power spectrum C_l^EE for single multipole.
        """
        if l < 2:
            return 0.0
        
        # Integration over k
        k_min, k_max = 1e-4, 1.0
        n_k = 200
        k_array = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_k)
        
        # Primordial scalar power spectrum
        P_R = self.temp_cl.A_s * (k_array / self.temp_cl.k_pivot)**(self.temp_cl.n_s - 1)
        
        # E-mode transfer function
        T_E_array = vmap(lambda k: self._polarization_transfer_jit(k, l, 'E'))(k_array)
        
        # Integrand
        integrand = P_R * T_E_array**2 * k_array**2
        
        # Integration
        d_ln_k = jnp.log(k_array[1] / k_array[0])
        C_l_EE = jnp.sum(integrand * k_array * d_ln_k) / jnp.pi
        
        # Convert to μK² units
        C_l_EE_microK2 = C_l_EE * (self.T_cmb * 1e6)**2
        
        # Add reionization bump
        C_l_EE_reio = self._reionization_EE_contribution(l)
        C_l_EE_microK2 += C_l_EE_reio
        
        return C_l_EE_microK2
    
    def _reionization_EE_contribution(self, l: int) -> float:
        """
        Reionization contribution to E-mode polarization.
        
        Creates characteristic bump at l ~ 10-30.
        """
        if l < 2 or l > 100:
            return 0.0
        
        # Reionization visibility function
        tau = self.tau_reio
        z_reio = 8.0  # Approximate reionization redshift
        
        # Reionization bump amplitude and shape
        l_peak = 20.0
        amplitude = 10.0 * tau**2  # μK²
        width = 15.0
        
        # Gaussian-like bump
        C_l_reio = amplitude * jnp.exp(-(l - l_peak)**2 / (2 * width**2))
        
        return C_l_reio
    
    def EE_power_spectrum(self, l_array: Union[int, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        E-mode polarization power spectrum C_l^EE in μK².
        
        Parameters
        ----------
        l_array : int or array_like
            Multipole moment(s)
            
        Returns
        -------
        float or array_like
            E-mode power spectrum in μK²
        """
        l_array = jnp.asarray(l_array, dtype=int)
        
        if l_array.ndim == 0:
            return self._cl_EE_single(l_array)
        else:
            return self._cl_EE_vmap(l_array)
    
    # ==================== B-Mode Polarization ====================
    
    def _tensor_transfer_function(self, k: float, l: int) -> Tuple[float, float]:
        """
        Tensor transfer functions for E and B modes from gravitational waves.
        
        Returns both E and B mode contributions from tensor perturbations.
        """
        if l < 2:
            return 0.0, 0.0
        
        # Tensor perturbations have different evolution than scalars
        chi_star = self.bg.distances.comoving_distance(self.z_star)
        x = k * chi_star
        
        # Simplified tensor transfer (full calculation requires tensor evolution)
        if l < 100:
            j_l = spherical_jn(l, x)
            j_l_minus2 = spherical_jn(max(0, l-2), x) if l >= 2 else 0.0
            j_l_plus2 = spherical_jn(l+2, x)
        else:
            # Asymptotic forms
            phase_l = x - l*jnp.pi/2 - jnp.pi/4
            j_l = jnp.sqrt(2/(jnp.pi*x)) * jnp.cos(phase_l)
            j_l_minus2 = jnp.sqrt(2/(jnp.pi*x)) * jnp.cos(x - (l-2)*jnp.pi/2 - jnp.pi/4) if l >= 2 else 0.0
            j_l_plus2 = jnp.sqrt(2/(jnp.pi*x)) * jnp.cos(x - (l+2)*jnp.pi/2 - jnp.pi/4)
        
        # Tensor coupling to E and B modes
        coupling = jnp.sqrt((l-1)*l*(l+1)*(l+2)) / (2*l + 1)
        
        # Both E and B get contributions from tensors (unlike scalars)
        T_E_tensor = coupling * (j_l_minus2 + j_l_plus2) / 4
        T_B_tensor = coupling * (j_l_minus2 - j_l_plus2) / 4
        
        # Include tensor evolution and visibility
        visibility = self.thermo.visibility_function(self.z_star)
        
        T_E_tensor *= visibility
        T_B_tensor *= visibility
        
        return T_E_tensor, T_B_tensor
    
    def _cl_BB_single(self, l: int) -> float:
        """
        B-mode polarization power spectrum C_l^BB for single multipole.
        
        Includes contributions from:
        1. Tensor perturbations (primordial gravitational waves)
        2. Gravitational lensing (conversion of E→B)
        """
        if l < 2:
            return 0.0
        
        C_l_BB_total = 0.0
        
        # 1. Tensor contribution (primordial B-modes)
        if self.r_tensor > 0:
            C_l_BB_tensor = self._tensor_BB_contribution(l)
            C_l_BB_total += C_l_BB_tensor
        
        # 2. Lensing-induced B-modes
        C_l_BB_lensing = self._lensing_BB_contribution(l)
        C_l_BB_total += C_l_BB_lensing
        
        return C_l_BB_total
    
    def _tensor_BB_contribution(self, l: int) -> float:
        """
        Tensor contribution to B-mode power spectrum.
        """
        # Integration over k  
        k_min, k_max = 1e-4, 1.0
        n_k = 200
        k_array = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_k)
        
        # Primordial tensor power spectrum
        # P_T(k) = r * P_R(k) where r is tensor-to-scalar ratio
        P_T = self.r_tensor * self.temp_cl.A_s * (k_array / self.temp_cl.k_pivot)**(self.temp_cl.n_s - 1)
        
        # Tensor B-mode transfer function
        T_B_tensor_array = vmap(lambda k: self._tensor_transfer_jit(k, l)[1])(k_array)
        
        # Integrand
        integrand = P_T * T_B_tensor_array**2 * k_array**2
        
        # Integration
        d_ln_k = jnp.log(k_array[1] / k_array[0])
        C_l_BB_tensor = jnp.sum(integrand * k_array * d_ln_k) / jnp.pi
        
        # Convert to μK²
        return C_l_BB_tensor * (self.T_cmb * 1e6)**2
    
    def _lensing_BB_contribution(self, l: int) -> float:
        """
        Lensing-induced B-mode contribution.
        
        Gravitational lensing converts E-modes to B-modes.
        """
        if l < 10 or l > 3000:
            return 0.0
        
        # Simplified lensing B-mode calculation
        # Full calculation requires lensing potential power spectrum
        
        # E-mode power at this l
        C_l_EE = self.EE_power_spectrum(l)
        
        # Lensing conversion efficiency (approximate)
        # Peaks around l ~ 1000-2000
        l_peak_lensing = 1500.0
        efficiency = 0.01 * (l / l_peak_lensing) * jnp.exp(-(l / l_peak_lensing - 1)**2)
        
        C_l_BB_lensing = efficiency * C_l_EE
        
        return C_l_BB_lensing
    
    def BB_power_spectrum(self, l_array: Union[int, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        B-mode polarization power spectrum C_l^BB in μK².
        
        Parameters
        ----------
        l_array : int or array_like
            Multipole moment(s)
            
        Returns
        -------
        float or array_like
            B-mode power spectrum in μK²
        """
        l_array = jnp.asarray(l_array, dtype=int)
        
        if l_array.ndim == 0:
            return self._cl_BB_single(l_array)
        else:
            return self._cl_BB_vmap(l_array)
    
    # ==================== Temperature-Polarization Cross-Correlation ====================
    
    def _cl_TE_single(self, l: int) -> float:
        """
        Temperature-E mode cross-correlation C_l^TE for single multipole.
        """
        if l < 2:
            return 0.0
        
        # Integration over k
        k_min, k_max = 1e-4, 1.0
        n_k = 200
        k_array = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_k)
        
        # Primordial power spectrum
        P_R = self.temp_cl.A_s * (k_array / self.temp_cl.k_pivot)**(self.temp_cl.n_s - 1)
        
        # Transfer functions
        T_T_array = vmap(lambda k: self.temp_cl._photon_transfer_jit(k, l))(k_array)
        T_E_array = vmap(lambda k: self._polarization_transfer_jit(k, l, 'E'))(k_array)
        
        # Cross-correlation integrand
        integrand = P_R * T_T_array * T_E_array * k_array**2
        
        # Integration
        d_ln_k = jnp.log(k_array[1] / k_array[0])
        C_l_TE = jnp.sum(integrand * k_array * d_ln_k) / jnp.pi
        
        # Convert to μK²
        C_l_TE_microK2 = C_l_TE * (self.T_cmb * 1e6)**2
        
        return C_l_TE_microK2
    
    def TE_cross_spectrum(self, l_array: Union[int, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Temperature-E mode cross power spectrum C_l^TE in μK².
        
        Parameters
        ----------
        l_array : int or array_like
            Multipole moment(s)
            
        Returns
        -------
        float or array_like
            TE cross power spectrum in μK²
        """
        l_array = jnp.asarray(l_array, dtype=int)
        
        if l_array.ndim == 0:
            return self._cl_TE_single(l_array)
        else:
            return self._cl_TE_vmap(l_array)
    
    # ==================== Tensor Mode Analysis ====================
    
    def set_tensor_to_scalar_ratio(self, r: float):
        """
        Set tensor-to-scalar ratio for B-mode calculations.
        
        Parameters
        ----------
        r : float
            Tensor-to-scalar ratio (r = 0 means no primordial B-modes)
        """
        self.r_tensor = r
    
    def tensor_contributions(self, l_array: jnp.ndarray, r: float = 0.1) -> Dict[str, jnp.ndarray]:
        """
        Compute tensor contributions to all polarization spectra.
        
        Parameters
        ----------
        l_array : jnp.ndarray
            Multipole array
        r : float
            Tensor-to-scalar ratio
            
        Returns
        -------
        dict
            Tensor contributions to TT, EE, BB, TE
        """
        # Temporarily set r
        r_old = self.r_tensor
        self.r_tensor = r
        
        # Compute tensor contributions
        # (This is simplified - full calculation would separate tensor/scalar)
        
        C_l_BB_tensor = self.BB_power_spectrum(l_array)
        
        # Tensor also contributes to TT and EE (though typically small)
        C_l_TT_tensor = 0.1 * C_l_BB_tensor  # Rough approximation
        C_l_EE_tensor = 0.5 * C_l_BB_tensor  # Rough approximation
        C_l_TE_tensor = jnp.zeros_like(l_array, dtype=float)  # Parity forbids scalar-tensor TE
        
        # Restore original r
        self.r_tensor = r_old
        
        return {
            'TT_tensor': C_l_TT_tensor,
            'EE_tensor': C_l_EE_tensor, 
            'BB_tensor': C_l_BB_tensor,
            'TE_tensor': C_l_TE_tensor
        }
    
    # ==================== Utility Methods ====================
    
    def polarization_summary(self, r_tensor: float = 0.0) -> str:
        """
        Generate comprehensive polarization summary.
        
        Parameters
        ----------
        r_tensor : float
            Tensor-to-scalar ratio for B-modes
            
        Returns
        -------
        str
            Formatted summary
        """
        # Set tensor ratio temporarily
        r_old = self.r_tensor
        self.r_tensor = r_tensor
        
        # Sample calculations
        l_sample = jnp.array([2, 10, 100, 1000])
        C_l_EE = self.EE_power_spectrum(l_sample)
        C_l_BB = self.BB_power_spectrum(l_sample)
        C_l_TE = self.TE_cross_spectrum(l_sample)
        
        # Reionization peak
        l_reio_peak = 20
        C_l_EE_reio = self.EE_power_spectrum(l_reio_peak)
        
        # Restore original tensor ratio
        self.r_tensor = r_old
        
        lines = [
            "CMB Polarization Summary",
            "=" * 30,
            f"Reionization optical depth: τ = {self.tau_reio:.4f}",
            f"Tensor-to-scalar ratio: r = {r_tensor:.3f}",
            "",
            "E-mode Power Spectrum C_l^EE (μK²):",
            f"  l=2:    {C_l_EE[0]:.2f}",
            f"  l=10:   {C_l_EE[1]:.2f} (reionization bump)",
            f"  l=100:  {C_l_EE[2]:.2f}",
            f"  l=1000: {C_l_EE[3]:.3f}",
            "",
            "B-mode Power Spectrum C_l^BB (μK²):",
            f"  l=2:    {C_l_BB[0]:.4f}",
            f"  l=10:   {C_l_BB[1]:.4f}",
            f"  l=100:  {C_l_BB[2]:.4f}",
            f"  l=1000: {C_l_BB[3]:.4f} (lensing)",
            "",
            "TE Cross-Spectrum C_l^TE (μK²):",
            f"  l=2:    {C_l_TE[0]:.2f}",
            f"  l=100:  {C_l_TE[2]:.2f}",
            f"  l=1000: {C_l_TE[3]:.3f}",
        ]
        
        return "\n".join(lines)
    
    def plot_polarization_spectra(self, l_min: int = 2, l_max: int = 2000,
                                r_tensor: float = 0.0) -> Tuple[jnp.ndarray, Dict]:
        """
        Generate data for plotting all polarization spectra.
        
        Parameters
        ----------
        l_min, l_max : int
            Multipole range
        r_tensor : float
            Tensor-to-scalar ratio
            
        Returns
        -------
        l_array : jnp.ndarray
            Multipole values
        spectra : dict
            Dictionary with all polarization spectra
        """
        # Set tensor ratio
        r_old = self.r_tensor
        self.r_tensor = r_tensor
        
        # Multipole array
        n_points = min(300, l_max - l_min + 1)
        l_array = jnp.logspace(jnp.log10(l_min), jnp.log10(l_max), n_points).astype(int)
        l_array = jnp.unique(l_array)
        
        # Compute all spectra
        spectra = {
            'EE': self.EE_power_spectrum(l_array),
            'BB': self.BB_power_spectrum(l_array),
            'TE': self.TE_cross_spectrum(l_array)
        }
        
        # Restore tensor ratio
        self.r_tensor = r_old
        
        return l_array, spectra
    
    def benchmark_polarization(self) -> Dict[str, float]:
        """
        Generate benchmark polarization values.
        
        Returns
        -------
        dict
            Benchmark quantities
        """
        # Standard multipoles
        l_low = 10   # Reionization bump
        l_mid = 100  # Acoustic peak region  
        l_high = 1000 # Small scales
        
        return {
            'C_l_EE_at_l_10': float(self.EE_power_spectrum(l_low)),
            'C_l_EE_at_l_100': float(self.EE_power_spectrum(l_mid)),
            'C_l_BB_at_l_1000': float(self.BB_power_spectrum(l_high)),
            'C_l_TE_at_l_100': float(self.TE_cross_spectrum(l_mid)),
            'reionization_tau': float(self.tau_reio),
            'tensor_to_scalar_ratio': float(self.r_tensor)
        }