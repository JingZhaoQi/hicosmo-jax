"""
CMB Lensing Power Spectrum Calculator
====================================

Professional implementation of CMB gravitational lensing calculations.
Computes lensing potential power spectrum and lensing effects on temperature/polarization.

Key features:
- Lensing potential power spectrum C_l^φφ
- Lensing reconstruction noise
- Lensing deflection field statistics
- Cross-correlations with large-scale structure
- Delensing for primordial B-mode searches

CMB lensing provides:
- Constraints on neutrino masses and dark energy
- Tests of general relativity on cosmological scales  
- Cross-correlation probes of dark matter
- Delensing for tensor mode detection
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from typing import Union, Tuple, Dict, Optional, Callable
import numpy as np

from ..background.background import BackgroundEvolution
from ..perturbations.linear_perturbations import LinearPerturbations
from ..powerspectrum.linear_power import LinearPowerSpectrum
from ..powerspectrum.nonlinear_corrections import NonlinearCorrections


class CMBLensingCalculator:
    """
    CMB lensing power spectrum calculator.
    
    Computes gravitational lensing effects on CMB temperature and polarization
    anisotropies from intervening large-scale structure.
    
    The lensing potential is related to the gravitational potential by:
    φ(n̂) = -2 ∫₀^η* dη (η*-η)/(η*η) Ψ(ηn̂, η)
    
    where η* is conformal time to last scattering.
    """
    
    def __init__(self, background: BackgroundEvolution,
                 perturbations: LinearPerturbations, 
                 linear_power: LinearPowerSpectrum,
                 nonlinear_power: NonlinearCorrections):
        """
        Initialize CMB lensing calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology
        perturbations : LinearPerturbations
            Linear perturbation theory
        linear_power : LinearPowerSpectrum
            Linear matter power spectrum
        nonlinear_power : NonlinearCorrections
            Nonlinear matter power spectrum
        """
        self.bg = background
        self.pert = perturbations
        self.linear = linear_power
        self.nonlinear = nonlinear_power
        self.params = background.model.params
        
        # CMB parameters
        self.z_star = 1089.9  # Last scattering redshift
        self.eta_star = self.bg.conformal_time(self.z_star)
        self.chi_star = self.bg.distances.comoving_distance(self.z_star)
        
        # Physical constants for lensing
        self.c_km_s = 299792.458
        self.H0 = self.bg.H0
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile JAX functions for performance."""
        
        # Lensing kernel integration
        self._lensing_kernel_jit = jit(self._lensing_kernel)
        
        # Vectorized lensing calculations
        self._cl_phiphi_vmap = jit(vmap(self._cl_phiphi_single))
        self._cl_deflection_vmap = jit(vmap(self._cl_deflection_single))
        
        # Reconstruction noise
        self._reconstruction_noise_jit = jit(self._reconstruction_noise_single)
    
    # ==================== Lensing Kernel ====================
    
    def _lensing_kernel(self, chi: float) -> float:
        """
        CMB lensing efficiency kernel W_φ(χ).
        
        W_φ(χ) = (3/2) Ω_m H₀² (χ*/χ) (χ*-χ) (1+z)
        
        Parameters
        ----------
        chi : float
            Comoving distance in Mpc
            
        Returns
        -------
        float
            Lensing kernel value
        """
        if chi >= self.chi_star or chi <= 0:
            return 0.0
        
        # Redshift at this distance
        z = self.bg.distances._invert_comoving_distance(chi)  # Approximate inversion
        
        # Lensing kernel
        Omega_m = self.params.get_value('Omega_m')
        H0_over_c = self.H0 / self.c_km_s  # (km/s/Mpc) / (km/s) = 1/Mpc
        
        kernel = (1.5 * Omega_m * H0_over_c**2 * 
                 (self.chi_star / chi) * (self.chi_star - chi) * (1 + z))
        
        return kernel
    
    def lensing_efficiency_function(self, z_array: jnp.ndarray) -> jnp.ndarray:
        """
        Lensing efficiency as function of redshift.
        
        Parameters
        ----------
        z_array : jnp.ndarray
            Redshift array
            
        Returns
        -------
        jnp.ndarray
            Lensing efficiency W_φ(z)
        """
        # Convert redshifts to comoving distances
        chi_array = vmap(self.bg.distances.comoving_distance)(z_array)
        
        # Compute lensing kernel
        kernel_array = vmap(self._lensing_kernel_jit)(chi_array)
        
        return kernel_array
    
    # ==================== Lensing Potential Power Spectrum ====================
    
    def _cl_phiphi_single(self, l: int) -> float:
        """
        Lensing potential power spectrum C_l^φφ for single multipole.
        
        C_l^φφ = ∫ dχ W_φ²(χ) P_Ψ(l/χ, z(χ)) / χ²
        
        where P_Ψ = (9/16) Ω_m² H₀⁴ P_δ is the gravitational potential power spectrum.
        """
        if l < 2:
            return 0.0
        
        # Integration over comoving distance
        chi_min = 10.0   # Mpc (avoid z → ∞)
        chi_max = self.chi_star
        n_chi = 100
        chi_array = jnp.linspace(chi_min, chi_max, n_chi)
        
        def integrand(chi):
            # Lensing kernel
            W_phi = self._lensing_kernel_jit(chi)
            
            # Wave number
            k = l / chi  # Limber approximation
            
            # Convert to h/Mpc units for power spectrum
            k_h = k * self.params.get_value('h')
            
            # Redshift at this distance
            z = self._chi_to_z_approx(chi)
            
            # Gravitational potential power spectrum
            # P_Ψ = (9/16) Ω_m² H₀⁴ (a²c²)⁻¹ P_δ = (9/4) Ω_m² H₀² (1+z)⁻² P_δ
            P_delta = self.nonlinear.nonlinear_power_spectrum(k_h, z)
            Omega_m = self.params.get_value('Omega_m')
            
            # Convert to Mpc units and include (1+z) factors
            P_psi = (2.25 * Omega_m**2 * (self.H0/self.c_km_s)**4 * 
                    (1 + z)**(-2) * P_delta)
            
            # Integrand
            return W_phi**2 * P_psi / chi**2
        
        # Numerical integration
        integrand_vals = vmap(integrand)(chi_array)
        C_l_phiphi = jnp.trapz(integrand_vals, chi_array)
        
        return C_l_phiphi
    
    def _chi_to_z_approx(self, chi: float) -> float:
        """Approximate conversion from comoving distance to redshift."""
        # Simple approximation for speed
        # More accurate would use root finding on comoving_distance(z) = chi
        return jnp.maximum(0.0, chi * self.H0 / self.c_km_s - 1.0)
    
    def lensing_potential_cl(self, l_array: Union[int, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        CMB lensing potential power spectrum C_l^φφ.
        
        Parameters
        ----------
        l_array : int or array_like
            Multipole moment(s)
            
        Returns
        -------
        float or array_like
            Lensing potential power spectrum
        """
        l_array = jnp.asarray(l_array, dtype=int)
        
        if l_array.ndim == 0:
            return self._cl_phiphi_single(l_array)
        else:
            return self._cl_phiphi_vmap(l_array)
    
    # ==================== Lensing Deflection Field ====================
    
    def _cl_deflection_single(self, l: int) -> float:
        """
        Lensing deflection power spectrum C_l^dd.
        
        Related to potential by: C_l^dd = l(l+1) C_l^φφ
        """
        if l < 2:
            return 0.0
        
        C_l_phiphi = self._cl_phiphi_single(l)
        C_l_dd = l * (l + 1) * C_l_phiphi
        
        return C_l_dd
    
    def lensing_deflection_cl(self, l_array: Union[int, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        CMB lensing deflection power spectrum C_l^dd.
        
        This is what's actually measured by lensing reconstruction.
        
        Parameters
        ----------
        l_array : int or array_like
            Multipole moment(s)
            
        Returns
        -------
        float or array_like
            Deflection power spectrum in (μrad)²
        """
        l_array = jnp.asarray(l_array, dtype=int)
        
        if l_array.ndim == 0:
            C_l_dd = self._cl_deflection_single(l_array)
        else:
            C_l_dd = self._cl_deflection_vmap(l_array)
        
        # Convert to microradians squared
        # φ is dimensionless, deflection α = ∇φ has units of angle
        C_l_dd_microrad2 = C_l_dd * (180 * 3600 * 1e6 / jnp.pi)**2
        
        return C_l_dd_microrad2
    
    # ==================== Lensing Reconstruction ====================
    
    def _reconstruction_noise_single(self, l: int, 
                                   C_l_TT: float, C_l_EE: float,
                                   noise_T: float = 10.0, noise_P: float = 14.14) -> float:
        """
        Lensing reconstruction noise N_l^φφ.
        
        Based on quadratic estimator reconstruction from temperature and polarization.
        
        Parameters
        ----------
        l : int
            Multipole
        C_l_TT, C_l_EE : float
            Temperature and E-mode power spectra
        noise_T, noise_P : float
            Instrument noise in μK-arcmin
            
        Returns
        -------
        float
            Reconstruction noise
        """
        if l < 2:
            return jnp.inf
        
        # Convert noise to power spectrum units
        # For white noise: N_l = (θ_beam * noise_level)²
        beam_fwhm_arcmin = 5.0  # Typical beam size
        beam_sigma = beam_fwhm_arcmin / (2 * jnp.sqrt(2 * jnp.log(2)))
        beam_sigma_rad = beam_sigma * jnp.pi / (180 * 60)
        
        # Beam and noise
        beam_factor = jnp.exp(l * (l + 1) * beam_sigma_rad**2)
        N_l_TT = (noise_T * jnp.pi / (180 * 60))**2 * beam_factor
        N_l_EE = (noise_P * jnp.pi / (180 * 60))**2 * beam_factor
        
        # Total observed power
        C_l_TT_obs = C_l_TT + N_l_TT
        C_l_EE_obs = C_l_EE + N_l_EE
        
        # Reconstruction noise (simplified quadratic estimator formula)
        # Full calculation requires detailed estimator analysis
        
        # Temperature reconstruction noise
        A_l_TT = l * (l + 1) / 2  # Estimator normalization
        N_l_phi_TT = A_l_TT / (C_l_TT_obs**2) if C_l_TT_obs > 0 else jnp.inf
        
        # Polarization reconstruction noise  
        A_l_EE = l * (l + 1) / 2
        N_l_phi_EE = A_l_EE / (C_l_EE_obs**2) if C_l_EE_obs > 0 else jnp.inf
        
        # Combined minimum variance estimator
        if N_l_phi_TT < jnp.inf and N_l_phi_EE < jnp.inf:
            N_l_phi = 1.0 / (1.0/N_l_phi_TT + 1.0/N_l_phi_EE)
        else:
            N_l_phi = min(N_l_phi_TT, N_l_phi_EE)
        
        return N_l_phi
    
    def reconstruction_noise_cl(self, l_array: jnp.ndarray,
                              C_l_TT: jnp.ndarray, C_l_EE: jnp.ndarray,
                              noise_T: float = 10.0, noise_P: float = 14.14) -> jnp.ndarray:
        """
        CMB lensing reconstruction noise power spectrum.
        
        Parameters
        ----------
        l_array : jnp.ndarray
            Multipole array
        C_l_TT, C_l_EE : jnp.ndarray
            CMB power spectra
        noise_T, noise_P : float
            Noise levels in μK-arcmin
            
        Returns
        -------
        jnp.ndarray
            Reconstruction noise N_l^φφ
        """
        def compute_noise_single(i):
            l = l_array[i]
            return self._reconstruction_noise_jit(l, C_l_TT[i], C_l_EE[i], noise_T, noise_P)
        
        N_l_array = vmap(compute_noise_single)(jnp.arange(len(l_array)))
        
        return N_l_array
    
    # ==================== Cross-Correlations ====================
    
    def cmb_lensing_galaxy_cross_cl(self, l_array: jnp.ndarray,
                                   z_galaxy: float,
                                   galaxy_bias: float = 1.0) -> jnp.ndarray:
        """
        Cross-correlation between CMB lensing and galaxy density.
        
        C_l^φg = ∫ dχ W_φ(χ) W_g(χ) P_δ(l/χ, z(χ)) / χ²
        
        Parameters
        ----------
        l_array : jnp.ndarray
            Multipole array
        z_galaxy : float
            Galaxy sample redshift
        galaxy_bias : float
            Galaxy bias parameter
            
        Returns
        -------
        jnp.ndarray
            Cross-correlation power spectrum
        """
        def cross_cl_single(l):
            if l < 2:
                return 0.0
            
            # Integration over comoving distance
            chi_min = 10.0
            chi_max = self.chi_star
            n_chi = 100
            chi_array = jnp.linspace(chi_min, chi_max, n_chi)
            
            def integrand(chi):
                # Lensing kernel
                W_phi = self._lensing_kernel_jit(chi)
                
                # Galaxy window (approximate delta function)
                chi_galaxy = self.bg.distances.comoving_distance(z_galaxy)
                sigma_chi = 50.0  # Mpc (window width)
                W_galaxy = jnp.exp(-(chi - chi_galaxy)**2 / (2 * sigma_chi**2))
                W_galaxy *= galaxy_bias
                
                # Matter power spectrum
                k = l / chi
                k_h = k * self.params.get_value('h')
                z = self._chi_to_z_approx(chi)
                P_delta = self.nonlinear.nonlinear_power_spectrum(k_h, z)
                
                return W_phi * W_galaxy * P_delta / chi**2
            
            integrand_vals = vmap(integrand)(chi_array)
            return jnp.trapz(integrand_vals, chi_array)
        
        C_l_cross = vmap(cross_cl_single)(l_array)
        
        return C_l_cross
    
    def cmb_lensing_cosmic_shear_cross_cl(self, l_array: jnp.ndarray,
                                        z_source: float) -> jnp.ndarray:
        """
        Cross-correlation between CMB lensing and cosmic shear.
        
        Both probe the same gravitational potential field.
        
        Parameters
        ----------
        l_array : jnp.ndarray
            Multipole array
        z_source : float
            Weak lensing source redshift
            
        Returns
        -------
        jnp.ndarray
            Cross-correlation power spectrum
        """
        def cross_cl_single(l):
            if l < 2:
                return 0.0
            
            # Integration
            chi_min = 10.0
            chi_max = min(self.chi_star, self.bg.distances.comoving_distance(z_source))
            n_chi = 100
            chi_array = jnp.linspace(chi_min, chi_max, n_chi)
            
            def integrand(chi):
                # CMB lensing kernel
                W_cmb = self._lensing_kernel_jit(chi)
                
                # Weak lensing kernel
                chi_source = self.bg.distances.comoving_distance(z_source)
                if chi >= chi_source:
                    W_wl = 0.0
                else:
                    z = self._chi_to_z_approx(chi)
                    Omega_m = self.params.get_value('Omega_m')
                    H0_over_c = self.H0 / self.c_km_s
                    
                    W_wl = (1.5 * Omega_m * H0_over_c**2 * (1 + z) * chi * 
                           (chi_source - chi) / chi_source)
                
                # Matter power spectrum
                k = l / chi
                k_h = k * self.params.get_value('h')
                z = self._chi_to_z_approx(chi)
                P_delta = self.nonlinear.nonlinear_power_spectrum(k_h, z)
                
                return W_cmb * W_wl * P_delta / chi**2
            
            integrand_vals = vmap(integrand)(chi_array)
            return jnp.trapz(integrand_vals, chi_array)
        
        C_l_cross = vmap(cross_cl_single)(l_array)
        
        return C_l_cross
    
    # ==================== Utility Methods ====================
    
    def lensing_summary(self, l_pivot: int = 100) -> str:
        """
        Generate comprehensive CMB lensing summary.
        
        Parameters
        ----------
        l_pivot : int
            Reference multipole for summary
            
        Returns
        -------
        str
            Formatted summary
        """
        # Key calculations
        C_l_phi_pivot = self.lensing_potential_cl(l_pivot)
        C_l_dd_pivot = self.lensing_deflection_cl(l_pivot)
        
        # Peak of lensing kernel (approximate)
        z_peak_lensing = 2.0
        W_peak = self.lensing_efficiency_function(jnp.array([z_peak_lensing]))[0]
        
        # Sample values at different scales
        l_sample = jnp.array([10, 100, 1000, 2000])
        C_l_phi_sample = self.lensing_potential_cl(l_sample)
        
        lines = [
            "CMB Lensing Summary",
            "=" * 25,
            f"Last scattering: z* = {self.z_star:.0f}",
            f"Comoving distance: χ* = {self.chi_star:.0f} Mpc",
            f"Lensing peak redshift: z ~ {z_peak_lensing:.1f}",
            f"Peak lensing efficiency: W = {W_peak:.2e}",
            "",
            f"Lensing Potential C_l^φφ at l = {l_pivot}:",
            f"  C_{l_pivot}^φφ = {C_l_phi_pivot:.2e}",
            "",
            f"Deflection C_l^dd at l = {l_pivot} (μrad²):",
            f"  C_{l_pivot}^dd = {C_l_dd_pivot:.2f}",
            "",
            "Lensing Power Spectrum Samples:",
            f"  C_10^φφ = {C_l_phi_sample[0]:.2e}",
            f"  C_100^φφ = {C_l_phi_sample[1]:.2e}",
            f"  C_1000^φφ = {C_l_phi_sample[2]:.2e}",
            f"  C_2000^φφ = {C_l_phi_sample[3]:.2e}",
        ]
        
        return "\n".join(lines)
    
    def plot_lensing_spectra(self, l_min: int = 10, l_max: int = 3000) -> Tuple[jnp.ndarray, Dict]:
        """
        Generate data for plotting CMB lensing spectra.
        
        Parameters
        ----------
        l_min, l_max : int
            Multipole range
            
        Returns
        -------
        l_array : jnp.ndarray
            Multipole values
        spectra : dict
            Dictionary with lensing spectra
        """
        # Logarithmic multipole spacing
        n_points = 100
        l_array = jnp.logspace(jnp.log10(l_min), jnp.log10(l_max), n_points).astype(int)
        l_array = jnp.unique(l_array)
        
        # Compute spectra
        spectra = {
            'phi_phi': self.lensing_potential_cl(l_array),
            'deflection': self.lensing_deflection_cl(l_array)
        }
        
        # Add cross-correlations if desired
        # spectra['cmb_galaxy_cross'] = self.cmb_lensing_galaxy_cross_cl(l_array, z_galaxy=0.5)
        
        return l_array, spectra
    
    def benchmark_lensing_values(self) -> Dict[str, float]:
        """
        Generate benchmark CMB lensing values.
        
        Returns
        -------
        dict
            Benchmark quantities
        """
        # Standard reference values
        l_ref = 100
        
        return {
            'C_l_phi_phi_at_100': float(self.lensing_potential_cl(l_ref)),
            'C_l_deflection_at_100_microrad2': float(self.lensing_deflection_cl(l_ref)),
            'last_scattering_redshift': float(self.z_star),
            'comoving_distance_to_ls': float(self.chi_star),
            'lensing_efficiency_peak': float(self.lensing_efficiency_function(jnp.array([2.0]))[0])
        }