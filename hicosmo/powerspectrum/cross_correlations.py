"""
Cross-Correlation Calculator
============================

Professional implementation of cross-correlation calculations for multi-probe cosmology.
Essential for combined analyses of different observational probes.

Key features:
- Galaxy-galaxy lensing
- Galaxy-CMB lensing cross-correlations  
- Clustering-lensing cross-correlations
- Multi-redshift bin correlations
- Limber approximation for projected quantities
- Full 3D correlation functions

Cross-correlations provide:
- Enhanced cosmological constraints
- Control of systematic effects
- Tests of modified gravity
- Validation of analysis pipelines
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from scipy.special import j1 as bessel_j1
from functools import partial
from typing import Union, Tuple, Dict, Optional, Callable, List
import numpy as np

from ..background.background import BackgroundEvolution
from ..perturbations.linear_perturbations import LinearPerturbations
from .linear_power import LinearPowerSpectrum
from .nonlinear_corrections import NonlinearCorrections


class CrossCorrelationCalculator:
    """
    Cross-correlation calculator for multi-probe cosmology.
    
    Computes cross-correlations between different tracers of the cosmic web:
    - Matter density fluctuations
    - Galaxy overdensities
    - Weak lensing shear
    - CMB temperature and lensing
    - 21cm intensity mapping
    """
    
    def __init__(self, background: BackgroundEvolution,
                 perturbations: LinearPerturbations,
                 linear_power: LinearPowerSpectrum,
                 nonlinear_power: NonlinearCorrections):
        """
        Initialize cross-correlation calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology
        perturbations : LinearPerturbations
            Linear perturbation theory
        linear_power : LinearPowerSpectrum
            Linear power spectrum
        nonlinear_power : NonlinearCorrections
            Nonlinear corrections
        """
        self.bg = background
        self.pert = perturbations
        self.linear = linear_power
        self.nonlinear = nonlinear_power
        self.params = background.model.params
        
        # Physical constants
        self.c_km_s = 299792.458  # Speed of light in km/s
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile JAX functions for performance."""
        
        # Limber integration kernels
        self._limber_kernel_jit = jit(self._limber_integration_kernel)
        
        # Lensing efficiency functions
        self._lensing_efficiency_jit = jit(self._lensing_efficiency)
        
        # Projection integrals
        self._project_power_spectrum_jit = jit(self._project_power_spectrum)
    
    # ==================== Lensing Cross-Correlations ====================
    
    def _lensing_efficiency(self, z: float, z_source: float) -> float:
        """
        Lensing efficiency function for weak lensing.
        
        W_lens(z) = (3/2) Ω_m H₀² (1+z) χ(z) [χ(z_s) - χ(z)] / χ(z_s)
        """
        if z >= z_source:
            return 0.0
        
        # Comoving distances
        chi_z = self.bg.distances.comoving_distance(z)
        chi_s = self.bg.distances.comoving_distance(z_source)
        
        # Efficiency
        H0_Mpc = self.bg.H0 * 3.24078e-20  # H0 in 1/Mpc units
        Omega_m = self.params.get_value('Omega_m')
        
        W_lens = (1.5 * Omega_m * H0_Mpc**2 * (1 + z) * chi_z * 
                 (chi_s - chi_z) / chi_s)
        
        return W_lens
    
    def galaxy_galaxy_lensing_power(self, ell: Union[float, jnp.ndarray],
                                  z_lens: float, z_source: float,
                                  linear_bias: float = 1.0) -> Union[float, jnp.ndarray]:
        """
        Galaxy-galaxy lensing power spectrum C_ℓ^{gκ}.
        
        Cross-correlation between foreground galaxies and background lensing.
        
        Parameters
        ----------
        ell : float or array_like
            Multipole moments
        z_lens : float
            Lens redshift
        z_source : float  
            Source redshift
        linear_bias : float
            Galaxy linear bias
            
        Returns
        -------
        float or array_like
            Galaxy-galaxy lensing power spectrum
        """
        ell = jnp.asarray(ell)
        
        def integrand(z):
            # Weight functions
            W_g = 1.0 if jnp.abs(z - z_lens) < 0.1 else 0.0  # Galaxy window (delta function)
            W_lens = self._lensing_efficiency(z, z_source)
            
            # Comoving distance and Hubble parameter
            chi = self.bg.distances.comoving_distance(z)
            H_z = self.bg.H_z(z)
            
            # Wave number
            k = (ell + 0.5) / chi
            
            # Power spectrum
            P_k = self.nonlinear.nonlinear_power_spectrum(k, z)
            
            # Integrand
            return W_g * W_lens * P_k / (chi**2 * H_z)
        
        # Integration over redshift
        z_array = jnp.linspace(0, max(z_lens, z_source) + 0.5, 100)
        integrand_vals = vmap(integrand)(z_array)
        
        # Numerical integration
        C_ell_gk = linear_bias * jnp.trapz(integrand_vals, z_array)
        
        return C_ell_gk
    
    # ==================== Clustering Cross-Correlations ====================
    
    def galaxy_clustering_power(self, ell: Union[float, jnp.ndarray],
                              z1: float, z2: float,
                              bias1: float = 1.0, bias2: float = 1.0,
                              window_width: float = 0.1) -> Union[float, jnp.ndarray]:
        """
        Galaxy clustering power spectrum between two redshift bins.
        
        Parameters
        ----------
        ell : float or array_like
            Multipole moments
        z1, z2 : float
            Central redshifts of two bins
        bias1, bias2 : float
            Linear bias parameters
        window_width : float
            Width of redshift windows
            
        Returns
        -------
        float or array_like
            Cross-clustering power spectrum
        """
        ell = jnp.asarray(ell)
        
        def integrand(z):
            # Window functions (top-hat)
            W1 = jnp.where(jnp.abs(z - z1) < window_width/2, 1.0, 0.0)
            W2 = jnp.where(jnp.abs(z - z2) < window_width/2, 1.0, 0.0)
            
            chi = self.bg.distances.comoving_distance(z)
            H_z = self.bg.H_z(z)
            
            k = (ell + 0.5) / chi
            P_k = self.nonlinear.nonlinear_power_spectrum(k, z)
            
            return W1 * W2 * P_k / (chi**2 * H_z)
        
        # Integration
        z_min = min(z1, z2) - window_width
        z_max = max(z1, z2) + window_width
        z_array = jnp.linspace(z_min, z_max, 100)
        
        integrand_vals = vmap(integrand)(z_array)
        C_ell_gg = bias1 * bias2 * jnp.trapz(integrand_vals, z_array)
        
        return C_ell_gg
    
    # ==================== Weak Lensing Auto/Cross-Correlations ====================
    
    def lensing_auto_power(self, ell: Union[float, jnp.ndarray],
                         z_source1: float, z_source2: float) -> Union[float, jnp.ndarray]:
        """
        Weak lensing auto/cross power spectrum C_ℓ^{κκ}.
        
        Parameters
        ----------
        ell : float or array_like
            Multipole moments
        z_source1, z_source2 : float
            Source redshifts for cross-correlation
            
        Returns
        -------
        float or array_like
            Lensing power spectrum
        """
        ell = jnp.asarray(ell)
        
        def integrand(z):
            W1 = self._lensing_efficiency(z, z_source1)
            W2 = self._lensing_efficiency(z, z_source2)
            
            chi = self.bg.distances.comoving_distance(z)
            H_z = self.bg.H_z(z)
            
            k = (ell + 0.5) / chi
            P_k = self.nonlinear.nonlinear_power_spectrum(k, z)
            
            return W1 * W2 * P_k / (chi**2 * H_z)
        
        # Integration over all lens redshifts
        z_max = min(z_source1, z_source2)
        z_array = jnp.linspace(0, z_max, 200)
        
        integrand_vals = vmap(integrand)(z_array)
        C_ell_kk = jnp.trapz(integrand_vals, z_array)
        
        return C_ell_kk
    
    # ==================== CMB Cross-Correlations ====================
    
    def cmb_lensing_galaxy_cross(self, ell: Union[float, jnp.ndarray],
                               z_galaxy: float,
                               linear_bias: float = 1.0) -> Union[float, jnp.ndarray]:
        """
        Cross-correlation between CMB lensing and galaxies.
        
        Parameters
        ----------
        ell : float or array_like
            Multipole moments
        z_galaxy : float
            Galaxy redshift
        linear_bias : float
            Galaxy bias
            
        Returns
        -------
        float or array_like
            CMB lensing - galaxy cross power
        """
        ell = jnp.asarray(ell)
        
        # CMB lensing kernel peaks around z ~ 2
        z_cmb = 1100.0  # Recombination
        
        def integrand(z):
            # Galaxy window
            W_g = jnp.where(jnp.abs(z - z_galaxy) < 0.1, 1.0, 0.0)
            
            # CMB lensing efficiency
            W_cmb = self._lensing_efficiency(z, z_cmb)
            
            chi = self.bg.distances.comoving_distance(z)
            H_z = self.bg.H_z(z)
            
            k = (ell + 0.5) / chi
            P_k = self.nonlinear.nonlinear_power_spectrum(k, z)
            
            return W_g * W_cmb * P_k / (chi**2 * H_z)
        
        z_array = jnp.linspace(0, z_galaxy + 1.0, 100)
        integrand_vals = vmap(integrand)(z_array)
        
        C_ell_cmb_g = linear_bias * jnp.trapz(integrand_vals, z_array)
        
        return C_ell_cmb_g
    
    # ==================== 3D Correlation Functions ====================
    
    def _project_power_spectrum(self, k: float, mu: float, P_k: float,
                              f_growth: float) -> float:
        """
        Project 3D power spectrum including RSD effects.
        
        P(k,μ) = P(k) * (1 + f*μ²)²
        """
        kaiser_factor = (1 + f_growth * mu**2)**2
        return P_k * kaiser_factor
    
    def correlation_function_3d(self, r: Union[float, jnp.ndarray],
                              z: float = 0,
                              n_k_points: int = 1000) -> Union[float, jnp.ndarray]:
        """
        3D matter correlation function ξ(r,z).
        
        ξ(r) = ∫ P(k) j₀(kr) k²dk/(2π²)
        
        where j₀ is spherical Bessel function.
        
        Parameters
        ----------
        r : float or array_like
            Separation in Mpc/h
        z : float
            Redshift
        n_k_points : int
            Integration points in k
            
        Returns
        -------
        float or array_like
            Correlation function
        """
        r = jnp.asarray(r)
        
        def xi_single(r_val):
            # k-space integration
            k_min, k_max = 1e-4, 100.0
            k_array = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_k_points)
            
            # Power spectrum
            P_k = self.nonlinear.nonlinear_power_spectrum(k_array, z)
            
            # Spherical Bessel function j₀(kr) = sin(kr)/(kr)
            kr = k_array * r_val
            j0 = jnp.where(kr < 1e-3, 1.0, jnp.sin(kr) / kr)
            
            # Integrand
            integrand = P_k * j0 * k_array**2
            
            # Integration
            d_ln_k = jnp.log(k_array[1] / k_array[0])
            xi = jnp.sum(integrand * k_array * d_ln_k) / (2 * jnp.pi**2)
            
            return xi
        
        if r.ndim == 0:
            return xi_single(r)
        else:
            return vmap(xi_single)(r)
    
    def two_point_correlation_multipoles(self, r: Union[float, jnp.ndarray],
                                       z: float = 0,
                                       n_mu_points: int = 50) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Multipoles of 2-point correlation function in redshift space.
        
        ξ_ℓ(r) = (2ℓ+1)/2 ∫₋₁¹ ξ(r,μ) L_ℓ(μ) dμ
        
        Parameters
        ----------
        r : float or array_like
            Separation in Mpc/h
        z : float
            Redshift
        n_mu_points : int
            Angular integration points
            
        Returns
        -------
        dict
            Multipoles ξ₀, ξ₂, ξ₄
        """
        r = jnp.asarray(r)
        
        # Gauss-Legendre quadrature
        mu_points, weights = jnp.polynomial.legendre.leggauss(n_mu_points)
        
        def compute_multipole(ell):
            def integrand_ell(mu_val):
                # RSD correlation function (approximate)
                xi_real = self.correlation_function_3d(r, z)
                f = self.pert.growth_rate(z)
                
                # Simple Kaiser approximation for correlation function
                xi_rsd = xi_real * (1 + (2/3)*f + (1/5)*f**2 * (3*mu_val**2 - 1))
                
                # Legendre polynomials
                if ell == 0:
                    L_ell = 1.0
                elif ell == 2:
                    L_ell = 0.5 * (3*mu_val**2 - 1)
                elif ell == 4:
                    L_ell = (35*mu_val**4 - 30*mu_val**2 + 3) / 8
                else:
                    L_ell = 1.0
                
                return xi_rsd * L_ell
            
            # Integration
            integrand_vals = vmap(integrand_ell)(mu_points)
            return (2*ell + 1) / 2 * jnp.sum(integrand_vals * weights, axis=-1)
        
        xi_0 = compute_multipole(0)  # Monopole
        xi_2 = compute_multipole(2)  # Quadrupole  
        xi_4 = compute_multipole(4)  # Hexadecapole
        
        return {'xi_0': xi_0, 'xi_2': xi_2, 'xi_4': xi_4}
    
    # ==================== Multi-Probe Analysis Tools ====================
    
    def multi_probe_covariance_matrix(self, observables: List[str],
                                    ell_array: jnp.ndarray,
                                    z_bins: List[float],
                                    noise_levels: Dict[str, float] = None) -> jnp.ndarray:
        """
        Covariance matrix for multi-probe analysis.
        
        Parameters
        ----------
        observables : List[str]
            List of observables ('clustering', 'lensing', 'cross')
        ell_array : jnp.ndarray
            Multipole array
        z_bins : List[float]
            Redshift bins
        noise_levels : dict
            Noise levels for different probes
            
        Returns
        -------
        jnp.ndarray
            Covariance matrix
        """
        if noise_levels is None:
            noise_levels = {'clustering': 1e-6, 'lensing': 1e-7, 'cross': 0.0}
        
        n_ell = len(ell_array)
        n_obs = len(observables) * len(z_bins)
        
        # Initialize covariance matrix
        cov_matrix = jnp.zeros((n_obs * n_ell, n_obs * n_ell))
        
        # This is a simplified implementation
        # Full covariance would include cosmic variance, shot noise, and correlations
        for i, obs1 in enumerate(observables):
            for j, obs2 in enumerate(observables):
                if i == j:  # Auto-covariance
                    noise = noise_levels.get(obs1, 0.0)
                    diagonal_block = jnp.diag(jnp.full(n_ell, noise))
                    
                    # Insert into full matrix
                    start_i = i * n_ell
                    end_i = (i + 1) * n_ell
                    cov_matrix = cov_matrix.at[start_i:end_i, start_i:end_i].set(diagonal_block)
        
        return cov_matrix
    
    def fisher_matrix_multi_probe(self, observables: List[str],
                                ell_array: jnp.ndarray,
                                z_bins: List[float],
                                parameter_derivatives: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Fisher matrix for multi-probe cosmological analysis.
        
        F_αβ = ∑_ij (∂C_i/∂θ_α) Cov⁻¹_ij (∂C_j/∂θ_β)
        
        Parameters
        ----------
        observables : List[str]
            Observable types
        ell_array : jnp.ndarray
            Multipole array  
        z_bins : List[float]
            Redshift bins
        parameter_derivatives : dict
            Derivatives w.r.t. cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Fisher information matrix
        """
        # Get covariance matrix
        cov_matrix = self.multi_probe_covariance_matrix(observables, ell_array, z_bins)
        cov_inv = jnp.linalg.inv(cov_matrix)
        
        # Number of parameters
        param_names = list(parameter_derivatives.keys())
        n_params = len(param_names)
        
        # Initialize Fisher matrix
        fisher_matrix = jnp.zeros((n_params, n_params))
        
        # Compute Fisher matrix elements
        for i, param1 in enumerate(param_names):
            for j, param2 in enumerate(param_names):
                deriv1 = parameter_derivatives[param1].flatten()
                deriv2 = parameter_derivatives[param2].flatten()
                
                fisher_ij = jnp.dot(deriv1, jnp.dot(cov_inv, deriv2))
                fisher_matrix = fisher_matrix.at[i, j].set(fisher_ij)
        
        return fisher_matrix
    
    # ==================== Utility Methods ====================
    
    def cross_correlation_summary(self, z_lens: float = 0.5, z_source: float = 1.0) -> str:
        """
        Generate summary of cross-correlation calculations.
        
        Parameters
        ----------
        z_lens : float
            Lens redshift
        z_source : float  
            Source redshift
            
        Returns
        -------
        str
            Formatted summary
        """
        # Sample calculations
        ell_sample = jnp.array([10, 100, 1000])
        
        C_gg = self.galaxy_clustering_power(ell_sample, z_lens, z_lens)
        C_gk = self.galaxy_galaxy_lensing_power(ell_sample, z_lens, z_source)
        C_kk = self.lensing_auto_power(ell_sample, z_source, z_source)
        
        # Correlation function
        r_sample = 10.0  # Mpc/h
        xi_3d = self.correlation_function_3d(r_sample, z_lens)
        
        lines = [
            f"Cross-Correlation Summary",
            "=" * 35,
            f"Lens redshift: z = {z_lens:.1f}",
            f"Source redshift: z = {z_source:.1f}",
            "",
            "Angular Power Spectra (sample values):",
            f"Galaxy clustering C_ℓ^gg:",
            f"  ℓ=10:   {C_gg[0]:.2e}",
            f"  ℓ=100:  {C_gg[1]:.2e}",
            f"  ℓ=1000: {C_gg[2]:.2e}",
            "",
            f"Galaxy-lensing C_ℓ^gκ:",
            f"  ℓ=10:   {C_gk[0]:.2e}",
            f"  ℓ=100:  {C_gk[1]:.2e}",
            f"  ℓ=1000: {C_gk[2]:.2e}",
            "",
            f"Lensing auto C_ℓ^κκ:",
            f"  ℓ=10:   {C_kk[0]:.2e}",
            f"  ℓ=100:  {C_kk[1]:.2e}",
            f"  ℓ=1000: {C_kk[2]:.2e}",
            "",
            f"3D correlation function:",
            f"ξ(r={r_sample} Mpc/h) = {xi_3d:.4f}",
        ]
        
        return "\n".join(lines)