"""
Linear Power Spectrum Calculator
===============================

Professional implementation of linear matter power spectrum calculations.
The foundation for all structure formation observables.

Key features:
- Primordial power spectrum (Harrison-Zel'dovich, running spectral index)
- Transfer function convolution
- Growth factor evolution
- Component power spectra (CDM, baryons, matter)
- BAO feature modeling
- High-performance JAX optimization

The linear power spectrum is:
P_L(k,z) = A_s * (k/k_pivot)^(n_s-1) * T²(k) * D²(z)

where A_s is primordial amplitude, n_s is spectral index, T(k) is 
transfer function, and D(z) is growth factor.
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from typing import Union, Tuple, Dict, Optional, Callable
import numpy as np

from ..models import LCDM  # Use models instead of background module
from ..perturbations.linear_perturbations import LinearPerturbations
from ..perturbations.transfer_functions import TransferFunctionCalculator
from ..utils.constants import c_km_s, h_planck, k_B


class LinearPowerSpectrum:
    """
    Linear power spectrum calculator.
    
    Computes the linear matter power spectrum from primordial fluctuations,
    transfer functions, and growth evolution. Provides the foundation for
    all higher-order calculations and observables.
    """
    
    def __init__(self, background: BackgroundEvolution,
                 perturbations: LinearPerturbations,
                 transfer: TransferFunctionCalculator):
        """
        Initialize linear power spectrum calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology
        perturbations : LinearPerturbations  
            Linear perturbation calculator
        transfer : TransferFunctionCalculator
            Transfer function calculator
        """
        self.bg = background
        self.pert = perturbations
        self.transfer = transfer
        self.params = background.model.params
        
        # Primordial parameters
        self.A_s = jnp.exp(self.params.get_value('ln_A_s_1e10')) * 1e-10
        self.n_s = self.params.get_value('n_s')
        self.k_pivot = 0.05  # Mpc^-1 (pivot scale)
        
        # Normalization
        self.sigma8_target = 0.811  # Will be computed from parameters
        if 'sigma8' in self.params._param_values:
            self.sigma8_target = self.params.get_value('sigma8')
        
        # Pre-compile functions for performance
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile JAX functions for optimal performance."""
        
        # Vectorized power spectrum calculations
        self._power_spectrum_vmap = jit(vmap(self._power_spectrum_single, in_axes=(0, None)))
        self._power_spectrum_2d_vmap = jit(vmap(vmap(self._power_spectrum_single, in_axes=(0, None)), in_axes=(None, 0)))
        
        # Primordial power spectrum
        self._primordial_power_jit = jit(self._primordial_power_single)
        
        # Sigma8 calculation
        self._compute_sigma8_jit = jit(self._compute_sigma8)
    
    # ==================== Primordial Power Spectrum ====================
    
    def _primordial_power_single(self, k: float) -> float:
        """
        Primordial scalar power spectrum P_R(k).
        
        P_R(k) = A_s * (k/k_pivot)^(n_s - 1)
        """
        return self.A_s * (k / self.k_pivot)**(self.n_s - 1)
    
    def primordial_power_spectrum(self, k: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Primordial scalar power spectrum.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in Mpc^-1 (comoving)
            
        Returns
        -------
        float or array_like
            Primordial power spectrum
        """
        k = jnp.asarray(k)
        if k.ndim == 0:
            return self._primordial_power_single(k)
        else:
            return vmap(self._primordial_power_single)(k)
    
    def primordial_power_running(self, k: Union[float, jnp.ndarray],
                               alpha_s: float = 0.0) -> Union[float, jnp.ndarray]:
        """
        Primordial power spectrum with running spectral index.
        
        n_s(k) = n_s + α_s * ln(k/k_pivot)
        P_R(k) = A_s * (k/k_pivot)^(n_s(k) - 1)
        
        Parameters
        ----------
        k : float or array_like
            Wave number in Mpc^-1
        alpha_s : float
            Running of spectral index
            
        Returns
        -------
        float or array_like
            Primordial power spectrum with running
        """
        k = jnp.asarray(k)
        ln_k_ratio = jnp.log(k / self.k_pivot)
        n_s_running = self.n_s + alpha_s * ln_k_ratio
        
        return self.A_s * (k / self.k_pivot)**(n_s_running - 1)
    
    # ==================== Linear Matter Power Spectrum ====================
    
    def _power_spectrum_single(self, k: float, z: float = 0) -> float:
        """
        Linear matter power spectrum for single (k,z).
        
        P_L(k,z) = P_R(k) * T²(k) * D²(z) * (2π²/k³)
        """
        # Convert to h/Mpc for transfer function
        k_h = k * self.params.get_value('h')
        
        # Components
        P_R = self._primordial_power_single(k)
        T_k = self.transfer.transfer_matter(k_h)
        D_z = self.pert.growth_factor(z)
        
        # Power spectrum (dimensionless)
        Delta_sq = P_R * T_k**2 * D_z**2
        
        # Convert to P(k) = Δ²(k) * 2π²/k³
        P_k = Delta_sq * 2 * jnp.pi**2 / k**3
        
        return P_k
    
    def linear_power_spectrum(self, k: Union[float, jnp.ndarray],
                            z: Union[float, jnp.ndarray] = 0) -> Union[float, jnp.ndarray]:
        """
        Linear matter power spectrum P_L(k,z) in (Mpc/h)³.
        
        The fundamental quantity for structure formation calculations.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float or array_like
            Redshift(s)
            
        Returns
        -------
        float or array_like
            Linear power spectrum in (Mpc/h)³
        """
        k = jnp.asarray(k)
        z = jnp.asarray(z)
        
        # Convert k from h/Mpc to Mpc^-1 for internal calculations
        k_comoving = k / self.params.get_value('h')
        
        if k.ndim == 0 and z.ndim == 0:
            # Single k, single z
            return self._power_spectrum_single(k_comoving, z)
        elif k.ndim == 1 and z.ndim == 0:
            # Array k, single z
            return self._power_spectrum_vmap(k_comoving, z)
        elif k.ndim == 0 and z.ndim == 1:
            # Single k, array z
            return vmap(partial(self._power_spectrum_single, k_comoving))(z)
        else:
            # Array k, array z - full grid
            return self._power_spectrum_2d_vmap(k_comoving, z)
    
    # ==================== Component Power Spectra ====================
    
    def cdm_power_spectrum(self, k: Union[float, jnp.ndarray],
                          z: Union[float, jnp.ndarray] = 0) -> Union[float, jnp.ndarray]:
        """
        Cold dark matter power spectrum.
        
        Uses CDM transfer function instead of total matter.
        """
        k = jnp.asarray(k)
        z = jnp.asarray(z)
        k_comoving = k / self.params.get_value('h')
        
        def cdm_power_single(k_val, z_val):
            P_R = self._primordial_power_single(k_val)
            T_cdm = self.transfer.transfer_cdm(k_val * self.params.get_value('h'))
            D_z = self.pert.growth_factor(z_val)
            
            Delta_sq = P_R * T_cdm**2 * D_z**2
            return Delta_sq * 2 * jnp.pi**2 / k_val**3
        
        if k.ndim == 0 and z.ndim == 0:
            return cdm_power_single(k_comoving, z)
        else:
            return vmap(vmap(cdm_power_single, in_axes=(0, None)), in_axes=(None, 0))(k_comoving, z)
    
    def baryon_power_spectrum(self, k: Union[float, jnp.ndarray],
                            z: Union[float, jnp.ndarray] = 0) -> Union[float, jnp.ndarray]:
        """
        Baryon power spectrum with BAO features.
        
        Uses baryon transfer function with acoustic oscillations.
        """
        k = jnp.asarray(k)
        z = jnp.asarray(z)
        k_comoving = k / self.params.get_value('h')
        
        def baryon_power_single(k_val, z_val):
            P_R = self._primordial_power_single(k_val)
            T_b = self.transfer.transfer_baryon(k_val * self.params.get_value('h'))
            D_z = self.pert.growth_factor(z_val)
            
            Delta_sq = P_R * T_b**2 * D_z**2
            return Delta_sq * 2 * jnp.pi**2 / k_val**3
        
        if k.ndim == 0 and z.ndim == 0:
            return baryon_power_single(k_comoving, z)
        else:
            return vmap(vmap(baryon_power_single, in_axes=(0, None)), in_axes=(None, 0))(k_comoving, z)
    
    # ==================== BAO Modeling ====================
    
    def power_spectrum_no_wiggle(self, k: Union[float, jnp.ndarray],
                               z: Union[float, jnp.ndarray] = 0) -> Union[float, jnp.ndarray]:
        """
        Smooth power spectrum without BAO wiggles.
        
        Useful for BAO feature extraction and analysis.
        """
        k = jnp.asarray(k)
        z = jnp.asarray(z)
        k_comoving = k / self.params.get_value('h')
        
        def no_wiggle_power_single(k_val, z_val):
            P_R = self._primordial_power_single(k_val)
            T_nw = self.transfer._transfer_no_wiggle(k_val * self.params.get_value('h'))
            D_z = self.pert.growth_factor(z_val)
            
            Delta_sq = P_R * T_nw**2 * D_z**2
            return Delta_sq * 2 * jnp.pi**2 / k_val**3
        
        if k.ndim == 0 and z.ndim == 0:
            return no_wiggle_power_single(k_comoving, z)
        else:
            return vmap(vmap(no_wiggle_power_single, in_axes=(0, None)), in_axes=(None, 0))(k_comoving, z)
    
    def bao_wiggle_amplitude(self, k: Union[float, jnp.ndarray],
                           z: Union[float, jnp.ndarray] = 0) -> Union[float, jnp.ndarray]:
        """
        BAO wiggle amplitude as fraction of smooth power spectrum.
        
        wiggle = (P_wiggle - P_no_wiggle) / P_no_wiggle
        """
        P_total = self.linear_power_spectrum(k, z)
        P_smooth = self.power_spectrum_no_wiggle(k, z)
        
        return (P_total - P_smooth) / P_smooth
    
    # ==================== Normalization and σ8 ====================
    
    @partial(jit, static_argnums=(1,))
    def _compute_sigma8(self, z: float = 0, n_points: int = 1000) -> float:
        """
        Compute σ8(z) from power spectrum integral.
        
        σ8² = ∫ P(k) W²(kR) k² dk/(2π²)
        
        where W(x) = 3(sin x - x cos x)/x³ is top-hat window at R = 8 Mpc/h.
        """
        # Integration range in h/Mpc
        k_min, k_max = 1e-4, 1e2
        k_array = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_points)
        
        # Power spectrum
        P_k = self.linear_power_spectrum(k_array, z)
        
        # Top-hat window function
        R8 = 8.0  # Mpc/h
        x = k_array * R8
        W_tophat = 3 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
        
        # Integrand
        integrand = P_k * W_tophat**2 * k_array**2 / (2 * jnp.pi**2)
        
        # Numerical integration (log spacing)
        d_ln_k = jnp.log(k_array[1] / k_array[0])
        sigma8_squared = jnp.sum(integrand * k_array * d_ln_k)
        
        return jnp.sqrt(sigma8_squared)
    
    def sigma8(self, z: Union[float, jnp.ndarray] = 0) -> Union[float, jnp.ndarray]:
        """
        Matter fluctuation amplitude σ8(z).
        
        Parameters
        ----------
        z : float or array_like
            Redshift(s)
            
        Returns
        -------
        float or array_like
            σ8(z) values
        """
        z = jnp.asarray(z)
        if z.ndim == 0:
            return self._compute_sigma8(z)
        else:
            return vmap(self._compute_sigma8)(z)
    
    def sigma_R(self, R: Union[float, jnp.ndarray], z: float = 0,
               n_points: int = 1000) -> Union[float, jnp.ndarray]:
        """
        RMS fluctuation on scale R in Mpc/h.
        
        Parameters
        ----------
        R : float or array_like
            Scale in Mpc/h
        z : float
            Redshift
        n_points : int
            Integration points
            
        Returns
        -------
        float or array_like
            σ(R,z) values
        """
        def sigma_R_single(R_val):
            # Integration
            k_min, k_max = 1e-4, 1e2
            k_array = jnp.logspace(jnp.log10(k_min), jnp.log10(k_max), n_points)
            
            P_k = self.linear_power_spectrum(k_array, z)
            
            # Top-hat window
            x = k_array * R_val
            W_tophat = 3 * (jnp.sin(x) - x * jnp.cos(x)) / x**3
            
            integrand = P_k * W_tophat**2 * k_array**2 / (2 * jnp.pi**2)
            d_ln_k = jnp.log(k_array[1] / k_array[0])
            sigma_sq = jnp.sum(integrand * k_array * d_ln_k)
            
            return jnp.sqrt(sigma_sq)
        
        R = jnp.asarray(R)
        if R.ndim == 0:
            return sigma_R_single(R)
        else:
            return vmap(sigma_R_single)(R)
    
    # ==================== Redshift-Space Distortions ====================
    
    def redshift_space_power_spectrum(self, k: Union[float, jnp.ndarray],
                                    mu: Union[float, jnp.ndarray],
                                    z: float = 0) -> Union[float, jnp.ndarray]:
        """
        Redshift-space power spectrum with Kaiser effect.
        
        P_s(k,μ,z) = P_real(k,z) * (1 + β(z) μ²)²
        
        where μ is cosine of angle to line of sight.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        mu : float or array_like
            Line-of-sight cosine
        z : float
            Redshift
            
        Returns
        -------
        float or array_like
            Redshift-space power spectrum
        """
        P_real = self.linear_power_spectrum(k, z)
        beta = self.pert.growth_rate(z)  # Assuming linear bias = 1
        
        mu = jnp.asarray(mu)
        kaiser_factor = (1 + beta * mu**2)**2
        
        return P_real * kaiser_factor
    
    # ==================== Utility Methods ====================
    
    def power_spectrum_multipoles(self, k: Union[float, jnp.ndarray],
                                z: float = 0) -> Dict[str, Union[float, jnp.ndarray]]:
        """
        Multipoles of redshift-space power spectrum.
        
        P_ℓ(k) = (2ℓ+1)/2 ∫₋₁¹ P_s(k,μ) L_ℓ(μ) dμ
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float
            Redshift
            
        Returns
        -------
        dict
            Multipoles P_0, P_2, P_4
        """
        # Quadrature points and weights for Gauss-Legendre integration
        n_mu = 50
        mu_points, weights = jnp.polynomial.legendre.leggauss(n_mu)
        
        def compute_multipole(ell):
            def integrand(mu_val):
                P_s = self.redshift_space_power_spectrum(k, mu_val, z)
                if ell == 0:
                    L_ell = 1.0
                elif ell == 2:
                    L_ell = 0.5 * (3 * mu_val**2 - 1)
                elif ell == 4:
                    L_ell = (35 * mu_val**4 - 30 * mu_val**2 + 3) / 8
                else:
                    L_ell = 1.0
                
                return P_s * L_ell
            
            # Vectorized integration
            integrand_vals = vmap(integrand)(mu_points)
            return (2*ell + 1) / 2 * jnp.sum(integrand_vals * weights)
        
        P_0 = compute_multipole(0)  # Monopole
        P_2 = compute_multipole(2)  # Quadrupole
        P_4 = compute_multipole(4)  # Hexadecapole
        
        return {'P_0': P_0, 'P_2': P_2, 'P_4': P_4}
    
    def power_spectrum_summary(self, z: float = 0) -> str:
        """
        Generate comprehensive power spectrum summary.
        
        Parameters
        ----------
        z : float
            Redshift for summary
            
        Returns
        -------
        str
            Formatted summary
        """
        # Key values
        sigma8_z = self.sigma8(z)
        
        # Sample power spectrum values
        k_sample = jnp.array([0.01, 0.1, 1.0])  # h/Mpc
        P_sample = self.linear_power_spectrum(k_sample, z)
        
        # Growth information
        D_z = self.pert.growth_factor(z)
        f_z = self.pert.growth_rate(z)
        
        lines = [
            f"Linear Power Spectrum Summary at z = {z:.2f}",
            "=" * 45,
            f"σ8(z) = {sigma8_z:.4f}",
            f"Growth factor: D(z) = {D_z:.4f}",
            f"Growth rate: f(z) = {f_z:.4f}",
            "",
            "Power Spectrum Samples:",
            f"P(k=0.01): {P_sample[0]:.2e} (Mpc/h)³",
            f"P(k=0.1):  {P_sample[1]:.2e} (Mpc/h)³",
            f"P(k=1.0):  {P_sample[2]:.2e} (Mpc/h)³",
            "",
            f"Primordial parameters:",
            f"A_s = {self.A_s:.2e}",
            f"n_s = {self.n_s:.4f}",
            f"k_pivot = {self.k_pivot:.3f} Mpc⁻¹",
        ]
        
        return "\n".join(lines)
    
    def benchmark_power_spectrum(self) -> Dict[str, float]:
        """
        Generate benchmark values for validation.
        
        Returns
        -------
        dict
            Benchmark values at standard scales and redshifts
        """
        # Standard scales
        k_ref = 0.1  # h/Mpc
        z_ref = 0.0
        
        P_ref = self.linear_power_spectrum(k_ref, z_ref)
        sigma8_ref = self.sigma8(z_ref)
        
        return {
            'k_reference': k_ref,
            'z_reference': z_ref,
            'P_reference': float(P_ref),
            'sigma8_reference': float(sigma8_ref),
            'A_s': float(self.A_s),
            'n_s': float(self.n_s)
        }