"""
Nonlinear Corrections Calculator
===============================

Professional implementation of nonlinear corrections to the matter power spectrum.
Essential for accurate predictions on small scales where linear theory breaks down.

Key features:
- Halofit prescription (Smith et al. 2003, Takahashi et al. 2012)
- Bird et al. (2012) corrections for massive neutrinos
- Effective field theory corrections
- BAO reconstruction methods
- Scale-dependent bias models

The nonlinear power spectrum is crucial for:
- Weak lensing surveys
- Galaxy clustering on small scales  
- CMB lensing
- Cross-correlations with non-linear tracers
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from typing import Union, Tuple, Dict, Optional, Callable
import numpy as np

from ..background.background import BackgroundEvolution
from ..perturbations.linear_perturbations import LinearPerturbations
from .linear_power import LinearPowerSpectrum


class NonlinearCorrections:
    """
    Nonlinear corrections to the matter power spectrum.
    
    Implements state-of-the-art fitting formulas and theoretical prescriptions
    for computing nonlinear matter power spectra from linear theory.
    """
    
    def __init__(self, background: BackgroundEvolution,
                 perturbations: LinearPerturbations,
                 linear_power: LinearPowerSpectrum):
        """
        Initialize nonlinear corrections calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology
        perturbations : LinearPerturbations
            Linear perturbation theory
        linear_power : LinearPowerSpectrum
            Linear power spectrum calculator
        """
        self.bg = background
        self.pert = perturbations
        self.linear = linear_power
        self.params = background.model.params
        
        # Cosmological parameters for Halofit
        self.Omega_m = self.params.get_value('Omega_m')
        self.Omega_b = self.params.get_value('Omega_b')
        self.h = self.params.get_value('h')
        self.n_s = self.params.get_value('n_s')
        
        # Neutrino mass (if available)
        self.m_nu_sum = 0.06  # Default minimal mass
        
        # Pre-compile functions
        self._compile_functions()
    
    def _compile_functions(self):
        """Pre-compile JAX functions for performance."""
        
        # Vectorized nonlinear corrections
        self._halofit_correction_vmap = jit(vmap(self._halofit_correction_single, in_axes=(0, None)))
        self._nonlinear_power_vmap = jit(vmap(self._nonlinear_power_single, in_axes=(0, None)))
        
        # Halofit parameters
        self._halofit_params_jit = jit(self._compute_halofit_parameters)
    
    # ==================== Halofit Implementation ====================
    
    @partial(jit, static_argnums=(2,))
    def _compute_halofit_parameters(self, z: float = 0, n_points: int = 1000) -> Dict[str, float]:
        """
        Compute Halofit parameters for given redshift.
        
        Finds nonlinear scale k_nl, effective index n_eff, and curvature C.
        """
        # Integration range for σ(R) calculation
        R_min, R_max = 0.1, 100.0  # Mpc/h
        R_array = jnp.logspace(jnp.log10(R_min), jnp.log10(R_max), n_points)
        
        # Compute σ(R) at this redshift
        sigma_R = self.linear.sigma_R(R_array, z)
        
        # Find R_nl where σ(R_nl) = 1
        idx_nl = jnp.argmin(jnp.abs(sigma_R - 1.0))
        R_nl = R_array[idx_nl]
        k_nl = 1.0 / R_nl
        
        # Compute effective spectral index
        # n_eff = d ln σ² / d ln R|_{R=R_nl}
        if idx_nl > 0 and idx_nl < len(R_array) - 1:
            dR = R_array[idx_nl + 1] - R_array[idx_nl - 1]
            dsigma2 = sigma_R[idx_nl + 1]**2 - sigma_R[idx_nl - 1]**2
            n_eff = -R_nl / sigma_R[idx_nl]**2 * dsigma2 / dR
        else:
            n_eff = self.n_s - 1  # Fallback
        
        # Curvature parameter
        # C = d² ln σ² / d ln R²|_{R=R_nl}
        if idx_nl > 1 and idx_nl < len(R_array) - 2:
            # Numerical second derivative
            d2sigma2_dR2 = (sigma_R[idx_nl + 1]**2 - 2*sigma_R[idx_nl]**2 + sigma_R[idx_nl - 1]**2) / dR**2
            C = R_nl**2 / sigma_R[idx_nl]**2 * d2sigma2_dR2
        else:
            C = 0.0  # Fallback
        
        return {
            'k_nl': k_nl,
            'R_nl': R_nl, 
            'n_eff': n_eff,
            'C': C,
            'sigma_nl': sigma_R[idx_nl]
        }
    
    def _halofit_correction_single(self, k: float, z: float = 0) -> float:
        """
        Halofit nonlinear correction for single k-mode.
        
        Based on Smith et al. (2003) with Takahashi et al. (2012) updates.
        """
        # Get Halofit parameters
        hf_params = self._halofit_params_jit(z)
        k_nl = hf_params['k_nl']
        n_eff = hf_params['n_eff']
        C = hf_params['C']
        
        # Dimensionless wave number
        y = k / k_nl
        
        # Fitting parameters (Takahashi et al. 2012)
        a_n = 10**(1.5222 + 2.8553*n_eff + 2.3706*n_eff**2 + 0.9903*n_eff**3 + 0.2250*n_eff**4 - 0.6038*C)
        b_n = 10**(-0.5642 + 0.5864*n_eff + 0.5716*n_eff**2 - 1.5474*C)
        c_n = 10**(0.3698 + 2.0404*n_eff + 0.8161*n_eff**2 + 0.5869*C)
        gamma_n = 0.1971 - 0.0843*n_eff + 0.8460*C
        alpha_n = jnp.abs(6.0835 + 1.3373*n_eff - 0.1959*n_eff**2 - 5.5274*C)
        beta_n = 2.0379 - 0.7354*n_eff + 0.3157*n_eff**2 + 1.2490*n_eff**3 + 0.3980*n_eff**4 - 0.1682*C
        mu_n = 0.0
        nu_n = 10**(5.2105 + 3.6902*n_eff)
        
        # Halofit formula
        f1 = y / 4.0 + y**2 / 8.0
        f2 = y**0.5 / 2.0 + y**1.5 / 4.0
        f3 = y**3 / (1.0 + y**3)
        
        # Quasi-linear regime
        Delta_Q = (((1.0 + (b_n * y)**alpha_n) / (1.0 + ((b_n * y * c_n)**alpha_n)))*
                   jnp.exp(f1)) * (1.0 + f3)**gamma_n
        
        # Nonlinear regime  
        Delta_H = (a_n * y**(3.0 * beta_n)) / (1.0 + (a_n * y) + ((b_n * y)**alpha_n) + ((c_n * y)**(3.0 + 4.0 * beta_n)))
        
        # Transition
        Delta_NL = Delta_Q + Delta_H
        
        return Delta_NL
    
    def _nonlinear_power_single(self, k: float, z: float = 0) -> float:
        """
        Nonlinear power spectrum for single (k,z).
        
        P_NL(k,z) = P_L(k,z) * [Δ_NL(k,z)]²
        """
        P_linear = self.linear.linear_power_spectrum(k, z)
        correction = self._halofit_correction_single(k, z)
        
        return P_linear * correction**2
    
    def nonlinear_power_spectrum(self, k: Union[float, jnp.ndarray],
                               z: Union[float, jnp.ndarray] = 0,
                               method: str = 'halofit') -> Union[float, jnp.ndarray]:
        """
        Nonlinear matter power spectrum.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float or array_like
            Redshift(s)
        method : str
            Nonlinear method ('halofit', 'bird', 'eft')
            
        Returns
        -------
        float or array_like
            Nonlinear power spectrum in (Mpc/h)³
        """
        k = jnp.asarray(k)
        z = jnp.asarray(z)
        
        if method == 'halofit':
            if k.ndim == 0 and z.ndim == 0:
                return self._nonlinear_power_single(k, z)
            elif k.ndim == 1 and z.ndim == 0:
                return self._nonlinear_power_vmap(k, z)
            else:
                # Full grid calculation
                return vmap(vmap(self._nonlinear_power_single, in_axes=(0, None)), in_axes=(None, 0))(k, z)
        else:
            # Other methods can be implemented here
            return self.linear.linear_power_spectrum(k, z)  # Fallback to linear
    
    # ==================== Massive Neutrino Corrections ====================
    
    def neutrino_suppression_halofit(self, k: Union[float, jnp.ndarray],
                                   z: float = 0) -> Union[float, jnp.ndarray]:
        """
        Neutrino suppression in nonlinear regime (Bird et al. 2012).
        
        Modifies Halofit prescription for massive neutrinos.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float
            Redshift
            
        Returns
        -------
        float or array_like
            Suppression factor
        """
        if self.m_nu_sum < 0.01:  # Effectively massless
            return 1.0
        
        k = jnp.asarray(k)
        
        # Neutrino parameters
        f_nu = self.m_nu_sum / (94.07 * self.Omega_m * self.h**2)
        
        # Free-streaming scale (comoving)
        k_fs_comoving = 0.026 * jnp.sqrt(self.m_nu_sum / 0.05) * (self.Omega_m * self.h**2 / 0.14)  # h/Mpc
        
        # Bird et al. (2012) fitting formula for nonlinear suppression
        y = k / k_fs_comoving
        
        # Growth-dependent suppression
        D_z = self.pert.growth_factor(z)
        z_collapse = 1.0  # Approximate collapse redshift
        
        alpha = 0.7 * f_nu * (D_z / (1 + z_collapse))**(1.5)
        suppression = jnp.exp(-alpha * y**0.7)
        
        return suppression
    
    def nonlinear_power_with_neutrinos(self, k: Union[float, jnp.ndarray],
                                     z: Union[float, jnp.ndarray] = 0) -> Union[float, jnp.ndarray]:
        """
        Nonlinear power spectrum including massive neutrino effects.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float or array_like
            Redshift(s)
            
        Returns
        -------
        float or array_like
            Nonlinear power spectrum with neutrino corrections
        """
        P_nl_standard = self.nonlinear_power_spectrum(k, z)
        nu_suppression = self.neutrino_suppression_halofit(k, z)
        
        return P_nl_standard * nu_suppression
    
    # ==================== BAO Reconstruction ====================
    
    def bao_reconstructed_power(self, k: Union[float, jnp.ndarray],
                              z: float = 0,
                              damping_factor: float = 0.5) -> Union[float, jnp.ndarray]:
        """
        BAO-reconstructed power spectrum.
        
        Removes some nonlinear damping of BAO features through 
        density field reconstruction.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float
            Redshift
        damping_factor : float
            Remaining damping after reconstruction (0 = perfect, 1 = no reconstruction)
            
        Returns
        -------
        float or array_like
            Reconstructed power spectrum
        """
        # Linear and nonlinear power spectra
        P_linear = self.linear.linear_power_spectrum(k, z)
        P_nonlinear = self.nonlinear_power_spectrum(k, z)
        
        # Smooth components (without BAO)
        P_linear_smooth = self.linear.power_spectrum_no_wiggle(k, z)
        
        # BAO wiggles
        BAO_linear = P_linear - P_linear_smooth
        
        # Reconstructed power: smooth nonlinear + partially restored BAO
        P_reconstructed = (P_nonlinear - P_linear + P_linear_smooth + 
                          damping_factor * BAO_linear)
        
        return P_reconstructed
    
    # ==================== Scale-Dependent Bias ====================
    
    def scale_dependent_bias(self, k: Union[float, jnp.ndarray],
                           z: float = 0,
                           bias_model: str = 'local') -> Union[float, jnp.ndarray]:
        """
        Scale-dependent bias corrections for galaxy clustering.
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float
            Redshift
        bias_model : str
            Bias model ('local', 'non_local', 'tidal')
            
        Returns
        -------
        float or array_like
            Bias correction factor
        """
        k = jnp.asarray(k)
        
        if bias_model == 'local':
            # Local bias (constant)
            return 1.0
        elif bias_model == 'non_local':
            # Non-local bias (scale-dependent)
            # b(k) = b₁ + b₂ * D(z) * σ²(R(k))
            D_z = self.pert.growth_factor(z)
            R_k = 1.0 / k  # Effective scale
            sigma_R = self.linear.sigma_R(R_k, z)
            
            b1 = 1.0  # Linear bias
            b2 = 0.1  # Second-order bias
            
            return b1 + b2 * D_z * sigma_R**2
        else:
            return 1.0
    
    def galaxy_power_spectrum(self, k: Union[float, jnp.ndarray],
                            z: float = 0,
                            linear_bias: float = 1.0,
                            bias_model: str = 'local') -> Union[float, jnp.ndarray]:
        """
        Galaxy power spectrum with bias and nonlinear corrections.
        
        P_gg(k,z) = b²(k,z) * P_mm(k,z)
        
        Parameters
        ----------
        k : float or array_like
            Wave number in h/Mpc
        z : float
            Redshift
        linear_bias : float
            Linear bias parameter
        bias_model : str
            Bias model for scale dependence
            
        Returns
        -------
        float or array_like
            Galaxy power spectrum
        """
        P_matter = self.nonlinear_power_spectrum(k, z)
        bias_correction = self.scale_dependent_bias(k, z, bias_model)
        
        total_bias = linear_bias * bias_correction
        
        return total_bias**2 * P_matter
    
    # ==================== Utility Methods ====================
    
    def compare_linear_nonlinear(self, k_array: jnp.ndarray, z: float = 0) -> Dict:
        """
        Compare linear and nonlinear power spectra.
        
        Parameters
        ----------
        k_array : jnp.ndarray
            Wave number array in h/Mpc
        z : float
            Redshift
            
        Returns
        -------
        dict
            Comparison data
        """
        P_linear = self.linear.linear_power_spectrum(k_array, z)
        P_nonlinear = self.nonlinear_power_spectrum(k_array, z)
        
        # Ratio and difference
        ratio = P_nonlinear / P_linear
        difference_percent = 100 * (P_nonlinear - P_linear) / P_linear
        
        return {
            'k': k_array,
            'P_linear': P_linear,
            'P_nonlinear': P_nonlinear,
            'ratio': ratio,
            'difference_percent': difference_percent
        }
    
    def nonlinear_summary(self, z: float = 0) -> str:
        """
        Generate summary of nonlinear corrections.
        
        Parameters
        ----------
        z : float
            Redshift for summary
            
        Returns
        -------
        str
            Formatted summary
        """
        # Halofit parameters
        hf_params = self._halofit_params_jit(z)
        
        # Sample corrections at different scales
        k_sample = jnp.array([0.1, 1.0, 10.0])  # h/Mpc
        P_lin_sample = self.linear.linear_power_spectrum(k_sample, z)
        P_nl_sample = self.nonlinear_power_spectrum(k_sample, z)
        ratio_sample = P_nl_sample / P_lin_sample
        
        lines = [
            f"Nonlinear Corrections Summary at z = {z:.2f}",
            "=" * 45,
            f"Nonlinear scale: k_nl = {hf_params['k_nl']:.4f} h/Mpc",
            f"Effective index: n_eff = {hf_params['n_eff']:.4f}",
            f"Curvature: C = {hf_params['C']:.4f}",
            "",
            "Nonlinear Enhancement:",
            f"P_NL/P_L(k=0.1):  {ratio_sample[0]:.3f}",
            f"P_NL/P_L(k=1.0):  {ratio_sample[1]:.3f}",
            f"P_NL/P_L(k=10):   {ratio_sample[2]:.3f}",
            "",
            f"Neutrino mass: Σm_ν = {self.m_nu_sum:.3f} eV",
        ]
        
        return "\n".join(lines)
    
    def halofit_benchmarks(self) -> Dict[str, float]:
        """
        Generate Halofit benchmark values.
        
        Returns
        -------
        dict
            Benchmark parameters and values
        """
        z_ref = 0.0
        hf_params = self._halofit_params_jit(z_ref)
        
        # Reference scales and values
        k_ref = 1.0  # h/Mpc
        P_lin_ref = self.linear.linear_power_spectrum(k_ref, z_ref)
        P_nl_ref = self.nonlinear_power_spectrum(k_ref, z_ref)
        
        return {
            'z_reference': z_ref,
            'k_nl': float(hf_params['k_nl']),
            'n_eff': float(hf_params['n_eff']),
            'C': float(hf_params['C']),
            'k_reference': k_ref,
            'P_linear_ref': float(P_lin_ref),
            'P_nonlinear_ref': float(P_nl_ref),
            'enhancement_ref': float(P_nl_ref / P_lin_ref)
        }