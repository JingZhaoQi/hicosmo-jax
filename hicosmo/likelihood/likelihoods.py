"""
Likelihood Calculator
====================

Professional likelihood calculations for cosmological parameter estimation.
Implements chi-squared and Gaussian likelihoods for various observational probes.

Key features:
- Type Ia Supernovae likelihood
- Baryon Acoustic Oscillations likelihood  
- CMB parameter likelihood
- Hubble constant likelihood
- Combined multi-probe likelihoods
- Systematic uncertainty handling
- Marginalization over nuisance parameters

Optimized with JAX for high-performance MCMC sampling.
"""

import jax.numpy as jnp
from jax import jit, vmap, grad
from jax.scipy.linalg import solve, cholesky
from jax.scipy.stats import norm
from functools import partial
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import numpy as np
from abc import ABC, abstractmethod

from .observational_data import Dataset, ObservationalDataManager
from ..background.background import BackgroundEvolution
from ..powerspectrum.linear_power import LinearPowerSpectrum


class BaseLikelihood(ABC):
    """Base class for all likelihood calculations."""
    
    def __init__(self, dataset: Dataset):
        """
        Initialize base likelihood.
        
        Parameters
        ----------
        dataset : Dataset
            Observational dataset
        """
        self.dataset = dataset
        self.name = dataset.name
        
    @abstractmethod
    def log_likelihood(self, theory_prediction: jnp.ndarray, **kwargs) -> float:
        """
        Compute log-likelihood given theory prediction.
        
        Parameters
        ----------
        theory_prediction : jnp.ndarray
            Theoretical prediction for observables
        **kwargs
            Additional parameters (nuisance parameters, etc.)
            
        Returns
        -------
        float
            Log-likelihood value
        """
        pass


class SupernovaeLikelihood(BaseLikelihood):
    """
    Type Ia Supernovae likelihood.
    
    Implements chi-squared likelihood for distance moduli with
    optional marginalization over absolute magnitude M_B.
    """
    
    def __init__(self, dataset: Dataset, marginalize_mb: bool = True):
        """
        Initialize SNe likelihood.
        
        Parameters
        ----------
        dataset : Dataset
            Supernovae dataset
        marginalize_mb : bool
            Whether to marginalize over absolute magnitude M_B
        """
        super().__init__(dataset)
        
        if dataset.data_type != 'sne':
            raise ValueError("Dataset must be of type 'sne'")
        
        self.marginalize_mb = marginalize_mb
        
        # Extract data
        self.z_obs = dataset.data[:, 0]  # Redshifts
        self.mu_obs = dataset.data[:, 1]  # Observed distance moduli
        
        # Covariance matrix
        if dataset.covariance is not None:
            self.cov_matrix = dataset.covariance
            self.inv_cov = jnp.linalg.inv(self.cov_matrix)
        else:
            # Diagonal covariance from uncertainties
            self.cov_matrix = jnp.diag(dataset.uncertainties**2)
            self.inv_cov = jnp.diag(1.0 / dataset.uncertainties**2)
        
        # Pre-compute marginalization quantities if needed
        if self.marginalize_mb:
            self._setup_mb_marginalization()
    
    def _setup_mb_marginalization(self):
        """Setup quantities for M_B marginalization."""
        # For marginalization over M_B, we need C^{-1} * 1 where 1 is vector of ones
        ones = jnp.ones(len(self.mu_obs))
        self.inv_cov_ones = jnp.dot(self.inv_cov, ones)
        self.ones_inv_cov_ones = jnp.dot(ones, self.inv_cov_ones)
    
    @jit
    def log_likelihood(self, mu_theory: jnp.ndarray, M_B: float = 0.0) -> float:
        """
        Compute SNe log-likelihood.
        
        Parameters
        ----------
        mu_theory : jnp.ndarray
            Theoretical distance moduli
        M_B : float
            Absolute magnitude (if not marginalizing)
            
        Returns
        -------
        float
            Log-likelihood value
        """
        # Residuals
        residuals = self.mu_obs - (mu_theory + M_B)
        
        if self.marginalize_mb:
            # Analytical marginalization over M_B
            # chi2 = (mu_obs - mu_theory)^T * A * (mu_obs - mu_theory)
            # where A = C^{-1} - C^{-1}*1*1^T*C^{-1} / (1^T*C^{-1}*1)
            
            delta_mu = self.mu_obs - mu_theory
            chi2_full = jnp.dot(delta_mu, jnp.dot(self.inv_cov, delta_mu))
            
            # Correction term for marginalization
            correction = (jnp.dot(delta_mu, self.inv_cov_ones)**2) / self.ones_inv_cov_ones
            
            chi2 = chi2_full - correction
            
            # Add marginalization normalization term
            log_norm = -0.5 * jnp.log(2 * jnp.pi * self.ones_inv_cov_ones)
        else:
            # Standard chi-squared without marginalization
            chi2 = jnp.dot(residuals, jnp.dot(self.inv_cov, residuals))
            log_norm = 0.0
        
        # Log-likelihood
        n_data = len(self.mu_obs)
        log_like = -0.5 * chi2 - 0.5 * n_data * jnp.log(2 * jnp.pi) + log_norm
        
        # Add log determinant term (constant for fixed covariance)
        # log_like -= 0.5 * jnp.log(jnp.linalg.det(self.cov_matrix))
        
        return log_like


class BAOLikelihood(BaseLikelihood):
    """
    Baryon Acoustic Oscillations likelihood.
    
    Computes likelihood for BAO distance measurements including
    D_V/r_s, D_A/r_s, and H*r_s observables.
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initialize BAO likelihood.
        
        Parameters
        ----------
        dataset : Dataset
            BAO dataset
        """
        super().__init__(dataset)
        
        if dataset.data_type != 'bao':
            raise ValueError("Dataset must be of type 'bao'")
        
        # Extract BAO measurements
        self.z_eff = dataset.data[:, 0]          # Effective redshifts
        self.DV_over_rs_obs = dataset.data[:, 1]  # D_V/r_s observations
        self.DA_over_rs_obs = dataset.data[:, 2]  # D_A/r_s observations  
        self.H_times_rs_obs = dataset.data[:, 3]  # H*r_s observations
        
        # Flatten observables and uncertainties for covariance handling
        self.observables_obs = jnp.concatenate([
            self.DV_over_rs_obs,
            self.DA_over_rs_obs,
            self.H_times_rs_obs
        ])
        
        # Covariance matrix
        if dataset.covariance is not None:
            self.cov_matrix = dataset.covariance
            self.inv_cov = jnp.linalg.inv(self.cov_matrix)
        else:
            # Diagonal from uncertainties
            uncertainties_flat = jnp.concatenate([
                dataset.uncertainties[:, 1],  # DV/rs errors
                dataset.uncertainties[:, 2],  # DA/rs errors
                dataset.uncertainties[:, 3]   # H*rs errors
            ])
            self.cov_matrix = jnp.diag(uncertainties_flat**2)
            self.inv_cov = jnp.diag(1.0 / uncertainties_flat**2)
        
        # Fiducial sound horizon (from metadata)
        self.rs_fid = dataset.metadata.get('sound_horizon_fid', 147.78)
    
    @jit
    def log_likelihood(self, theory_bao: Dict[str, jnp.ndarray]) -> float:
        """
        Compute BAO log-likelihood.
        
        Parameters
        ----------
        theory_bao : dict
            Dictionary with theoretical BAO predictions:
            - 'DV_over_rs': D_V/r_s values
            - 'DA_over_rs': D_A/r_s values  
            - 'H_times_rs': H*r_s values
            
        Returns
        -------
        float
            Log-likelihood value
        """
        # Combine theoretical predictions
        theory_observables = jnp.concatenate([
            theory_bao['DV_over_rs'],
            theory_bao['DA_over_rs'],
            theory_bao['H_times_rs']
        ])
        
        # Residuals
        residuals = self.observables_obs - theory_observables
        
        # Chi-squared
        chi2 = jnp.dot(residuals, jnp.dot(self.inv_cov, residuals))
        
        # Log-likelihood
        n_data = len(self.observables_obs)
        log_like = -0.5 * chi2 - 0.5 * n_data * jnp.log(2 * jnp.pi)
        
        return log_like


class CMBLikelihood(BaseLikelihood):
    """
    CMB parameter likelihood.
    
    Simplified CMB likelihood using Gaussian approximation
    for cosmological parameters from CMB data.
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initialize CMB likelihood.
        
        Parameters
        ---------- 
        dataset : Dataset
            CMB parameter dataset
        """
        super().__init__(dataset)
        
        if dataset.data_type != 'cmb':
            raise ValueError("Dataset must be of type 'cmb'")
        
        # Parameter names and values
        self.param_names = dataset.metadata['parameter_names']
        self.param_means = dataset.data
        self.param_uncertainties = dataset.uncertainties
        
        # Covariance matrix
        self.cov_matrix = dataset.covariance
        self.inv_cov = jnp.linalg.inv(self.cov_matrix)
    
    @jit
    def log_likelihood(self, param_values: jnp.ndarray) -> float:
        """
        Compute CMB parameter log-likelihood.
        
        Parameters
        ----------
        param_values : jnp.ndarray
            Parameter values in same order as param_names
            
        Returns
        -------
        float
            Log-likelihood value
        """
        # Residuals
        residuals = param_values - self.param_means
        
        # Chi-squared
        chi2 = jnp.dot(residuals, jnp.dot(self.inv_cov, residuals))
        
        # Log-likelihood
        n_params = len(param_values)
        log_like = -0.5 * chi2 - 0.5 * n_params * jnp.log(2 * jnp.pi)
        
        return log_like


class H0Likelihood(BaseLikelihood):
    """
    Local H0 measurement likelihood.
    
    Simple Gaussian likelihood for local Hubble constant measurements.
    """
    
    def __init__(self, dataset: Dataset):
        """
        Initialize H0 likelihood.
        
        Parameters
        ----------
        dataset : Dataset
            H0 measurement dataset
        """
        super().__init__(dataset)
        
        if dataset.data_type != 'h0':
            raise ValueError("Dataset must be of type 'h0'")
        
        self.H0_obs = dataset.data[0]
        self.H0_uncertainty = dataset.uncertainties[0]
    
    @jit
    def log_likelihood(self, H0_theory: float) -> float:
        """
        Compute H0 log-likelihood.
        
        Parameters
        ----------
        H0_theory : float
            Theoretical H0 value
            
        Returns
        -------
        float
            Log-likelihood value
        """
        residual = H0_theory - self.H0_obs
        chi2 = (residual / self.H0_uncertainty)**2
        
        log_like = -0.5 * chi2 - 0.5 * jnp.log(2 * jnp.pi * self.H0_uncertainty**2)
        
        return log_like


class LikelihoodCalculator:
    """
    Main likelihood calculator for cosmological parameter estimation.
    
    Coordinates computation of theoretical predictions and likelihood
    evaluations for multiple observational probes.
    """
    
    def __init__(self, 
                 background: BackgroundEvolution,
                 linear_power: Optional[LinearPowerSpectrum] = None,
                 data_manager: Optional[ObservationalDataManager] = None):
        """
        Initialize likelihood calculator.
        
        Parameters
        ----------
        background : BackgroundEvolution
            Background cosmology calculator
        linear_power : LinearPowerSpectrum, optional
            Power spectrum calculator
        data_manager : ObservationalDataManager, optional
            Observational data manager
        """
        self.background = background
        self.linear_power = linear_power
        self.data_manager = data_manager or ObservationalDataManager()
        
        # Storage for likelihood objects
        self.likelihoods: Dict[str, BaseLikelihood] = {}
        
        # Pre-compile theory prediction functions
        self._compile_theory_functions()
    
    def _compile_theory_functions(self):
        """Pre-compile theory calculation functions."""
        
        # Distance modulus calculation
        @jit
        def distance_modulus_single(z, H0):
            D_L = self.background.distances.luminosity_distance(z)
            # Rescale by H0 (distance proportional to c/H0)
            D_L_scaled = D_L * (67.4 / H0)  # Assuming fiducial H0=67.4
            return 5 * jnp.log10(D_L_scaled / 1e-5)  # 10 pc = 1e-5 Mpc
        
        self._distance_modulus_jit = jit(vmap(distance_modulus_single, in_axes=(0, None)))
        
        # BAO observables
        @jit
        def bao_observables_single(z, rs):
            D_A = self.background.distances.angular_diameter_distance(z)
            H_z = self.background.H_z(z)
            
            # Volume-averaged distance
            D_V = (z * D_A**2 * 299792.458 / H_z)**(1/3)  # c/H(z) factor
            
            return D_V / rs, D_A / rs, H_z * rs / 299792.458
        
        self._bao_observables_jit = jit(vmap(bao_observables_single, in_axes=(0, None)))
    
    # ==================== Theory Predictions ====================
    
    def compute_sne_theory(self, z_array: jnp.ndarray, H0: float) -> jnp.ndarray:
        """
        Compute theoretical distance moduli for supernovae.
        
        Parameters
        ----------
        z_array : jnp.ndarray
            Redshift array
        H0 : float
            Hubble constant
            
        Returns
        -------
        jnp.ndarray
            Theoretical distance moduli
        """
        return self._distance_modulus_jit(z_array, H0)
    
    def compute_bao_theory(self, z_array: jnp.ndarray, rs: float) -> Dict[str, jnp.ndarray]:
        """
        Compute theoretical BAO observables.
        
        Parameters
        ----------
        z_array : jnp.ndarray
            Effective redshift array
        rs : float
            Sound horizon at drag epoch
            
        Returns
        -------
        dict
            BAO observables dictionary
        """
        DV_rs, DA_rs, H_rs = self._bao_observables_jit(z_array, rs)
        
        return {
            'DV_over_rs': DV_rs,
            'DA_over_rs': DA_rs,
            'H_times_rs': H_rs
        }
    
    def compute_cmb_theory(self, cosmological_params: Dict[str, float]) -> jnp.ndarray:
        """
        Compute theoretical CMB parameters.
        
        Parameters
        ----------
        cosmological_params : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        jnp.ndarray
            Theoretical parameter values
        """
        # Extract parameters in the order expected by CMB likelihood
        # This would typically involve running the full CMB calculation
        # For now, return the input parameters (identity mapping)
        
        param_names = ['Omega_b_h2', 'Omega_c_h2', 'theta_s', 'tau_reio', 'ln_A_s_1e10', 'n_s']
        
        theory_values = []
        for param in param_names:
            if param == 'Omega_b_h2':
                Omega_b = cosmological_params.get('Omega_b', 0.049)
                h = cosmological_params.get('h', 0.674)
                theory_values.append(Omega_b * h**2)
            elif param == 'Omega_c_h2':
                Omega_m = cosmological_params.get('Omega_m', 0.315)
                Omega_b = cosmological_params.get('Omega_b', 0.049)
                h = cosmological_params.get('h', 0.674)
                Omega_c = Omega_m - Omega_b
                theory_values.append(Omega_c * h**2)
            elif param == 'theta_s':
                # Sound horizon angle (would need CMB calculation)
                theory_values.append(cosmological_params.get('theta_s', 1.04092))
            else:
                theory_values.append(cosmological_params.get(param, 0.0))
        
        return jnp.array(theory_values)
    
    # ==================== Likelihood Management ====================
    
    def add_sne_likelihood(self, dataset_name: str = 'pantheon_plus'):
        """
        Add Type Ia supernovae likelihood.
        
        Parameters
        ----------
        dataset_name : str
            Name of SNe dataset to load
        """
        if dataset_name not in self.data_manager.datasets:
            if dataset_name == 'pantheon_plus':
                dataset = self.data_manager.load_pantheon_plus_sne()
            else:
                raise ValueError(f"Unknown SNe dataset: {dataset_name}")
        else:
            dataset = self.data_manager.datasets[dataset_name]
        
        likelihood = SupernovaeLikelihood(dataset, marginalize_mb=True)
        self.likelihoods[f'sne_{dataset_name}'] = likelihood
    
    def add_bao_likelihood(self, dataset_name: str = 'boss_bao'):
        """
        Add BAO likelihood.
        
        Parameters
        ----------
        dataset_name : str
            Name of BAO dataset to load
        """
        if dataset_name not in self.data_manager.datasets:
            if dataset_name == 'boss_bao':
                dataset = self.data_manager.load_boss_bao_data()
            else:
                raise ValueError(f"Unknown BAO dataset: {dataset_name}")
        else:
            dataset = self.data_manager.datasets[dataset_name]
        
        likelihood = BAOLikelihood(dataset)
        self.likelihoods[f'bao_{dataset_name}'] = likelihood
    
    def add_cmb_likelihood(self, dataset_name: str = 'planck_cmb'):
        """
        Add CMB likelihood.
        
        Parameters
        ----------
        dataset_name : str
            Name of CMB dataset to load
        """
        if dataset_name not in self.data_manager.datasets:
            if dataset_name == 'planck_cmb':
                dataset = self.data_manager.load_planck_cmb_data()
            else:
                raise ValueError(f"Unknown CMB dataset: {dataset_name}")
        else:
            dataset = self.data_manager.datasets[dataset_name]
        
        likelihood = CMBLikelihood(dataset)
        self.likelihoods[f'cmb_{dataset_name}'] = likelihood
    
    def add_h0_likelihood(self, dataset_name: str = 'local_h0'):
        """
        Add H0 likelihood.
        
        Parameters
        ----------
        dataset_name : str
            Name of H0 dataset to load
        """
        if dataset_name not in self.data_manager.datasets:
            if dataset_name == 'local_h0':
                dataset = self.data_manager.load_h0_measurement()
            else:
                raise ValueError(f"Unknown H0 dataset: {dataset_name}")
        else:
            dataset = self.data_manager.datasets[dataset_name]
        
        likelihood = H0Likelihood(dataset)
        self.likelihoods[f'h0_{dataset_name}'] = likelihood
    
    # ==================== Combined Likelihood ====================
    
    @jit
    def total_log_likelihood(self, cosmological_params: Dict[str, float]) -> float:
        """
        Compute total log-likelihood from all active likelihoods.
        
        Parameters
        ----------
        cosmological_params : dict
            Dictionary of cosmological parameters
            
        Returns
        -------
        float
            Total log-likelihood
        """
        total_log_like = 0.0
        
        # Extract common parameters
        H0 = cosmological_params.get('H0', 67.4)
        rs = cosmological_params.get('rs_drag', 147.78)
        
        # Loop through active likelihoods
        for name, likelihood in self.likelihoods.items():
            
            if isinstance(likelihood, SupernovaeLikelihood):
                # Compute SNe theory
                mu_theory = self.compute_sne_theory(likelihood.z_obs, H0)
                log_like = likelihood.log_likelihood(mu_theory)
                
            elif isinstance(likelihood, BAOLikelihood):
                # Compute BAO theory
                bao_theory = self.compute_bao_theory(likelihood.z_eff, rs)
                log_like = likelihood.log_likelihood(bao_theory)
                
            elif isinstance(likelihood, CMBLikelihood):
                # Compute CMB theory
                cmb_theory = self.compute_cmb_theory(cosmological_params)
                log_like = likelihood.log_likelihood(cmb_theory)
                
            elif isinstance(likelihood, H0Likelihood):
                # H0 theory is just the parameter value
                log_like = likelihood.log_likelihood(H0)
                
            else:
                continue
            
            total_log_like += log_like
        
        return total_log_like
    
    # ==================== Utility Methods ====================
    
    def likelihood_summary(self) -> str:
        """
        Generate summary of active likelihoods.
        
        Returns
        -------
        str
            Formatted summary
        """
        if not self.likelihoods:
            return "No likelihoods active."
        
        lines = [
            "Active Likelihoods",
            "=" * 20,
        ]
        
        for name, likelihood in self.likelihoods.items():
            dataset = likelihood.dataset
            lines.extend([
                f"Likelihood: {name}",
                f"  Dataset: {dataset.name}",
                f"  Type: {dataset.data_type}",
                f"  Data points: {dataset.data.shape[0]}",
                f"  Description: {dataset.metadata.get('description', 'N/A')}",
                ""
            ])
        
        return "\n".join(lines)
    
    def remove_likelihood(self, name: str):
        """Remove likelihood by name."""
        if name in self.likelihoods:
            del self.likelihoods[name]
        else:
            raise ValueError(f"Likelihood {name} not found")
    
    def clear_likelihoods(self):
        """Clear all likelihoods."""
        self.likelihoods.clear()
    
    def list_active_likelihoods(self) -> List[str]:
        """
        List names of active likelihoods.
        
        Returns
        -------
        List[str]
            Active likelihood names
        """
        return list(self.likelihoods.keys())