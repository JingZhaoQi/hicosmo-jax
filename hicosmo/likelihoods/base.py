"""
Base likelihood class compatible with Cobaya and optimized for JAX.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import jax.numpy as jnp
from jax import jit, grad
import yaml
import os
from pathlib import Path


class Likelihood(ABC):
    """
    Abstract base class for all likelihoods in HIcosmo.
    
    This class is designed to be compatible with Cobaya's likelihood interface
    while leveraging JAX for automatic differentiation and GPU acceleration.
    """
    
    def __init__(self, name: Optional[str] = None, 
                 data_path: Optional[str] = None,
                 **kwargs):
        """
        Initialize the likelihood.
        
        Args:
            name: Name of the likelihood
            data_path: Path to data files
            **kwargs: Additional configuration parameters
        """
        self.name = name or self.__class__.__name__
        self.data_path = data_path
        self.config = kwargs
        self._initialized = False
        self._nuisance_params = {}
        self._requirements = {}
        
    def initialize(self) -> None:
        """
        Initialize the likelihood by loading data and covariance matrix.
        This method should be called before first use.
        """
        if self._initialized:
            return
            
        self._load_data()
        self._setup_covariance()
        self._initialize_nuisance()
        self._initialized = True
    
    @abstractmethod
    def _load_data(self) -> None:
        """
        Load observational data.
        Must be implemented by derived classes.
        """
        pass
    
    @abstractmethod
    def _setup_covariance(self) -> None:
        """
        Setup covariance matrix and its inverse.
        Must be implemented by derived classes.
        """
        pass
    
    def _initialize_nuisance(self) -> None:
        """
        Initialize nuisance parameters.
        Can be overridden by derived classes that have nuisance parameters.
        """
        pass
    
    @abstractmethod
    def get_requirements(self) -> Dict[str, Any]:
        """
        Return the cosmological quantities required by this likelihood.
        
        This method tells the sampler what cosmological quantities need to be
        computed for this likelihood.
        
        Example return value:
        {
            'luminosity_distance': {'z': self.z_data},
            'H': {'z': self.z_data}
        }
        
        Returns:
            Dictionary of required quantities
        """
        pass
    
    def get_nuisance_params(self) -> Dict[str, Dict[str, Any]]:
        """
        Return the nuisance parameters for this likelihood.
        
        Returns:
            Dictionary of nuisance parameters with their properties
            Example: {'M': {'prior': {'min': -20, 'max': -18}, 'ref': -19.3}}
        """
        return self._nuisance_params
    
    @abstractmethod
    def theory(self, **kwargs) -> jnp.ndarray:
        """
        Compute theoretical prediction given cosmological parameters.
        
        Args:
            **kwargs: Cosmological quantities computed by the theory code
            
        Returns:
            Theory vector matching the data vector shape
        """
        pass
    
    @jit
    def chi2(self, theory_vec: jnp.ndarray, data_vec: jnp.ndarray, 
             inv_cov: jnp.ndarray) -> float:
        """
        Compute chi-squared statistic.
        
        Args:
            theory_vec: Theoretical predictions
            data_vec: Observational data
            inv_cov: Inverse covariance matrix
            
        Returns:
            Chi-squared value
        """
        residual = data_vec - theory_vec
        return jnp.dot(residual, jnp.dot(inv_cov, residual))
    
    def logp(self, **params_values) -> float:
        """
        Compute log-likelihood.
        
        This is the main method called by the sampler.
        
        Args:
            **params_values: Dictionary of all parameters (cosmological + nuisance)
            
        Returns:
            Log-likelihood value
        """
        if not self._initialized:
            self.initialize()
        
        # Compute theory vector
        theory_vec = self.theory(**params_values)
        
        # Compute chi2
        chi2_val = self.chi2(theory_vec, self.data_vec, self.inv_cov)
        
        # Return log-likelihood
        return -0.5 * chi2_val
    
    def logp_grad(self, **params_values) -> Tuple[float, Dict[str, float]]:
        """
        Compute log-likelihood and its gradient.
        
        This method uses JAX's automatic differentiation.
        
        Args:
            **params_values: Dictionary of all parameters
            
        Returns:
            Tuple of (log-likelihood, gradient dictionary)
        """
        # Create a function that takes a single array and returns scalar
        param_names = list(params_values.keys())
        param_array = jnp.array([params_values[name] for name in param_names])
        
        def logp_func(param_array):
            params_dict = {name: param_array[i] for i, name in enumerate(param_names)}
            return self.logp(**params_dict)
        
        # Compute value and gradient
        logp_val, grad_array = jax.value_and_grad(logp_func)(param_array)
        
        # Convert gradient back to dictionary
        grad_dict = {name: grad_array[i] for i, name in enumerate(param_names)}
        
        return logp_val, grad_dict
    
    @classmethod
    def from_yaml(cls, yaml_file: str) -> 'Likelihood':
        """
        Create likelihood instance from YAML configuration file.
        
        Args:
            yaml_file: Path to YAML configuration file
            
        Returns:
            Likelihood instance
        """
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
        return cls(**config)
    
    def to_yaml(self, yaml_file: str) -> None:
        """
        Save likelihood configuration to YAML file.
        
        Args:
            yaml_file: Path to output YAML file
        """
        config = {
            'name': self.name,
            'data_path': self.data_path,
            **self.config
        }
        with open(yaml_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class GaussianLikelihood(Likelihood):
    """
    Base class for Gaussian likelihoods with simple chi2 computation.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data_vec = None
        self.cov = None
        self.inv_cov = None
        
    def _setup_covariance(self) -> None:
        """
        Setup covariance matrix and compute its inverse.
        Assumes self.cov has been set by _load_data().
        """
        if self.cov is None:
            raise ValueError("Covariance matrix not loaded")
        
        # Convert to JAX array
        self.cov = jnp.array(self.cov)
        
        # Compute inverse (with regularization for numerical stability)
        try:
            self.inv_cov = jnp.linalg.inv(self.cov)
        except np.linalg.LinAlgError:
            # Add small regularization
            eps = 1e-10
            self.inv_cov = jnp.linalg.inv(self.cov + eps * jnp.eye(len(self.cov)))
    
    def validate_data(self) -> bool:
        """
        Validate loaded data and covariance matrix.
        
        Returns:
            True if data is valid, raises ValueError otherwise
        """
        if self.data_vec is None:
            raise ValueError("Data vector not loaded")
        
        if self.cov is None:
            raise ValueError("Covariance matrix not loaded")
        
        n_data = len(self.data_vec)
        if self.cov.shape != (n_data, n_data):
            raise ValueError(f"Covariance shape {self.cov.shape} doesn't match data length {n_data}")
        
        # Check if covariance is symmetric
        if not jnp.allclose(self.cov, self.cov.T):
            raise ValueError("Covariance matrix is not symmetric")
        
        # Check if covariance is positive definite
        eigenvals = jnp.linalg.eigvalsh(self.cov)
        if jnp.any(eigenvals <= 0):
            raise ValueError("Covariance matrix is not positive definite")
        
        return True