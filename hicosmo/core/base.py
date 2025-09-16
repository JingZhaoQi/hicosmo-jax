"""
HIcosmo Base Classes - Simplified and Clean
===========================================

Minimal base classes for all cosmological models in HIcosmo.
This replaces the bloated cosmology.py with a clean, focused interface.

Key principles:
- Single responsibility
- No implementation details in base class
- Clear interface contracts
- No redundant methods
"""

from abc import ABC, abstractmethod
from typing import Dict, Union
import jax.numpy as jnp


class CosmologyBase(ABC):
    """
    Minimal abstract base class for all cosmological models.
    
    Defines the essential interface that all models must implement,
    without any concrete implementations to avoid code duplication.
    """
    
    def __init__(self, **params):
        """Initialize cosmological model with parameters."""
        self.params = params.copy()
        self._validate_params()
        
    @abstractmethod
    def _validate_params(self) -> None:
        """Validate input parameters for physical consistency."""
        pass
    
    # ==================== Core Interface ====================
    
    @abstractmethod
    def E_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.
        
        This is the fundamental quantity that defines the cosmological model.
        All other quantities are derived from this.
        """
        pass
    
    @abstractmethod
    def comoving_distance(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Comoving distance in Mpc.
        
        Must be implemented with high performance integration.
        """
        pass
    
    @abstractmethod  
    def angular_diameter_distance(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Angular diameter distance in Mpc."""
        pass
    
    @abstractmethod
    def luminosity_distance(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Luminosity distance in Mpc."""
        pass
    
    @abstractmethod
    def distance_modulus(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """Distance modulus in magnitudes."""
        pass
    
    # ==================== Optional Interface ====================
    
    def w_z(self, z: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
        """
        Dark energy equation of state w(z).
        
        Default implementation for ΛCDM (w = -1).
        Can be overridden by extensions.
        """
        return jnp.full_like(jnp.asarray(z), -1.0)
    
    def get_parameters(self) -> Dict[str, float]:
        """Get copy of model parameters."""
        return self.params.copy()
    
    def update_parameters(self, new_params: Dict[str, float]) -> None:
        """
        Update model parameters.
        
        Subclasses should override this to handle parameter updates properly.
        """
        self.params.update(new_params)
        self._validate_params()


class DistanceCalculator(ABC):
    """
    Abstract base for distance calculation engines.
    
    This allows different integration methods to be plugged into models
    without changing the model interface.
    """
    
    @abstractmethod
    def comoving_distance(
        self, 
        z: Union[float, jnp.ndarray],
        E_z_func: callable
    ) -> Union[float, jnp.ndarray]:
        """Calculate comoving distance given E(z) function."""
        pass


# Type aliases for clarity
Redshift = Union[float, jnp.ndarray]
Distance = Union[float, jnp.ndarray]
Parameters = Dict[str, float]