"""
Unified Parameter Management for HIcosmo
========================================

Single source of truth for all cosmological parameter handling.
Replaces multiple parameter management systems with one clean interface.

Features:
- Parameter validation
- Default values  
- Type checking
- Range validation
- Derived parameter calculation
"""

from typing import Dict, Optional, Union, List, Any
from dataclasses import dataclass
import warnings
import numpy as np
import jax.numpy as jnp
import jax
import jax.errors


@dataclass
class ParameterSpec:
    """Specification for a cosmological parameter."""
    default: float
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    description: str = ""
    unit: str = ""


class CosmologicalParameters:
    """
    Unified cosmological parameter manager.
    
    Handles validation, defaults, and derived parameters for all models.
    """
    
    # Parameter specifications
    PARAM_SPECS = {
        # Primary parameters
        'H0': ParameterSpec(67.36, 20.0, 200.0, "Hubble constant", "km/s/Mpc"),
        'Omega_m': ParameterSpec(0.3153, 0.01, 1.0, "Total matter density parameter", ""),
        'Omega_b': ParameterSpec(0.0493, 0.005, 0.1, "Baryon density parameter", ""),
        'Omega_k': ParameterSpec(0.0, -0.5, 0.5, "Curvature density parameter", ""),
        'sigma8': ParameterSpec(0.8111, 0.1, 2.0, "Matter fluctuation amplitude", ""),
        'n_s': ParameterSpec(0.9649, 0.8, 1.2, "Scalar spectral index", ""),
        
        # CMB and recombination  
        'T_cmb': ParameterSpec(2.7255, 2.0, 3.0, "CMB temperature", "K"),
        'N_eff': ParameterSpec(3.046, 1.0, 10.0, "Effective neutrino species", ""),
        'Y_p': ParameterSpec(0.2453, 0.2, 0.3, "Primordial helium fraction", ""),
        
        # Dark energy
        'w0': ParameterSpec(-1.0, -3.0, 0.0, "Dark energy EoS today", ""),
        'wa': ParameterSpec(0.0, -3.0, 3.0, "Dark energy EoS evolution", ""),
        
        # Neutrinos
        'mnu': ParameterSpec(0.06, 0.0, 2.0, "Sum of neutrino masses", "eV"),
        'Neff': ParameterSpec(3.046, 1.0, 10.0, "Effective neutrino number", ""),
        
        # Modified gravity
        'mu_0': ParameterSpec(0.0, -1.0, 1.0, "Modified gravity parameter", ""),
        'sigma_0': ParameterSpec(0.0, -1.0, 1.0, "Modified gravity parameter", ""),
    }
    
    def __init__(self, **params):
        """
        Initialize parameters.
        
        Parameters
        ----------
        **params : dict
            Cosmological parameters to set
        """
        self._params = {}
        self._derived = {}
        
        # Set defaults first
        for name, spec in self.PARAM_SPECS.items():
            self._params[name] = spec.default
            
        # Override with user values
        for name, value in params.items():
            self.set_parameter(name, value)
            
        # Compute derived parameters
        self._compute_derived()
        
    def set_parameter(self, name: str, value) -> None:
        """Set a parameter with validation. JAX-compatible - accepts tracers."""
        if name not in self.PARAM_SPECS:
            warnings.warn(f"Unknown parameter '{name}'. Adding without validation.")
            self._params[name] = value  # No float() conversion!
            return

        spec = self.PARAM_SPECS[name]
        # NO float() conversion - JAX tracer compatibility!

        # Skip validation for JAX tracers (during MCMC)
        try:
            # Try range validation only if concrete value
            concrete_value = float(value)
            if spec.min_val is not None and concrete_value < spec.min_val:
                raise ValueError(f"{name} = {concrete_value} below minimum {spec.min_val}")
            if spec.max_val is not None and concrete_value > spec.max_val:
                raise ValueError(f"{name} = {concrete_value} above maximum {spec.max_val}")
        except (TypeError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
            # JAX tracer - skip validation during MCMC sampling
            pass

        self._params[name] = value  # Store original value (tracer or float)
        # Skip derived computation for tracers
        try:
            float(value)
            self._compute_derived()
        except (TypeError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
            pass
        
    def get_parameter(self, name: str) -> float:
        """Get parameter value."""
        if name in self._params:
            return self._params[name]
        elif name in self._derived:
            return self._derived[name]
        else:
            raise KeyError(f"Parameter '{name}' not found")
            
    def get_all_parameters(self) -> Dict[str, float]:
        """Get all parameters (primary + derived)."""
        result = self._params.copy()
        result.update(self._derived)
        return result
        
    def _compute_derived(self) -> None:
        """Compute derived parameters."""
        params = self._params
        
        # Hubble parameter  
        self._derived['h'] = params['H0'] / 100.0
        
        # Radiation density
        T_cmb = params['T_cmb']
        N_eff = params['N_eff']
        h = self._derived['h']
        
        # Photon density parameter
        Omega_gamma_h2 = 2.47e-5 * (T_cmb / 2.7255)**4
        self._derived['Omega_gamma'] = Omega_gamma_h2 / h**2
        
        # Total radiation (photons + neutrinos)
        self._derived['Omega_r'] = self._derived['Omega_gamma'] * (1 + 0.2271 * N_eff)
        
        # Dark matter
        self._derived['Omega_c'] = params['Omega_m'] - params['Omega_b']
        
        # Dark energy (enforces closure)
        self._derived['Omega_Lambda'] = (
            1.0 - params['Omega_m'] - params['Omega_k'] - self._derived['Omega_r']
        )
        
        # Hubble distance and time
        from ..utils.constants import c_km_s, Gyr
        self._derived['D_H'] = c_km_s / params['H0']  # Mpc
        self._derived['t_H'] = 9.777952 / h  # Gyr (1/H0 in appropriate units)
        
        # Age of universe (rough approximation)
        Om = params['Omega_m']
        OL = self._derived['Omega_Lambda']
        # Use JAX-compatible conditional for tracers
        self._derived['age'] = jnp.where(
            OL > 0,
            (2.0/3.0) * self._derived['t_H'] / jnp.sqrt(OL),
            (2.0/3.0) * self._derived['t_H']
        )
            
        # Critical density
        self._derived['rho_crit'] = 2.77536627e11 * h**2  # M_sun/Mpc^3
        
    def validate_closure(self, tolerance: float = 1e-6) -> None:
        """Validate closure relation."""
        total = (
            self._params['Omega_m'] +
            self._params['Omega_k'] +
            self._derived['Omega_r'] +
            self._derived['Omega_Lambda']
        )

        # Skip validation for JAX tracers during MCMC sampling
        try:
            concrete_total = float(total)
            if abs(concrete_total - 1.0) > tolerance:
                raise ValueError(f"Closure relation violated: Ω_total = {concrete_total:.6f} ≠ 1")
        except (TypeError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
            # JAX tracer - skip validation during MCMC sampling
            pass
            
    def validate_physics(self) -> None:
        """Check for unphysical parameter combinations."""
        # Skip validation for JAX tracers during MCMC sampling
        try:
            # Dark energy density should be positive
            OL = float(self._derived['Omega_Lambda'])
            if OL < -1e-6:
                raise ValueError(f"Omega_Lambda = {OL:.6f} < 0 unphysical")

            # Dark matter should be positive
            Oc = float(self._derived['Omega_c'])
            if Oc < -1e-6:
                raise ValueError(f"Omega_c = {Oc:.6f} < 0 unphysical")

            # Baryon fraction should be reasonable
            Ob = float(self._params['Omega_b'])
            Om = float(self._params['Omega_m'])
            if Ob >= Om:
                raise ValueError("Omega_b >= Omega_m unphysical")
        except (TypeError, jax.errors.TracerIntegerConversionError, jax.errors.ConcretizationTypeError):
            # JAX tracer - skip validation during MCMC sampling
            pass
            
    def summary(self) -> str:
        """Generate parameter summary."""
        lines = ["Cosmological Parameters Summary"]
        lines.append("=" * 40)
        
        # Primary parameters
        lines.append("\nPrimary Parameters:")
        for name in ['H0', 'Omega_m', 'Omega_b', 'Omega_k', 'sigma8', 'n_s']:
            if name in self._params:
                spec = self.PARAM_SPECS[name] 
                value = self._params[name]
                lines.append(f"  {name:<12} = {value:8.4f}  {spec.unit}")
                
        # Derived parameters
        lines.append("\nDerived Parameters:")
        derived_show = ['h', 'Omega_c', 'Omega_r', 'Omega_Lambda', 'D_H', 'age']
        for name in derived_show:
            if name in self._derived:
                value = self._derived[name]
                if name == 'D_H':
                    lines.append(f"  {name:<12} = {value:8.1f}  Mpc")
                elif name == 'age':
                    lines.append(f"  {name:<12} = {value:8.2f}  Gyr")
                else:
                    lines.append(f"  {name:<12} = {value:8.4f}")
                    
        return "\n".join(lines)
        
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary (for compatibility)."""
        return self.get_all_parameters()
        
    def __getitem__(self, key: str) -> float:
        """Dictionary-like access."""
        return self.get_parameter(key)
        
    def __setitem__(self, key: str, value: float) -> None:
        """Dictionary-like setting."""
        self.set_parameter(key, value)
        
    def __contains__(self, key: str) -> bool:
        """Check if parameter exists."""
        return key in self._params or key in self._derived


# Default parameter sets
PLANCK_2018 = CosmologicalParameters(
    H0=67.36, Omega_m=0.3153, Omega_b=0.0493, Omega_k=0.0,
    sigma8=0.8111, n_s=0.9649, T_cmb=2.7255
)

PLANCK_2015 = CosmologicalParameters(
    H0=67.74, Omega_m=0.3089, Omega_b=0.0486, Omega_k=0.0,
    sigma8=0.8159, n_s=0.9667, T_cmb=2.7255
)

WMAP9 = CosmologicalParameters(
    H0=69.32, Omega_m=0.2865, Omega_b=0.0463, Omega_k=0.0,
    sigma8=0.820, n_s=0.9608, T_cmb=2.725
)