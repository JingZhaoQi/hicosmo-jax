#!/usr/bin/env python3
"""
Parameter Configuration System for HiCosmo MCMC.

This module provides a dictionary-driven parameter management system that bridges
user-friendly configuration with NumPyro's flexible model definition.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import numpyro.distributions as dist
from enum import Enum


class PriorType(Enum):
    """Supported prior distribution types."""
    UNIFORM = "uniform"
    NORMAL = "normal" 
    TRUNCNORM = "truncnorm"
    LOGNORMAL = "lognormal"
    BETA = "beta"
    GAMMA = "gamma"
    HALF_NORMAL = "halfnormal"
    HALF_CAUCHY = "halfcauchy"
    EXPONENTIAL = "exponential"


@dataclass
class AutoParameter:
    """
    Automatic parameter definition with validation and prior specification.
    
    This class encapsulates all information needed to define a parameter for MCMC
    sampling, including prior distributions, bounds, reference values, and metadata.
    
    Parameters
    ----------
    name : str
        Parameter name (must be valid Python identifier).
    prior : Dict[str, Any]
        Prior distribution configuration.
    ref : Optional[float]
        Reference value for initialization.
    bounds : Optional[Tuple[float, float]]
        Hard bounds for parameter (min, max).
    latex : Optional[str]
        LaTeX representation for plotting.
    description : Optional[str]
        Human-readable description.
    derived : bool
        Whether this is a derived parameter.
    
    Examples
    --------
    >>> # Uniform prior
    >>> H0 = AutoParameter(
    ...     name="H0",
    ...     prior={"dist": "uniform", "min": 50, "max": 100},
    ...     ref=70.0,
    ...     latex=r"H_0"
    ... )
    
    >>> # Normal prior with bounds
    >>> Omega_m = AutoParameter(
    ...     name="Omega_m", 
    ...     prior={"dist": "normal", "loc": 0.3, "scale": 0.05},
    ...     bounds=(0.1, 0.5),
    ...     latex=r"\\Omega_m"
    ... )
    
    >>> # Truncated normal (automatically inferred from bounds + normal)
    >>> sigma8 = AutoParameter(
    ...     name="sigma8",
    ...     prior={"dist": "truncnorm", "loc": 0.8, "scale": 0.1, "low": 0.5, "high": 1.2}
    ... )
    """
    
    name: str
    prior: Dict[str, Any]
    ref: Optional[float] = None
    bounds: Optional[Tuple[float, float]] = None
    latex: Optional[str] = None
    description: Optional[str] = None
    derived: bool = False
    
    def __post_init__(self):
        """Validate parameter configuration after initialization."""
        self._validate_name()
        self._validate_prior()
        self._validate_bounds()
        self._set_defaults()
    
    def _validate_name(self):
        """Validate parameter name is a valid identifier."""
        if not self.name.isidentifier():
            raise ValueError(f"Parameter name '{self.name}' is not a valid Python identifier")
    
    def _validate_prior(self):
        """Validate prior distribution configuration."""
        if not isinstance(self.prior, dict):
            raise ValueError(f"Prior for '{self.name}' must be a dictionary")
        
        if "dist" not in self.prior:
            raise ValueError(f"Prior for '{self.name}' must specify 'dist' key")
        
        dist_type = self.prior["dist"].lower()
        if dist_type not in [pt.value for pt in PriorType]:
            valid_types = [pt.value for pt in PriorType]
            raise ValueError(f"Unknown prior type '{dist_type}' for '{self.name}'. "
                           f"Valid types: {valid_types}")
        
        # Validate required parameters for each distribution type
        self._validate_prior_params(dist_type)
    
    def _validate_prior_params(self, dist_type: str):
        """Validate required parameters for specific distribution types."""
        required_params = {
            "uniform": ["min", "max"],
            "normal": ["loc", "scale"],
            "truncnorm": ["loc", "scale", "low", "high"],
            "lognormal": ["loc", "scale"],
            "beta": ["alpha", "beta"],
            "gamma": ["concentration", "rate"],
            "halfnormal": ["scale"],
            "halfcauchy": ["scale"],
            "exponential": ["rate"]
        }
        
        if dist_type in required_params:
            for param in required_params[dist_type]:
                if param not in self.prior:
                    raise ValueError(f"Prior '{dist_type}' for '{self.name}' missing required parameter '{param}'")
    
    def _validate_bounds(self):
        """Validate bounds are consistent with prior."""
        if self.bounds is not None:
            if len(self.bounds) != 2:
                raise ValueError(f"Bounds for '{self.name}' must be tuple of (min, max)")
            
            if self.bounds[0] >= self.bounds[1]:
                raise ValueError(f"Bounds for '{self.name}' must have min < max")
            
            # Check consistency with uniform priors
            if self.prior.get("dist") == "uniform":
                prior_min, prior_max = self.prior["min"], self.prior["max"]
                if self.bounds != (prior_min, prior_max):
                    raise ValueError(f"Bounds {self.bounds} inconsistent with uniform prior "
                                   f"({prior_min}, {prior_max}) for '{self.name}'")
    
    def _set_defaults(self):
        """Set reasonable defaults for missing optional fields."""
        if self.latex is None:
            self.latex = self.name
            
        if self.ref is None:
            self.ref = self._infer_reference_value()
    
    def _infer_reference_value(self) -> float:
        """Infer a reasonable reference value from prior."""
        dist_type = self.prior["dist"]
        
        if dist_type == "uniform":
            return (self.prior["min"] + self.prior["max"]) / 2
        elif dist_type in ["normal", "truncnorm", "lognormal"]:
            return self.prior["loc"]
        elif dist_type == "beta":
            # Mean of Beta(α, β) = α / (α + β)
            alpha, beta = self.prior["alpha"], self.prior["beta"]
            return alpha / (alpha + beta)
        elif dist_type == "gamma":
            # Mean of Gamma = concentration / rate
            return self.prior["concentration"] / self.prior["rate"]
        elif dist_type in ["halfnormal", "halfcauchy"]:
            return self.prior["scale"]
        elif dist_type == "exponential":
            return 1.0 / self.prior["rate"]  # Mean of exponential
        else:
            return 1.0  # Fallback
    
    def to_numpyro_dist(self):
        """Convert to NumPyro distribution object."""
        dist_type = self.prior["dist"]
        
        if dist_type == "uniform":
            return dist.Uniform(self.prior["min"], self.prior["max"])
        elif dist_type == "normal":
            return dist.Normal(self.prior["loc"], self.prior["scale"])
        elif dist_type == "truncnorm":
            return dist.TruncatedNormal(
                self.prior["loc"], 
                self.prior["scale"],
                low=self.prior["low"], 
                high=self.prior["high"]
            )
        elif dist_type == "lognormal":
            return dist.LogNormal(self.prior["loc"], self.prior["scale"])
        elif dist_type == "beta":
            beta_dist = dist.Beta(self.prior["alpha"], self.prior["beta"])
            # Scale to bounds if provided
            if self.bounds:
                # Transform Beta(0,1) to (min, max)
                low, high = self.bounds
                return dist.TransformedDistribution(
                    beta_dist,
                    dist.transforms.AffineTransform(
                        loc=low, 
                        scale=high - low
                    )
                )
            return beta_dist
        elif dist_type == "gamma":
            return dist.Gamma(self.prior["concentration"], self.prior["rate"])
        elif dist_type == "halfnormal":
            return dist.HalfNormal(self.prior["scale"])
        elif dist_type == "halfcauchy":
            return dist.HalfCauchy(self.prior["scale"])
        elif dist_type == "exponential":
            return dist.Exponential(self.prior["rate"])
        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], name: Optional[str] = None) -> 'AutoParameter':
        """Create AutoParameter from dictionary.
        
        Parameters
        ----------
        data : Dict[str, Any]
            Parameter data dictionary.
        name : Optional[str]
            Parameter name (can be provided separately or in data).
        """
        # If name is provided separately, add it to data
        if name and 'name' not in data:
            data = {**data, 'name': name}
        return cls(**data)
    
    @classmethod
    def from_tuple(
        cls,
        name: str,
        values: Union[Tuple, List, set]
    ) -> 'AutoParameter':
        """
        Create AutoParameter from simplified tuple/list/set format.
        
        Parameters
        ----------
        name : str
            Parameter name.
        values : tuple, list or set
            Parameter values in format: (initial, min, max, latex) or {initial, min, max, latex}
            Where latex is optional.
            
        Returns
        -------
        AutoParameter
            Configured parameter object.
            
        Examples
        --------
        >>> param = AutoParameter.from_tuple('a', (3.5, 0, 10, '$a$'))
        >>> param = AutoParameter.from_tuple('b', [2.0, 0, 4])
        >>> param = AutoParameter.from_tuple('c', {1.0, 0, 2, '$c$'})
        """
        # Convert set to list if needed (sets don't preserve order)
        if isinstance(values, set):
            # Separate strings and numbers
            str_vals = [v for v in values if isinstance(v, str)]
            num_vals = [v for v in values if not isinstance(v, str)]
            
            # Sort numbers to get consistent ordering (initial, min, max)
            num_vals = sorted(num_vals)
            
            # Combine: numbers first, then string (latex) if present
            if str_vals:
                values = num_vals + str_vals
            else:
                values = num_vals
        else:
            values = list(values)
        
        if len(values) < 3:
            raise ValueError(f"Parameter '{name}' needs at least (initial, min, max)")
        
        initial = float(values[0])
        min_val = float(values[1])
        max_val = float(values[2])
        latex = values[3] if len(values) > 3 else None
        
        return cls.from_simple_config(name, initial, min_val, max_val, latex)
    
    @classmethod
    def from_simple_config(
        cls, 
        name: str, 
        initial: float, 
        min_val: float, 
        max_val: float,
        latex: Optional[str] = None
    ) -> 'AutoParameter':
        """
        Create AutoParameter from simple configuration (qcosmc-style).
        
        Parameters
        ----------
        name : str
            Parameter name.
        initial : float
            Initial/reference value.
        min_val : float
            Minimum value.
        max_val : float
            Maximum value.
        latex : str, optional
            LaTeX representation.
            
        Returns
        -------
        AutoParameter
            Configured parameter object.
        """
        return cls(
            name=name,
            prior={"dist": "uniform", "min": min_val, "max": max_val},
            ref=initial,
            bounds=(min_val, max_val),
            latex=latex
        )
    
    def __repr__(self) -> str:
        """String representation."""
        return f"AutoParameter(name='{self.name}', prior={self.prior}, ref={self.ref})"


@dataclass 
class ParameterConfig:
    """
    Complete parameter configuration for MCMC sampling.
    
    This class manages a collection of parameters and provides validation,
    conversion utilities, and integration with the MCMC sampling system.
    
    Parameters
    ----------
    parameters : Dict[str, AutoParameter]
        Dictionary mapping parameter names to AutoParameter objects.
    derived : List[str], optional
        List of derived parameter names.
    mcmc : Dict[str, Any], optional
        MCMC sampling configuration.
    
    Examples
    --------
    >>> config = ParameterConfig(
    ...     parameters={
    ...         'H0': AutoParameter('H0', {'dist': 'uniform', 'min': 50, 'max': 100}),
    ...         'Omega_m': AutoParameter('Omega_m', {'dist': 'normal', 'loc': 0.3, 'scale': 0.05})
    ...     },
    ...     derived=['age', 'Omega_Lambda'],
    ...     mcmc={'num_samples': 4000, 'num_chains': 4}
    ... )
    """
    
    parameters: Dict[str, AutoParameter] = field(default_factory=dict)
    derived: List[str] = field(default_factory=list)
    mcmc: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_parameters()
        self._validate_derived()
        self._set_mcmc_defaults()
    
    def _validate_parameters(self):
        """Validate parameter definitions."""
        if not self.parameters:
            raise ValueError("At least one parameter must be defined")
        
        # Check for name conflicts
        param_names = set(self.parameters.keys())
        if len(param_names) != len(self.parameters):
            raise ValueError("Duplicate parameter names found")
        
        # Validate each parameter
        for name, param in self.parameters.items():
            if param.name != name:
                raise ValueError(f"Parameter name mismatch: key '{name}' vs param.name '{param.name}'")
    
    def _validate_derived(self):
        """Validate derived parameter names."""
        param_names = set(self.parameters.keys())
        for derived_name in self.derived:
            if derived_name in param_names:
                raise ValueError(f"Derived parameter '{derived_name}' conflicts with sampled parameter")
            
            if not derived_name.isidentifier():
                raise ValueError(f"Derived parameter '{derived_name}' is not a valid identifier")
    
    def _set_mcmc_defaults(self):
        """Set default MCMC configuration."""
        defaults = {
            'num_warmup': 0,  # 默认不需要warmup，从优化初始值开始
            'num_samples': 4000,
            'num_chains': 4,
            'verbose': True
        }
        
        for key, default_value in defaults.items():
            if key not in self.mcmc:
                self.mcmc[key] = default_value
    
    def add_parameter(self, param: AutoParameter):
        """Add a parameter to the configuration."""
        if param.name in self.parameters:
            raise ValueError(f"Parameter '{param.name}' already exists")
        self.parameters[param.name] = param
    
    def remove_parameter(self, name: str):
        """Remove a parameter from the configuration."""
        if name not in self.parameters:
            raise ValueError(f"Parameter '{name}' not found")
        del self.parameters[name]
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self.parameters.keys())
    
    def get_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get bounds for all parameters."""
        bounds = {}
        for name, param in self.parameters.items():
            if param.bounds:
                bounds[name] = param.bounds
            else:
                # Infer from prior
                if param.prior["dist"] == "uniform":
                    bounds[name] = (param.prior["min"], param.prior["max"])
                else:
                    # Use wide bounds as fallback
                    bounds[name] = (-np.inf, np.inf)
        return bounds
    
    def get_reference_values(self) -> Dict[str, float]:
        """Get reference values for all parameters."""
        return {name: param.ref for name, param in self.parameters.items()}
    
    def get_numpyro_priors(self) -> Dict[str, Any]:
        """Get NumPyro distribution objects for all parameters."""
        return {name: param.to_numpyro_dist() for name, param in self.parameters.items()}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'parameters': {name: param.to_dict() for name, param in self.parameters.items()},
            'derived': self.derived,
            'mcmc': self.mcmc
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterConfig':
        """
        Create ParameterConfig from dictionary.
        
        Supports both full dictionary format and simplified tuple/list/set format:
        
        Full format:
        {
            'parameters': {
                'a': {'prior': {'dist': 'uniform', 'min': 0, 'max': 10}, 'ref': 3.5}
            }
        }
        
        Simplified format:
        {
            'parameters': {
                'a': (3.5, 0, 10, '$a$'),  # or [3.5, 0, 10] or {3.5, 0, 10, '$a$'}
                'b': [2.0, 0, 4, '$b$'],
                'c': {1.0, 0, 2}
            }
        }
        """
        parameters = {}
        for name, param_data in data.get('parameters', {}).items():
            # Check if it's simplified format (tuple, list, or set)
            if isinstance(param_data, (tuple, list, set)):
                parameters[name] = AutoParameter.from_tuple(name, param_data)
            # Check if it's a dictionary with 'prior' key (full format)
            elif isinstance(param_data, dict) and 'prior' in param_data:
                parameters[name] = AutoParameter.from_dict(param_data, name=name)
            # Check if it's a dict but looks like simplified format (numbers as values)
            elif isinstance(param_data, dict) and not any(k in param_data for k in ['prior', 'name', 'bounds']):
                # Try to convert dict values to list
                values = list(param_data.values()) if param_data else []
                if values and all(isinstance(v, (int, float, str)) for v in values):
                    parameters[name] = AutoParameter.from_tuple(name, values)
                else:
                    raise ValueError(f"Invalid parameter format for '{name}'")
            else:
                # Assume it's meant to be full format
                parameters[name] = AutoParameter.from_dict(param_data, name=name)
        
        return cls(
            parameters=parameters,
            derived=data.get('derived', []),
            mcmc=data.get('mcmc', {})
        )
    
    @classmethod
    def from_simple_list(cls, param_list: List[List], **mcmc_kwargs) -> 'ParameterConfig':
        """
        Create ParameterConfig from simple parameter list (qcosmc-style).
        
        Parameters
        ----------
        param_list : List[List]
            List of parameter definitions: [[name, initial, min, max], ...]
        **mcmc_kwargs
            MCMC configuration options.
            
        Returns
        -------
        ParameterConfig
            Configured parameter object.
            
        Examples
        --------
        >>> config = ParameterConfig.from_simple_list([
        ...     ['H0', 70, 50, 100],
        ...     ['Omega_m', 0.3, 0.1, 0.5],
        ... ], num_samples=2000)
        """
        parameters = {}
        for param_def in param_list:
            if len(param_def) < 4:
                raise ValueError(f"Parameter definition {param_def} must have [name, initial, min, max]")
            
            name, initial, min_val, max_val = param_def[:4]
            latex = param_def[4] if len(param_def) > 4 else None
            
            parameters[name] = AutoParameter.from_simple_config(
                name, initial, min_val, max_val, latex
            )
        
        return cls(parameters=parameters, mcmc=mcmc_kwargs)
    
    def __repr__(self) -> str:
        """String representation."""
        param_names = list(self.parameters.keys())
        return f"ParameterConfig(parameters={param_names}, derived={self.derived})"