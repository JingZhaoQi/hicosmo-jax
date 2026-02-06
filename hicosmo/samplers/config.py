#!/usr/bin/env python3
"""
Parameter Configuration System for HIcosmo MCMC.

This module provides a dictionary-driven parameter management system that bridges
user-friendly configuration with NumPyro's flexible model definition.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Union, List, Tuple
import numpy as np
import numpyro.distributions as dist
from enum import Enum

# Import unified Parameter class
from ..parameters import Parameter


@dataclass
class ParameterConfig:
    """
    Complete parameter configuration for MCMC sampling.

    This class manages a collection of parameters and provides validation,
    conversion utilities, and integration with the MCMC sampling system.

    Parameters
    ----------
    parameters : Dict[str, Parameter]
        Dictionary mapping parameter names to Parameter objects.
    derived : List[str], optional
        List of derived parameter names.
    mcmc : Dict[str, Any], optional
        MCMC sampling configuration.

    Examples
    --------
    >>> config = ParameterConfig(
    ...     parameters={
    ...         'H0': Parameter('H0', value=70, free=True,
    ...                        prior={'dist': 'uniform', 'min': 50, 'max': 100}),
    ...         'Omega_m': Parameter('Omega_m', value=0.3, free=True,
    ...                             prior={'dist': 'normal', 'loc': 0.3, 'scale': 0.05})
    ...     },
    ...     derived=['age', 'Omega_Lambda'],
    ...     mcmc={'num_samples': 4000, 'num_chains': 4}
    ... )
    """

    parameters: Dict[str, Parameter] = field(default_factory=dict)
    derived: List[str] = field(default_factory=list)
    mcmc: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self._validate_parameters()
        self._validate_derived()
        self._set_mcmc_defaults()

    def _validate_parameters(self):
        """Validate parameter definitions using unified Parameter class."""
        if not self.parameters:
            raise ValueError("At least one parameter must be defined")

        # Normalize simple definitions into Parameter instances
        for name, param in list(self.parameters.items()):
            # Already a Parameter instance
            if isinstance(param, Parameter):
                continue

            # Note: AutoParameter has been removed, no conversion needed

            # Tuple/list/set format
            if isinstance(param, (list, tuple, set)):
                self.parameters[name] = Parameter.from_tuple(name, param)
                continue

            # Dictionary format
            if isinstance(param, dict):
                self.parameters[name] = Parameter.from_simple_config(name, param)
                continue

            # Unsupported type
            raise ValueError(
                f"Unsupported parameter specification for '{name}'. "
                "Provide a Parameter, tuple/list/set, or dict definition."
            )

        # Check for name conflicts
        param_names = set(self.parameters.keys())
        if len(param_names) != len(self.parameters):
            raise ValueError("Duplicate parameter names found")

        # Validate each parameter
        for name, param in self.parameters.items():
            if param.name != name:
                raise ValueError(
                    f"Parameter name mismatch: key '{name}' vs param.name '{param.name}'"
                )

    def _validate_derived(self):
        """Validate derived parameter names."""
        param_names = set(self.parameters.keys())
        for derived_name in self.derived:
            if derived_name in param_names:
                raise ValueError(
                    f"Derived parameter '{derived_name}' conflicts with sampled parameter"
                )

            if not derived_name.isidentifier():
                raise ValueError(
                    f"Derived parameter '{derived_name}' is not a valid identifier"
                )

    def _set_mcmc_defaults(self):
        """Set basic MCMC configuration defaults.

        Note: num_samples and num_chains are now set intelligently by MCMC class,
        so we don't set hardcoded defaults here anymore.
        """
        defaults = {
            "verbose": True  # Only set basic defaults, let MCMC handle intelligent ones
        }

        for key, default_value in defaults.items():
            if key not in self.mcmc:
                self.mcmc[key] = default_value

    def add_parameter(self, param: Parameter):
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
        """Get bounds for all free parameters."""
        bounds = {}
        for name, param in self.parameters.items():
            if not param.free:
                continue

            if param.bounds:
                bounds[name] = param.bounds
            elif param.prior and param.prior.get("dist") == "uniform":
                bounds[name] = (param.prior["min"], param.prior["max"])
            else:
                # Use wide bounds as fallback
                bounds[name] = (-np.inf, np.inf)
        return bounds

    def get_labels(self) -> Dict[str, str]:
        """Get LaTeX labels for all parameters.

        Returns dictionary mapping parameter names to their LaTeX labels.
        Parameter.latex_label is already filled by Parameter._set_defaults().
        """
        return {
            name: param.latex_label or name for name, param in self.parameters.items()
        }

    def get_reference_values(self) -> Dict[str, float]:
        """Get reference values for all parameters."""
        return {
            name: param.value
            for name, param in self.parameters.items()
            if param.value is not None
        }

    def get_numpyro_priors(self) -> Dict[str, Any]:
        """Get NumPyro distribution objects for all free parameters."""
        return {
            name: param.to_numpyro_dist()
            for name, param in self.parameters.items()
            if param.free
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "parameters": {
                name: param.to_dict() for name, param in self.parameters.items()
            },
            "derived": self.derived,
            "mcmc": self.mcmc,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterConfig":
        """
        Create ParameterConfig from dictionary.

        Supports both nested and flat dictionary formats:

        Nested format (with 'parameters' key):
        {
            'parameters': {
                'a': {'prior': {'dist': 'uniform', 'min': 0, 'max': 10}, 'ref': 3.5}
            },
            'mcmc': {'num_samples': 1000}
        }

        Flat format (parameters at top level):
        {
            'a': {'prior': {'dist': 'uniform', 'min': 0, 'max': 10}, 'ref': 3.5},
            'mcmc': {'num_samples': 1000}
        }

        Simplified tuple/list/set format:
        {
            'parameters': {
                'a': (3.5, 0, 10, '$a$'),  # or [3.5, 0, 10] or {3.5, 0, 10, '$a$'}
                'b': [2.0, 0, 4, '$b$']
            }
        }
        """
        # Extract MCMC config first
        mcmc_config = data.get("mcmc", {})
        derived_params = data.get("derived", [])

        # Determine if data is in nested format (has 'parameters' key) or flat format
        if "parameters" in data:
            # Nested format: parameters are under 'parameters' key
            param_source = data["parameters"]
        else:
            # Flat format: parameters are at top level
            # Extract all keys except 'mcmc' and 'derived'
            param_source = {
                key: value
                for key, value in data.items()
                if key not in ("mcmc", "derived")
            }

        # Parse parameters from the source dictionary
        parameters = {}
        for name, param_data in param_source.items():
            # Check if it's simplified format (tuple, list, or set)
            if isinstance(param_data, (tuple, list, set)):
                parameters[name] = Parameter.from_tuple(name, param_data)
            # Check if it's a dictionary
            elif isinstance(param_data, dict):
                parameters[name] = Parameter.from_simple_config(name, param_data)
            else:
                raise ValueError(f"Invalid parameter format for '{name}'")

        return cls(parameters=parameters, derived=derived_params, mcmc=mcmc_config)

    @classmethod
    def from_simple_list(
        cls, param_list: List[List], **mcmc_kwargs
    ) -> "ParameterConfig":
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
                raise ValueError(
                    f"Parameter definition {param_def} must have [name, initial, min, max]"
                )

            name, initial, min_val, max_val = param_def[:4]
            latex = param_def[4] if len(param_def) > 4 else None

            parameters[name] = Parameter.from_tuple(
                name, (initial, min_val, max_val, latex)
            )

        return cls(parameters=parameters, mcmc=mcmc_kwargs)

    def __repr__(self) -> str:
        """String representation."""
        param_names = list(self.parameters.keys())
        return f"ParameterConfig(parameters={param_names}, derived={self.derived})"
